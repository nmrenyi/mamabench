#!/usr/bin/env python3
"""Classify upstream rows with the unified OBGYN classifier prompt.

Reads rows from a source file (HealthBench JSONL, Kenya xlsx, or MedQA-USMLE
JSONL), sends each as one chat turn to an OpenAI-compatible chat completion
endpoint (vLLM / TGI / SGLang / Ollama on a local cluster), parses the JSON
verdict, and appends a record to the output JSONL. The run is resumable:
rows whose ``row_id`` is already in the output file are skipped.

Example:
    export OPENAI_BASE_URL=http://cluster:8000/v1
    export OPENAI_API_KEY=EMPTY
    python scripts/classify_obgyn.py \\
        --source healthbench --subset oss_eval --mode openended \\
        --input  ~/Downloads/healthbench/data/2025-05-07-06-14-12_oss_eval.jsonl \\
        --output benchmark/v0.2/classification_verdicts/healthbench_oss_eval.jsonl \\
        --model  Qwen/Qwen2.5-72B-Instruct

Guided JSON structured generation is enabled by default; pass --no-guided-json
to disable if your server doesn't support it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.obgyn_classifier import (  # noqa: E402
    ClassifierError,
    classify_row,
    make_openai_completer,
    vllm_guided_json_extra_body,
)
from mamabench.obgyn_sources import (  # noqa: E402
    iter_healthbench,
    iter_kenya,
    iter_medqa_usmle,
)
from mamabench.prompts import PROMPT_VERSION, load_classifier_prompt  # noqa: E402


SOURCE_ITERATORS = {
    "healthbench": iter_healthbench,
    "kenya": iter_kenya,
    "medqa_usmle": iter_medqa_usmle,
}

DEFAULT_MODE = {
    "healthbench": "openended",
    "kenya": "openended",
    "medqa_usmle": "mcq",
}


def load_existing_row_ids(output_path: Path) -> set[str]:
    """Return the set of row_ids already present in the output JSONL."""
    if not output_path.exists():
        return set()
    seen: set[str] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = record.get("row_id")
            if isinstance(row_id, str):
                seen.add(row_id)
    return seen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_ITERATORS),
        required=True,
        help="Upstream source to classify rows from.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subset label (e.g. oss_eval, consensus, hard) recorded "
        "in each output row. Defaults to the --source value.",
    )
    parser.add_argument(
        "--mode",
        choices=("openended", "mcq"),
        default=None,
        help="Classifier prompt mode. Defaults to openended for "
        "healthbench/kenya, mcq for medqa_usmle.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source file (JSONL for HealthBench / MedQA-USMLE, "
        "xlsx for Kenya).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the verdicts JSONL. Appended to if it exists; rows "
        "already present (by row_id) are skipped.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id served by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (e.g. http://host:8000/v1). "
        "Defaults to the OPENAI_BASE_URL env var or the OpenAI default.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the OpenAI-compatible endpoint. vLLM ignores this; "
        "set to any non-empty string.",
    )
    parser.add_argument(
        "--guided-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM's guided_json structured generation. Default: enabled. "
        "Pass --no-guided-json to disable (e.g. for servers without guided "
        "decoding support).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0 for deterministic verdicts).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after classifying this many new rows. Useful for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a progress line every N rows classified (default 50).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"error: --input not found: {args.input}", file=sys.stderr)
        return 2

    subset = args.subset or args.source
    mode = args.mode or DEFAULT_MODE[args.source]
    source_iter = SOURCE_ITERATORS[args.source]

    system_prompt = load_classifier_prompt(mode)
    complete = make_openai_completer(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        extra_body=vllm_guided_json_extra_body() if args.guided_json else None,
    )

    already_done = load_existing_row_ids(args.output)
    if already_done:
        print(
            f"resume: skipping {len(already_done)} rows already in "
            f"{args.output.relative_to(ROOT) if args.output.is_absolute() and args.output.is_relative_to(ROOT) else args.output}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skipped_resume = 0
    n_errors = 0
    t_start = time.time()

    with args.output.open("a", encoding="utf-8") as out:
        for row_id, user_message in source_iter(args.input):
            if row_id in already_done:
                n_skipped_resume += 1
                continue

            try:
                verdict = classify_row(
                    complete=complete,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )
            except ClassifierError as e:
                n_errors += 1
                print(f"  ⚠ [{row_id}] {e}", file=sys.stderr)
                continue

            record = {
                "row_id": row_id,
                "source": subset,
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "mode": mode,
                "category": verdict["category"],
                "rationale": verdict["rationale"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1

            if n_done % args.progress_every == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(f"  classified {n_done} rows ({rate:.1f} rows/s)")

            if args.limit is not None and n_done >= args.limit:
                break

    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0.0
    print()
    print(
        f"done: {n_done} classified, {n_skipped_resume} skipped (resume), "
        f"{n_errors} parse errors"
    )
    if n_done:
        print(f"elapsed: {elapsed:.1f}s ({rate:.1f} rows/s)")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
