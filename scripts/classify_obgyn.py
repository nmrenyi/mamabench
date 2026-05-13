#!/usr/bin/env python3
"""Classify upstream rows with the unified OBGYN classifier prompt.

Reads rows from a source file (HealthBench JSONL, Kenya xlsx, or MedQA-USMLE
JSONL), sends each as one chat turn to an OpenAI-compatible chat completion
endpoint (vLLM / TGI / SGLang / Ollama on a local cluster), parses the JSON
verdict, and appends a record to the output JSONL.

The run is **resumable**: rows whose ``row_id`` is already in the output file
are skipped on restart. It also supports **cross-job sharding** so multiple
parallel jobs (one per GPU on a cluster) can split the work, and
**in-process concurrency** so a single job can keep one GPU saturated.

Example:
    export OPENAI_BASE_URL=http://cluster:8000/v1
    export OPENAI_API_KEY=EMPTY
    python scripts/classify_obgyn.py \\
        --source healthbench --subset oss_eval --mode openended \\
        --input  ~/Downloads/healthbench/data/2025-05-07-06-14-12_oss_eval.jsonl \\
        --output benchmark/v0.2/classification_verdicts/healthbench_oss_eval.jsonl \\
        --model  Qwen/Qwen3.6-27B-FP8 \\
        --workers 8

Guided JSON structured generation is enabled by default; pass --no-guided-json
to disable if your server doesn't support it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def select_rows(
    source_iter,
    input_path: Path,
    *,
    already_done: set[str],
    shard: tuple[int, int] | None,
    limit: int | None,
) -> tuple[list[tuple[str, str]], int, int]:
    """Walk the source iterator and return the rows that this run should
    actually classify, along with skip counts.

    Filters applied (in order):
      1. shard: keep rows where ``source_row_index % shard_count == shard_index``
      2. already_done: skip rows whose row_id is already in the output
      3. limit: stop after collecting this many rows

    Returns (rows_to_process, n_skipped_shard, n_skipped_resume).
    """
    rows: list[tuple[str, str]] = []
    n_skipped_shard = 0
    n_skipped_resume = 0
    for i, (row_id, user_message) in enumerate(source_iter(input_path)):
        if shard is not None:
            shard_index, shard_count = shard
            if i % shard_count != shard_index:
                n_skipped_shard += 1
                continue
        if row_id in already_done:
            n_skipped_resume += 1
            continue
        rows.append((row_id, user_message))
        if limit is not None and len(rows) >= limit:
            break
    return rows, n_skipped_shard, n_skipped_resume


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
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent in-flight requests to the LLM (default 1, "
        "sync). On a cluster GPU, 8 is a common choice — keeps the GPU "
        "saturated under vLLM's continuous batching and avoids the 2-hour "
        "idle-GPU shutdown.",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "COUNT"),
        default=None,
        help="If set, process only rows where source_row_index %% COUNT == "
        "INDEX. Use to split work across parallel runai jobs (each job "
        "passes its own --shard i N).",
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

    if args.workers < 1:
        print(f"error: --workers must be >= 1 (got {args.workers})", file=sys.stderr)
        return 2

    if args.shard is not None:
        shard_index, shard_count = args.shard
        if shard_count < 1 or not (0 <= shard_index < shard_count):
            print(
                f"error: --shard INDEX COUNT must satisfy 0 <= INDEX < COUNT and COUNT >= 1 "
                f"(got INDEX={shard_index}, COUNT={shard_count})",
                file=sys.stderr,
            )
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
        print(f"resume: skipping {len(already_done)} rows already in {args.output}")

    shard_tuple = tuple(args.shard) if args.shard is not None else None
    rows_to_process, n_skipped_shard, n_skipped_resume = select_rows(
        source_iter,
        args.input,
        already_done=already_done,
        shard=shard_tuple,
        limit=args.limit,
    )

    if shard_tuple is not None:
        print(
            f"shard {shard_tuple[0]}/{shard_tuple[1]}: skipped {n_skipped_shard} "
            f"out-of-shard rows"
        )
    print(f"about to classify {len(rows_to_process)} rows with {args.workers} workers")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    counts = {"done": 0, "errors": 0}
    t_start = time.time()

    def process_row(row_id: str, user_message: str) -> None:
        try:
            verdict = classify_row(
                complete=complete,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except ClassifierError as e:
            with write_lock:
                counts["errors"] += 1
                print(f"  ⚠ [{row_id}] {e}", file=sys.stderr)
            return
        except Exception as e:  # noqa: BLE001 — keep the run going on transient errors
            with write_lock:
                counts["errors"] += 1
                print(
                    f"  ⚠ [{row_id}] unexpected {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return

        record = {
            "row_id": row_id,
            "source": subset,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "mode": mode,
            "category": verdict["category"],
            "rationale": verdict["rationale"],
        }
        with write_lock:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            counts["done"] += 1
            if counts["done"] % args.progress_every == 0:
                elapsed = time.time() - t_start
                rate = counts["done"] / elapsed if elapsed > 0 else 0.0
                print(f"  classified {counts['done']} rows ({rate:.1f} rows/s)")

    with args.output.open("a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_row, row_id, msg)
                for row_id, msg in rows_to_process
            ]
            for fut in as_completed(futures):
                fut.result()  # surface any unexpected exceptions

    elapsed = time.time() - t_start
    rate = counts["done"] / elapsed if elapsed > 0 else 0.0
    print()
    print(
        f"done: {counts['done']} classified, {n_skipped_resume} skipped (resume), "
        f"{n_skipped_shard} skipped (shard), {counts['errors']} errors"
    )
    if counts["done"]:
        print(f"elapsed: {elapsed:.1f}s ({rate:.1f} rows/s)")

    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
