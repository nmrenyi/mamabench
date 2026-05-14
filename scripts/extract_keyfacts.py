#!/usr/bin/env python3
"""Extract atomic must-cover key_facts from open-ended mamabench rows.

Reads a v0.4 open_ended mamabench JSONL (one of: kenya.jsonl,
afrimedqa_saq.jsonl, whb.jsonl), sends each row's (question, answer) as one
chat turn to an OpenAI-compatible chat completion endpoint (vLLM / TGI /
SGLang on a local cluster), parses the JSON extraction result, and appends a
record to the output JSONL.

The run is **resumable**: rows whose ``row_id`` is already in the output
file are skipped on restart. It also supports **cross-job sharding** so
multiple parallel jobs (one per GPU on a cluster) can split the work, and
**in-process concurrency** so a single job can keep one GPU saturated.

Example:
    export OPENAI_BASE_URL=http://cluster:8000/v1
    export OPENAI_API_KEY=EMPTY
    python scripts/extract_keyfacts.py \\
        --input  benchmark/v0.2/kenya.jsonl \\
        --output benchmark/v0.2/key_facts/kenya_keyfacts.jsonl \\
        --model  Qwen/Qwen3.5-397B-A17B-FP8 \\
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

from mamabench.adapters._keyfact_extraction import (  # noqa: E402
    EXTRACTION_JSON_SCHEMA,
    ExtractionError,
    extract_row_with_reasoning,
)
from mamabench.obgyn_classifier import (  # noqa: E402
    make_openai_completer_with_reasoning,
)
from mamabench.prompts import (  # noqa: E402
    KEYFACT_EXTRACTOR_PROMPT_VERSION,
    load_keyfact_extractor_prompt,
)


def reasoning_path_for(output_path: Path) -> Path:
    """Derive the reasoning side-file path from the main output path.

    ``foo/bar_keyfacts.jsonl`` → ``foo/bar_keyfacts_reasoning.jsonl``.
    """
    return output_path.with_name(output_path.stem + "_reasoning.jsonl")


def iter_open_ended_jsonl(jsonl_path: Path):
    """Yield ``(row_id, question, reference)`` for each open_ended row.

    Rows that are not ``set_type == 'open_ended'`` are skipped silently —
    we only score rows that have a single reference response.
    """
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("set_type") != "open_ended":
                continue
            row_id = row.get("id")
            question = row.get("question")
            answer = row.get("answer")
            if not (isinstance(row_id, str) and isinstance(question, str) and isinstance(answer, str)):
                continue
            yield row_id, question, answer


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
    input_path: Path,
    *,
    already_done: set[str],
    shard: tuple[int, int] | None,
    limit: int | None,
) -> tuple[list[tuple[str, str, str]], int, int]:
    """Walk the input and return rows this run should process.

    Filters applied (in order):
      1. shard: keep rows where source_row_index %% shard_count == shard_index
      2. already_done: skip rows whose row_id is already in the output
      3. limit: stop after collecting this many rows

    Returns (rows_to_process, n_skipped_shard, n_skipped_resume).
    """
    rows: list[tuple[str, str, str]] = []
    n_skipped_shard = 0
    n_skipped_resume = 0
    for i, (row_id, question, reference) in enumerate(iter_open_ended_jsonl(input_path)):
        if shard is not None:
            shard_index, shard_count = shard
            if i % shard_count != shard_index:
                n_skipped_shard += 1
                continue
        if row_id in already_done:
            n_skipped_resume += 1
            continue
        rows.append((row_id, question, reference))
        if limit is not None and len(rows) >= limit:
            break
    return rows, n_skipped_shard, n_skipped_resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a v0.4 open_ended mamabench JSONL (kenya / afrimedqa_saq / whb).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the key_facts JSONL. Appended to if it exists; rows "
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
        help="Enable OpenAI-compatible structured generation (response_format "
        "with json_schema). Default: enabled. Pass --no-guided-json to disable "
        "for servers without structured-output support.",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable Qwen3+ thinking mode via chat_template_kwargs (default "
        "OFF — thinking is enabled by default for the key-fact extractor "
        "because the task has subtle judgment calls that benefit from CoT). "
        "Pass --disable-thinking to turn thinking off (faster, smaller "
        "context budget).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0 for deterministic extractions).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=23552,
        help="Soft cap (Qwen3+ chat_template_kwargs.thinking_budget) on the "
        "reasoning portion of the response, in tokens. Default 23,552 — sized "
        "to leave room (under a 32K vLLM MAX_MODEL_LEN with ~4K worst-case "
        "input) for the model to wrap up thinking and emit the JSON before "
        "the server's hard context limit. Ignored when thinking is disabled. "
        "This is a soft, model-honored signal; the only hard ceiling is "
        "MAX_MODEL_LEN itself.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent in-flight requests to the LLM (default 1). "
        "On a cluster GPU, 8 is a common choice for vLLM continuous batching.",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "COUNT"),
        default=None,
        help="If set, process only rows where source_row_index %% COUNT == "
        "INDEX. Use to split work across parallel runai jobs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after extracting this many new rows. Useful for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print a progress line every N rows extracted (default 20).",
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

    system_prompt = load_keyfact_extractor_prompt()
    complete = make_openai_completer_with_reasoning(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        json_schema=EXTRACTION_JSON_SCHEMA if args.guided_json else None,
        disable_thinking=args.disable_thinking,
        thinking_budget=args.thinking_budget,
        schema_name="extraction",
    )

    reasoning_output_path = reasoning_path_for(args.output)
    already_done = load_existing_row_ids(args.output)
    if already_done:
        print(
            f"resume: skipping {len(already_done)} rows already in {args.output}"
        )

    shard_tuple = tuple(args.shard) if args.shard is not None else None
    rows_to_process, n_skipped_shard, n_skipped_resume = select_rows(
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
    print(f"about to extract {len(rows_to_process)} rows with {args.workers} workers")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    counts = {"done": 0, "errors": 0, "reasoning_missing": 0}
    t_start = time.time()

    def process_row(row_id: str, question: str, reference: str) -> None:
        try:
            result, reasoning = extract_row_with_reasoning(
                complete=complete,
                system_prompt=system_prompt,
                question=question,
                reference=reference,
            )
        except ExtractionError as e:
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
            "model": args.model,
            "prompt_version": KEYFACT_EXTRACTOR_PROMPT_VERSION,
            "summary": result["summary"],
            "key_facts": result["key_facts"],
        }
        reasoning_record = {
            "row_id": row_id,
            "model": args.model,
            "prompt_version": KEYFACT_EXTRACTOR_PROMPT_VERSION,
            "reasoning": reasoning or "",
        }
        with write_lock:
            # Write reasoning first so the main side-file is the resume key
            # — if a crash interrupts a row, the worst case is an orphan
            # reasoning record without a matching keyfacts record (harmless;
            # the next resume just re-extracts that row).
            reasoning_file.write(
                json.dumps(reasoning_record, ensure_ascii=False) + "\n"
            )
            reasoning_file.flush()
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            counts["done"] += 1
            if reasoning is None:
                counts["reasoning_missing"] += 1
            if counts["done"] % args.progress_every == 0:
                elapsed = time.time() - t_start
                rate = counts["done"] / elapsed if elapsed > 0 else 0.0
                print(
                    f"  extracted {counts['done']} rows ({rate:.1f} rows/s)"
                )

    reasoning_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out_file, \
            reasoning_output_path.open("a", encoding="utf-8") as reasoning_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_row, row_id, q, ref)
                for row_id, q, ref in rows_to_process
            ]
            for fut in as_completed(futures):
                fut.result()

    elapsed = time.time() - t_start
    rate = counts["done"] / elapsed if elapsed > 0 else 0.0
    print()
    print(
        f"done: {counts['done']} extracted, {n_skipped_resume} skipped (resume), "
        f"{n_skipped_shard} skipped (shard), {counts['errors']} errors"
    )
    if counts["done"]:
        print(f"elapsed: {elapsed:.1f}s ({rate:.1f} rows/s)")
    if counts["reasoning_missing"] > 0:
        print(
            f"  note: {counts['reasoning_missing']}/{counts['done']} rows had "
            f"no reasoning_content from the server — check that vLLM was "
            f"started with --enable-reasoning --reasoning-parser qwen3."
        )
    print(f"  keyfacts:  {args.output}")
    print(f"  reasoning: {reasoning_output_path}")

    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
