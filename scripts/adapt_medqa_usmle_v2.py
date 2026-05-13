#!/usr/bin/env python3
"""Adapt the full MedQA-USMLE US_qbank.jsonl into mamabench v0.4 mcq rows.

v0.2 refilters from the upstream raw JSONL (14,369 questions) using the
unified Qwen3.6-27B-FP8 OBGYN classifier verdicts. The row set differs
from v0.1's pre-filtered 1,025-row TSV input; that change is documented
in the v0.2 release notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.adapters.medqa_usmle_v2 import (  # noqa: E402
    SCHEMA_VERSION,
    MedQAUSMLEv2AdapterError,
    build_medqa_usmle_v2_source_metadata,
    load_medqa_usmle_v2,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to US_qbank.jsonl (the full 14,369-row MedQA-USMLE source).",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        required=True,
        help="Path to MedQA-USMLE OBGYN classifier verdicts JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/v0.2/medqa_usmle.jsonl"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("benchmark/v0.2/manifests/medqa_usmle_manifest.json"),
    )
    parser.add_argument("--benchmark-version", default="v0.2")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"error: --input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.verdicts.is_file():
        print(f"error: --verdicts not found: {args.verdicts}", file=sys.stderr)
        return 2

    try:
        rows, stats = load_medqa_usmle_v2(
            args.input,
            args.verdicts,
            benchmark_version=args.benchmark_version,
            limit=args.limit,
        )
    except (OSError, json.JSONDecodeError, MedQAUSMLEv2AdapterError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _write_jsonl(args.output, rows)

    counts_by_category: dict[str, int] = {}
    counts_by_step: dict[str, int] = {}
    for row in rows:
        meta = row["source"]["metadata"]
        cat = meta["obgyn_classification"]["category"]
        counts_by_category[cat] = counts_by_category.get(cat, 0) + 1
        step = meta.get("meta_info") or "unknown"
        counts_by_step[step] = counts_by_step.get(step, 0) + 1

    manifest = {
        "benchmark_version": args.benchmark_version,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utcnow_iso(),
        "total_item_count": len(rows),
        "counts_by_set_type": {"mcq": len(rows)},
        "counts_by_source_dataset": {"MedQA-USMLE": len(rows)},
        "counts_by_category": counts_by_category,
        "counts_by_step": counts_by_step,
        "filter": {
            "type": "obgyn_classifier_verdict",
            "stats": stats.as_dict(),
        },
        "source_datasets": build_medqa_usmle_v2_source_metadata(),
        "outputs": {"rows": str(args.output)},
        "release_notes": (
            "v0.2 refilters MedQA-USMLE from the upstream raw 14,369-row "
            "US_qbank.jsonl using the unified Qwen3.6-27B-FP8 OBGYN classifier. "
            "Row set differs from v0.1 (which used a pre-filtered 1,025-row "
            "TSV produced by an earlier Gemini classifier)."
        ),
    }
    _write_json(args.manifest_output, manifest)

    print(
        json.dumps(
            {
                "ok": True,
                "rows_written": len(rows),
                "counts_by_category": counts_by_category,
                "counts_by_step": counts_by_step,
                "filter_stats": stats.as_dict(),
                "output": str(args.output),
                "manifest": str(args.manifest_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
