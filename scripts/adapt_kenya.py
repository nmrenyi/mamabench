#!/usr/bin/env python3
"""Adapt Kenya Clinical Vignettes source xlsx into mamabench v0.4 open_ended rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.adapters.kenya import (  # noqa: E402
    SCHEMA_VERSION,
    KenyaAdapterError,
    build_kenya_source_metadata,
    load_kenya,
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
        help="Path to Kenya `Prompt responses.xlsx` (the unfiltered source).",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        required=True,
        help="Path to Kenya OBGYN classifier verdicts JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/v0.2/kenya.jsonl"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("benchmark/v0.2/manifests/kenya_manifest.json"),
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
        rows, stats = load_kenya(
            args.input,
            args.verdicts,
            benchmark_version=args.benchmark_version,
            limit=args.limit,
        )
    except (OSError, json.JSONDecodeError, KenyaAdapterError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _write_jsonl(args.output, rows)

    counts_by_category: dict[str, int] = {}
    for row in rows:
        cat = row["source"]["metadata"]["obgyn_classification"]["category"]
        counts_by_category[cat] = counts_by_category.get(cat, 0) + 1

    manifest = {
        "benchmark_version": args.benchmark_version,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utcnow_iso(),
        "total_item_count": len(rows),
        "counts_by_set_type": {"open_ended": len(rows)},
        "counts_by_source_dataset": {"Kenya-Clinical-Vignettes": len(rows)},
        "counts_by_category": counts_by_category,
        "filter": {
            "type": "obgyn_classifier_verdict",
            "stats": stats.as_dict(),
        },
        "source_datasets": build_kenya_source_metadata(),
        "outputs": {"rows": str(args.output)},
    }
    _write_json(args.manifest_output, manifest)

    print(
        json.dumps(
            {
                "ok": True,
                "rows_written": len(rows),
                "counts_by_category": counts_by_category,
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
