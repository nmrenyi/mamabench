#!/usr/bin/env python3
"""Adapt the Women's Health Benchmark stumps TSV into mamabench v0.4 open_ended rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.adapters.whb import (  # noqa: E402
    SCHEMA_VERSION,
    WHBAdapterError,
    build_whb_source_metadata,
    load_whb,
)
from mamabench.adapters._v0_4_validation import validate_rows  # noqa: E402


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
        help="Path to womens_health_stumps.tsv from the obgyn-qa-collection repo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/v0.2/whb.jsonl"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("benchmark/v0.2/manifests/whb_manifest.json"),
    )
    parser.add_argument("--benchmark-version", default="v0.2")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"error: --input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        rows, stats = load_whb(
            args.input,
            benchmark_version=args.benchmark_version,
            limit=args.limit,
        )
    except (OSError, WHBAdapterError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _write_jsonl(args.output, rows)
    validation = validate_rows(rows)
    manifest = {
        "benchmark_version": args.benchmark_version,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utcnow_iso(),
        "total_item_count": len(rows),
        "counts_by_set_type": {"open_ended": len(rows)},
        "counts_by_source_dataset": {"WHB": len(rows)},
        "validation": validation,
        "filter": {"type": "upstream-curated", "stats": stats.as_dict()},
        "source_datasets": build_whb_source_metadata(),
        "outputs": {"rows": str(args.output)},
    }
    _write_json(args.manifest_output, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "rows_written": len(rows),
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
