#!/usr/bin/env python3
"""Normalize filtered MedMCQA TSV rows into mamabench JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.adapters.medmcqa import (  # noqa: E402
    MedMCQAAdapterError,
    load_medmcqa_tsv,
)
from mamabench.io import write_json, write_jsonl  # noqa: E402
from mamabench.manifest import build_manifest  # noqa: E402
from mamabench.validate import validate_items  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", help="Filtered MedMCQA TSV input file.")
    parser.add_argument("output_jsonl", help="Path for normalized mamabench JSONL.")
    parser.add_argument(
        "--benchmark-version",
        default="v0.1",
        help="Benchmark version used in generated item ids. Default: v0.1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Normalize at most N rows, useful for dry runs.",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional path to write a manifest JSON summary.",
    )
    parser.add_argument(
        "--validation-report-output",
        default=None,
        help="Optional path to write a validation report JSON.",
    )
    args = parser.parse_args(argv)

    try:
        rows = load_medmcqa_tsv(
            args.input_tsv,
            benchmark_version=args.benchmark_version,
            limit=args.limit,
        )
        report = validate_items(rows)
        manifest = build_manifest(
            rows,
            benchmark_version=args.benchmark_version,
            validation_report=report,
        )
    except (OSError, MedMCQAAdapterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.validation_report_output:
        write_json(args.validation_report_output, report.to_dict())
    if args.manifest_output:
        write_json(args.manifest_output, manifest)

    if not report.ok:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 1

    write_jsonl(args.output_jsonl, rows)
    print(
        json.dumps(
            {
                "ok": True,
                "output_jsonl": args.output_jsonl,
                "item_count": len(rows),
                "manifest": manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
