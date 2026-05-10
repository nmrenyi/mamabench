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
    MEDMCQA_SOURCE_METADATA,
    MedMCQAAdapterError,
    load_medmcqa_tsv,
)
from mamabench.config import (  # noqa: E402
    load_project_config,
    normalize_benchmark_version,
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
        default=None,
        help=(
            "Benchmark version used in generated item ids. "
            "Default: benchmark_version from mamabench.json."
        ),
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
    args = parser.parse_args(argv)

    try:
        config = load_project_config(ROOT / "mamabench.json")
        benchmark_version = normalize_benchmark_version(
            args.benchmark_version or config.benchmark_version
        )
        rows = load_medmcqa_tsv(
            args.input_tsv,
            benchmark_version=benchmark_version,
            limit=args.limit,
        )
        report = validate_items(rows)
        manifest = build_manifest(
            rows,
            benchmark_version=benchmark_version,
            schema_version=config.schema_version,
            source_dataset_metadata=MEDMCQA_SOURCE_METADATA,
            validation_report=report,
        )
    except (OSError, MedMCQAAdapterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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
