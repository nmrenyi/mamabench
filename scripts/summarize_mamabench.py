#!/usr/bin/env python3
"""Print a lightweight summary for a mamabench JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.config import (  # noqa: E402
    load_project_config,
    normalize_benchmark_version,
)
from mamabench.io import read_jsonl  # noqa: E402
from mamabench.manifest import build_manifest  # noqa: E402
from mamabench.validate import validate_items  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", help="Path to a normalized mamabench JSONL file.")
    parser.add_argument(
        "--benchmark-version",
        default=None,
        help=(
            "Benchmark version to include in the summary. "
            "Default: benchmark_version from mamabench.json."
        ),
    )
    args = parser.parse_args(argv)

    try:
        config = load_project_config(ROOT / "mamabench.json")
        benchmark_version = normalize_benchmark_version(
            args.benchmark_version or config.benchmark_version
        )
        rows = read_jsonl(args.jsonl_path)
        report = validate_items(rows)
        summary = build_manifest(
            rows,
            benchmark_version=benchmark_version,
            schema_version=config.schema_version,
            validation_report=report,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
