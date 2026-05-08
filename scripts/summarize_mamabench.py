#!/usr/bin/env python3
"""Summarize a mamabench JSONL file as a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.io import read_jsonl  # noqa: E402
from mamabench.manifest import build_manifest  # noqa: E402
from mamabench.validate import validate_items  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", help="Path to a normalized mamabench JSONL file.")
    parser.add_argument(
        "--benchmark-version",
        default="v0.1",
        help="Benchmark version to include in the manifest. Default: v0.1.",
    )
    args = parser.parse_args(argv)

    try:
        rows = read_jsonl(args.jsonl_path)
        report = validate_items(rows)
        manifest = build_manifest(
            rows,
            benchmark_version=args.benchmark_version,
            validation_report=report,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

