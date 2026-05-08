#!/usr/bin/env python3
"""Validate a mamabench JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mamabench.io import read_jsonl  # noqa: E402
from mamabench.validate import validate_items  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", help="Path to a normalized mamabench JSONL file.")
    parser.add_argument(
        "--check-perturbation-refs",
        action="store_true",
        help="Require every perturbation_of value to reference a known item id.",
    )
    parser.add_argument(
        "--known-ids-jsonl",
        action="append",
        default=[],
        help=(
            "Additional JSONL file whose item ids are valid perturbation targets. "
            "Can be passed more than once."
        ),
    )
    args = parser.parse_args(argv)

    try:
        rows = read_jsonl(args.jsonl_path)
        known_ids = _read_known_ids(args.known_ids_jsonl)
        report = validate_items(
            rows,
            known_ids=known_ids,
            check_perturbation_references=(
                args.check_perturbation_refs or bool(known_ids)
            ),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _read_known_ids(paths: list[str]) -> set[str]:
    known_ids: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            item_id = row.get("id")
            if isinstance(item_id, str) and item_id.strip():
                known_ids.add(item_id)
    return known_ids


if __name__ == "__main__":
    raise SystemExit(main())
