#!/usr/bin/env python3
"""Adapt HealthBench source JSONLs into mamabench v0.4 open_ended_rubric rows.

Reads HealthBench's three subset files (``oss_eval``, ``consensus``,
``hard``) and their corresponding OBGYN-classifier verdict JSONLs, drops
rows whose verdict category is NONE (or rows without a verdict), and
emits per-subset mamabench JSONL plus a shared criteria side-table and
a manifest.

Example:
    python scripts/adapt_healthbench.py \\
        --oss-eval-input  ~/Downloads/healthbench/data/2025-05-07-06-14-12_oss_eval.jsonl \\
        --consensus-input ~/Downloads/healthbench/data/consensus_2025-05-09-20-00-46.jsonl \\
        --hard-input      ~/Downloads/healthbench/data/hard_2025-05-08-21-00-10.jsonl \\
        --oss-eval-verdicts  benchmark/v0.2/classification_verdicts/healthbench_oss_eval.jsonl \\
        --consensus-verdicts benchmark/v0.2/classification_verdicts/healthbench_consensus.jsonl \\
        --hard-verdicts      benchmark/v0.2/classification_verdicts/healthbench_hard.jsonl \\
        --output-dir         benchmark/v0.2 \\
        --manifest-output    benchmark/v0.2/manifests/healthbench_manifest.json
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

from mamabench.adapters.healthbench import (  # noqa: E402
    SCHEMA_VERSION,
    CriteriaIndex,
    HealthBenchAdapterError,
    build_healthbench_source_metadata,
    load_healthbench_subset,
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


def _per_axis_counts(criteria_index: CriteriaIndex) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in criteria_index.rows():
        counts[row["axis"]] = counts.get(row["axis"], 0) + 1
    return counts


def _per_level_counts(criteria_index: CriteriaIndex) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in criteria_index.rows():
        counts[row["level"]] = counts.get(row["level"], 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss-eval-input", type=Path, required=True)
    parser.add_argument("--consensus-input", type=Path, required=True)
    parser.add_argument("--hard-input", type=Path, required=True)
    parser.add_argument("--oss-eval-verdicts", type=Path, required=True)
    parser.add_argument("--consensus-verdicts", type=Path, required=True)
    parser.add_argument("--hard-verdicts", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/v0.2"),
        help="Directory under which healthbench_{oss_eval,consensus,hard}.jsonl + healthbench_criteria.jsonl are written.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("benchmark/v0.2/manifests/healthbench_manifest.json"),
    )
    parser.add_argument(
        "--benchmark-version",
        default="v0.2",
        help="Benchmark version segment used in generated row ids (default v0.2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows per subset (smoke-test only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    criteria_index = CriteriaIndex()
    subset_specs = [
        ("oss_eval", args.oss_eval_input, args.oss_eval_verdicts),
        ("consensus", args.consensus_input, args.consensus_verdicts),
        ("hard", args.hard_input, args.hard_verdicts),
    ]

    rows_by_subset: dict[str, list[dict[str, Any]]] = {}
    stats_by_subset: dict[str, dict[str, int]] = {}

    try:
        for subset, source_path, verdicts_path in subset_specs:
            if not source_path.is_file():
                raise HealthBenchAdapterError(f"source file not found: {source_path}")
            if not verdicts_path.is_file():
                raise HealthBenchAdapterError(f"verdicts file not found: {verdicts_path}")
            rows, stats = load_healthbench_subset(
                source_path,
                verdicts_path,
                subset=subset,
                benchmark_version=args.benchmark_version,
                criteria_index=criteria_index,
                limit=args.limit,
            )
            rows_by_subset[subset] = rows
            stats_by_subset[subset] = stats.as_dict()
    except (OSError, json.JSONDecodeError, HealthBenchAdapterError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Cross-subset duplicate-id check. Should never fire because the subset is
    # part of every mamabench row id, but cheap to verify.
    all_ids: dict[str, str] = {}
    for subset, rows in rows_by_subset.items():
        for row in rows:
            rid = row["id"]
            if rid in all_ids:
                print(
                    f"error: duplicate row id {rid!r} between {all_ids[rid]} and {subset}",
                    file=sys.stderr,
                )
                return 1
            all_ids[rid] = subset

    output_dir = args.output_dir
    output_paths: dict[str, Path] = {}
    for subset, rows in rows_by_subset.items():
        path = output_dir / f"healthbench_{subset}.jsonl"
        _write_jsonl(path, rows)
        output_paths[subset] = path

    criteria_path = output_dir / "healthbench_criteria.jsonl"
    _write_jsonl(criteria_path, criteria_index.rows())

    total_item_count = sum(len(rows) for rows in rows_by_subset.values())
    total_rubric_items = sum(
        len(row["rubrics"]) for rows in rows_by_subset.values() for row in rows
    )
    counts_by_subset = {s: len(rs) for s, rs in rows_by_subset.items()}
    counts_by_category: dict[str, int] = {}
    for rows in rows_by_subset.values():
        for row in rows:
            cat = row["source"]["metadata"]["obgyn_classification"]["category"]
            counts_by_category[cat] = counts_by_category.get(cat, 0) + 1

    all_rows = [row for rows in rows_by_subset.values() for row in rows]
    validation = validate_rows(all_rows)

    manifest = {
        "benchmark_version": args.benchmark_version,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utcnow_iso(),
        "total_item_count": total_item_count,
        "counts_by_set_type": {"open_ended_rubric": total_item_count},
        "counts_by_source_dataset": {"HealthBench": total_item_count},
        "counts_by_subset": counts_by_subset,
        "counts_by_category": counts_by_category,
        "validation": validation,
        "rubric_stats": {
            "total_rubric_items": total_rubric_items,
            "unique_criteria": len(criteria_index.rows()),
            "per_level_counts": _per_level_counts(criteria_index),
            "per_axis_counts": _per_axis_counts(criteria_index),
        },
        "filter": {
            "type": "obgyn_classifier_verdict",
            "stats_by_subset": stats_by_subset,
        },
        "source_datasets": build_healthbench_source_metadata(),
        "outputs": {
            "rows_by_subset": {s: str(p) for s, p in output_paths.items()},
            "criteria_side_table": str(criteria_path),
        },
    }
    _write_json(args.manifest_output, manifest)

    print(
        json.dumps(
            {
                "ok": True,
                "rows_written": counts_by_subset,
                "unique_criteria": len(criteria_index.rows()),
                "criteria_side_table": str(criteria_path),
                "manifest_path": str(args.manifest_output),
                "filter_stats_by_subset": stats_by_subset,
                "counts_by_category": counts_by_category,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
