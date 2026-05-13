#!/usr/bin/env python3
"""Re-emit MedMCQA / AfriMed-MCQ rows for the v0.2 release at schema v0.4.

These two sources keep their v0.1 row content unchanged for v0.2 — the
upstream column-based filters are trusted, no OBGYN classifier verdict
filter is applied — but the v0.2 release uses schema_version "0.4" and
ids prefixed `mamabench_v0.2_...`. This script reuses the existing v0.1
adapter modules (`mamabench.adapters.medmcqa`,
`mamabench.adapters.afrimedqa`), passes `benchmark_version="v0.2"` so
the ids carry the v0.2 prefix, and stamps `schema_version="0.4"` on the
output rows.

Each row is then validated row-by-row against
`schemas/mamabench_v0.4.schema.json` (via jsonschema-py if available).
The existing `mamabench.validate.validate_items` is *not* used, because
it's still pinned to v0.3 — keeping that path intact avoids a
v0.1-breaking refactor of `mamabench.schema` / `mamabench.validate` for
the v0.2 release.
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

from mamabench.adapters.medmcqa import (  # noqa: E402
    build_medmcqa_source_metadata,
    load_medmcqa_tsv,
)
from mamabench.adapters.afrimedqa import (  # noqa: E402
    build_afrimedqa_source_metadata,
    load_afrimedqa_tsv,
)
from mamabench.adapters._v0_4_validation import validate_rows  # noqa: E402


TARGET_SCHEMA_VERSION = "0.4"


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
        "--source",
        choices=("medmcqa", "afrimedqa"),
        required=True,
        help="Which MCQ source to re-emit at v0.2 / schema 0.4.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source TSV (same input the v0.1 adapter took).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Default: benchmark/v0.2/<source>.jsonl",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Manifest JSON path. Default: benchmark/v0.2/manifests/<source>_manifest.json",
    )
    parser.add_argument("--benchmark-version", default="v0.2")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"error: --input not found: {args.input}", file=sys.stderr)
        return 2

    output_path = args.output or Path(f"benchmark/v0.2/{args.source}.jsonl")
    manifest_path = args.manifest_output or Path(
        f"benchmark/v0.2/manifests/{args.source}_manifest.json"
    )

    try:
        if args.source == "medmcqa":
            rows = load_medmcqa_tsv(
                args.input,
                benchmark_version=args.benchmark_version,
                limit=args.limit,
            )
            source_metadata = build_medmcqa_source_metadata(args.input)
            source_dataset_name = next(iter(source_metadata))
        else:
            rows, _afri_stats = load_afrimedqa_tsv(
                args.input,
                benchmark_version=args.benchmark_version,
                limit=args.limit,
            )
            source_metadata = build_afrimedqa_source_metadata(args.input)
            source_dataset_name = next(iter(source_metadata))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The adapter modules wrote `mamabench.schema.SCHEMA_VERSION` ("0.3") into
    # each row; stamp the v0.4 value before validation and write-out.
    for row in rows:
        row["schema_version"] = TARGET_SCHEMA_VERSION

    validation = validate_rows(rows)
    if not validation["ok"]:
        print("schema validation errors:", file=sys.stderr)
        for issue in validation["issues"][:10]:
            print(f"  row {issue['line_number']} id={issue['item_id']}: {issue['message']}", file=sys.stderr)
        return 1

    _write_jsonl(output_path, rows)

    manifest = {
        "benchmark_version": args.benchmark_version,
        "schema_version": TARGET_SCHEMA_VERSION,
        "created_at": _utcnow_iso(),
        "total_item_count": len(rows),
        "counts_by_set_type": {"mcq": len(rows)},
        "counts_by_source_dataset": {source_dataset_name: len(rows)},
        "validation": validation,
        "filter": {"type": "structural-upstream"},
        "source_datasets": source_metadata,
        "outputs": {"rows": str(output_path)},
        "release_notes": (
            f"v0.2 re-emits {source_dataset_name} from the same upstream TSV "
            "as v0.1 with no content changes — only the benchmark_version "
            "(v0.1 → v0.2) and schema_version (0.3 → 0.4) update."
        ),
    }
    _write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "ok": True,
                "source": args.source,
                "rows_written": len(rows),
                "output": str(output_path),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
