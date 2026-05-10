"""Manifest and summary helpers for mamabench artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from mamabench.schema import SCHEMA_VERSION
from mamabench.validate import ValidationReport


def build_manifest(
    items: Iterable[Mapping[str, Any]],
    *,
    benchmark_version: str = "v0.2",
    schema_version: str = SCHEMA_VERSION,
    validation_report: ValidationReport | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a manifest-style summary for a collection of benchmark items."""

    rows = list(items)
    created = created_at or datetime.now(timezone.utc)

    return {
        "benchmark_version": benchmark_version,
        "schema_version": schema_version,
        "created_at": _format_timestamp(created),
        "total_item_count": len(rows),
        "counts_by_set_type": _counts(rows, "set_type"),
        "counts_by_source_dataset": _source_counts(rows, "dataset"),
        "source_datasets": _source_dataset_notes(rows),
        "validation": _validation_summary(validation_report),
    }


def _counts(rows: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[_counter_key(row.get(field))] += 1
    return dict(sorted(counter.items()))


def _source_dataset_notes(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    notes: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"licenses": set()})

    for row in rows:
        source = row.get("source")
        source_dataset = _source_value(source, "dataset")
        license_value = source.get("license") if isinstance(source, Mapping) else None
        if isinstance(license_value, str) and license_value.strip():
            notes[source_dataset]["licenses"].add(license_value)

    return {
        source_dataset: {
            "licenses": sorted(values["licenses"]),
        }
        for source_dataset, values in sorted(notes.items())
    }


def _source_counts(rows: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[_source_value(row.get("source"), field)] += 1
    return dict(sorted(counter.items()))


def _source_value(source: Any, field: str) -> str:
    if not isinstance(source, Mapping):
        return "null"
    return _counter_key(source.get(field))


def _validation_summary(report: ValidationReport | None) -> dict[str, Any]:
    if report is None:
        return {"ok": None, "error_count": None}
    return {"ok": report.ok, "error_count": report.error_count}


def _counter_key(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "null"
    return str(value)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
