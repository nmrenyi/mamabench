"""Manifest and summary helpers for mamabench artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from mamabench.config import normalize_benchmark_version
from mamabench.schema import SCHEMA_VERSION
from mamabench.validate import ValidationReport


class ReleaseManifestError(ValueError):
    """Raised when per-source manifests cannot be aggregated."""


def build_manifest(
    items: Iterable[Mapping[str, Any]],
    *,
    benchmark_version: str,
    schema_version: str = SCHEMA_VERSION,
    source_dataset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    validation_report: ValidationReport | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build artifact summary fields shared by manifests and summaries."""

    rows = list(items)
    created = created_at or datetime.now(timezone.utc)

    return {
        "benchmark_version": normalize_benchmark_version(benchmark_version),
        "schema_version": schema_version,
        "created_at": _format_timestamp(created),
        "total_item_count": len(rows),
        "counts_by_set_type": _counts(rows, "set_type"),
        "counts_by_source_dataset": _source_counts(rows, "dataset"),
        "source_datasets": _source_dataset_notes(rows, source_dataset_metadata),
        "validation": _validation_report(validation_report),
    }


def build_release_manifest(
    per_source_manifests: Sequence[Mapping[str, Any]],
    *,
    benchmark_version: str,
    schema_version: str = SCHEMA_VERSION,
    manifest_paths: Sequence[str] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate per-source manifests into a top-level release manifest.

    Each ``per_source_manifests`` entry must be a manifest produced by
    :func:`build_manifest` (a single-source artifact summary). The result
    contains aggregated counts, a per-source breakdown with pointers to the
    individual manifest files (when ``manifest_paths`` is given), and an
    overall validation status that is OK only when every per-source manifest
    is OK.

    Raises :class:`ReleaseManifestError` when the per-source manifests
    disagree on ``benchmark_version`` or ``schema_version``.
    """

    expected_version = normalize_benchmark_version(benchmark_version)
    if manifest_paths is not None and len(manifest_paths) != len(per_source_manifests):
        raise ReleaseManifestError(
            "manifest_paths must have the same length as per_source_manifests"
        )

    aggregated_set_type: Counter[str] = Counter()
    aggregated_source: Counter[str] = Counter()
    total_item_count = 0
    total_error_count = 0
    all_validations_ok = True
    sources: dict[str, dict[str, Any]] = {}

    for index, manifest in enumerate(per_source_manifests):
        _check_release_consistency(manifest, expected_version, schema_version, index)

        for set_type, count in manifest.get("counts_by_set_type", {}).items():
            aggregated_set_type[set_type] += count
        for source_dataset, count in manifest.get("counts_by_source_dataset", {}).items():
            aggregated_source[source_dataset] += count

        total_item_count += manifest.get("total_item_count", 0)

        validation = manifest.get("validation") or {}
        ok = validation.get("ok")
        error_count = validation.get("error_count") or 0
        total_error_count += error_count
        if ok is False:
            all_validations_ok = False
        elif ok is None:
            all_validations_ok = False

        source_dataset = _primary_source_dataset(manifest, fallback=f"source_{index}")
        # When multiple manifests collapse onto the same primary source
        # dataset (e.g. AfriMed-QA MCQ + AfriMed-QA SAQ), disambiguate using
        # the manifest filename stem so the second entry doesn't silently
        # overwrite the first.
        if source_dataset in sources:
            if manifest_paths is not None:
                key = _stem_from_manifest_path(manifest_paths[index])
            else:
                key = f"{source_dataset}_{index}"
        else:
            key = source_dataset
        sources[key] = {
            "source_dataset": source_dataset,
            "item_count": manifest.get("total_item_count", 0),
            "validation": {
                "ok": ok,
                "item_count": validation.get("item_count"),
                "error_count": error_count,
            },
        }
        if manifest_paths is not None:
            sources[key]["manifest_path"] = manifest_paths[index]

    created = created_at or datetime.now(timezone.utc)
    return {
        "benchmark_version": expected_version,
        "schema_version": schema_version,
        "created_at": _format_timestamp(created),
        "total_item_count": total_item_count,
        "counts_by_set_type": dict(sorted(aggregated_set_type.items())),
        "counts_by_source_dataset": dict(sorted(aggregated_source.items())),
        "sources": {key: sources[key] for key in sorted(sources)},
        "validation": {
            "ok": all_validations_ok,
            "total_item_count": total_item_count,
            "total_error_count": total_error_count,
            "source_count": len(per_source_manifests),
        },
    }


def _check_release_consistency(
    manifest: Mapping[str, Any],
    expected_version: str,
    expected_schema_version: str,
    index: int,
) -> None:
    actual_version = manifest.get("benchmark_version")
    if actual_version != expected_version:
        raise ReleaseManifestError(
            f"per_source_manifests[{index}]: benchmark_version "
            f"{actual_version!r} does not match release version "
            f"{expected_version!r}"
        )
    actual_schema = manifest.get("schema_version")
    if actual_schema != expected_schema_version:
        raise ReleaseManifestError(
            f"per_source_manifests[{index}]: schema_version "
            f"{actual_schema!r} does not match release schema version "
            f"{expected_schema_version!r}"
        )


def _primary_source_dataset(
    manifest: Mapping[str, Any], *, fallback: str
) -> str:
    counts = manifest.get("counts_by_source_dataset") or {}
    if not counts:
        return fallback
    # Pick the source dataset with the most rows; ties broken by name.
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _stem_from_manifest_path(manifest_path: str) -> str:
    """Strip directory + ``_manifest.json`` from a manifest path for use as a key.

    e.g. ``"manifests/afrimedqa_saq_manifest.json"`` → ``"afrimedqa_saq"``.
    """
    from pathlib import Path

    name = Path(manifest_path).stem
    if name.endswith("_manifest"):
        name = name[: -len("_manifest")]
    return name


def _counts(rows: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[_counter_key(row.get(field))] += 1
    return dict(sorted(counter.items()))


def _source_dataset_notes(
    rows: list[Mapping[str, Any]],
    source_dataset_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    notes: dict[str, dict[str, Any]] = defaultdict(dict)

    for row in rows:
        source = row.get("source")
        source_dataset = _source_value(source, "dataset")
        notes[source_dataset]

    for source_dataset, metadata in (source_dataset_metadata or {}).items():
        if not isinstance(metadata, Mapping):
            continue
        note = notes[_counter_key(source_dataset)]
        for key, value in metadata.items():
            if isinstance(value, str):
                note[key] = value.strip() or None
            else:
                note[key] = value

    return {
        source_dataset: values
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


def _validation_report(report: ValidationReport | None) -> dict[str, Any]:
    if report is None:
        return {"ok": None, "item_count": None, "error_count": None, "issues": []}
    return report.to_dict()


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
