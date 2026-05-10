"""Project-level configuration for mamabench tooling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mamabench.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class ProjectConfig:
    """Current benchmark and schema paths loaded from mamabench.json."""

    benchmark_version: str
    schema_version: str
    schema_file: Path
    benchmark_dir: Path
    manifest_dir: Path
    inspection_dir: Path


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load and validate the root mamabench.json pointer file."""

    config_path = Path(path)
    payload = _read_config_payload(config_path)
    root = config_path.parent

    config = ProjectConfig(
        benchmark_version=normalize_benchmark_version(
            _required_string(payload, "benchmark_version")
        ),
        schema_version=_required_string(payload, "schema_version"),
        schema_file=root / _required_string(payload, "schema_file"),
        benchmark_dir=root / _required_string(payload, "benchmark_dir"),
        manifest_dir=root / _required_string(payload, "manifest_dir"),
        inspection_dir=root / _required_string(payload, "inspection_dir"),
    )

    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{config_path}: schema_version {config.schema_version!r} does not "
            f"match code schema version {SCHEMA_VERSION!r}"
        )

    if not config.schema_file.is_file():
        raise ValueError(
            f"{config_path}: schema_file does not exist: {config.schema_file}"
        )

    return config


def normalize_benchmark_version(value: str) -> str:
    """Return a canonical benchmark release label with a leading v."""

    version = value.strip()
    if not version:
        raise ValueError("benchmark_version must be a non-empty string")
    if version.startswith("v"):
        return version
    return f"v{version}"


def _read_config_payload(config_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path}: invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"{config_path}: expected a JSON object")
    return payload


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"mamabench.json: {field} must be a non-empty string")
    return value
