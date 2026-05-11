#!/usr/bin/env python3
"""Aggregate per-source manifests into a release-level manifest."""

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
from mamabench.io import write_json  # noqa: E402
from mamabench.manifest import ReleaseManifestError, build_release_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "per_source_manifests",
        nargs="+",
        help="Paths to per-source manifest JSON files to aggregate.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the aggregated release manifest JSON.",
    )
    parser.add_argument(
        "--benchmark-version",
        default=None,
        help=(
            "Benchmark version this release should match. "
            "Default: benchmark_version from mamabench.json."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "Optional root used to make per-source manifest paths relative in "
            "the output. Defaults to the parent directory of --output."
        ),
    )
    args = parser.parse_args(argv)

    try:
        config = load_project_config(ROOT / "mamabench.json")
        benchmark_version = normalize_benchmark_version(
            args.benchmark_version or config.benchmark_version
        )

        manifest_paths = [Path(path) for path in args.per_source_manifests]
        per_source_manifests = [_read_manifest(path) for path in manifest_paths]

        manifest_root = (
            Path(args.manifest_root).resolve()
            if args.manifest_root
            else Path(args.output).resolve().parent
        )
        relative_paths = [
            _relative_to(path.resolve(), manifest_root) for path in manifest_paths
        ]

        release = build_release_manifest(
            per_source_manifests,
            benchmark_version=benchmark_version,
            schema_version=config.schema_version,
            manifest_paths=relative_paths,
        )
    except (OSError, ReleaseManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_json(args.output, release)
    print(
        json.dumps(
            {
                "ok": release["validation"]["ok"],
                "output": args.output,
                "total_item_count": release["total_item_count"],
                "source_count": release["validation"]["source_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if release["validation"]["ok"] else 1


def _read_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseManifestError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ReleaseManifestError(f"{path}: expected a JSON object")
    return payload


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
