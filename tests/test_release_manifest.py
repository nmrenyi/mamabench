from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mamabench.manifest import ReleaseManifestError, build_release_manifest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_VERSION = "v0.1"
SCHEMA_VERSION = "0.3"


def _per_source_manifest(
    *,
    dataset: str,
    item_count: int,
    error_count: int = 0,
    ok: bool = True,
    set_type: str = "mcq",
    benchmark_version: str = BENCHMARK_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    return {
        "benchmark_version": benchmark_version,
        "schema_version": schema_version,
        "created_at": "2026-05-11T00:00:00Z",
        "total_item_count": item_count,
        "counts_by_set_type": {set_type: item_count},
        "counts_by_source_dataset": {dataset: item_count},
        "source_datasets": {dataset: {}},
        "validation": {
            "ok": ok,
            "item_count": item_count,
            "error_count": error_count,
            "issues": [],
        },
    }


class BuildReleaseManifestTests(unittest.TestCase):
    def test_aggregates_counts_across_sources(self) -> None:
        release = build_release_manifest(
            [
                _per_source_manifest(dataset="MedMCQA", item_count=18508),
                _per_source_manifest(dataset="MedQA-USMLE", item_count=1025),
                _per_source_manifest(dataset="AfriMed-QA", item_count=534),
            ],
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertEqual(release["benchmark_version"], BENCHMARK_VERSION)
        self.assertEqual(release["schema_version"], SCHEMA_VERSION)
        self.assertEqual(release["total_item_count"], 18508 + 1025 + 534)
        self.assertEqual(release["counts_by_set_type"], {"mcq": 18508 + 1025 + 534})
        self.assertEqual(
            release["counts_by_source_dataset"],
            {"AfriMed-QA": 534, "MedMCQA": 18508, "MedQA-USMLE": 1025},
        )
        self.assertEqual(set(release["sources"]), {"MedMCQA", "MedQA-USMLE", "AfriMed-QA"})
        self.assertEqual(release["sources"]["MedMCQA"]["item_count"], 18508)
        self.assertTrue(release["validation"]["ok"])
        self.assertEqual(release["validation"]["total_error_count"], 0)
        self.assertEqual(release["validation"]["source_count"], 3)

    def test_propagates_validation_failure(self) -> None:
        release = build_release_manifest(
            [
                _per_source_manifest(dataset="MedMCQA", item_count=10),
                _per_source_manifest(
                    dataset="AfriMed-QA", item_count=5, error_count=2, ok=False
                ),
            ],
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertFalse(release["validation"]["ok"])
        self.assertEqual(release["validation"]["total_error_count"], 2)

    def test_rejects_benchmark_version_mismatch(self) -> None:
        manifests = [
            _per_source_manifest(dataset="MedMCQA", item_count=10),
            _per_source_manifest(
                dataset="AfriMed-QA",
                item_count=5,
                benchmark_version="v0.2",
            ),
        ]

        with self.assertRaisesRegex(ReleaseManifestError, "benchmark_version"):
            build_release_manifest(manifests, benchmark_version=BENCHMARK_VERSION)

    def test_rejects_schema_version_mismatch(self) -> None:
        manifests = [
            _per_source_manifest(dataset="MedMCQA", item_count=10),
            _per_source_manifest(
                dataset="AfriMed-QA", item_count=5, schema_version="0.4"
            ),
        ]

        with self.assertRaisesRegex(ReleaseManifestError, "schema_version"):
            build_release_manifest(manifests, benchmark_version=BENCHMARK_VERSION)

    def test_threads_manifest_paths_into_sources_block(self) -> None:
        release = build_release_manifest(
            [
                _per_source_manifest(dataset="MedMCQA", item_count=10),
                _per_source_manifest(dataset="AfriMed-QA", item_count=5),
            ],
            benchmark_version=BENCHMARK_VERSION,
            manifest_paths=["medmcqa_manifest.json", "afrimedqa_manifest.json"],
        )

        self.assertEqual(
            release["sources"]["MedMCQA"]["manifest_path"], "medmcqa_manifest.json"
        )
        self.assertEqual(
            release["sources"]["AfriMed-QA"]["manifest_path"],
            "afrimedqa_manifest.json",
        )

    def test_rejects_manifest_paths_length_mismatch(self) -> None:
        manifests = [_per_source_manifest(dataset="MedMCQA", item_count=10)]

        with self.assertRaisesRegex(ReleaseManifestError, "same length"):
            build_release_manifest(
                manifests,
                benchmark_version=BENCHMARK_VERSION,
                manifest_paths=["a.json", "b.json"],
            )


class BuildReleaseManifestCliTests(unittest.TestCase):
    def test_cli_writes_aggregated_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            a_path = tmpdir_path / "medmcqa_manifest.json"
            b_path = tmpdir_path / "afrimedqa_manifest.json"
            output_path = tmpdir_path / "release_manifest.json"

            a_path.write_text(
                json.dumps(_per_source_manifest(dataset="MedMCQA", item_count=10))
            )
            b_path.write_text(
                json.dumps(_per_source_manifest(dataset="AfriMed-QA", item_count=5))
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_release_manifest.py"),
                    str(a_path),
                    str(b_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            release = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(release["total_item_count"], 15)
            self.assertEqual(
                release["sources"]["MedMCQA"]["manifest_path"],
                "medmcqa_manifest.json",
            )
            self.assertEqual(
                release["sources"]["AfriMed-QA"]["manifest_path"],
                "afrimedqa_manifest.json",
            )
            self.assertTrue(release["validation"]["ok"])

    def test_cli_returns_nonzero_when_a_per_source_manifest_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            failed_path = tmpdir_path / "broken_manifest.json"
            output_path = tmpdir_path / "release_manifest.json"

            failed_path.write_text(
                json.dumps(
                    _per_source_manifest(
                        dataset="AfriMed-QA",
                        item_count=5,
                        error_count=3,
                        ok=False,
                    )
                )
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_release_manifest.py"),
                    str(failed_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            release = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(release["validation"]["ok"])
            self.assertEqual(release["validation"]["total_error_count"], 3)


if __name__ == "__main__":
    unittest.main()
