from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.afrimedqa import (
    CONTENT_HASH_LENGTH,
    AfriMedQAAdapterError,
    AfriMedQAFilterStats,
    build_afrimedqa_source_metadata,
    load_afrimedqa_tsv,
    normalize_afrimedqa_row,
    parse_options,
)
from mamabench.validate import validate_items


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "afrimedqa_obgyn_sample.tsv"
BENCHMARK_VERSION = "v0.1"
HASH_PATTERN = re.compile(rf"^[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$")
ID_PATTERN = re.compile(
    rf"^mamabench_v0\.1_afrimedqa_[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$"
)
FIXTURE_HEADER = (
    "question_clean\toptions_formatted\tcorrect_letter\n"
)


def _init_repo_with_fixture(repo: Path, tsv_relpath: Path) -> str:
    """Initialize a throwaway git repo containing an empty AfriMed-QA TSV."""

    tsv = repo / tsv_relpath
    tsv.parent.mkdir(parents=True)
    tsv.write_text(FIXTURE_HEADER)

    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=mamabench",
            "-c",
            "user.email=mamabench@example.test",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


class AfriMedQAAdapterTests(unittest.TestCase):
    def test_source_metadata_records_prepared_input_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "obgyn-qa-collection"
            tsv_relpath = Path("afrimedqa") / "data" / "obgyn_mcq.tsv"

            tsv = repo / tsv_relpath
            tsv.parent.mkdir(parents=True)
            tsv.write_text(FIXTURE_HEADER)

            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(
                [
                    "git",
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/nmrenyi/obgyn-qa-collection.git",
                ],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=mamabench",
                    "-c",
                    "user.email=mamabench@example.test",
                    "commit",
                    "-m",
                    "fixture",
                ],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
            )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()

            metadata = build_afrimedqa_source_metadata(tsv)
            prepared = metadata["AfriMed-QA"]["prepared_input"]

            self.assertEqual(
                prepared["repository"],
                "https://github.com/nmrenyi/obgyn-qa-collection",
            )
            self.assertEqual(prepared["path"], "afrimedqa/data/obgyn_mcq.tsv")
            self.assertEqual(prepared["commit"], commit)
            self.assertFalse(prepared["git_dirty"])
            self.assertTrue(prepared["verified"])
            self.assertNotIn("expected", prepared)
            self.assertNotIn("actual", prepared)

    def test_source_metadata_does_not_verify_unexpected_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "obgyn-qa-collection"
            tsv_relpath = Path("afrimedqa") / "data" / "obgyn_mcq.tsv"
            tsv = repo / tsv_relpath
            tsv.parent.mkdir(parents=True)
            tsv.write_text(FIXTURE_HEADER)

            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(
                [
                    "git",
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/example/obgyn-qa-collection.git",
                ],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=mamabench",
                    "-c",
                    "user.email=mamabench@example.test",
                    "commit",
                    "-m",
                    "fixture",
                ],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
            )

            metadata = build_afrimedqa_source_metadata(tsv)
            prepared = metadata["AfriMed-QA"]["prepared_input"]

            self.assertEqual(
                prepared["expected"]["repository"],
                "https://github.com/nmrenyi/obgyn-qa-collection",
            )
            self.assertEqual(
                prepared["expected"]["path"], "afrimedqa/data/obgyn_mcq.tsv"
            )
            self.assertEqual(
                prepared["actual"]["repository"],
                "https://github.com/example/obgyn-qa-collection.git",
            )
            self.assertEqual(
                prepared["actual"]["path"], "afrimedqa/data/obgyn_mcq.tsv"
            )
            self.assertFalse(prepared["verified"])
            self.assertNotIn("repository", prepared)
            self.assertNotIn("path", prepared)
            self.assertNotIn("commit", prepared)
            self.assertNotIn("git_dirty", prepared)

    def test_source_metadata_does_not_verify_noncanonical_repo_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "obgyn-qa-collection"
            _init_repo_with_fixture(repo, Path("other") / "source.tsv")
            metadata = build_afrimedqa_source_metadata(repo / "other" / "source.tsv")
            prepared = metadata["AfriMed-QA"]["prepared_input"]

            self.assertEqual(
                prepared["expected"]["path"], "afrimedqa/data/obgyn_mcq.tsv"
            )
            self.assertEqual(prepared["actual"]["path"], "other/source.tsv")
            self.assertFalse(prepared["verified"])
            self.assertNotIn("path", prepared)
            self.assertNotIn("commit", prepared)
            self.assertNotIn("git_dirty", prepared)

    def test_source_metadata_does_not_claim_random_local_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv = Path(tmpdir) / "source.tsv"
            tsv.write_text(FIXTURE_HEADER)

            metadata = build_afrimedqa_source_metadata(tsv)
            prepared = metadata["AfriMed-QA"]["prepared_input"]

            self.assertEqual(
                prepared["expected"]["path"], "afrimedqa/data/obgyn_mcq.tsv"
            )
            self.assertFalse(prepared["verified"])
            self.assertNotIn("actual", prepared)
            self.assertNotIn("path", prepared)
            self.assertNotIn("commit", prepared)
            self.assertNotIn("git_dirty", prepared)

    def test_source_metadata_carries_noncommercial_license_notes(self) -> None:
        metadata = build_afrimedqa_source_metadata(None)
        block = metadata["AfriMed-QA"]

        self.assertEqual(block["license"], "CC BY-NC-SA 4.0")
        self.assertIn("Non-commercial", block["license_notes"])
        self.assertIn("share-alike", block["license_notes"])

    def test_source_metadata_includes_filter_stats_when_provided(self) -> None:
        stats = AfriMedQAFilterStats(
            total_source_rows=10,
            single_answer_rows=8,
            multi_answer_rows_skipped=2,
        )

        metadata = build_afrimedqa_source_metadata(None, filter_stats=stats)

        self.assertEqual(
            metadata["AfriMed-QA"]["filter"],
            {
                "total_source_rows": 10,
                "single_answer_rows": 8,
                "multi_answer_rows_skipped": 2,
            },
        )

    def test_source_metadata_omits_filter_block_when_stats_missing(self) -> None:
        metadata = build_afrimedqa_source_metadata(None)

        self.assertNotIn("filter", metadata["AfriMed-QA"])

    def test_load_fixture_emits_valid_rows(self) -> None:
        rows, _ = load_afrimedqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)

        report = validate_items(rows)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(len(rows), 4)

    def test_load_fixture_filter_stats_record_multi_answer_skip(self) -> None:
        _, stats = load_afrimedqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)

        self.assertEqual(
            stats,
            AfriMedQAFilterStats(
                total_source_rows=5,
                single_answer_rows=4,
                multi_answer_rows_skipped=1,
            ),
        )

    def test_normalizes_answer_text_index_and_source_answer_key(self) -> None:
        rows, _ = load_afrimedqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)
        row = rows[1]

        self.assertEqual(row["source"]["answer"], "B")
        self.assertEqual(row["answer_index"], 1)
        self.assertEqual(row["answer"], "Chorionic villus sampling.")
        self.assertEqual(len(row["choices"]), 5)
        self.assertTrue(row["choices"][1].startswith("Chorionic"))

    def test_preserves_minimal_afrimedqa_source(self) -> None:
        rows, _ = load_afrimedqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)
        row = rows[0]

        self.assertEqual(row["schema_version"], "0.3")
        self.assertRegex(row["id"], ID_PATTERN)
        self.assertEqual(row["source"]["dataset"], "AfriMed-QA")
        self.assertRegex(row["source"]["id"], HASH_PATTERN)
        self.assertTrue(row["id"].endswith(row["source"]["id"]))
        self.assertNotIn("license", row["source"])
        self.assertNotIn("url", row["source"])

    def test_does_not_emit_source_specific_input_fields(self) -> None:
        rows, _ = load_afrimedqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)
        row = rows[0]

        # `question_clean` must be normalized to canonical `question`.
        self.assertNotIn("question_clean", row)
        self.assertNotIn("options_formatted", row)
        self.assertNotIn("correct_letter", row)

    def test_handles_five_options(self) -> None:
        row = normalize_afrimedqa_row(
            {
                "question_clean": "Five-option question?",
                "options_formatted": (
                    "A. First | B. Second | C. Third | D. Fourth | E. Fifth"
                ),
                "correct_letter": "E",
            },
            row_number=1,
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertEqual(row["answer"], "Fifth")
        self.assertEqual(row["answer_index"], 4)
        self.assertEqual(len(row["choices"]), 5)

    def test_content_hash_is_stable_across_row_number(self) -> None:
        source_row = {
            "question_clean": "Stable hash question?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "C",
        }

        at_row_1 = normalize_afrimedqa_row(
            source_row, row_number=1, benchmark_version=BENCHMARK_VERSION
        )
        at_row_500 = normalize_afrimedqa_row(
            source_row, row_number=500, benchmark_version=BENCHMARK_VERSION
        )

        self.assertEqual(at_row_1["id"], at_row_500["id"])
        self.assertEqual(at_row_1["source"]["id"], at_row_500["source"]["id"])

    def test_content_hash_is_stable_across_option_permutation(self) -> None:
        base = normalize_afrimedqa_row(
            {
                "question_clean": "Permutation question?",
                "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
                "correct_letter": "C",
            },
            row_number=1,
            benchmark_version=BENCHMARK_VERSION,
        )
        permuted = normalize_afrimedqa_row(
            {
                "question_clean": "Permutation question?",
                "options_formatted": "A. Second | B. Fourth | C. Third | D. First",
                "correct_letter": "C",
            },
            row_number=2,
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertEqual(base["source"]["id"], permuted["source"]["id"])

    def test_duplicate_content_in_same_dataset_is_flagged_by_validator(self) -> None:
        source_row = {
            "question_clean": "Duplicate-content question?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "A",
        }

        first = normalize_afrimedqa_row(
            source_row, row_number=1, benchmark_version=BENCHMARK_VERSION
        )
        second = normalize_afrimedqa_row(
            source_row, row_number=2, benchmark_version=BENCHMARK_VERSION
        )

        report = validate_items([first, second])

        self.assertFalse(report.ok)
        fields_with_issues = {issue.field for issue in report.issues}
        self.assertIn("id", fields_with_issues)
        self.assertIn("source.id", fields_with_issues)

    def test_limit_caps_loaded_rows(self) -> None:
        rows, stats = load_afrimedqa_tsv(
            FIXTURE, benchmark_version=BENCHMARK_VERSION, limit=2
        )

        self.assertEqual(len(rows), 2)
        # First two source rows are both single-answer, so no multi-answer was scanned.
        self.assertEqual(stats.total_source_rows, 2)
        self.assertEqual(stats.multi_answer_rows_skipped, 0)

    def test_limit_zero_returns_no_rows(self) -> None:
        rows, stats = load_afrimedqa_tsv(
            FIXTURE, benchmark_version=BENCHMARK_VERSION, limit=0
        )

        self.assertEqual(rows, [])
        self.assertEqual(stats.total_source_rows, 0)
        self.assertEqual(stats.single_answer_rows, 0)
        self.assertEqual(stats.multi_answer_rows_skipped, 0)

    def test_negative_limit_fails(self) -> None:
        with self.assertRaisesRegex(
            AfriMedQAAdapterError, "limit must be non-negative"
        ):
            load_afrimedqa_tsv(
                FIXTURE, benchmark_version=BENCHMARK_VERSION, limit=-1
            )

    def test_cli_limit_zero_writes_empty_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "out.jsonl"
            manifest_json = Path(tmpdir) / "manifest.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "adapt_afrimedqa.py"),
                    str(FIXTURE),
                    str(output_jsonl),
                    "--limit",
                    "0",
                    "--manifest-output",
                    str(manifest_json),
                ],
                check=False,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output_jsonl.read_text(encoding="utf-8"), "")
            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_version"], BENCHMARK_VERSION)
            self.assertEqual(manifest["total_item_count"], 0)
            self.assertTrue(manifest["validation"]["ok"])

    def test_cli_normalizes_benchmark_version_in_rows_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "out.jsonl"
            manifest_json = Path(tmpdir) / "manifest.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "adapt_afrimedqa.py"),
                    str(FIXTURE),
                    str(output_jsonl),
                    "--benchmark-version",
                    "0.2",
                    "--limit",
                    "1",
                    "--manifest-output",
                    str(manifest_json),
                ],
                check=False,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(output_jsonl.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertTrue(row["id"].startswith("mamabench_v0.2_afrimedqa_"))
            self.assertEqual(manifest["benchmark_version"], "v0.2")

    def test_cli_manifest_records_filter_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "out.jsonl"
            manifest_json = Path(tmpdir) / "manifest.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "adapt_afrimedqa.py"),
                    str(FIXTURE),
                    str(output_jsonl),
                    "--manifest-output",
                    str(manifest_json),
                ],
                check=False,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            block = manifest["source_datasets"]["AfriMed-QA"]
            self.assertEqual(
                block["filter"],
                {
                    "total_source_rows": 5,
                    "single_answer_rows": 4,
                    "multi_answer_rows_skipped": 1,
                },
            )
            self.assertEqual(block["license"], "CC BY-NC-SA 4.0")
            self.assertIn("Non-commercial", block["license_notes"])

    def test_parse_options_preserves_pipes_inside_option_text(self) -> None:
        options = parse_options(
            "A. | First | B. | Second | C. | Third | D. | Fourth",
            row_number=1,
        )

        self.assertEqual(options["A"], "First")
        self.assertEqual(options["B"], "Second")

    def test_parse_options_rejects_missing_markers(self) -> None:
        with self.assertRaisesRegex(
            AfriMedQAAdapterError, "options_formatted is empty"
        ):
            parse_options("First | Second", row_number=1)

    def test_unknown_answer_letter_fails(self) -> None:
        row = {
            "question_clean": "Which answer is correct?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "Z",
        }

        with self.assertRaisesRegex(AfriMedQAAdapterError, "correct_letter"):
            normalize_afrimedqa_row(
                row, row_number=1, benchmark_version=BENCHMARK_VERSION
            )

    def test_normalize_refuses_multi_answer_row(self) -> None:
        row = {
            "question_clean": "Which are correct?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "A,B,C",
        }

        with self.assertRaisesRegex(AfriMedQAAdapterError, "multi-answer"):
            normalize_afrimedqa_row(
                row, row_number=1, benchmark_version=BENCHMARK_VERSION
            )

    def test_multi_answer_detection_treats_space_padded_values_as_multi(self) -> None:
        rows, stats = load_afrimedqa_tsv(
            self._tmp_tsv_with_correct_letter("A, B"),
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertEqual(rows, [])
        self.assertEqual(stats.multi_answer_rows_skipped, 1)

    def test_empty_correct_letter_raises(self) -> None:
        path = self._tmp_tsv_with_correct_letter("")

        with self.assertRaisesRegex(AfriMedQAAdapterError, "missing correct_letter"):
            load_afrimedqa_tsv(path, benchmark_version=BENCHMARK_VERSION)

    def _tmp_tsv_with_correct_letter(self, correct_letter: str) -> Path:
        tmp = Path(tempfile.mkstemp(suffix=".tsv")[1])
        self.addCleanup(tmp.unlink)
        tmp.write_text(
            FIXTURE_HEADER
            + (
                "Multi-answer or empty?\t"
                "A. First | B. Second | C. Third | D. Fourth\t"
                f"{correct_letter}\n"
            ),
            encoding="utf-8",
        )
        return tmp


if __name__ == "__main__":
    unittest.main()
