from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.medmcqa import (
    MedMCQAAdapterError,
    build_medmcqa_source_metadata,
    load_medmcqa_tsv,
    normalize_medmcqa_row,
    parse_options,
)
from mamabench.validate import validate_items

import _adapter_provenance_tests as provenance


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "medmcqa_obgyn_sample.tsv"
BENCHMARK_VERSION = "v0.1"
PROVENANCE_CASE = dict(
    dataset_name="MedMCQA",
    expected_path="medmcqa/data/obgyn_mcq.tsv",
    tsv_header="id\tquestion\toptions_formatted\tcorrect_letter\n",
    build_metadata=build_medmcqa_source_metadata,
)


class MedMCQAAdapterTests(unittest.TestCase):
    def test_source_metadata_records_prepared_input_commit(self) -> None:
        provenance.assert_verified_prepared_input(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_verify_unexpected_remote(self) -> None:
        provenance.assert_unexpected_remote(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_verify_missing_remote(self) -> None:
        provenance.assert_missing_remote(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_verify_noncanonical_repo_path(self) -> None:
        provenance.assert_noncanonical_path(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_claim_random_local_input_path(self) -> None:
        provenance.assert_random_local_input_path(self, **PROVENANCE_CASE)

    def test_load_fixture_emits_valid_rows(self) -> None:
        rows = load_medmcqa_tsv(
            FIXTURE,
            benchmark_version=BENCHMARK_VERSION,
        )

        report = validate_items(rows)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(len(rows), 4)

    def test_normalizes_answer_text_index_and_source_answer_key(self) -> None:
        rows = load_medmcqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)
        row = rows[1]

        self.assertEqual(row["source"]["answer"], "C")
        self.assertEqual(row["answer_index"], 2)
        self.assertEqual(row["answer"], "Levonorgestrel")
        self.assertEqual(
            row["choices"],
            ["OCP", "Danazol", "Levonorgestrel", "Mifepristone"],
        )

    def test_preserves_minimal_medmcqa_source(self) -> None:
        row = load_medmcqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)[1]

        self.assertEqual(row["schema_version"], "0.3")
        self.assertEqual(
            row["id"],
            "mamabench_v0.1_medmcqa_0036cad0-d22f-453c-b075-322479d19d6e",
        )
        self.assertEqual(row["source"]["dataset"], "MedMCQA")
        self.assertEqual(row["source"]["id"], "0036cad0-d22f-453c-b075-322479d19d6e")
        self.assertNotIn("license", row["source"])
        self.assertNotIn("url", row["source"])

    def test_does_not_emit_v0_1_label_fields(self) -> None:
        row = load_medmcqa_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)[1]

        self.assertNotIn("clinical_domain", row)
        self.assertNotIn("age_group", row)
        self.assertNotIn("task_type", row)
        self.assertNotIn("tags", row)
        self.assertNotIn("provenance", row)

    def test_limit_caps_loaded_rows(self) -> None:
        rows = load_medmcqa_tsv(
            FIXTURE,
            benchmark_version=BENCHMARK_VERSION,
            limit=2,
        )

        self.assertEqual(len(rows), 2)

    def test_limit_zero_returns_no_rows(self) -> None:
        rows = load_medmcqa_tsv(
            FIXTURE,
            benchmark_version=BENCHMARK_VERSION,
            limit=0,
        )

        self.assertEqual(rows, [])

    def test_negative_limit_fails(self) -> None:
        with self.assertRaisesRegex(MedMCQAAdapterError, "limit must be non-negative"):
            load_medmcqa_tsv(
                FIXTURE,
                benchmark_version=BENCHMARK_VERSION,
                limit=-1,
            )

    def test_cli_limit_zero_writes_empty_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "out.jsonl"
            manifest_json = Path(tmpdir) / "manifest.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "adapt_medmcqa.py"),
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
                    str(ROOT / "scripts" / "adapt_medmcqa.py"),
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
            self.assertTrue(row["id"].startswith("mamabench_v0.2_medmcqa_"))
            self.assertEqual(manifest["benchmark_version"], "v0.2")

    def test_parse_options_preserves_pipes_inside_option_text(self) -> None:
        options = parse_options(
            "A. | FSH | B. | Cholestsrol | C. | Androgen | D. | Cholesterol",
            row_number=1,
        )

        self.assertEqual(options["A"], "FSH")
        self.assertEqual(options["B"], "Cholestsrol")

    def test_parse_options_rejects_missing_markers(self) -> None:
        with self.assertRaisesRegex(MedMCQAAdapterError, "options_formatted is empty"):
            parse_options("First | Second", row_number=1)

    def test_unknown_answer_letter_fails(self) -> None:
        row = {
            "id": "bad-answer",
            "question": "Which answer is correct?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "E",
            "explanation": "",
            "subject": "Gynaecology & Obstetrics",
            "topic": "",
            "choice_type": "single",
            "split": "train",
        }

        with self.assertRaisesRegex(MedMCQAAdapterError, "correct_letter"):
            normalize_medmcqa_row(
                row,
                row_number=1,
                benchmark_version=BENCHMARK_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
