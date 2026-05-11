from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.medqa_usmle import (
    CONTENT_HASH_LENGTH,
    MedQAUSMLEAdapterError,
    build_medqa_usmle_source_metadata,
    load_medqa_usmle_tsv,
    normalize_medqa_usmle_row,
    parse_options,
)
from mamabench.validate import validate_items

import _adapter_provenance_tests as provenance


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "medqa_usmle_obgyn_sample.tsv"
BENCHMARK_VERSION = "v0.1"
HASH_PATTERN = re.compile(rf"^[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$")
ID_PATTERN = re.compile(
    rf"^mamabench_v0\.1_medqa_usmle_[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$"
)
PROVENANCE_CASE = dict(
    dataset_name="MedQA-USMLE",
    expected_path="medqa-usmle/data/obgyn_usmle.tsv",
    tsv_header=(
        "question\toptions_formatted\tcorrect_letter\tanswer\tcategory\tmeta_info\n"
    ),
    build_metadata=build_medqa_usmle_source_metadata,
)


class MedQAUSMLEAdapterTests(unittest.TestCase):
    def test_source_metadata_records_prepared_input_commit(self) -> None:
        provenance.assert_verified_prepared_input(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_verify_unexpected_remote(self) -> None:
        provenance.assert_unexpected_remote(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_verify_noncanonical_repo_path(self) -> None:
        provenance.assert_noncanonical_path(self, **PROVENANCE_CASE)

    def test_source_metadata_does_not_claim_random_local_input_path(self) -> None:
        provenance.assert_random_local_input_path(self, **PROVENANCE_CASE)

    def test_load_fixture_emits_valid_rows(self) -> None:
        rows = load_medqa_usmle_tsv(
            FIXTURE,
            benchmark_version=BENCHMARK_VERSION,
        )

        report = validate_items(rows)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(len(rows), 4)

    def test_normalizes_answer_text_index_and_source_answer_key(self) -> None:
        rows = load_medqa_usmle_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)
        row = rows[1]

        self.assertEqual(row["source"]["answer"], "B")
        self.assertEqual(row["answer_index"], 1)
        self.assertEqual(row["answer"], "MRI of the pituitary")
        self.assertEqual(
            row["choices"],
            [
                "Pelvic ultrasound",
                "MRI of the pituitary",
                "Thyroid biopsy",
                "Endometrial biopsy",
            ],
        )

    def test_preserves_minimal_medqa_usmle_source(self) -> None:
        row = load_medqa_usmle_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)[1]

        self.assertEqual(row["schema_version"], "0.3")
        self.assertRegex(row["id"], ID_PATTERN)
        self.assertEqual(row["source"]["dataset"], "MedQA-USMLE")
        self.assertRegex(row["source"]["id"], HASH_PATTERN)
        self.assertTrue(row["id"].endswith(row["source"]["id"]))
        self.assertNotIn("license", row["source"])
        self.assertNotIn("url", row["source"])

    def test_does_not_emit_category_or_meta_info_fields(self) -> None:
        row = load_medqa_usmle_tsv(FIXTURE, benchmark_version=BENCHMARK_VERSION)[0]

        self.assertNotIn("category", row)
        self.assertNotIn("meta_info", row)
        self.assertNotIn("category", row["source"])
        self.assertNotIn("meta_info", row["source"])

    def test_handles_more_than_four_options(self) -> None:
        row = normalize_medqa_usmle_row(
            {
                "question": "Five-option question?",
                "options_formatted": (
                    "A. First | B. Second | C. Third | D. Fourth | E. Fifth"
                ),
                "correct_letter": "E",
                "answer": "Fifth",
            },
            row_number=42,
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertEqual(row["answer"], "Fifth")
        self.assertEqual(row["answer_index"], 4)
        self.assertEqual(len(row["choices"]), 5)
        self.assertRegex(row["id"], ID_PATTERN)
        self.assertTrue(row["id"].endswith(row["source"]["id"]))

    def test_content_hash_is_stable_across_row_number(self) -> None:
        source_row = {
            "question": "Stable hash question?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "C",
            "answer": "Third",
        }

        at_row_1 = normalize_medqa_usmle_row(
            source_row, row_number=1, benchmark_version=BENCHMARK_VERSION
        )
        at_row_500 = normalize_medqa_usmle_row(
            source_row, row_number=500, benchmark_version=BENCHMARK_VERSION
        )

        self.assertEqual(at_row_1["id"], at_row_500["id"])
        self.assertEqual(at_row_1["source"]["id"], at_row_500["source"]["id"])

    def test_content_hash_is_stable_across_option_permutation(self) -> None:
        base_row = normalize_medqa_usmle_row(
            {
                "question": "Permutation question?",
                "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
                "correct_letter": "C",
                "answer": "Third",
            },
            row_number=1,
            benchmark_version=BENCHMARK_VERSION,
        )
        permuted_row = normalize_medqa_usmle_row(
            {
                "question": "Permutation question?",
                "options_formatted": "A. Second | B. Fourth | C. Third | D. First",
                "correct_letter": "C",
                "answer": "Third",
            },
            row_number=2,
            benchmark_version=BENCHMARK_VERSION,
        )

        self.assertEqual(base_row["source"]["id"], permuted_row["source"]["id"])

    def test_duplicate_content_in_same_dataset_is_flagged_by_validator(self) -> None:
        source_row = {
            "question": "Duplicate-content question?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "A",
            "answer": "First",
        }

        first = normalize_medqa_usmle_row(
            source_row, row_number=1, benchmark_version=BENCHMARK_VERSION
        )
        second = normalize_medqa_usmle_row(
            source_row, row_number=2, benchmark_version=BENCHMARK_VERSION
        )

        report = validate_items([first, second])

        self.assertFalse(report.ok)
        fields_with_issues = {issue.field for issue in report.issues}
        self.assertIn("id", fields_with_issues)
        self.assertIn("source.id", fields_with_issues)

    def test_limit_caps_loaded_rows(self) -> None:
        rows = load_medqa_usmle_tsv(
            FIXTURE,
            benchmark_version=BENCHMARK_VERSION,
            limit=2,
        )

        self.assertEqual(len(rows), 2)

    def test_limit_zero_returns_no_rows(self) -> None:
        rows = load_medqa_usmle_tsv(
            FIXTURE,
            benchmark_version=BENCHMARK_VERSION,
            limit=0,
        )

        self.assertEqual(rows, [])

    def test_negative_limit_fails(self) -> None:
        with self.assertRaisesRegex(
            MedQAUSMLEAdapterError, "limit must be non-negative"
        ):
            load_medqa_usmle_tsv(
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
                    str(ROOT / "scripts" / "adapt_medqa_usmle.py"),
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
                    str(ROOT / "scripts" / "adapt_medqa_usmle.py"),
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
            self.assertTrue(row["id"].startswith("mamabench_v0.2_medqa_usmle_"))
            self.assertEqual(manifest["benchmark_version"], "v0.2")

    def test_parse_options_preserves_pipes_inside_option_text(self) -> None:
        options = parse_options(
            "A. | FSH | B. | Cholesterol | C. | Androgen | D. | Estrogen",
            row_number=1,
        )

        self.assertEqual(options["A"], "FSH")
        self.assertEqual(options["B"], "Cholesterol")

    def test_parse_options_rejects_missing_markers(self) -> None:
        with self.assertRaisesRegex(
            MedQAUSMLEAdapterError, "options_formatted is empty"
        ):
            parse_options("First | Second", row_number=1)

    def test_unknown_answer_letter_fails(self) -> None:
        row = {
            "question": "Which answer is correct?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "E",
            "answer": "Fifth",
            "category": "OBSTETRICS",
            "meta_info": "step2",
        }

        with self.assertRaisesRegex(MedQAUSMLEAdapterError, "correct_letter"):
            normalize_medqa_usmle_row(
                row,
                row_number=1,
                benchmark_version=BENCHMARK_VERSION,
            )

    def test_mismatched_source_answer_text_fails(self) -> None:
        row = {
            "question": "Which answer is correct?",
            "options_formatted": "A. First | B. Second | C. Third | D. Fourth",
            "correct_letter": "B",
            "answer": "Third",
            "category": "OBSTETRICS",
            "meta_info": "step2",
        }

        with self.assertRaisesRegex(
            MedQAUSMLEAdapterError, "does not match parsed option"
        ):
            normalize_medqa_usmle_row(
                row,
                row_number=1,
                benchmark_version=BENCHMARK_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
