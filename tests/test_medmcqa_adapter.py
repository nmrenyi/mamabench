from __future__ import annotations

import unittest
from pathlib import Path

from mamabench.adapters.medmcqa import (
    MedMCQAAdapterError,
    load_medmcqa_tsv,
    normalize_medmcqa_row,
    parse_options,
)
from mamabench.validate import validate_items


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "medmcqa_obgyn_sample.tsv"


class MedMCQAAdapterTests(unittest.TestCase):
    def test_load_fixture_emits_valid_rows(self) -> None:
        rows = load_medmcqa_tsv(
            FIXTURE,
            benchmark_version="v0.1",
            source_version="test-fixture",
        )

        report = validate_items(rows)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(len(rows), 3)

    def test_normalizes_answer_text_index_and_source_answer(self) -> None:
        rows = load_medmcqa_tsv(FIXTURE, source_version="test-fixture")
        row = rows[1]

        self.assertEqual(row["source_answer"], "C")
        self.assertEqual(row["answer_index"], 2)
        self.assertEqual(row["answer"], "Levonorgestrel")
        self.assertEqual(
            row["choices"],
            ["OCP", "Danazol", "Levonorgestrel", "Mifepristone"],
        )

    def test_preserves_medmcqa_metadata(self) -> None:
        row = load_medmcqa_tsv(FIXTURE, source_version="test-fixture")[1]

        self.assertEqual(row["source_dataset"], "MedMCQA")
        self.assertEqual(row["source_id"], "0036cad0-d22f-453c-b075-322479d19d6e")
        self.assertEqual(row["license"], "Apache-2.0")
        self.assertEqual(row["contamination_risk"], "high")
        self.assertEqual(row["provenance"]["source_split"], "train")
        self.assertEqual(row["provenance"]["source_version"], "test-fixture")
        self.assertEqual(row["provenance"]["source_topic"], "Contraceptives")
        self.assertEqual(row["task_type"], "prevention")
        self.assertIn("contraceptives", row["tags"])

    def test_classifies_obgyn_and_neonatal_rows(self) -> None:
        rows = load_medmcqa_tsv(FIXTURE, source_version="test-fixture")

        self.assertEqual(rows[0]["clinical_domain"], "obgyn")
        self.assertEqual(rows[0]["age_group"], "adult")
        self.assertEqual(rows[2]["clinical_domain"], "neonatal")
        self.assertEqual(rows[2]["age_group"], "neonate")

    def test_limit_caps_loaded_rows(self) -> None:
        rows = load_medmcqa_tsv(FIXTURE, source_version="test-fixture", limit=2)

        self.assertEqual(len(rows), 2)

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
            normalize_medmcqa_row(row, row_number=1)


if __name__ == "__main__":
    unittest.main()
