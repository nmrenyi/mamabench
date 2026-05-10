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
        )

        report = validate_items(rows)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(len(rows), 4)

    def test_normalizes_answer_text_index_and_source_answer_key(self) -> None:
        rows = load_medmcqa_tsv(FIXTURE)
        row = rows[1]

        self.assertEqual(row["source"]["answer"], "C")
        self.assertEqual(row["answer_index"], 2)
        self.assertEqual(row["answer"], "Levonorgestrel")
        self.assertEqual(
            row["choices"],
            ["OCP", "Danazol", "Levonorgestrel", "Mifepristone"],
        )

    def test_preserves_minimal_medmcqa_source(self) -> None:
        row = load_medmcqa_tsv(FIXTURE)[1]

        self.assertEqual(row["schema_version"], "0.2")
        self.assertEqual(
            row["id"],
            "mamabench_v0.1_medmcqa_0036cad0-d22f-453c-b075-322479d19d6e",
        )
        self.assertEqual(row["source"]["dataset"], "MedMCQA")
        self.assertEqual(row["source"]["id"], "0036cad0-d22f-453c-b075-322479d19d6e")
        self.assertEqual(row["source"]["license"], "Apache-2.0")
        self.assertEqual(
            row["source"]["url"],
            "https://huggingface.co/datasets/openlifescienceai/medmcqa",
        )

    def test_does_not_emit_v0_1_label_fields(self) -> None:
        row = load_medmcqa_tsv(FIXTURE)[1]

        self.assertNotIn("clinical_domain", row)
        self.assertNotIn("age_group", row)
        self.assertNotIn("task_type", row)
        self.assertNotIn("tags", row)
        self.assertNotIn("provenance", row)

    def test_limit_caps_loaded_rows(self) -> None:
        rows = load_medmcqa_tsv(FIXTURE, limit=2)

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
