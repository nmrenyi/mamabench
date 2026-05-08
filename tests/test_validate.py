from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from mamabench.io import read_jsonl
from mamabench.validate import ValidationReport, validate_items


ROOT = Path(__file__).resolve().parents[1]


def valid_mcq(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "mamabench_v0.1_mcq_test",
        "schema_version": "0.1",
        "set_type": "mcq",
        "source_dataset": "unit_test",
        "source_id": "unit-mcq-001",
        "question": "What is the safest next action?",
        "clinical_domain": "obgyn",
        "age_group": "maternal",
        "task_type": "triage",
        "safety_type": None,
        "choices": ["Refer urgently.", "Wait.", "Ignore.", "Reassure only."],
        "answer": "Refer urgently.",
        "answer_index": 0,
        "rubric": None,
        "tags": ["pregnancy"],
        "icd10_codes": ["O14"],
        "perturbation_of": None,
        "perturbation_type": None,
        "contamination_risk": "low",
        "license": "synthetic",
        "provenance": {
            "source_url": None,
            "source_split": "pilot",
            "source_version": "unit-v0.1",
        },
        "split": "pilot",
    }
    row.update(overrides)
    return row


def valid_open_ended(**overrides: Any) -> dict[str, Any]:
    row = valid_mcq(
        id="mamabench_v0.1_open_test",
        set_type="open_ended",
        source_id="unit-open-001",
        task_type="counseling",
        choices=None,
        answer=None,
        answer_index=None,
        rubric={"criteria": [{"name": "action", "points": 1}]},
    )
    row.update(overrides)
    return row


def valid_safety(**overrides: Any) -> dict[str, Any]:
    row = valid_mcq(
        id="mamabench_v0.1_safety_test",
        set_type="safety",
        source_id="unit-safety-001",
        clinical_domain="general_maternal",
        task_type="safety",
        safety_type="equity",
    )
    row.update(overrides)
    return row


class ValidateItemsTests(unittest.TestCase):
    def test_sample_file_is_valid(self) -> None:
        rows = read_jsonl(ROOT / "data" / "samples" / "sample.jsonl")
        report = validate_items(rows)
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.item_count, 4)

    def test_duplicate_ids_are_reported(self) -> None:
        first = valid_mcq(id="duplicate", source_id="source-1")
        second = valid_mcq(id="duplicate", source_id="source-2")

        report = validate_items([first, second])

        self.assert_issue(report, "id", "duplicate id")

    def test_duplicate_source_ids_are_reported(self) -> None:
        first = valid_mcq(id="item-1", source_id="same-source")
        second = valid_mcq(id="item-2", source_id="same-source")

        report = validate_items([first, second])

        self.assert_issue(report, "source_id", "duplicate source_dataset + source_id")

    def test_bad_mcq_answer_index_is_reported(self) -> None:
        report = validate_items([valid_mcq(answer_index=9)])

        self.assert_issue(report, "answer_index", "out of bounds")

    def test_mcq_answer_key_is_allowed_when_answer_index_is_set(self) -> None:
        report = validate_items([valid_mcq(answer="A", answer_index=0)])

        self.assertTrue(report.ok, report.to_dict())

    def test_mcq_answer_must_match_choice_when_answer_index_is_missing(self) -> None:
        report = validate_items([valid_mcq(answer="A", answer_index=None)])

        self.assert_issue(report, "answer", "must appear in choices")

    def test_open_ended_requires_rubric(self) -> None:
        report = validate_items([valid_open_ended(rubric=None)])

        self.assert_issue(report, "rubric", "require a rubric")

    def test_safety_requires_safety_type(self) -> None:
        report = validate_items([valid_safety(safety_type=None)])

        self.assert_issue(report, "safety_type", "require safety_type")

    def test_perturbation_requires_link_fields(self) -> None:
        report = validate_items([valid_safety(perturbation_type="paraphrase")])

        self.assert_issue(report, "perturbation_of", "required")

    def test_perturbation_reference_can_point_outside_current_file(self) -> None:
        report = validate_items(
            [
                valid_safety(
                    perturbation_of="missing-id",
                    perturbation_type="paraphrase",
                )
            ]
        )

        self.assertTrue(report.ok, report.to_dict())

    def test_strict_perturbation_reference_check_reports_missing_id(self) -> None:
        report = validate_items(
            [
                valid_safety(
                    perturbation_of="missing-id",
                    perturbation_type="paraphrase",
                )
            ],
            check_perturbation_references=True,
        )

        self.assert_issue(report, "perturbation_of", "does not reference")

    def test_unknown_controlled_value_is_reported(self) -> None:
        report = validate_items([valid_mcq(clinical_domain="cardiology")])

        self.assert_issue(report, "clinical_domain", "unknown value")

    def assert_issue(
        self, report: ValidationReport, field: str, message_part: str
    ) -> None:
        for issue in report.issues:
            if issue.field == field and message_part in issue.message:
                return
        self.fail(f"missing issue field={field!r} containing {message_part!r}: {report}")


if __name__ == "__main__":
    unittest.main()

