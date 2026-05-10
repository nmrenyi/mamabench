from __future__ import annotations

import unittest
from typing import Any

from mamabench.validate import ValidationReport, validate_items


def valid_mcq(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "mamabench_v0.1_unit_test",
        "schema_version": "0.2",
        "set_type": "mcq",
        "question": "What is the safest next action?",
        "choices": ["Refer urgently.", "Wait.", "Ignore.", "Reassure only."],
        "answer": "Refer urgently.",
        "answer_index": 0,
        "source": {
            "dataset": "unit_test",
            "id": "unit-mcq-001",
            "url": "https://example.test/unit",
            "license": "synthetic",
            "answer": "A",
        },
    }
    row.update(overrides)
    return row


def with_source(**overrides: Any) -> dict[str, Any]:
    row = valid_mcq()
    source = dict(row["source"])
    source.update(overrides)
    row["source"] = source
    return row


class ValidateItemsTests(unittest.TestCase):
    def test_valid_mcq_is_valid(self) -> None:
        report = validate_items([valid_mcq()])

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.item_count, 1)

    def test_duplicate_ids_are_reported(self) -> None:
        first = valid_mcq(
            id="duplicate",
            source=with_source(id="source-1")["source"],
        )
        second = valid_mcq(
            id="duplicate",
            source=with_source(id="source-2")["source"],
        )

        report = validate_items([first, second])

        self.assert_issue(report, "id", "duplicate id")

    def test_duplicate_source_ids_are_reported(self) -> None:
        first = valid_mcq(id="item-1")
        second = valid_mcq(id="item-2")

        report = validate_items([first, second])

        self.assert_issue(
            report, "source.id", "duplicate source.dataset + source.id"
        )

    def test_schema_version_must_match_current_version(self) -> None:
        report = validate_items([valid_mcq(schema_version="0.1")])

        self.assert_issue(report, "schema_version", "expected schema_version '0.2'")

    def test_rejects_unexpected_v0_1_fields(self) -> None:
        report = validate_items([valid_mcq(clinical_domain="obgyn")])

        self.assert_issue(report, "clinical_domain", "unexpected canonical field")

    def test_missing_canonical_field_is_reported(self) -> None:
        row = valid_mcq()
        del row["answer_index"]

        report = validate_items([row])

        self.assert_issue(report, "answer_index", "missing canonical field")

    def test_unknown_set_type_is_reported(self) -> None:
        report = validate_items([valid_mcq(set_type="open_ended")])

        self.assert_issue(report, "set_type", "unknown value")

    def test_bad_mcq_answer_index_is_reported(self) -> None:
        report = validate_items([valid_mcq(answer_index=9)])

        self.assert_issue(report, "answer_index", "out of bounds")

    def test_answer_index_must_be_integer(self) -> None:
        report = validate_items([valid_mcq(answer_index=None)])

        self.assert_issue(report, "answer_index", "must be an integer")

    def test_mcq_answer_must_match_choice_when_answer_index_is_set(self) -> None:
        report = validate_items([valid_mcq(answer="A", answer_index=0)])

        self.assert_issue(report, "answer", "does not match")

    def test_choices_must_be_non_empty_strings(self) -> None:
        report = validate_items([valid_mcq(choices=["Refer urgently.", ""])])

        self.assert_issue(report, "choices", "choice at index 1")

    def test_source_requires_minimal_audit_fields(self) -> None:
        row = valid_mcq()
        del row["source"]["license"]

        report = validate_items([row])

        self.assert_issue(report, "source", "missing source.license")

    def test_source_id_can_be_null(self) -> None:
        report = validate_items([with_source(id=None)])

        self.assertTrue(report.ok, report.to_dict())

    def test_source_id_rejects_empty_strings(self) -> None:
        report = validate_items([with_source(id="")])

        self.assert_issue(report, "source.id", "string or null")

    def test_source_answer_field_preserves_source_key(self) -> None:
        report = validate_items([with_source(answer="A")])

        self.assertTrue(report.ok, report.to_dict())

    def test_source_answer_field_allows_integer_keys(self) -> None:
        report = validate_items([with_source(answer=1)])

        self.assertTrue(report.ok, report.to_dict())

    def test_source_answer_field_rejects_empty_strings(self) -> None:
        report = validate_items([with_source(answer="")])

        self.assert_issue(report, "source.answer", "non-empty string")

    def test_source_answer_field_rejects_bools(self) -> None:
        report = validate_items([with_source(answer=True)])

        self.assert_issue(report, "source.answer", "integer")

    def test_rejects_unexpected_source_fields(self) -> None:
        report = validate_items([with_source(subject="Gynaecology & Obstetrics")])

        self.assert_issue(report, "source.subject", "unexpected source field")

    def assert_issue(
        self, report: ValidationReport, field: str, message_part: str
    ) -> None:
        for issue in report.issues:
            if issue.field == field and message_part in issue.message:
                return
        self.fail(f"missing issue field={field!r} containing {message_part!r}: {report}")


if __name__ == "__main__":
    unittest.main()
