"""Validation utilities for normalized mamabench JSONL rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from mamabench.schema import (
    CANONICAL_FIELDS,
    CONTROLLED_VOCABULARIES,
    PROVENANCE_FIELDS,
    SCHEMA_VERSION,
)


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation problem."""

    message: str
    field: str | None = None
    item_id: str | None = None
    line_number: int | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "line_number": self.line_number,
            "item_id": self.item_id,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for a collection of benchmark items."""

    item_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def error_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "item_count": self.item_count,
            "error_count": self.error_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_items(
    items: Iterable[Mapping[str, Any]],
    *,
    known_ids: Iterable[str] | None = None,
    check_perturbation_references: bool = False,
) -> ValidationReport:
    """Validate a collection of normalized benchmark items."""

    rows = list(items)
    issues: list[ValidationIssue] = []
    first_id_line: dict[str, int] = {}
    first_source_line: dict[tuple[str, str], int] = {}

    for line_number, item in enumerate(rows, start=1):
        item_id = _item_id(item)
        issues.extend(_validate_item(item, line_number))

        if _is_nonblank_string(item.get("id")):
            row_id = item["id"]
            if row_id in first_id_line:
                issues.append(
                    _issue(
                        item,
                        line_number,
                        "id",
                        f"duplicate id also appears on line {first_id_line[row_id]}",
                    )
                )
            else:
                first_id_line[row_id] = line_number

        source_dataset = item.get("source_dataset")
        source_id = item.get("source_id")
        if _is_nonblank_string(source_dataset) and _is_nonblank_string(source_id):
            source_key = (source_dataset, source_id)
            if source_key in first_source_line:
                issues.append(
                    _issue(
                        item,
                        line_number,
                        "source_id",
                        "duplicate source_dataset + source_id also appears on "
                        f"line {first_source_line[source_key]}",
                    )
                )
            else:
                first_source_line[source_key] = line_number

        if item_id is None:
            continue

    local_ids = {
        item["id"]
        for item in rows
        if isinstance(item, Mapping) and _is_nonblank_string(item.get("id"))
    }
    valid_perturbation_targets = local_ids | {
        item_id
        for item_id in known_ids or ()
        if _is_nonblank_string(item_id)
    }
    for line_number, item in enumerate(rows, start=1):
        perturbation_of = item.get("perturbation_of")
        if _is_nonblank_string(perturbation_of):
            if perturbation_of == item.get("id"):
                issues.append(
                    _issue(
                        item,
                        line_number,
                        "perturbation_of",
                        "perturbation_of cannot reference the same item id",
                    )
                )
            elif (
                check_perturbation_references
                and perturbation_of not in valid_perturbation_targets
            ):
                issues.append(
                    _issue(
                        item,
                        line_number,
                        "perturbation_of",
                        "perturbation_of does not reference a known item id",
                    )
                )

    return ValidationReport(item_count=len(rows), issues=tuple(issues))


def _validate_item(item: Mapping[str, Any], line_number: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for field in CANONICAL_FIELDS:
        if field not in item:
            issues.append(_issue(item, line_number, field, "missing canonical field"))

    for field, allowed_values in CONTROLLED_VOCABULARIES.items():
        value = item.get(field)
        if value is None:
            continue
        if value not in allowed_values:
            issues.append(
                _issue(
                    item,
                    line_number,
                    field,
                    f"unknown value {value!r}; expected one of {sorted(allowed_values)}",
                )
            )

    if item.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            _issue(
                item,
                line_number,
                "schema_version",
                f"expected schema_version {SCHEMA_VERSION!r}",
            )
        )

    for field in (
        "id",
        "source_dataset",
        "question",
        "clinical_domain",
        "age_group",
        "task_type",
        "contamination_risk",
        "license",
        "split",
    ):
        if field in item and not _is_nonblank_string(item.get(field)):
            issues.append(_issue(item, line_number, field, "must be a non-empty string"))

    _validate_string_list(item, line_number, "tags", issues)
    _validate_string_list(item, line_number, "icd10_codes", issues)
    _validate_provenance(item, line_number, issues)
    _validate_choices_and_answer(item, line_number, issues)

    set_type = item.get("set_type")
    if set_type == "mcq":
        _validate_mcq(item, line_number, issues)
    elif set_type == "open_ended":
        _validate_open_ended(item, line_number, issues)
    elif set_type == "safety":
        _validate_safety(item, line_number, issues)

    if _is_nonblank_string(item.get("perturbation_of")) and not _is_nonblank_string(
        item.get("perturbation_type")
    ):
        issues.append(
            _issue(
                item,
                line_number,
                "perturbation_type",
                "required when perturbation_of is set",
            )
        )

    if _is_nonblank_string(item.get("perturbation_type")) and not _is_nonblank_string(
        item.get("perturbation_of")
    ):
        issues.append(
            _issue(
                item,
                line_number,
                "perturbation_of",
                "required when perturbation_type is set",
            )
        )

    return issues


def _validate_mcq(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    choices = item.get("choices")
    answer = item.get("answer")

    if not isinstance(choices, list) or len(choices) < 2:
        issues.append(
            _issue(item, line_number, "choices", "mcq rows require at least 2 choices")
        )

    if not _is_nonblank_string(answer):
        issues.append(_issue(item, line_number, "answer", "mcq rows require an answer"))

    if isinstance(choices, list) and _is_nonblank_string(answer):
        answer_index = item.get("answer_index")
        if answer_index is None and answer not in choices:
            issues.append(
                _issue(
                    item,
                    line_number,
                    "answer",
                    "answer must appear in choices when answer_index is not set",
                )
            )


def _validate_open_ended(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    choices = item.get("choices")
    if choices not in (None, []):
        issues.append(
            _issue(
                item,
                line_number,
                "choices",
                "open_ended rows must not include choices",
            )
        )

    if _is_empty(item.get("rubric")):
        issues.append(
            _issue(item, line_number, "rubric", "open_ended rows require a rubric")
        )


def _validate_safety(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    if not _is_nonblank_string(item.get("safety_type")):
        issues.append(
            _issue(item, line_number, "safety_type", "safety rows require safety_type")
        )


def _validate_choices_and_answer(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    choices = item.get("choices")
    answer = item.get("answer")
    answer_index = item.get("answer_index")

    if choices is not None:
        if not isinstance(choices, list):
            issues.append(_issue(item, line_number, "choices", "must be a list or null"))
            return
        for index, choice in enumerate(choices):
            if not _is_nonblank_string(choice):
                issues.append(
                    _issue(
                        item,
                        line_number,
                        "choices",
                        f"choice at index {index} must be a non-empty string",
                    )
                )

    if answer is not None and not _is_nonblank_string(answer):
        issues.append(_issue(item, line_number, "answer", "must be a string or null"))

    if answer_index is not None:
        if not _is_int(answer_index):
            issues.append(
                _issue(item, line_number, "answer_index", "must be an integer or null")
            )
            return

        if not isinstance(choices, list):
            issues.append(
                _issue(
                    item,
                    line_number,
                    "answer_index",
                    "cannot be set when choices is not a list",
                )
            )
            return

        if answer_index < 0 or answer_index >= len(choices):
            issues.append(
                _issue(item, line_number, "answer_index", "answer_index out of bounds")
            )
            return

        if _is_nonblank_string(answer) and answer != choices[answer_index]:
            issues.append(
                _issue(
                    item,
                    line_number,
                    "answer",
                    "answer does not match choices[answer_index]",
                )
            )


def _validate_string_list(
    item: Mapping[str, Any],
    line_number: int,
    field: str,
    issues: list[ValidationIssue],
) -> None:
    value = item.get(field)
    if not isinstance(value, list):
        issues.append(_issue(item, line_number, field, "must be a list of strings"))
        return

    for index, entry in enumerate(value):
        if not isinstance(entry, str):
            issues.append(
                _issue(
                    item,
                    line_number,
                    field,
                    f"entry at index {index} must be a string",
                )
            )


def _validate_provenance(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping):
        issues.append(_issue(item, line_number, "provenance", "must be an object"))
        return

    for field in PROVENANCE_FIELDS:
        if field not in provenance:
            issues.append(
                _issue(
                    item,
                    line_number,
                    "provenance",
                    f"missing provenance.{field}",
                )
            )
            continue

        value = provenance[field]
        if value is not None and not isinstance(value, str):
            issues.append(
                _issue(
                    item,
                    line_number,
                    "provenance",
                    f"provenance.{field} must be a string or null",
                )
            )


def _issue(
    item: Mapping[str, Any],
    line_number: int,
    field: str | None,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(
        item_id=_item_id(item),
        line_number=line_number,
        field=field,
        message=message,
    )


def _item_id(item: Mapping[str, Any]) -> str | None:
    value = item.get("id")
    return value if isinstance(value, str) else None


def _is_nonblank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

