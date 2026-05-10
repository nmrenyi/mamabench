"""Validation utilities for normalized mamabench JSONL rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from mamabench.schema import (
    CANONICAL_FIELDS,
    CONTROLLED_VOCABULARIES,
    REQUIRED_SOURCE_FIELDS,
    SCHEMA_VERSION,
    SOURCE_FIELDS,
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


def validate_items(items: Iterable[Mapping[str, Any]]) -> ValidationReport:
    """Validate a collection of normalized benchmark items."""

    rows = list(items)
    issues: list[ValidationIssue] = []
    first_id_line: dict[str, int] = {}
    first_source_line: dict[tuple[str, str], int] = {}

    for line_number, item in enumerate(rows, start=1):
        issues.extend(_validate_item(item, line_number))

        row_id = item.get("id")
        if _is_nonblank_string(row_id):
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

        source = item.get("source")
        if isinstance(source, Mapping):
            source_dataset = source.get("dataset")
            source_id = source.get("id")
            if _is_nonblank_string(source_dataset) and _is_nonblank_string(source_id):
                source_key = (source_dataset, source_id)
                if source_key in first_source_line:
                    issues.append(
                        _issue(
                            item,
                            line_number,
                            "source.id",
                            "duplicate source.dataset + source.id also appears on "
                            f"line {first_source_line[source_key]}",
                        )
                    )
                else:
                    first_source_line[source_key] = line_number

    return ValidationReport(item_count=len(rows), issues=tuple(issues))


def _validate_item(item: Mapping[str, Any], line_number: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for field in CANONICAL_FIELDS:
        if field not in item:
            issues.append(_issue(item, line_number, field, "missing canonical field"))

    for field in item:
        if field not in CANONICAL_FIELDS:
            issues.append(_issue(item, line_number, field, "unexpected canonical field"))

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

    for field in ("id", "schema_version", "set_type", "question"):
        if field in item and not _is_nonblank_string(item.get(field)):
            issues.append(_issue(item, line_number, field, "must be a non-empty string"))

    _validate_source(item, line_number, issues)
    _validate_mcq(item, line_number, issues)

    return issues


def _validate_source(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    source = item.get("source")
    if not isinstance(source, Mapping):
        issues.append(_issue(item, line_number, "source", "must be an object"))
        return

    for field in REQUIRED_SOURCE_FIELDS:
        if field not in source:
            issues.append(
                _issue(item, line_number, "source", f"missing source.{field}")
            )

    for field in source:
        if field not in SOURCE_FIELDS:
            issues.append(
                _issue(item, line_number, f"source.{field}", "unexpected source field")
            )

    for field in ("dataset",):
        if field in source and not _is_nonblank_string(source.get(field)):
            issues.append(
                _issue(
                    item,
                    line_number,
                    f"source.{field}",
                    "must be a non-empty string",
                )
            )

    if "id" in source:
        source_id = source.get("id")
        if source_id is not None and not _is_nonblank_string(source_id):
            issues.append(
                _issue(
                    item,
                    line_number,
                    "source.id",
                    "must be a non-empty string or null",
                )
            )

    if "answer" in source:
        _validate_source_answer(item, line_number, source.get("answer"), issues)


def _validate_mcq(
    item: Mapping[str, Any], line_number: int, issues: list[ValidationIssue]
) -> None:
    choices = item.get("choices")
    answer = item.get("answer")
    answer_index = item.get("answer_index")

    if not isinstance(choices, list):
        issues.append(_issue(item, line_number, "choices", "must be a list"))
        return

    if len(choices) < 2:
        issues.append(
            _issue(item, line_number, "choices", "mcq rows require at least 2 choices")
        )

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

    if not _is_nonblank_string(answer):
        issues.append(
            _issue(item, line_number, "answer", "must be a non-empty string")
        )

    if not _is_int(answer_index):
        issues.append(
            _issue(item, line_number, "answer_index", "must be an integer")
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


def _validate_source_answer(
    item: Mapping[str, Any],
    line_number: int,
    value: Any,
    issues: list[ValidationIssue],
) -> None:
    if value is None:
        return
    if _is_nonblank_string(value):
        return
    if _is_int(value):
        return

    issues.append(
        _issue(
            item,
            line_number,
            "source.answer",
            "must be a non-empty string, integer, or null",
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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
