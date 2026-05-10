"""Adapter for the filtered MedMCQA OBGYN/Pediatrics TSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

from mamabench.schema import SCHEMA_VERSION


MEDMCQA_SOURCE_DATASET = "MedMCQA"
MEDMCQA_SOURCE_URL = "https://huggingface.co/datasets/openlifescienceai/medmcqa"
MEDMCQA_LICENSE = "Apache-2.0"

REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "question",
        "options_formatted",
        "correct_letter",
    }
)

OPTION_MARKER_PATTERN = re.compile(r"(?:^|\s\|\s)([A-Z])\.\s*")


class MedMCQAAdapterError(ValueError):
    """Raised when a MedMCQA row cannot be normalized."""


def load_medmcqa_tsv(
    path: str | Path,
    *,
    benchmark_version: str = "v0.1",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a filtered MedMCQA TSV and normalize it to mamabench rows."""

    tsv_path = Path(path)
    rows: list[dict[str, Any]] = []

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise MedMCQAAdapterError(f"{tsv_path}: missing TSV header")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise MedMCQAAdapterError(
                f"{tsv_path}: missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=1):
            rows.append(
                normalize_medmcqa_row(
                    row,
                    row_number=row_number,
                    benchmark_version=benchmark_version,
                )
            )
            if limit is not None and len(rows) >= limit:
                break

    return rows


def normalize_medmcqa_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    benchmark_version: str = "v0.1",
) -> dict[str, Any]:
    """Normalize one MedMCQA TSV row into the mamabench schema."""

    source_id = _required_text(row, "id", row_number)
    question = _required_text(row, "question", row_number)
    correct_letter = _required_text(row, "correct_letter", row_number).upper()
    choices_by_letter = parse_options(row.get("options_formatted", ""), row_number)

    if correct_letter not in choices_by_letter:
        raise MedMCQAAdapterError(
            f"row {row_number} ({source_id}): correct_letter {correct_letter!r} "
            "does not match any parsed option"
        )

    letters = list(choices_by_letter.keys())
    choices = list(choices_by_letter.values())
    answer_index = letters.index(correct_letter)
    answer = choices[answer_index]
    return {
        "id": _benchmark_id(benchmark_version, source_id),
        "schema_version": SCHEMA_VERSION,
        "set_type": "mcq",
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "source": {
            "dataset": MEDMCQA_SOURCE_DATASET,
            "id": source_id,
            "url": MEDMCQA_SOURCE_URL,
            "license": MEDMCQA_LICENSE,
            "answer": correct_letter,
        },
    }


def parse_options(options_formatted: str, row_number: int) -> dict[str, str]:
    """Parse `A. ... | B. ...` option text into an ordered letter map."""

    matches = list(OPTION_MARKER_PATTERN.finditer(options_formatted))
    if not matches:
        raise MedMCQAAdapterError(f"row {row_number}: options_formatted is empty")

    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1)
        next_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(options_formatted)
        )
        text = _clean_option_text(options_formatted[match.end() : next_start])
        if letter in parsed:
            raise MedMCQAAdapterError(
                f"row {row_number}: duplicate option letter {letter!r}"
            )
        if not text:
            raise MedMCQAAdapterError(
                f"row {row_number}: empty option text for {letter!r}"
            )
        parsed[letter] = text

    return parsed


def _required_text(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = _clean_text(row.get(field, ""))
    if not value:
        raise MedMCQAAdapterError(f"row {row_number}: missing {field}")
    return value


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\\n", " ").split())


def _clean_option_text(value: str) -> str:
    return _clean_text(value).strip(" |")


def _benchmark_id(benchmark_version: str, source_id: str) -> str:
    version = (
        benchmark_version
        if benchmark_version.startswith("v")
        else f"v{benchmark_version}"
    )
    return f"mamabench_{version}_medmcqa_{source_id}"
