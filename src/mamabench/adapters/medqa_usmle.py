"""Adapter for the filtered MedQA-USMLE OBGYN TSV."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping

from mamabench.adapters import _options, _provenance
from mamabench.config import normalize_benchmark_version
from mamabench.schema import SCHEMA_VERSION


MEDQA_USMLE_SOURCE_DATASET = "MedQA-USMLE"
MEDQA_USMLE_SOURCE_URL = "https://github.com/jind11/MedQA"
MEDQA_USMLE_LICENSE = "MIT"
PREPARED_INPUT_REPOSITORY = "https://github.com/nmrenyi/obgyn-qa-collection"
PREPARED_INPUT_PATH = "medqa-usmle/data/obgyn_usmle.tsv"
PREPARED_INPUT_BASE_METADATA: dict[str, Any] = {
    "description": (
        "Pre-filtered OBGYN subset of the MedQA US question bank used as "
        "mamabench adapter input."
    ),
    "filtering_done_outside_mamabench": True,
}

REQUIRED_COLUMNS = frozenset(
    {
        "question",
        "options_formatted",
        "correct_letter",
        "answer",
    }
)

CONTENT_HASH_LENGTH = 12


class MedQAUSMLEAdapterError(ValueError):
    """Raised when a MedQA-USMLE row cannot be normalized."""


def build_medqa_usmle_source_metadata(
    input_tsv: str | Path | None = None,
) -> dict[str, Any]:
    """Build dataset-level MedQA-USMLE provenance for the manifest."""

    return {
        MEDQA_USMLE_SOURCE_DATASET: {
            "url": MEDQA_USMLE_SOURCE_URL,
            "license": MEDQA_USMLE_LICENSE,
            "prepared_input": _provenance.build_prepared_input_metadata(
                input_tsv,
                expected_repository=PREPARED_INPUT_REPOSITORY,
                expected_path=PREPARED_INPUT_PATH,
                base_metadata=PREPARED_INPUT_BASE_METADATA,
            ),
        }
    }


def load_medqa_usmle_tsv(
    path: str | Path,
    *,
    benchmark_version: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a filtered MedQA-USMLE TSV and normalize it to mamabench rows."""

    if limit is not None and limit < 0:
        raise MedQAUSMLEAdapterError("limit must be non-negative")

    tsv_path = Path(path)
    rows: list[dict[str, Any]] = []

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise MedQAUSMLEAdapterError(f"{tsv_path}: missing TSV header")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise MedQAUSMLEAdapterError(
                f"{tsv_path}: missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=1):
            if limit is not None and len(rows) >= limit:
                break

            rows.append(
                normalize_medqa_usmle_row(
                    row,
                    row_number=row_number,
                    benchmark_version=benchmark_version,
                )
            )
    return rows


def normalize_medqa_usmle_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    benchmark_version: str,
) -> dict[str, Any]:
    """Normalize one MedQA-USMLE TSV row into the mamabench schema."""

    question = _required_text(row, "question", row_number)
    correct_letter = _required_text(row, "correct_letter", row_number).upper()
    source_answer_text = _required_text(row, "answer", row_number)
    choices_by_letter = parse_options(row.get("options_formatted", ""), row_number)

    if correct_letter not in choices_by_letter:
        raise MedQAUSMLEAdapterError(
            f"row {row_number}: correct_letter {correct_letter!r} "
            "does not match any parsed option"
        )

    letters = list(choices_by_letter.keys())
    choices = list(choices_by_letter.values())
    answer_index = letters.index(correct_letter)
    answer = choices[answer_index]

    if _options.clean_text(source_answer_text) != answer:
        raise MedQAUSMLEAdapterError(
            f"row {row_number}: source answer text {source_answer_text!r} "
            f"does not match parsed option {answer!r} at letter {correct_letter}"
        )

    content_hash = _content_hash(question, choices, answer)
    return {
        "id": _benchmark_id(benchmark_version, content_hash),
        "schema_version": SCHEMA_VERSION,
        "set_type": "mcq",
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "source": {
            "dataset": MEDQA_USMLE_SOURCE_DATASET,
            "id": content_hash,
            "answer": correct_letter,
        },
    }


def parse_options(options_formatted: str, row_number: int) -> dict[str, str]:
    """Parse `A. ... | B. ...` option text into an ordered letter map."""

    return _options.parse_options(
        options_formatted,
        row_number=row_number,
        error_cls=MedQAUSMLEAdapterError,
    )


def _required_text(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = _options.clean_text(row.get(field, ""))
    if not value:
        raise MedQAUSMLEAdapterError(f"row {row_number}: missing {field}")
    return value


def _benchmark_id(benchmark_version: str, content_hash: str) -> str:
    version = normalize_benchmark_version(benchmark_version)
    return f"mamabench_{version}_medqa_usmle_{content_hash}"


def _content_hash(question: str, choices: list[str], answer: str) -> str:
    # Sorting choices keeps the hash stable under option permutation, since the
    # MCQ task is unchanged by reordering the option list. Including the answer
    # text separates rows that share a question stem but key on different options.
    payload = "\n".join(
        [
            question.strip(),
            "\n".join(sorted(choice.strip() for choice in choices)),
            answer.strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONTENT_HASH_LENGTH]
