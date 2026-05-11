"""Adapter for the filtered MedMCQA OBGYN/Pediatrics TSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from mamabench.adapters import _options, _provenance
from mamabench.config import normalize_benchmark_version
from mamabench.schema import SCHEMA_VERSION


MEDMCQA_SOURCE_DATASET = "MedMCQA"
MEDMCQA_SOURCE_URL = "https://huggingface.co/datasets/openlifescienceai/medmcqa"
MEDMCQA_LICENSE = "Apache-2.0"
PREPARED_INPUT_REPOSITORY = "https://github.com/nmrenyi/obgyn-qa-collection"
PREPARED_INPUT_PATH = "medmcqa/data/obgyn_mcq.tsv"
PREPARED_INPUT_BASE_METADATA: dict[str, Any] = {
    "description": (
        "Pre-filtered OBGYN/Pediatrics MedMCQA subset used as mamabench "
        "adapter input."
    ),
    "filtering_done_outside_mamabench": True,
}

REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "question",
        "options_formatted",
        "correct_letter",
    }
)


class MedMCQAAdapterError(ValueError):
    """Raised when a MedMCQA row cannot be normalized."""


def build_medmcqa_source_metadata(input_tsv: str | Path | None = None) -> dict[str, Any]:
    """Build dataset-level MedMCQA provenance for the manifest."""

    return {
        MEDMCQA_SOURCE_DATASET: {
            "url": MEDMCQA_SOURCE_URL,
            "license": MEDMCQA_LICENSE,
            "prepared_input": _provenance.build_prepared_input_metadata(
                input_tsv,
                expected_repository=PREPARED_INPUT_REPOSITORY,
                expected_path=PREPARED_INPUT_PATH,
                base_metadata=PREPARED_INPUT_BASE_METADATA,
            ),
        }
    }


def load_medmcqa_tsv(
    path: str | Path,
    *,
    benchmark_version: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a filtered MedMCQA TSV and normalize it to mamabench rows."""

    if limit is not None and limit < 0:
        raise MedMCQAAdapterError("limit must be non-negative")

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
            if limit is not None and len(rows) >= limit:
                break

            rows.append(
                normalize_medmcqa_row(
                    row,
                    row_number=row_number,
                    benchmark_version=benchmark_version,
                )
            )
    return rows


def normalize_medmcqa_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    benchmark_version: str,
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
            "answer": correct_letter,
        },
    }


def parse_options(options_formatted: str, row_number: int) -> dict[str, str]:
    """Parse `A. ... | B. ...` option text into an ordered letter map."""

    return _options.parse_options(
        options_formatted,
        row_number=row_number,
        error_cls=MedMCQAAdapterError,
    )


def _required_text(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = _options.clean_text(row.get(field, ""))
    if not value:
        raise MedMCQAAdapterError(f"row {row_number}: missing {field}")
    return value


def _benchmark_id(benchmark_version: str, source_id: str) -> str:
    version = normalize_benchmark_version(benchmark_version)
    return f"mamabench_{version}_medmcqa_{source_id}"
