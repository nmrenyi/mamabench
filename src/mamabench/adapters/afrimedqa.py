"""Adapter for the filtered AfriMed-QA OBGYN MCQ TSV.

Only single-answer rows are normalized. Rows whose `correct_letter` is
comma-separated (e.g. `A,C,D`) cannot be represented by the v0.3 schema's
single `answer_index` and are skipped; the count is preserved in the manifest's
source-dataset `filter` block.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mamabench.adapters import _options, _provenance
from mamabench.config import normalize_benchmark_version
from mamabench.schema import SCHEMA_VERSION


AFRIMEDQA_SOURCE_DATASET = "AfriMed-QA"
AFRIMEDQA_SOURCE_URL = "https://huggingface.co/datasets/intronhealth/afrimedqa_v2"
AFRIMEDQA_LICENSE = "CC BY-NC-SA 4.0"
AFRIMEDQA_LICENSE_NOTES = (
    "Non-commercial use only. Derivative works must be released under the "
    "same license (share-alike)."
)
PREPARED_INPUT_REPOSITORY = "https://github.com/nmrenyi/obgyn-qa-collection"
PREPARED_INPUT_PATH = "afrimedqa/data/obgyn_mcq.tsv"
PREPARED_INPUT_BASE_METADATA: dict[str, Any] = {
    "description": (
        "Pre-filtered OBGYN subset of AfriMed-QA v2 used as mamabench "
        "adapter input."
    ),
    "filtering_done_outside_mamabench": True,
}

REQUIRED_COLUMNS = frozenset(
    {
        "question_clean",
        "options_formatted",
        "correct_letter",
    }
)

CONTENT_HASH_LENGTH = 12


class AfriMedQAAdapterError(ValueError):
    """Raised when an AfriMed-QA row cannot be normalized."""


@dataclass(frozen=True)
class AfriMedQAFilterStats:
    """Outcome of the multi-answer filter applied during loading."""

    total_source_rows: int
    single_answer_rows: int
    multi_answer_rows_skipped: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_source_rows": self.total_source_rows,
            "single_answer_rows": self.single_answer_rows,
            "multi_answer_rows_skipped": self.multi_answer_rows_skipped,
        }


def build_afrimedqa_source_metadata(
    input_tsv: str | Path | None = None,
    *,
    filter_stats: AfriMedQAFilterStats | None = None,
) -> dict[str, Any]:
    """Build dataset-level AfriMed-QA provenance for the manifest."""

    block: dict[str, Any] = {
        "url": AFRIMEDQA_SOURCE_URL,
        "license": AFRIMEDQA_LICENSE,
        "license_notes": AFRIMEDQA_LICENSE_NOTES,
        "prepared_input": _provenance.build_prepared_input_metadata(
            input_tsv,
            expected_repository=PREPARED_INPUT_REPOSITORY,
            expected_path=PREPARED_INPUT_PATH,
            base_metadata=PREPARED_INPUT_BASE_METADATA,
        ),
    }
    if filter_stats is not None:
        block["filter"] = filter_stats.to_dict()
    return {AFRIMEDQA_SOURCE_DATASET: block}


def load_afrimedqa_tsv(
    path: str | Path,
    *,
    benchmark_version: str,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], AfriMedQAFilterStats]:
    """Load an AfriMed-QA TSV and normalize single-answer MCQ rows."""

    if limit is not None and limit < 0:
        raise AfriMedQAAdapterError("limit must be non-negative")

    tsv_path = Path(path)
    rows: list[dict[str, Any]] = []
    total_scanned = 0
    multi_answer_skipped = 0

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise AfriMedQAAdapterError(f"{tsv_path}: missing TSV header")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise AfriMedQAAdapterError(
                f"{tsv_path}: missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=1):
            if limit is not None and len(rows) >= limit:
                break
            total_scanned += 1

            if _is_multi_answer(row.get("correct_letter", "")):
                multi_answer_skipped += 1
                continue

            rows.append(
                normalize_afrimedqa_row(
                    row,
                    row_number=row_number,
                    benchmark_version=benchmark_version,
                )
            )

    stats = AfriMedQAFilterStats(
        total_source_rows=total_scanned,
        single_answer_rows=len(rows),
        multi_answer_rows_skipped=multi_answer_skipped,
    )
    return rows, stats


def normalize_afrimedqa_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    benchmark_version: str,
) -> dict[str, Any]:
    """Normalize one AfriMed-QA TSV row into the mamabench schema."""

    correct_letter_raw = (row.get("correct_letter") or "").strip()
    if _is_multi_answer(correct_letter_raw):
        raise AfriMedQAAdapterError(
            f"row {row_number}: refusing to normalize multi-answer row "
            f"({correct_letter_raw!r}); filter at load time instead"
        )

    question = _required_text(row, "question_clean", row_number)
    correct_letter = _required_text(row, "correct_letter", row_number).upper()
    choices_by_letter = parse_options(row.get("options_formatted", ""), row_number)

    if correct_letter not in choices_by_letter:
        raise AfriMedQAAdapterError(
            f"row {row_number}: correct_letter {correct_letter!r} "
            "does not match any parsed option"
        )

    letters = list(choices_by_letter.keys())
    choices = list(choices_by_letter.values())
    answer_index = letters.index(correct_letter)
    answer = choices[answer_index]

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
            "dataset": AFRIMEDQA_SOURCE_DATASET,
            "id": content_hash,
            "answer": correct_letter,
        },
    }


def parse_options(options_formatted: str, row_number: int) -> dict[str, str]:
    """Parse `A. ... | B. ...` option text into an ordered letter map."""

    return _options.parse_options(
        options_formatted,
        row_number=row_number,
        error_cls=AfriMedQAAdapterError,
    )


def _required_text(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = _options.clean_text(row.get(field, ""))
    if not value:
        raise AfriMedQAAdapterError(f"row {row_number}: missing {field}")
    return value


def _is_multi_answer(correct_letter_raw: str) -> bool:
    return "," in (correct_letter_raw or "").strip()


def _benchmark_id(benchmark_version: str, content_hash: str) -> str:
    version = normalize_benchmark_version(benchmark_version)
    return f"mamabench_{version}_afrimedqa_{content_hash}"


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
