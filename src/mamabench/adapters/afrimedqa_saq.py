"""Adapter for the AfriMed-QA short-answer-question (SAQ) TSV.

The AfriMed-QA SAQ subset is the 37 expert-tier short-answer questions
already filtered upstream by the `obgyn-qa-collection` repo. No OBGYN
classifier verdict filter is applied — the upstream column-based filter
(`specialty contains "Obstetric" AND tier == "expert" AND question_type
== "saq"`) is trusted, matching the v0.1 design decision for AfriMed-QA
MCQ rows.

Each row becomes a v0.4 `open_ended` mamabench row with the
`question_clean` field as `question` and `answer_rationale` as the
reference `answer`. There is no native row id, so we use a content-hash
of (question + answer) — same convention the v0.1 AfriMed-QA MCQ
adapter uses.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


AFRIMEDQA_SAQ_SOURCE_DATASET = "AfriMed-QA"
AFRIMEDQA_SAQ_SUBSET = "saq"
AFRIMEDQA_SAQ_SOURCE_URL = "https://huggingface.co/datasets/intronhealth/afrimedqa_v2"
AFRIMEDQA_SAQ_LICENSE = "CC-BY-NC-SA-4.0"

SCHEMA_VERSION = "0.4"
CONTENT_HASH_LENGTH = 12
REQUIRED_COLUMNS = frozenset({"question_clean", "answer_rationale"})


class AfriMedQASAQAdapterError(ValueError):
    """Raised when an AfriMed-QA SAQ row cannot be normalized."""


@dataclass
class AfriMedQASAQStats:
    """Filter accounting; SAQ has no row-drop filter so kept == total."""

    total: int = 0
    kept: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "kept": self.kept}


def _content_hash(question: str, answer: str) -> str:
    payload = "\n".join([question.strip(), answer.strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: CONTENT_HASH_LENGTH]


def _required(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise AfriMedQASAQAdapterError(f"row {row_number}: missing {field}")
    return value


def normalize_afrimedqa_saq_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    benchmark_version: str,
    key_facts_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one v0.4 `open_ended` mamabench row from an AfriMed-SAQ TSV row.

    When ``key_facts_metadata`` is provided, it is nested under
    ``source.metadata.key_fact_extraction`` alongside the existing subset tag.
    """
    question = _required(row, "question_clean", row_number)
    answer = _required(row, "answer_rationale", row_number)
    content_hash = _content_hash(question, answer)
    metadata: dict[str, Any] = {"subset": AFRIMEDQA_SAQ_SUBSET}
    if key_facts_metadata is not None:
        metadata["key_fact_extraction"] = dict(key_facts_metadata)
    return {
        "id": f"mamabench_{benchmark_version}_afrimedqa-saq_{content_hash}",
        "schema_version": SCHEMA_VERSION,
        "set_type": "open_ended",
        "question": question,
        "answer": answer,
        "source": {
            "dataset": AFRIMEDQA_SAQ_SOURCE_DATASET,
            "id": content_hash,
            "metadata": metadata,
        },
    }


def load_afrimedqa_saq(
    source_tsv: str | Path,
    *,
    benchmark_version: str,
    limit: int | None = None,
    keyfacts_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], AfriMedQASAQStats]:
    """Load AfriMed-SAQ TSV and normalize to v0.4 mamabench rows.

    When ``keyfacts_path`` is provided, the keyfact extractor side-file is
    loaded and each row matched by id gets
    ``source.metadata.key_fact_extraction`` populated.
    """
    if limit is not None and limit < 0:
        raise AfriMedQASAQAdapterError("limit must be non-negative")

    keyfacts_by_id: dict[str, dict[str, Any]] = {}
    if keyfacts_path is not None:
        from ._keyfact_extraction import load_keyfacts_by_row_id

        keyfacts_by_id = load_keyfacts_by_row_id(keyfacts_path)

    path = Path(source_tsv)
    rows: list[dict[str, Any]] = []
    stats = AfriMedQASAQStats()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise AfriMedQASAQAdapterError(f"{path}: missing TSV header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise AfriMedQASAQAdapterError(
                f"{path}: missing required columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=1):
            stats.total += 1
            partial = normalize_afrimedqa_saq_row(
                row, row_number=row_number, benchmark_version=benchmark_version
            )
            key_facts_metadata = keyfacts_by_id.get(partial["id"])
            if key_facts_metadata is not None:
                partial = normalize_afrimedqa_saq_row(
                    row,
                    row_number=row_number,
                    benchmark_version=benchmark_version,
                    key_facts_metadata=key_facts_metadata,
                )
            rows.append(partial)
            stats.kept += 1
            if limit is not None and len(rows) >= limit:
                break
    return rows, stats


def build_afrimedqa_saq_source_metadata() -> dict[str, Any]:
    """Build dataset-level AfriMed-QA SAQ provenance for the manifest."""
    return {
        AFRIMEDQA_SAQ_SOURCE_DATASET: {
            "url": AFRIMEDQA_SAQ_SOURCE_URL,
            "license": AFRIMEDQA_SAQ_LICENSE,
        }
    }
