"""Adapter for the Women's Health Benchmark (WHB) stumps TSV.

WHB is 20 expert-crafted "model stumps" — clinical prompts paired with
an expert justification describing how LLMs typically fail. All 20 rows
are in mamabench's OBGYN scope by upstream curation; no classifier
filter is applied.

Each row becomes a v0.4 `open_ended` mamabench row: `question_clean`
is the question, `expert_justification` is the reference response that
documents what a correct answer should address and what failure modes
to avoid.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


WHB_SOURCE_DATASET = "WHB"
WHB_SOURCE_URL = "https://huggingface.co/datasets/TheLumos/WHB_subset"
WHB_LICENSE = "CC-BY-SA-4.0"

SCHEMA_VERSION = "0.4"
CONTENT_HASH_LENGTH = 12
REQUIRED_COLUMNS = frozenset({"question_clean", "expert_justification"})


class WHBAdapterError(ValueError):
    """Raised when a WHB row cannot be normalized."""


@dataclass
class WHBStats:
    """No row-drop filter for WHB; kept == total."""

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
        raise WHBAdapterError(f"row {row_number}: missing {field}")
    return value


def normalize_whb_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    benchmark_version: str,
    key_facts_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one v0.4 `open_ended` mamabench row from a WHB TSV row.

    When ``key_facts_metadata`` is provided (loaded from the keyfact
    extractor side-file by row_id), it is nested under
    ``source.metadata.key_fact_extraction``.
    """
    question = _required(row, "question_clean", row_number)
    answer = _required(row, "expert_justification", row_number)
    content_hash = _content_hash(question, answer)
    source: dict[str, Any] = {
        "dataset": WHB_SOURCE_DATASET,
        "id": content_hash,
    }
    if key_facts_metadata is not None:
        source["metadata"] = {"key_fact_extraction": dict(key_facts_metadata)}
    return {
        "id": f"mamabench_{benchmark_version}_whb_{content_hash}",
        "schema_version": SCHEMA_VERSION,
        "set_type": "open_ended",
        "question": question,
        "answer": answer,
        "source": source,
    }


def load_whb(
    source_tsv: str | Path,
    *,
    benchmark_version: str,
    limit: int | None = None,
    keyfacts_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], WHBStats]:
    """Load WHB TSV and normalize to v0.4 open_ended rows.

    When ``keyfacts_path`` is provided, the keyfact extractor side-file is
    loaded and each row that matches by id gets ``source.metadata.key_fact_extraction``
    populated. Rows without a matching keyfact entry are emitted without
    the field (no error — supports partial extraction state).
    """
    if limit is not None and limit < 0:
        raise WHBAdapterError("limit must be non-negative")

    keyfacts_by_id: dict[str, dict[str, Any]] = {}
    if keyfacts_path is not None:
        # Local import to keep import graph cheap when keyfacts aren't used.
        from ._keyfact_extraction import load_keyfacts_by_row_id

        keyfacts_by_id = load_keyfacts_by_row_id(keyfacts_path)

    path = Path(source_tsv)
    rows: list[dict[str, Any]] = []
    stats = WHBStats()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise WHBAdapterError(f"{path}: missing TSV header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise WHBAdapterError(
                f"{path}: missing required columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=1):
            stats.total += 1
            # First pass: get the row's mamabench id so we can look up its key_facts.
            partial = normalize_whb_row(
                row, row_number=row_number, benchmark_version=benchmark_version
            )
            key_facts_metadata = keyfacts_by_id.get(partial["id"])
            if key_facts_metadata is not None:
                partial = normalize_whb_row(
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


def build_whb_source_metadata() -> dict[str, Any]:
    return {
        WHB_SOURCE_DATASET: {
            "url": WHB_SOURCE_URL,
            "license": WHB_LICENSE,
        }
    }
