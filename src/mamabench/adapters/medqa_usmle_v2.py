"""v0.2 MedQA-USMLE adapter: refilter from the original 14,369-row JSONL.

The v0.1 adapter (`mamabench.adapters.medqa_usmle`) reads from
`obgyn-qa-collection/medqa-usmle/data/obgyn_usmle.tsv` — 1,025 rows
pre-filtered by an earlier Gemini-based OBGYN classifier. v0.2 refilters
from the upstream source `US_qbank.jsonl` (all 14,369 USMLE questions)
using the unified Qwen3.6-27B-FP8 OBGYN classifier verdicts in
`benchmark/v0.2/classification_verdicts/medqa_usmle.jsonl`.

The row set for v0.2 is therefore different from v0.1; that change is
documented in the v0.2 dataset card release notes. The v0.1 adapter is
left untouched for retro-reproducibility of the v0.1 release.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MEDQA_USMLE_SOURCE_DATASET = "MedQA-USMLE"
MEDQA_USMLE_SOURCE_URL = "https://github.com/jind11/MedQA"
MEDQA_USMLE_LICENSE = "MIT"

SCHEMA_VERSION = "0.4"
CONTENT_HASH_LENGTH = 12
VALID_CATEGORIES = frozenset(
    {"MATERNAL", "NEONATAL", "CHILD_HEALTH", "SEXUAL_AND_REPRODUCTIVE_HEALTH", "NONE"}
)


class MedQAUSMLEv2AdapterError(ValueError):
    """Raised when a MedQA-USMLE source row cannot be normalized."""


@dataclass
class MedQAUSMLEv2Stats:
    """Filter accounting; invariant: total == kept + skipped_none + skipped_no_verdict."""

    total: int = 0
    kept: int = 0
    skipped_none: int = 0
    skipped_no_verdict: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "kept": self.kept,
            "skipped_none": self.skipped_none,
            "skipped_no_verdict": self.skipped_no_verdict,
        }


def synth_row_id(row_index: int) -> str:
    """Stable synthesized id for a US_qbank.jsonl row, matching the classifier convention."""
    return f"usmle_{row_index:05d}"


def _content_hash(question: str, choices: list[str], answer_text: str) -> str:
    """Content-stable id: matches the v0.1 adapter's hash convention exactly."""
    payload = "\n".join(
        [
            question.strip(),
            "\n".join(sorted(choice.strip() for choice in choices)),
            answer_text.strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: CONTENT_HASH_LENGTH]


def load_verdicts(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read the MedQA-USMLE verdicts JSONL into ``{row_id: verdict}``."""
    verdicts: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            row_id = v.get("row_id")
            category = v.get("category")
            if not isinstance(row_id, str) or category not in VALID_CATEGORIES:
                continue
            verdicts[row_id] = v
    return verdicts


def normalize_medqa_usmle_source_row(
    source_row: Mapping[str, Any],
    *,
    row_index: int,
    verdict: Mapping[str, Any],
    benchmark_version: str,
) -> dict[str, Any]:
    """Build one v0.4 mcq mamabench row from a US_qbank.jsonl row."""
    question_raw = source_row.get("question")
    options = source_row.get("options")
    answer_letter = source_row.get("answer")
    meta_info = source_row.get("meta_info")

    if not isinstance(question_raw, str) or not question_raw.strip():
        raise MedQAUSMLEv2AdapterError(f"row {row_index}: missing question")
    if not isinstance(options, Mapping) or len(options) < 2:
        raise MedQAUSMLEv2AdapterError(
            f"row {row_index}: options must be a dict of at least 2 entries"
        )
    if not isinstance(answer_letter, str) or answer_letter not in options:
        raise MedQAUSMLEv2AdapterError(
            f"row {row_index}: answer {answer_letter!r} not in options {sorted(options)}"
        )

    question = question_raw.strip()
    letters = sorted(options.keys())
    choices = [str(options[letter]).strip() for letter in letters]
    for letter, choice in zip(letters, choices):
        if not choice:
            raise MedQAUSMLEv2AdapterError(
                f"row {row_index}: empty option text at letter {letter!r}"
            )
    answer_index = letters.index(answer_letter)
    answer_text = choices[answer_index]
    content_hash = _content_hash(question, choices, answer_text)

    metadata: dict[str, Any] = {
        "upstream_index": synth_row_id(row_index),
        "obgyn_classification": {
            "model": verdict.get("model"),
            "prompt_version": verdict.get("prompt_version"),
            "category": verdict["category"],
            "rationale": verdict.get("rationale", ""),
        },
    }
    if isinstance(meta_info, str) and meta_info.strip():
        metadata["meta_info"] = meta_info.strip()

    return {
        "id": f"mamabench_{benchmark_version}_medqa_usmle_{content_hash}",
        "schema_version": SCHEMA_VERSION,
        "set_type": "mcq",
        "question": question,
        "choices": choices,
        "answer": answer_text,
        "answer_index": answer_index,
        "source": {
            "dataset": MEDQA_USMLE_SOURCE_DATASET,
            "id": content_hash,
            "answer": answer_letter,
            "metadata": metadata,
        },
    }


def load_medqa_usmle_v2(
    source_jsonl: str | Path,
    verdicts_path: str | Path,
    *,
    benchmark_version: str,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], MedQAUSMLEv2Stats]:
    """Walk US_qbank.jsonl, filter by verdict, normalize to v0.4 mcq rows."""
    if limit is not None and limit < 0:
        raise MedQAUSMLEv2AdapterError("limit must be non-negative")

    verdicts = load_verdicts(verdicts_path)
    stats = MedQAUSMLEv2Stats()
    rows: list[dict[str, Any]] = []

    with Path(source_jsonl).open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            stats.total += 1
            source_row = json.loads(line)
            verdict = verdicts.get(synth_row_id(row_index))
            if verdict is None:
                stats.skipped_no_verdict += 1
                continue
            if verdict["category"] == "NONE":
                stats.skipped_none += 1
                continue
            rows.append(
                normalize_medqa_usmle_source_row(
                    source_row,
                    row_index=row_index,
                    verdict=verdict,
                    benchmark_version=benchmark_version,
                )
            )
            stats.kept += 1
            if limit is not None and len(rows) >= limit:
                break

    if stats.total != stats.kept + stats.skipped_none + stats.skipped_no_verdict:
        raise MedQAUSMLEv2AdapterError(
            f"filter accounting inconsistent — total={stats.total}, "
            f"kept={stats.kept}, skipped_none={stats.skipped_none}, "
            f"skipped_no_verdict={stats.skipped_no_verdict}"
        )
    return rows, stats


def build_medqa_usmle_v2_source_metadata() -> dict[str, Any]:
    """Build dataset-level MedQA-USMLE provenance for the manifest."""
    return {
        MEDQA_USMLE_SOURCE_DATASET: {
            "url": MEDQA_USMLE_SOURCE_URL,
            "license": MEDQA_USMLE_LICENSE,
        }
    }
