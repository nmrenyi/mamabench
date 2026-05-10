"""Canonical schema definitions for mamabench JSONL rows."""

from __future__ import annotations

from typing import Any, Final, TypedDict


SCHEMA_VERSION: Final[str] = "0.1"


class Provenance(TypedDict, total=False):
    """Common source metadata carried through normalized benchmark rows.

    Adapters may carry additional source-specific provenance keys at runtime.
    """

    source_url: str | None
    source_split: str | None
    source_version: str | None


class BenchmarkItem(TypedDict, total=False):
    """A normalized mamabench benchmark item."""

    id: str
    schema_version: str
    set_type: str
    source_dataset: str
    source_id: str | None
    question: str
    clinical_domain: str
    age_group: str
    task_type: str
    safety_type: str | None
    choices: list[str] | None
    answer: str | None
    answer_index: int | None
    source_answer: str | int | None
    rubric: Any
    tags: list[str]
    icd10_codes: list[str]
    perturbation_of: str | None
    perturbation_type: str | None
    contamination_risk: str
    license: str
    provenance: Provenance
    split: str


CANONICAL_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "schema_version",
    "set_type",
    "source_dataset",
    "source_id",
    "question",
    "clinical_domain",
    "age_group",
    "task_type",
    "safety_type",
    "choices",
    "answer",
    "answer_index",
    "source_answer",
    "rubric",
    "tags",
    "icd10_codes",
    "perturbation_of",
    "perturbation_type",
    "contamination_risk",
    "license",
    "provenance",
    "split",
)

PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "source_url",
    "source_split",
    "source_version",
)

CONTROLLED_VOCABULARIES: Final[dict[str, frozenset[str]]] = {
    "set_type": frozenset({"mcq", "open_ended", "safety"}),
    "clinical_domain": frozenset(
        {
            "obgyn",
            "neonatal",
            "infant",
            "pediatric",
            "reproductive",
            "general_maternal",
            "unknown",
        }
    ),
    "age_group": frozenset(
        {"maternal", "neonate", "infant", "child", "adult", "unknown"}
    ),
    "task_type": frozenset(
        {
            "diagnosis",
            "treatment",
            "triage",
            "dosage",
            "procedure",
            "prevention",
            "counseling",
            "case_reasoning",
            "factual_lookup",
            "safety",
            "unknown",
        }
    ),
    "contamination_risk": frozenset({"high", "medium", "low", "unknown"}),
    "split": frozenset({"dev", "test", "pilot"}),
}
