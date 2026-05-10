"""Canonical schema definitions for mamabench JSONL rows."""

from __future__ import annotations

from typing import Final, TypedDict


SCHEMA_VERSION: Final[str] = "0.3"


class Source(TypedDict, total=False):
    """Minimal source metadata carried through normalized benchmark rows."""

    dataset: str
    id: str | None
    answer: str | int | None


class BenchmarkItem(TypedDict, total=False):
    """A normalized mamabench benchmark item."""

    id: str
    schema_version: str
    set_type: str
    question: str
    choices: list[str]
    answer: str
    answer_index: int
    source: Source


CANONICAL_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "schema_version",
    "set_type",
    "question",
    "choices",
    "answer",
    "answer_index",
    "source",
)

SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "dataset",
    "id",
    "answer",
)

REQUIRED_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "dataset",
    "id",
)

CONTROLLED_VOCABULARIES: Final[dict[str, frozenset[str]]] = {
    "set_type": frozenset({"mcq"}),
}
