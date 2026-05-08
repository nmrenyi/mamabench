"""Adapter for the filtered MedMCQA OBGYN/Pediatrics TSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

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
        "explanation",
        "subject",
        "topic",
        "choice_type",
        "split",
    }
)

OPTION_MARKER_PATTERN = re.compile(r"(?:^|\s\|\s)([A-Z])\.\s*")
MATERNAL_KEYWORDS = frozenset(
    {
        "pregnan",
        "gestation",
        "fetal",
        "fetus",
        "obstetric",
        "antenatal",
        "prenatal",
        "labor",
        "labour",
        "delivery",
        "postpartum",
        "preeclampsia",
        "eclampsia",
        "placenta",
        "ectopic",
    }
)
NEONATAL_KEYWORDS = frozenset(
    {"neonate", "neonatal", "newborn", "birth asphyxia", "premature"}
)
INFANT_KEYWORDS = frozenset({"infant", "baby", "breastfeeding", "lactation"})


class MedMCQAAdapterError(ValueError):
    """Raised when a MedMCQA row cannot be normalized."""


def load_medmcqa_tsv(
    path: str | Path,
    *,
    benchmark_version: str = "v0.1",
    benchmark_split: str = "test",
    source_version: str | None = None,
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
                    benchmark_split=benchmark_split,
                    source_version=source_version,
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
    benchmark_split: str = "test",
    source_version: str | None = None,
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
    subject = _clean_text(row.get("subject", ""))
    topic = _clean_text(row.get("topic", ""))
    choice_type = _clean_text(row.get("choice_type", ""))
    source_split = _clean_text(row.get("split", ""))
    explanation = _optional_text(row.get("explanation"))
    clinical_domain, age_group = _classify_domain_and_age(
        subject=subject,
        topic=topic,
        question=question,
    )

    return {
        "id": _benchmark_id(benchmark_version, source_id),
        "schema_version": SCHEMA_VERSION,
        "set_type": "mcq",
        "source_dataset": MEDMCQA_SOURCE_DATASET,
        "source_id": source_id,
        "question": question,
        "clinical_domain": clinical_domain,
        "age_group": age_group,
        "task_type": _classify_task_type(question, topic),
        "safety_type": None,
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "source_answer": correct_letter,
        "rubric": None,
        "tags": _tags(subject=subject, topic=topic, choice_type=choice_type),
        "icd10_codes": [],
        "perturbation_of": None,
        "perturbation_type": None,
        "contamination_risk": "high",
        "license": MEDMCQA_LICENSE,
        "provenance": {
            "source_url": MEDMCQA_SOURCE_URL,
            "source_split": source_split or None,
            "source_version": source_version,
            "source_subject": subject or None,
            "source_topic": topic or None,
            "source_choice_type": choice_type or None,
            "source_explanation": explanation,
        },
        "split": benchmark_split,
    }


def parse_options(options_formatted: str, row_number: int) -> dict[str, str]:
    """Parse `A. ... | B. ...` option text into an ordered letter map."""

    matches = list(OPTION_MARKER_PATTERN.finditer(options_formatted))
    if not matches:
        raise MedMCQAAdapterError(f"row {row_number}: options_formatted is empty")

    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1)
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(options_formatted)
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


def _optional_text(value: str | None) -> str | None:
    cleaned = _clean_text(value or "")
    return cleaned or None


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\\n", " ").split())


def _clean_option_text(value: str) -> str:
    return _clean_text(value).strip(" |")


def _benchmark_id(benchmark_version: str, source_id: str) -> str:
    version = benchmark_version if benchmark_version.startswith("v") else f"v{benchmark_version}"
    return f"mamabench_{version}_mcq_medmcqa_{source_id}"


def _classify_domain_and_age(
    *, subject: str, topic: str, question: str
) -> tuple[str, str]:
    text = f"{topic} {question}".lower()

    if any(keyword in text for keyword in NEONATAL_KEYWORDS):
        return "neonatal", "neonate"
    if any(keyword in text for keyword in INFANT_KEYWORDS):
        return "infant", "infant"

    if subject == "Gynaecology & Obstetrics":
        age_group = "maternal" if any(k in text for k in MATERNAL_KEYWORDS) else "adult"
        return "obgyn", age_group

    if subject == "Pediatrics":
        return "pediatric", "child"

    return "unknown", "unknown"


def _classify_task_type(question: str, topic: str) -> str:
    text = f"{question} {topic}".lower()
    if "dose" in text or "dosage" in text:
        return "dosage"
    if any(word in text for word in ("treatment", "management", "drug", "therapy")):
        return "treatment"
    if any(word in text for word in ("diagnosis", "investigation", "marker", "cause")):
        return "diagnosis"
    if any(word in text for word in ("prevent", "prophylaxis", "contraceptive")):
        return "prevention"
    if any(word in text for word in ("emergency", "urgent", "refer")):
        return "triage"
    return "factual_lookup"


def _tags(*, subject: str, topic: str, choice_type: str) -> list[str]:
    tags = ["medmcqa"]
    tags.extend(_tag_tokens(subject))
    tags.extend(_tag_tokens(topic))
    if choice_type:
        tags.append(f"choice_type:{choice_type.lower()}")
    return _dedupe(tags)


def _tag_tokens(value: str) -> Iterable[str]:
    cleaned = value.strip().lower()
    if not cleaned:
        return []
    return [re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
