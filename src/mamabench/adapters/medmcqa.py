"""Adapter for the filtered MedMCQA OBGYN/Pediatrics TSV."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from mamabench.config import normalize_benchmark_version
from mamabench.schema import SCHEMA_VERSION


MEDMCQA_SOURCE_DATASET = "MedMCQA"
MEDMCQA_SOURCE_URL = "https://huggingface.co/datasets/openlifescienceai/medmcqa"
MEDMCQA_LICENSE = "Apache-2.0"
PREPARED_INPUT_REPOSITORY = "https://github.com/nmrenyi/obgyn-qa-collection"
PREPARED_INPUT_REPO_NAME = "obgyn-qa-collection"
PREPARED_INPUT_PATH = "medmcqa/data/obgyn_mcq.tsv"
PREPARED_INPUT_METADATA: dict[str, Any] = {
    "expected_repository": PREPARED_INPUT_REPOSITORY,
    "expected_path": PREPARED_INPUT_PATH,
    "description": (
        "Pre-filtered OBGYN/Pediatrics MedMCQA subset used as mamabench "
        "adapter input."
    ),
    "filtering_done_outside_mamabench": True,
    "verified": False,
}

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


def build_medmcqa_source_metadata(input_tsv: str | Path | None = None) -> dict[str, Any]:
    """Build dataset-level MedMCQA provenance for the manifest."""

    prepared_input = dict(PREPARED_INPUT_METADATA)
    git_metadata = _prepared_input_git_metadata(input_tsv)
    if git_metadata is not None:
        prepared_input.update(git_metadata)

    return {
        MEDMCQA_SOURCE_DATASET: {
            "url": MEDMCQA_SOURCE_URL,
            "license": MEDMCQA_LICENSE,
            "prepared_input": prepared_input,
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
    version = normalize_benchmark_version(benchmark_version)
    return f"mamabench_{version}_medmcqa_{source_id}"


def _prepared_input_git_metadata(input_tsv: str | Path | None) -> dict[str, Any] | None:
    if input_tsv is None:
        return None

    input_path = Path(input_tsv).resolve()
    git_directory = input_path.parent if input_path.is_file() else input_path

    try:
        repo_root = Path(
            _git_output(git_directory, "rev-parse", "--show-toplevel")
        ).resolve()
    except (OSError, subprocess.CalledProcessError):
        return None

    if repo_root.name != PREPARED_INPUT_REPO_NAME:
        return None

    try:
        relative_path = input_path.relative_to(repo_root).as_posix()
        if relative_path != PREPARED_INPUT_PATH:
            return {"actual_path": relative_path}

        commit = _git_output(repo_root, "rev-parse", "HEAD")
        dirty = bool(_git_output(repo_root, "status", "--porcelain"))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None

    return {
        "repository": PREPARED_INPUT_REPOSITORY,
        "path": relative_path,
        "commit": commit,
        "git_dirty": dirty,
        "verified": True,
    }


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


MEDMCQA_SOURCE_METADATA = build_medmcqa_source_metadata()
