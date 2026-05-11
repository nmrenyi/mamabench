"""Shared option-parsing and text-cleaning helpers for MCQ source adapters."""

from __future__ import annotations

import re


OPTION_MARKER_PATTERN = re.compile(r"(?:^|\s\|\s)([A-Z])\.\s*")


def parse_options(
    options_formatted: str,
    *,
    row_number: int,
    error_cls: type[Exception],
) -> dict[str, str]:
    """Parse `A. ... | B. ...` option text into an ordered letter map."""

    matches = list(OPTION_MARKER_PATTERN.finditer(options_formatted))
    if not matches:
        raise error_cls(f"row {row_number}: options_formatted is empty")

    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1)
        next_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(options_formatted)
        )
        text = clean_option_text(options_formatted[match.end() : next_start])
        if letter in parsed:
            raise error_cls(
                f"row {row_number}: duplicate option letter {letter!r}"
            )
        if not text:
            raise error_cls(
                f"row {row_number}: empty option text for {letter!r}"
            )
        parsed[letter] = text

    return parsed


def clean_text(value: str) -> str:
    return " ".join(str(value).replace("\\n", " ").split())


def clean_option_text(value: str) -> str:
    return clean_text(value).strip(" |")
