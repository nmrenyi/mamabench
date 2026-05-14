"""Key-fact extractor — extraction-result parsing, schema, and LLM driver.

The extractor sends a system prompt + a single user message (one row at a
time) to a chat-style LLM and expects a single JSON object back of the shape:

    {
      "summary": "<1-2 sentence summary>",
      "key_facts": ["<atomic claim>", "<atomic claim>", ...]
    }

This module is structured to mirror :mod:`mamabench.obgyn_classifier`:

- :func:`parse_extraction` is pure: takes the LLM's raw string output,
  returns a validated result dict or raises :class:`ExtractionError`.
- :func:`format_user_message` builds the user-message text from a question
  and reference response.
- :func:`extract_row` is the thin glue layer that calls a pluggable chat
  completion callable and parses its result.

The OpenAI-compatible chat completer used in cluster runs lives in
:mod:`mamabench.obgyn_classifier` (``make_openai_completer``) and is reused
verbatim — the structured-output / disable-thinking knobs apply identically.
"""

from __future__ import annotations

import json
from typing import Any, Callable

# Per-claim length cap. Matches the value documented in
# prompts/keyfact_extractor/quality_rules.md (rule 4 — concise).
MAX_CLAIM_LEN = 200
# Summary length cap. Headroom over "1–2 sentences" to allow the model some
# flex without producing paragraph-long summaries.
MAX_SUMMARY_LEN = 500


# JSON Schema for guided / structured generation runtimes (vLLM
# response_format with json_schema, TGI grammar, etc.).
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SUMMARY_LEN,
        },
        "key_facts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CLAIM_LEN,
            },
        },
    },
    "required": ["summary", "key_facts"],
    "additionalProperties": False,
}


ChatCompleter = Callable[[list[dict[str, str]]], str]
ChatCompleterWithReasoning = Callable[
    [list[dict[str, str]]], tuple[str, str | None]
]


class ExtractionError(Exception):
    """Raised when the LLM output cannot be parsed as a valid extraction."""


def _strip_code_fence(text: str) -> str:
    """Remove a single surrounding ``` or ```json fence if present."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline == -1:
        return s
    s = s[newline + 1 :]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[: -len("```")]
    return s.strip()


def parse_extraction(raw: str) -> dict[str, Any]:
    """Parse an LLM extraction response into ``{"summary", "key_facts"}``.

    Raises :class:`ExtractionError` with a short, debuggable message when the
    raw output cannot be parsed or the parsed result fails validation.
    """
    text = _strip_code_fence(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ExtractionError(
            f"could not parse JSON from LLM output ({e.msg} at pos {e.pos}); "
            f"raw output (first 200 chars): {raw[:200]!r}"
        ) from None
    if not isinstance(obj, dict):
        raise ExtractionError(
            f"LLM returned non-object JSON ({type(obj).__name__}); "
            f"raw output (first 200 chars): {raw[:200]!r}"
        )
    summary = obj.get("summary")
    key_facts = obj.get("key_facts")
    if not isinstance(summary, str) or not summary.strip():
        raise ExtractionError("summary must be a non-empty string")
    if not isinstance(key_facts, list) or not key_facts:
        raise ExtractionError("key_facts must be a non-empty list")
    cleaned_facts: list[str] = []
    for i, fact in enumerate(key_facts):
        if not isinstance(fact, str):
            raise ExtractionError(
                f"key_facts[{i}] must be a string, got {type(fact).__name__}"
            )
        stripped = fact.strip()
        if not stripped:
            raise ExtractionError(f"key_facts[{i}] is empty after stripping whitespace")
        if len(stripped) > MAX_CLAIM_LEN:
            raise ExtractionError(
                f"key_facts[{i}] exceeds {MAX_CLAIM_LEN} chars (got {len(stripped)})"
            )
        cleaned_facts.append(stripped)
    return {"summary": summary.strip(), "key_facts": cleaned_facts}


def format_user_message(question: str, reference: str) -> str:
    """Build the user-message text from a question + reference response.

    The shape is fixed by ``prompts/keyfact_extractor/input_format.md`` and
    must stay in sync with that module.
    """
    return f"Question:\n{question.strip()}\n\nReference response:\n{reference.strip()}"


def extract_row(
    *,
    complete: ChatCompleter,
    system_prompt: str,
    question: str,
    reference: str,
) -> dict[str, Any]:
    """Send one (system, user) chat to ``complete`` and parse the extraction.

    For the audit-grade path that also captures the model's reasoning_content,
    use :func:`extract_row_with_reasoning`.
    """
    user_message = format_user_message(question, reference)
    raw = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return parse_extraction(raw)


def extract_row_with_reasoning(
    *,
    complete: ChatCompleterWithReasoning,
    system_prompt: str,
    question: str,
    reference: str,
) -> tuple[dict[str, Any], str | None]:
    """Send one (system, user) chat and return both the parsed extraction
    and the model's reasoning_content (or ``None`` if the server isn't
    parsing reasoning out).

    Returns ``(extraction_dict, reasoning_text_or_None)``.
    """
    user_message = format_user_message(question, reference)
    content, reasoning = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    extraction = parse_extraction(content)
    return extraction, reasoning
