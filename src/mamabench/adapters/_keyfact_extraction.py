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
from pathlib import Path
from typing import Any, Callable

# Per-claim length cap. Matches the value documented in the
# prompts/keyfact_extractor.md quality_rules section (rule 4 — concise).
MAX_CLAIM_LEN = 200
# Summary length cap. Headroom over "1–2 sentences" to allow the model some
# flex without producing paragraph-long summaries.
MAX_SUMMARY_LEN = 500
# Reasoning length cap. The model writes its chain-of-thought through the
# 3-step extraction procedure into this field; 10,000 chars (~2,500 tokens)
# gives ample room for thorough reasoning on a complex multi-paragraph
# reference without letting reasoning runaway eat the whole context.
MAX_REASONING_LEN = 10000


# JSON Schema for guided / structured generation runtimes (vLLM
# response_format with json_schema, TGI grammar, etc.). The ``reasoning``
# field captures the model's chain-of-thought as a structured output
# field — needed because vLLM 0.20.2 silently disables reasoning_content
# parsing when response_format is set on a request, even with
# --reasoning-parser configured. Putting reasoning inside the schema is
# the only reliable way to capture it without giving up structured output.
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REASONING_LEN,
        },
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
    "required": ["reasoning", "summary", "key_facts"],
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
    """Parse an LLM extraction response.

    Returns ``{"reasoning", "summary", "key_facts"}`` on success. Raises
    :class:`ExtractionError` with a short, debuggable message when the raw
    output cannot be parsed or the parsed result fails validation.
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
    reasoning = obj.get("reasoning")
    summary = obj.get("summary")
    key_facts = obj.get("key_facts")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ExtractionError("reasoning must be a non-empty string")
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
    return {
        "reasoning": reasoning.strip(),
        "summary": summary.strip(),
        "key_facts": cleaned_facts,
    }


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


# NOTE: an earlier extract_row_with_reasoning variant captured Qwen's
# native thinking-mode reasoning_content via a separate completer. That
# path is dead — vLLM 0.20.2 silently suppresses reasoning_content when
# json_schema is set, and the --structured-outputs-config.enable_in_reasoning
# flag that's supposed to fix it caused 100% empty content on both
# Qwen3.5-397B-A17B-FP8 and Qwen3.6-27B-FP8. Reasoning now lives in the
# JSON schema as a required string field (see EXTRACTION_JSON_SCHEMA).
# Callers should use :func:`extract_row` and read ``result["reasoning"]``.


def load_keyfacts_by_row_id(
    keyfacts_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load a keyfacts side-file JSONL into ``{row_id: extraction_metadata}``.

    Each value contains the full per-row extraction metadata that gets folded
    into ``source.metadata.key_fact_extraction`` of the open-ended adapter
    rows:

        {
          "model":          str,
          "prompt_version": str,
          "summary":        str,
          "key_facts":      list[str],
          "reasoning":      str,   # merged from the parallel *_reasoning.jsonl
        }

    The ``reasoning`` field is auto-merged from a sibling
    ``<source>_keyfacts_reasoning.jsonl`` file in the same directory (if it
    exists). Surfacing reasoning on the row — rather than only in the
    side-file — makes the HF release fully self-describing: each entry
    carries the model's chain-of-thought for the extraction, so dataset
    users can audit the filtering/extraction process without needing to
    download a separate side-file.

    Lines missing ``row_id`` (or with unparseable JSON) are skipped silently.
    """
    keyfacts_path = Path(keyfacts_path)
    by_id: dict[str, dict[str, Any]] = {}
    with keyfacts_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = rec.get("row_id")
            if not isinstance(row_id, str):
                continue
            by_id[row_id] = {
                "model": rec.get("model"),
                "prompt_version": rec.get("prompt_version"),
                "summary": rec.get("summary"),
                "key_facts": rec.get("key_facts", []),
                "reasoning": "",
            }

    # Merge reasoning from the parallel side-file when present. We derive
    # the path from the keyfacts filename: foo_keyfacts.jsonl →
    # foo_keyfacts_reasoning.jsonl. Missing reasoning file is not an error
    # — the row just keeps reasoning="" and the consumer can detect it.
    reasoning_path = keyfacts_path.with_name(keyfacts_path.stem + "_reasoning.jsonl")
    if reasoning_path.is_file():
        with reasoning_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_id = rec.get("row_id")
                reasoning = rec.get("reasoning")
                if (
                    isinstance(row_id, str)
                    and isinstance(reasoning, str)
                    and row_id in by_id
                ):
                    by_id[row_id]["reasoning"] = reasoning

    return by_id
