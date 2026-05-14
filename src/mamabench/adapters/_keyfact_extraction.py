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


# JSON Schema kept as documentation of the expected output shape. NOT
# passed as response_format/json_schema on the vLLM request — that path
# silently disables reasoning_content extraction in vLLM 0.20.2 (V1 engine
# limitation; the only known fixes are dropping json_schema or switching
# to the V0 engine). We instead let the model emit free-form JSON
# alongside its native <think>...</think> reasoning, then parse content
# manually (parse_extraction handles code-fence stripping and validation).
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
    Native chain-of-thought lives in ``message.reasoning_content`` (captured
    separately by the with-reasoning completer) — *not* in this output.
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
    and the model's native reasoning_content.

    Returns ``(extraction_dict, reasoning_text_or_None)`` where the dict has
    ``{summary, key_facts}`` and the reasoning is the model's native
    <think>...</think> chain-of-thought parsed out by vLLM's
    ``--reasoning-parser qwen3``. Reasoning is ``None`` when the server
    isn't configured to parse it or the model isn't a thinking model.
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


def load_keyfacts_by_row_id(
    keyfacts_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load a keyfacts side-file JSONL into ``{row_id: extraction_metadata}``.

    Each value contains the per-row extraction metadata that gets folded
    into ``source.metadata.key_fact_extraction`` of the open-ended adapter
    rows:

        {"model": str, "prompt_version": str, "summary": str, "key_facts": list[str]}

    The full chain-of-thought ``reasoning`` lives in the parallel
    ``<source>_keyfacts_reasoning.jsonl`` side-file and is *not* surfaced on
    the row. Rationale: keeping rows lean (~1 KB each vs. ~3 KB with
    reasoning inlined) makes the dataset slimmer to ship and load, while
    the side-file remains available alongside it for full audit. The
    reasoning side-file is shipped to HuggingFace under the same key_facts/
    directory so consumers can join by row_id when they want the audit
    trail.

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
            }
    return by_id
