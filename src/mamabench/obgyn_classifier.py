"""OBGYN specialty classifier — verdict parsing, schema, and LLM driver.

The classifier sends a system prompt + a single user message (one row at a
time) to a chat-style LLM and expects a single JSON object back of the shape:

    {"category": "<one of 5>", "rationale": "<≤120 chars>"}

This module is deliberately split from the I/O bits:

- :func:`parse_verdict` is pure: takes the LLM's raw string output, returns a
  validated verdict dict or raises :class:`ClassifierError`.
- :func:`classify_row` is the thin glue layer that calls a pluggable chat
  completion callable and parses its result.
- :func:`make_openai_completer` builds a chat completer backed by the OpenAI
  Python SDK, which works against vLLM / TGI / SGLang / Ollama's
  OpenAI-compatible endpoints.

For unit tests, supply any callable matching :data:`ChatCompleter` (e.g. a
stub that returns canned JSON strings).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable


CATEGORIES: frozenset[str] = frozenset(
    {"MATERNAL", "NEONATAL", "CHILD_HEALTH", "SEXUAL_AND_REPRODUCTIVE_HEALTH", "NONE"}
)


# JSON Schema for guided / structured generation runtimes (vLLM `guided_json`,
# TGI `grammar`, OpenAI `response_format=json_schema`, etc.).
VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": sorted(CATEGORIES),
        },
        "rationale": {
            "type": "string",
            "maxLength": 200,
        },
    },
    "required": ["category", "rationale"],
    "additionalProperties": False,
}


# Chat completion callable contract. Takes a list of role/content messages and
# returns the assistant content as a string. Errors raised by the underlying
# client propagate unchanged.
ChatCompleter = Callable[[list[dict[str, str]]], str]


class ClassifierError(Exception):
    """Raised when the LLM output cannot be parsed as a valid verdict."""


def _strip_code_fence(text: str) -> str:
    """Remove a single surrounding ``` or ```json fence if present."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    # drop the opening fence line (e.g. ``` or ```json)
    newline = s.find("\n")
    if newline == -1:
        return s
    s = s[newline + 1 :]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[: -len("```")]
    return s.strip()


def parse_verdict(raw: str) -> dict[str, str]:
    """Parse an LLM verdict response into ``{"category", "rationale"}``.

    Raises :class:`ClassifierError` with a short, debuggable message when the
    raw output cannot be parsed or the parsed verdict fails validation.
    """
    text = _strip_code_fence(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClassifierError(
            f"could not parse JSON from LLM output ({e.msg} at pos {e.pos}); "
            f"raw output (first 200 chars): {raw[:200]!r}"
        ) from None
    if not isinstance(obj, dict):
        raise ClassifierError(
            f"LLM returned non-object JSON ({type(obj).__name__}); "
            f"raw output (first 200 chars): {raw[:200]!r}"
        )
    category = obj.get("category")
    rationale = obj.get("rationale")
    if category not in CATEGORIES:
        raise ClassifierError(
            f"invalid category {category!r}; expected one of {sorted(CATEGORIES)}"
        )
    if not isinstance(rationale, str):
        raise ClassifierError(
            f"rationale must be a string, got {type(rationale).__name__}"
        )
    return {"category": category, "rationale": rationale}


def classify_row(
    *,
    complete: ChatCompleter,
    system_prompt: str,
    user_message: str,
) -> dict[str, str]:
    """Send one (system, user) chat to ``complete`` and parse the verdict."""
    raw = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return parse_verdict(raw)


def make_openai_completer(
    *,
    model: str,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    response_format: dict | None = None,
    extra_body: dict | None = None,
    timeout: float = 120.0,
) -> ChatCompleter:
    """Build a :data:`ChatCompleter` backed by the OpenAI Python SDK.

    Works against vLLM / TGI / SGLang / Ollama via their OpenAI-compatible
    endpoints — point ``base_url`` at the server (e.g. ``http://host:8000/v1``).
    ``response_format`` and ``extra_body`` are passed through unmodified so
    callers can configure structured generation per their runtime (vLLM uses
    ``extra_body={"guided_json": schema}``; OpenAI/TGI accept
    ``response_format={"type": "json_schema", ...}``).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai SDK is required for make_openai_completer — "
            "install with `pip install openai>=1.0`"
        ) from e
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(messages: Iterable[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    return complete


def vllm_guided_json_extra_body(schema: dict | None = None) -> dict:
    """Return the ``extra_body`` dict that enables vLLM's guided-JSON mode.

    When ``schema`` is ``None``, the default :data:`VERDICT_JSON_SCHEMA` is
    used. Pass the result as ``extra_body=`` to :func:`make_openai_completer`.
    """
    return {"guided_json": schema if schema is not None else VERDICT_JSON_SCHEMA}
