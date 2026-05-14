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


# JSON Schema kept as documentation of the expected output shape. NOT
# passed as response_format/json_schema on the vLLM request — that path
# silently disables reasoning_content extraction in vLLM 0.20.2's V1
# engine. We let the model emit free-form JSON alongside its native
# <think>...</think> reasoning, then parse content manually
# (parse_verdict handles code-fence stripping and validation).
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

# Same shape as ChatCompleter but returns ``(content, reasoning_content)`` —
# the latter is the model's thinking-mode chain-of-thought, exposed by vLLM
# only when the server is started with ``--enable-reasoning --reasoning-parser
# qwen3`` (or equivalent for other model families). ``reasoning_content`` is
# ``None`` when the model isn't a thinking model, when thinking is disabled
# in the request, or when the server isn't parsing reasoning out of the
# completion content.
ChatCompleterWithReasoning = Callable[
    [list[dict[str, str]]], tuple[str, str | None]
]


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
    Native chain-of-thought lives in ``message.reasoning_content`` — *not*
    in this output.
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


def _make_openai_client(
    *,
    base_url: str | None,
    api_key: str,
    timeout: float,
):
    """Build an OpenAI client; raise with an install hint if the SDK is missing."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai SDK is required for make_openai_completer — "
            "install with `pip install openai>=1.0`"
        ) from e
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _build_request_kwargs(
    *,
    model: str,
    temperature: float,
    json_schema: dict | None,
    disable_thinking: bool,
    thinking_budget: int | None,
    max_tokens: int | None,
    extra_body: dict | None,
    schema_name: str = "verdict",
) -> dict[str, Any]:
    """Shared request-kwargs builder for the OpenAI-compatible chat endpoint.

    Captures structured-output, thinking-mode, and budget knobs in one place
    so multiple completer flavors (content-only / with-reasoning) stay in sync.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            },
        }
    merged_extra: dict[str, Any] = {}
    chat_template_kwargs: dict[str, Any] = {}
    if disable_thinking:
        chat_template_kwargs["enable_thinking"] = False
    elif thinking_budget is not None:
        chat_template_kwargs["thinking_budget"] = thinking_budget
    if chat_template_kwargs:
        merged_extra["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        merged_extra.update(extra_body)
    if merged_extra:
        kwargs["extra_body"] = merged_extra
    return kwargs


def make_openai_completer(
    *,
    model: str,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    timeout: float = 300.0,
    json_schema: dict | None = None,
    disable_thinking: bool = True,
    thinking_budget: int | None = None,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
) -> ChatCompleter:
    """Build a :data:`ChatCompleter` backed by the OpenAI Python SDK.

    Works against vLLM / TGI / SGLang / Ollama via their OpenAI-compatible
    endpoints — point ``base_url`` at the server (e.g. ``http://host:8000/v1``).

    Structured output:

    - When ``json_schema`` is provided, the request includes
      ``response_format={"type": "json_schema", ...}`` — this is the
      OpenAI-standard structured-generation mechanism, supported by vLLM
      0.10+, TGI, SGLang, and OpenAI itself. The previous ``guided_json``
      extra_body knob was not honored on vLLM 0.20.2 in our cluster runs,
      so this is the path we rely on.

    Qwen3+ thinking mode:

    - When ``disable_thinking`` is True, the request adds
      ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.
      Qwen3 / Qwen3.5 / Qwen3.6 default to thinking mode, which produces
      verbose chain-of-thought before any JSON output. For a fast
      classification task we usually don't want that — disabling thinking
      gives clean direct JSON.
    - When thinking is *enabled* (``disable_thinking=False``) the optional
      ``thinking_budget`` (an integer token count) is forwarded as
      ``extra_body.chat_template_kwargs.thinking_budget`` — Qwen3+ uses this
      as a soft cap on the reasoning portion so the model self-wraps and
      starts emitting the final answer before the budget is exhausted.

    Hard output cap:

    - ``max_tokens`` (when provided) is forwarded as the OpenAI-standard
      ``max_tokens`` parameter and caps the total completion length
      (reasoning + final answer combined). It's the universal safety net in
      case ``thinking_budget`` is ignored by the server.

    ``extra_body`` is merged after the thinking-mode setting, so callers can
    pass additional vLLM-specific knobs without clobbering it.

    Returns content only. If you also need the model's reasoning_content
    (thinking-mode CoT, parsed out by vLLM with ``--reasoning-parser ...``),
    use :func:`make_openai_completer_with_reasoning` instead.
    """
    client = _make_openai_client(
        base_url=base_url, api_key=api_key, timeout=timeout
    )

    def complete(messages: Iterable[dict[str, str]]) -> str:
        kwargs = _build_request_kwargs(
            model=model,
            temperature=temperature,
            json_schema=json_schema,
            disable_thinking=disable_thinking,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        kwargs["messages"] = list(messages)
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    return complete


def make_openai_completer_with_reasoning(
    *,
    model: str,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    timeout: float = 300.0,
    json_schema: dict | None = None,
    disable_thinking: bool = False,
    thinking_budget: int | None = None,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
    schema_name: str = "extraction",
) -> ChatCompleterWithReasoning:
    """Like :func:`make_openai_completer` but the returned callable yields
    ``(content, reasoning_content)``.

    ``reasoning_content`` is the model's thinking-mode CoT exposed by vLLM
    in ``response.choices[0].message.reasoning_content`` when the server is
    started with ``--enable-reasoning --reasoning-parser qwen3`` (or
    equivalent). It is ``None`` when reasoning is disabled, when the model
    isn't a thinking model, or when the server isn't parsing reasoning out.

    Default ``disable_thinking=False`` — this completer is intended for
    tasks that want the model's reasoning, so thinking is on by default.
    """
    client = _make_openai_client(
        base_url=base_url, api_key=api_key, timeout=timeout
    )

    def complete(
        messages: Iterable[dict[str, str]],
    ) -> tuple[str, str | None]:
        kwargs = _build_request_kwargs(
            model=model,
            temperature=temperature,
            json_schema=json_schema,
            disable_thinking=disable_thinking,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            extra_body=extra_body,
            schema_name=schema_name,
        )
        kwargs["messages"] = list(messages)
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        return content, reasoning

    return complete
