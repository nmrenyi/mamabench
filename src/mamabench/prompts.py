"""Assemble mamabench LLM-pipeline system prompts from markdown sources.

Two pipelines live here:

- **OBGYN classifier** (``load_classifier_prompt``): per-row category verdict
  from one of five categories. Two modes (``openended`` / ``mcq``) share three
  modules and swap three, so the prompt is split across multiple files under
  ``prompts/obgyn_classifier/`` and assembled at runtime.
- **Key-fact extractor** (``load_keyfact_extractor_prompt``): per-row atomic
  must-cover claim list extracted from a reference response. Single mode,
  single file: ``prompts/keyfact_extractor.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PROMPT_VERSION = "v8"
KEYFACT_EXTRACTOR_PROMPT_VERSION = "v1"

Mode = Literal["openended", "mcq"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / "prompts" / "obgyn_classifier"
_KEYFACT_PROMPT_PATH = _REPO_ROOT / "prompts" / "keyfact_extractor.md"


def _section_order(mode: Mode) -> list[str]:
    if mode not in ("openended", "mcq"):
        raise ValueError(f"unknown classifier mode: {mode!r}")
    return [
        f"intro_{mode}.md",
        "categories.md",
        "decision_rule.md",
        f"guidance_{mode}.md",
        f"input_format_{mode}.md",
        "output_format.md",
    ]


def load_classifier_prompt(mode: Mode, *, modules_dir: Path | None = None) -> str:
    """Return the assembled OBGYN classifier system prompt for the given mode.

    Modules are concatenated with a blank-line separator, each trimmed of
    surrounding whitespace so the assembled prompt has clean section breaks.
    """
    base = modules_dir or _MODULES_DIR
    sections = []
    for name in _section_order(mode):
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"missing prompt module: {path}")
        sections.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(sections)


def load_keyfact_extractor_prompt(*, prompt_path: Path | None = None) -> str:
    """Return the key-fact extractor system prompt.

    Read from a single file (``prompts/keyfact_extractor.md`` by default).
    Trailing/leading whitespace is stripped so the prompt has a clean shape
    when concatenated into chat messages.
    """
    path = prompt_path or _KEYFACT_PROMPT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()
