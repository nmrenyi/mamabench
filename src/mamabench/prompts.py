"""Assemble the OBGYN classifier system prompt from modular markdown sections.

The prompt lives under `prompts/obgyn_classifier/` as small markdown modules
(intro, categories, decision rule, guidance, input format, output format).
Two modes — `openended` (HealthBench, Kenya Clinical Vignettes) and `mcq`
(MedQA-USMLE) — share the same categories / decision rule / output format
modules, and differ only in their intro, guidance, and input-format modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PROMPT_VERSION = "v6"

Mode = Literal["openended", "mcq"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / "prompts" / "obgyn_classifier"


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
    """Return the assembled system prompt for the given mode.

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
