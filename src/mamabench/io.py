"""JSON and JSONL helpers for mamabench artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""

    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"{jsonl_path}:{line_number}: expected a JSON object"
                )

            rows.append(value)

    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of JSON objects as JSONL."""

    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON document with stable formatting."""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

