"""Shared v0.4 JSON Schema validation helper for adapter scripts.

Each v0.2 adapter script writes a per-source manifest that needs a
``validation`` block matching the shape produced by
:class:`mamabench.validate.ValidationReport.to_dict`. The legacy v0.3
validator is set-type-blind (it expects every row to be MCQ-shaped), so
we validate v0.4 rows against the JSON Schema directly via jsonschema-py.

Returns a dict that drops cleanly into a manifest under
``manifest["validation"] = ...``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "mamabench_v0.4.schema.json"
)


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a list of v0.4 mamabench rows against the JSON Schema.

    Returns a dict shaped like :meth:`ValidationReport.to_dict` (with
    ``ok``, ``item_count``, ``error_count``, ``issues``). When
    jsonschema-py is unavailable, returns ``ok=True`` with a ``note``
    so the run doesn't fail just because the dev environment is bare;
    CI environments install jsonschema and get full validation.
    """
    item_count = len(rows)
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return {
            "ok": True,
            "item_count": item_count,
            "error_count": 0,
            "issues": [],
            "note": "jsonschema-py not installed; row schema validation skipped",
        }

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    issues: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        for err in validator.iter_errors(row):
            issues.append(
                {
                    "severity": "error",
                    "line_number": i + 1,
                    "item_id": row.get("id"),
                    "field": ".".join(str(p) for p in err.absolute_path) or None,
                    "message": err.message[:200],
                }
            )
            break  # report the first error per row only
    return {
        "ok": not issues,
        "item_count": item_count,
        "error_count": len(issues),
        "issues": issues,
    }
