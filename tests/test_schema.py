from __future__ import annotations

import json
import unittest
from pathlib import Path

from mamabench.config import load_project_config
from mamabench.schema import (
    CANONICAL_FIELDS,
    CONTROLLED_VOCABULARIES,
    REQUIRED_SOURCE_FIELDS,
    SCHEMA_VERSION,
    SOURCE_FIELDS,
)


ROOT = Path(__file__).resolve().parents[1]


class SchemaConsistencyTests(unittest.TestCase):
    def test_json_schema_matches_python_schema_constants(self) -> None:
        config = load_project_config(ROOT / "mamabench.json")
        schema = json.loads(config.schema_file.read_text(encoding="utf-8"))

        self.assertEqual(schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertEqual(tuple(schema["required"]), CANONICAL_FIELDS)
        self.assertEqual(tuple(schema["properties"]), CANONICAL_FIELDS)

        source_schema = schema["properties"]["source"]
        self.assertEqual(tuple(source_schema["required"]), REQUIRED_SOURCE_FIELDS)
        self.assertEqual(tuple(source_schema["properties"]), SOURCE_FIELDS)

        allowed_set_types = CONTROLLED_VOCABULARIES["set_type"]
        self.assertEqual(
            allowed_set_types,
            frozenset({schema["properties"]["set_type"]["const"]}),
        )


if __name__ == "__main__":
    unittest.main()
