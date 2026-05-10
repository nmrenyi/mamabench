from __future__ import annotations

import json
import unittest
from pathlib import Path

from mamabench.config import load_project_config, normalize_benchmark_version
from mamabench.schema import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigTests(unittest.TestCase):
    def test_normalize_benchmark_version_adds_v_prefix(self) -> None:
        self.assertEqual(normalize_benchmark_version("0.2"), "v0.2")
        self.assertEqual(normalize_benchmark_version(" v0.2 "), "v0.2")

    def test_normalize_benchmark_version_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark_version"):
            normalize_benchmark_version(" ")

    def test_current_config_matches_code_and_schema_file(self) -> None:
        payload = json.loads((ROOT / "mamabench.json").read_text(encoding="utf-8"))
        config = load_project_config(ROOT / "mamabench.json")

        self.assertEqual(config.schema_version, SCHEMA_VERSION)
        self.assertEqual(config.benchmark_version, "v0.1")
        self.assertEqual(payload["benchmark_version"], config.benchmark_version)
        self.assertTrue(config.schema_file.is_file())
        self.assertEqual(config.benchmark_dir, ROOT / "benchmark" / "v0.1")


if __name__ == "__main__":
    unittest.main()
