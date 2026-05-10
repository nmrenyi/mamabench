from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mamabench.manifest import build_manifest
from mamabench.validate import validate_items


class BuildManifestTests(unittest.TestCase):
    def test_manifest_embeds_full_validation_report(self) -> None:
        row = {
            "id": "mamabench_v0.1_unit_test",
            "schema_version": "0.3",
            "set_type": "mcq",
            "question": "Which answer is correct?",
            "choices": ["Correct", "Incorrect"],
            "answer": "A",
            "answer_index": 0,
            "source": {
                "dataset": "unit_test",
                "id": "source-1",
                "answer": "A",
            },
        }
        report = validate_items([row])

        manifest = build_manifest(
            [row],
            benchmark_version="v0.1",
            source_dataset_metadata={
                "unit_test": {
                    "url": "https://example.test/unit",
                    "license": "synthetic",
                }
            },
            validation_report=report,
            created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        )

        self.assertEqual(
            manifest["source_datasets"]["unit_test"],
            {"license": "synthetic", "url": "https://example.test/unit"},
        )
        self.assertEqual(manifest["validation"], report.to_dict())
        self.assertFalse(manifest["validation"]["ok"])
        self.assertEqual(manifest["validation"]["item_count"], 1)
        self.assertGreater(len(manifest["validation"]["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
