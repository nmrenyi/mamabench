from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.whb import (
    WHBAdapterError,
    build_whb_source_metadata,
    load_whb,
    normalize_whb_row,
)


class NormalizeWhbRowTests(unittest.TestCase):
    def test_emits_open_ended_v0_4_row(self) -> None:
        row = normalize_whb_row(
            {
                "question_clean": "I have a milk bleb. What should I do?",
                "expert_justification": "Treat with steroid cream and sunflower lecithin.",
            },
            row_number=1,
            benchmark_version="v0.2",
        )
        self.assertEqual(row["schema_version"], "0.4")
        self.assertEqual(row["set_type"], "open_ended")
        self.assertTrue(row["id"].startswith("mamabench_v0.2_whb_"))
        self.assertEqual(row["source"]["dataset"], "WHB")
        self.assertNotIn("metadata", row["source"])

    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(WHBAdapterError, "missing question_clean"):
            normalize_whb_row(
                {"question_clean": " ", "expert_justification": "A"},
                row_number=1,
                benchmark_version="v0.2",
            )

    def test_rejects_empty_justification(self) -> None:
        with self.assertRaisesRegex(WHBAdapterError, "missing expert_justification"):
            normalize_whb_row(
                {"question_clean": "Q", "expert_justification": " "},
                row_number=1,
                benchmark_version="v0.2",
            )


class LoadWhbTests(unittest.TestCase):
    def test_walks_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "whb.tsv"
            path.write_text(
                "question_clean\texpert_justification\n" "Q1\tA1\nQ2\tA2\n",
                encoding="utf-8",
            )
            rows, stats = load_whb(path, benchmark_version="v0.2")
            self.assertEqual(len(rows), 2)
            self.assertEqual(stats.kept, 2)

    def test_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "whb.tsv"
            path.write_text("foo\tbar\n1\t2\n", encoding="utf-8")
            with self.assertRaisesRegex(WHBAdapterError, "missing required columns"):
                load_whb(path, benchmark_version="v0.2")


class BuildSourceMetadataTests(unittest.TestCase):
    def test_url_and_license(self) -> None:
        meta = build_whb_source_metadata()
        whb = meta["WHB"]
        self.assertEqual(whb["license"], "CC-BY-SA-4.0")
        self.assertTrue(whb["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
