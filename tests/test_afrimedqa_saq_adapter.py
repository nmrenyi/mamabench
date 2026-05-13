from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.afrimedqa_saq import (
    AfriMedQASAQAdapterError,
    build_afrimedqa_saq_source_metadata,
    load_afrimedqa_saq,
    normalize_afrimedqa_saq_row,
)


class NormalizeAfrimedqaSaqRowTests(unittest.TestCase):
    def test_emits_open_ended_v0_4_row(self) -> None:
        row = normalize_afrimedqa_saq_row(
            {
                "question_clean": "List the complications of IUCDs.",
                "answer_rationale": "Perforation, expulsion, infection, dysmenorrhea...",
            },
            row_number=1,
            benchmark_version="v0.2",
        )
        self.assertEqual(row["schema_version"], "0.4")
        self.assertEqual(row["set_type"], "open_ended")
        self.assertTrue(row["id"].startswith("mamabench_v0.2_afrimedqa-saq_"))
        self.assertEqual(row["question"], "List the complications of IUCDs.")
        self.assertEqual(row["answer"], "Perforation, expulsion, infection, dysmenorrhea...")
        self.assertEqual(row["source"]["dataset"], "AfriMed-QA")
        self.assertEqual(row["source"]["metadata"], {"subset": "saq"})
        for forbidden in ("choices", "answer_index", "rubrics"):
            self.assertNotIn(forbidden, row)

    def test_id_stable_for_identical_content(self) -> None:
        a = normalize_afrimedqa_saq_row(
            {"question_clean": "Q.", "answer_rationale": "A."},
            row_number=1,
            benchmark_version="v0.2",
        )
        b = normalize_afrimedqa_saq_row(
            {"question_clean": "Q.", "answer_rationale": "A."},
            row_number=2,
            benchmark_version="v0.2",
        )
        self.assertEqual(a["id"], b["id"])

    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(AfriMedQASAQAdapterError, "missing question_clean"):
            normalize_afrimedqa_saq_row(
                {"question_clean": "  ", "answer_rationale": "A"},
                row_number=1,
                benchmark_version="v0.2",
            )

    def test_rejects_empty_answer(self) -> None:
        with self.assertRaisesRegex(AfriMedQASAQAdapterError, "missing answer_rationale"):
            normalize_afrimedqa_saq_row(
                {"question_clean": "Q", "answer_rationale": "  "},
                row_number=1,
                benchmark_version="v0.2",
            )


class LoadAfrimedqaSaqTests(unittest.TestCase):
    def test_walks_tsv_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "saq.tsv"
            path.write_text(
                "question_clean\tanswer_rationale\n"
                "Q1\tA1\n"
                "Q2\tA2\n",
                encoding="utf-8",
            )
            rows, stats = load_afrimedqa_saq(path, benchmark_version="v0.2")
            self.assertEqual(len(rows), 2)
            self.assertEqual(stats.total, 2)
            self.assertEqual(stats.kept, 2)

    def test_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "saq.tsv"
            path.write_text(
                "question\tresponse\nQ\tR\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AfriMedQASAQAdapterError, "missing required columns"):
                load_afrimedqa_saq(path, benchmark_version="v0.2")

    def test_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "saq.tsv"
            path.write_text(
                "question_clean\tanswer_rationale\n"
                + "\n".join(f"Q{i}\tA{i}" for i in range(10))
                + "\n",
                encoding="utf-8",
            )
            rows, _ = load_afrimedqa_saq(path, benchmark_version="v0.2", limit=3)
            self.assertEqual(len(rows), 3)


class BuildSourceMetadataTests(unittest.TestCase):
    def test_carries_url_and_license(self) -> None:
        meta = build_afrimedqa_saq_source_metadata()
        afri = meta["AfriMed-QA"]
        self.assertEqual(afri["license"], "CC-BY-NC-SA-4.0")
        self.assertTrue(afri["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
