from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.kenya import (
    KenyaAdapterError,
    KenyaAdapterStats,
    KenyaSourceRow,
    build_kenya_source_metadata,
    load_kenya,
    load_verdicts,
    normalize_kenya_row,
)


def make_verdict(category: str = "MATERNAL", row_id: str = "7") -> dict:
    return {
        "row_id": row_id,
        "model": "Qwen/Qwen3.6-27B-FP8",
        "prompt_version": "v6",
        "category": category,
        "rationale": "Postpartum care case",
    }


class NormalizeKenyaRowTests(unittest.TestCase):
    def test_emits_open_ended_v0_4_row(self) -> None:
        row = normalize_kenya_row(
            KenyaSourceRow(
                study_id="7",
                scenario="A 28-year-old G2P1 in labor with preeclampsia signs.",
                clinician_response="Refer urgently, give magnesium sulfate.",
            ),
            verdict=make_verdict("MATERNAL"),
            benchmark_version="v0.2",
        )
        self.assertEqual(row["schema_version"], "0.4")
        self.assertEqual(row["set_type"], "open_ended")
        self.assertEqual(row["id"], "mamabench_v0.2_kenya_7")
        self.assertEqual(row["question"], "A 28-year-old G2P1 in labor with preeclampsia signs.")
        self.assertEqual(row["answer"], "Refer urgently, give magnesium sulfate.")
        self.assertEqual(row["source"]["dataset"], "Kenya-Clinical-Vignettes")
        self.assertEqual(row["source"]["id"], "7")
        meta = row["source"]["metadata"]
        self.assertEqual(meta["obgyn_classification"]["category"], "MATERNAL")
        # open_ended rows must NOT have choices / answer_index / rubrics
        for forbidden in ("choices", "answer_index", "rubrics"):
            self.assertNotIn(forbidden, row)

    def test_strips_whitespace(self) -> None:
        row = normalize_kenya_row(
            KenyaSourceRow(
                study_id="9",
                scenario="\n\n  scenario text  \n\n",
                clinician_response="\n  response  \n",
            ),
            verdict=make_verdict("CHILD_HEALTH", row_id="9"),
            benchmark_version="v0.2",
        )
        self.assertEqual(row["question"], "scenario text")
        self.assertEqual(row["answer"], "response")

    def test_rejects_empty_scenario(self) -> None:
        with self.assertRaisesRegex(KenyaAdapterError, "empty scenario"):
            normalize_kenya_row(
                KenyaSourceRow(study_id="x", scenario="   ", clinician_response="ok"),
                verdict=make_verdict("MATERNAL"),
                benchmark_version="v0.2",
            )

    def test_rejects_empty_response(self) -> None:
        with self.assertRaisesRegex(KenyaAdapterError, "empty clinician_response"):
            normalize_kenya_row(
                KenyaSourceRow(study_id="x", scenario="ok", clinician_response="   "),
                verdict=make_verdict("MATERNAL"),
                benchmark_version="v0.2",
            )


class LoadVerdictsTests(unittest.TestCase):
    def test_filters_invalid_categories_and_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v.jsonl"
            path.write_text(
                json.dumps(make_verdict("MATERNAL", row_id="1")) + "\n"
                + json.dumps({**make_verdict(), "row_id": "2", "category": "OBGYN"}) + "\n"
                + json.dumps({**make_verdict(), "row_id": "3", "category": "NONE"}) + "\n",
                encoding="utf-8",
            )
            verdicts = load_verdicts(path)
            self.assertEqual(set(verdicts.keys()), {"1", "3"})  # OBGYN dropped


class LoadKenyaTests(unittest.TestCase):
    def _make_xlsx(self, tmpdir: Path) -> Path:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")
        path = tmpdir / "prompts.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["StudyID", "User Prompt", "Clinician response", "category"])
        ws.append([1, "Maternal scenario", "Maternal response", "MATERNAL"])
        ws.append([2, "Cardiac scenario", "Cardiac response", "OTHER"])
        ws.append([3, "Pediatric scenario", "Pediatric response", "CHILD_HEALTH"])
        ws.append([None, None, None, None])  # trailing empty row
        wb.save(str(path))
        return path

    def test_filters_by_verdict_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            xlsx_path = self._make_xlsx(tmp)
            verdicts_path = tmp / "verdicts.jsonl"
            verdicts_path.write_text(
                json.dumps(make_verdict("MATERNAL", row_id="1")) + "\n"
                + json.dumps(make_verdict("NONE", row_id="2")) + "\n"
                + json.dumps(make_verdict("CHILD_HEALTH", row_id="3")) + "\n",
                encoding="utf-8",
            )

            rows, stats = load_kenya(
                xlsx_path,
                verdicts_path,
                benchmark_version="v0.2",
            )
            self.assertEqual(len(rows), 2)
            ids = {row["source"]["id"] for row in rows}
            self.assertEqual(ids, {"1", "3"})
            self.assertEqual(stats.total, 3)
            self.assertEqual(stats.kept, 2)
            self.assertEqual(stats.skipped_none, 1)
            self.assertEqual(stats.skipped_no_verdict, 0)

    def test_skipped_no_verdict_counts_when_verdict_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            xlsx_path = self._make_xlsx(tmp)
            verdicts_path = tmp / "verdicts.jsonl"
            # only verdict for row 1; rows 2 and 3 have no verdict
            verdicts_path.write_text(
                json.dumps(make_verdict("MATERNAL", row_id="1")) + "\n",
                encoding="utf-8",
            )
            rows, stats = load_kenya(
                xlsx_path, verdicts_path, benchmark_version="v0.2"
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(stats.skipped_no_verdict, 2)


class BuildKenyaSourceMetadataTests(unittest.TestCase):
    def test_contains_url_and_license(self) -> None:
        meta = build_kenya_source_metadata()
        kenya = meta["Kenya-Clinical-Vignettes"]
        self.assertEqual(kenya["license"], "MIT")
        self.assertTrue(kenya["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
