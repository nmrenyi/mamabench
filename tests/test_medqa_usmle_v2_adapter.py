from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mamabench.adapters.medqa_usmle_v2 import (
    MedQAUSMLEv2AdapterError,
    build_medqa_usmle_v2_source_metadata,
    load_medqa_usmle_v2,
    normalize_medqa_usmle_source_row,
    synth_row_id,
)


def make_source_row(answer: str = "A") -> dict:
    return {
        "question": "A 32-year-old G2P1 at 36 weeks gestation presents with BP 162/110...",
        "options": {
            "A": "Magnesium sulfate and delivery plan",
            "B": "Outpatient monitoring",
            "C": "Discharge home",
            "D": "Routine prenatal visit",
        },
        "answer": answer,
        "meta_info": "step2",
    }


def make_verdict(category: str = "MATERNAL", row_id: str = "usmle_00000") -> dict:
    return {
        "row_id": row_id,
        "model": "Qwen/Qwen3.6-27B-FP8",
        "prompt_version": "v6",
        "category": category,
        "rationale": "Severe preeclampsia management",
    }


class SynthRowIdTests(unittest.TestCase):
    def test_zero_padded_5_digit(self) -> None:
        self.assertEqual(synth_row_id(0), "usmle_00000")
        self.assertEqual(synth_row_id(42), "usmle_00042")
        self.assertEqual(synth_row_id(14368), "usmle_14368")


class NormalizeMedqaUsmleSourceRowTests(unittest.TestCase):
    def test_emits_mcq_v0_4_row(self) -> None:
        row = normalize_medqa_usmle_source_row(
            make_source_row("A"),
            row_index=42,
            verdict=make_verdict("MATERNAL"),
            benchmark_version="v0.2",
        )
        self.assertEqual(row["schema_version"], "0.4")
        self.assertEqual(row["set_type"], "mcq")
        self.assertEqual(len(row["choices"]), 4)
        self.assertEqual(row["choices"][row["answer_index"]], row["answer"])
        self.assertEqual(row["source"]["dataset"], "MedQA-USMLE")
        self.assertEqual(row["source"]["answer"], "A")
        self.assertEqual(row["source"]["metadata"]["upstream_index"], "usmle_00042")
        self.assertEqual(row["source"]["metadata"]["meta_info"], "step2")
        self.assertEqual(
            row["source"]["metadata"]["obgyn_classification"]["category"],
            "MATERNAL",
        )
        # mcq rows must NOT have rubrics
        self.assertNotIn("rubrics", row)

    def test_id_format_includes_content_hash(self) -> None:
        row = normalize_medqa_usmle_source_row(
            make_source_row(),
            row_index=0,
            verdict=make_verdict(),
            benchmark_version="v0.2",
        )
        self.assertTrue(row["id"].startswith("mamabench_v0.2_medqa_usmle_"))
        # content hash is 12 hex chars at the tail
        tail = row["id"].rsplit("_", 1)[-1]
        self.assertEqual(len(tail), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in tail))

    def test_id_stable_under_option_permutation(self) -> None:
        # Reordering options shouldn't change the content hash because the hash
        # sorts choices before hashing.
        row_a = normalize_medqa_usmle_source_row(
            make_source_row("A"),
            row_index=0,
            verdict=make_verdict(),
            benchmark_version="v0.2",
        )
        # Same data, but options dict insertion order shuffled
        shuffled = {
            "B": "Outpatient monitoring",
            "A": "Magnesium sulfate and delivery plan",
            "D": "Routine prenatal visit",
            "C": "Discharge home",
        }
        row_b = normalize_medqa_usmle_source_row(
            {**make_source_row("A"), "options": shuffled},
            row_index=0,
            verdict=make_verdict(),
            benchmark_version="v0.2",
        )
        self.assertEqual(row_a["id"], row_b["id"])

    def test_rejects_missing_answer(self) -> None:
        with self.assertRaisesRegex(MedQAUSMLEv2AdapterError, "not in options"):
            normalize_medqa_usmle_source_row(
                {**make_source_row(), "answer": "Z"},
                row_index=0,
                verdict=make_verdict(),
                benchmark_version="v0.2",
            )

    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(MedQAUSMLEv2AdapterError, "missing question"):
            normalize_medqa_usmle_source_row(
                {**make_source_row(), "question": "   "},
                row_index=0,
                verdict=make_verdict(),
                benchmark_version="v0.2",
            )


class LoadMedqaUsmleV2Tests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_filters_by_verdict_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_jsonl(
                tmp / "src.jsonl",
                [make_source_row("A"), make_source_row("B"), make_source_row("C")],
            )
            self._write_jsonl(
                tmp / "v.jsonl",
                [
                    make_verdict("MATERNAL", row_id="usmle_00000"),
                    make_verdict("NONE", row_id="usmle_00001"),
                    # usmle_00002 has no verdict
                ],
            )
            rows, stats = load_medqa_usmle_v2(
                tmp / "src.jsonl",
                tmp / "v.jsonl",
                benchmark_version="v0.2",
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(stats.total, 3)
            self.assertEqual(stats.kept, 1)
            self.assertEqual(stats.skipped_none, 1)
            self.assertEqual(stats.skipped_no_verdict, 1)


class BuildSourceMetadataTests(unittest.TestCase):
    def test_url_and_license(self) -> None:
        meta = build_medqa_usmle_v2_source_metadata()
        usmle = meta["MedQA-USMLE"]
        self.assertEqual(usmle["license"], "MIT")
        self.assertTrue(usmle["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
