from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_kenya_parity", ROOT / "scripts" / "compare_kenya_parity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()


class KenyaToMamabenchCategoryTests(unittest.TestCase):
    def test_srh_maps_to_full_spelling(self) -> None:
        self.assertEqual(
            SCRIPT.kenya_to_mamabench_category("SRH"),
            "SEXUAL_AND_REPRODUCTIVE_HEALTH",
        )

    def test_other_categories_passthrough(self) -> None:
        for c in ("MATERNAL", "NEONATAL", "CHILD_HEALTH", "NONE"):
            with self.subTest(c=c):
                self.assertEqual(SCRIPT.kenya_to_mamabench_category(c), c)


class LoadersTests(unittest.TestCase):
    def test_verdicts_skip_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v.jsonl"
            path.write_text(
                json.dumps({"row_id": "7", "category": "MATERNAL", "rationale": "x"}) + "\n"
                + json.dumps({"row_id": "8", "category": "OBGYN", "rationale": "bad cat"}) + "\n"
                + "this is not json\n"
                + json.dumps({"category": "NEONATAL", "rationale": "no row_id"}) + "\n"
                + json.dumps({"row_id": "9", "category": "NONE", "rationale": "ok"}) + "\n",
                encoding="utf-8",
            )
            verdicts = SCRIPT.load_verdicts(path)
            self.assertEqual(verdicts, {"7": "MATERNAL", "9": "NONE"})

    def test_kenya_labels_normalises_srh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "k.tsv"
            path.write_text(
                "study_id\tscenario\tclinician_response\tcategory\n"
                "1\ts1\tr1\tMATERNAL\n"
                "2\ts2\tr2\tSRH\n"
                "3\ts3\tr3\tCHILD_HEALTH\n",
                encoding="utf-8",
            )
            labels = SCRIPT.load_kenya_labels(path)
            self.assertEqual(
                labels,
                {
                    "1": "MATERNAL",
                    "2": "SEXUAL_AND_REPRODUCTIVE_HEALTH",
                    "3": "CHILD_HEALTH",
                },
            )


class ComputeMetricsTests(unittest.TestCase):
    def test_perfect_agreement_yields_100_percent(self) -> None:
        verdicts = {"1": "MATERNAL", "2": "NEONATAL", "3": "NONE"}
        kenya = {"1": "MATERNAL", "2": "NEONATAL"}  # "3" absent → Kenya said NONE
        m = SCRIPT.compute_metrics(verdicts, kenya)
        self.assertEqual(m["n_total"], 3)
        self.assertEqual(m["n_agree"], 3)
        self.assertEqual(m["agreement"], 1.0)
        self.assertEqual(m["disagreements"], [])

    def test_missing_kenya_label_treated_as_none(self) -> None:
        verdicts = {"99": "MATERNAL"}
        kenya: dict[str, str] = {}
        m = SCRIPT.compute_metrics(verdicts, kenya)
        # We said MATERNAL, Kenya implicitly said NONE → disagree
        self.assertEqual(m["n_agree"], 0)
        self.assertEqual(m["disagreements"], [("99", "NONE", "MATERNAL")])

    def test_per_category_precision_and_recall(self) -> None:
        # 2 MATERNAL TPs, 1 MATERNAL FP (we said MATERNAL, Kenya said NONE),
        # 1 MATERNAL FN (Kenya said MATERNAL, we said NEONATAL).
        verdicts = {
            "a": "MATERNAL",
            "b": "MATERNAL",
            "c": "MATERNAL",  # FP — Kenya said NONE
            "d": "NEONATAL",  # FN — Kenya said MATERNAL
        }
        kenya = {
            "a": "MATERNAL",
            "b": "MATERNAL",
            "d": "MATERNAL",
            # "c" missing → Kenya NONE
        }
        m = SCRIPT.compute_metrics(verdicts, kenya)
        mat = m["per_category"]["MATERNAL"]
        self.assertEqual(mat["tp"], 2)
        self.assertEqual(mat["fp"], 1)
        self.assertEqual(mat["fn"], 1)
        self.assertAlmostEqual(mat["precision"], 2 / 3, places=3)
        self.assertAlmostEqual(mat["recall"], 2 / 3, places=3)

    def test_srh_disagreement_after_normalisation(self) -> None:
        # Kenya TSV stores "SRH"; verdicts use the full spelling. Loader has
        # already normalised Kenya side, so equal categories should agree.
        verdicts = {"1": "SEXUAL_AND_REPRODUCTIVE_HEALTH"}
        kenya = {"1": "SEXUAL_AND_REPRODUCTIVE_HEALTH"}  # post-normalisation
        m = SCRIPT.compute_metrics(verdicts, kenya)
        self.assertEqual(m["n_agree"], 1)

    def test_confusion_matrix_indexed_by_kenya_then_ours(self) -> None:
        verdicts = {"a": "NEONATAL"}
        kenya = {"a": "MATERNAL"}
        m = SCRIPT.compute_metrics(verdicts, kenya)
        self.assertEqual(m["confusion_matrix"]["MATERNAL"]["NEONATAL"], 1)
        self.assertEqual(m["confusion_matrix"]["MATERNAL"].get("MATERNAL", 0), 0)


if __name__ == "__main__":
    unittest.main()
