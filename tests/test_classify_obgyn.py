from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    """Load scripts/classify_obgyn.py as a module for direct testing."""
    spec = importlib.util.spec_from_file_location(
        "classify_obgyn", ROOT / "scripts" / "classify_obgyn.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()


def fake_iter(rows: list[tuple[str, str]]):
    """Return a function that, when called with a path arg, yields the given rows."""

    def _iter(_path):
        for row in rows:
            yield row

    return _iter


class SelectRowsTests(unittest.TestCase):
    def test_no_filters_returns_all_rows(self) -> None:
        rows = [("r0", "t0"), ("r1", "t1"), ("r2", "t2")]
        selected, n_shard, n_resume = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done=set(),
            shard=None,
            limit=None,
        )
        self.assertEqual(selected, rows)
        self.assertEqual(n_shard, 0)
        self.assertEqual(n_resume, 0)

    def test_already_done_rows_are_skipped(self) -> None:
        rows = [("r0", "t0"), ("r1", "t1"), ("r2", "t2"), ("r3", "t3")]
        selected, _, n_resume = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done={"r1", "r3"},
            shard=None,
            limit=None,
        )
        self.assertEqual(selected, [("r0", "t0"), ("r2", "t2")])
        self.assertEqual(n_resume, 2)

    def test_shard_filters_by_source_index_modulo(self) -> None:
        rows = [(f"r{i}", f"t{i}") for i in range(10)]
        # shard 0 of 3 → keep i in {0, 3, 6, 9}
        selected_0, n_shard_0, _ = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done=set(),
            shard=(0, 3),
            limit=None,
        )
        self.assertEqual([r[0] for r in selected_0], ["r0", "r3", "r6", "r9"])
        self.assertEqual(n_shard_0, 6)

        # shard 1 of 3 → keep i in {1, 4, 7}
        selected_1, _, _ = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done=set(),
            shard=(1, 3),
            limit=None,
        )
        self.assertEqual([r[0] for r in selected_1], ["r1", "r4", "r7"])

        # union of all shards must equal full set
        selected_2, _, _ = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done=set(),
            shard=(2, 3),
            limit=None,
        )
        union = set(r[0] for r in selected_0) | set(r[0] for r in selected_1) | set(r[0] for r in selected_2)
        self.assertEqual(union, {f"r{i}" for i in range(10)})

    def test_shard_runs_independently_per_resume_state(self) -> None:
        rows = [(f"r{i}", f"t{i}") for i in range(6)]
        # shard 0/2 means even-indexed source rows; already-done filters AFTER shard
        selected, n_shard, n_resume = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done={"r0"},
            shard=(0, 2),
            limit=None,
        )
        # In-shard rows are r0, r2, r4. Resume drops r0 → r2, r4 remain.
        self.assertEqual([r[0] for r in selected], ["r2", "r4"])
        self.assertEqual(n_shard, 3)  # r1, r3, r5 dropped by shard filter
        self.assertEqual(n_resume, 1)

    def test_limit_caps_the_collected_rows(self) -> None:
        rows = [(f"r{i}", f"t{i}") for i in range(100)]
        selected, _, _ = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done=set(),
            shard=None,
            limit=5,
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual([r[0] for r in selected], ["r0", "r1", "r2", "r3", "r4"])

    def test_limit_counts_only_kept_rows_not_skipped_ones(self) -> None:
        rows = [(f"r{i}", f"t{i}") for i in range(20)]
        # shard 0/2 keeps even indices: r0, r2, r4, r6, ..., r18 (10 rows).
        # Limit of 3 should give r0, r2, r4 — not consume the limit budget on
        # skipped odd-indexed rows.
        selected, _, _ = SCRIPT.select_rows(
            fake_iter(rows),
            Path("dummy"),
            already_done=set(),
            shard=(0, 2),
            limit=3,
        )
        self.assertEqual([r[0] for r in selected], ["r0", "r2", "r4"])


class LoadExistingRowIdsTests(unittest.TestCase):
    def test_returns_empty_set_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                SCRIPT.load_existing_row_ids(Path(tmpdir) / "nope.jsonl"),
                set(),
            )

    def test_reads_row_ids_and_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v.jsonl"
            path.write_text(
                json.dumps({"row_id": "a", "category": "MATERNAL"}) + "\n"
                "this is not json\n"
                "\n"
                + json.dumps({"row_id": "b", "category": "NONE"}) + "\n",
                encoding="utf-8",
            )
            ids = SCRIPT.load_existing_row_ids(path)
            self.assertEqual(ids, {"a", "b"})


class CliWorkersIntegrationTests(unittest.TestCase):
    """End-to-end smoke test: stub out the OpenAI completer and run main."""

    def test_workers_processes_all_rows_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hb.jsonl"
            output_path = Path(tmpdir) / "verdicts.jsonl"
            input_rows = [
                {"prompt_id": f"p{i}", "prompt": [{"role": "user", "content": f"q{i}"}]}
                for i in range(7)
            ]
            input_path.write_text(
                "\n".join(json.dumps(r) for r in input_rows) + "\n",
                encoding="utf-8",
            )

            call_lock = threading.Lock()
            calls = {"n": 0}

            def fake_completer(_messages):
                with call_lock:
                    calls["n"] += 1
                return (
                    json.dumps({"category": "MATERNAL", "rationale": "stub verdict"}),
                    "stub reasoning",
                )

            # Patch make_openai_completer to return our stub
            original_factory = SCRIPT.make_openai_completer_with_reasoning
            SCRIPT.make_openai_completer_with_reasoning = lambda **kwargs: fake_completer
            try:
                rc = SCRIPT.main(
                    [
                        "--source", "healthbench",
                        "--input", str(input_path),
                        "--output", str(output_path),
                        "--model", "stub-model",
                        "--no-guided-json",
                        "--workers", "4",
                    ]
                )
            finally:
                SCRIPT.make_openai_completer_with_reasoning = original_factory

            self.assertEqual(rc, 0)
            self.assertEqual(calls["n"], 7)

            written = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(written), 7)
            ids = {json.loads(line)["row_id"] for line in written}
            self.assertEqual(ids, {f"p{i}" for i in range(7)})
            for line in written:
                rec = json.loads(line)
                self.assertEqual(rec["category"], "MATERNAL")
                self.assertEqual(rec["source"], "healthbench")
                self.assertEqual(rec["model"], "stub-model")
                self.assertEqual(rec["mode"], "openended")

    def test_shard_subdivision_only_processes_assigned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hb.jsonl"
            output_path = Path(tmpdir) / "verdicts.jsonl"
            input_rows = [
                {"prompt_id": f"p{i}", "prompt": [{"role": "user", "content": f"q{i}"}]}
                for i in range(10)
            ]
            input_path.write_text(
                "\n".join(json.dumps(r) for r in input_rows) + "\n",
                encoding="utf-8",
            )

            def fake_completer(_messages):
                return (
                    json.dumps({"category": "NONE", "rationale": "x"}),
                    "stub reasoning",
                )

            original_factory = SCRIPT.make_openai_completer_with_reasoning
            SCRIPT.make_openai_completer_with_reasoning = lambda **kwargs: fake_completer
            try:
                rc = SCRIPT.main(
                    [
                        "--source", "healthbench",
                        "--input", str(input_path),
                        "--output", str(output_path),
                        "--model", "stub-model",
                        "--no-guided-json",
                        "--workers", "2",
                        "--shard", "1", "3",
                    ]
                )
            finally:
                SCRIPT.make_openai_completer_with_reasoning = original_factory

            self.assertEqual(rc, 0)
            written = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
            # shard 1 of 3 → source indices {1, 4, 7} → prompt_ids p1, p4, p7
            self.assertEqual(
                sorted(r["row_id"] for r in written),
                ["p1", "p4", "p7"],
            )

    def test_resume_skips_rows_already_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hb.jsonl"
            output_path = Path(tmpdir) / "verdicts.jsonl"
            input_rows = [
                {"prompt_id": f"p{i}", "prompt": [{"role": "user", "content": f"q{i}"}]}
                for i in range(5)
            ]
            input_path.write_text(
                "\n".join(json.dumps(r) for r in input_rows) + "\n",
                encoding="utf-8",
            )
            # Pre-populate output with p1 and p3 already classified
            output_path.write_text(
                json.dumps({"row_id": "p1", "category": "MATERNAL", "rationale": "pre"}) + "\n"
                + json.dumps({"row_id": "p3", "category": "NEONATAL", "rationale": "pre"}) + "\n",
                encoding="utf-8",
            )

            def fake_completer(_messages):
                return (
                    json.dumps({"category": "NONE", "rationale": "new"}),
                    "stub reasoning",
                )

            original_factory = SCRIPT.make_openai_completer_with_reasoning
            SCRIPT.make_openai_completer_with_reasoning = lambda **kwargs: fake_completer
            try:
                SCRIPT.main(
                    [
                        "--source", "healthbench",
                        "--input", str(input_path),
                        "--output", str(output_path),
                        "--model", "stub-model",
                        "--no-guided-json",
                    ]
                )
            finally:
                SCRIPT.make_openai_completer_with_reasoning = original_factory

            written = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
            # original 2 + 3 newly classified = 5
            self.assertEqual(len(written), 5)
            # the pre-existing verdicts should still be there unchanged
            preexisting = [r for r in written if r["rationale"] == "pre"]
            self.assertEqual(len(preexisting), 2)
            self.assertEqual(sorted(r["row_id"] for r in preexisting), ["p1", "p3"])


if __name__ == "__main__":
    unittest.main()
