from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mamabench.obgyn_sources import (
    iter_healthbench,
    iter_medqa_usmle,
    render_healthbench_messages,
    render_medqa_options,
)


class RenderHealthBenchMessagesTests(unittest.TestCase):
    def test_single_turn_user(self) -> None:
        rendered = render_healthbench_messages(
            [{"role": "user", "content": "what's a healthy postpartum diet?"}]
        )
        self.assertEqual(rendered, "User: what's a healthy postpartum diet?")

    def test_multi_turn_uses_role_prefixes_and_preserves_order(self) -> None:
        rendered = render_healthbench_messages(
            [
                {"role": "user", "content": "what foods help with postpartum recovery?"},
                {"role": "assistant", "content": "iron, protein, fluids"},
                {"role": "user", "content": "what about vitamin D specifically?"},
            ]
        )
        self.assertEqual(
            rendered,
            "User: what foods help with postpartum recovery?\n"
            "Assistant: iron, protein, fluids\n"
            "User: what about vitamin D specifically?",
        )


class RenderMedqaOptionsTests(unittest.TestCase):
    def test_sorts_options_alphabetically_and_uses_pipe_separator(self) -> None:
        rendered = render_medqa_options(
            "What is the most appropriate next step?",
            {"D": "Pin sleeve to the shirt", "A": "Nerve conduction", "B": "Surgical fixation", "C": "PT"},
        )
        self.assertEqual(
            rendered,
            "What is the most appropriate next step?\n"
            "Options: A. Nerve conduction | B. Surgical fixation | C. PT | D. Pin sleeve to the shirt",
        )

    def test_handles_empty_options_dict(self) -> None:
        rendered = render_medqa_options("Some question.", {})
        self.assertEqual(rendered, "Some question.\nOptions: ")


class IterHealthBenchTests(unittest.TestCase):
    def test_yields_prompt_id_and_rendered_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hb.jsonl"
            rows = [
                {
                    "prompt_id": "abc-1",
                    "prompt": [{"role": "user", "content": "first question"}],
                },
                {
                    "prompt_id": "abc-2",
                    "prompt": [
                        {"role": "user", "content": "turn 1"},
                        {"role": "assistant", "content": "reply 1"},
                        {"role": "user", "content": "turn 2"},
                    ],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

            out = list(iter_healthbench(path))

            self.assertEqual(out[0], ("abc-1", "User: first question"))
            self.assertEqual(
                out[1],
                (
                    "abc-2",
                    "User: turn 1\nAssistant: reply 1\nUser: turn 2",
                ),
            )

    def test_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hb.jsonl"
            row = {"prompt_id": "x", "prompt": [{"role": "user", "content": "q"}]}
            path.write_text(
                "\n\n" + json.dumps(row) + "\n\n", encoding="utf-8"
            )
            out = list(iter_healthbench(path))
            self.assertEqual(out, [("x", "User: q")])


class IterMedqaUsmleTests(unittest.TestCase):
    def test_synthesises_zero_padded_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "usmle.jsonl"
            rows = [
                {
                    "question": "A pregnant patient with mild HTN. Next step?",
                    "options": {"A": "Methyldopa", "B": "ACE inhibitor", "C": "Observation", "D": "Delivery"},
                    "answer": "A",
                },
                {
                    "question": "A 65-year-old with chest pain after exertion. Most likely diagnosis?",
                    "options": {"A": "GERD", "B": "MI", "C": "Anxiety"},
                    "answer": "B",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

            out = list(iter_medqa_usmle(path))

            self.assertEqual(len(out), 2)
            self.assertEqual(out[0][0], "usmle_00000")
            self.assertEqual(out[1][0], "usmle_00001")
            self.assertIn("Options: A. Methyldopa | B. ACE inhibitor", out[0][1])

    def test_handles_row_with_no_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "usmle.jsonl"
            path.write_text(
                json.dumps({"question": "Q only.", "answer": "A"}) + "\n",
                encoding="utf-8",
            )
            out = list(iter_medqa_usmle(path))
            self.assertEqual(out, [("usmle_00000", "Q only.\nOptions: ")])


class IterKenyaTests(unittest.TestCase):
    def test_iter_kenya_returns_study_id_and_user_prompt(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")
        from mamabench.obgyn_sources import iter_kenya

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.append(["StudyID", "User Prompt", "Clinician response"])
            ws.append([7, "A 15-year-old male with stool incontinence.", "Refer to surgery"])
            ws.append([8, "A 28-year-old G2P1 with preeclampsia signs.", "Magnesium sulfate"])
            ws.append([None, None, None])  # empty trailing row should be skipped
            wb.save(str(path))

            out = list(iter_kenya(path))

            self.assertEqual(len(out), 2)
            self.assertEqual(out[0], ("7", "A 15-year-old male with stool incontinence."))
            self.assertEqual(out[1], ("8", "A 28-year-old G2P1 with preeclampsia signs."))

    def test_iter_kenya_raises_clearly_on_missing_column(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")
        from mamabench.obgyn_sources import iter_kenya

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wrong.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.append(["id", "text"])
            ws.append([1, "something"])
            wb.save(str(path))

            with self.assertRaisesRegex(ValueError, "StudyID"):
                list(iter_kenya(path))


if __name__ == "__main__":
    unittest.main()
