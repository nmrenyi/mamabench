from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mamabench.prompts import (
    PROMPT_VERSION,
    load_classifier_prompt,
)


class LoadClassifierPromptTests(unittest.TestCase):
    def test_prompt_version_is_pinned(self) -> None:
        self.assertEqual(PROMPT_VERSION, "v6")

    def test_openended_assembles_in_expected_order(self) -> None:
        prompt = load_classifier_prompt("openended")
        self.assertIn("OBGYN Specialty Classifier — Open-ended", prompt)
        self.assertIn("## Categories", prompt)
        self.assertIn("## Decision rule", prompt)
        self.assertIn("### Multi-turn conversations", prompt)
        self.assertIn("### Clinical vignettes with a narrator self-description", prompt)
        self.assertIn("## Input format", prompt)
        self.assertIn("## Output format", prompt)
        # mcq-specific guidance must not leak into openended
        self.assertNotIn("Reading the answer options", prompt)
        self.assertNotIn("Options:", prompt[: prompt.find("Output format")])
        # order: intro before categories before decision rule before guidance
        self.assertLess(prompt.find("OBGYN Specialty Classifier"), prompt.find("## Categories"))
        self.assertLess(prompt.find("## Categories"), prompt.find("## Decision rule"))
        self.assertLess(prompt.find("## Decision rule"), prompt.find("## Additional decision guidance"))
        self.assertLess(prompt.find("## Additional decision guidance"), prompt.find("## Input format"))
        self.assertLess(prompt.find("## Input format"), prompt.find("## Output format"))

    def test_mcq_assembles_with_mcq_specific_sections(self) -> None:
        prompt = load_classifier_prompt("mcq")
        self.assertIn("OBGYN Specialty Classifier — Multiple-Choice", prompt)
        self.assertIn("### Reading the answer options", prompt)
        # openended-only guidance must not leak into mcq
        self.assertNotIn("### Multi-turn conversations", prompt)
        self.assertNotIn("### Clinical vignettes with a narrator self-description", prompt)

    def test_both_modes_share_categories_and_output_format(self) -> None:
        openended = load_classifier_prompt("openended")
        mcq = load_classifier_prompt("mcq")
        for shared in (
            "**MATERNAL** — pregnancy, labor",
            "**NEONATAL** — care of newborns",
            "**CHILD_HEALTH**",
            "**SEXUAL_AND_REPRODUCTIVE_HEALTH**",
            "Return a single JSON object",
        ):
            with self.subTest(shared=shared):
                self.assertIn(shared, openended)
                self.assertIn(shared, mcq)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown classifier mode"):
            load_classifier_prompt("free_response")  # type: ignore[arg-type]

    def test_missing_module_raises_with_clear_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # only write a subset
            (tmp / "intro_openended.md").write_text("# Open\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "categories.md"):
                load_classifier_prompt("openended", modules_dir=tmp)


if __name__ == "__main__":
    unittest.main()
