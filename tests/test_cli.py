from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_script(script_name: str) -> ModuleType:
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliHelpTests(unittest.TestCase):
    def test_adapt_medmcqa_help_does_not_load_project_config(self) -> None:
        self._assert_help_skips_config("adapt_medmcqa.py")

    def test_adapt_medqa_usmle_help_does_not_load_project_config(self) -> None:
        self._assert_help_skips_config("adapt_medqa_usmle.py")

    def test_adapt_afrimedqa_help_does_not_load_project_config(self) -> None:
        self._assert_help_skips_config("adapt_afrimedqa.py")

    def test_summarize_help_does_not_load_project_config(self) -> None:
        self._assert_help_skips_config("summarize_mamabench.py")

    def test_classify_obgyn_help_does_not_init_llm(self) -> None:
        module = load_script("classify_obgyn.py")
        module.make_openai_completer = self.fail_if_llm_inits

        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stdout(
            io.StringIO()
        ):
            module.main(["--help"])

        self.assertEqual(exc.exception.code, 0)

    def _assert_help_skips_config(self, script_name: str) -> None:
        module = load_script(script_name)
        module.load_project_config = self.fail_if_config_loads

        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stdout(
            io.StringIO()
        ):
            module.main(["--help"])

        self.assertEqual(exc.exception.code, 0)

    def fail_if_config_loads(self, _path: object) -> object:
        self.fail("help should be handled before loading mamabench.json")

    def fail_if_llm_inits(self, *_args: object, **_kwargs: object) -> object:
        self.fail("help should be handled before initializing the LLM client")


if __name__ == "__main__":
    unittest.main()
