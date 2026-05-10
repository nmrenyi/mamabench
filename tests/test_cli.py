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
    def test_adapt_help_does_not_load_project_config(self) -> None:
        module = load_script("adapt_medmcqa.py")
        module.load_project_config = self.fail_if_config_loads

        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stdout(
            io.StringIO()
        ):
            module.main(["--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_summarize_help_does_not_load_project_config(self) -> None:
        module = load_script("summarize_mamabench.py")
        module.load_project_config = self.fail_if_config_loads

        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stdout(
            io.StringIO()
        ):
            module.main(["--help"])

        self.assertEqual(exc.exception.code, 0)

    def fail_if_config_loads(self, _path: object) -> object:
        self.fail("help should be handled before loading mamabench.json")


if __name__ == "__main__":
    unittest.main()
