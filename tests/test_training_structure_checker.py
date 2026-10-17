from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_training_structure import discover_sources, inspect_module


class TrainingStructureCheckerTests(unittest.TestCase):
    def test_reports_missing_signature_docs_and_hidden_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.py"
            path.write_text(
                "def update(value):\n    global CACHE\n    return value\n",
                encoding="utf-8",
            )

            errors = inspect_module(path, max_lines=20, max_function_lines=20)

        self.assertTrue(any("parameter annotations missing for value" in error for error in errors))
        self.assertTrue(any("return annotation missing" in error for error in errors))
        self.assertTrue(any("document the public function 'update'" in error for error in errors))
        self.assertTrue(any("pass state explicitly" in error for error in errors))

    def test_reports_module_and_function_size_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "large.py"
            path.write_text(
                "def calculate() -> int:\n"
                "    first = 1\n"
                "    second = 2\n"
                "    return first + second\n",
                encoding="utf-8",
            )

            errors = inspect_module(path, max_lines=3, max_function_lines=3)

        self.assertTrue(any("module exceeds 3 lines" in error for error in errors))
        self.assertTrue(any("function 'calculate' exceeds 3 lines" in error for error in errors))

    def test_reports_syntax_errors_with_source_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.py"
            path.write_text("def broken(:\n", encoding="utf-8")

            errors = inspect_module(path, max_lines=20, max_function_lines=20)

        self.assertEqual(len(errors), 1)
        self.assertIn(f"{path}:1:", errors[0])
        self.assertIn("invalid Python syntax", errors[0])

    def test_discovery_fails_when_training_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths, errors = discover_sources(Path(temporary_directory))

        self.assertEqual(paths, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("required training source directory is missing", errors[0])

    def test_cli_fails_clearly_when_training_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checker = Path(__file__).resolve().parents[1] / "scripts/check_training_structure.py"
            result = subprocess.run(
                [sys.executable, str(checker), "--root", temporary_directory],
                capture_output=True,
                check=False,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("required training source directory is missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_discovery_includes_trainer_entrypoint_and_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_dir = root / "src/wnn/training"
            scripts_dir = root / "scripts"
            training_dir.mkdir(parents=True)
            scripts_dir.mkdir()
            (training_dir / "__init__.py").write_text("\n", encoding="utf-8")
            (training_dir / "runner.py").write_text("\n", encoding="utf-8")
            (scripts_dir / "train_rnn.py").write_text("\n", encoding="utf-8")
            (scripts_dir / "check_training_structure.py").write_text("\n", encoding="utf-8")

            paths, errors = discover_sources(root)

        self.assertEqual(errors, [])
        self.assertEqual(
            {path.relative_to(root.resolve()) for path in paths},
            {
                Path("src/wnn/training/__init__.py"),
                Path("src/wnn/training/runner.py"),
                Path("scripts/train_rnn.py"),
                Path("scripts/check_training_structure.py"),
            },
        )


if __name__ == "__main__":
    unittest.main()
