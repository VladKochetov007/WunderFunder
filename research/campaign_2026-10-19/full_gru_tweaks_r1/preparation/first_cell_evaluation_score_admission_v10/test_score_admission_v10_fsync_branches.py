from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V9_DIR = ROOT / "preparation/first_cell_evaluation_score_admission_v9"
V10_DIR = Path(__file__).resolve().parent
BODY_PATH = V10_DIR / "REMOTE_SUBMISSION_V10.py"
BINDING_PATH = V10_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V10.json"
BLOCK4 = (
    "    directory_fd = os.open(\n"
    '        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)\n'
    "    )\n"
    "    try:\n"
    "        os.fsync(directory_fd)\n"
    "    finally:\n"
    "        os.close(directory_fd)\n"
)
BLOCK8 = "".join(
    "    " + line if line.strip() else line for line in BLOCK4.splitlines(keepends=True)
)


class FsyncBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BODY_PATH.read_text()
        cls.binding = json.loads(BINDING_PATH.read_text())
        cls.tree = ast.parse(cls.source)

    def test_success_and_unknown_receipt_branches_fsync_before_return_or_raise(self) -> None:
        self.assertEqual(self.source.count(BLOCK4), 2)
        self.assertEqual(self.source.count(BLOCK8), 1)
        unresolved = (
            '        write_json_exclusive(\n            SCORE_D / "LAUNCH_UNRESOLVED.json",\n'
        )
        unresolved_at = self.source.index(unresolved)
        unknown_raise_at = self.source.index(
            "        raise RuntimeError(\n"
            '            "score invocation ID unresolved; durable unknown outcome retained, never retry"',
            unresolved_at,
        )
        self.assertLess(self.source.index(BLOCK8, unresolved_at), unknown_raise_at)

        launch_write = '    write_json_exclusive(SCORE_D / "LAUNCH.json", launch)\n'
        launch_at = self.source.index(launch_write)
        outcome_check_at = self.source.index(
            '    if submission.get("outcome") != "submitted"', launch_at
        )
        self.assertEqual(self.source[launch_at + len(launch_write) :].startswith(BLOCK4), True)
        self.assertLess(self.source.index(BLOCK4, launch_at), outcome_check_at)

        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        submit_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "submit_once"
        )
        self.assertTrue(
            isinstance(main.body[submit_index + 1], ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "admission_raw"
                for target in ast.walk(main.body[submit_index + 1])
            )
        )

    def test_initial_pre_submit_fsync_and_six_payload_pins_remain(self) -> None:
        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        submit_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "submit_once"
        )
        self.assertTrue(any(isinstance(node, ast.Try) for node in main.body[:submit_index]))
        v9_binding = json.loads((V9_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V9.json").read_text())
        self.assertEqual(v9_binding["scorer_payloads"], self.binding["scorer_payloads"])
        self.assertEqual(len(self.binding["scorer_payloads"]), 6)
        self.assertEqual(v9_binding["score_unit"], self.binding["score_unit"])
        self.assertEqual(v9_binding["resource_limits"], self.binding["resource_limits"])


if __name__ == "__main__":
    unittest.main()
