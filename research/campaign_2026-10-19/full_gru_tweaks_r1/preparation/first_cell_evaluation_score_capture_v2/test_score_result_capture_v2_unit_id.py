"""Focused synthetic boundary checks for exact systemd unit identity."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "SCORE_RESULT_CAPTURE_TEMPLATE_V2.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("score_result_capture_v2", TEMPLATE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load score result capture V2")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScoreResultCaptureV2UnitIdentityTests(unittest.TestCase):
    def test_exact_unit_id_is_accepted_and_wrong_or_missing_id_rejected(self) -> None:
        MODULE._require_exact_unit_identity({"Id": MODULE.SCORE_UNIT})
        for properties in ({"Id": "other.service"}, {}, {"Id": ""}):
            with (
                self.subTest(properties=properties),
                self.assertRaisesRegex(ValueError, "different score unit Id"),
            ):
                MODULE._require_exact_unit_identity(properties)

    def test_systemd_requests_id_and_checks_before_result_capture(self) -> None:
        self.assertIn("Id", MODULE.UNIT_PROPERTIES)
        tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        unit_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "unit" for target in node.targets)
        )
        self.assertIsInstance(main.body[unit_index + 1], ast.Expr)
        guard_call = main.body[unit_index + 1].value
        self.assertIsInstance(guard_call, ast.Call)
        self.assertIsInstance(guard_call.func, ast.Name)
        self.assertEqual(guard_call.func.id, "_require_exact_unit_identity")
        result_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "result_capture"
                for target in node.targets
            )
        )
        self.assertLess(unit_index + 1, result_index)

    def test_result_root_tracks_export_v3(self) -> None:
        self.assertEqual(
            str(MODULE.RESULT_PATH),
            "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v2/RESULT.json",
        )
        self.assertEqual(
            MODULE.SCORE_UNIT, "computer-vision-homework-fg19-first-cell-score-v1.service"
        )


if __name__ == "__main__":
    unittest.main()
