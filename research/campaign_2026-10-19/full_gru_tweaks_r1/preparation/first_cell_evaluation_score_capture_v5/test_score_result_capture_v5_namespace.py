"""Focused AST-only checks for the score-v2 capture namespace rebasing."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
V4 = BASE / "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
V5 = HERE / "SCORE_RESULT_CAPTURE_TEMPLATE_V5.py"
REBASED_ASSIGNMENTS = {"SCORE_D", "SCORE_UNIT", "SCORER_CODE_PINS"}
CORE_PIN = (
    26_965,
    "961f566eaeec0cd7e9ab48788666d03da9b0482810a11f16a019b083cfc65005",
)
UNCHANGED_CODE_PINS = {
    "first_cell_score_contract.py": (
        20_165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}


def _source_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignments(tree: ast.Module) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = node.value
    return result


def _without_rebased_assignments(tree: ast.Module) -> ast.Module:
    retained = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in REBASED_ASSIGNMENTS
        ):
            continue
        retained.append(node)
    tree.body = retained
    return tree


class ScoreResultCaptureV5NamespaceTests(unittest.TestCase):
    def test_ast_diff_is_exactly_the_three_namespace_and_core_assignments(self) -> None:
        before = _source_tree(V4)
        after = _source_tree(V5)
        before_assignments = _assignments(before)
        after_assignments = _assignments(after)
        changed = {
            name
            for name in before_assignments.keys() | after_assignments.keys()
            if name not in before_assignments
            or name not in after_assignments
            or ast.dump(before_assignments[name], include_attributes=False)
            != ast.dump(after_assignments[name], include_attributes=False)
        }
        self.assertEqual(changed, REBASED_ASSIGNMENTS)
        self.assertEqual(
            ast.dump(_without_rebased_assignments(before), include_attributes=False),
            ast.dump(_without_rebased_assignments(after), include_attributes=False),
        )

    def test_namespace_and_frozen_core_pin_match_the_score_v2_admission(self) -> None:
        assignments = _assignments(_source_tree(V5))
        score_d = assignments["SCORE_D"]
        self.assertEqual(
            ast.dump(score_d, include_attributes=False),
            ast.dump(
                ast.BinOp(
                    left=ast.Name(id="SCRATCH_ROOT", ctx=ast.Load()),
                    op=ast.Div(),
                    right=ast.Constant(value="fg19_first_cell_f0_s1001_score_v2"),
                ),
                include_attributes=False,
            ),
        )
        self.assertEqual(
            ast.literal_eval(assignments["SCORE_UNIT"]),
            "computer-vision-homework-fg19-first-cell-score-v2.service",
        )
        pins = ast.literal_eval(assignments["SCORER_CODE_PINS"])
        self.assertEqual(pins["first_cell_score_core.py"], CORE_PIN)
        self.assertEqual({name: pins[name] for name in UNCHANGED_CODE_PINS}, UNCHANGED_CODE_PINS)
        self.assertEqual(
            pins.keys(),
            {
                "first_cell_score_contract.py",
                "first_cell_score_core.py",
                "first_cell_score_predictions.py",
                "score_first_cell.py",
            },
        )

    def test_actual_admission_literals_remain_unbound(self) -> None:
        assignments = _assignments(_source_tree(V5))
        self.assertIsNone(ast.literal_eval(assignments["SCORE_ADMISSION_BYTES"]))
        self.assertEqual(
            ast.literal_eval(assignments["SCORE_ADMISSION_SHA256"]),
            "UNBOUND_ROOT_AFTER_ACTUAL_SCORE_ADMISSION",
        )
        self.assertIsNone(ast.literal_eval(assignments["SCORE_LAUNCH_BYTES"]))
        self.assertEqual(
            ast.literal_eval(assignments["SCORE_LAUNCH_SHA256"]),
            "UNBOUND_ROOT_AFTER_ACTUAL_SCORE_ADMISSION",
        )
        self.assertEqual(
            ast.literal_eval(assignments["SCORE_INVOCATION_ID"]),
            "UNBOUND_ROOT_AFTER_ACTUAL_SCORE_ADMISSION",
        )


if __name__ == "__main__":
    unittest.main()
