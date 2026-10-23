"""Focused source-only checks for rebinding the ROOT status driver to reader V5."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
V2_DRIVER = PACKET / "ROOT_CAPTURE_SCORE_STATUS_V2.py"
V3_DRIVER = PACKET / "ROOT_CAPTURE_SCORE_STATUS_V3.py"
V5_TEMPLATE = (
    PREPARATION / "first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py"
)
V5_REVIEW = PREPARATION / "first_cell_evaluation_score_capture_v5/INDEPENDENT_SOURCE_REVIEW_V5.json"
EXPECTED_TEMPLATE_SHA256 = "8d3c37f89376a23db1a548001989c11dd7eb5b83720061c4b229e59f75d1f69c"
EXPECTED_REVIEW_SHA256 = "6be421f63040c941368af2c08670b22ce1699434a8b264c4f100dd253e229b95"
REBASABLE = {"TEMPLATE", "TEMPLATE_SHA", "REVIEW", "REVIEW_SHA"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _normalize_driver(path: Path) -> ast.Module:
    tree = _tree(path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in REBASABLE:
                node.value = ast.Constant(value=f"<{target.id}>")
    return tree


def _assignments(path: Path) -> dict[str, ast.expr]:
    return {
        node.targets[0].id: node.value
        for node in _tree(path).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


class RootScoreStatusDriverV3Tests(unittest.TestCase):
    def test_driver_ast_diff_is_exactly_four_reader_review_bindings(self) -> None:
        self.assertEqual(
            ast.dump(_normalize_driver(V2_DRIVER), include_attributes=False),
            ast.dump(_normalize_driver(V3_DRIVER), include_attributes=False),
        )

    def test_driver_selects_exact_independently_reviewed_reader_v5(self) -> None:
        values = _assignments(V3_DRIVER)
        template = values["TEMPLATE"]
        self.assertIsInstance(template, ast.BinOp)
        self.assertIsInstance(template.left, ast.Name)
        self.assertEqual(template.left.id, "C")
        self.assertIsInstance(template.right, ast.Constant)
        self.assertEqual(
            template.right.value,
            "preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py",
        )
        review_path = values["REVIEW"]
        self.assertIsInstance(review_path, ast.BinOp)
        self.assertIsInstance(review_path.left, ast.Attribute)
        self.assertIsInstance(review_path.left.value, ast.Name)
        self.assertEqual(review_path.left.value.id, "TEMPLATE")
        self.assertEqual(review_path.left.attr, "parent")
        self.assertIsInstance(review_path.right, ast.Constant)
        self.assertEqual(review_path.right.value, "INDEPENDENT_SOURCE_REVIEW_V5.json")
        self.assertEqual(ast.literal_eval(values["TEMPLATE_SHA"]), EXPECTED_TEMPLATE_SHA256)
        self.assertEqual(ast.literal_eval(values["REVIEW_SHA"]), EXPECTED_REVIEW_SHA256)
        review = json.loads(V5_REVIEW.read_text(encoding="utf-8"))
        self.assertEqual(review["status"], "PASS_SOURCE_ONLY_READER_V5_HARD_BOUNDED_UNBOUND")
        self.assertEqual(review["candidate"]["sha256"], EXPECTED_TEMPLATE_SHA256)
        self.assertEqual(len(V5_TEMPLATE.read_bytes()), review["candidate"]["bytes"])

    def test_reader_actual_score_receipt_literals_remain_unbound(self) -> None:
        values = _assignments(V5_TEMPLATE)
        self.assertIsNone(ast.literal_eval(values["SCORE_ADMISSION_BYTES"]))
        self.assertIsNone(ast.literal_eval(values["SCORE_LAUNCH_BYTES"]))
        for name in ("SCORE_ADMISSION_SHA256", "SCORE_LAUNCH_SHA256", "SCORE_INVOCATION_ID"):
            self.assertIn("UNBOUND", ast.literal_eval(values[name]))


if __name__ == "__main__":
    unittest.main()
