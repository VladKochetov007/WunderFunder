"""Metadata-only checks for rebinding the root score-status driver to reader V4."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).parent
PREPARATION = PACKET.parent
V1_DRIVER = PACKET / "ROOT_CAPTURE_SCORE_STATUS_V1.py"
V2_DRIVER = PACKET / "ROOT_CAPTURE_SCORE_STATUS_V2.py"
V4_TEMPLATE = (
    PREPARATION / "first_cell_evaluation_score_capture_v4" / "SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
)
V4_REVIEW = (
    PREPARATION / "first_cell_evaluation_score_capture_v4" / "INDEPENDENT_SOURCE_REVIEW_V4.json"
)
EXPECTED_TEMPLATE_SHA256 = "88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6"
EXPECTED_REVIEW_SHA256 = "b05b06a6e56888348e30ebe0ded0272a5e9f3ca709037feb975b938b4d0bb7d0"
REBASABLE = {"TEMPLATE", "TEMPLATE_SHA", "REVIEW", "REVIEW_SHA"}


def _tree(path: Path) -> ast.Module:
    """Parse one local source file without importing its executable entrypoint."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _normalize_driver(path: Path) -> ast.Module:
    """Replace only the four versioned reader/review assignments."""
    tree = _tree(path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in REBASABLE:
                node.value = ast.Constant(value=f"<{target.id}>")
    return tree


def _top_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = node.value
    return values


class RootScoreStatusDriverV2Rebase(unittest.TestCase):
    def test_driver_delta_is_exactly_the_four_reader_and_review_pins(self) -> None:
        self.assertEqual(
            ast.dump(_normalize_driver(V1_DRIVER), include_attributes=False),
            ast.dump(_normalize_driver(V2_DRIVER), include_attributes=False),
        )

    def test_driver_targets_the_exact_reviewed_v4_reader(self) -> None:
        values = _top_assignments(_tree(V2_DRIVER))
        template_path = values["TEMPLATE"]
        review_path = values["REVIEW"]
        self.assertIsInstance(template_path, ast.BinOp)
        self.assertIsInstance(template_path.left, ast.Name)
        self.assertEqual(template_path.left.id, "C")
        self.assertIsInstance(template_path.right, ast.Constant)
        self.assertEqual(
            template_path.right.value,
            "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py",
        )
        self.assertIsInstance(review_path, ast.BinOp)
        self.assertEqual(ast.unparse(review_path.left), "TEMPLATE.parent")
        self.assertIsInstance(review_path.right, ast.Constant)
        self.assertEqual(review_path.right.value, "INDEPENDENT_SOURCE_REVIEW_V4.json")
        self.assertEqual(ast.literal_eval(values["TEMPLATE_SHA"]), EXPECTED_TEMPLATE_SHA256)
        self.assertEqual(ast.literal_eval(values["REVIEW_SHA"]), EXPECTED_REVIEW_SHA256)
        for path, expected in (
            (V4_TEMPLATE, EXPECTED_TEMPLATE_SHA256),
            (V4_REVIEW, EXPECTED_REVIEW_SHA256),
        ):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_reader_review_is_passed_and_actual_score_pins_stay_unbound(self) -> None:
        review = json.loads(V4_REVIEW.read_text(encoding="utf-8"))
        self.assertEqual(review["status"], "PASS_SOURCE_ONLY_READER_V4_HARD_BOUNDED_UNBOUND")
        self.assertEqual(
            review["candidate"],
            {
                "path": (
                    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                    "first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
                ),
                "bytes": 26_354,
                "sha256": EXPECTED_TEMPLATE_SHA256,
            },
        )
        values = _top_assignments(_tree(V4_TEMPLATE))
        self.assertIsNone(ast.literal_eval(values["SCORE_ADMISSION_BYTES"]))
        self.assertIsNone(ast.literal_eval(values["SCORE_LAUNCH_BYTES"]))
        self.assertIn("UNBOUND", ast.literal_eval(values["SCORE_INVOCATION_ID"]))
        driver_tree = _tree(V2_DRIVER)
        guarded_main = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "main"
                for child in ast.walk(node)
            )
            for node in driver_tree.body
        )
        self.assertTrue(guarded_main)


if __name__ == "__main__":
    unittest.main()
