"""Focused source and receipt-binding fixtures for score admission V8."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
BODY_V7 = (
    PREPARATION / "first_cell_evaluation_score_admission_v7" / "SCORE_ADMISSION_BODY_TEMPLATE_V7.py"
)
BODY_V8 = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V8.py"
PLAN_V8 = PACKET / "SCORE_ADMISSION_BINDING_PLAN_V8.json"
RENDERER_V8 = PACKET / "render_score_admission_v8.py"
REVIEW_V4 = PREPARATION / "first_cell_evaluation_export_v4" / "EXPORT_SOURCE_REVIEW_V4.json"
SCORE_V5 = PREPARATION / "first_cell_evaluation_score_v5"
EXPECTED_REVIEW_PIN = (8712, "64c93d8279a2daeb15f2750e6366a0fe94a268c7b8e9a04dbdb28e4a36a56e73")
EXPECTED_BINDING_PIN = (15796, "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd")
EXPECTED_SCORER_PINS = {
    "first_cell_score_contract.py": (
        20165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
    ),
    "first_cell_score_core.py": (
        27226,
        "30c456e7db875e2c075a6200c857c375087b18d9c952c354ae9bd3dceb0a0143",
    ),
    "first_cell_score_predictions.py": (
        11056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}


def _load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_admission_v8_renderer", RENDERER_V8)
    if spec is None or spec.loader is None:
        raise AssertionError("V8 renderer module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
    return result


def _normalize_rebased_body(tree: ast.Module, *, candidate: bool) -> ast.Module:
    names = {
        "EXPORT_D",
        "EXPORT_UNIT",
        "EXPORT_BINDING_BYTES",
        "EXPORT_BINDING_SHA256",
        "SOURCE_REVIEW_BYTES",
        "SOURCE_REVIEW_SHA256",
        "SCORER_FILES",
    }
    assignments = _assignments(tree)
    for name in names:
        assignments[name] = ast.Constant(value=f"<{name}>")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    node.value = assignments[target.id]
    if candidate:
        old_name = "EXPORT_BINDING_V4.json"
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == old_name:
                node.value = "EXPORT_BINDING_V3.json"
                count += 1
        if count != 3:
            raise AssertionError(f"unexpected V8 binding-name rebasing count: {count}")
    return tree


class ScoreAdmissionV8PairBinding(unittest.TestCase):
    def test_actual_v4_review_bytes_and_all_twelve_rows_match_parser(self) -> None:
        renderer = _load_renderer()
        raw = REVIEW_V4.read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), EXPECTED_REVIEW_PIN)
        review = renderer.validate_source_review_bytes(raw)
        self.assertEqual(review["schema"], "fg19.first-cell-export-source-review.v4")
        self.assertEqual(review["status"], "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR")
        self.assertEqual(len(review["reviewed_files"]), 12)
        self.assertEqual(renderer._validate_source_review_object(review)["reviewed_file_count"], 12)

    def test_v4_review_parser_rejects_wrong_row_or_join_metadata(self) -> None:
        renderer = _load_renderer()
        actual = json.loads(REVIEW_V4.read_bytes())
        wrong_pin = copy.deepcopy(actual)
        wrong_pin["reviewed_files"][5]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "file pin differs"):
            renderer._validate_source_review_object(wrong_pin)
        missing_row = copy.deepcopy(actual)
        missing_row["reviewed_files"].pop()
        with self.assertRaisesRegex(ValueError, "membership differs"):
            renderer._validate_source_review_object(missing_row)
        wrong_join = copy.deepcopy(actual)
        key = next(iter(wrong_join["producer_consumer_join"]))
        wrong_join["producer_consumer_join"][key] = False
        with self.assertRaisesRegex(ValueError, "join is not a complete pass"):
            renderer._validate_source_review_object(wrong_join)

    def test_plan_binds_v4_v5_pair_and_leaves_export_receipts_unbound(self) -> None:
        plan = json.loads(PLAN_V8.read_bytes())
        self.assertEqual(plan["schema"], "fg19.first-cell-score-admission-binding-plan.v8")
        self.assertEqual(
            plan["status"], "LOCAL_CONDITIONAL_TEMPLATE_ONLY_UNBOUND_NOT_REVIEWED_OR_ADMITTED_V8"
        )
        self.assertEqual(
            plan["export_authority"]["expected_unit"],
            "computer-vision-homework-fg19-first-cell-export-v3.service",
        )
        for name in ("acceptance", "exact_stop"):
            self.assertIsNone(plan["export_authority"][name]["path"])
            self.assertIsNone(plan["export_authority"][name]["bytes"])
            self.assertIsNone(plan["export_authority"][name]["sha256"])
        self.assertEqual(
            plan["frozen_inputs"]["export_binding"],
            {
                "bytes": EXPECTED_BINDING_PIN[0],
                "path": "../first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json",
                "sha256": EXPECTED_BINDING_PIN[1],
            },
        )
        self.assertEqual(plan["frozen_inputs"]["source_review"]["reviewed_file_count"], 12)
        self.assertEqual(
            plan["frozen_inputs"]["score_dependency_manifest"]["path"],
            "../first_cell_evaluation_score_v5/dependency_manifest.json",
        )
        for row in plan["frozen_inputs"]["score_modules"]:
            expected = EXPECTED_SCORER_PINS[row["path"].rsplit("/", 1)[1]]
            self.assertEqual((row["bytes"], row["sha256"]), expected)
        self.assertTrue(
            all("evaluation_v3" in path for path in plan["read_only_export_paths"].values())
        )

    def test_body_is_only_rebased_identity_and_keeps_inventory_gate(self) -> None:
        v7_tree, v8_tree = _tree(BODY_V7), _tree(BODY_V8)
        self.assertEqual(
            ast.dump(_normalize_rebased_body(v7_tree, candidate=False), include_attributes=False),
            ast.dump(_normalize_rebased_body(v8_tree, candidate=True), include_attributes=False),
        )
        v7_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in v7_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"inventory_exact_source_files", "check_source_tree"}
        }
        v8_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in v8_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"inventory_exact_source_files", "check_source_tree"}
        }
        self.assertEqual(v7_functions, v8_functions)
        values = _assignments(_tree(BODY_V8))
        self.assertEqual(ast.literal_eval(values["EXPORT_BINDING_BYTES"]), EXPECTED_BINDING_PIN[0])
        self.assertEqual(ast.literal_eval(values["EXPORT_BINDING_SHA256"]), EXPECTED_BINDING_PIN[1])
        self.assertEqual(ast.literal_eval(values["SOURCE_REVIEW_BYTES"]), EXPECTED_REVIEW_PIN[0])
        self.assertEqual(ast.literal_eval(values["SOURCE_REVIEW_SHA256"]), EXPECTED_REVIEW_PIN[1])
        self.assertEqual(
            ast.unparse(values["EXPORT_D"]),
            "SCRATCH_ROOT / 'fg19_first_cell_f0_s1001_evaluation_v3'",
        )
        self.assertEqual(
            ast.literal_eval(values["EXPORT_UNIT"]),
            "computer-vision-homework-fg19-first-cell-export-v3.service",
        )
        self.assertEqual(
            ast.literal_eval(values["UNIT"]),
            "computer-vision-homework-fg19-first-cell-score-v1.service",
        )
        scorer_pins = ast.literal_eval(values["SCORER_FILES"])
        self.assertEqual(scorer_pins, EXPECTED_SCORER_PINS)
        self.assertEqual(ast.literal_eval(values["EXPORT_ACCEPTANCE_BYTES"]), 0)
        self.assertEqual(ast.literal_eval(values["EXPORT_EXACT_STOP_BYTES"]), 0)
        for name in ("EXPORT_ACCEPTANCE_SHA256", "EXPORT_EXACT_STOP_SHA256"):
            self.assertIsInstance(values[name], ast.Name)
            self.assertEqual(values[name].id, "UNBOUND")


if __name__ == "__main__":
    unittest.main()
