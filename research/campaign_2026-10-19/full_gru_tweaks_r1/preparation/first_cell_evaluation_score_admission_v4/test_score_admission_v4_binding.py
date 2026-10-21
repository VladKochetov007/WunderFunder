"""Metadata-only join for conditional admission V4 and scorer/exporter V2."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
BODY_V3 = (
    PREPARATION / "first_cell_evaluation_score_admission_v3" / "SCORE_ADMISSION_BODY_TEMPLATE_V3.py"
)
BODY_V4 = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V4.py"
EXPORT_PACKET = PREPARATION / "first_cell_evaluation_export_v2"
SCORE_PACKET = PREPARATION / "first_cell_evaluation_score_v2"
PLAN_V4 = PACKET / "SCORE_ADMISSION_BINDING_PLAN_V4.json"
REVIEW_SIZE = 10_250
REVIEW_SHA256 = "f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af"
BINDING_SIZE = 15_798
BINDING_SHA256 = "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9"
CORE_SHA256 = "1b624ad568b436615f01f50e60567ece4a483ae9cc1def69ab2277aa0a392604"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path.name}")
    return value


def module_assignments(path: Path) -> tuple[ast.Module, dict[str, ast.Assign]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments: dict[str, ast.Assign] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id in assignments:
                raise AssertionError(f"duplicate top-level assignment: {target.id}")
            assignments[target.id] = node
    return tree, assignments


def replace_selected_assignment_values(
    tree: ast.Module,
    assignments: dict[str, ast.Assign],
    names: set[str],
    values: dict[str, ast.expr],
) -> ast.Module:
    found: set[str] = set()
    for name in names:
        if name not in assignments or name not in values:
            raise AssertionError(f"missing expected assignment: {name}")
        assignments[name].value = copy.deepcopy(values[name])
        found.add(name)
    if found != names:
        raise AssertionError("assignment normalization was incomplete")
    return tree


class AdmissionBindingFixture(unittest.TestCase):
    def test_v4_binds_exact_export_and_scorer_v2_pins_only(self) -> None:
        binding_path = EXPORT_PACKET / "EXPORT_BINDING_V1.json"
        binding_raw = binding_path.read_bytes()
        self.assertEqual(
            (len(binding_raw), hashlib.sha256(binding_raw).hexdigest()),
            (BINDING_SIZE, BINDING_SHA256),
        )
        review_path = EXPORT_PACKET / "EXPORT_SOURCE_REVIEW_V2.json"
        review_raw = review_path.read_bytes()
        self.assertEqual(
            (len(review_raw), hashlib.sha256(review_raw).hexdigest()), (REVIEW_SIZE, REVIEW_SHA256)
        )
        review = read_object(review_path)
        self.assertEqual(review["status"], "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR")

        binding = read_object(binding_path)
        self.assertEqual(
            binding["source"]["restricted_loader"]["relative_path"],
            "parent_fit_terminal_check_v3.py",
        )
        self.assertEqual(
            binding["source"]["restricted_loader_support"]["relative_path"],
            "propensity_terminal_support.py",
        )
        manifest = read_object(SCORE_PACKET / "dependency_manifest.json")
        plan = read_object(PLAN_V4)
        body_tree, body_assignments = module_assignments(BODY_V4)
        pin_names = {
            "EXPORT_BINDING_BYTES",
            "EXPORT_BINDING_SHA256",
            "SOURCE_REVIEW_BYTES",
            "SOURCE_REVIEW_SHA256",
            "SCORER_FILES",
        }
        body_values = {name: ast.literal_eval(body_assignments[name].value) for name in pin_names}
        self.assertEqual(body_values["EXPORT_BINDING_BYTES"], BINDING_SIZE)
        self.assertEqual(body_values["EXPORT_BINDING_SHA256"], BINDING_SHA256)
        self.assertEqual(body_values["SOURCE_REVIEW_BYTES"], REVIEW_SIZE)
        self.assertEqual(body_values["SOURCE_REVIEW_SHA256"], REVIEW_SHA256)
        self.assertEqual(
            body_values["SCORER_FILES"]["first_cell_score_core.py"], (27_226, CORE_SHA256)
        )
        self.assertEqual(
            body_values["SCORER_FILES"]["first_cell_score_contract.py"],
            (19_485, "09155537feec1d9cd398f6a729ab88201b82e814c1a8c4ac796fe4283940a90d"),
        )
        self.assertEqual(
            body_values["SCORER_FILES"]["first_cell_score_predictions.py"],
            (11_056, "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d"),
        )
        self.assertEqual(
            body_values["SCORER_FILES"]["score_first_cell.py"],
            (10_630, "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6"),
        )

        producer = manifest["producer_contract"]
        self.assertEqual(producer["binding"]["bytes"], BINDING_SIZE)
        self.assertEqual(producer["binding"]["sha256"], BINDING_SHA256)
        self.assertEqual(
            producer["source_review"]["path"],
            "../first_cell_evaluation_export_v2/EXPORT_SOURCE_REVIEW_V2.json",
        )
        self.assertEqual(plan["schema"], "fg19.first-cell-score-admission-binding-plan.v4")
        frozen = plan["frozen_inputs"]
        self.assertEqual(frozen["export_binding"]["sha256"], BINDING_SHA256)
        self.assertEqual(frozen["source_review"]["sha256"], REVIEW_SHA256)
        self.assertEqual(
            frozen["score_dependency_manifest"]["sha256"],
            hashlib.sha256((SCORE_PACKET / "dependency_manifest.json").read_bytes()).hexdigest(),
        )
        module_pins = {Path(item["path"]).name: item for item in frozen["score_modules"]}
        for name, expected in body_values["SCORER_FILES"].items():
            self.assertEqual((module_pins[name]["bytes"], module_pins[name]["sha256"]), expected)

        renderer_path = PACKET / "render_score_admission_v4.py"
        renderer_tree, renderer_assignments = module_assignments(renderer_path)
        renderer_values = {
            name: ast.literal_eval(renderer_assignments[name].value)
            for name in ("SOURCE_PINS", "SOURCE_REVIEW_PIN")
        }
        self.assertEqual(renderer_values["SOURCE_REVIEW_PIN"], (REVIEW_SIZE, REVIEW_SHA256))
        self.assertEqual(
            renderer_values["SOURCE_PINS"],
            {name: tuple(value) for name, value in body_values["SCORER_FILES"].items()},
        )
        renderer_text = renderer_path.read_text(encoding="utf-8")
        self.assertIn("first_cell_evaluation_score_v2", renderer_text)
        self.assertIn("first_cell_evaluation_export_v2", renderer_text)
        binding_keywords = [
            keyword.value
            for node in ast.walk(renderer_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate_export_authority"
            for keyword in node.keywords
            if keyword.arg == "binding_sha256"
        ]
        self.assertEqual([ast.literal_eval(node) for node in binding_keywords], [BINDING_SHA256])

        v3_tree, v3_assignments = module_assignments(BODY_V3)
        changed = {
            "EXPORT_BINDING_BYTES",
            "EXPORT_BINDING_SHA256",
            "SOURCE_REVIEW_BYTES",
            "SOURCE_REVIEW_SHA256",
            "SCORER_FILES",
        }
        v4_values = {name: body_assignments[name].value for name in changed}
        normalized_v3 = replace_selected_assignment_values(
            v3_tree, v3_assignments, changed, v4_values
        )
        self.assertEqual(
            ast.dump(normalized_v3, include_attributes=False),
            ast.dump(body_tree, include_attributes=False),
        )


if __name__ == "__main__":
    unittest.main()
