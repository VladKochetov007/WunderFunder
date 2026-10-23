"""Metadata-only fixture for the scorer's new flat-support binding pin."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
SCORE_V1 = PREPARATION / "first_cell_evaluation_score_v1"
EXPORT_V1 = PREPARATION / "first_cell_evaluation_export_v1"
EXPORT_V2 = PREPARATION / "first_cell_evaluation_export_v2"
BINDING_NAME = "EXPORT_BINDING_V1.json"
BINDING_BYTE_COUNT = 15_798
BINDING_SHA256 = "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9"
FLAT_HELPERS = {
    "restricted_loader": "parent_fit_terminal_check_v3.py",
    "restricted_loader_support": "propensity_terminal_support.py",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path.name}")
    return value


def literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {
            "EXPECTED_BINDING_BYTES",
            "EXPECTED_BINDING_SHA256",
        }:
            if target.id in result:
                raise AssertionError(f"duplicate binding constant: {target.id}")
            result[target.id] = ast.literal_eval(node.value)
    return result


def core_tree_without_binding_pin(path: Path, replace_with: dict[str, Any]) -> ast.Module:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    observed: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in replace_with:
            observed.add(target.id)
            node.value = ast.Constant(value=replace_with[target.id])
    if observed != set(replace_with):
        raise AssertionError("expected binding pin assignments were not found exactly once")
    return tree


class BindingMetadataFixture(unittest.TestCase):
    def test_binding_paths_pin_and_scorer_core_are_exactly_versioned(self) -> None:
        binding_v1_path = EXPORT_V1 / BINDING_NAME
        binding_v2_path = EXPORT_V2 / BINDING_NAME
        binding_v1_raw = binding_v1_path.read_bytes()
        binding_v2_raw = binding_v2_path.read_bytes()
        self.assertEqual(len(binding_v2_raw), BINDING_BYTE_COUNT)
        self.assertEqual(hashlib.sha256(binding_v2_raw).hexdigest(), BINDING_SHA256)

        binding_v1 = json.loads(binding_v1_raw)
        binding_v2 = json.loads(binding_v2_raw)
        self.assertEqual(binding_v2["schema"], "fg19.first-cell-evaluation-export-binding.v1")
        self.assertEqual(
            binding_v2["source"]["restricted_loader"]["relative_path"],
            FLAT_HELPERS["restricted_loader"],
        )
        self.assertEqual(
            binding_v2["source"]["restricted_loader_support"]["relative_path"],
            FLAT_HELPERS["restricted_loader_support"],
        )
        binding_v1["source"]["restricted_loader"]["relative_path"] = FLAT_HELPERS[
            "restricted_loader"
        ]
        binding_v1["source"]["restricted_loader_support"]["relative_path"] = FLAT_HELPERS[
            "restricted_loader_support"
        ]
        self.assertEqual(binding_v1, binding_v2)

        expected = {
            "EXPECTED_BINDING_BYTES": BINDING_BYTE_COUNT,
            "EXPECTED_BINDING_SHA256": BINDING_SHA256,
        }
        core_v1 = SCORE_V1 / "first_cell_score_core.py"
        core_v2 = PACKET / "first_cell_score_core.py"
        self.assertEqual(literal_assignments(core_v2), expected)
        self.assertEqual(
            ast.dump(
                core_tree_without_binding_pin(core_v1, literal_assignments(core_v2)),
                include_attributes=False,
            ),
            ast.dump(
                core_tree_without_binding_pin(core_v2, literal_assignments(core_v2)),
                include_attributes=False,
            ),
        )

        manifest = read_json(PACKET / "dependency_manifest.json")
        self.assertEqual(manifest["schema"], "fg19.first-cell-score-dependency-manifest.v2")
        self.assertEqual(manifest["producer_contract"]["binding"]["bytes"], BINDING_BYTE_COUNT)
        self.assertEqual(manifest["producer_contract"]["binding"]["sha256"], BINDING_SHA256)
        core_pin = next(
            item
            for item in manifest["score_consumer_files"]
            if item["path"] == "first_cell_score_core.py"
        )
        core_v2_raw = core_v2.read_bytes()
        self.assertEqual(core_pin["bytes"], len(core_v2_raw))
        self.assertEqual(core_pin["sha256"], hashlib.sha256(core_v2_raw).hexdigest())
        fixture_pin = next(
            item
            for item in manifest["score_consumer_files"]
            if item["path"] == "test_score_v2_binding_metadata.py"
        )
        fixture_raw = Path(__file__).read_bytes()
        self.assertEqual(fixture_pin["bytes"], len(fixture_raw))
        self.assertEqual(fixture_pin["sha256"], hashlib.sha256(fixture_raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
