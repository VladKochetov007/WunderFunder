"""Metadata-only producer/consumer pin tests for scorer V4."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
SCORE_V3 = PREPARATION / "first_cell_evaluation_score_v3"
EXPORT_V3 = PREPARATION / "first_cell_evaluation_export_v3"
EXPORT_V4 = PREPARATION / "first_cell_evaluation_export_v4"

V3_CODE_PINS = {
    "first_cell_score_core.py": (
        27_226,
        "ffe8a6e7aad38b131a78783504ef55c761d4e8037c13bac279f20c5e55f6249e",
    ),
    "first_cell_score_contract.py": (
        19_485,
        "7784c91713dbfd7d11ec3cfd276eb8a62607289067999f4ecd5c03a3f9a18038",
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
EXPORTER_V4_PIN = (66_790, "87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e")
BINDING_V4_PIN = (15_796, "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd")
SOURCE_MANIFEST_PIN = (27_569, "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621")
V3_BINDING_PIN = (15_798, "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730")
V3_BINDING_SHA = V3_BINDING_PIN[1]
V4_BINDING_SHA = BINDING_V4_PIN[1]


def _pin(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _normalized_ast(path: Path, replacements: dict[tuple[type, object], tuple[object, int]]) -> str:
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8", errors="strict"), filename=str(path))
    seen = {key: 0 for key in replacements}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            key = (type(node.value), node.value)
            if key in replacements:
                node.value = replacements[key][0]
                seen[key] += 1
    expected = {key: count for key, (_, count) in replacements.items()}
    if seen != expected:
        raise AssertionError(f"V4 source adaptation counts differ: {seen!r}")
    return ast.dump(tree, include_attributes=False)


def _replace_evidence_v4_values(value: object) -> tuple[object, int]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        count = 0
        for key, item in value.items():
            result[key], nested = _replace_evidence_v4_values(item)
            count += nested
        return result, count
    if isinstance(value, list):
        result_list = []
        count = 0
        for item in value:
            normalized, nested = _replace_evidence_v4_values(item)
            result_list.append(normalized)
            count += nested
        return result_list, count
    if isinstance(value, str) and "fg19_first_cell_f0_s1001_evaluation_v3" in value:
        return value.replace(
            "fg19_first_cell_f0_s1001_evaluation_v3", "fg19_first_cell_f0_s1001_evaluation_v2"
        ), 1
    return value, 0


class ScoreV4BindingMetadata(unittest.TestCase):
    def test_frozen_exporter_and_binding_pins_match_exact_files(self) -> None:
        self.assertEqual(_pin(EXPORT_V4 / "export_first_cell_v4.py"), EXPORTER_V4_PIN)
        self.assertEqual(_pin(EXPORT_V4 / "EXPORT_BINDING_V4.json"), BINDING_V4_PIN)
        source_manifest = EXPORT_V4 / "EXPORT_SOURCE_MANIFEST.json"
        self.assertEqual(_pin(source_manifest), SOURCE_MANIFEST_PIN)
        self.assertEqual(
            source_manifest.read_bytes(), (EXPORT_V3 / "EXPORT_SOURCE_MANIFEST.json").read_bytes()
        )

    def test_v4_binding_is_v3_path_rebase_plus_empty_prop_input_default(self) -> None:
        old_raw = (EXPORT_V3 / "EXPORT_BINDING_V3.json").read_bytes()
        new_raw = (EXPORT_V4 / "EXPORT_BINDING_V4.json").read_bytes()
        self.assertEqual(_pin(EXPORT_V3 / "EXPORT_BINDING_V3.json"), V3_BINDING_PIN)
        old = json.loads(old_raw)
        new = json.loads(new_raw)
        self.assertEqual(new["expected_checkpoint"]["prop_input"], "")
        new_without_prop_default = copy.deepcopy(new)
        new_without_prop_default["expected_checkpoint"]["prop_input"] = None
        normalized, path_count = _replace_evidence_v4_values(new_without_prop_default)
        self.assertEqual(path_count, 4)
        self.assertEqual(normalized, old)

    def test_v4_score_code_diff_is_only_binding_and_fixed_evidence_paths(self) -> None:
        core_replacements = {
            (int, 15_796): (15_798, 1),
            (str, V4_BINDING_SHA): (V3_BINDING_SHA, 1),
        }
        contract_replacements = {
            (str, "EXPORT_BINDING_V4.json"): ("EXPORT_BINDING_V3.json", 2),
            (str, "fg19_first_cell_f0_s1001_evaluation_v3"): (
                "fg19_first_cell_f0_s1001_evaluation_v2",
                1,
            ),
        }
        self.assertEqual(
            _normalized_ast(PACKET / "first_cell_score_core.py", core_replacements),
            ast.dump(
                ast.parse((SCORE_V3 / "first_cell_score_core.py").read_text(encoding="utf-8")),
                include_attributes=False,
            ),
        )
        self.assertEqual(
            _normalized_ast(PACKET / "first_cell_score_contract.py", contract_replacements),
            ast.dump(
                ast.parse((SCORE_V3 / "first_cell_score_contract.py").read_text(encoding="utf-8")),
                include_attributes=False,
            ),
        )
        for name in ("first_cell_score_predictions.py", "score_first_cell.py"):
            self.assertEqual((PACKET / name).read_bytes(), (SCORE_V3 / name).read_bytes())

    def test_only_frozen_runtime_binding_name_and_evidence_root_are_v4(self) -> None:
        contract = ast.parse((PACKET / "first_cell_score_contract.py").read_text(encoding="utf-8"))
        literals = [
            node.value
            for node in ast.walk(contract)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertEqual(literals.count("EXPORT_BINDING_V4.json"), 2)
        self.assertEqual(literals.count("fg19_first_cell_f0_s1001_evaluation_v3"), 1)
        core = ast.parse((PACKET / "first_cell_score_core.py").read_text(encoding="utf-8"))
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in core.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"EXPECTED_BINDING_BYTES", "EXPECTED_BINDING_SHA256"}
        }
        self.assertEqual(
            assignments,
            {
                "EXPECTED_BINDING_BYTES": BINDING_V4_PIN[0],
                "EXPECTED_BINDING_SHA256": BINDING_V4_PIN[1],
            },
        )


if __name__ == "__main__":
    unittest.main()
