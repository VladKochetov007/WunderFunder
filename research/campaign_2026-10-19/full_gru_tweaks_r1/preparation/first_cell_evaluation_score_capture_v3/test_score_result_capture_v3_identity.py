"""Focused metadata checks for the score-result capture V3 rebase."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
TEMPLATE_V2 = (
    PREPARATION / "first_cell_evaluation_score_capture_v2" / "SCORE_RESULT_CAPTURE_TEMPLATE_V2.py"
)
TEMPLATE_V3 = PACKET / "SCORE_RESULT_CAPTURE_TEMPLATE_V3.py"
SCORE_V5 = PREPARATION / "first_cell_evaluation_score_v5"
EXPECTED_PINS = {
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
SCORE_UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"
EXPORT_UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
RESULT_PATH = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/"
    "fg19_first_cell_f0_s1001_evaluation_v3/RESULT.json"
)


def _load_template() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_result_capture_v3", TEMPLATE_V3)
    if spec is None or spec.loader is None:
        raise AssertionError("V3 result-capture template could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
    return values


def _normalized(tree: ast.Module) -> ast.Module:
    values = _top_assignments(tree)
    for name in ("EXPORT_D", "EXPORT_UNIT", "SCORER_CODE_PINS"):
        replacement = ast.Constant(value=f"<{name}>")
        values[name] = replacement
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                node.value = replacement
    return tree


class ScoreResultCaptureV3Identity(unittest.TestCase):
    def test_v3_is_only_the_export_root_unit_and_scorer_pin_rebase(self) -> None:
        v2_tree, v3_tree = _tree(TEMPLATE_V2), _tree(TEMPLATE_V3)
        self.assertEqual(
            ast.dump(_normalized(v2_tree), include_attributes=False),
            ast.dump(_normalized(v3_tree), include_attributes=False),
        )

    def test_v3_binds_export_v3_result_root_and_exact_v5_scorer_files(self) -> None:
        module = _load_template()
        self.assertEqual(str(module.RESULT_PATH), RESULT_PATH)
        self.assertEqual(module.SCORE_UNIT, SCORE_UNIT)
        self.assertEqual(module.EXPORT_UNIT, EXPORT_UNIT)
        self.assertEqual(module.SCORER_CODE_PINS, EXPECTED_PINS)
        for name, expected in EXPECTED_PINS.items():
            raw = (SCORE_V5 / name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), expected)

    def test_exact_unit_id_receipts_and_strict_native_success_gate(self) -> None:
        module = _load_template()
        module._require_exact_unit_identity({"Id": SCORE_UNIT})
        for properties in ({"Id": "other.service"}, {}, {"Id": ""}):
            with (
                self.subTest(properties=properties),
                self.assertRaisesRegex(ValueError, "different score unit Id"),
            ):
                module._require_exact_unit_identity(properties)
        self.assertIn("Id", module.UNIT_PROPERTIES)
        self.assertTrue(
            module._strict_terminal_success(
                {
                    "Result": "success",
                    "ActiveState": "active",
                    "SubState": "exited",
                    "MainPID": "0",
                    "ExecMainCode": "1",
                    "ExecMainStatus": "0",
                }
            )
        )
        self.assertFalse(
            module._strict_terminal_success(
                {
                    "Result": "success",
                    "ActiveState": "active",
                    "SubState": "running",
                    "MainPID": "123",
                    "ExecMainCode": "1",
                    "ExecMainStatus": "0",
                }
            )
        )

    def test_synthetic_receipts_join_v3_export_and_v5_scorer_without_binding_actual_pins(
        self,
    ) -> None:
        module = _load_template()
        admission = {
            "schema": "fg19.first-cell-score-admission.v1",
            "status": "FRESH_GATES_PASSED_SUBMISSION_PENDING",
            "unit": SCORE_UNIT,
            "result_path": RESULT_PATH,
            "export_unit": EXPORT_UNIT,
            "export_invocation_id": "a" * 32,
            "result_create_only_and_absent_before_mutation": True,
            "export_acceptance": {
                "bytes": 100,
                "sha256": "1" * 64,
                "staged_path": "accepted_export/EXPORT_ACCEPTANCE.json",
            },
            "export_exact_stop": {
                "bytes": 101,
                "sha256": "2" * 64,
                "staged_path": "accepted_export/EXPORT_EXACT_STOP.json",
            },
            "staged_files": {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in EXPECTED_PINS.items()
            },
        }
        admission_raw = (
            json.dumps(admission, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        launch = {
            "schema": "fg19.first-cell-score-launch.v1",
            "unit": SCORE_UNIT,
            "invocation_id": "b" * 32,
            "export_unit": EXPORT_UNIT,
            "export_invocation_id": admission["export_invocation_id"],
            "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
            "submission": {"outcome": "submitted", "returncode": 0},
        }
        launch_raw = (json.dumps(launch, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        joined = module._validate_receipt_metadata(admission_raw, launch_raw, "b" * 32)
        self.assertEqual(joined["export_invocation_id"], "a" * 32)
        self.assertEqual(
            joined["staged_code_pins"],
            {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in EXPECTED_PINS.items()
            },
        )

    def test_actual_score_receipt_literals_remain_unbound(self) -> None:
        module = _load_template()
        for name in (
            "SCORE_ADMISSION_BYTES",
            "SCORE_ADMISSION_SHA256",
            "SCORE_LAUNCH_BYTES",
            "SCORE_LAUNCH_SHA256",
            "SCORE_INVOCATION_ID",
        ):
            self.assertTrue(getattr(module, name) is None or "UNBOUND" in getattr(module, name))
        with self.assertRaisesRegex(RuntimeError, "has not bound exact score admission bytes"):
            module._require_bound_pins()


if __name__ == "__main__":
    unittest.main()
