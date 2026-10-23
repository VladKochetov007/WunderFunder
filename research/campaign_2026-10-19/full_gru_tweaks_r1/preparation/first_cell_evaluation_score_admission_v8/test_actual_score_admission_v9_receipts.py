"""Exercise the actual fixed-export receipt contract and six rendered payloads."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType

REPOSITORY = Path(__file__).resolve().parents[5]
PREPARATION = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
    "first_cell_evaluation_score_admission_v8"
)
RECORD = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/export_v5"
)
TEMPLATE = PREPARATION / "SCORE_ADMISSION_BODY_TEMPLATE_V8.py"
ACTUAL_PLAN = PREPARATION / "ACTUAL_BINDING_PLAN_V8.json"
BODY = PREPARATION / "REMOTE_SUBMISSION_TEMPLATE_V8.py"
SOURCE_BINDING = PREPARATION / "SCORE_ADMISSION_SOURCE_BINDING_V8.json"
UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
INVOCATION = "78fa25d0abc44d199e3696d172dbc4f9"
ACCEPTANCE_PIN = (
    8833,
    "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d",
)
STOP_PIN = (
    1140,
    "695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f",
)
EXPORT_BINDING_SHA256 = "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd"
EXPORT_REVIEW_SHA256 = "64c93d8279a2daeb15f2750e6366a0fe94a268c7b8e9a04dbdb28e4a36a56e73"


def sha256(raw: bytes) -> str:
    """Hash exact bytes used by the actual receipt and payload joins."""
    return hashlib.sha256(raw).hexdigest()


def load_template_module() -> ModuleType:
    """Load only the standard-library score-body module for its pure receipt leaf."""
    spec = importlib.util.spec_from_file_location(
        "actual_score_admission_v8_receipt_contract", TEMPLATE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("score-body template could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rendered_payloads(raw: bytes) -> dict[str, bytes]:
    """Decode the one literal six-payload mapping from rendered source AST."""
    tree = ast.parse(raw.decode("utf-8", errors="strict"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PAYLOAD_B64"
    ]
    if len(nodes) != 1:
        raise AssertionError("rendered body must contain exactly one PAYLOAD_B64 mapping")
    encoded = ast.literal_eval(nodes[0].value)
    if not isinstance(encoded, dict) or any(
        not isinstance(value, str) for value in encoded.values()
    ):
        raise AssertionError("rendered PAYLOAD_B64 mapping is malformed")
    return {name: base64.b64decode(value, validate=True) for name, value in encoded.items()}


class ActualExportReceiptContractTests(unittest.TestCase):
    """Check the accepted actual export and the rendered six-source payload bundle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(ACTUAL_PLAN.read_bytes())
        authority = cls.plan["export_authority"]
        cls.acceptance_path = Path(authority["acceptance"]["path"])
        cls.stop_path = Path(authority["exact_stop"]["path"])
        cls.acceptance_raw = cls.acceptance_path.read_bytes()
        cls.stop_raw = cls.stop_path.read_bytes()
        cls.acceptance = json.loads(cls.acceptance_raw)
        cls.stop = json.loads(cls.stop_raw)
        cls.template_module = load_template_module()

    def test_actual_acceptance_and_stop_pass_consumer_leaf(self) -> None:
        """The actual ROOT acceptance and exact stop satisfy the 13-key consumer gate."""
        self.assertEqual((len(self.acceptance_raw), sha256(self.acceptance_raw)), ACCEPTANCE_PIN)
        self.assertEqual((len(self.stop_raw), sha256(self.stop_raw)), STOP_PIN)
        self.assertEqual(len(self.stop), 13)
        self.assertEqual(
            self.plan["export_authority"]["acceptance"],
            {
                "bytes": ACCEPTANCE_PIN[0],
                "expected_schema": "fg19.first-cell-export-acceptance.v1",
                "expected_status": "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED",
                "path": str(self.acceptance_path),
                "sha256": ACCEPTANCE_PIN[1],
            },
        )
        self.assertEqual(
            self.plan["export_authority"]["exact_stop"],
            {
                "bytes": STOP_PIN[0],
                "expected_schema": "fg19-first-cell-export-exact-owned-stop.v1",
                "path": str(self.stop_path),
                "sha256": STOP_PIN[1],
            },
        )
        invocation = self.template_module.validate_export_authority(
            self.acceptance,
            self.stop,
            acceptance_sha256=ACCEPTANCE_PIN[1],
            binding_sha256=EXPORT_BINDING_SHA256,
            review_sha256=EXPORT_REVIEW_SHA256,
            expected_unit=UNIT,
        )
        self.assertEqual(invocation, INVOCATION)

    def test_rendered_body_and_binding_join_all_six_exact_payloads(self) -> None:
        """Rendered source embeds four pinned code files and both ROOT authority receipts."""
        body_raw = BODY.read_bytes()
        binding = json.loads(SOURCE_BINDING.read_bytes())
        rendered_pin = binding["rendered_source"]
        self.assertEqual(
            (rendered_pin["path"], rendered_pin["bytes"], rendered_pin["sha256"]),
            (
                BODY.relative_to(REPOSITORY).as_posix(),
                len(body_raw),
                sha256(body_raw),
            ),
        )
        tree = ast.parse(body_raw.decode("utf-8", errors="strict"))
        self.assertFalse(
            any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree))
        )
        self.assertEqual(
            sum(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body),
            1,
        )

        expected_names = {
            "first_cell_score_contract.py",
            "first_cell_score_core.py",
            "first_cell_score_predictions.py",
            "score_first_cell.py",
            "EXPORT_ACCEPTANCE.json",
            "EXPORT_EXACT_STOP.json",
        }
        payloads = rendered_payloads(body_raw)
        self.assertEqual(set(payloads), expected_names)
        self.assertEqual(
            set(binding["scorer_payloads"]),
            expected_names,
        )
        self.assertEqual(set(binding["export_authority"]), {"acceptance", "exact_stop"})

        expected_files: dict[str, tuple[Path, int, str]] = {}
        for row in self.plan["frozen_inputs"]["score_modules"]:
            path = (PREPARATION / row["path"]).resolve()
            expected_files[path.name] = (path, row["bytes"], row["sha256"])
        expected_files["EXPORT_ACCEPTANCE.json"] = (
            self.acceptance_path,
            ACCEPTANCE_PIN[0],
            ACCEPTANCE_PIN[1],
        )
        expected_files["EXPORT_EXACT_STOP.json"] = (
            self.stop_path,
            STOP_PIN[0],
            STOP_PIN[1],
        )
        for name, (path, expected_bytes, expected_sha) in expected_files.items():
            raw = path.read_bytes()
            self.assertEqual((len(raw), sha256(raw)), (expected_bytes, expected_sha), name)
            self.assertEqual(payloads[name], raw, name)
            if name in binding["scorer_payloads"]:
                expected_path = (
                    str(path)
                    if name in {"EXPORT_ACCEPTANCE.json", "EXPORT_EXACT_STOP.json"}
                    else path.relative_to(REPOSITORY).as_posix()
                )
                self.assertEqual(
                    binding["scorer_payloads"][name],
                    {
                        "path": expected_path,
                        "bytes": expected_bytes,
                        "sha256": expected_sha,
                    },
                )
        self.assertEqual(
            binding["export_authority"]["acceptance"],
            {
                "path": str(self.acceptance_path),
                "bytes": ACCEPTANCE_PIN[0],
                "sha256": ACCEPTANCE_PIN[1],
            },
        )
        self.assertEqual(
            binding["export_authority"]["exact_stop"],
            {
                "path": str(self.stop_path),
                "bytes": STOP_PIN[0],
                "sha256": STOP_PIN[1],
            },
        )


if __name__ == "__main__":
    unittest.main()
