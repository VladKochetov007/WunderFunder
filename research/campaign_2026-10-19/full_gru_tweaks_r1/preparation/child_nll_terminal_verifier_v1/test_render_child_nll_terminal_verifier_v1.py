"""Focused negative checks for the conditional NLL renderer."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET))

renderer = importlib.import_module("render_child_nll_terminal_verifier_v1")


class ConditionalNllRendererTests(unittest.TestCase):
    """Check render prerequisites without creating a body or valid fake proof."""

    def test_static_root_preparation_must_not_claim_remote_execution(self) -> None:
        identity = {
            "run_name": renderer.RUN,
            "unit": renderer.UNIT,
            "invocation_id": renderer.INVOCATION,
            "source_commit": renderer.COMMIT,
            "source_files": 167,
            "source_manifest_sha256": renderer.MANIFEST_SHA,
        }
        root = {
            "schema": renderer.ROOT_SCHEMA,
            "status": "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED",
            "identity": identity,
            "dependency_binding_condition_satisfied": True,
            "native_terminal_evidence_present": True,
            "actual_B_staging_admission_or_execution": False,
        }
        with self.assertRaisesRegex(ValueError, "root-preparation identity/status differs"):
            renderer._check_root_preparation(json.dumps(root).encode())

    def test_static_root_preparation_requires_source_review_and_binding(self) -> None:
        root = {
            "schema": renderer.ROOT_SCHEMA,
            "status": "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED",
            "identity": {},
            "dependency_binding_condition_satisfied": False,
            "native_terminal_evidence_present": False,
            "actual_B_staging_admission_or_execution": False,
        }
        with self.assertRaisesRegex(ValueError, "root-preparation identity/status differs"):
            renderer._check_root_preparation(json.dumps(root).encode())

    def test_terminal_native_exit_rejects_boolean_as_integer(self) -> None:
        capture_raw = json.dumps(
            {
                "schema": "fg19-child-nll-startup-capture.v1",
                "expected_run": renderer.RUN,
                "expected_unit": renderer.UNIT,
                "expected_invocation_id": renderer.INVOCATION,
                "captured_utc": "2026-10-21T00:00:00+00:00",
                "unit_properties": {
                    "InvocationID": renderer.INVOCATION,
                    "ActiveState": "active",
                    "SubState": "exited",
                    "MainPID": "0",
                    "ExecMainCode": "1",
                    "ExecMainStatus": "0",
                    "Result": "success",
                },
            },
            sort_keys=True,
        ).encode()
        terminal_raw = json.dumps(
            {
                "schema": renderer.NATIVE_SCHEMA,
                "run_name": renderer.RUN,
                "unit": renderer.UNIT,
                "invocation_id": renderer.INVOCATION,
                "active_state": "active",
                "sub_state": "exited",
                "main_pid": True,
                "exec_main_code": 1,
                "exec_main_status": 0,
                "raw_path": (
                    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/"
                    "terminal_native_capture_v1/STDOUT.raw.json"
                ),
                "raw_bytes": len(capture_raw),
                "raw_sha256": hashlib.sha256(capture_raw).hexdigest(),
                "captured_utc": "2026-10-21T00:00:00+00:00",
                "native_exit_utc": "2026-10-21T00:00:00+00:00",
            },
            sort_keys=True,
        ).encode()
        with self.assertRaisesRegex(ValueError, "main_pid is not the exact integer 0"):
            renderer._check_native_proof(terminal_raw, capture_raw)

    def test_native_proof_rejects_wrong_invocation_before_accepting_state(self) -> None:
        wrong = {
            "schema": renderer.NATIVE_SCHEMA,
            "run_name": renderer.RUN,
            "unit": renderer.UNIT,
            "invocation_id": "0" * 32,
            "active_state": "active",
            "sub_state": "exited",
        }
        with self.assertRaisesRegex(ValueError, "identity/state differs"):
            renderer._check_native_proof(json.dumps(wrong).encode(), b"{}")

    def test_template_keeps_source_import_path_effective(self) -> None:
        template = (PACKET / "REMOTE_SUBMISSION_TEMPLATE.py").read_text(encoding="utf-8")
        checker = renderer.CHECKER.read_text(encoding="utf-8")
        self.assertEqual(
            hashlib.sha256(template.encode("utf-8")).hexdigest(), renderer.TEMPLATE_BASE_SHA
        )
        self.assertIn("str(CANONICAL_PYTHON)", template)
        self.assertIn('"-E"', template)
        source_cli_call = checker.find("launch_args = _launch_cli_args(launch, source_root)")
        source_module_import = checker.find(
            'representation_module = importlib.import_module("wnn.training.input_representation")'
        )
        self.assertGreaterEqual(source_cli_call, 0)
        self.assertGreater(source_module_import, source_cli_call)

    def test_actual_root_pins_v2_proof_without_legacy_cold_proof_name(self) -> None:
        root_raw = renderer._read_regular(renderer.ROOT_PREPARATION, renderer.MAX_METADATA_BYTES)
        root, code, _, pins = renderer._check_root_preparation(root_raw)
        self.assertEqual(set(code), set(renderer.CODE_INPUTS))
        self.assertEqual(root["accepted_import_boundary_proof"], renderer.IMPORT_BOUNDARY_PROOF_V2)
        self.assertFalse(root["local_proof_scope"]["local_proof_is_worker_runtime_proof"])
        paths = {str(item["path"]) for item in pins}
        self.assertTrue(any(path.endswith("IMPORT_BOUNDARY_PROOF_V2.json") for path in paths))
        self.assertFalse(any("COLD_IMPORT_PROOF" in path for path in paths))

    def test_renderer_refuses_output_without_native_terminal_bytes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(renderer, "BODY", Path(temporary_directory) / "REMOTE_SUBMISSION.py"),
            patch.object(
                renderer, "TERMINAL", Path(temporary_directory) / "TERMINAL_EVIDENCE.json"
            ),
        ):
            body_path = Path(temporary_directory) / "REMOTE_SUBMISSION.py"
            try:
                renderer.render_body()
            except FileNotFoundError:
                pass
            else:
                self.fail("renderer accepted missing typed terminal evidence")
            self.assertFalse(body_path.exists())

    def test_renderer_has_no_legacy_import_proof_filename_dependency(self) -> None:
        source = (PACKET / "render_child_nll_terminal_verifier_v1.py").read_text(encoding="utf-8")
        self.assertIn('root.get("metadata_and_proof_pins")', source)
        self.assertNotIn("COLD_IMPORT_PROOF", source)
        self.assertNotIn("COLD_IMPORT_PROOF.json", source)


if __name__ == "__main__":
    unittest.main()
