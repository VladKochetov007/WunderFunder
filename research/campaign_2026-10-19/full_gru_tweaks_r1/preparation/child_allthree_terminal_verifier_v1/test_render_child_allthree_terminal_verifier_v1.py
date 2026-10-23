"""Focused synthetic failure fixtures for the conditional AllThree renderer."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import unittest

import render_child_allthree_terminal_verifier_v1 as renderer


def valid_native_fixture() -> tuple[bytes, bytes]:
    """Create a tiny matching typed proof/raw capture pair without disk I/O."""
    captured = "2026-10-21T05:30:00+00:00"
    native_exit = "2026-10-21T05:29:00+00:00"
    capture = {
        "schema": renderer.RAW_SCHEMA,
        "expected_run": renderer.RUN,
        "expected_unit": renderer.UNIT,
        "expected_invocation_id": renderer.INVOCATION,
        "captured_utc": captured,
        "unit_properties": {
            "InvocationID": renderer.INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
        },
    }
    raw = json.dumps(capture, sort_keys=True).encode()
    terminal = {
        "schema": renderer.NATIVE_SCHEMA,
        "run_name": renderer.RUN,
        "unit": renderer.UNIT,
        "invocation_id": renderer.INVOCATION,
        "active_state": "active",
        "sub_state": "exited",
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_path": str(renderer.NATIVE_RAW.relative_to(renderer.REPOSITORY)),
        "captured_utc": captured,
        "native_exit_utc": native_exit,
    }
    return json.dumps(terminal, sort_keys=True).encode(), raw


def valid_import_boundary_fixture() -> dict[str, object]:
    """Build the minimal exact proof shape accepted by the source validator."""
    pins = {
        name: {"bytes": size, "sha256": digest}
        for name, (size, digest) in renderer.CODE_PINS.items()
    }
    helpers = {
        name: pin for name, pin in pins.items() if name != "child_allthree_terminal_check_v1.py"
    }
    return {
        "schema": "fg19.child-allthree-terminal-import-boundary-proof.v1",
        "status": "PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V1",
        "private_closure": {
            "pre_copy_source_pins": pins,
            "temporary_copy_pins_before_child_import": pins,
            "child_pre_import_pins": pins,
            "child_post_import_pins": pins,
            "child_preloaded_module_names": [],
            "helper_closure_recheck": helpers,
            "child_module_origins": {name: f"/tmp/private/{name}" for name in pins},
            "exact_file_count": 7,
            "exact_file_set_before_and_after_import": list(pins),
        },
        "entrypoint_scope": {
            "checker_functions_called": ["assert_helper_closure"],
            "verify_terminal_called": False,
            "main_or_cli_called": False,
            "fit_or_evaluation_called": False,
            "source_cli_run_identity_representation_or_model_imports_exercised": False,
            "wnn_source_modules_loaded": 0,
        },
        "interpreter_binding": {
            "child_python_flags": ["-E", "-B", "-I"],
            "child_dont_write_bytecode": True,
            "child_pycache_prefix": None,
            "private_bytecode_absent_before_import": True,
            "private_bytecode_absent_after_import": True,
        },
        "harness": {
            "path": str(renderer.IMPORT_BOUNDARY_HARNESS.resolve(strict=True)),
            "bytes": 23_177,
            "sha256": "6f4318aaa3eea9c88acc693b3d7a822a6b0e9ae61450202f6dedf79ef5bceda8",
        },
    }


def valid_body(
    payload: bytes = b"tiny", *, wrong_pin: bool = False
) -> tuple[str, dict[str, bytes]]:
    raw_sha = hashlib.sha256(payload).hexdigest()
    wrong_sha = "0" * 64 if wrong_pin else raw_sha
    payloads = {"tiny.py": payload}
    packaged = {
        "tiny.py": {
            "base64": base64.b64encode(payload).decode("ascii"),
            "bytes": len(payload),
            "sha256": raw_sha,
        }
    }
    fixed = {"tiny.py": {"bytes": len(payload), "sha256": wrong_sha}}
    body = (
        f"PAYLOAD = {packaged!r}\n"
        f"FIXED_PAYLOAD_PINS = {fixed!r}\n"
        "RUNNER_API = 'result = checker.verify_terminal(*tuple(Path(value) for value in ()))'\n"
        "def check_selected_gpu_ownership():\n    return {}\n"
        "def preflight(payload):\n    gpu = check_selected_gpu_ownership()\n    return {'gpu': gpu}\n"
        "def payload_bytes():\n    return {}\n"
        "class Output:\n    def mkdir(self):\n        pass\n"
        "D = Output()\n"
        "def main():\n    payload = payload_bytes()\n    gates = preflight(payload)\n    D.mkdir()\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    return body, payloads


class NativeProofTests(unittest.TestCase):
    def test_matching_strict_native_exit_and_raw_capture_pass(self) -> None:
        terminal_raw, capture_raw = valid_native_fixture()
        proof = renderer.check_native_proof(terminal_raw, capture_raw)
        self.assertEqual(proof["raw_bytes"], len(capture_raw))

    def test_boolean_exit_fields_are_not_strict_integers(self) -> None:
        terminal_raw, capture_raw = valid_native_fixture()
        terminal = json.loads(terminal_raw)
        terminal["main_pid"] = True
        with self.assertRaises(ValueError):
            renderer.check_native_proof(json.dumps(terminal).encode(), capture_raw)

    def test_boolean_raw_byte_count_is_rejected(self) -> None:
        terminal_raw, capture_raw = valid_native_fixture()
        terminal = json.loads(terminal_raw)
        terminal["raw_bytes"] = True
        with self.assertRaises(ValueError):
            renderer.check_native_proof(json.dumps(terminal).encode(), capture_raw)

    def test_wrong_capture_hash_and_naive_timestamp_are_rejected(self) -> None:
        terminal_raw, capture_raw = valid_native_fixture()
        terminal = json.loads(terminal_raw)
        terminal["raw_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            renderer.check_native_proof(json.dumps(terminal).encode(), capture_raw)
        terminal_raw, capture_raw = valid_native_fixture()
        terminal = json.loads(terminal_raw)
        terminal["captured_utc"] = "2026-10-21T05:30:00"
        with self.assertRaises(ValueError):
            renderer.check_native_proof(json.dumps(terminal).encode(), capture_raw)


class ImportBoundaryProofTests(unittest.TestCase):
    def test_final_local_import_proof_passes_frozen_contract(self) -> None:
        proof = json.loads(renderer.IMPORT_BOUNDARY_PROOF.read_text(encoding="utf-8"))
        renderer.check_import_boundary_proof(proof)

    def test_exact_private_import_boundary_passes(self) -> None:
        renderer.check_import_boundary_proof(valid_import_boundary_fixture())

    def test_preloaded_private_module_is_rejected(self) -> None:
        proof = valid_import_boundary_fixture()
        proof["private_closure"]["child_preloaded_module_names"] = [
            "child_allthree_terminal_check_v1"
        ]
        with self.assertRaisesRegex(ValueError, "preloaded private module"):
            renderer.check_import_boundary_proof(proof)

    def test_bytecode_and_verifier_entrypoint_must_remain_disabled(self) -> None:
        proof = valid_import_boundary_fixture()
        proof["interpreter_binding"]["child_pycache_prefix"] = "/tmp/pycache"
        with self.assertRaises(ValueError):
            renderer.check_import_boundary_proof(proof)
        proof = valid_import_boundary_fixture()
        proof["entrypoint_scope"]["verify_terminal_called"] = True
        with self.assertRaises(ValueError):
            renderer.check_import_boundary_proof(proof)


class RenderedBodyGuardTests(unittest.TestCase):
    def test_minimal_exact_payload_and_gate_order_pass(self) -> None:
        body, payloads = valid_body()
        self.assertTrue(renderer.check_rendered_body(body, payloads))

    def test_worker_self_read_and_body_hash_claim_fail(self) -> None:
        body, payloads = valid_body()
        with self.assertRaisesRegex(ValueError, "self-read"):
            renderer.check_rendered_body(body + "# Path(__file__)\n", payloads)
        with self.assertRaisesRegex(ValueError, "self-read"):
            renderer.check_rendered_body(body + '# body_sha256_claimed": True\n', payloads)

    def test_fixed_payload_pin_mismatch_fails(self) -> None:
        body, payloads = valid_body(wrong_pin=True)
        with self.assertRaisesRegex(ValueError, "fixed pin"):
            renderer.check_rendered_body(body, payloads)

    def test_checker_paths_must_be_path_objects(self) -> None:
        body, payloads = valid_body()
        with self.assertRaisesRegex(ValueError, "pathlib.Path"):
            renderer.check_rendered_body(
                body.replace("checker.verify_terminal", "checker.other"), payloads
            )

    def test_gpu_gate_and_output_mutation_order_are_required(self) -> None:
        body, payloads = valid_body()
        no_gpu = body.replace("    gpu = check_selected_gpu_ownership()\n", "")
        with self.assertRaisesRegex(ValueError, "GPU ownership"):
            renderer.check_rendered_body(no_gpu, payloads)
        bad_order = body.replace(
            "    gates = preflight(payload)\n    D.mkdir()",
            "    D.mkdir()\n    gates = preflight(payload)",
        )
        with self.assertRaisesRegex(ValueError, "fresh gates"):
            renderer.check_rendered_body(bad_order, payloads)


class LaunchProducerConsumerTests(unittest.TestCase):
    def test_template_launch_record_satisfies_capture_identity_contract(self) -> None:
        template_path = renderer.TEMPLATE
        spec = importlib.util.spec_from_file_location(
            "allthree_worker_template_fixture", template_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        support_invocation = "0123456789abcdef0123456789abcdef"
        post = {
            "outcome": "captured",
            "properties": {
                "InvocationID": support_invocation,
                "ActiveState": "exited",
                "SubState": "success",
            },
        }
        launch = module.make_launch_record({"outcome": "submitted", "returncode": 0}, post)
        expected = {
            "schema": "fg19.child-allthree-terminal-verifier-launch.v1",
            "unit": module.UNIT,
            "invocation_id": support_invocation,
            "native_unit": module.NATIVE_UNIT,
            "native_invocation_id": module.INVOCATION,
        }
        for key, value in expected.items():
            self.assertEqual(launch.get(key), value)
        capture_path = (
            renderer.CAMPAIGN
            / "preparation/child_allthree_terminal_result_capture_v1/RESULT_CAPTURE_TEMPLATE.py"
        )
        capture_source = capture_path.read_text(encoding="utf-8")
        for key in expected:
            self.assertIn(f'launch.get("{key}")', capture_source)
        self.assertNotIn("body_sha256_claimed", launch)


class TemplatePinTests(unittest.TestCase):
    def test_template_bytes_match_renderer_precondition(self) -> None:
        template = renderer.TEMPLATE.read_bytes()
        self.assertEqual(renderer.sha256(template), renderer.TEMPLATE_BASE_SHA)


if __name__ == "__main__":
    unittest.main()
