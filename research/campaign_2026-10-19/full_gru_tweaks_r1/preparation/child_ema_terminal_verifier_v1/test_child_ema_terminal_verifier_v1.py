"""Focused bindings for the EMA verifier admission adapter."""

from __future__ import annotations

import ast
import base64
import json
import unittest

import prepare_child_ema_terminal_verifier_v1 as packet


class EmaTerminalVerifierBindingTests(unittest.TestCase):
    """Exercise actual stored metadata and narrow admission invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = {
            name: packet.stable_read(path, 100_000) for name, path in packet.LOCAL_INPUTS.items()
        }

    def test_actual_launch_acceptance_and_native_capture_bind(self) -> None:
        result = packet.validate_metadata_bindings(
            self.metadata["ADMISSION.json"],
            self.metadata["LAUNCH.json"],
            self.metadata["LAUNCH_ACCEPTANCE.json"],
            self.metadata["TERMINAL_EVIDENCE.json"],
            self.metadata["NATIVE_STDOUT.raw.json"],
        )
        self.assertEqual(
            result["native_exit_fields"],
            {"main_pid": 0, "exec_main_code": 1, "exec_main_status": 0},
        )
        self.assertEqual(result["existing_receipt_count"], 8)

    def test_boolean_main_pid_is_not_an_integer_zero(self) -> None:
        terminal = json.loads(self.metadata["TERMINAL_EVIDENCE.json"])
        terminal["main_pid"] = False
        with self.assertRaisesRegex(ValueError, "exact integer 0"):
            packet.validate_terminal_object(terminal)

    def test_wrong_native_invocation_is_rejected(self) -> None:
        terminal = json.loads(self.metadata["TERMINAL_EVIDENCE.json"])
        terminal["invocation_id"] = "0" * 32
        with self.assertRaisesRegex(ValueError, "identity differs"):
            packet.validate_terminal_object(terminal)

    def test_root_acceptance_body_rebinding_is_rejected(self) -> None:
        acceptance = json.loads(self.metadata["LAUNCH_ACCEPTANCE.json"])
        acceptance["body"]["sha256"] = "0" * 64
        altered = json.dumps(acceptance, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "acceptance digest differs"):
            packet.validate_metadata_bindings(
                self.metadata["ADMISSION.json"],
                self.metadata["LAUNCH.json"],
                altered,
                self.metadata["TERMINAL_EVIDENCE.json"],
                self.metadata["NATIVE_STDOUT.raw.json"],
            )

    def test_rendered_payload_bytes_match_fixed_local_inputs(self) -> None:
        body, expected, _ = packet.render()
        module = ast.parse(body)
        module.body = [
            node
            for node in module.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "main"
            )
        ]
        namespace: dict[str, object] = {}
        exec(compile(module, "REMOTE_SUBMISSION.py", "exec"), namespace)
        embedded = namespace["payload_bytes"]()
        self.assertEqual(embedded, expected)
        self.assertEqual(len(embedded), 9)
        self.assertEqual(
            base64.b64decode(namespace["PAYLOAD"]["NATIVE_STDOUT.raw.json"]["base64"]),
            self.metadata["NATIVE_STDOUT.raw.json"],
        )


if __name__ == "__main__":
    unittest.main()
