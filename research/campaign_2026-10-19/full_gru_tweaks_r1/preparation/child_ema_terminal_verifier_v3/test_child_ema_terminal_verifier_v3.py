"""Focused V3 tests for the V2 failure binding and direct helper references."""

from __future__ import annotations

import unittest

import prepare_child_ema_terminal_verifier_v3 as packet


class EmaTerminalVerifierV3Tests(unittest.TestCase):
    """Check only pinned failure metadata and static helper names."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = packet.stable_read(
            packet.PAYLOAD_INPUTS["child_ema_terminal_check_v3.py"], 50_000
        )
        cls.helpers = {
            name: packet.stable_read(packet.PAYLOAD_INPUTS[name], 100_000)
            for name in (
                "child_control_terminal_check_v3.py",
                "parent_fit_terminal_check_v3.py",
                "propensity_terminal_support.py",
                "teacher_s5_terminal_check_v1.py",
            )
        }
        cls.v2_failure = {
            name: packet.stable_read(packet.PAYLOAD_INPUTS[name], 40_000)
            for name in ("V2_FAILURE_STDOUT.raw.json", "V2_FAILURE_UNIT_LOG.txt")
        }

    def test_frozen_checker_references_bind_to_exact_helper_modules(self) -> None:
        audit = packet.audit_helper_attribute_bindings(self.checker, self.helpers)
        self.assertEqual(audit["module_count"], 4)
        self.assertEqual(audit["unresolved"], [])
        self.assertEqual(audit["direct_attribute_reference_count"], 79)
        self.assertEqual(audit["unique_direct_attribute_reference_count"], 50)

    def test_wrong_namespace_reference_is_rejected(self) -> None:
        wrong = self.checker.replace(
            b"control_v3.PARENT_ACCEPTANCE_SHA", b"parent_v3.PARENT_ACCEPTANCE_SHA"
        )
        self.assertNotEqual(wrong, self.checker)
        with self.assertRaisesRegex(
            ValueError, "parent_fit_terminal_check_v3.PARENT_ACCEPTANCE_SHA"
        ):
            packet.audit_helper_attribute_bindings(wrong, self.helpers)

    def test_actual_v2_failed_launch_and_traceback_are_bound(self) -> None:
        result = packet.validate_v2_failure_binding(self.v2_failure)
        self.assertEqual(result["v2_support_invocation"], packet.V2_SUPPORT_INVOCATION)
        self.assertEqual(result["v2_body_sha256"], packet.V2_BODY_SHA)
        self.assertEqual(result["failure"], "parent_v3.PARENT_ACCEPTANCE_SHA undefined")

    def test_v2_failure_capture_and_log_drift_are_rejected(self) -> None:
        changed_capture = dict(self.v2_failure)
        changed_capture["V2_FAILURE_STDOUT.raw.json"] += b" "
        with self.assertRaisesRegex(RuntimeError, "V2 failure evidence digest differs"):
            packet.validate_v2_failure_binding(changed_capture)
        changed_log = dict(self.v2_failure)
        changed_log["V2_FAILURE_UNIT_LOG.txt"] += b" "
        with self.assertRaisesRegex(RuntimeError, "V2 failure evidence digest differs"):
            packet.validate_v2_failure_binding(changed_log)


if __name__ == "__main__":
    unittest.main()
