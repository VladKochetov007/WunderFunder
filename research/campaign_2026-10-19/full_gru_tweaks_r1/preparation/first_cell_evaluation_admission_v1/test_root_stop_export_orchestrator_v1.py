"""Exercise only the stop driver's local literal-binding and receipt gates."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

DRIVER = Path(__file__).with_name("ROOT_STOP_EXPORT_ORCHESTRATOR_V1.py")


def load_driver():
    """Load the root driver without invoking its command-line entrypoint."""
    spec = importlib.util.spec_from_file_location(
        "root_stop_export_orchestrator_v1_fixture", DRIVER
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load exact-stop orchestrator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RootStopExportOrchestratorTests(unittest.TestCase):
    def test_four_literal_binding_uses_only_exact_stop_template_slots(self) -> None:
        driver = load_driver()
        template, pin, _, _ = driver.validate_template_and_review()
        self.assertEqual(pin["bytes"], 6_299)
        acceptance_raw = b'{"fixture":"accepted-five-arm-export"}\n'
        acceptance = {
            "unit": driver.UNIT,
            "invocation_id": "555122f41c884850a6bd4e847c14e82f",
        }
        body = driver.bind_body(template, acceptance_raw, acceptance)
        text = body.decode("utf-8")
        self.assertIn(f"INVOCATION = {acceptance['invocation_id']!r}", text)
        self.assertIn(f"ACCEPTANCE_BYTES = {len(acceptance_raw)}", text)
        self.assertIn(f"ACCEPTANCE_SHA256 = {driver.sha(acceptance_raw)!r}", text)
        self.assertNotIn("UNBOUND_ACTUAL_EXPORT_INVOCATION", text)
        self.assertNotIn("UNBOUND_ROOT_EXPORT_ACCEPTANCE", text)
        self.assertNotIn("UNBOUND_ROOT_EXPORT_ACCEPTANCE_BASE64", text)

    def test_exact_stop_receipt_contract_accepts_only_success_and_empty_slice(self) -> None:
        driver = load_driver()
        invocation = "555122f41c884850a6bd4e847c14e82f"
        acceptance = {"invocation_id": invocation}
        acceptance_sha = "a" * 64
        receipt = {
            "schema": "fg19-first-cell-export-exact-owned-stop.v1",
            "captured_utc": "2026-10-21T09:50:00+00:00",
            "unit": driver.UNIT,
            "invocation_id": invocation,
            "acceptance_sha256": acceptance_sha,
            "before": {
                "Id": driver.UNIT,
                "InvocationID": invocation,
                "ActiveState": "active",
                "SubState": "exited",
                "MainPID": "0",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "Result": "success",
            },
            "after": {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": "0",
                "InvocationID": "",
            },
            "populated_before": {},
            "populated_after": {},
            "stop_returncode": 0,
            "stop_stdout": "",
            "stop_stderr": "",
            "scope": "Only the one successful exited export unit. All old native/support and foreign units excluded; no file or payload mutation.",
        }
        raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        self.assertEqual(driver.validate_remote_receipt(raw, acceptance, acceptance_sha), receipt)

        occupied = {**receipt, "populated_before": {"/campaign": "123\n"}}
        with self.assertRaisesRegex(ValueError, "campaign population"):
            driver.validate_remote_receipt(
                (json.dumps(occupied) + "\n").encode(), acceptance, acceptance_sha
            )

        wrong_acceptance = {**receipt, "acceptance_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "acceptance digest"):
            driver.validate_remote_receipt(
                (json.dumps(wrong_acceptance) + "\n").encode(), acceptance, acceptance_sha
            )


if __name__ == "__main__":
    unittest.main()
