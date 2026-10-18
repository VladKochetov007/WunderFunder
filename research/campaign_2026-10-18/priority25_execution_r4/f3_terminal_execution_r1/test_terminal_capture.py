from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import capture_terminal_v1 as capture
from terminal_gate import GateError


def pack(raw: bytes) -> dict[str, object]:
    """Build the exact bounded raw-file envelope returned by B."""
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode(),
    }


class CaptureLifecycleTests(unittest.TestCase):
    """Exercise raw receipt persistence before stopping the support unit."""

    def test_capture_rejects_traversal_before_packet_or_record_lookup(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "capture_terminal_v1.py",
                    "--arm",
                    "wide",
                    "--operation",
                    "collector",
                    "--invocation",
                    "../outside",
                ],
            ),
            patch.object(capture, "verify_local_packet") as verify_packet,
            self.assertRaises(GateError),
        ):
            capture.main()
        verify_packet.assert_not_called()

    def _run_capture(
        self, *, helper_exit_status: int, unit_success: bool, launch_status: str = "LAUNCHED"
    ) -> tuple[str, dict[str, object], dict[str, object]]:
        invocation = "t20261018-abcdef12"
        unit = f"computer-vision-homework-r01-f3-wide-terminal-collector-{invocation}.service"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records"
            local_dir = records / invocation
            local_dir.mkdir(parents=True)
            manifest = {"runtime_files": {"terminal_remote_v1.py": {"sha256": "a" * 64}}}
            (root / "PREPARATION_MANIFEST.json").write_text(json.dumps(manifest))
            (local_dir / "LOCAL_LAUNCH_INTENT.json").write_text(
                json.dumps(
                    {
                        "arm": "wide",
                        "operation": "collector",
                        "invocation": invocation,
                        "manifest_sha256": "b" * 64,
                        "review_sha256": "c" * 64,
                    }
                )
            )
            (local_dir / "REMOTE_LAUNCH_RESPONSE.json").write_text(
                json.dumps({"status": launch_status, "invocation": invocation, "unit": unit})
            )
            receipt_name = "RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json"
            raw_files = {
                "ADMISSION_V1.json": json.dumps({"invocation": invocation}).encode(),
                "LAUNCH_V1.json": json.dumps({"invocation": invocation}).encode(),
                receipt_name: json.dumps(
                    {
                        "schema": "priority25.f3_checkpoint_collection.v1",
                        "arm": "wide",
                        "phase": "terminal",
                        "plan_sha256": capture.gate.PLAN_SHA256,
                        "attempt_evidence": {
                            "status": "finished",
                            "exit_status": helper_exit_status,
                        },
                        "source_snapshot_hash_match": helper_exit_status == 0,
                    }
                ).encode(),
            }
            response = {
                "schema": "priority25.f3_terminal_support_capture.v1",
                "status": "TERMINAL_CAPTURE",
                "unit": unit,
                "unit_success": unit_success,
                "raw_files": {name: pack(raw) for name, raw in raw_files.items()},
                "journal_tail": "bounded journal",
                "parent_memory_events": "oom 0\n",
                "populated_descendant_paths_before_stop": [],
            }
            command_count = 0

            def run(_args: list[str], **_kwargs: object) -> SimpleNamespace:
                nonlocal command_count
                command_count += 1
                if command_count == 1:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps(response),
                        stderr="",
                    )
                self.assertTrue((local_dir / "REMOTE_CAPTURE_RESPONSE.json").is_file())
                self.assertTrue((local_dir / "REMOTE_HELPER_RECEIPT.json").is_file())
                self.assertTrue((local_dir / "LOCAL_EXECUTION_V1.json").is_file())
                return SimpleNamespace(returncode=0, stdout='{"returncode":0}', stderr="")

            output = io.StringIO()
            with (
                patch.object(capture, "PACKET", root),
                patch.object(
                    capture, "verify_local_packet", return_value=(manifest, "b" * 64, "c" * 64)
                ),
                patch.object(capture.subprocess, "run", side_effect=run),
                patch.object(
                    sys,
                    "argv",
                    [
                        "capture_terminal_v1.py",
                        "--arm",
                        "wide",
                        "--operation",
                        "collector",
                        "--invocation",
                        invocation,
                    ],
                ),
                contextlib.redirect_stdout(output),
            ):
                capture.main()
            self.assertEqual(command_count, 2)
            self.assertTrue((local_dir / "LOCAL_STOP_V1.json").is_file())
            execution = json.loads((local_dir / "LOCAL_EXECUTION_V1.json").read_bytes())
            capture_response = json.loads((local_dir / "REMOTE_CAPTURE_RESPONSE.json").read_bytes())
            return output.getvalue(), execution, capture_response

    def test_successful_capture_is_persisted_once_before_stop(self) -> None:
        output, _execution, response = self._run_capture(helper_exit_status=0, unit_success=True)
        self.assertIn("CAPTURED_AND_OWN_UNIT_STOPPED", output)
        self.assertEqual(response["status"], "TERMINAL_CAPTURE")

    def test_failed_support_is_captured_before_only_its_unit_is_stopped(self) -> None:
        output, execution, _response = self._run_capture(helper_exit_status=1, unit_success=False)
        self.assertIn("FAILED_SUPPORT_CAPTURED_AND_OWN_UNIT_STOPPED", output)
        self.assertFalse(execution["unit_success"])
        self.assertFalse(execution["helper_success"])
        self.assertTrue(execution["capture_saved_before_stop"])

    def test_unconfirmed_launch_record_can_capture_exact_fast_exited_unit(self) -> None:
        output, execution, _response = self._run_capture(
            helper_exit_status=1,
            unit_success=False,
            launch_status="LAUNCH_UNCONFIRMED",
        )
        self.assertIn("FAILED_SUPPORT_CAPTURED_AND_OWN_UNIT_STOPPED", output)
        self.assertFalse(execution["unit_success"])


if __name__ == "__main__":
    unittest.main()
