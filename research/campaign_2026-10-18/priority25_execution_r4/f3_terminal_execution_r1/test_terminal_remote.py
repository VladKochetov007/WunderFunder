from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import terminal_remote_v1 as remote
from terminal_gate import GateError


class TerminalOutputGateTests(unittest.TestCase):
    """Verify canonical result collisions and symlinked inputs fail closed."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "terminal_verification.json"
        self.snapshot = self.root / "terminal_snapshot"
        self.collection = self.root / "terminal_collection.json"
        self.collection_snapshot = self.root / "terminal_snapshot"
        self.patchers = [
            patch.object(
                remote,
                "output_paths",
                side_effect=lambda _arm, _operation: (self.output, self.snapshot),
            ),
            patch.object(
                remote,
                "collect_paths",
                side_effect=lambda _arm: {
                    "receipt": self.collection,
                    "snapshot": self.collection_snapshot,
                },
            ),
            patch.object(remote, "verify_collection_receipt"),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_collector_requires_both_outputs_absent_including_broken_symlink(self) -> None:
        self.assertEqual(
            remote.validate_output_state("wide", "collector"), (self.output, self.snapshot)
        )
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(GateError):
            remote.validate_output_state("wide", "collector")
        self.assertTrue(self.output.is_symlink())
        self.output.unlink()
        self.snapshot.mkdir()
        with self.assertRaises(GateError):
            remote.validate_output_state("wide", "collector")
        self.assertTrue(self.snapshot.is_dir())

    def test_verifier_requires_real_collection_inputs_and_absent_receipt(self) -> None:
        with self.assertRaises(GateError):
            remote.validate_output_state("deep", "verifier")
        self.collection.write_text("{}\n")
        self.collection_snapshot.mkdir()
        self.assertEqual(
            remote.validate_output_state("deep", "verifier"), (self.output, self.snapshot)
        )
        self.output.write_text("preserve\n")
        with self.assertRaises(GateError):
            remote.validate_output_state("deep", "verifier")
        self.assertEqual(self.output.read_text(), "preserve\n")

    def test_verifier_rejects_symlinked_collection_snapshot(self) -> None:
        self.collection.write_text("{}\n")
        self.collection_snapshot.symlink_to(self.root / "missing", target_is_directory=True)
        with self.assertRaises(GateError):
            remote.validate_output_state("wide", "verifier")
        self.assertTrue(self.collection_snapshot.is_symlink())

    def test_existing_capture_is_read_for_recovery_without_rewrite(self) -> None:
        capture_path = self.root / "CAPTURE_V1.json"
        unit = "computer-vision-homework-r01-f3-wide-terminal-collector-t12345678.service"
        self.assertIsNone(remote.read_existing_capture(capture_path, unit))
        raw = (
            '{"schema":"priority25.f3_terminal_support_capture.v1",'
            '"status":"TERMINAL_CAPTURE","unit":"' + unit + '"}\n'
        ).encode()
        capture_path.write_bytes(raw)
        result = remote.read_existing_capture(capture_path, unit)
        self.assertEqual(result["unit"], unit)
        self.assertEqual(capture_path.read_bytes(), raw)

    def test_existing_capture_wrong_identity_is_not_reused(self) -> None:
        capture_path = self.root / "CAPTURE_V1.json"
        capture_path.write_text(
            '{"schema":"priority25.f3_terminal_support_capture.v1",'
            '"status":"TERMINAL_CAPTURE","unit":"other.service"}\n'
        )
        with self.assertRaises(GateError):
            remote.read_existing_capture(
                capture_path,
                "computer-vision-homework-r01-f3-wide-terminal-collector-t12345678.service",
            )


if __name__ == "__main__":
    unittest.main()
