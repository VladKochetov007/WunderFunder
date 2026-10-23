from __future__ import annotations

import base64
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "root_capture_export_status_v5_test", HERE / "ROOT_CAPTURE_EXPORT_STATUS_V5.py"
)
assert SPEC is not None and SPEC.loader is not None
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)
ARMS = ("control", "ema", "sparse", "nll", "allthree")


def capture(raw: bytes) -> dict[str, object]:
    return {
        "base64": base64.b64encode(raw).decode("ascii"),
        "bytes": len(raw),
        "sha256": STATUS.sha(raw),
    }


def complete_metadata() -> dict[str, dict[str, object]]:
    items = {"EXPORT_MANIFEST_V1.json": b"manifest"}
    items.update({f"{arm}/PREDICTION_SIDECAR.json": arm.encode() for arm in ARMS})
    return {name: capture(raw) for name, raw in items.items()}


class StatusV5TerminalMetadataTests(unittest.TestCase):
    def test_active_unit_accepts_empty_terminal_metadata_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            active = {
                "ActiveState": "active",
                "SubState": "running",
                "MainPID": "1875143",
                "ExecMainCode": "0",
                "ExecMainStatus": "0",
                "Result": "success",
            }
            self.assertFalse(STATUS.is_terminal_success(active))
            names = STATUS.write_terminal_metadata(
                record, {}, terminal_success=STATUS.is_terminal_success(active)
            )
            self.assertEqual(names, [])
            self.assertEqual(list(record.iterdir()), [])

    def test_failed_unit_accepts_empty_terminal_metadata_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            failed = {
                "ActiveState": "failed",
                "SubState": "failed",
                "MainPID": "0",
                "ExecMainCode": "1",
                "ExecMainStatus": "1",
                "Result": "exit-code",
            }
            self.assertFalse(STATUS.is_terminal_success(failed))
            names = STATUS.write_terminal_metadata(
                record, {}, terminal_success=STATUS.is_terminal_success(failed)
            )
            self.assertEqual(names, [])
            self.assertEqual(list(record.iterdir()), [])

    def test_successful_terminal_requires_and_writes_exact_six_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            metadata = complete_metadata()
            terminal_success = {
                "ActiveState": "active",
                "SubState": "exited",
                "MainPID": "0",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "Result": "success",
            }
            self.assertTrue(STATUS.is_terminal_success(terminal_success))
            names = STATUS.write_terminal_metadata(
                record, metadata, terminal_success=STATUS.is_terminal_success(terminal_success)
            )
            self.assertEqual(set(names), set(metadata))
            self.assertEqual((record / "EXPORT_MANIFEST_V1.json").read_bytes(), b"manifest")
            self.assertEqual({path.name for path in record.iterdir() if path.is_dir()}, set(ARMS))
            for arm in ARMS:
                self.assertEqual(
                    (record / arm / "PREDICTION_SIDECAR.json").read_bytes(), arm.encode()
                )

    def test_partial_terminal_metadata_is_rejected_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            metadata = complete_metadata()
            metadata.pop("allthree/PREDICTION_SIDECAR.json")
            with self.assertRaisesRegex(ValueError, "manifest and five fixed sidecars"):
                STATUS.write_terminal_metadata(record, metadata, terminal_success=True)
            self.assertEqual(list(record.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
