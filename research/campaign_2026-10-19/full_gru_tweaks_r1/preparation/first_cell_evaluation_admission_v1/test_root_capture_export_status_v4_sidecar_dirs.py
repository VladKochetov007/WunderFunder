from __future__ import annotations

import base64
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "root_capture_export_status_v4_test", HERE / "ROOT_CAPTURE_EXPORT_STATUS_V4.py"
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


def valid_metadata() -> dict[str, dict[str, object]]:
    items = {"EXPORT_MANIFEST_V1.json": b"manifest"}
    items.update({f"{arm}/PREDICTION_SIDECAR.json": arm.encode() for arm in ARMS})
    return {name: capture(raw) for name, raw in items.items()}


class StatusV4SidecarDirectoryTests(unittest.TestCase):
    def test_creates_only_five_fixed_arm_directories_and_persists_each(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            with patch.object(STATUS, "fsync_directory", wraps=STATUS.fsync_directory) as sync:
                names = STATUS.write_terminal_metadata(record, valid_metadata())
            self.assertEqual(set(names), set(valid_metadata()))
            self.assertEqual({path.name for path in record.iterdir() if path.is_dir()}, set(ARMS))
            self.assertEqual((record / "EXPORT_MANIFEST_V1.json").read_bytes(), b"manifest")
            for arm in ARMS:
                self.assertEqual(
                    (record / arm / "PREDICTION_SIDECAR.json").read_bytes(), arm.encode()
                )
            self.assertEqual(
                [call.args[0] for call in sync.call_args_list],
                [record, *(record / arm for arm in ARMS), record],
            )

    def test_rejects_path_escape_before_creating_arm_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            metadata = valid_metadata()
            metadata["../outside/PREDICTION_SIDECAR.json"] = metadata.pop(
                "control/PREDICTION_SIDECAR.json"
            )
            with self.assertRaisesRegex(ValueError, "five fixed sidecars"):
                STATUS.write_terminal_metadata(record, metadata)
            self.assertEqual(list(record.iterdir()), [])

    def test_rejects_incomplete_or_extra_membership_before_mutation(self) -> None:
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                record = Path(temporary)
                metadata = valid_metadata()
                if mutation == "missing":
                    metadata.pop("allthree/PREDICTION_SIDECAR.json")
                else:
                    metadata["other/PREDICTION_SIDECAR.json"] = capture(b"other")
                with self.assertRaises(ValueError):
                    STATUS.write_terminal_metadata(record, metadata)
                self.assertEqual(list(record.iterdir()), [])

    def test_write_does_not_create_unenumerated_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "arbitrary"
            with self.assertRaisesRegex(ValueError, "not an enumerated real directory"):
                STATUS.write(missing / "x.json", b"x")
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
