"""Focused safety tests for mounted-input resolution and staging."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from profile_staging import assemble, copy_mounted, discover_mounted_files


class MountedDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mounts = {f"DS{index}": self.root / f"DS{index}" for index in range(1, 5)}
        for mount in self.mounts.values():
            mount.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_resolves_names_without_reading_or_guessing_slots(self) -> None:
        source = self.mounts["DS3"] / "nested" / "train.parquet.part00"
        source.parent.mkdir()
        source.write_bytes(b"part-0")
        target = self.mounts["DS1"] / "cache__train_targets.npz"
        target.write_bytes(b"targets")

        resolved, scans = discover_mounted_files(
            self.mounts, {"train.parquet.part00", "cache__train_targets.npz"}
        )

        self.assertEqual(resolved["train.parquet.part00"]["dataset_slot"], "DS3")
        self.assertEqual(
            resolved["train.parquet.part00"]["relative_path"], "nested/train.parquet.part00"
        )
        self.assertEqual(resolved["cache__train_targets.npz"]["dataset_slot"], "DS1")
        self.assertEqual(scans["DS3"]["entries"], 2)

    def test_missing_name_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            discover_mounted_files(self.mounts, {"train.parquet.part00"})

    def test_duplicate_name_fails_closed_across_mounts(self) -> None:
        (self.mounts["DS1"] / "train.parquet.part00").write_bytes(b"one")
        (self.mounts["DS4"] / "train.parquet.part00").write_bytes(b"two")

        with self.assertRaisesRegex(ValueError, "duplicate mounted basename"):
            discover_mounted_files(self.mounts, {"train.parquet.part00"})

    def test_expected_symlink_is_rejected(self) -> None:
        target = self.root / "outside"
        target.write_bytes(b"part")
        (self.mounts["DS2"] / "train.parquet.part00").symlink_to(target)

        with self.assertRaisesRegex(ValueError, "is a symlink"):
            discover_mounted_files(self.mounts, {"train.parquet.part00"})

    def test_metadata_scan_entry_limit_is_enforced(self) -> None:
        for index in range(3):
            (self.mounts["DS1"] / f"irrelevant-{index}").write_bytes(b"x")

        with (
            patch("profile_staging.MAX_MOUNT_ENTRIES", 2),
            self.assertRaisesRegex(ValueError, "entry bound"),
        ):
            discover_mounted_files(self.mounts, {"wanted.bin"})

    def test_assembly_verifies_ordered_whole_file_hash(self) -> None:
        payloads = [b"part-a", b"part-b"]
        names = ["train.parquet.part00", "train.parquet.part01"]
        for name, payload in zip(names, payloads, strict=True):
            (self.mounts["DS2"] / name).write_bytes(payload)
        expected = hashlib.sha256(b"".join(payloads)).hexdigest()
        discovered, _ = discover_mounted_files(self.mounts, set(names))

        result = assemble(
            "train",
            {
                "assembled_path": "datasets/train.parquet",
                "bytes": len(b"part-apart-b"),
                "sha256": expected,
                "parts": [{"filename": name} for name in names],
            },
            self.root / "staged",
            discovered,
        )

        self.assertEqual(Path(result["path"]).read_bytes(), b"part-apart-b")
        self.assertEqual(result["sha256"], expected)
        self.assertEqual([part["filename"] for part in result["parts"]], names)

    def test_target_copy_checks_hash(self) -> None:
        name = "cache__train_targets.npz"
        payload = b"target-index"
        (self.mounts["DS4"] / name).write_bytes(payload)
        discovered, _ = discover_mounted_files(self.mounts, {name})

        result = copy_mounted(
            "cache/train_targets.npz",
            {
                "source_filename": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            self.root / "staged",
            discovered,
        )

        self.assertEqual(Path(result["path"]).read_bytes(), payload)
        self.assertEqual(result["dataset_slot"], "DS4")

    def test_target_copy_rejects_hash_mismatch(self) -> None:
        name = "cache__train_targets.npz"
        payload = b"target-index"
        (self.mounts["DS4"] / name).write_bytes(payload)
        discovered, _ = discover_mounted_files(self.mounts, {name})

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            copy_mounted(
                "cache/train_targets.npz",
                {"source_filename": name, "bytes": len(payload), "sha256": "0" * 64},
                self.root / "staged",
                discovered,
            )


if __name__ == "__main__":
    unittest.main()
