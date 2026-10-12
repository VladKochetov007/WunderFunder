"""Focused synthetic tests for the J1 runner's source and admission guards."""

from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from j1_panel_runner_r1 import (
    MappedNpy,
    StoredTargetArchive,
    _read_npy_header,
    load_release,
    sha256_bytes,
    sha256_path,
)


class J1PanelRunnerSafetyTests(unittest.TestCase):
    """Exercise bounded parser and fail-closed source-reader behavior."""

    def test_npy_header_requires_magic_and_bounded_length(self) -> None:
        invalid_magic = io.BytesIO(b"BADMAGIC\x01\x00\x08\x00" + b" " * 8)
        with self.assertRaisesRegex(ValueError, "invalid NPY magic"):
            _read_npy_header(invalid_magic, 28)

        oversized = io.BytesIO(b"\x93NUMPY\x02\x00" + struct.pack("<I", 65_537))
        with self.assertRaisesRegex(ValueError, "outside bound"):
            _read_npy_header(oversized, 65_549)

    def test_mapped_npy_checks_full_hash_stat_and_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "member.npy"
            expected = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
            np.save(path, expected, allow_pickle=False)
            stat = path.stat()
            identity = {
                "inode": stat.st_ino,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            with MappedNpy(
                path,
                expected_sha256=sha256_path(path),
                expected_stat_identity=identity,
                expected_shape=(4, 3, 2),
                expected_dtype="<f4",
            ) as mapped:
                assert mapped.array is not None
                np.testing.assert_array_equal(mapped.array, expected)

    def test_target_archive_maps_only_stored_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "targets.npz"
            arrays = {
                "y": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "mask": np.asarray([[True, False, True], [False, True, True]]),
                "row_group": np.asarray([0, 1], dtype=np.int64),
                "seq_ix": np.asarray([18, 24], dtype=np.int64),
            }
            np.savez(path, **arrays)
            with zipfile.ZipFile(path) as archive:
                self.assertTrue(
                    all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
                )
            stat = path.stat()
            identity = {
                "inode": stat.st_ino,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            with StoredTargetArchive(path, identity, sha256_path(path)) as archive:
                mapped = archive.array("y.npy", (2, 3, 4))
                self.assertEqual(mapped.dtype, np.dtype("<f4"))
                np.testing.assert_array_equal(mapped[1], arrays["y"][1])
                del mapped

    def test_root_release_requires_exact_manifest_and_limits(self) -> None:
        manifest = {
            "frozen_comparisons": {"max_unique_weight_vectors": 11},
            "future_execution_contract": {
                "threads": 1,
                "conditional_future_resource_profile": {
                    "memory_max_bytes": 1_500_000_000,
                    "runtime_max_seconds": 120,
                    "cpu_quota": "100%",
                    "swap_bytes": 0,
                },
            },
        }
        payload = {
            "schema": "wnn.j1-root-admission.v1",
            "status": "pending",
            "manifest_sha256": "a" * 64,
            "runner_sha256": "b" * 64,
            "allow_blend_weight_optimization": True,
            "allow_model_fit_or_prediction_generation": False,
            "allow_provider_or_gpu": False,
            "phases": ["calibration", "evaluation"],
            "max_unique_weight_vectors": 11,
            "resource_limits": {
                "memory_max_bytes": 1_500_000_000,
                "runtime_max_seconds": 120,
                "cpu_quota": "100%",
                "swap_bytes": 0,
                "threads": 1,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "release.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status"):
                load_release(path, manifest, "a" * 64, "b" * 64)

            payload["status"] = "admitted_j1_exposed_f0_component_blend"
            path.write_text(json.dumps(payload), encoding="utf-8")
            release, digest = load_release(path, manifest, "a" * 64, "b" * 64)
            self.assertEqual(release["status"], payload["status"])
            self.assertEqual(digest, sha256_bytes(path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
