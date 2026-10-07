from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("e463_tpu_probe_v2", HERE / "tpu_probe_v2.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("TPU v2 probe cannot be imported")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class VerifiedSourceMaterializationTests(unittest.TestCase):
    """Exercise the private source mount resolution without importing JAX."""

    def make_fixture(self, root: Path, member_name: str) -> tuple[Path, Path]:
        input_root = root / "input"
        source_mount = (
            input_root
            / "datasets"
            / "vladkochetov07"
            / "wnn-jax-realdata-e463-tpu-20261004-v1"
        )
        source_mount.mkdir(parents=True)
        member_bytes = b"tiny fixture model source\n"
        archive_path = source_mount / "source_bundle.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(member_name, member_bytes)
        archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        manifest = {
            "private": True,
            "internet": False,
            "gpu": False,
            "tpu": True,
            "archive_sha256": archive_hash,
            "files": {
                member_name: {
                    "sha256": hashlib.sha256(member_bytes).hexdigest(),
                    "size_bytes": len(member_bytes),
                    "source_path": member_name,
                }
            },
        }
        manifest_path = source_mount / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        (input_root / "datasets" / "vladkochetov07" / "wunder-funder").mkdir()
        probe.INPUT_ROOT = input_root
        probe.SOURCE_MANIFEST_SHA256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        probe.SOURCE_ARCHIVE_SHA256 = archive_hash
        probe.DATASET = root / "working" / "source"
        probe.ASSET_ROOT = input_root / "datasets" / "vladkochetov07" / "wunder-funder"
        return manifest_path, member_bytes

    def test_verifies_and_extracts_exact_manifest_bound_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, expected_bytes = self.make_fixture(root, "src/wnn_jax/rnn.py")
            receipt = probe.materialize_verified_source()
            self.assertEqual(receipt["status"], "verified_and_materialized")
            self.assertEqual(receipt["mount_path"], str(manifest_path.parent))
            self.assertEqual(receipt["verified_file_count"], 1)
            self.assertEqual((probe.DATASET / "src/wnn_jax/rnn.py").read_bytes(), expected_bytes)
            self.assertTrue(receipt["asset_root_exists"])

    def test_rejects_parent_traversal_before_writing_outside_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root, "../escape.py")
            with self.assertRaisesRegex(ValueError, "unsafe E463 archive member"):
                probe.materialize_verified_source()
            self.assertFalse((root / "working" / "escape.py").exists())

    def test_materialized_packet_hashes_match_its_preregistration(self) -> None:
        packet = HERE / "tpu_packet_v2"
        receipt = json.loads((HERE / "tpu_prereg_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["packet_id"], "wnn-e463-real-tpu-fullshape-preflight-20261007-v2")
        self.assertEqual(
            hashlib.sha256((packet / "probe.py").read_bytes()).hexdigest(),
            receipt["packet_files"]["probe.py"],
        )
        self.assertEqual(
            hashlib.sha256((packet / "kernel-metadata.json").read_bytes()).hexdigest(),
            receipt["packet_files"]["kernel-metadata.json"],
        )
        self.assertTrue(receipt["metadata"]["is_private"])
        self.assertFalse(receipt["metadata"]["enable_internet"])
        self.assertTrue(receipt["metadata"]["enable_tpu"])


if __name__ == "__main__":
    unittest.main()
