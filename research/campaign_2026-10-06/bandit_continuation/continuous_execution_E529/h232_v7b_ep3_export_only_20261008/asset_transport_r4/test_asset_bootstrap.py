"""Offline tests for the nested private-asset payload bootstrap."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_asset_transport_packet as builder  # noqa: E402
import h232_export_asset_bootstrap as bootstrap  # noqa: E402


class AssetBootstrapTests(unittest.TestCase):
    """Check mounted asset discovery, byte identity, safe extraction and pinning."""

    def _make_fake_mount(self, input_root: Path) -> tuple[Path, bytes, dict[str, str]]:
        dataset_id = "vladkochetov07/h232-test-payload"
        directory = input_root / "nested" / "input" / dataset_id.split("/")[1]
        directory.mkdir(parents=True)
        buffer = io.BytesIO()
        member = "export_worker.py"
        member_bytes = b"def main(packet_root):\n    return None\n"
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member, member_bytes)
        payload = buffer.getvalue()
        payload_sha = hashlib.sha256(payload).hexdigest()
        filename = f"{payload_sha}.wnn"
        asset_path = directory / filename
        asset_path.write_bytes(payload)
        manifest = {
            "version": 1,
            "dataset": dataset_id,
            "private": True,
            "files": [
                {
                    "path": "payload.wnn",
                    "size": len(payload),
                    "sha256": payload_sha,
                    "filename": filename,
                }
            ],
        }
        (directory / "assets-manifest.json").write_text(json.dumps(manifest))
        return asset_path, payload, {member: hashlib.sha256(member_bytes).hexdigest()}

    def test_nested_mount_finds_and_verifies_the_exact_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary) / "input"
            input_root.mkdir()
            asset, payload, _ = self._make_fake_mount(input_root)
            result = bootstrap.find_verified_asset(
                input_root,
                "vladkochetov07/h232-test-payload",
                asset.name,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(result, asset)

    def test_modified_mount_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary) / "input"
            input_root.mkdir()
            asset, payload, _ = self._make_fake_mount(input_root)
            asset.write_bytes(payload + b"modified")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                bootstrap.find_verified_asset(
                    input_root,
                    "vladkochetov07/h232-test-payload",
                    asset.name,
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )

    def test_nested_payload_members_are_verified_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            asset, payload, members = self._make_fake_mount(input_root)
            extracted = root / "extracted"
            result = bootstrap.extract_verified_payload(
                asset,
                extracted,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                members,
            )
            self.assertEqual(result, members)
            self.assertEqual(
                (extracted / "export_worker.py").read_bytes(),
                b"def main(packet_root):\n    return None\n",
            )

    def test_unsafe_zip_paths_are_rejected_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "payload.wnn"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.py", b"unsafe")
            expected = {"../outside.py": hashlib.sha256(b"unsafe").hexdigest()}
            destination = root / "extract"
            with self.assertRaisesRegex(ValueError, "unsafe member path"):
                bootstrap.extract_verified_payload(
                    archive_path,
                    destination,
                    archive_path.stat().st_size,
                    hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    expected,
                )
            self.assertFalse(destination.exists())

    def test_built_kernel_pins_version_and_fits_source_limit(self) -> None:
        receipt_path = builder.ASSET_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R4_BUILD_RECEIPT_R6.json"
        if not receipt_path.is_file():
            self.skipTest("asset transport packet has not been built")
        receipt = json.loads(receipt_path.read_text())
        packet = Path(receipt["kernel_packet_path"])
        asset_upload = Path(receipt["asset_upload_path"])
        asset_file = asset_upload / receipt["asset_file"]
        metadata = json.loads((packet / "kernel-metadata.json").read_text())
        source = (packet / "kernel.py").read_text()
        self.assertEqual(asset_file.stat().st_size, receipt["asset_bytes"])
        self.assertEqual(
            hashlib.sha256(asset_file.read_bytes()).hexdigest(), receipt["asset_sha256"]
        )
        self.assertEqual(metadata["id"], receipt["kernel_id"])
        self.assertEqual(metadata["title"], receipt["kernel_title"])
        self.assertEqual(
            metadata["dataset_sources"][-1],
            f"{receipt['asset_dataset_id']}/{receipt['asset_dataset_version']}",
        )
        self.assertLess((packet / "kernel.py").stat().st_size, 1_000_000)
        self.assertIn(receipt["payload_zip_sha256"], source)
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(receipt["provider_kernel_push_authorized"])
        builder.validate_kernel_identity(metadata["id"], metadata["title"])


if __name__ == "__main__":
    unittest.main()
