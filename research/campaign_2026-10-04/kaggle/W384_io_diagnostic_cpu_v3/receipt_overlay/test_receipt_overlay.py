from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from receipt_overlay import (
    F_GET_SEALS,
    REQUIRED_SEALS,
    DatasetBinding,
    MountIdentity,
    ReceiptError,
    ReceiptOverlay,
    ReceiptRequest,
    _memfd_create,
    build_receipt_set,
    install_hash_overlay,
    load_input_manifest,
    read_sealed_receipts,
    validate_receipt_entry,
)

KERNEL_ID = "vladkochetov07/wnn-receipt-fixture-v1-runner"
VERSION_MAP = {"owner/private-fixture": 1}


def prototype_sha256() -> str:
    source = Path(__file__).with_name("receipt_overlay.py")
    return hashlib.sha256(source.read_bytes()).hexdigest()


class ReceiptOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "source.bin"
        self.content = b"immutable fixture content\n"
        self.path.write_bytes(self.content)
        self.sha256 = hashlib.sha256(self.content).hexdigest()
        self.binding = DatasetBinding("owner/private-fixture", 1, self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def request(
        self, path: Path | None = None, expected_sha256: str | None = None
    ) -> ReceiptRequest:
        source = path or self.path
        return ReceiptRequest(
            path=source,
            expected_sha256=expected_sha256 or self.sha256,
            expected_size=len(self.content),
            dataset=self.binding,
            logical_paths=(str(source),),
        )

    def mount(self, path: Path) -> MountIdentity:
        return MountIdentity(77, str(self.root), path.stat().st_dev, True)

    def build(self, requests: list[ReceiptRequest] | None = None):
        return build_receipt_set(
            KERNEL_ID,
            VERSION_MAP,
            requests or [self.request()],
            mount_probe=self.mount,
            allowed_mount_roots=(Path("/tmp"),),
        )

    def test_full_hash_receipt_reuses_only_matching_held_readonly_fd(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            self.assertEqual(overlay.sha256_file(self.path), self.sha256)
            self.assertEqual(overlay.reused_count, 1)
            self.assertEqual(overlay.fallback_count, 0)
            self.assertEqual(len(receipt_set.pass_fds()), 2)
        finally:
            receipt_set.close()

    def test_receipt_memfd_is_sealed_against_write(self) -> None:
        receipt_set = self.build()
        try:
            self.assertEqual(
                fcntl.fcntl(receipt_set.manifest_fd, F_GET_SEALS) & REQUIRED_SEALS,
                REQUIRED_SEALS,
            )
            with self.assertRaises(OSError):
                os.write(receipt_set.manifest_fd, b"tamper")
        finally:
            receipt_set.close()

    def test_child_manifest_reads_do_not_share_or_mutate_memfd_offset(self) -> None:
        receipt_set = self.build()
        try:
            os.lseek(receipt_set.manifest_fd, 7, os.SEEK_SET)
            entries = read_sealed_receipts(
                receipt_set.manifest_fd,
                KERNEL_ID,
                VERSION_MAP,
                receipt_set.root_manifest_sha256,
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(os.lseek(receipt_set.manifest_fd, 0, os.SEEK_CUR), 7)
        finally:
            receipt_set.close()

    def test_receipt_and_source_fds_survive_child_exec(self) -> None:
        receipt_set = self.build()
        module_dir = Path(__file__).parent
        script = "\n".join(
            (
                "import os, sys",
                f"sys.path.insert(0, {str(module_dir)!r})",
                "from receipt_overlay import read_sealed_receipts",
                "manifest_fd, source_fd = map(int, sys.argv[1:3])",
                "entries = read_sealed_receipts(",
                "    manifest_fd, sys.argv[3], {'owner/private-fixture': 1}, sys.argv[4]",
                ")",
                "assert len(entries) == 1",
                "assert entries[0]['source_fd'] == source_fd",
                "assert os.fstat(source_fd).st_size == entries[0]['identity']['size']",
                "print('inherited-receipt-ok')",
            )
        )
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(receipt_set.manifest_fd),
                    str(receipt_set.source_fds[0]),
                    KERNEL_ID,
                    receipt_set.root_manifest_sha256,
                ],
                check=True,
                capture_output=True,
                pass_fds=receipt_set.pass_fds(),
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "inherited-receipt-ok")
        finally:
            receipt_set.close()

    def test_wrong_kernel_identity_rejects_manifest(self) -> None:
        receipt_set = self.build()
        try:
            with self.assertRaises(ReceiptError):
                read_sealed_receipts(
                    receipt_set.manifest_fd,
                    "different-kernel",
                    VERSION_MAP,
                    receipt_set.root_manifest_sha256,
                )
        finally:
            receipt_set.close()

    def test_wrong_dataset_version_rejects_manifest(self) -> None:
        receipt_set = self.build()
        try:
            with self.assertRaises(ReceiptError):
                read_sealed_receipts(
                    receipt_set.manifest_fd,
                    KERNEL_ID,
                    {"owner/private-fixture": 2},
                    receipt_set.root_manifest_sha256,
                )
        finally:
            receipt_set.close()

    def test_wrong_root_manifest_sha_rejects_manifest(self) -> None:
        receipt_set = self.build()
        try:
            with self.assertRaises(ReceiptError):
                read_sealed_receipts(
                    receipt_set.manifest_fd,
                    KERNEL_ID,
                    VERSION_MAP,
                    "0" * 64,
                )
        finally:
            receipt_set.close()

    def test_unsealed_manifest_is_rejected(self) -> None:
        fd = _memfd_create("unsealed-fixture")
        try:
            os.write(
                fd,
                json.dumps(
                    {
                        "schema_version": 1,
                        "kernel_id": KERNEL_ID,
                        "dataset_versions": VERSION_MAP,
                        "entries": [],
                    }
                ).encode(),
            )
            os.lseek(fd, 0, os.SEEK_SET)
            with self.assertRaises(ReceiptError):
                read_sealed_receipts(fd, KERNEL_ID, VERSION_MAP, "0" * 64)
        finally:
            os.close(fd)

    def test_expected_hash_mismatch_fails_receipt_creation(self) -> None:
        with self.assertRaises(ReceiptError):
            self.build([self.request(expected_sha256="0" * 64)])

    def test_writable_mount_fails_receipt_creation(self) -> None:
        with self.assertRaises(ReceiptError):
            build_receipt_set(
                KERNEL_ID,
                VERSION_MAP,
                [self.request()],
                mount_probe=lambda path: MountIdentity(
                    77, str(self.root), path.stat().st_dev, False
                ),
                allowed_mount_roots=(Path("/tmp"),),
            )

    def test_path_outside_bound_dataset_mount_fails(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside"
        outside.write_bytes(self.content)
        try:
            with self.assertRaises(ReceiptError):
                self.build([self.request(outside)])
        finally:
            outside.unlink()

    def test_conflicting_descriptor_hashes_for_same_file_fail(self) -> None:
        with self.assertRaises(ReceiptError):
            self.build([self.request(), self.request(expected_sha256="1" * 64)])

    def test_same_size_content_mutation_falls_back_to_full_hash(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            self.path.write_bytes(b"Immutable fixture content\n")
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            observed = overlay.sha256_file(self.path)
            self.assertEqual(observed, hashlib.sha256(self.path.read_bytes()).hexdigest())
            self.assertNotEqual(observed, self.sha256)
            self.assertEqual(overlay.reused_count, 0)
            self.assertEqual(overlay.fallback_count, 1)
            self.assertIn("file_identity_changed", overlay.fallback_reasons)
        finally:
            receipt_set.close()

    def test_metadata_only_change_invalidates_held_receipt(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd,
                KERNEL_ID,
                VERSION_MAP,
                receipt_set.root_manifest_sha256,
            )
            before = self.path.stat()
            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
            valid, reason = validate_receipt_entry(
                entries[0],
                self.path,
                self.sha256,
                KERNEL_ID,
                VERSION_MAP,
                self.mount,
                (Path("/tmp"),),
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "file_identity_changed")
        finally:
            receipt_set.close()

    def test_replaced_inode_falls_back_to_full_hash(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            replacement = self.root / "replacement.bin"
            replacement.write_bytes(b"another fixture object\n")
            replacement.replace(self.path)
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            self.assertEqual(
                overlay.sha256_file(self.path), hashlib.sha256(self.path.read_bytes()).hexdigest()
            )
            self.assertEqual(overlay.reused_count, 0)
            self.assertIn("file_identity_changed", overlay.fallback_reasons)
        finally:
            receipt_set.close()

    def test_symlink_retarget_never_reuses_receipt(self) -> None:
        alias = self.root / "alias.bin"
        alias.symlink_to(self.path)
        receipt_set = self.build([self.request(alias)])
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            other = self.root / "other.bin"
            other.write_bytes(self.content)
            alias.unlink()
            alias.symlink_to(other)
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            self.assertEqual(overlay.sha256_file(alias), self.sha256)
            self.assertEqual(overlay.reused_count, 0)
            self.assertEqual(overlay.fallback_count, 1)
            self.assertIn("no_exact_receipt", overlay.fallback_reasons)
        finally:
            receipt_set.close()

    def test_writable_inherited_fd_is_rejected(self) -> None:
        receipt_set = self.build()
        writable_fd = os.open(self.path, os.O_RDWR)
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            tampered = dict(entries[0], source_fd=writable_fd)
            valid, reason = validate_receipt_entry(
                tampered,
                self.path,
                self.sha256,
                KERNEL_ID,
                VERSION_MAP,
                self.mount,
                (Path("/tmp"),),
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "source_fd_not_readonly")
        finally:
            os.close(writable_fd)
            receipt_set.close()

    def test_closed_inherited_fd_falls_back_to_full_hash(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd,
                KERNEL_ID,
                VERSION_MAP,
                receipt_set.root_manifest_sha256,
            )
            os.close(receipt_set.source_fds[0])
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            self.assertEqual(overlay.sha256_file(self.path), self.sha256)
            self.assertEqual(overlay.reused_count, 0)
            self.assertIn("receipt_validation_error", overlay.fallback_reasons)
        finally:
            receipt_set.close()

    def test_mount_change_falls_back_to_full_hash(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                lambda path: MountIdentity(88, str(self.root), path.stat().st_dev, True),
                (Path("/tmp"),),
            )
            self.assertEqual(overlay.sha256_file(self.path), self.sha256)
            self.assertEqual(overlay.reused_count, 0)
            self.assertIn("mount_identity_or_readonly_changed", overlay.fallback_reasons)
        finally:
            receipt_set.close()

    def test_writable_remount_invalidates_receipt(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd,
                KERNEL_ID,
                VERSION_MAP,
                receipt_set.root_manifest_sha256,
            )
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                lambda path: MountIdentity(77, str(self.root), path.stat().st_dev, False),
                (Path("/tmp"),),
            )
            self.assertEqual(overlay.sha256_file(self.path), self.sha256)
            self.assertEqual(overlay.reused_count, 0)
            self.assertEqual(overlay.fallback_count, 1)
            self.assertIn("mount_identity_or_readonly_changed", overlay.fallback_reasons)
        finally:
            receipt_set.close()

    def test_changed_mount_device_invalidates_receipt_creation(self) -> None:
        with self.assertRaises(ReceiptError):
            build_receipt_set(
                KERNEL_ID,
                VERSION_MAP,
                [self.request()],
                mount_probe=lambda path: MountIdentity(
                    77, str(self.root), path.stat().st_dev + 1, True
                ),
                allowed_mount_roots=(Path("/tmp"),),
            )

    def test_swapped_held_fd_target_is_rejected(self) -> None:
        receipt_set = self.build()
        other = self.root / "other.bin"
        other.write_bytes(self.content)
        other_fd = os.open(other, os.O_RDONLY)
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd,
                KERNEL_ID,
                VERSION_MAP,
                receipt_set.root_manifest_sha256,
            )
            tampered = dict(entries[0], source_fd=other_fd)
            valid, reason = validate_receipt_entry(
                tampered,
                self.path,
                self.sha256,
                KERNEL_ID,
                VERSION_MAP,
                self.mount,
                (Path("/tmp"),),
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "file_identity_changed")
        finally:
            os.close(other_fd)
            receipt_set.close()

    def test_corrupted_receipt_digest_is_rejected(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd,
                KERNEL_ID,
                VERSION_MAP,
                receipt_set.root_manifest_sha256,
            )
            tampered = dict(entries[0], observed_sha256="0" * 64)
            valid, reason = validate_receipt_entry(
                tampered,
                self.path,
                self.sha256,
                KERNEL_ID,
                VERSION_MAP,
                self.mount,
                (Path("/tmp"),),
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "expected_digest_mismatch")
        finally:
            receipt_set.close()

    def test_short_initial_hash_fails_receipt_creation(self) -> None:
        with (
            patch("receipt_overlay._fd_sha256", return_value="0" * 64),
            self.assertRaises(ReceiptError),
        ):
            self.build()

    def test_duplicate_logical_alias_cannot_bind_distinct_inodes(self) -> None:
        alias = self.root / "alias.bin"
        alias.symlink_to(self.path)
        other = self.root / "other.bin"
        other.write_bytes(b"different content bytes\n")
        request = ReceiptRequest(
            path=other,
            expected_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
            expected_size=other.stat().st_size,
            dataset=self.binding,
            logical_paths=(str(alias),),
        )
        with self.assertRaises(ReceiptError):
            self.build([self.request(alias), request])

    def test_overlay_installs_only_after_exact_module_pin(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            module_path = self.root / "verifier.py"
            module_path.write_text("pinned verifier fixture")
            module_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
            calls = []
            module = SimpleNamespace(
                __file__=str(module_path),
                sha256_file=lambda path: calls.append(str(path)) or "strict",
            )
            uninstall = install_hash_overlay(overlay, [module], {str(module_path): module_hash})
            try:
                self.assertEqual(module.sha256_file(self.path), self.sha256)
                self.assertEqual(calls, [])
            finally:
                uninstall()
            self.assertEqual(module.sha256_file(self.path), "strict")
            self.assertEqual(calls, [str(self.path)])
        finally:
            receipt_set.close()

    def test_unpinned_module_is_not_patched(self) -> None:
        receipt_set = self.build()
        try:
            entries = read_sealed_receipts(
                receipt_set.manifest_fd, KERNEL_ID, VERSION_MAP, receipt_set.root_manifest_sha256
            )
            overlay = ReceiptOverlay(
                entries,
                KERNEL_ID,
                VERSION_MAP,
                mount_probe=self.mount,
                allowed_mount_roots=(Path("/tmp"),),
            )
            module = SimpleNamespace(__file__=str(self.path), sha256_file=lambda _path: "strict")
            with self.assertRaises(ReceiptError):
                install_hash_overlay(overlay, [module], {})
        finally:
            receipt_set.close()

    def test_input_manifest_binds_version_hash_and_experiment_alias(self) -> None:
        manifest = {
            "schema_version": 1,
            "proposed_kernel_id": KERNEL_ID,
            "provider_dataset_versions": VERSION_MAP,
            "overlay_prototype": {"sha256": prototype_sha256()},
            "sources": [
                {
                    "dataset_slug": "owner/private-fixture",
                    "dataset_version": 1,
                    "dataset_mount_root": "/kaggle/input/datasets/owner/private-fixture",
                    "filename": "source.bin",
                    "expected_size": len(self.content),
                    "expected_sha256": self.sha256,
                    "logical_paths": ["datasets/source.bin"],
                }
            ],
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["root_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = self.root / "input_manifest.json"
        path.write_text(json.dumps(manifest))
        requests = load_input_manifest(
            path, self.root, KERNEL_ID, VERSION_MAP, manifest["root_manifest_sha256"]
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].expected_sha256, self.sha256)
        self.assertEqual(
            requests[0].path,
            Path("/kaggle/input/datasets/owner/private-fixture/source.bin"),
        )
        self.assertEqual(requests[0].logical_paths, (str(self.root / "datasets/source.bin"),))

    def test_input_manifest_rejects_modified_fields_and_wrong_version(self) -> None:
        manifest = {
            "schema_version": 1,
            "proposed_kernel_id": KERNEL_ID,
            "provider_dataset_versions": VERSION_MAP,
            "overlay_prototype": {"sha256": prototype_sha256()},
            "sources": [
                {
                    "dataset_slug": "owner/private-fixture",
                    "dataset_version": 1,
                    "dataset_mount_root": "/kaggle/input/datasets/owner/private-fixture",
                    "filename": "source.bin",
                    "expected_size": len(self.content),
                    "expected_sha256": self.sha256,
                    "logical_paths": ["datasets/source.bin"],
                }
            ],
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["root_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = self.root / "input_manifest.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ReceiptError):
            load_input_manifest(
                path, self.root, "other-kernel", VERSION_MAP, manifest["root_manifest_sha256"]
            )
        with self.assertRaises(ReceiptError):
            load_input_manifest(
                path,
                self.root,
                KERNEL_ID,
                {"owner/private-fixture": 2},
                manifest["root_manifest_sha256"],
            )
        manifest["overlay_prototype"]["sha256"] = "0" * 64
        canonical = json.dumps(
            {key: value for key, value in manifest.items() if key != "root_manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        manifest["root_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ReceiptError):
            load_input_manifest(
                path, self.root, KERNEL_ID, VERSION_MAP, manifest["root_manifest_sha256"]
            )
        manifest["overlay_prototype"]["sha256"] = prototype_sha256()
        canonical = json.dumps(
            {key: value for key, value in manifest.items() if key != "root_manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        manifest["root_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        manifest["sources"][0]["expected_size"] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ReceiptError):
            load_input_manifest(
                path, self.root, KERNEL_ID, VERSION_MAP, manifest["root_manifest_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
