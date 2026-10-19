"""Exercise the registered in-Beleriand cache-link admission rule."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

PREPARATION = Path(__file__).resolve().parent
CORE = PREPARATION / "propensity_fit_admission_remote_v2.py"
MARKER = 'PAYLOAD_B64 = "__PAYLOAD_B64__"'


def _load_core() -> dict[str, object]:
    source = CORE.read_text()
    if source.count(MARKER) != 1:
        raise AssertionError("remote payload marker is missing or duplicated")
    encoded = base64.b64encode(b"{}").decode("ascii")
    source = source.replace(MARKER, "PAYLOAD_B64 = " + repr(encoded), 1)
    namespace: dict[str, object] = {"__name__": "admission_test", "__builtins__": __builtins__}
    exec(compile(source, str(CORE), "exec"), namespace)
    return namespace


class DataSymlinkAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="propensity-data-link-")
        self.base = Path(self.temporary.name)
        self.b_root = self.base / "computer-vision-homework-20261018"
        self.data = self.b_root / "data"
        self.cache = self.data / "cache"
        self.receipts = self.b_root / "receipts"
        self.registry_path = self.b_root / "registry.json"
        self.cache.mkdir(parents=True)
        self.receipts.mkdir()
        self.namespace = _load_core()
        self.namespace["B"] = self.b_root
        self.namespace["DATA"] = self.data
        self.namespace["REGISTRY"] = self.registry_path
        self._create_sources()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_sources(self) -> None:
        target = self.receipts / "chain_folds_derived.npz"
        target.write_bytes(b"sealed-derived-fold-cache\n")
        link = self.cache / "chain_folds.npz"
        link.symlink_to(Path("../../receipts/chain_folds_derived.npz"))
        regular = self.cache / "chains.npz"
        regular.write_bytes(b"registered-regular-cache\n")
        self.namespace["PAYLOAD"] = {
            "profile_data_device": self.b_root.stat().st_dev,
            "profile_data_stats": {
                "cache/chain_folds.npz": self._identity(target),
                "cache/chains.npz": self._identity(regular),
            },
        }
        registry = {
            "sources": {
                "cache/chain_folds.npz": self._registry_entry(target),
                "cache/chains.npz": self._registry_entry(regular),
            }
        }
        self.registry_path.write_text(json.dumps(registry))

    @staticmethod
    def _identity(path: Path) -> dict[str, object]:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
            "registered_sha256": digest,
            "evidence": "fresh stat; full payload validation remains inside capped profile load",
        }

    @classmethod
    def _registry_entry(cls, path: Path) -> dict[str, object]:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "sha256": digest,
            "original_identity": {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            },
        }

    def test_registered_cache_link_and_regular_file_pass_with_exact_stats(self) -> None:
        result = self.namespace["_data"]()
        link = result["cache/chain_folds.npz"]
        self.assertTrue(link["registered_symlink"])
        self.assertEqual(
            link["resolved_path"], str(self.receipts.resolve() / "chain_folds_derived.npz")
        )
        self.assertEqual(
            link["inode"],
            self.namespace["PAYLOAD"]["profile_data_stats"]["cache/chain_folds.npz"]["inode"],
        )
        self.assertFalse(link["payload_hashed"])
        self.assertFalse(result["cache/chains.npz"]["registered_symlink"])

    def test_external_symlink_target_is_rejected(self) -> None:
        outside = self.base / "outside-cache.npz"
        outside.write_bytes(b"sealed-derived-fold-cache\n")
        link = self.cache / "chain_folds.npz"
        link.unlink()
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes physical B"):
            self.namespace["_data"]()

    def test_changed_profile_identity_is_rejected(self) -> None:
        self.namespace["PAYLOAD"]["profile_data_stats"]["cache/chain_folds.npz"]["inode"] += 1
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.namespace["_data"]()

    def test_wrong_profile_device_is_rejected(self) -> None:
        self.namespace["PAYLOAD"]["profile_data_device"] += 1
        with self.assertRaisesRegex(ValueError, "size/type changed"):
            self.namespace["_data"]()


if __name__ == "__main__":
    unittest.main()
