"""Exercise exact registered cache and dataset path resolution."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

PREPARATION = Path(__file__).resolve().parent
CORE = PREPARATION / "propensity_fit_admission_remote_v3.py"
MARKER = 'PAYLOAD_B64 = "__PAYLOAD_B64__"'
RELATIVE = (
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "datasets/train.parquet",
    "datasets/valid.parquet",
)


def _load_core() -> dict[str, object]:
    source = CORE.read_text()
    if source.count(MARKER) != 1:
        raise AssertionError("remote payload marker is missing or duplicated")
    encoded = base64.b64encode(b"{}").decode("ascii")
    source = source.replace(MARKER, "PAYLOAD_B64 = " + repr(encoded), 1)
    namespace: dict[str, object] = {"__name__": "admission_test", "__builtins__": __builtins__}
    exec(compile(source, str(CORE), "exec"), namespace)
    return namespace


class ExactDataResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="propensity-data-resolution-")
        self.base = Path(self.temporary.name)
        self.b_root = self.base / "computer-vision-homework-20261018"
        self.data = self.b_root / "data"
        self.cache = self.data / "cache"
        self.receipts = self.b_root / "receipts"
        self.external = self.base / "WunderFunder"
        self.external_cache = self.external / "cache"
        self.external_datasets = self.external / "datasets"
        self.registry_path = self.b_root / "registry.json"
        self.cache.mkdir(parents=True)
        self.receipts.mkdir(parents=True)
        self.external_cache.mkdir(parents=True)
        self.external_datasets.mkdir(parents=True)
        (self.data / "datasets").symlink_to(self.external_datasets, target_is_directory=True)
        self.namespace = _load_core()
        self.namespace["B"] = self.b_root
        self.namespace["DATA"] = self.data
        self.namespace["REGISTRY"] = self.registry_path
        self._create_sources()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_sources(self) -> None:
        targets = {
            "cache/chain_folds.npz": self.receipts / "chain_folds_derived.npz",
            "cache/chains.npz": self.external_cache / "chains.npz",
            "cache/train_targets.npz": self.external_cache / "train_targets.npz",
            "cache/valid_targets.npz": self.external_cache / "valid_targets.npz",
            "datasets/train.parquet": self.external_datasets / "train.parquet",
            "datasets/valid.parquet": self.external_datasets / "valid.parquet",
        }
        sources = {}
        stats = {}
        path_map = {}
        for relative, target in targets.items():
            target.write_bytes((relative + "\n").encode())
            source = self.data / relative
            if relative == "datasets/train.parquet" or relative == "datasets/valid.parquet":
                pass  # The data/datasets parent is the registered alias.
            else:
                source.symlink_to(target)
            current = source.resolve(strict=True).stat()
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            stats[relative] = {
                "bytes": current.st_size,
                "mtime_ns": current.st_mtime_ns,
                "inode": current.st_ino,
                "registered_sha256": digest,
                "evidence": "profile identity; no admission-time payload read",
            }
            sources[relative] = {
                "sha256": digest,
                "original_identity": {
                    "bytes": current.st_size,
                    "mtime_ns": current.st_mtime_ns,
                },
            }
            path_map[relative] = {
                "path": str(source),
                "resolved": str(source.resolve(strict=True)),
                "islink": source.is_symlink(),
                "device": current.st_dev,
                "inode": current.st_ino,
                "mtime_ns": current.st_mtime_ns,
                "bytes": current.st_size,
                "registered_sha256": digest,
            }
        self.namespace["PAYLOAD"] = {
            "profile_data_device": self.b_root.stat().st_dev,
            "profile_data_stats": stats,
            "profile_data_resolved_paths": path_map,
        }
        self.registry_path.write_text(json.dumps({"sources": sources}))

    def test_exact_cache_links_and_dataset_parent_alias_pass(self) -> None:
        result = self.namespace["_data"]()
        self.assertEqual(set(result), set(RELATIVE))
        self.assertTrue(result["cache/chain_folds.npz"]["registered_symlink"])
        self.assertTrue(result["cache/chains.npz"]["registered_symlink"])
        parquet = self.data / "datasets/train.parquet"
        self.assertFalse(parquet.is_symlink())
        self.assertFalse(parquet.resolve().is_relative_to(self.b_root))
        self.assertEqual(result["datasets/train.parquet"]["resolved_path"], str(parquet.resolve()))
        self.assertFalse(result["datasets/train.parquet"]["payload_hashed"])

    def test_wrong_external_target_is_rejected_even_with_same_bytes(self) -> None:
        relative = "cache/chains.npz"
        wrong = self.external_cache / "wrong-chains.npz"
        wrong.write_bytes((relative + "\n").encode())
        source = self.data / relative
        source.unlink()
        source.symlink_to(wrong)
        with self.assertRaisesRegex(ValueError, "resolution differs from frozen map"):
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
