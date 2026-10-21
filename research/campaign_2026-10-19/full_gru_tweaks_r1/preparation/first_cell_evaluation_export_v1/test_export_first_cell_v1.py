"""Synthetic CPU-only contract fixtures for the first-cell export adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import export_first_cell_v1 as exporter
import numpy as np


class FirstCellExportContractTests(unittest.TestCase):
    def test_fixed_binding_and_source_manifest_pins(self) -> None:
        binding = exporter._load_binding(Path(__file__).resolve().parent)
        self.assertEqual(binding["data_root"], f"{binding['project_base']}/data")
        self.assertEqual(len(binding["arms"]), 5)
        self.assertEqual(binding["outputs"]["shape"], [468, 20_000, 2])
        self.assertEqual(binding["split"]["sequence_rows"], 20_000)
        self.assertEqual(binding["split"]["fit_valid_group_count"], 1_405)
        self.assertEqual(binding["split"]["evaluation_group_count"], 468)
        self.assertEqual(binding["split"]["valid_row_group_count"], 1_873)
        self.assertEqual(
            binding["registered_data"]["valid_parquet"]["resolved"],
            "/home/grinchenkoeg/WunderFunder/datasets/valid.parquet",
        )

    def test_split_inventory_derives_registered_full_valid_count(self) -> None:
        fit_valid = tuple(range(1_405))
        evaluation = tuple(range(1_405, 1_873))
        components = tuple(
            [("valid", group, group) for group in fit_valid + evaluation] + [("train", 0, 1_873)]
        )
        split = SimpleNamespace(
            valid_groups=fit_valid,
            evaluation_groups=evaluation,
            components=components,
        )
        binding = {
            "fit_valid_group_count": 1_405,
            "evaluation_group_count": 468,
            "valid_row_group_count": 1_873,
        }
        groups, valid_count = exporter._validate_split_inventory(split, binding)
        self.assertEqual(groups, list(evaluation))
        self.assertEqual(valid_count, 1_873)

    def test_local_allthree_admission_matches_registered_data_pin(self) -> None:
        binding = exporter._load_binding(Path(__file__).resolve().parent)
        receipt_path = (
            Path(__file__).resolve().parents[2]
            / "records"
            / "child_allthree_fit_v1"
            / "ADMISSION.json"
        )
        receipt_bytes = receipt_path.read_bytes()
        self.assertEqual(len(receipt_bytes), 12_912)
        self.assertEqual(
            hashlib.sha256(receipt_bytes).hexdigest(),
            binding["registered_data"]["admission"]["sha256"],
        )
        receipt = json.loads(receipt_bytes)
        self.assertEqual(receipt["schema"], "fg19-child-allthree-fit-admission.v1")
        self.assertEqual(
            receipt["data_identity_stats_only"]["datasets/valid.parquet"],
            binding["registered_data"]["valid_parquet"],
        )
        self.assertEqual(
            receipt["data_identity_stats_only"],
            binding["registered_data"]["source_identities"],
        )
        self.assertEqual(receipt["split_sha256"], binding["split"]["sha256"])

    def test_local_portable_registry_joins_all_six_admitted_identities(self) -> None:
        binding = exporter._load_binding(Path(__file__).resolve().parent)
        repository_root = Path(__file__).resolve().parents[5]
        registry_path = (
            repository_root
            / "research"
            / "campaign_2026-10-18"
            / "priority25_execution_r4"
            / "receipts"
            / "PORTABLE_SOURCES.json"
        )
        registry_bytes = registry_path.read_bytes()
        self.assertEqual(len(registry_bytes), exporter.PORTABLE_MANIFEST_BYTES)
        self.assertEqual(
            hashlib.sha256(registry_bytes).hexdigest(), exporter.PORTABLE_MANIFEST_SHA256
        )
        registry = json.loads(registry_bytes)
        exporter._validate_portable_manifest_payload(
            registry,
            binding["registered_data"]["source_identities"],
            binding["split"]["sha256"],
        )
        corrupt_registry = json.loads(registry_bytes)
        corrupt_registry["sources"]["cache/chains.npz"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            exporter._validate_portable_manifest_payload(
                corrupt_registry,
                binding["registered_data"]["source_identities"],
                binding["split"]["sha256"],
            )

    def test_six_source_stat_observer_checks_registered_aliases_without_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "project" / "data"
            target_root = root / "registered"
            source_identities: dict[str, dict[str, object]] = {}
            for index, name in enumerate(exporter.PORTABLE_SOURCE_NAMES):
                path = data_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if index < 4:
                    target = target_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"tiny synthetic source metadata")
                    path.symlink_to(target)
                else:
                    path.write_bytes(b"tiny synthetic source metadata")
                file_stat = path.stat()
                source_identities[name] = {
                    "bytes": file_stat.st_size,
                    "device": file_stat.st_dev,
                    "inode": file_stat.st_ino,
                    "islink": path.is_symlink(),
                    "mtime_ns": file_stat.st_mtime_ns,
                    "path": str(path),
                    "registered_sha256": "0" * 64,
                    "resolved": str(path.resolve()),
                }

            observed = exporter._observe_registered_source_identities(data_root, source_identities)
            self.assertEqual(list(observed), list(exporter.PORTABLE_SOURCE_NAMES))
            self.assertTrue(observed["cache/chains.npz"]["islink"])
            self.assertEqual(
                observed["datasets/valid.parquet"]["resolved"],
                source_identities["datasets/valid.parquet"]["resolved"],
            )
            changed = dict(source_identities)
            changed["cache/chains.npz"] = dict(source_identities["cache/chains.npz"])
            changed["cache/chains.npz"]["inode"] = -1
            with self.assertRaises(ValueError):
                exporter._observe_registered_source_identities(data_root, changed)

    def test_split_inventory_rejects_missing_component_group(self) -> None:
        split = SimpleNamespace(
            valid_groups=(0, 1),
            evaluation_groups=(2,),
            components=(("valid", 0, 0), ("valid", 2, 1)),
        )
        binding = {
            "fit_valid_group_count": 2,
            "evaluation_group_count": 1,
            "valid_row_group_count": 3,
        }
        with self.assertRaises(ValueError):
            exporter._validate_split_inventory(split, binding)

    def test_registered_validation_path_allows_only_exact_parent_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "project" / "data"
            data_root.mkdir(parents=True)
            target_parent = root / "registered" / "datasets"
            target_parent.mkdir(parents=True)
            registered_file = target_parent / "valid.parquet"
            registered_file.write_bytes(b"synthetic metadata-only fixture")
            (data_root / "datasets").symlink_to(target_parent, target_is_directory=True)
            path = data_root / "datasets" / "valid.parquet"
            data_stat = path.stat()
            expected = {
                "bytes": data_stat.st_size,
                "device": data_stat.st_dev,
                "inode": data_stat.st_ino,
                "islink": False,
                "mtime_ns": data_stat.st_mtime_ns,
                "path": str(path),
                "registered_sha256": "0" * 64,
                "resolved": str(registered_file.resolve()),
            }
            observation = exporter._observe_registered_data_identity(data_root, expected)
            self.assertEqual(observation["resolved"], str(registered_file.resolve()))
            self.assertEqual(observation["parent"]["resolved"], str(target_parent.resolve()))
            self.assertEqual(observation["parent"]["path"], str(data_root / "datasets"))
            wrong_target = dict(expected, resolved=str(root / "other" / "valid.parquet"))
            with self.assertRaises(ValueError):
                exporter._observe_registered_data_identity(data_root, wrong_target)
            (data_root / "datasets").unlink()
            (data_root / "datasets").symlink_to(registered_file)
            with self.assertRaises((OSError, ValueError)):
                exporter._observe_registered_data_identity(data_root, expected)

    def test_ordered_membership_encoding_and_count(self) -> None:
        groups = exporter._validate_evaluation_groups(range(468), 468)
        self.assertEqual(groups, list(range(468)))
        self.assertEqual(
            exporter._membership_sha256([1, 10]), hashlib.sha256(b"1\n10\n").hexdigest()
        )

    def test_membership_rejects_duplicate_boolean_and_fractional_ids(self) -> None:
        with self.assertRaises(ValueError):
            exporter._validate_evaluation_groups([1, 1], 2)
        with self.assertRaises(ValueError):
            exporter._validate_evaluation_groups([True], 1)
        with self.assertRaises(ValueError):
            exporter._validate_evaluation_groups([1.5], 1)

    def test_acceptance_representation_mode_binds_legacy_control_marker(self) -> None:
        exporter._check_acceptance_representation_mode(
            {"mode": None}, {"arm": "control", "representation_mode": "base"}
        )
        exporter._check_acceptance_representation_mode(
            {"mode": "ema"}, {"arm": "ema", "representation_mode": "ema"}
        )
        with self.assertRaises(ValueError):
            exporter._check_acceptance_representation_mode(
                {}, {"arm": "control", "representation_mode": "base"}
            )
        with self.assertRaises(ValueError):
            exporter._check_acceptance_representation_mode(
                {"mode": None}, {"arm": "ema", "representation_mode": "ema"}
            )

    def test_feature_reader_disables_targets_and_mask(self) -> None:
        class FakeSequenceFile:
            def __init__(self) -> None:
                self.calls: list[tuple[int, bool, bool]] = []

            def read(self, group: int, *, targets: bool, mask: bool) -> SimpleNamespace:
                self.calls.append((group, targets, mask))
                return SimpleNamespace(
                    seq_ix=101,
                    features=np.zeros((3, 2), dtype=np.float32),
                    targets=None,
                    scored=None,
                )

        source = FakeSequenceFile()
        sequence = exporter._read_feature_only(source, 17, (3, 2), np)
        self.assertEqual(source.calls, [(17, False, False)])
        self.assertEqual(sequence.seq_ix, 101)
        self.assertIsNone(sequence.targets)
        self.assertIsNone(sequence.scored)

        class InvalidSequenceIdReader:
            def read(self, group: int, *, targets: bool, mask: bool) -> SimpleNamespace:
                return SimpleNamespace(
                    seq_ix=True,
                    features=np.zeros((3, 2), dtype=np.float32),
                    targets=None,
                    scored=None,
                )

        with self.assertRaises(ValueError):
            exporter._read_feature_only(InvalidSequenceIdReader(), 17, (3, 2), np)

    def test_prediction_memmap_is_npy_readable_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.npy"
            stream, mapped, header_bytes, data_bytes = exporter._create_prediction_memmap(
                path, (2, 3, 2), np
            )
            mapped[:] = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            mapped.flush()
            del mapped
            stream.flush()
            stream.close()
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            self.assertEqual(loaded.shape, (2, 3, 2))
            self.assertEqual(loaded.dtype, np.dtype("<f4"))
            self.assertEqual(data_bytes, 48)
            self.assertEqual(path.stat().st_size, header_bytes + data_bytes)
            with self.assertRaises(FileExistsError):
                exporter._create_prediction_memmap(path, (2, 3, 2), np)

    def test_safe_child_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                exporter._safe_child(root, "../outside.json")
            with self.assertRaises(ValueError):
                exporter._safe_child(root, "/outside.json")

    def test_portable_manifest_environment_is_set_to_only_approved_registry(self) -> None:
        binding = {
            "registered_data": {
                "portable_manifest": {"absolute_path": exporter.PORTABLE_MANIFEST_PATH}
            }
        }
        with patch.dict(exporter.os.environ, {"WNN_PORTABLE_DATA_MANIFEST": ""}):
            exporter._configure_portable_manifest(binding)
            self.assertEqual(
                exporter.os.environ["WNN_PORTABLE_DATA_MANIFEST"],
                exporter.PORTABLE_MANIFEST_PATH,
            )
        with patch.dict(
            exporter.os.environ,
            {"WNN_PORTABLE_DATA_MANIFEST": exporter.PORTABLE_MANIFEST_PATH},
        ):
            exporter._configure_portable_manifest(binding)
        with (
            patch.dict(
                exporter.os.environ,
                {"WNN_PORTABLE_DATA_MANIFEST": "/tmp/unreviewed-portable-manifest.json"},
            ),
            self.assertRaises(RuntimeError),
        ):
            exporter._configure_portable_manifest(binding)

    def test_output_gate_reserves_space_for_all_five_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scratch = root / "scratch"
            evidence = scratch / "evaluation"
            evidence.mkdir(parents=True)
            output = evidence / "exports"
            binding = {
                "scratch_root": str(scratch),
                "evidence_root": str(evidence),
                "output_dir": str(output),
                "resource": {
                    "minimum_free_bytes": 500_000_000,
                    "outputs_bytes": 374_400_000,
                    "free_reserve_bytes": 1_000_000_000,
                },
            }
            with (
                patch.object(
                    exporter.shutil, "disk_usage", return_value=SimpleNamespace(free=1_374_399_999)
                ),
                self.assertRaises(RuntimeError),
            ):
                exporter._check_output_location(output, binding)
            self.assertFalse(output.exists())
            with patch.object(
                exporter.shutil, "disk_usage", return_value=SimpleNamespace(free=1_374_400_000)
            ):
                exporter._check_output_location(output, binding)
            self.assertFalse(output.exists())

    def test_json_metadata_is_create_only_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.json"
            byte_count, digest = exporter._write_json_exclusive(path, {"b": 2, "a": 1})
            raw = path.read_bytes()
            self.assertEqual(byte_count, len(raw))
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            with self.assertRaises(FileExistsError):
                exporter._write_json_exclusive(path, {"different": True})
            self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
