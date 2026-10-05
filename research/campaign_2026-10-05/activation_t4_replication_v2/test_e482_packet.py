"""Bounded CPU-only checks for the unlaunched E482 transport packet."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import e482_remote as remote
import prepare_e482 as prepare


class E482PacketTests(unittest.TestCase):
    """Exercise immutable arm binding and failure-closed pre-fit gates."""

    def test_real_worker_cli_preserves_output_for_both_phases_without_cuda(self) -> None:
        for phase in ("probe", "fit"):
            for optimizer in remote.OPTIMIZERS:
                output = Path("/kaggle/working/e482-fixture") / phase / optimizer
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("e482_remote.py")),
                    "--worker",
                    "--phase",
                    phase,
                    "--optimizer",
                    optimizer,
                    "--output",
                    str(output),
                    "--parse-only",
                ]
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[3],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                receipt = json.loads(completed.stdout)
                self.assertEqual(receipt["status"], "worker_cli_arguments_validated_no_work")
                self.assertEqual(receipt["phase"], phase)
                self.assertEqual(receipt["optimizer"], optimizer)
                self.assertEqual(receipt["output"], str(output))

    def test_registered_cells_are_exactly_the_six_seed_271828_recipes(self) -> None:
        spec = prepare.e482_spec([], [{"path": "cohort", "sha256": prepare.COHORT_SHA256}])
        observed = {(cell["seed"], cell["optimizer"], cell["shape"]) for cell in spec["cells"]}
        expected = {
            (271828, optimizer, shape)
            for optimizer in ("adamw", "muon_adamw")
            for shape in ("baseline", "sine", "cubic")
        }
        self.assertEqual(observed, expected)
        self.assertEqual(spec["execution"]["processes"], 2)
        self.assertEqual(spec["execution"]["maximum_device_hours"], 2)
        self.assertEqual(spec["execution"]["new_admission_cutoff_utc"], "2026-10-05T07:00:00Z")
        remote.require_admission_window(datetime(2026, 10, 5, 6, 59, tzinfo=UTC))
        with self.assertRaisesRegex(ValueError, "cutoff"):
            remote.require_admission_window(datetime(2026, 10, 5, 7, 0, tzinfo=UTC))

    def test_admission_rejects_insufficient_cpu_ram_gpu_count_and_gpu_type(self) -> None:
        good_host = {
            "capacity_bytes": 32 * 1024**3,
            "available_bytes": 30 * 1024**3,
            "affinity": [0, 1, 2, 3],
        }
        devices = [
            {
                "index": 0,
                "uuid": "GPU-a",
                "name": "Tesla T4",
                "total_bytes": 16 * 1024**3,
                "used_bytes": 0,
            },
            {
                "index": 1,
                "uuid": "GPU-b",
                "name": "Tesla T4",
                "total_bytes": 16 * 1024**3,
                "used_bytes": 0,
            },
        ]
        remote.admit_workers(good_host, devices)
        with self.assertRaisesRegex(ValueError, "three available CPUs"):
            remote.admit_workers({**good_host, "affinity": [0, 1]}, devices)
        with self.assertRaisesRegex(ValueError, "host budget"):
            remote.admit_workers(
                {
                    "capacity_bytes": 15 * 1024**3,
                    "available_bytes": 15 * 1024**3,
                    "affinity": [0, 1, 2],
                },
                devices,
            )
        with self.assertRaisesRegex(ValueError, "exactly two visible T4"):
            remote.admit_workers(good_host, devices[:1])
        with self.assertRaisesRegex(ValueError, "exactly two visible T4"):
            remote.admit_workers(good_host, [devices[0], {**devices[1], "name": "A100"}])
        with self.assertRaisesRegex(ValueError, "unique T4 indices and UUIDs"):
            remote.admit_workers(good_host, [devices[0], {**devices[1], "uuid": "GPU-a"}])
        self.assertEqual(
            remote.validate_worker_device(devices, "GPU-b", 1, "Tesla T4"),
            {"uuid": "GPU-b", "name": "Tesla T4", "total_memory_bytes": 16 * 1024**3},
        )
        with self.assertRaisesRegex(ValueError, "exactly its assigned T4"):
            remote.validate_worker_device(devices, "GPU-b", 2, "Tesla T4")
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            remote.validate_worker_device(devices, "GPU-missing", 1, "Tesla T4")

    def test_probe_gate_fails_closed_for_incomplete_or_failed_optimizer(self) -> None:
        arms = [
            {
                "seed": 271828,
                "optimizer": optimizer,
                "shape": shape,
                "batch": 4,
                "chunk_rows": 256,
                "updates": 1,
                "changed_parameter_tensors": 1,
            }
            for optimizer in remote.OPTIMIZERS
            for shape in remote.SHAPES
        ]
        receipts = [
            {
                "status": "probe_passed",
                "optimizer": optimizer,
                "arms": [arm for arm in arms if arm["optimizer"] == optimizer],
                "dataset_sha256": remote.DATASET_SHA256,
                "cohort_sha256": remote.COHORT_SHA256,
                "fit_only": True,
                "test_rows_loaded": False,
                "native_parity": {
                    "optimizer": optimizer,
                    "output_state_input_gradient_parameter_gradient_and_clipped_step": "passed",
                },
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
                "source_sha256": remote.source_hashes(),
            }
            for optimizer in remote.OPTIMIZERS
        ]
        expected_sources = remote.source_hashes()
        remote.validate_probe_receipts(receipts, expected_sources=expected_sources)
        with self.assertRaisesRegex(ValueError, "source or input hashes"):
            remote.validate_probe_receipts(
                [{**receipts[0], "source_sha256": {"e481": "0" * 64}}, receipts[1]],
                expected_sources=expected_sources,
            )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            remote.validate_probe_receipts(receipts[:1])
        with self.assertRaisesRegex(ValueError, "failed"):
            remote.validate_probe_receipts([{**receipts[0], "status": "probe_failed"}, receipts[1]])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            remote.validate_probe_receipts([receipts[0], {**receipts[1], "arms": []}])

    def test_built_archive_omits_external_parquet_and_bootstrap_keeps_guards(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix="e482-packet-") as temporary:
            destination = Path(temporary) / "draft"
            receipt = prepare.build_draft(root, destination)
            bundle = destination / "bundle"
            manifest = json.loads((bundle / "bundle-manifest.json").read_text())
            with tarfile.open(bundle / "bundle.wnn", "r:gz") as archive:
                archive_entries = archive.getmembers()
                members = {member.name for member in archive_entries}
                declared_bytes = sum(
                    item["size"]
                    for item in manifest["files"]
                    if item["path"] not in manifest["external_mounts"]
                )
                spec_member = archive.getmember(manifest["spec_paths"][0])
                self.assertEqual(manifest["expanded_bytes"], declared_bytes + spec_member.size)
            self.assertNotIn("datasets/valid.parquet", members)
            self.assertIn("datasets/valid.parquet", manifest["external_mounts"])
            self.assertTrue(manifest["private"])
            self.assertTrue(receipt["contains_raw_dataset"] is False)
            bootstrap = (bundle / "kernel/kernel.py").read_text()
            self.assertIn("archive_members = seen", bootstrap)
            self.assertIn("timeout_seconds=3_500", bootstrap)
            self.assertIn("e482_remote.py", bootstrap)
            self.assertNotIn("phase_runner.py", bootstrap)
            metadata = json.loads((bundle / "kernel/kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertEqual(
                metadata["dataset_sources"], [prepare.DATASET_SLUG, prepare.INPUT_DATASET]
            )
            specification = importlib.util.spec_from_file_location(
                "e482_kernel_fixture", bundle / "kernel/kernel.py"
            )
            self.assertIsNotNone(specification)
            self.assertIsNotNone(specification.loader)
            kernel = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(kernel)
            self.assertEqual(
                kernel.EXPECTED_MANIFEST_SHA256,
                prepare.sha256_file(bundle / "bundle-manifest.json"),
            )
            allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
            external = set(manifest["external_mounts"])
            with tarfile.open(bundle / "bundle.wnn", "r:gz") as archive:
                kernel.validate_archive_members(archive.getmembers(), allowed, external)
            with tempfile.TemporaryDirectory(prefix="e482-extract-") as temporary_root:
                extraction_root = Path(temporary_root)
                kernel.validate_external_mounts(
                    extraction_root,
                    manifest["external_mounts"],
                    set(),
                    manifest["external_mounts"],
                )
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    kernel.validate_external_mounts(
                        extraction_root,
                        manifest["external_mounts"],
                        external,
                        manifest["external_mounts"],
                    )
                escaped_mount = {
                    "../escape": {
                        "dataset": prepare.INPUT_DATASET,
                        "filename": "valid.parquet",
                    }
                }
                with self.assertRaisesRegex(ValueError, "mapping differs|Unsafe"):
                    kernel.validate_external_mounts(
                        extraction_root, escaped_mount, set(), escaped_mount
                    )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                kernel.validate_archive_members([], allowed, external)
            with self.assertRaisesRegex(ValueError, "escapes"):
                kernel.validate_archive_members([tarfile.TarInfo("../escape")], allowed, external)
            symlink = tarfile.TarInfo("src/wnn/escape.py")
            symlink.type = tarfile.SYMTYPE
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                kernel.validate_archive_members([symlink], allowed, external)
            hardlink = tarfile.TarInfo("src/wnn/linked.py")
            hardlink.type = tarfile.LNKTYPE
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                kernel.validate_archive_members([hardlink], allowed, external)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                kernel.validate_archive_members(
                    [tarfile.TarInfo("src/wnn/__init__.py")] * 2, allowed, external
                )
            with tempfile.TemporaryDirectory(prefix="e482-input-") as temporary_inputs:
                input_root = Path(temporary_inputs)
                dataset_root = input_root / "wunder-funder"
                dataset_root.mkdir()
                mounted = dataset_root / "valid.parquet"
                mounted.write_text("fixture")
                self.assertEqual(
                    kernel.find_external_candidate(
                        input_root, prepare.INPUT_DATASET, "valid.parquet"
                    ),
                    mounted.resolve(),
                )
                outside = input_root / "outside.parquet"
                outside.write_text("escape")
                mounted.unlink()
                mounted.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "outside"):
                    kernel.find_external_candidate(
                        input_root, prepare.INPUT_DATASET, "valid.parquet"
                    )
                with self.assertRaisesRegex(ValueError, "Invalid external"):
                    kernel.find_external_candidate(
                        input_root, prepare.INPUT_DATASET, "../outside.parquet"
                    )


if __name__ == "__main__":
    unittest.main()
