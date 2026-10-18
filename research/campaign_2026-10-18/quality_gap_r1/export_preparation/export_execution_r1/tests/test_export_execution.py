"""Small local tests for bounded export bindings and capture validation."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import capture
from bindings import (
    B_TMP,
    EXECUTION_DIR,
    MEMBERSHIP,
    PINS,
    captured_preinference_record_path,
    output_paths,
    require_outputs_absent,
    require_project_cache_directory,
    validate_preflight_receipt,
    validate_root_admission,
    verify_runtime_source_inventory,
    worker_preinference_record_path,
)
from capture import (
    _archive_preinference_record,
    _ordered_row_groups,
    _read_npy_header,
    _validate_h390_sidecar,
)


def _npy_bytes(shape: tuple[int, ...], descr: str, values: bytes) -> bytes:
    """Build a small valid version-1 NPY payload for parser tests."""
    header = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode("latin1")
    padding = (-((10 + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + values


class ExportExecutionTests(unittest.TestCase):
    """Exercise create-only and metadata contracts without GPU or large arrays."""

    def test_actual_captured_cpu_preflight_receipt_is_semantically_bound(self) -> None:
        path = (
            MEMBERSHIP.parent
            / "preflight_execution_r2/records/preflight-r2-20261018t1520z/COMPATIBILITY_RECEIPT_R2.json"
        )
        result = validate_preflight_receipt(
            path,
            overlay_manifest=json.loads(
                (
                    MEMBERSHIP.parent / "compatibility_overlay_r2/OVERLAY_MANIFEST_R2.json"
                ).read_text()
            ),
        )
        self.assertEqual(result["status"], "PASS_CPU_LOADER_PREFLIGHT")
        self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_root_admission_binds_attempt_preflight_and_physical_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.json"
            admission = {
                "schema": "quality_gap.strong_control_export_admission.r1",
                "status": "ADMITTED_ROOT_ONE_GPU_EXPORT",
                "owner": "root",
                "attempt_id": "control-export-r1-test",
                "preflight_receipt_sha256": "a" * 64,
                "one_gpu_only": True,
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "created_utc": datetime.now(UTC).isoformat(),
                "resource_check": {
                    "aggregate_cpu_cores": 2,
                    "parent_memory_high_bytes": 18_000_000_000,
                    "parent_memory_max_bytes": 20_000_000_000,
                    "parent_swap_max_bytes": 0,
                    "campaign_active_gpu_claims": [],
                    "campaign_active_gpu_workers": [],
                    "physical_scratch_path": str(B_TMP),
                    "physical_scratch_free_bytes": 2_000_000_000,
                    "selected_gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                    "selected_gpu_free_bytes": 5_000_000_000,
                },
            }
            path.write_text(json.dumps(admission), encoding="utf-8")
            result = validate_root_admission(
                path,
                attempt_id="control-export-r1-test",
                preflight_sha256="a" * 64,
            )
            self.assertEqual(result["gpu_uuid"], admission["gpu_uuid"])
            admission["resource_check"]["campaign_active_gpu_workers"] = ["active campaign worker"]
            path.write_text(json.dumps(admission), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aggregate/GPU claim"):
                validate_root_admission(
                    path,
                    attempt_id="control-export-r1-test",
                    preflight_sha256="a" * 64,
                )

    def test_output_preflight_rejects_every_predictor_collision_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "cache").mkdir()
            self.assertEqual(
                set(require_outputs_absent(project)),
                {"prediction", "sidecar", "partial", "family_prediction", "stale_seqsum"},
            )
            for key in output_paths(project):
                path = output_paths(project)[key]
                path.write_bytes(b"prior evidence")
                with self.assertRaises(FileExistsError, msg=key):
                    require_outputs_absent(project)
                path.unlink()

    def test_output_directory_must_not_redirect_outside_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "external-cache"
            target.mkdir()
            (root / "cache").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                require_project_cache_directory(root)

    def test_runtime_inventory_rejects_unreviewed_module(self) -> None:
        source = EXECUTION_DIR / "unreviewed_shadow.py"
        source.write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "runtime Python inventory"):
                verify_runtime_source_inventory()
        finally:
            source.unlink()

    def test_runtime_inventory_rejects_cached_bytecode(self) -> None:
        cache = EXECUTION_DIR / "forbidden.pyc"
        cache.write_bytes(b"not executable bytecode")
        try:
            with self.assertRaisesRegex(ValueError, "cached bytecode"):
                verify_runtime_source_inventory()
        finally:
            cache.unlink()

    def test_preinference_producer_and_capture_archive_are_distinct_create_only_files(
        self,
    ) -> None:
        attempt = "control-export-preinference-lifecycle-test"
        source = worker_preinference_record_path(attempt)
        captured = captured_preinference_record_path(attempt)
        source.parent.mkdir(parents=True, exist_ok=True)
        raw = b'{"status":"PRE_INFERENCE_SETTINGS_VERIFIED"}\n'
        try:
            source.write_bytes(raw)
            archived_path, archived_raw = _archive_preinference_record(attempt)
            self.assertEqual(archived_path, captured)
            self.assertNotEqual(archived_path, source)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(archived_path.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                _archive_preinference_record(attempt)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(archived_raw, raw)
        finally:
            source.unlink(missing_ok=True)
            captured.unlink(missing_ok=True)
            source.parent.rmdir()

    def test_capture_main_writes_full_receipt_without_recreating_worker_record(self) -> None:
        records_root = EXECUTION_DIR / "records"
        records_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capture-main-lifecycle-", dir=records_root
        ) as attempt_dir_name:
            attempt_dir = Path(attempt_dir_name)
            attempt = attempt_dir.name
            project = attempt_dir / "project"
            cache = project / "cache"
            cache.mkdir(parents=True)
            project_outputs = output_paths(project)
            project_outputs["prediction"].write_bytes(b"bounded fixture")
            project_outputs["sidecar"].write_text("{}\n", encoding="utf-8")

            contract = {
                "attempt_id": attempt,
                "project_root": str(project),
                "root_admission": {"path": str(attempt_dir / "admission.json")},
                "preflight_receipt": {
                    "sha256": "a" * 64,
                    "status": "PASS_CPU_LOADER_PREFLIGHT",
                },
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "pins": {"predictor_sha256": "b" * 64},
                "execution": {"command": ["fixture"], "pythonpath": "overlay:source"},
                "unverified_gates": ["runtime is unmeasured"],
            }
            contract_path = attempt_dir / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            attempt_record = {
                "id": attempt,
                "status": "finished",
                "phase": "terminal",
                "exit_status": 0,
                "failure": None,
                "unit": f"wnn-bandit-{attempt}.service",
                "elapsed_seconds": 0.1,
                "reservation": {"cpus": 1, "ram_bytes": 8_000_000_000},
                "final_memory_event_deltas": {},
                "final_parent_memory_event_deltas": {},
                "final_parent_hierarchical_memory_event_deltas": {},
            }
            attempt_record_path = attempt_dir / "fake_attempt.json"
            raw_attempt = json.dumps(attempt_record).encode() + b"\n"
            attempt_record_path.write_bytes(raw_attempt)
            preinference = {
                "schema": "quality_gap.strong_control_pre_inference_runtime.r1",
                "status": "PRE_INFERENCE_SETTINGS_VERIFIED",
                "attempt_id": attempt,
                "cuda_visible_devices": contract["gpu_uuid"],
                "wnn_gpu_device": contract["gpu_uuid"],
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "float32_matmul_precision": "highest",
                "nvidia_tf32_override": "0",
                "inference_precision": "fp32",
                "torch_version": "fixture-torch",
                "torch_cuda_version": "fixture-cuda",
            }
            raw_preinference = json.dumps(preinference).encode() + b"\n"
            worker_preinference_record_path(attempt).write_bytes(raw_preinference)
            synthetic_outputs = {
                "prediction": {
                    "path": str(project_outputs["prediction"]),
                    "bytes": 15,
                    "sha256": "c" * 64,
                },
                "sidecar": {
                    "path": str(project_outputs["sidecar"]),
                    "bytes": 3,
                    "sha256": "d" * 64,
                },
                "row_groups": {
                    "count": 1873,
                    "ordered_int64_le_sha256": "e" * 64,
                    "first": 0,
                    "last": 1872,
                },
            }
            args = [
                "capture.py",
                "--contract",
                str(contract_path),
                "--attempt-record",
                str(attempt_record_path),
            ]
            with (
                mock.patch.object(sys, "argv", args),
                mock.patch.object(capture, "build_contract", return_value=contract),
                mock.patch.object(capture, "_verify_guard_attempt"),
                mock.patch.object(capture, "_check_outputs", return_value=synthetic_outputs),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(capture.main(), 0)

            source_path = worker_preinference_record_path(attempt)
            archive_path = captured_preinference_record_path(attempt)
            receipt = json.loads((attempt_dir / "EXPORT_RECEIPT.json").read_text())
            self.assertEqual(source_path.read_bytes(), raw_preinference)
            self.assertEqual(archive_path.read_bytes(), raw_preinference)
            self.assertEqual(receipt["pre_inference_runtime"]["path"], str(source_path))
            self.assertEqual(
                receipt["captured_raw_evidence"]["pre_inference_runtime"]["path"],
                str(archive_path),
            )
            self.assertEqual((attempt_dir / "BANDIT_ATTEMPT.json").read_bytes(), raw_attempt)
            self.assertEqual((attempt_dir / "PREDICTION_SIDECAR.json").read_bytes(), b"{}\n")
            with (
                mock.patch.object(sys, "argv", args),
                mock.patch.object(capture, "build_contract", return_value=contract),
                redirect_stdout(io.StringIO()),
                self.assertRaises(FileExistsError),
            ):
                capture.main()
            self.assertEqual(archive_path.read_bytes(), raw_preinference)

    def test_row_group_reader_touches_one_bounded_npz_member(self) -> None:
        groups = list(range(1873))
        raw = struct.pack("<1873q", *groups)
        npy = _npy_bytes((1873,), "<i8", raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid_targets.npz"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("row_group.npy", npy)
                archive.writestr("targets.npy", b"unread-target-member")
            loaded, digest = _ordered_row_groups(path)
        self.assertEqual(loaded, groups)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_h390_sidecar_requires_its_actual_feature_only_fields(self) -> None:
        groups = list(range(1873))
        digest = "b" * 64
        contract = {
            "artifacts": {"control_checkpoint": {"sha256": "c" * 64}},
        }
        sidecar = {
            "run": "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1",
            "split": "valid",
            "epoch": 5,
            "checkpoint_sha256": "c" * 64,
            "prediction_sha256": digest,
            "split_manifest_sha256": PINS["old_split_sha256"],
            "quote_cache": None,
            "gate_mode": "soft",
            "inference_precision": "fp32",
            "quantization_preset": None,
            "recurrent_rank": None,
            "sentinel_preprocessing": None,
            "numeric_preprocessing": None,
            "row_groups": groups,
            "evaluation_membership_sha256": None,
            "fit_subset_manifest_sha256": None,
            "campaign_split_manifest_sha256": PINS["old_split_sha256"],
            "campaign_fit_scope": "full-valid-feature-only",
            "campaign_evaluation_membership_sha256": None,
        }
        _validate_h390_sidecar(contract, sidecar, groups, digest)
        sidecar["campaign_export_scope"] = "feature_only"
        with self.assertRaisesRegex(ValueError, "field inventory"):
            _validate_h390_sidecar(contract, sidecar, groups, digest)

    def test_npy_header_checks_shape_and_dtype_without_loading_array_body(self) -> None:
        encoded = _npy_bytes((1873, 20000, 2), "<f4", b"")
        shape, dtype, fortran, offset = _read_npy_header(io.BytesIO(encoded))
        self.assertEqual(shape, (1873, 20000, 2))
        self.assertEqual(dtype, "<f4")
        self.assertFalse(fortran)
        self.assertEqual(offset, len(encoded))


if __name__ == "__main__":
    unittest.main()
