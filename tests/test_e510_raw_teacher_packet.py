from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PACKET_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/e510_raw_teacher_pair"
WORKER_PATH = PACKET_DIR / "packet_worker.py"
PREPARE_PATH = PACKET_DIR / "prepare_packet.py"


def load_builder() -> ModuleType:
    """Load the pure packet builder helpers without materializing a packet."""
    spec = importlib.util.spec_from_file_location("e510_prepare_packet", PREPARE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E510 packet builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_receipt() -> dict[str, object]:
    """Return a minimal synthetic, no-fit E508 receipt for gate tests."""
    raw_files = {
        "datasets/train.parquet": {
            "expected": {
                "dataset_ref": "vladkochetov07/wunder-funder",
                "dataset_version_at_preregistration": 1,
                "dataset_version_observed_at_utc": "2026-10-07T00:11:43+00:00",
                "logical_path": "datasets/train.parquet",
                "bytes": 29_240_219_011,
                "footer_bytes": 60_437_285,
                "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
                "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
            },
            "observed": {
                "dataset_ref": "vladkochetov07/wunder-funder",
                "path": "/kaggle/input/wunder-funder/train.parquet",
                "bytes": 29_240_219_011,
                "footer_bytes": 60_437_285,
                "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
                "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
            },
            "footer_identity_matches": True,
            "full_file_sha256_matches": True,
        },
        "datasets/valid.parquet": {
            "expected": {
                "dataset_ref": "vladkochetov07/wunder-funder",
                "dataset_version_at_preregistration": 1,
                "dataset_version_observed_at_utc": "2026-10-07T00:11:43+00:00",
                "logical_path": "datasets/valid.parquet",
                "bytes": 5_615_509_128,
                "footer_bytes": 10_686_926,
                "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
                "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
            },
            "observed": {
                "dataset_ref": "vladkochetov07/wunder-funder",
                "path": "/kaggle/input/wunder-funder/valid.parquet",
                "bytes": 5_615_509_128,
                "footer_bytes": 10_686_926,
                "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
                "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
            },
            "footer_identity_matches": True,
            "full_file_sha256_matches": True,
        },
    }
    return {
        "packet": "E508_RAW_BASELINE_V3",
        "status": "profile_ready_awaiting_controller_review",
        "fit_admitted": False,
        "fit_stages_started": [],
        "split_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
        "fit_group_counts": {"train": 10_535, "valid": 1_405},
        "evaluation_valid_groups": 468,
        "target_row_counts": {"train": 10_606, "valid": 1_873},
        "profile": {
            "metrics": {"step_wall": 12.5},
            "peak_reserved_bytes": 1_500_000_000,
            "estimated_pair_fit_and_export_seconds": 3_000.0,
        },
        "hardware": {
            "cuda_available": True,
            "cuda_devices": [{"name": "Tesla T4", "total_memory": 16_000_000_000}],
        },
        "input_map": {
            "portable_registry": {
                "sha256": "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
            },
            "raw_files": raw_files,
        },
    }


def function_namespace(
    path: Path, function_name: str, globals_: dict[str, object]
) -> dict[str, object]:
    """Compile one standalone pure worker helper without importing Torch."""
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class E510PacketGateTests(unittest.TestCase):
    """Check profile admission, source mapping, and embedded receipt identity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()
        cls.receipt_bytes = (json.dumps(profile_receipt(), sort_keys=True) + "\n").encode()
        cls.receipt_sha = hashlib.sha256(cls.receipt_bytes).hexdigest()

    def test_profile_receipt_requires_matching_review_hash_and_no_fit_status(self) -> None:
        result = self.builder.validate_profile_receipt(
            self.receipt_bytes,
            expected_sha256=self.receipt_sha,
        )
        self.assertEqual(result["packet"], "E508_RAW_BASELINE_V3")
        with self.assertRaisesRegex(ValueError, "supplied digest"):
            self.builder.validate_profile_receipt(self.receipt_bytes, expected_sha256="0" * 64)

        rejected = profile_receipt()
        rejected["fit_stages_started"] = ["fit_seed_3"]
        changed = (json.dumps(rejected, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "no-fit profile"):
            self.builder.validate_profile_receipt(changed, expected_sha256=digest)

        wrong_identity = profile_receipt()
        wrong_identity["input_map"]["raw_files"]["datasets/train.parquet"][
            "full_file_sha256_matches"
        ] = False
        changed = (json.dumps(wrong_identity, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "raw input identity"):
            self.builder.validate_profile_receipt(changed, expected_sha256=digest)

        wrong_version = profile_receipt()
        wrong_version["input_map"]["raw_files"]["datasets/train.parquet"]["expected"][
            "dataset_version_at_preregistration"
        ] = 2
        changed = (json.dumps(wrong_version, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "raw input identity"):
            self.builder.validate_profile_receipt(changed, expected_sha256=digest)

    def test_profile_timing_allocator_and_group_counts_fail_closed(self) -> None:
        for mutate, message in (
            (
                lambda row: row["profile"].update(estimated_pair_fit_and_export_seconds=6_301),
                "timing",
            ),
            (lambda row: row["profile"].update(peak_reserved_bytes=4_300_000_001), "timing"),
            (lambda row: row.update(fit_group_counts={"train": 10_534, "valid": 1_405}), "fold-0"),
        ):
            row = profile_receipt()
            mutate(row)
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            digest = hashlib.sha256(encoded).hexdigest()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.builder.validate_profile_receipt(encoded, expected_sha256=digest)

    def test_packet_builder_refuses_to_materialize_without_real_profile_receipt(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate controller review receipt"):
            self.builder.build_packet()
        self.assertFalse((PACKET_DIR / "packet").exists())

    def test_generated_kernel_embeds_profile_but_no_self_attested_review(self) -> None:
        source_families = (PACKET_DIR.parent / "SOURCE_FAMILIES.json").read_bytes()
        kernel_bytes = self.builder.kernel_source(
            b"synthetic-archive",
            {"scripts/train_rnn.py": "a" * 64},
            b"print('synthetic worker')\n",
            self.receipt_bytes,
            self.receipt_sha,
            source_families,
        )
        namespace: dict[str, object] = {"__name__": "e510_kernel_test"}
        exec(compile(kernel_bytes, "kernel.py", "exec"), namespace)
        decode_source = namespace["decode_source"]
        recovered = decode_source(namespace["PROFILE_RECEIPT"], namespace["PROFILE_RECEIPT_SHA256"])
        self.assertEqual(recovered, self.receipt_bytes)
        self.assertEqual(namespace["PROFILE_RECEIPT_SHA256"], self.receipt_sha)
        self.assertIn(
            '"WNN_E508_PROFILE_RECEIPT_SHA256": PROFILE_RECEIPT_SHA256', kernel_bytes.decode()
        )
        self.assertNotIn("WNN_E508_PROFILE_REVIEWED_SHA256", kernel_bytes.decode())

    def test_separate_review_binds_real_profile_output_code_and_resource_budget(self) -> None:
        code_identities = {
            "kernel_sha256": "1" * 64,
            "worker_sha256": "2" * 64,
            "builder_sha256": "3" * 64,
            "trainer_sha256": "4" * 64,
            "source_archive_sha256": "5" * 64,
        }
        now = datetime.now(UTC)
        review = {
            "format": "wnn.controller.e510-review.v1",
            "version": 1,
            "status": "approved",
            "packet_id": "E510_RAW_TEACHER_PAIR_V1",
            "profile_review": {
                "status": "approved",
                "packet_id": "E508_RAW_BASELINE_V3",
                "receipt_sha256": self.receipt_sha,
                "terminal_output_sha256": "6" * 64,
            },
            "e510_code_review": {"status": "approved", **code_identities},
            "production_budget": {
                "status": "approved",
                "provider": "kaggle",
                "provider_watch_seconds": 7_200,
                "worker_wall_cap_seconds": 6_300,
                "worst_case_dual_t4_gpu_hours": 4.0,
                "ram_headroom_fraction": 0.2,
                "ram_monitor_required": True,
                "fresh_quota_checked": True,
                "gpu_hours_remaining": 4.5,
                "gpu_hours_reserved": 4.0,
                "quota_checked_at_utc": now.isoformat(),
            },
        }
        review_bytes = (json.dumps(review, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(review_bytes).hexdigest()
        accepted = self.builder.validate_controller_review_receipt(
            review_bytes,
            expected_review_sha256=digest,
            profile_receipt_sha256=self.receipt_sha,
            terminal_output_sha256="6" * 64,
            code_identities=code_identities,
            now_utc=now,
        )
        self.assertEqual(accepted["status"], "approved")

        for path, value in (
            (("profile_review", "terminal_output_sha256"), "7" * 64),
            (("e510_code_review", "kernel_sha256"), "8" * 64),
            (("production_budget", "ram_monitor_required"), False),
            (("production_budget", "gpu_hours_remaining"), 3.9),
        ):
            changed = json.loads(review_bytes)
            changed[path[0]][path[1]] = value
            encoded = (json.dumps(changed, sort_keys=True) + "\n").encode()
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.builder.validate_controller_review_receipt(
                    encoded,
                    expected_review_sha256=hashlib.sha256(encoded).hexdigest(),
                    profile_receipt_sha256=self.receipt_sha,
                    terminal_output_sha256="6" * 64,
                    code_identities=code_identities,
                    now_utc=now,
                )

    def test_source_manifest_checkpoint_mapping_uses_exact_logical_path(self) -> None:
        namespace = function_namespace(
            WORKER_PATH,
            "validate_audit_checkpoint_entry",
            {
                "AUDIT_CHECKPOINT_SHA256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
            },
        )
        validate_entry = namespace["validate_audit_checkpoint_entry"]
        accepted = {
            "files": {
                "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt": {
                    "sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
                    "size_bytes": 3_977_183,
                }
            }
        }
        self.assertEqual(validate_entry(accepted)["size_bytes"], 3_977_183)
        for files in (
            {
                "source_bundle/artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt": accepted[
                    "files"
                ]["artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"]
            },
            {
                "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt": {
                    **accepted["files"]["artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"],
                    "sha256": "f" * 64,
                }
            },
        ):
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, "not bound"):
                validate_entry({"files": files})

    def test_parquet_footer_hash_includes_validated_trailer(self) -> None:
        namespace = function_namespace(
            WORKER_PATH,
            "parquet_footer_identity",
            {"Path": Path, "hashlib": hashlib, "os": os},
        )
        prefix = b"parquet-data-prefix"
        footer = b"serialized row-group and schema metadata"
        trailer = len(footer).to_bytes(4, "little") + b"PAR1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            path.write_bytes(prefix + footer + trailer)
            actual = namespace["parquet_footer_identity"](path)
            self.assertEqual(actual["footer_range_bytes_including_trailer"], len(footer) + 8)
            self.assertEqual(actual["footer_range_start_offset"], len(prefix))
            self.assertEqual(actual["footer_sha256"], hashlib.sha256(footer + trailer).hexdigest())
            path.write_bytes(prefix + footer + len(footer).to_bytes(4, "little") + b"BAD!")
            with self.assertRaisesRegex(ValueError, "invalid parquet trailer"):
                namespace["parquet_footer_identity"](path)

    def test_identity_report_matches_only_when_both_hashes_match(self) -> None:
        namespace = function_namespace(WORKER_PATH, "input_identity_status", {})
        status = namespace["input_identity_status"]
        self.assertEqual(status(True, True), "matched")
        self.assertEqual(status(False, True), "mismatch")
        self.assertEqual(status(True, False), "mismatch")

    def test_worker_checks_profile_hash_before_input_inventory(self) -> None:
        tree = ast.parse(WORKER_PATH.read_text())
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls: list[str] = []
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        self.assertIn("validate_external_profile_receipt", calls)
        self.assertIn("input_map", calls)
        main_source = ast.get_source_segment(WORKER_PATH.read_text(), main)
        self.assertLess(
            main_source.index("validate_external_profile_receipt()"),
            main_source.index('mark_phase("input_inventory")'),
        )

    def test_worker_requires_continuous_twenty_percent_ram_headroom(self) -> None:
        namespace = function_namespace(
            WORKER_PATH,
            "require_memory_headroom",
            {
                "MIN_HOST_HEADROOM_BYTES": 4 * 1024**3,
                "MIN_HOST_HEADROOM_FRACTION": 0.2,
            },
        )
        require_headroom = namespace["require_memory_headroom"]
        accepted = require_headroom(
            {"memory_capacity_bytes": 32 * 1024**3, "memory_headroom_bytes": 8 * 1024**3}
        )
        self.assertEqual(accepted["required_headroom_bytes"], int(32 * 1024**3 * 0.2))
        with self.assertRaisesRegex(RuntimeError, "below required"):
            require_headroom(
                {
                    "memory_capacity_bytes": 32 * 1024**3,
                    "memory_headroom_bytes": int(32 * 1024**3 * 0.2) - 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
