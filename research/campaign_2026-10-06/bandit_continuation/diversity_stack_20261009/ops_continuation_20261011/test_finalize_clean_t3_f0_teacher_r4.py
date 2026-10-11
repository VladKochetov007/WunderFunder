"""Offline private-copy tests for the R4 create-only release finalizer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

OPS = Path(__file__).resolve().parent
ROOT = OPS.parents[4]
R3_PACKET = ROOT / Path(
    "research",
    "campaign_2026-10-06",
    "bandit_continuation",
    "diversity_stack_20261009",
    "ops_continuation_20261011",
    "clean_t3_fold0_h390_r3_teacher_cohort",
)
FINALIZER_PATH = OPS / "finalize_clean_t3_f0_teacher_r4.py"
SPEC = importlib.util.spec_from_file_location("r4_finalizer_under_test", FINALIZER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the R4 finalizer")
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)


def authorized_receipt() -> dict:
    """Make a test-only authority shape; it is never used outside temp paths."""
    return {
        "schema": FINALIZER.ROOT_SCHEMA,
        "status": "admitted",
        "authorized": True,
        "owner": "test_fixture_only_not_root_authority",
        "target": {
            "dataset": FINALIZER.R4_DATASET,
            "kernel_id": f"{FINALIZER.R4_DATASET}-runner",
        },
        "allowed_delta": FINALIZER.EXPECTED_ALLOWED_DELTA,
        "binding": {
            "preseal_r3": FINALIZER.r3_packet_bindings(R3_PACKET),
            "previous_profile_collection": {"file_count": 77, "sha256": "a" * 64},
            "quota_budget": {
                "wall_ceiling_seconds": 14_400,
                "two_device_gpu_seconds_upper_bound": 28_800,
                "reserved_gpu_seconds": 28_800,
                "paid_spending_authorized": False,
                "campaign_headroom_gpu_seconds": 28_800,
                "provider_headroom_gpu_seconds": 28_800,
            },
        },
    }


class R4FinalizerTests(unittest.TestCase):
    """Exercise refusal and successful packaging only in temporary copies."""

    def test_pending_receipt_refuses_without_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wnn-r4-pending-test-") as temp_name:
            temp = Path(temp_name)
            receipt = authorized_receipt()
            receipt["status"] = "pending"
            receipt["authorized"] = False
            receipt_path = temp / "pending.json"
            receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
            receipt_path.write_bytes(receipt_bytes)
            output = temp / "r4"
            stage = temp / "short-stage"
            with self.assertRaises(FINALIZER.FinalizationError):
                FINALIZER.finalize(
                    R3_PACKET,
                    receipt_path,
                    FINALIZER.sha256_bytes(receipt_bytes),
                    output,
                    stage,
                )
            self.assertFalse(output.exists())
            self.assertFalse(stage.exists())

    def test_nonfinite_or_boolean_quota_headroom_refuses(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wnn-r4-invalid-quota-test-") as temp_name:
            temp = Path(temp_name)
            receipt_path = temp / "root-release.json"
            packet_bindings = FINALIZER.r3_packet_bindings(R3_PACKET)
            invalid_values = (float("nan"), float("inf"), float("-inf"), True)
            for field_name in (
                "campaign_headroom_gpu_seconds",
                "provider_headroom_gpu_seconds",
            ):
                for value in invalid_values:
                    with self.subTest(field=field_name, value=value):
                        receipt = authorized_receipt()
                        receipt["binding"]["quota_budget"][field_name] = value
                        receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
                        receipt_path.write_bytes(receipt_bytes)
                        with self.assertRaises(FINALIZER.FinalizationError):
                            FINALIZER.validate_root_release(
                                receipt_path,
                                FINALIZER.sha256_bytes(receipt_bytes),
                                packet_bindings,
                            )

    def test_authorized_shape_reseals_private_copy_only(self) -> None:
        source_archive = R3_PACKET / "bundle_teacher_cohort_r3/bundle.wnn"
        source_gate = R3_PACKET / FINALIZER.R3_GATE_NAME
        archive_before = FINALIZER.sha256_file(source_archive)
        gate_before = FINALIZER.sha256_file(source_gate)
        with tempfile.TemporaryDirectory(prefix="wnn-r4-admitted-test-") as temp_name:
            temp = Path(temp_name)
            receipt = authorized_receipt()
            receipt_path = temp / "root-release-test-fixture.json"
            receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
            receipt_path.write_bytes(receipt_bytes)
            output = temp / FINALIZER.R4_NAME
            stage = temp / "short-upload-stage"
            result = FINALIZER.finalize(
                R3_PACKET,
                receipt_path,
                FINALIZER.sha256_bytes(receipt_bytes),
                output,
                stage,
            )
            checked = FINALIZER.validate_r4(
                output, FINALIZER.sha256_bytes(receipt_bytes), R3_PACKET
            )
            bootstrap_path = output / "bundle_teacher_cohort_r3/kernel/kernel.py"
            bootstrap_namespace = {
                "__name__": "r4_generated_bootstrap_test",
                "__file__": str(bootstrap_path),
            }
            exec(
                compile(
                    bootstrap_path.read_text(encoding="utf-8"),
                    str(bootstrap_path),
                    "exec",
                ),
                bootstrap_namespace,
            )
            require_headroom = bootstrap_namespace["_require_finite_quota_headroom"]
            self.assertIsNone(require_headroom(28_800, "integer test"))
            self.assertIsNone(require_headroom(28_800.0, "float test"))
            for value in (float("nan"), float("inf"), float("-inf"), True):
                with (
                    self.subTest(bootstrap_value=value),
                    self.assertRaises(PermissionError),
                ):
                    require_headroom(value, "test field")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(checked["worker_spec_count"], 14)
            self.assertEqual(checked["dataset"], FINALIZER.R4_DATASET)
            self.assertTrue((stage / "dataset/bundle.wnn").is_file())
            self.assertTrue((stage / "kernel/kernel.py").is_file())
            self.assertFalse((output / "ROOT_FINAL_ADMISSION_R4.json").exists())
        self.assertEqual(FINALIZER.sha256_file(source_archive), archive_before)
        self.assertEqual(FINALIZER.sha256_file(source_gate), gate_before)


if __name__ == "__main__":
    unittest.main()
