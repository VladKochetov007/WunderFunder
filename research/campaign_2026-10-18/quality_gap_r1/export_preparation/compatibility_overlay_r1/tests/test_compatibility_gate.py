"""Small CPU tests for the isolated F0 cache compatibility gate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

OVERLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OVERLAY_ROOT / "src"))

from wnn.compatibility_f0 import (  # noqa: E402
    _F0_CANONICAL_SPLIT_SHA256,
    _F0_CURRENT_CHAIN_FOLDS_BYTES,
    _F0_CURRENT_CHAIN_FOLDS_SHA256,
    _F0_MEMBERSHIP_RECEIPT_SHA256,
    _F0_OLD_SPLIT_SHA256,
    _F0_PORTABLE_REGISTRY_SHA256,
    _load_f0_membership_receipt,
    _membership_bindings,
    _validate_f0_compatibility_receipt,
    _verify_overlay_manifest,
)
from wnn.compatibility_f0_artifacts import expected_f0_checkpoint_bindings  # noqa: E402


class CompatibilityGateTests(unittest.TestCase):
    """Verify exact approval bindings and fail-closed behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.membership_path = OVERLAY_ROOT.parent / "F0_MEMBERSHIP_COMPARISON.json"
        cls.membership_receipt, cls.membership_sha256 = _load_f0_membership_receipt(
            cls.membership_path
        )
        cls.membership_counts, cls.membership_digests = _membership_bindings(cls.membership_receipt)
        cls.overlay_sha256 = _verify_overlay_manifest()

    def valid_receipt(self) -> dict[str, object]:
        """Build an in-memory approval with every required immutable binding."""
        return {
            "schema": "quality_gap.f0_compatibility_preflight.v1",
            "status": "PASS_CPU_LOADER_PREFLIGHT",
            "old_split_manifest_sha256": _F0_OLD_SPLIT_SHA256,
            "canonical_split_manifest_sha256": _F0_CANONICAL_SPLIT_SHA256,
            "membership_comparison_receipt_sha256": self.membership_sha256,
            "portable_inputs_registry_sha256": _F0_PORTABLE_REGISTRY_SHA256,
            "overlay_manifest_sha256": self.overlay_sha256,
            "chain_folds": {
                "path": "cache/chain_folds.npz",
                "bytes": _F0_CURRENT_CHAIN_FOLDS_BYTES,
                "sha256": _F0_CURRENT_CHAIN_FOLDS_SHA256,
            },
            "artifact_bindings": expected_f0_checkpoint_bindings(),
            "membership_counts": self.membership_counts,
            "membership_ordered_sha256": self.membership_digests,
            "checks": {
                "loader_returned": True,
                "component_map_matches_feature_graph": True,
                "evaluation_membership_matches_chain_folds": True,
                "fit_and_evaluation_components_are_disjoint": True,
            },
        }

    def validate(self, receipt: dict[str, object]) -> None:
        _validate_f0_compatibility_receipt(
            receipt,
            split_sha256=_F0_OLD_SPLIT_SHA256,
            membership_sha256=self.membership_sha256,
            registry_sha256=_F0_PORTABLE_REGISTRY_SHA256,
            overlay_sha256=self.overlay_sha256,
            artifact_bindings=expected_f0_checkpoint_bindings(),
            membership_counts=self.membership_counts,
            membership_digests=self.membership_digests,
        )

    def test_real_membership_receipt_is_exact_and_overlay_is_self_consistent(self) -> None:
        self.assertEqual(self.membership_sha256, _F0_MEMBERSHIP_RECEIPT_SHA256)
        self.assertEqual(len(self.membership_counts), 6)
        self.assertEqual(len(self.overlay_sha256), 64)

    def test_artifact_bindings_pin_the_restored_control_and_propensity_files(self) -> None:
        bindings = expected_f0_checkpoint_bindings()
        self.assertEqual(
            bindings["control_checkpoint"]["restore_path"],
            "/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt",
        )
        self.assertEqual(
            bindings["propensity_checkpoint"]["restore_path"],
            "/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-propensity-v1/artifacts/runs/c20260929_prop_f0/model.pt",
        )
        self.assertEqual(
            bindings["control_checkpoint"]["sha256"],
            "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda",
        )
        self.assertEqual(
            bindings["propensity_checkpoint"]["sha256"],
            "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7",
        )

    def test_complete_cpu_preflight_receipt_passes(self) -> None:
        self.validate(self.valid_receipt())

    def test_pending_status_is_rejected(self) -> None:
        receipt = self.valid_receipt()
        receipt["status"] = "PENDING_CPU_PREFLIGHT"
        with self.assertRaisesRegex(ValueError, "not an exact PASS"):
            self.validate(receipt)

    def test_wrong_cache_or_split_identity_is_rejected(self) -> None:
        for key, value in (
            ("old_split_manifest_sha256", "0" * 64),
            ("canonical_split_manifest_sha256", "1" * 64),
        ):
            with self.subTest(key=key):
                receipt = self.valid_receipt()
                receipt[key] = value
                with self.assertRaisesRegex(ValueError, "not an exact PASS"):
                    self.validate(receipt)
        receipt = self.valid_receipt()
        receipt["chain_folds"]["sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "not an exact PASS"):
            self.validate(receipt)

    def test_receipt_must_bind_reviewed_code_and_all_loader_checks(self) -> None:
        receipt = self.valid_receipt()
        receipt["overlay_manifest_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "not an exact PASS"):
            self.validate(receipt)
        receipt = self.valid_receipt()
        receipt["artifact_bindings"]["propensity_checkpoint"]["sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "not an exact PASS"):
            self.validate(receipt)
        receipt = self.valid_receipt()
        receipt["checks"]["component_map_matches_feature_graph"] = False
        with self.assertRaisesRegex(ValueError, "not an exact PASS"):
            self.validate(receipt)

    def test_modified_membership_receipt_is_rejected(self) -> None:
        tampered = json.loads(self.membership_path.read_text(encoding="utf-8"))
        tampered["status"] = "PASS"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "membership.json"
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "does not match"):
                _load_f0_membership_receipt(path)


if __name__ == "__main__":
    unittest.main()
