from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from collect_f3_checkpoint import reserve_archive
from f3_checkpoint_common import (
    arm_spec,
    expected_counts,
    is_resume_model_checkpoint,
    sha256_file,
    validate_history,
    validate_parent_headroom,
    validate_plan,
    validate_systemd_runtime_limit,
)
from f3_checkpoint_membership import validate_manifest_membership
from f3_checkpoint_semantics import validate_adam_state_descriptors, validate_pair_contracts
from f3_checkpoint_tensors import _describe_adam_state, _tensor_map_equal, _tensor_map_same_layout


def _history_fixture() -> list[dict[str, object]]:
    """Build six rows in the frozen a266 history schema (no diagnostic epoch_index)."""
    rows = []
    for epoch in range(6):
        epoch_counts = expected_counts(1)
        rows.append(
            {
                "epoch": epoch,
                "train_seconds": 1.0,
                "evaluation": "withheld",
                "update_diagnostics": {
                    "epoch": epoch_counts,
                    "cumulative": expected_counts(epoch + 1),
                    "grad_stats": {
                        "epoch": epoch,
                        "steps": 14_880,
                        "nonfinite_steps": 0,
                    },
                },
                "fit_summary": {
                    "objective_mean": 1.0,
                    "optimizer_windows": 14_880,
                    "optimizer_updates": dict(epoch_counts),
                    "learning_rate_by_group": [
                        {"group": 0, "first": 0.1, "last": 0.01, "mean": 0.05}
                    ],
                },
            }
        )
    return rows


class F3CheckpointSupportTests(unittest.TestCase):
    def test_plan_is_frozen_and_resolves_seed1001_fold3_steps(self) -> None:
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "f3_beleriand_preparation_r1"
            / "F3_BELERIAND_PLAN.json"
        )
        plan = validate_plan(plan_path)
        wide_stage1 = arm_spec(plan, "wide", "stage1")
        deep_terminal = arm_spec(plan, "deep", "terminal")
        self.assertEqual(wide_stage1["run"], "rr18_r01_wide_f3_s1001")
        self.assertEqual(
            wide_stage1["attempt"], "computer-vision-homework-r01-wide-f3s1001-ep01-v1"
        )
        self.assertEqual(deep_terminal["run"], "rr18_r01_deep_f3_s1001_e6")
        self.assertEqual((deep_terminal["hidden"], deep_terminal["layers"]), (384, 2))

    def test_wrong_or_legacy_fold_manifest_is_rejected_by_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold3.json"
            path.write_text('{"version":1,"fold":3,"n_folds":4}')
            with self.assertRaisesRegex(ValueError, "fold3 manifest SHA256"):
                validate_manifest_membership(path, {}, {})

    def test_local_legacy_fold3_manifest_is_rejected_by_sha(self) -> None:
        legacy = (
            Path(__file__).resolve().parents[4]
            / "research/campaign_2026-09-29/artifacts/splits/fold3.json"
        )
        self.assertEqual(
            sha256_file(legacy),
            "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343",
        )
        with self.assertRaisesRegex(ValueError, "fold3 manifest SHA256"):
            validate_manifest_membership(legacy, {}, {})

    def test_static_f3_counts(self) -> None:
        self.assertEqual(
            expected_counts(1),
            {
                "optimizer_attempts": 14_880,
                "amp_skipped_updates": 0,
                "optimizer_completed_updates": 14_880,
                "ema_updates": 0,
                "completed_batches": 1_488,
            },
        )
        self.assertEqual(expected_counts(2)["optimizer_completed_updates"], 29_760)
        self.assertEqual(expected_counts(6)["optimizer_completed_updates"], 89_280)

    def test_membership_group_counts_match_pinned_f3_metadata(self) -> None:
        metadata_path = (
            Path(__file__).resolve().parents[1]
            / "f3_beleriand_preparation_r1"
            / "F3_SPLIT_METADATA_V1.json"
        )
        metadata = json.loads(metadata_path.read_text())
        counts = metadata["counts"]
        self.assertEqual(
            (counts["fit_train"], counts["fit_valid"], counts["evaluation_valid"]),
            (10_539, 1_368, 505),
        )
        self.assertEqual(counts["evaluation_components"], 145)

    def test_history_matches_frozen_producer_schema(self) -> None:
        history = _history_fixture()
        self.assertNotIn("epoch_index", history[0]["update_diagnostics"])
        self.assertEqual(len(validate_history(history, 6)), 6)

    def test_history_rejects_bad_counters(self) -> None:
        history = _history_fixture()
        history[3]["fit_summary"]["optimizer_updates"]["amp_skipped_updates"] = 1
        with self.assertRaisesRegex(ValueError, "fit update dictionary"):
            validate_history(history, 6)

    def test_stage1_only_final_epoch_checkpoint_matches_resume_model(self) -> None:
        self.assertFalse(is_resume_model_checkpoint("model_ep0.pt", "stage1"))
        self.assertTrue(is_resume_model_checkpoint("model_ep1.pt", "stage1"))
        self.assertTrue(is_resume_model_checkpoint("model_ep5.pt", "terminal"))
        self.assertTrue(is_resume_model_checkpoint("model.pt", "terminal"))

    def test_pair_contract_allows_only_declared_architecture_changes(self) -> None:
        wide = {"options": {"hidden": 512, "layers": 1, "valid_fold": 3}, "runtime": {"fp32": True}}
        deep = {"options": {"hidden": 384, "layers": 2, "valid_fold": 3}, "runtime": {"fp32": True}}
        validate_pair_contracts(wide, deep)
        deep["options"]["valid_fold"] = 0
        with self.assertRaisesRegex(ValueError, "paired option differences"):
            validate_pair_contracts(wide, deep)

    def test_headroom_keeps_worker_and_collector_reserves(self) -> None:
        validate_parent_headroom(8_000_000_000, 17_000_000_000)
        with self.assertRaisesRegex(ValueError, "parent cgroup memory headroom"):
            validate_parent_headroom(8_600_000_000, 17_000_000_000)
        validate_systemd_runtime_limit("2min")
        with self.assertRaisesRegex(ValueError, "collector systemd runtime limit"):
            validate_systemd_runtime_limit("120000000")

    def test_adam_state_requires_both_moments_and_matches_model_shapes(self) -> None:
        state = {
            "state_fields": ("exp_avg", "exp_avg_sq", "step"),
            "step_numel": 1,
            "step_finite": True,
            "step_value": 14_880.0,
            "exp_avg_dtype": "torch.float32",
            "exp_avg_sq_dtype": "torch.float32",
            "moments_finite": True,
            "exp_avg_shape": (4, 3),
            "exp_avg_sq_shape": (4, 3),
        }
        validate_adam_state_descriptors([(4, 3)], [state], 14_880)
        with self.assertRaisesRegex(ValueError, "step value"):
            validate_adam_state_descriptors([(4, 3)], [state], 29_760)
        state["state_fields"] = ("step",)
        with self.assertRaisesRegex(ValueError, "Adam state 0 fields"):
            validate_adam_state_descriptors([(4, 3)], [state], 14_880)

    def test_tiny_real_tensor_guards_when_torch_is_available(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Torch is not installed in the local lightweight environment")
        valid = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.ones(2),
            "exp_avg_sq": torch.ones(2),
        }
        _describe_adam_state({0: valid}, [0], [(2,)], torch, 3)
        for missing in ("exp_avg", "exp_avg_sq"):
            with (
                self.subTest(missing=missing),
                self.assertRaisesRegex(ValueError, "Adam state fields"),
            ):
                _describe_adam_state(
                    {0: {key: value for key, value in valid.items() if key != missing}},
                    [0],
                    [(2,)],
                    torch,
                    3,
                )
        nan_state = {**valid, "exp_avg": torch.tensor([float("nan"), 1.0])}
        with self.assertRaisesRegex(ValueError, "moments finite"):
            _describe_adam_state({0: nan_state}, [0], [(2,)], torch, 3)
        nan_step = {**valid, "step": torch.tensor(float("nan"))}
        with self.assertRaisesRegex(ValueError, "step"):
            _describe_adam_state({0: nan_step}, [0], [(2,)], torch, 3)
        wrong_shape = {**valid, "exp_avg_sq": torch.ones(3)}
        with self.assertRaisesRegex(ValueError, "moment shapes"):
            _describe_adam_state({0: wrong_shape}, [0], [(2,)], torch, 3)
        epoch0 = {"weight": torch.zeros(2)}
        epoch1 = {"weight": torch.ones(2)}
        self.assertFalse(_tensor_map_equal(epoch0, epoch1, torch))
        self.assertTrue(_tensor_map_same_layout(epoch0, epoch1))
        self.assertFalse(_tensor_map_same_layout(epoch0, {"weight": torch.ones(3)}))
        self.assertFalse(is_resume_model_checkpoint("model_ep0.pt", "stage1"))
        self.assertTrue(is_resume_model_checkpoint("model_ep1.pt", "stage1"))

    def test_archive_reservation_never_replaces_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "snapshot"
            reserve_archive(destination)
            marker = destination / "preserve.txt"
            marker.write_text("existing evidence")
            with self.assertRaises(FileExistsError):
                reserve_archive(destination)
            self.assertEqual(marker.read_text(), "existing evidence")


if __name__ == "__main__":
    unittest.main()
