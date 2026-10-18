from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from collect_f1_checkpoint import reserve_archive
from f1_checkpoint_common import (
    ARCHIVE_MANIFEST_SHA256,
    FOLD1_SHA256,
    PLAN_SHA256,
    REGISTRY_SHA256,
    REMOTE_ROOT,
    SOURCE_REVISION,
    arm_spec,
    expected_counts,
    is_resume_model_checkpoint,
    validate_adam_state_descriptors,
    validate_history,
    validate_optional_epoch0_binding,
    validate_pair_contracts,
    validate_parent_headroom,
    validate_plan,
    validate_systemd_runtime_limit,
)
from f1_checkpoint_lineage import verify_lineage
from f1_checkpoint_tensors import _describe_adam_state, _tensor_map_equal, _tensor_map_same_layout


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
                        "steps": 14_930,
                        "nonfinite_steps": 0,
                    },
                },
                "fit_summary": {
                    "objective_mean": 1.0,
                    "optimizer_windows": 14_930,
                    "optimizer_updates": dict(epoch_counts),
                    "learning_rate_by_group": [
                        {"group": 0, "first": 0.1, "last": 0.01, "mean": 0.05}
                    ],
                },
            }
        )
    return rows


class F1CheckpointSupportTests(unittest.TestCase):
    def test_plan_is_frozen_and_resolves_seed1001_fold1_steps(self) -> None:
        packet = Path(__file__).parent.parent
        plan = validate_plan(packet / "F1_BELERIAND_PLAN.json")
        wide_stage1 = arm_spec(plan, "wide", "stage1")
        deep_terminal = arm_spec(plan, "deep", "terminal")
        self.assertEqual(wide_stage1["run"], "rr18_r01_wide_f1_s1001")
        self.assertEqual(
            wide_stage1["attempt"], "computer-vision-homework-r01-wide-f1s1001-ep01-v1"
        )
        self.assertEqual(deep_terminal["run"], "rr18_r01_deep_f1_s1001_e6")
        self.assertEqual((deep_terminal["hidden"], deep_terminal["layers"]), (384, 2))

    def test_static_f1_counts(self) -> None:
        self.assertEqual(
            expected_counts(1),
            {
                "optimizer_attempts": 14_930,
                "amp_skipped_updates": 0,
                "optimizer_completed_updates": 14_930,
                "ema_updates": 0,
                "completed_batches": 1_493,
            },
        )
        self.assertEqual(expected_counts(2)["optimizer_completed_updates"], 29_860)
        self.assertEqual(expected_counts(6)["optimizer_completed_updates"], 89_580)

    def test_terminal_lineage_rejects_wrong_parent_attempt_and_history(self) -> None:
        packet = Path(__file__).parent.parent
        plan = validate_plan(packet / "F1_BELERIAND_PLAN.json")
        wide = arm_spec(plan, "wide", "stage1")
        deep = arm_spec(plan, "deep", "stage1")
        history = _history_fixture()[:2]
        common = {"fit_membership": {"fold": 1}, "normalization": {"mean": "same"}}
        wide_contract = {**common, "options": {"hidden": 512, "layers": 1, "valid_fold": 1}}
        deep_contract = {**common, "options": {"hidden": 384, "layers": 2, "valid_fold": 1}}
        helper_hash = "c" * 64
        dependencies = {"f1_checkpoint_common.py": "d" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wide_dir = root / "wide"
            deep_dir = root / "deep"
            wide_dir.mkdir()
            deep_dir.mkdir()
            wide_state_path = wide_dir / "training_state.pt"
            deep_state_path = deep_dir / "training_state.pt"
            wide_state_path.write_bytes(b"wide stage1")
            deep_state_path.write_bytes(b"deep stage1")
            wide_state = {
                "next_epoch": 2,
                "contract": wide_contract,
                "progress": {"history": history, "cumulative_updates": expected_counts(2)},
            }
            deep_state = {
                "next_epoch": 2,
                "contract": deep_contract,
                "progress": {"history": history, "cumulative_updates": expected_counts(2)},
            }

            def receipt(arm: str, spec: dict[str, object], path: Path) -> dict[str, object]:
                return {
                    "schema": "priority25.f1_checkpoint_verification.v1",
                    "status": "passed",
                    "phase": "stage1",
                    "arm": arm,
                    "run": spec["run"],
                    "next_epoch": 2,
                    "plan_sha256": PLAN_SHA256,
                    "fold1_manifest_sha256": FOLD1_SHA256,
                    "portable_registry_sha256": REGISTRY_SHA256,
                    "source_archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
                    "source_revision": SOURCE_REVISION,
                    "verification_helper_sha256": helper_hash,
                    "verification_dependencies": dependencies,
                    "completed_update_counts": expected_counts(2),
                    "attempt": {
                        "id": spec["attempt"],
                        "command": spec["command"],
                        "status": "finished",
                        "exit_status": 0,
                    },
                    "artifacts": {
                        "training_state.pt": {
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                        }
                    },
                }

            wide_receipt = root / "wide.json"
            deep_receipt = root / "deep.json"
            wide_receipt.write_text(json.dumps(receipt("wide", wide, wide_state_path)))
            deep_receipt.write_text(json.dumps(receipt("deep", deep, deep_state_path)))
            args = Namespace(
                phase="terminal",
                arm="wide",
                stage1_snapshot_dir=wide_dir,
                stage1_verification=wide_receipt,
                peer_stage1_snapshot_dir=deep_dir,
                peer_stage1_verification=deep_receipt,
            )
            spec = arm_spec(plan, "wide", "terminal")
            parent_path = (
                REMOTE_ROOT / "data/artifacts/runs" / str(wide["run"]) / "training_state.pt"
            )
            resumed = {
                "path": str(parent_path),
                "sha256": hashlib.sha256(wide_state_path.read_bytes()).hexdigest(),
                "next_epoch": 2,
            }
            provenance = {"resumed_from": resumed}
            collection = {"resumed_from": resumed}
            states = {str(wide_state_path): wide_state, str(deep_state_path): deep_state}

            def load_state(path: Path) -> object:
                return states[str(path)]

            self.assertEqual(
                verify_lineage(
                    args,
                    plan,
                    spec,
                    wide_contract,
                    history,
                    provenance,
                    collection,
                    load_state,
                    helper_hash,
                    dependencies,
                ),
                resumed["sha256"],
            )

            bad_receipt = receipt("wide", wide, wide_state_path)
            bad_receipt["attempt"]["id"] = "wrong-attempt"
            wide_receipt.write_text(json.dumps(bad_receipt))
            with self.assertRaisesRegex(ValueError, "verified stage1 attempt identity"):
                verify_lineage(
                    args,
                    plan,
                    spec,
                    wide_contract,
                    history,
                    provenance,
                    collection,
                    load_state,
                    helper_hash,
                    dependencies,
                )

            wide_receipt.write_text(json.dumps(receipt("wide", wide, wide_state_path)))
            changed_history = [dict(history[0]), history[1]]
            changed_history[0]["fit_summary"] = {"objective_mean": 9.0}
            states[str(wide_state_path)]["progress"]["history"] = changed_history
            with self.assertRaisesRegex(ValueError, "stage1 history changed"):
                verify_lineage(
                    args,
                    plan,
                    spec,
                    wide_contract,
                    history,
                    provenance,
                    collection,
                    load_state,
                    helper_hash,
                    dependencies,
                )

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

    def test_epoch0_receipt_is_optional_but_bound_when_supplied(self) -> None:
        self.assertFalse(validate_optional_epoch0_binding(None, "wide", "run-wide", "a" * 64))
        receipt = {
            "schema": "priority25.f1_checkpoint_collection.v1",
            "arm": "wide",
            "phase": "epoch0",
            "run": "run-wide",
            "artifacts": {"model_ep0.pt": {"sha256": "a" * 64}},
        }
        self.assertTrue(validate_optional_epoch0_binding(receipt, "wide", "run-wide", "a" * 64))
        with self.assertRaisesRegex(ValueError, "prior epoch0 collection identity"):
            validate_optional_epoch0_binding(receipt, "deep", "run-wide", "a" * 64)

    def test_pair_contract_allows_only_declared_architecture_changes(self) -> None:
        wide = {"options": {"hidden": 512, "layers": 1, "valid_fold": 1}, "runtime": {"fp32": True}}
        deep = {"options": {"hidden": 384, "layers": 2, "valid_fold": 1}, "runtime": {"fp32": True}}
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
            "step_value": 14_930.0,
            "exp_avg_dtype": "torch.float32",
            "exp_avg_sq_dtype": "torch.float32",
            "moments_finite": True,
            "exp_avg_shape": (4, 3),
            "exp_avg_sq_shape": (4, 3),
        }
        validate_adam_state_descriptors([(4, 3)], [state], 14_930)
        with self.assertRaisesRegex(ValueError, "step value"):
            validate_adam_state_descriptors([(4, 3)], [state], 29_860)
        state["state_fields"] = ("step",)
        with self.assertRaisesRegex(ValueError, "Adam state 0 fields"):
            validate_adam_state_descriptors([(4, 3)], [state], 14_930)

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
