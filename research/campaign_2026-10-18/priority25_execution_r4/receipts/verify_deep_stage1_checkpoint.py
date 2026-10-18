#!/usr/bin/env python3
"""Independently verify a collected deep F0 epoch-1 checkpoint boundary."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import verify_deep_epoch0_checkpoint as epoch0

RUN_NAME = "rr18_r01_deep_f0_s1001"
ATTEMPT_NAME = "computer-vision-homework-r01-deep-f0s1001-ep01-v1"
STEPS_PER_EPOCH = 14_920
STAGE1_UPDATES = STEPS_PER_EPOCH * 2
TOTAL_SCHEDULE_STEPS = STEPS_PER_EPOCH * 6
EXPECTED_PARAMETERS = 2_151_554
ARTIFACTS = ("model_ep0.pt", "model_ep1.pt", "provenance.json", "training_state.pt", "history.json")


def _verify_attempt(record_path: Path, plan_path: Path) -> dict[str, object]:
    """Bind structural checks to the frozen, successful stage1 attempt."""
    record = json.loads(record_path.read_text())
    plan = json.loads(plan_path.read_text())
    steps = [step for step in plan["steps"] if step["attempt"] == ATTEMPT_NAME]
    epoch0.require(len(steps) == 1, "frozen plan must contain one deep stage1 attempt")
    command = steps[0]["expected_command"]
    epoch0.require(record["id"] == ATTEMPT_NAME, "attempt identity differs")
    epoch0.require(
        record["status"] == "finished" and record["exit_status"] == 0,
        "attempt did not finish successfully",
    )
    epoch0.require(record["command"] == command, "actual attempt command differs from frozen plan")
    epoch0.require(
        record["working_directory"].endswith("/source-a266b036"), "attempt source checkout differs"
    )
    epoch0.require(command[command.index("--name") + 1] == RUN_NAME, "attempt run name differs")
    epoch0.require(
        command[command.index("--stop-after-epoch") + 1] == "1",
        "attempt was not capped after epoch 1",
    )
    return {
        "attempt": ATTEMPT_NAME,
        "attempt_record_sha256": epoch0.sha256_file(record_path),
        "plan_sha256": epoch0.sha256_file(plan_path),
        "command_matches_frozen_plan": True,
        "exit_status": 0,
        "working_directory": record["working_directory"],
    }


def _verify_collection(
    snapshot: Path, receipt_path: Path, epoch0_receipt_path: Path
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Require stable source metadata and byte-identical collected artifacts."""
    collection = json.loads(receipt_path.read_text())
    previous = json.loads(epoch0_receipt_path.read_text())
    epoch0.require(collection["run"] == RUN_NAME, "collection run identity differs")
    epoch0.require(collection["attempt"] == ATTEMPT_NAME, "collection attempt identity differs")
    epoch0.require(collection["epoch_indices"] == [0, 1], "collection epoch indices differ")
    epoch0.require(
        collection.get("source_snapshot_hash_match") is True,
        "collection did not establish stable copies",
    )
    epoch0.require(
        collection.get("source_stats_before") == collection.get("source_stats_after"),
        "source changed during collection",
    )
    epoch0.require(
        set(collection["artifacts"]) == set(ARTIFACTS), "collection artifact set differs"
    )
    prior_model = previous["artifacts"]["model_ep0.pt"]
    epoch0.require(
        collection["artifacts"]["model_ep0.pt"]["sha256"] == prior_model["sha256"],
        "epoch-0 snapshot changed",
    )
    result = {}
    for name in ARTIFACTS:
        path = snapshot / name
        entry = collection["artifacts"][name]
        digest = epoch0.sha256_file(path)
        size = path.stat().st_size
        epoch0.require(digest == entry["sha256"], f"collected artifact hash differs: {name}")
        epoch0.require(size == entry["bytes"], f"collected artifact size differs: {name}")
        result[name] = {"bytes": size, "sha256": digest}
    return result, collection


def _verify_history(history: list[dict[str, object]], progress: Mapping[str, object]) -> None:
    """Validate both fit-only epochs and their optimizer/scheduler accounting."""
    epoch0.require(len(history) == 2, "stage1 history must contain two epochs")
    epoch0.require([row["epoch"] for row in history] == [0, 1], "history epochs differ")
    epoch0.require(progress["history"] == history, "history file and training-state history differ")
    for index, row in enumerate(history):
        epoch0.require(row["evaluation"] == "withheld", "validation was not withheld during fit")
        updates = row["update_diagnostics"]
        counts = {
            "optimizer_attempts": STEPS_PER_EPOCH,
            "amp_skipped_updates": 0,
            "optimizer_completed_updates": STEPS_PER_EPOCH,
            "ema_updates": 0,
            "completed_batches": 1492,
        }
        epoch0.require(updates["epoch"] == counts, f"epoch {index} update counts differ")
        cumulative = counts.copy()
        completed = (index + 1) * STEPS_PER_EPOCH
        cumulative.update(
            optimizer_attempts=completed,
            optimizer_completed_updates=completed,
            completed_batches=(index + 1) * 1492,
        )
        epoch0.require(
            updates["cumulative"] == cumulative, f"epoch {index} cumulative counts differ"
        )
        epoch0.require(
            updates["grad_stats"]["nonfinite_steps"] == 0, f"epoch {index} had nonfinite gradients"
        )
        fit = row["fit_summary"]
        epoch0.require(
            math.isfinite(float(fit["objective_mean"])), f"epoch {index} fit objective is nonfinite"
        )
        epoch0.require(
            fit["optimizer_windows"] == STEPS_PER_EPOCH,
            f"epoch {index} optimizer window count differs",
        )
        epoch0.require(
            len(fit["learning_rate_by_group"]) == 1, f"epoch {index} LR group count differs"
        )
    expected_total = {
        "optimizer_attempts": STAGE1_UPDATES,
        "amp_skipped_updates": 0,
        "optimizer_completed_updates": STAGE1_UPDATES,
        "ema_updates": 0,
        "completed_batches": 2984,
    }
    epoch0.require(
        progress["cumulative_updates"] == expected_total, "cumulative training-state counts differ"
    )


def verify(args: argparse.Namespace) -> dict[str, object]:
    """Check stage1 state, ancestry, hashes and finite FP32 tensors on CPU."""
    source_root = args.source_root.resolve()
    snapshot = args.snapshot_dir.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from wnn.training_state import load_training_state

    attempt_info = _verify_attempt(args.attempt_record, args.plan)
    artifacts, collection = _verify_collection(
        snapshot, args.collection_receipt, args.epoch0_collection_receipt
    )
    state = load_training_state(snapshot / "training_state.pt")
    deep0 = load_training_state(args.epoch0_snapshot_dir / "training_state.pt")
    wide1 = load_training_state(args.wide_stage1_snapshot_dir / "training_state.pt")
    epoch0_verification = json.loads(args.epoch0_verification.read_text())
    wide_verification = json.loads(args.wide_stage1_verification.read_text())
    contract = state["contract"]
    epoch0.require(state["next_epoch"] == 2, "checkpoint is not the epoch-1 boundary next_epoch=2")
    epoch0.require(deep0["next_epoch"] == 1, "deep epoch-0 baseline is not next_epoch=1")
    epoch0.require(contract == deep0["contract"], "deep stage1 contract changed from epoch0")
    epoch0.require(
        contract["epochs"] == 6 and contract["total_steps"] == TOTAL_SCHEDULE_STEPS,
        "fixed six-epoch schedule differs",
    )
    options = contract["options"]
    epoch0.require(
        options["seed"] == 1001 and options["valid_fold"] == 0, "fit cell identity differs"
    )
    epoch0.require(options["precision"] == "fp32", "fit precision is not FP32")
    epoch0.require(
        options["distill"] == "" and options["propensity"] == "", "teacher or propensity was used"
    )
    epoch0.require(
        options["init"] == ""
        and not options["final_distill_initializer"]
        and not options["inherit_init_preprocessing"],
        "fit was not cold initialized",
    )
    epoch0.require(
        contract["runtime"]["matmul_tf32"] is False and contract["runtime"]["cudnn_tf32"] is False,
        "strict FP32 runtime flags differ",
    )
    epoch0.require(
        contract["teacher_predictions"] == {} and contract["consumed_caches"] == {},
        "teacher/auxiliary caches are present",
    )
    epoch0.require(
        contract["ancestors"]["initial_checkpoint_sha256"] is None
        and contract["ancestors"]["propensity_checkpoint_sha256"] is None,
        "checkpoint/propensity ancestry is present",
    )
    epoch0.require(
        state["scaler"] == {} and state["ema"] is None and state["swa"] is None,
        "unexpected scaler/averaging state",
    )
    epoch0.require(
        epoch0_verification["status"] == "passed" and wide_verification["status"] == "passed",
        "baseline structural receipt failed",
    )
    epoch0.require(wide1["next_epoch"] == 2, "wide baseline is not at next_epoch=2")

    source_count = epoch0._verify_sources(source_root, contract)
    epoch0._verify_source_archive(args.archive_manifest)
    registry_sha = epoch0._verify_registry(contract, args.data_registry)
    epoch0.require(
        contract["sources"]["trainer"] == epoch0.sha256_file(source_root / "scripts/train_rnn.py"),
        "trainer source binding differs",
    )
    epoch0._assert_equal_contracts(wide1["contract"], contract)
    for identity in ("fit_membership", "normalization"):
        epoch0.require(
            contract[identity] == deep0["contract"][identity] == wide1["contract"][identity],
            f"{identity} identity differs",
        )
    epoch0.require(
        epoch0.sha256_file(args.epoch0_snapshot_dir / "training_state.pt")
        == epoch0_verification["artifacts"]["training_state.pt"]["sha256"],
        "deep epoch0 state hash differs from its receipt",
    )
    epoch0.require(
        epoch0.sha256_file(args.wide_stage1_snapshot_dir / "training_state.pt")
        == wide_verification["artifacts"]["training_state.pt"]["sha256"],
        "wide stage1 state hash differs from its receipt",
    )

    provenance = json.loads((snapshot / "provenance.json").read_text())
    history = json.loads((snapshot / "history.json").read_text())
    epoch0.require(
        provenance["fit_membership_identity"] == contract["fit_membership"],
        "provenance membership differs",
    )
    epoch0.require(
        provenance["normalization_identity"] == contract["normalization"],
        "provenance normalization differs",
    )
    epoch0.require(
        provenance["cumulative_updates"] == state["progress"]["cumulative_updates"],
        "provenance cumulative update counts differ",
    )
    for name in ARTIFACTS[:2]:
        epoch0.require(
            provenance["checkpoint_sha256"][name] == artifacts[name]["sha256"],
            f"provenance does not bind {name}",
        )
        raw = torch.load(snapshot / name, map_location="cpu", weights_only=False)
        epoch0.require(raw["epoch"] == int(name[8]), f"{name} epoch index differs")
        epoch0.require(
            raw["args"]["hidden"] == options["hidden"]
            and raw["args"]["layers"] == options["layers"],
            f"{name} architecture differs",
        )
        epoch0.require(
            raw["args"]["seed"] == options["seed"]
            and raw["args"]["valid_fold"] == options["valid_fold"],
            f"{name} cell identity differs",
        )
        epoch0.require(
            raw["args"]["precision"] == options["precision"], f"{name} precision differs"
        )
        for field in (
            "args",
            "trainer_implementation",
            "fit_membership_identity",
            "normalization_identity",
        ):
            if field in provenance:
                epoch0.require(
                    raw["provenance"].get(field) == provenance[field],
                    f"{name} embedded provenance differs: {field}",
                )
        model = raw["model"]
        epoch0.require(
            set(model) == set(state["model"]), f"{name} tensor names differ from continuation model"
        )
        if name == "model_ep1.pt":
            for tensor_name in model:
                epoch0.require(
                    torch.equal(model[tensor_name], state["model"][tensor_name]),
                    f"resume model differs from {name}:{tensor_name}",
                )
    old_model = json.loads(args.epoch0_verification.read_text())["artifacts"]["model_ep0.pt"]
    epoch0.require(
        artifacts["model_ep0.pt"]["sha256"] == old_model["sha256"],
        "stage1 epoch0 inference checkpoint differs from archived epoch0",
    )

    model_tensors = epoch0._tensor_values(state["model"])
    epoch0.require(
        all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for _, t in model_tensors),
        "model is not finite FP32",
    )
    parameter_count = sum(
        t.numel() for name, t in state["model"].items() if name not in {"mean", "std"}
    )
    epoch0.require(parameter_count == EXPECTED_PARAMETERS, "deep parameter count differs")
    optimizer = state["optimizer"]
    parameter_ids = [pid for group in optimizer["param_groups"] for pid in group["params"]]
    optimizer_state = optimizer["state"]
    epoch0.require(
        len(parameter_ids) == len(set(parameter_ids))
        and set(parameter_ids) == set(optimizer_state),
        "optimizer state does not cover unique model parameters",
    )
    optimizer_tensors = [
        tensor for val in optimizer_state.values() for _, tensor in epoch0._tensor_values(val)
    ]
    floating = [tensor for tensor in optimizer_tensors if tensor.is_floating_point()]
    epoch0.require(
        all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in floating),
        "optimizer floating tensors are not finite FP32",
    )

    progress = state["progress"]
    _verify_history(history, progress)
    scheduler = state["scheduler"]
    epoch0.require(
        scheduler["last_epoch"] == 29_840 and scheduler["_step_count"] == 29_841,
        "scheduler progress differs",
    )
    epoch0.require(not torch.cuda.is_initialized(), "CUDA initialized during verification")
    epoch0.require(
        all(t.device.type == "cpu" for _, t in model_tensors)
        and all(t.device.type == "cpu" for t in optimizer_tensors),
        "checkpoint tensors were not kept on CPU",
    )
    return {
        "kind": "actual_stage1_checkpoint_structural_verification",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "run": RUN_NAME,
        "attempt": attempt_info,
        "epoch_indices": [0, 1],
        "next_epoch": state["next_epoch"],
        "fixed_endpoint_epoch": 5,
        "planned_schedule_epochs": 6,
        "snapshot_directory": str(snapshot),
        "collection_receipt_sha256": epoch0.sha256_file(args.collection_receipt),
        "collector_helper_sha256": collection["collector_sha256"],
        "verification_helper_sha256": epoch0.sha256_file(Path(__file__)),
        "epoch0_verifier_dependency_sha256": epoch0.sha256_file(Path(epoch0.__file__).resolve()),
        "artifacts": artifacts,
        "fitting_source_revision": epoch0.SOURCE_COMMIT,
        "source_archive_sha256": epoch0.SOURCE_ARCHIVE_SHA256,
        "source_files_verified": source_count,
        "dataset_registry_sha256": registry_sha,
        "dataset_hashes_match_sealed_registry": True,
        "deep_epoch0_contract_exact_match": True,
        "paired_contract_architecture_only": {
            "wide": {"hidden": 512, "layers": 1},
            "deep": {"hidden": 384, "layers": 2},
        },
        "fit_membership_identity": contract["fit_membership"],
        "normalization_identity": contract["normalization"],
        "model_tensors_verified": len(model_tensors),
        "trainable_parameter_count": parameter_count,
        "optimizer_states_verified": len(optimizer_state),
        "optimizer_floating_tensors_verified": len(floating),
        "completed_update_counts": progress["cumulative_updates"],
        "fixed_schedule_total_steps": contract["total_steps"],
        "scheduler_last_epoch": scheduler["last_epoch"],
        "scheduler_step_count": scheduler["_step_count"],
        "raw_epoch1_checkpoint_resume_model_equal": True,
        "all_model_and_optimizer_floating_tensors_finite_fp32": True,
        "epochs_fit_only_withheld_validation": True,
        "epoch_summaries": history,
        "cuda_initialized_during_verification": torch.cuda.is_initialized(),
        "limitations": [
            "Structural checkpoint and source/registry checks only; no measured GPU resume parity.",
            "Parquet payloads were not rehashed during this bounded checkpoint check; identities were matched to the sealed registry.",
            "Two intermediate epochs, not the epoch-5 quality endpoint, independent grade, CP30 result, or convergence proof.",
        ],
    }


def main() -> None:
    """Run CPU-only verification and write a single deterministic-schema receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "snapshot-dir",
        "source-root",
        "epoch0-snapshot-dir",
        "epoch0-collection-receipt",
        "epoch0-verification",
        "wide-stage1-snapshot-dir",
        "wide-stage1-verification",
        "collection-receipt",
        "attempt-record",
        "plan",
        "data-registry",
        "archive-manifest",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
