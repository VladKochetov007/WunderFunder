#!/usr/bin/env python3
"""Verify the first resumed wide E6 checkpoint against its exact stage1 parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

RUN_NAME = "rr18_r01_wide_f0_s1001_e6"
ATTEMPT_NAME = "computer-vision-homework-r01-wide-f0s1001-ep25-v1"
BASE_RUN_NAME = "rr18_r01_wide_f0_s1001"
SOURCE_COMMIT = "a266b036b02e8e012c6eb31072291dad618a169d"
SOURCE_ARCHIVE_SHA256 = "288b4c9dd67b48114240bbf593901fc539de15aa709a1aa63388dccf64a4478a"
PLAN_SHA256 = "dcafeddb5d1d744ea0d6f981f7d70154403c5f6e4cc199bbd8060b1c961f3177"
SOURCE_FILES = 138
STEPS_PER_EPOCH = 14_920
EXPECTED_NEXT_EPOCH = 3
EXPECTED_UPDATES = STEPS_PER_EPOCH * EXPECTED_NEXT_EPOCH
TOTAL_SCHEDULE_STEPS = STEPS_PER_EPOCH * 6
EXPECTED_PARAMETERS = 2_210_306
ARTIFACTS = ("model_ep2.pt", "provenance.json", "training_state.pt")


def sha256_file(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject any mismatch in the collected continuation artifact."""
    if not condition:
        raise ValueError(message)


def tensor_values(value: object) -> list[torch.Tensor]:
    """Flatten tensors in nested optimizer state without schema coercion."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for child in value.values() for tensor in tensor_values(child)]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in tensor_values(child)]
    return []


def verify_sources(source_root: Path, contract: Mapping[str, object]) -> int:
    """Check all source files bound into the canonical training contract."""
    sources = contract.get("sources")
    require(isinstance(sources, Mapping) and len(sources) == SOURCE_FILES, "source map must bind 138 files")
    for relative, expected in sources.items():
        path = source_root / "scripts/train_rnn.py" if relative == "trainer" else source_root / "src/wnn" / str(relative)
        require(path.is_file(), f"source file missing: {relative}")
        require(sha256_file(path) == expected, f"source hash differs: {relative}")
    return len(sources)


def verify_registry(contract: Mapping[str, object], registry_path: Path) -> str:
    """Match recorded train/validation data identities to the sealed registry."""
    registry = json.loads(registry_path.read_text())
    datasets = contract["datasets"]
    for split in ("train", "valid"):
        entry = registry["sources"][f"datasets/{split}.parquet"]
        require(
            datasets[split]["bytes"] == entry["original_identity"]["bytes"]
            and datasets[split]["sha256"] == entry["sha256"],
            f"{split} dataset identity differs from registry",
        )
    return sha256_file(registry_path)


def verify_collection(snapshot: Path, receipt_path: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Validate the stable copied artifacts against the collection receipt."""
    receipt = json.loads(receipt_path.read_text())
    require(receipt["run"] == RUN_NAME and receipt["attempt"] == ATTEMPT_NAME, "collection run/attempt differs")
    require(receipt["source_snapshot_hash_match"] is True, "collection did not verify source/copy hashes")
    require(receipt["source_stats_before"] == receipt["source_stats_after"], "source changed during collection")
    require(set(receipt["artifacts"]) == set(ARTIFACTS), "collection file set differs")
    result: dict[str, dict[str, object]] = {}
    for name in ARTIFACTS:
        path = snapshot / name
        expected = receipt["artifacts"][name]
        digest = sha256_file(path)
        size = path.stat().st_size
        require(digest == expected["sha256"] and size == expected["bytes"], f"snapshot artifact differs: {name}")
        result[name] = {"bytes": size, "sha256": digest}
    return result, receipt


def verify_attempt(record_path: Path, plan_path: Path) -> dict[str, object]:
    """Bind the intermediate checkpoint to its frozen active/terminal attempt."""
    record = json.loads(record_path.read_text())
    plan = json.loads(plan_path.read_text())
    steps = [step for step in plan["steps"] if step["attempt"] == ATTEMPT_NAME]
    require(len(steps) == 1, "frozen plan lacks unique wide E6 attempt")
    command = steps[0]["expected_command"]
    require(sha256_file(plan_path) == PLAN_SHA256, "frozen F0 plan bytes differ")
    require(record["id"] == ATTEMPT_NAME, "attempt ID differs")
    status = record["status"]
    exit_status = record.get("exit_status")
    require(
        (status == "running" and exit_status is None)
        or (status == "finished" and exit_status == 0),
        "attempt is neither active nor successfully terminal",
    )
    require(record["command"] == command, "actual command differs from frozen plan")
    require(record["working_directory"].endswith("/source-a266b036"), "trainer checkout differs")
    require(
        record["unit"] == "wnn-bandit-computer-vision-homework-r01-wide-f0s1001-ep25-v1.service",
        "attempt systemd unit differs",
    )
    require(command[command.index("--name") + 1] == RUN_NAME, "run name differs")
    require(
        command[command.index("--resume-state") + 1]
        == f"/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/runs/{BASE_RUN_NAME}/training_state.pt",
        "resume input path differs",
    )
    return {
        "attempt": ATTEMPT_NAME,
        "attempt_record_sha256": sha256_file(record_path),
        "plan_sha256": sha256_file(plan_path),
        "command_matches_frozen_plan": True,
        "status_in_attempt_snapshot": status,
        "exit_status": exit_status,
        "started_utc": record.get("started_utc"),
        "heartbeat_utc": record.get("heartbeat_utc"),
        "unit": record["unit"],
        "working_directory": record["working_directory"],
    }


def verify(args: argparse.Namespace) -> dict[str, object]:
    """Check stage2 checkpoint ancestry, progress, model, optimizer and source on CPU."""
    source_root = args.source_root.resolve()
    snapshot = args.snapshot_dir.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from wnn.training_state import load_training_state

    attempt = verify_attempt(args.attempt_record, args.plan)
    artifacts, collection = verify_collection(snapshot, args.collection_receipt)
    state = load_training_state(snapshot / "training_state.pt")
    parent_state = load_training_state(args.stage1_snapshot_dir / "training_state.pt")
    stage1_receipt = json.loads(args.stage1_verification.read_text())
    contract = state["contract"]
    require(stage1_receipt["status"] == "passed" and stage1_receipt["next_epoch"] == 2, "stage1 receipt invalid")
    require(
        stage1_receipt["artifacts"]["training_state.pt"]["sha256"]
        == sha256_file(args.stage1_snapshot_dir / "training_state.pt"),
        "stage1 parent checkpoint hash differs from its verification receipt",
    )
    require(state["next_epoch"] == EXPECTED_NEXT_EPOCH, "checkpoint is not next_epoch=3")
    require(parent_state["next_epoch"] == 2 and contract == parent_state["contract"], "resumed training contract changed")
    require(contract["epochs"] == 6 and contract["total_steps"] == TOTAL_SCHEDULE_STEPS, "fixed six-epoch schedule differs")
    options = contract["options"]
    require(options["hidden"] == 512 and options["layers"] == 1, "wide architecture differs")
    require(options["seed"] == 1001 and options["valid_fold"] == 0, "fit cell identity differs")
    require(options["precision"] == "fp32", "fit precision differs")
    require(options["distill"] == "" and options["propensity"] == "", "fit has teacher/propensity ancestry")
    require(options["init"] == "" and not options["final_distill_initializer"] and not options["inherit_init_preprocessing"], "fit was not cold initialized")
    require(contract["runtime"]["matmul_tf32"] is False and contract["runtime"]["cudnn_tf32"] is False, "strict FP32 runtime flags differ")
    require(contract["teacher_predictions"] == {} and contract["consumed_caches"] == {}, "teacher/auxiliary cache exists")
    require(contract["ancestors"]["initial_checkpoint_sha256"] is None and contract["ancestors"]["propensity_checkpoint_sha256"] is None, "unexpected initialization ancestry")
    require(state["scaler"] == {} and state["ema"] is None and state["swa"] is None, "unexpected scaler/averaging state")

    source_count = verify_sources(source_root, contract)
    archive = json.loads(args.archive_manifest.read_text())
    require(archive["source_revision"] == "a266b036", "source archive revision differs")
    require(any(item["sha256"] == SOURCE_ARCHIVE_SHA256 for item in archive["archives"]), "source archive hash differs")
    registry_sha = verify_registry(contract, args.data_registry)
    stage1_contract = parent_state["contract"]
    require(contract["sources"] == stage1_contract["sources"], "resumed source map changed")
    require(contract["fit_membership"] == stage1_contract["fit_membership"], "fit membership changed")
    require(contract["normalization"] == stage1_contract["normalization"], "normalization changed")

    stage1_sha = sha256_file(args.stage1_snapshot_dir / "training_state.pt")
    require(
        collection["resumed_from"]
        == {
            "path": f"/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/runs/{BASE_RUN_NAME}/training_state.pt",
            "sha256": stage1_sha,
            "next_epoch": 2,
        },
        "collection resume ancestry differs from the stage1 parent",
    )
    provenance = json.loads((snapshot / "provenance.json").read_text())
    resumed_from = provenance["resumed_from"]
    expected_resume_path = f"/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/runs/{BASE_RUN_NAME}/training_state.pt"
    require(resumed_from["path"] == expected_resume_path, "persisted resumed_from path differs")
    require(resumed_from["sha256"] == stage1_sha, "persisted resumed_from SHA differs from archived stage1")
    require(resumed_from["next_epoch"] == 2, "persisted resumed_from boundary differs")
    require(provenance["fit_membership_identity"] == contract["fit_membership"], "provenance membership differs")
    require(provenance["normalization_identity"] == contract["normalization"], "provenance normalization differs")
    require(provenance["cumulative_updates"]["optimizer_completed_updates"] == EXPECTED_UPDATES, "provenance progress differs")
    require(provenance["checkpoint_sha256"]["model_ep2.pt"] == artifacts["model_ep2.pt"]["sha256"], "provenance does not bind raw model")
    raw = torch.load(snapshot / "model_ep2.pt", map_location="cpu", weights_only=False)
    require(raw["epoch"] == 2, "raw checkpoint is not epoch2")
    require(raw["provenance"]["resumed_from"] == resumed_from, "raw checkpoint ancestry differs from provenance")
    require(raw["args"]["hidden"] == 512 and raw["args"]["layers"] == 1, "raw checkpoint architecture differs")
    require(raw["args"]["seed"] == 1001 and raw["args"]["valid_fold"] == 0, "raw checkpoint cell identity differs")
    require(raw["args"]["precision"] == "fp32", "raw checkpoint precision differs")
    for field in ("args", "trainer_implementation", "fit_membership_identity", "normalization_identity", "resumed_from"):
        if field in provenance:
            require(raw["provenance"].get(field) == provenance[field], f"embedded provenance differs: {field}")

    model = state["model"]
    require(set(raw["model"]) == set(model), "raw and resume tensor names differ")
    require(all(torch.equal(raw["model"][key], model[key]) for key in model), "raw model_ep2 differs from resume state")
    model_tensors = tensor_values(model)
    require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in model_tensors), "model is not finite FP32")
    parameter_count = sum(t.numel() for key, t in model.items() if key not in {"mean", "std"})
    require(parameter_count == EXPECTED_PARAMETERS, "wide model parameter count differs")
    optimizer = state["optimizer"]
    parameter_ids = [pid for group in optimizer["param_groups"] for pid in group["params"]]
    optimizer_state = optimizer["state"]
    require(len(parameter_ids) == len(set(parameter_ids)) and set(parameter_ids) == set(optimizer_state), "optimizer state coverage differs")
    optimizer_tensors = [tensor for value in optimizer_state.values() for tensor in tensor_values(value)]
    floating_optimizer = [tensor for tensor in optimizer_tensors if tensor.is_floating_point()]
    require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in floating_optimizer), "optimizer state is not finite FP32")

    progress = state["progress"]
    history = progress["history"]
    require(len(history) == 3, "training-state history must contain epochs 0 through 2")
    require([row["epoch"] for row in history] == [0, 1, 2], "history epoch indices differ")
    require(history[:2] == parent_state["progress"]["history"], "prior stage1 history changed")
    for index, row in enumerate(history):
        require(row["evaluation"] == "withheld", f"validation was not withheld for epoch {index}")
        counts = row["update_diagnostics"]["epoch"]
        require(counts["optimizer_attempts"] == STEPS_PER_EPOCH, f"epoch {index} attempts differ")
        require(counts["optimizer_completed_updates"] == STEPS_PER_EPOCH, f"epoch {index} updates differ")
        require(counts["amp_skipped_updates"] == 0 and counts["completed_batches"] == 1492, f"epoch {index} batch/skip counts differ")
        require(row["update_diagnostics"]["grad_stats"]["nonfinite_steps"] == 0, f"epoch {index} nonfinite gradient recorded")
        require(math.isfinite(float(row["fit_summary"]["objective_mean"])), f"epoch {index} fit objective is nonfinite")
        require(row["fit_summary"]["optimizer_windows"] == STEPS_PER_EPOCH, f"epoch {index} optimizer window count differs")
        require(len(row["fit_summary"]["learning_rate_by_group"]) == 1, f"epoch {index} learning-rate group count differs")
    cumulative = progress["cumulative_updates"]
    require(cumulative["optimizer_attempts"] == EXPECTED_UPDATES, "total attempts differ")
    require(cumulative["optimizer_completed_updates"] == EXPECTED_UPDATES, "total completed updates differ")
    require(cumulative["amp_skipped_updates"] == 0 and cumulative["completed_batches"] == 4476, "total skips/batches differ")
    require(cumulative["ema_updates"] == 0 and progress["step_count"] == EXPECTED_UPDATES, "training step clock differs")
    scheduler = state["scheduler"]
    require(scheduler["last_epoch"] == EXPECTED_UPDATES and scheduler["_step_count"] == EXPECTED_UPDATES + 1, "scheduler clock differs")
    require(not torch.cuda.is_initialized(), "CUDA initialized during verification")
    require(all(t.device.type == "cpu" for t in model_tensors + optimizer_tensors), "loaded tensors are not on CPU")

    return {
        "kind": "actual_resumed_epoch_checkpoint_structural_verification",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "run": RUN_NAME,
        "attempt": attempt,
        "epoch_index": 2,
        "next_epoch": state["next_epoch"],
        "fixed_endpoint_epoch": 5,
        "planned_schedule_epochs": 6,
        "snapshot_directory": str(snapshot),
        "collection_receipt_sha256": sha256_file(args.collection_receipt),
        "verification_helper_sha256": sha256_file(Path(__file__)),
        "artifacts": artifacts,
        "fitting_source_revision": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_files_verified": source_count,
        "dataset_registry_sha256": registry_sha,
        "dataset_hashes_match_sealed_registry": True,
        "stage1_training_state_sha256": stage1_sha,
        "stage1_verification_receipt_sha256": sha256_file(args.stage1_verification),
        "frozen_plan_sha256": PLAN_SHA256,
        "persisted_resumed_from": resumed_from,
        "stage1_contract_exact_match": True,
        "fit_membership_identity": contract["fit_membership"],
        "normalization_identity": contract["normalization"],
        "model_tensors_verified": len(model_tensors),
        "trainable_parameter_count": parameter_count,
        "optimizer_states_verified": len(optimizer_state),
        "optimizer_floating_tensors_verified": len(floating_optimizer),
        "completed_update_counts": cumulative,
        "fixed_schedule_total_steps": contract["total_steps"],
        "scheduler_last_epoch": scheduler["last_epoch"],
        "scheduler_step_count": scheduler["_step_count"],
        "raw_epoch2_checkpoint_resume_model_equal": True,
        "all_model_and_optimizer_floating_tensors_finite_fp32": True,
        "history_epochs_fit_only_from_training_state": [
            {"epoch": row["epoch"], "train_seconds": row["train_seconds"], "fit_summary": row["fit_summary"]}
            for row in history
        ],
        "cuda_initialized_during_verification": torch.cuda.is_initialized(),
        "limitations": [
            "Structural continuation checks only; measured GPU resumed-versus-uninterrupted parity is not established.",
            "Raw parquets were not rehashed; identities were matched to the sealed registry.",
            "This is an intermediate epoch2 checkpoint, not the epoch5 quality endpoint, independent grade, CP30 result, or convergence proof.",
        ],
    }


def main() -> None:
    """Run one bounded CPU structural check and write its JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "snapshot-dir", "source-root", "stage1-snapshot-dir", "stage1-verification",
        "collection-receipt", "attempt-record", "plan",
        "data-registry", "archive-manifest", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
