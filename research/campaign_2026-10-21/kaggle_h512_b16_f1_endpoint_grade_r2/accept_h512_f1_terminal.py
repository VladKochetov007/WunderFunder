"""Accept downloaded B16 artifacts after independent finite and trajectory checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from f1_grade_contract import (
    DERIVED_SHA,
    FIT_NAME,
    ORIGINAL_SHA,
    SWA_NAME,
    artifact_root,
    check_manifests,
    pinned,
    read_json,
    sha256,
)
from f1_grade_outputs import check_worker_outputs
from grade_h512_f1_screen import PLAN, PLAN_SHA256


def check_training_history(base: Path, worker: dict[str, Any]) -> dict[str, Any]:
    """Check six completed epochs and the recorded strict FP32 update evidence."""
    run = base / "artifacts/runs" / FIT_NAME
    history_path = run / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    provenance = read_json(run / "provenance.json")
    if (
        not isinstance(history, list)
        or len(history) != 6
        or [item.get("epoch") for item in history] != list(range(6))
        or any(item.get("evaluation") != "withheld" for item in history)
    ):
        raise ValueError("Expected six complete fixed-recipe epochs without fit-time evaluation")
    expected_args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 512,
        "layers": 1,
        "dropout": 0.0,
        "valid_fold": 1,
        "n_valid_folds": 4,
        "train_seqs": 10607,
        "train_source": "train+valid",
        "valid_train_fraction": 1.0,
        "valid_repeat": 1,
        "full_valid": True,
        "eval_source": "vfold",
        "batch": 16,
        "chunk": 2000,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "no_eval_during_fit": True,
        "seed": 1004,
        "precision": "fp32",
        "loss": "wmse",
        "loss_norm": "chunk",
        "batch_order": "seeded",
        "deterministic": True,
        "sync_free": True,
        "fused_adam": True,
        "grad_stats": True,
        "optim": "adamw",
        "lr": 0.0001,
        "beta2": 0.999,
        "wd": 0.01,
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "head_lr_mult": 1.0,
        "clip": 1.0,
    }
    if any(provenance.get("args", {}).get(key) != value for key, value in expected_args.items()):
        raise ValueError("Fit arguments differ from the frozen B16 scientific recipe")
    diagnostics = provenance.get("update_diagnostics", [])
    if len(diagnostics) != 6:
        raise ValueError("Incomplete epoch update diagnostics")
    for epoch, item in enumerate(diagnostics):
        counts = item.get("epoch", {})
        if (
            item.get("epoch_index") != epoch
            or item.get("normalization_identity") != provenance.get("normalization_identity")
            or counts.get("optimizer_attempts") != 7460
            or counts.get("optimizer_completed_updates") != 7460
            or counts.get("completed_batches") != 746
            or counts.get("amp_skipped_updates") != 0
        ):
            raise ValueError(f"Incomplete B16 update or normalization evidence at epoch {epoch}")
    precision = worker.get("precision_audit", {})
    if precision.get("nvidia_tf32_override") != "0":
        raise ValueError("Missing worker TF32 override evidence")
    for phase in (
        "flags_after_profile",
        "flags_before_full_fit",
        "flags_after_full_fit",
        "flags_after_exports",
    ):
        if precision.get(phase) != {"matmul_tf32": False, "cudnn_tf32": False}:
            raise ValueError(f"Missing strict FP32 runtime evidence: {phase}")
    return {"path": str(history_path), "sha256": sha256(history_path), "epochs": 6}


def check_trajectory(root: Path, base: Path) -> dict[str, Any]:
    """Recompute the exact canonical equal-weight mean from downloaded CP3/4/5."""
    import importlib.util

    import torch

    helper_path = root / "scripts/average_trajectory_checkpoints.py"
    helper_sha = "8646ccc12b537a1bf2f2544dffd85d10287076451e5bab86a4b2d6379e39bbb3"
    if sha256(helper_path) != helper_sha:
        raise ValueError("Canonical trajectory helper differs from the reviewed source")
    spec = importlib.util.spec_from_file_location("accepted_trajectory_mean", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the canonical trajectory mean helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoints = [
            torch.load(
                base / "artifacts/runs" / FIT_NAME / f"model_ep{epoch}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for epoch in (3, 4, 5)
        ]
        derived = torch.load(
            base / "artifacts/runs" / SWA_NAME / "model_ep5.pt",
            map_location="cpu",
            weights_only=True,
        )
    for epoch, checkpoint in zip((3, 4, 5), checkpoints, strict=True):
        if (
            checkpoint.get("epoch") != epoch
            or checkpoint.get("n_in") != 211
            or checkpoint.get("args") != checkpoints[-1].get("args")
            or checkpoint.get("args", {}).get("split_manifest_sha256") != DERIVED_SHA
        ):
            raise ValueError("Source checkpoint epoch, input width or fitting arguments differ")
    if derived.get("epoch") != 5 or derived.get("n_in") != 211:
        raise ValueError("Derived checkpoint metadata differs from CP5 H512")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for checkpoint in [*checkpoints, derived]
        for value in checkpoint["model"].values()
    ):
        raise ValueError("Checkpoint contains floating model tensors outside FP32")
    expected = helper.mean_state([checkpoint["model"] for checkpoint in checkpoints])
    if expected.keys() != derived["model"].keys() or any(
        not torch.equal(value, derived["model"][name]) for name, value in expected.items()
    ):
        raise ValueError("Downloaded SWA weights differ from the exact canonical CP3/4/5 mean")
    if helper_sha != sha256(helper_path):
        raise ValueError("Trajectory helper changed during recomputation")
    return {"tensors_bitwise_equal": len(expected), "helper_sha256": helper_sha}


def accept(root: Path, base: Path) -> Path:
    """Write root acceptance only after checking actual downloaded model outputs."""
    plan = read_json(pinned(root, {"path": PLAN.as_posix(), "sha256": PLAN_SHA256}))
    if plan.get("status") != "FROZEN_SCREEN_NOT_GRADED":
        raise ValueError("B16 plan is not frozen")
    contract = plan["acceptance"]
    destination = root / contract["path"]
    if destination.exists():
        raise FileExistsError(destination)
    worker_path = root / contract["worker_receipt_path"]
    worker = read_json(worker_path)
    worker_sha = sha256(worker_path)
    if (
        worker.get("status") != "CONDITIONAL_FIT_AND_EXPORT_COMPLETE"
        or worker.get("schema") != "wnn.kaggle-h512-f1-conditional-worker-receipt.v1"
        or worker.get("run_names") != {"cp5": FIT_NAME, "swa345": SWA_NAME}
        or worker.get("original_manifest_sha256") != ORIGINAL_SHA
        or worker.get("derived_manifest_sha256") != DERIVED_SHA
        or worker.get("quality_fit_claim") is not None
    ):
        raise ValueError("Worker has not completed the exact B16 fit and both exports")
    _, derived = check_manifests(root, plan, worker)
    fit_dir = base / "artifacts/runs" / FIT_NAME
    swa = worker["swa345"]
    record = {
        "schema": contract["schema"],
        "acceptance_code_sha256": sha256(Path(__file__).resolve()),
        "plan_sha256": PLAN_SHA256,
        "status": contract["status"],
        "fit_run": FIT_NAME,
        "original_split_manifest_sha256": ORIGINAL_SHA,
        "execution_split_manifest_sha256": DERIVED_SHA,
        "artifact_root": str(base),
        "worker_receipt_path": contract["worker_receipt_path"],
        "worker_receipt_sha256": worker_sha,
        "fit_provenance_sha256": sha256(fit_dir / "provenance.json"),
        "checkpoint_sha256": {
            f"model_ep{epoch}.pt": sha256(fit_dir / f"model_ep{epoch}.pt") for epoch in (3, 4, 5)
        },
        "swa_derivation_receipt_path": swa["weight_derivation_receipt_path"],
        "swa_derivation_receipt_sha256": swa["weight_derivation_receipt_sha256"],
        "exports": {
            key: next(item for item in worker["exports"] if item["run_name"] == name)
            for key, name in (("cp5", FIT_NAME), ("swa345", SWA_NAME))
        },
    }
    predictions, tracked = check_worker_outputs(plan, derived, record, worker, base)
    before = {key: sha256(path) for key, path in tracked.items()}
    for name, values in predictions.items():
        if any(not np.isfinite(group).all() for group in values):
            raise ValueError(f"Nonfinite complete export: {name}")
    record["independent_history_check"] = check_training_history(base, worker)
    record["independent_trajectory_check"] = check_trajectory(root, base)
    if before != {key: sha256(path) for key, path in tracked.items()} or worker_sha != sha256(
        worker_path
    ):
        raise ValueError("Downloaded artifacts changed during acceptance")
    record["verified_artifact_hashes"] = before
    record["acceptance_scope"] = "Artifact/recipe acceptance only; not a score or promotion"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return destination


def main() -> None:
    """Accept one successful collection without launching or scoring a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print(accept(root, artifact_root(root, str(args.artifact_root))))


if __name__ == "__main__":
    main()
