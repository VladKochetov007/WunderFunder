from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from kaggle_h512_f1_worker_support import (
    EPOCH_CHUNK_UPDATES,
    EXPECTED,
    FIT_NAME,
    SWA_NAME,
    TOTAL_CHUNK_UPDATES,
    read_object,
    sha256,
)


def validate_fit(
    root: Path,
    manifest_sha256: str,
    expected_train_groups: list[int],
    expected_valid_groups: list[int],
    expected_evaluation_groups: list[int],
) -> dict[str, Any]:
    """Audit the cold fixed CP5 fit, membership, normalization and update counts."""
    import torch

    runtime_tf32_flags = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    if any(runtime_tf32_flags.values()):
        raise ValueError("Full-fit runtime enabled TF32 despite the strict-FP32 requirement")

    run_dir = root / "artifacts/runs" / FIT_NAME
    provenance = read_object(run_dir / "provenance.json")
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or len(history) != 6
        or [record.get("epoch") for record in history] != list(range(6))
        or any(record.get("evaluation") != "withheld" for record in history)
    ):
        raise ValueError("Full fit did not produce six withheld-evaluation epoch records")
    args = provenance.get("args")
    normalization = provenance.get("normalization_identity")
    initial_sha = provenance.get("initial_parameter_sha256")
    if not isinstance(args, dict) or args.get("seed") != 1004 or args.get("init"):
        raise ValueError("Fit provenance is not a fresh seed-1004 initialization")
    expected_args = {
        "hidden": 512,
        "layers": 1,
        "valid_fold": 1,
        "batch": 16,
        "chunk": 2000,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "precision": "fp32",
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "train_seqs": 10607,
        "no_eval_during_fit": True,
        "resume_state": None,
    }
    if any(args.get(name) != value for name, value in expected_args.items()):
        raise ValueError("Fit trainer arguments differ from the frozen seed-1004 recipe")
    if args.get("raw_only") is not False or args.get("research_features") != "none":
        raise ValueError("Fit provenance changed the frozen 211-input representation")
    if (
        args.get("split_manifest_sha256") != manifest_sha256
        or provenance.get("split_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Fit provenance is not bound to the derived F1 view")
    if provenance.get("initial_checkpoint_sha256") is not None:
        raise ValueError("Full fit unexpectedly inherited an initializer")
    if not isinstance(initial_sha, str) or len(initial_sha) != 64:
        raise ValueError("Missing fresh initial parameter identity")
    if not isinstance(normalization, dict) or set(normalization) != {"mean", "std"}:
        raise ValueError("Missing fixed input normalization identity")
    if any(
        value.get("shape") != [211]
        or value.get("dtype") != "torch.float32"
        or len(value.get("bytes_sha256", "")) != 64
        for value in normalization.values()
    ):
        raise ValueError("Normalization buffers do not match the 211-input FP32 representation")
    fit_groups = provenance.get("fit_row_groups")
    if not isinstance(fit_groups, dict):
        raise ValueError("Fit provenance lacks source-qualified group membership")
    if set(map(int, fit_groups.get("train", []))) != set(expected_train_groups):
        raise ValueError("Fit training groups differ from canonical F1 membership")
    if list(map(int, fit_groups.get("valid", []))) != expected_valid_groups:
        raise ValueError("Fit validation groups differ from canonical F1 membership")
    if set(map(int, provenance.get("evaluation_row_groups", []))) != set(
        expected_evaluation_groups
    ):
        raise ValueError("Fit provenance evaluation groups differ from canonical F1 membership")
    diagnostics = provenance.get("update_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 6:
        raise ValueError("Fit update diagnostics do not contain six epochs")
    for epoch, item in enumerate(diagnostics):
        if item.get("epoch_index") != epoch or item.get("normalization_identity") != normalization:
            raise ValueError("Epoch normalization identity changed during fitting")
        counts = item.get("epoch", {})
        if (
            counts.get("optimizer_attempts") != EPOCH_CHUNK_UPDATES
            or counts.get("optimizer_completed_updates") != EPOCH_CHUNK_UPDATES
            or counts.get("amp_skipped_updates") != 0
            or counts.get("completed_batches") != 746
        ):
            raise ValueError(f"Unexpected optimizer update count at epoch {epoch}")
    cumulative = provenance.get("cumulative_updates", {})
    if (
        cumulative.get("optimizer_attempts") != TOTAL_CHUNK_UPDATES
        or cumulative.get("optimizer_completed_updates") != TOTAL_CHUNK_UPDATES
        or cumulative.get("amp_skipped_updates") != 0
    ):
        raise ValueError("Cumulative optimizer count differs from 44,760 finite updates")
    checkpoint_path = run_dir / "model_ep5.pt"
    if not checkpoint_path.is_file():
        raise ValueError("Fixed CP5 checkpoint is missing")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != 5 or checkpoint.get("n_in") != 211:
        raise ValueError("CP5 checkpoint epoch or input width differs")
    if checkpoint.get("args", {}).get("split_manifest_sha256") != manifest_sha256:
        raise ValueError("CP5 checkpoint split identity differs")
    checkpoint_provenance = checkpoint.get("provenance", {})
    if (
        checkpoint_provenance.get("initial_parameter_sha256") != initial_sha
        or checkpoint_provenance.get("normalization_identity") != normalization
    ):
        raise ValueError("CP5 checkpoint initialization or normalization identity differs")
    return {
        "checkpoint": f"artifacts/runs/{FIT_NAME}/model_ep5.pt",
        "checkpoint_sha256": sha256(checkpoint_path),
        "fit_provenance_sha256": sha256(run_dir / "provenance.json"),
        "initial_parameter_sha256": initial_sha,
        "initialization": "fresh seed 1004; no init checkpoint or resume state",
        "profile_fit_tensor_identity_comparison": "not measured; runpy creates a fresh trainer run and canonical validation resets the seed",
        "normalization_identity": normalization,
        "normalization_invariant_across_epochs": True,
        "runtime_tf32_flags_after_fit": runtime_tf32_flags,
        "input_width": 211,
        "epoch_count": len(history),
        "optimizer_chunk_updates": TOTAL_CHUNK_UPDATES,
        "split_manifest_sha256": manifest_sha256,
    }


def derive_swa345(root: Path, fit_audit: dict[str, Any]) -> dict[str, Any]:
    """Average immutable CP3/4/5 states with canonical mean_state, and receipt it."""
    import torch

    sys.path.insert(0, str(root))
    from scripts.average_trajectory_checkpoints import mean_state

    source = root / "artifacts/runs" / FIT_NAME
    paths = [source / f"model_ep{epoch}.pt" for epoch in (3, 4, 5)]
    hashes = [sha256(path) for path in paths]
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    ancestry_reference = checkpoints[0].get("provenance", {})
    cp5_provenance = checkpoints[-1].get("provenance", {})
    for epoch, checkpoint in zip((3, 4, 5), checkpoints, strict=True):
        provenance = checkpoint.get("provenance", {})
        if checkpoint.get("epoch") != epoch or checkpoint.get("n_in") != 211:
            raise ValueError("SWA source checkpoint epoch or input width differs")
        if checkpoint.get("args") != checkpoints[0].get("args"):
            raise ValueError("SWA source checkpoints have different trainer arguments")
        for field in (
            "split_manifest_sha256",
            "fit_membership_identity",
            "initial_parameter_sha256",
            "normalization_identity",
        ):
            if provenance.get(field) != ancestry_reference.get(field):
                raise ValueError(f"SWA source ancestry differs: {field}")
        if provenance.get("initial_parameter_sha256") != fit_audit["initial_parameter_sha256"]:
            raise ValueError("SWA source initialization differs from audited CP5")
    averaged = copy.deepcopy(checkpoints[-1])
    averaged["model"] = mean_state([checkpoint["model"] for checkpoint in checkpoints])
    derivation = {
        "schema": "wnn.kaggle-h512-swa345-weight-derivation.v1",
        "status": "DERIVED_UNGRADED",
        "method": "equal_parameter_mean_epochs_3_4_5",
        "source_run": FIT_NAME,
        "destination_run": SWA_NAME,
        "source_checkpoints": [
            {
                "epoch": epoch,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
            }
            for epoch, path, digest in zip((3, 4, 5), paths, hashes, strict=True)
        ],
        "input_manifest": {
            "path": "research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_profile_view.json",
            "sha256": fit_audit["split_manifest_sha256"],
        },
        "original_canonical_manifest": {
            "path": "research/campaign_2026-09-29/artifacts/splits/fold1.json",
            "sha256": EXPECTED["original_manifest_sha256"],
        },
        "initial_parameter_sha256": fit_audit["initial_parameter_sha256"],
        "normalization_identity": fit_audit["normalization_identity"],
        "training_run": False,
        "quality_evaluated": False,
    }
    averaged["weight_derivation"] = derivation
    averaged["provenance"]["weight_derivation"] = derivation
    destination = root / "artifacts/runs" / SWA_NAME
    destination.mkdir(parents=True, exist_ok=False)
    model_path = destination / "model_ep5.pt"
    torch.save(averaged, model_path)
    (destination / "model.pt").symlink_to(model_path.name)
    provenance = {
        **cp5_provenance,
        "weight_derivation": derivation,
        "checkpoint_sha256": {
            "model_ep5.pt": sha256(model_path),
            "model.pt": sha256(model_path),
        },
    }
    provenance_path = destination / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    model_link = destination / "model.pt"
    receipt = {
        **derivation,
        "outputs": {
            "model_ep5.pt": {
                "path": model_path.relative_to(root).as_posix(),
                "sha256": sha256(model_path),
                "size_bytes": model_path.stat().st_size,
            },
            "provenance.json": {
                "path": provenance_path.relative_to(root).as_posix(),
                "sha256": sha256(provenance_path),
                "size_bytes": provenance_path.stat().st_size,
            },
            "model.pt": {
                "path": model_link.relative_to(root).as_posix(),
                "sha256": sha256(model_link),
                "size_bytes": model_link.stat().st_size,
            },
        },
    }
    receipt_path = destination / "weight_derivation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "run": SWA_NAME,
        "model_ep5_path": model_path.relative_to(root).as_posix(),
        "model_ep5_sha256": sha256(model_path),
        "provenance_path": provenance_path.relative_to(root).as_posix(),
        "provenance_sha256": sha256(provenance_path),
        "weight_derivation_receipt_path": receipt_path.relative_to(root).as_posix(),
        "weight_derivation_receipt_sha256": sha256(receipt_path),
        "derivation": derivation,
    }
