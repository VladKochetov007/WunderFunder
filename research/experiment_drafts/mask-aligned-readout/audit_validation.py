"""Exact checkpoint, membership, and inherited-input validation helpers."""

from __future__ import annotations

import json

# Any represents schema-checked campaign and checkpoint records from canonical sources.
from pathlib import Path
from typing import Any

import numpy as np
from receiver_assets import REPO_ROOT, Cell, ReceiverAssets, file_sha256, load_json

INHERITED_FIELDS = (
    "hidden",
    "encoder_width",
    "recurrent_width",
    "readout_width",
    "head_layers",
    "layers",
    "dropout",
    "arch",
    "decay_init",
    "memory_arm",
    "cell",
    "enc",
    "tok_dim",
    "att_layers",
    "conv_k",
    "readout_activation",
    "tcn_mix",
    "event_hidden",
    "expert_hidden",
    "num_experts",
    "specialist_route",
    "hash_seed",
    "adapter_rank",
    "adapter_mode",
    "prefix_rows",
    "online_norm",
    "raw_only",
    "heuristics",
    "research_features",
    "research_feature_control",
    "privileged_mode",
    "privileged_horizon",
    "sentinel_values",
    "sentinel_indicators",
    "sentinel_base_width",
    "prop_input",
    "aux",
    "aux_struct",
    "aux_next",
    "aux_next_k",
    "aux_tier",
    "aux_mask",
    "aux_sign",
    "aux_quote",
    "quote_cache_root",
)


def validate_embedded_provenance(
    checkpoint: dict[str, Any], provenance: dict[str, Any], keys: tuple[str, ...]
) -> None:
    """Compare stable source/lineage fields in checkpoint and companion JSON."""
    embedded = checkpoint.get("provenance", {})
    for key in keys:
        if embedded.get(key) != provenance.get(key):
            raise ValueError(f"checkpoint and companion provenance disagree on {key}")
    if checkpoint.get("args") != provenance.get("args"):
        raise ValueError("checkpoint and companion provenance args differ")


def validate_child_args(
    cell: Cell, arm: str, split_sha: str, checkpoint: dict[str, Any], provenance: dict[str, Any]
) -> None:
    """Check fit source, paired seed/parent, loss weighting, and checkpoint policy."""
    args = checkpoint.get("args", {})
    if provenance.get("args") != args:
        raise ValueError("child checkpoint args and provenance args differ")
    expected = {
        "name": cell.run_name(arm),
        "init": f"{cell.parent_run}@ep3",
        "inherit_init_preprocessing": True,
        "split_manifest_sha256": split_sha,
        "valid_fold": cell.fold_index,
        "train_source": "valid",
        "freeze": "inp,gru",
        "precision": "fp32",
        "mask_weight": arm == "is_scored_weighted",
        "propensity": "",
        "valid_true_mask": -1.0,
        "loss": "wmse",
        "aux_next": 0.3,
        "aux_struct": 0.3,
        "since_weight": 32.0,
        "since_floor": 0.25,
        "episode_weight": 0.0,
        "class_weight": "",
        "optim": "adamw",
        "lr": 0.0005,
        "wd": 0.01,
        "sched": "onecycle",
        "epochs": cell.epochs,
        "batch": 32,
        "chunk": 2000,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": cell.checkpoint_epoch,
        "resume_state": None,
        "save_training_state": False,
        "no_eval_during_fit": True,
        "capture_execution_runtime": True,
        "seed": cell.seed,
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{cell.fold}/{arm} arguments differ from the fixed experiment recipe")
    if provenance.get("initial_checkpoint_sha256") is None:
        raise ValueError("child provenance lacks its loaded initializer hash")


def validate_child_membership(
    cell: Cell, assets: ReceiverAssets, provenance: dict[str, Any]
) -> None:
    """Require exact valid-side fit membership and full held-out evaluation coverage."""
    split = assets.splits[cell.fold]
    fit = provenance.get("fit_row_groups", {})
    valid_pool = np.sort(np.asarray(split.valid_groups, dtype=np.int64))
    expected_fit = np.random.default_rng(cell.seed).permutation(valid_pool).tolist()
    if fit.get("train") != [] or fit.get("valid") != expected_fit:
        raise ValueError(
            f"{cell.fold} child fit membership differs from its seeded full valid pool"
        )
    observed_eval = provenance.get("evaluation_row_groups", [])
    if len(observed_eval) != len(split.evaluation_groups) or set(observed_eval) != set(
        split.evaluation_groups
    ):
        raise ValueError(
            f"{cell.fold} child evaluation membership is not the complete held-out fold"
        )


def validate_fixed_endpoint(
    cell: Cell,
    arm: str,
    assets: ReceiverAssets,
    checkpoint: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, int]:
    """Verify every full fit epoch and the recipe-selected fixed checkpoint."""
    args = checkpoint.get("args", {})
    if (
        checkpoint.get("epoch") != cell.checkpoint_epoch
        or provenance.get("selected_epoch") != cell.checkpoint_epoch
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != cell.checkpoint_epoch
        or args.get("epochs") != cell.epochs
        or args.get("max_batches") != 0
        or args.get("profile_steps") != 0
        or args.get("no_eval_during_fit") is not True
    ):
        raise ValueError(
            f"{cell.fold}/{arm} checkpoint is not the complete fixed epoch-{cell.checkpoint_epoch} fit"
        )
    split = assets.splits[cell.fold]
    expected_batches = len(split.valid_groups) // 32
    if expected_batches <= 0:
        raise ValueError(f"{cell.fold} full-valid fit has no complete training batch")
    update_rows = int(provenance.get("training_clocks", {}).get("update_rows", 0))
    sequence_rows = 20_000
    if update_rows != 2_000 or sequence_rows % update_rows:
        raise ValueError(f"{cell.fold} update clock differs from the fixed 2,000-row recipe")
    expected_attempts_per_epoch = expected_batches * (sequence_rows // update_rows)
    expected_total_attempts = expected_attempts_per_epoch * cell.epochs
    diagnostics = provenance.get("update_diagnostics", [])
    if not isinstance(diagnostics, list) or len(diagnostics) != cell.epochs:
        raise ValueError(f"{cell.fold}/{arm} provenance lacks all {cell.epochs} fit epochs")
    for epoch_index, diagnostic in enumerate(diagnostics):
        epoch = diagnostic.get("epoch", {})
        if (
            diagnostic.get("epoch_index") != epoch_index
            or epoch.get("optimizer_attempts") != expected_attempts_per_epoch
            or epoch.get("optimizer_completed_updates") != expected_attempts_per_epoch
            or epoch.get("amp_skipped_updates") != 0
            or epoch.get("completed_batches") != expected_batches
            or diagnostic.get("normalization_identity") != provenance.get("normalization_identity")
        ):
            raise ValueError(
                f"{cell.fold}/{arm} epoch {epoch_index} missed its full-data update budget"
            )
    history = json.loads(
        (assets.root / "artifacts" / "runs" / cell.run_name(arm) / "history.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(history, list) or len(history) != cell.epochs:
        raise ValueError(f"{cell.fold}/{arm} fit history does not contain all {cell.epochs} epochs")
    for epoch_index, record in enumerate(history):
        summary = record.get("fit_summary", {})
        updates = summary.get("optimizer_updates", {})
        if (
            record.get("epoch") != epoch_index
            or record.get("evaluation") != "withheld"
            or summary.get("optimizer_windows") != expected_attempts_per_epoch
            or updates.get("optimizer_attempts") != expected_attempts_per_epoch
            or updates.get("optimizer_completed_updates") != expected_attempts_per_epoch
            or updates.get("amp_skipped_updates") != 0
        ):
            raise ValueError(
                f"{cell.fold}/{arm} history fails the full no-eval epoch {epoch_index} endpoint"
            )
    return {
        "epochs": cell.epochs,
        "checkpoint_epoch": cell.checkpoint_epoch,
        "expected_batches_per_epoch": expected_batches,
        "expected_optimizer_attempts_per_epoch": expected_attempts_per_epoch,
        "expected_total_optimizer_attempts": expected_total_attempts,
    }


def compare_inherited_identity(
    cell: Cell,
    assets: ReceiverAssets,
    parent: dict[str, Any],
    child: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """Bind model shape, inherited input arguments, and normalization to the parent."""
    parent_args, child_args = parent.get("args", {}), child.get("args", {})
    if any(parent_args.get(key) != child_args.get(key) for key in INHERITED_FIELDS):
        raise ValueError(f"{cell.fold} child changed inherited parent architecture or inputs")
    if child.get("n_in") != parent.get("n_in"):
        raise ValueError(f"{cell.fold} child input width differs from its parent")
    parent_provenance = load_json(assets.parent_provenance[cell.fold])
    if provenance.get("normalization_identity") != parent_provenance.get("normalization_identity"):
        raise ValueError(f"{cell.fold} child normalization differs from its parent")


def validate_trainer_provenance(
    cell: Cell, assets: ReceiverAssets, provenance: dict[str, Any]
) -> None:
    """Require exact current trainer source identity in each child checkpoint."""
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.source_identity import training_source_identity

    identity = provenance.get("trainer_implementation")
    if not isinstance(identity, dict) or not identity.get("sha256") or not identity.get("files"):
        raise ValueError(f"{cell.fold} child lacks canonical trainer source identity")
    expected = training_source_identity(
        TrainingEnvironment(source_root=REPO_ROOT, root=assets.root)
    )
    if identity != expected:
        raise ValueError(f"{cell.fold} child was produced by a different trainer implementation")
    parent = load_json(assets.parent_provenance[cell.fold])
    historical_source_sha = parent.get("run_identity_source_sha256")
    if (
        not isinstance(historical_source_sha, str)
        or len(historical_source_sha) != 64
        or any(character not in "0123456789abcdef" for character in historical_source_sha.lower())
    ):
        raise ValueError(f"{cell.fold} parent lacks its recorded historical run-identity hash")


def validate_training_runtime(cell: Cell, provenance: dict[str, Any]) -> dict[str, Any]:
    """Require strict-FP32 B300 telemetry recorded by the actual trainer process."""
    runtime = provenance.get("execution_runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{cell.fold} child provenance lacks trainer-process runtime evidence")
    if (
        runtime.get("schema_version") != 1
        or runtime.get("capture_stage") != "trainer_process_start"
        or "b300" not in str(runtime.get("device", "")).lower()
        or not runtime.get("device_uuid")
        or not runtime.get("driver_version")
        or runtime.get("precision") != "fp32"
        or runtime.get("matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("autocast_policy") != "disabled"
        or runtime.get("autocast_active_at_capture") is not False
        or runtime.get("default_dtype") != "torch.float32"
        or any(
            not isinstance(runtime.get(key), int) or runtime[key] <= 0
            for key in (
                "torch_total_memory_bytes",
                "nvidia_smi_total_memory_bytes",
                "cuda_runtime_total_memory_bytes",
            )
        )
    ):
        raise ValueError(f"{cell.fold} child lacks verified strict-FP32 B300 runtime telemetry")
    return runtime


def validate_profile_receipt(cell: Cell, assets: ReceiverAssets, path: Path) -> dict[str, Any]:
    """Require the canonical, source-current B300 profile selected for this fold."""
    from b300_profile import _require_receipt, profile_receipt_path

    expected_path = profile_receipt_path(assets.root, cell).resolve()
    if path.resolve() != expected_path:
        raise ValueError(f"{cell.fold} prefit must bind its canonical profile receipt")
    receipt = _require_receipt(assets, cell)
    if file_sha256(expected_path) != receipt["sha256"]:
        raise ValueError(f"{cell.fold} profile receipt changed during prefit admission")
    return receipt
