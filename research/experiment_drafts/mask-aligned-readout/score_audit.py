"""Receipt and child-runtime checks required before a grade can be written."""

from __future__ import annotations

# Any represents stage receipts loaded from JSON and validated before comparison.
from typing import Any

from receiver_assets import (
    RECIPE_PATH,
    REPO_ROOT,
    Cell,
    ReceiverAssets,
    child_checkpoint,
    file_sha256,
    load_json,
    output_root,
)


def check_audit_receipts(
    assets: ReceiverAssets, cell: Cell, arms: tuple[str, ...]
) -> dict[str, Any]:
    """Check live child bytes, trainer identity and immutable audit receipts."""
    root = assets.root
    audit_dir = output_root(root) / "audit"
    recipe = load_json(RECIPE_PATH)
    recipe_cell = next(item for item in recipe["cells"] if item["fold"] == cell.fold)
    recipe_sha = file_sha256(RECIPE_PATH)
    split_sha = file_sha256(assets.split_paths[cell.fold])
    parent_sha = file_sha256(assets.parent_checkpoints[cell.fold])
    parent_provenance_sha = file_sha256(assets.parent_provenance[cell.fold])
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.source_identity import training_source_identity

    current_trainer_source = training_source_identity(
        TrainingEnvironment(source_root=REPO_ROOT, root=root)
    )
    from b300_profile import require_profile_receipts

    profile = require_profile_receipts(assets, (cell,))[cell.fold]
    prefit = audit_dir / f"prefit_{cell.fold}.json"
    if not prefit.is_file():
        raise FileNotFoundError(f"missing prefit audit receipt: {prefit}")
    prefit_data = load_json(prefit)
    prefit_required = {
        "parent_checkpoint_sha256",
        "parent_provenance_sha256",
        "split_manifest_sha256",
        "recipe_sha256",
        "profile_receipt_sha256",
        "trainable_parameter_names",
        "non_head_tensor_sha256",
        "normalization_identity",
    }
    if (
        not prefit_required.issubset(set(prefit_data.get("validated_fields", [])))
        or prefit_data.get("parent_checkpoint_sha256") != parent_sha
        or prefit_data.get("parent_provenance_sha256") != parent_provenance_sha
        or prefit_data.get("split_manifest_sha256") != split_sha
        or prefit_data.get("recipe_sha256") != recipe_sha
        or prefit_data.get("profile_receipt_sha256") != profile["sha256"]
        or not prefit_data.get("trainable_parameter_names")
        or any(not name.startswith("head.") for name in prefit_data["trainable_parameter_names"])
        or not prefit_data.get("non_head_tensor_sha256")
        or not isinstance(prefit_data.get("device"), dict)
        or "b300" not in str(prefit_data.get("device", {}).get("device", "")).lower()
    ):
        raise ValueError(f"{cell.fold} prefit audit lacks head-only and tensor evidence")
    result: dict[str, Any] = {
        "prefit": {
            "path": str(prefit),
            "sha256": file_sha256(prefit),
            "validated_fields": sorted(prefit_required),
        },
        "arms": {},
    }
    for arm in arms:
        path = audit_dir / f"after_{cell.fold}_{arm}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing post-fit audit receipt: {path}")
        receipt = load_json(path)
        child_name = cell.run_name(arm)
        child_path = child_checkpoint(root, cell, arm)
        provenance_path = child_path.with_name("provenance.json")
        if not child_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"{cell.fold}/{arm} checkpoint or provenance is missing")
        provenance = load_json(provenance_path)
        child_sha = file_sha256(child_path)
        child_provenance_sha = file_sha256(provenance_path)
        source_identity = provenance.get("trainer_implementation")
        required_fields = {
            "child_checkpoint_sha256",
            "child_provenance_sha256",
            "parent_checkpoint_sha256",
            "parent_provenance_sha256",
            "split_manifest_sha256",
            "recipe_sha256",
            "trainer_implementation_sha256",
            "prefit_receipt_sha256",
            "profile_receipt_sha256",
            "profile_memory_evidence",
            "receiver_hardware_observed_after_fit",
            "training_process_runtime",
            "fixed_fit_endpoint",
            "training_precision_bound_by_trainer_runtime",
            "non_head_tensor_sha256",
            "normalization_identity",
            "frozen_tensor_unchanged",
        }
        args = provenance.get("args", {})
        runtime = provenance.get("execution_runtime")
        observed = receipt.get("receiver_hardware_observed_after_fit", {})
        if (
            not required_fields.issubset(set(receipt.get("validated_fields", [])))
            or receipt.get("fold") != cell.fold
            or receipt.get("arm") != arm
            or receipt.get("child_run") != child_name
            or receipt.get("child_checkpoint_sha256") != child_sha
            or receipt.get("child_provenance_sha256") != child_provenance_sha
            or receipt.get("parent_checkpoint_sha256") != parent_sha
            or receipt.get("parent_provenance_sha256") != parent_provenance_sha
            or receipt.get("split_manifest_sha256") != split_sha
            or receipt.get("recipe_sha256") != recipe_sha
            or receipt.get("prefit_receipt_sha256") != file_sha256(prefit)
            or receipt.get("profile_receipt_sha256") != profile["sha256"]
            or receipt.get("profile_memory_evidence") != prefit_data.get("profile_memory_evidence")
            or not isinstance(args, dict)
            or not isinstance(source_identity, dict)
            or receipt.get("trainer_implementation_sha256") != source_identity.get("sha256")
            or source_identity != current_trainer_source
            or provenance.get("initial_checkpoint_sha256") != parent_sha
            or provenance.get("split_manifest_sha256") != recipe_cell["split_manifest_sha256"]
            or args.get("freeze") != "inp,gru"
            or args.get("mask_weight") != (arm == "is_scored_weighted")
            or args.get("propensity") != ""
            or args.get("valid_true_mask") != -1.0
            or args.get("precision") != "fp32"
            or args.get("device") != "cuda"
            or args.get("capture_execution_runtime") is not True
            or not isinstance(runtime, dict)
            or receipt.get("training_process_runtime") != runtime
            or "b300" not in str(runtime.get("device", "")).lower()
            or runtime.get("precision") != "fp32"
            or runtime.get("matmul_tf32") is not False
            or runtime.get("cudnn_tf32") is not False
            or runtime.get("autocast_policy") != "disabled"
            or runtime.get("autocast_active_at_capture") is not False
            or receipt.get("non_head_tensor_sha256") != prefit_data["non_head_tensor_sha256"]
            or receipt.get("normalization_identity") != prefit_data["normalization_identity"]
            or receipt.get("expected_trainable_parameter_names_by_canonical_freeze")
            != prefit_data.get("trainable_parameter_names")
            or not receipt.get("frozen_tensor_unchanged")
            or not isinstance(observed, dict)
            or "b300" not in str(observed.get("device", "")).lower()
            or receipt.get("training_precision_bound_by_trainer_runtime") != "strict fp32"
            or any(
                runtime.get(key) != observed.get(key)
                for key in (
                    "device",
                    "device_uuid",
                    "nvidia_smi_device_uuid",
                    "driver_version",
                    "compute_capability",
                    "torch_version",
                    "torch_cuda_version",
                )
            )
        ):
            raise ValueError(f"{cell.fold}/{arm} frozen tensor audit did not pass")
        result["arms"][arm] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "validated_fields": sorted(required_fields),
            "checkpoint_sha256": child_sha,
            "provenance_sha256": child_provenance_sha,
            "trainer_implementation": source_identity,
            "training_process_runtime": runtime,
        }
    treatment_receipt = load_json(audit_dir / f"after_{cell.fold}_is_scored_weighted.json")
    if not treatment_receipt.get("paired_fit_args_match_except_mask_weight"):
        raise ValueError(f"{cell.fold} paired fit settings lack an exact comparison receipt")
    return result
