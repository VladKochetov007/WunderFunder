"""Record the B300 prefit freeze and profile receipt for one fold."""

from __future__ import annotations

# Any represents canonical JSON and checkpoint payloads validated at the boundary.
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audit_model import load_model_state, non_head_hashes
from audit_receipts import write_receipt
from audit_runtime import require_b300_runtime, require_same_device
from audit_validation import validate_embedded_provenance, validate_profile_receipt
from receiver_assets import RECIPE_PATH, Cell, file_sha256, load_json, load_receiver_assets


def prefit(cell: Cell, assets_path: Path, receipt_path: Path, profile_path: Path) -> None:
    """Check B300 receipt, clean parent, parameter freeze, and tensor hashes."""
    assets = load_receiver_assets(assets_path)
    hardware = require_b300_runtime()
    profile = validate_profile_receipt(cell, assets, profile_path)
    require_same_device(profile["execution_runtime"]["process_end"], hardware)
    checkpoint_path = assets.parent_checkpoints[cell.fold]
    checkpoint, model = load_model_state(checkpoint_path)
    from wnn.training.inheritance import validate_campaign_checkpoint

    parent_provenance = load_json(assets.parent_provenance[cell.fold])
    validate_campaign_checkpoint(
        checkpoint,
        SimpleNamespace(
            split_manifest=True,
            split_manifest_sha256=parent_provenance["split_manifest_sha256"],
            checkpoint_epoch=3,
        ),
    )
    validate_embedded_provenance(
        checkpoint,
        parent_provenance,
        (
            "split_manifest_sha256",
            "fit_row_groups",
            "evaluation_row_groups",
            "initial_checkpoint_sha256",
            "propensity_checkpoint_sha256",
            "normalization_identity",
            "run_identity_source_sha256",
        ),
    )
    for name, parameter in model.named_parameters():
        if name.startswith(("inp", "gru")):
            parameter.requires_grad_(False)
    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    all_head = sorted(name for name, _ in model.named_parameters() if name.startswith("head."))
    if not trainable or trainable != all_head:
        raise ValueError(f"inp,gru freeze does not leave exactly the head trainable: {trainable}")
    if checkpoint.get("args", {}).get("split_manifest_sha256") != parent_provenance.get(
        "split_manifest_sha256"
    ):
        raise ValueError("loaded parent checkpoint and provenance split hashes disagree")
    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": "prefit",
        "fold": cell.fold,
        "parent_run": cell.parent_run,
        "parent_checkpoint_sha256": file_sha256(checkpoint_path),
        "parent_provenance_sha256": file_sha256(assets.parent_provenance[cell.fold]),
        "split_manifest_sha256": file_sha256(assets.split_paths[cell.fold]),
        "recipe_sha256": file_sha256(RECIPE_PATH),
        "profile_receipt_sha256": profile["sha256"],
        "profile_execution_runtime": profile["execution_runtime"],
        "profile_memory_evidence": {
            key: profile["canonical_metrics"][key]
            for key in (
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "cuda_free_memory_bytes",
                "cuda_total_memory_bytes",
            )
        },
        "device": hardware,
        "trainable_parameter_names": trainable,
        "frozen_parameter_count": sum(not p.requires_grad for _, p in model.named_parameters()),
        "non_head_tensor_sha256": non_head_hashes(checkpoint["model"]),
        "normalization_identity": parent_provenance.get("normalization_identity"),
        "precision_policy": "historical parent FP16, checkpoint loaded into default FP32 model",
        "validated_fields": [
            "parent_checkpoint_sha256",
            "parent_provenance_sha256",
            "split_manifest_sha256",
            "recipe_sha256",
            "profile_receipt_sha256",
            "profile_execution_runtime",
            "profile_memory_evidence",
            "trainable_parameter_names",
            "non_head_tensor_sha256",
            "normalization_identity",
        ],
    }
    write_receipt(receipt_path, record)
