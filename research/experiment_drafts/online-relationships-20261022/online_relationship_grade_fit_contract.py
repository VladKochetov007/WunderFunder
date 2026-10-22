"""Checks for row weights, inherited normalization and fixed training records."""

from __future__ import annotations

import hashlib
from typing import Any  # JSON metadata is validated at runtime.

import numpy as np
from online_relationship_grade_contract import ROW_WEIGHT_CONTRACT, ROWS


def verify_row_weight_contract(run: str, value: dict[str, Any]) -> None:
    """Bind candidate provenance to the shared canonical FP32 propensity weights."""
    if value.get("args") != ROW_WEIGHT_CONTRACT:
        raise ValueError(f"{run}: fit row-weight arguments differ from the recipe")
    if value.get("batch_semantics") != (
        "one complete sequence per main_row_weights call; true-mask mass scale is sequence-local"
    ):
        raise ValueError(f"{run}: fitting normalization is not sequence-local")


def verify_parent_normalization(fold: dict[str, Any], normalization: dict[str, Any]) -> None:
    """Check the readout inherits the exact selected parent's mean and standard deviation."""
    parent = fold["parent"]
    for field, identity in (("parent_mean", "mean_identity"), ("parent_std", "std_identity")):
        value = np.asarray(normalization.get(field), dtype="<f4")
        if value.shape != (211,) or not np.isfinite(value).all():
            raise ValueError(f"F{fold['fold']} candidate {field} is malformed")
        if hashlib.sha256(value.tobytes()).hexdigest() != parent[identity]:
            raise ValueError(f"F{fold['fold']} candidate {field} differs from the frozen parent")
        if field == "parent_std" and np.any(value <= 0):
            raise ValueError(f"F{fold['fold']} candidate parent std is not positive")


def verify_fit_records(
    run: str,
    recipe: dict[str, Any],
    cell: dict[str, Any],
    fit_refs: list[dict[str, Any]],
    fit: dict[str, Any],
    normalizer: dict[str, Any],
    normalization: dict[str, Any],
) -> None:
    """Require the exact three-epoch, fixed-order full-membership optimization receipt."""
    records = fit.get("epoch_records")
    expected_rows = len(fit_refs) * ROWS
    expected_updates = (expected_rows + int(recipe["readout"]["batch_rows"]) - 1) // int(
        recipe["readout"]["batch_rows"]
    )
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError(f"{run}: provenance must contain three epoch records")
    for epoch_index, record in enumerate(records):
        if (
            record.get("epoch") != epoch_index
            or record.get("updates") != expected_updates
            or record.get("finite_updates") != expected_updates
            or record.get("fit_sequences") != len(fit_refs)
            or record.get("fit_rows") != expected_rows
        ):
            raise ValueError(f"{run}: epoch {epoch_index} lacks the complete fixed update budget")
        for metric in (
            "mean_batch_primary_pair_wmse",
            "mean_preclip_gradient_norm",
            "max_preclip_gradient_norm",
        ):
            if not np.isfinite(float(record.get(metric, np.nan))):
                raise ValueError(f"{run}: epoch {epoch_index} has a nonfinite {metric}")
    metadata = fit.get("metadata", {})
    expected_hparams = {
        "arm": str(cell["arm"]),
        "lag": int(cell["lag"]),
        "seed": int(cell["seed"]),
        "epochs": 3,
        "fixed_checkpoint_epoch": 2,
        "batch_rows": int(recipe["readout"]["batch_rows"]),
        "learning_rate": float(recipe["readout"]["lr"]),
        "weight_decay": float(recipe["readout"]["weight_decay"]),
        "gradient_clip": float(recipe["readout"]["clip"]),
    }
    if any(metadata.get(key) != value for key, value in expected_hparams.items()):
        raise ValueError(f"{run}: fitter metadata differs from the frozen head recipe")
    if metadata.get("relationship_normalizer") != normalizer:
        raise ValueError(f"{run}: fitter metadata has a different relationship normalizer")
    if metadata.get("parent_mean") != normalization.get("parent_mean") or metadata.get(
        "parent_std"
    ) != normalization.get("parent_std"):
        raise ValueError(f"{run}: fitter metadata does not inherit the frozen parent normalization")
