from __future__ import annotations

from collections.abc import Mapping

import torch

from wnn.training.sequence_ensemble_config import (
    MIXTURE_MODE,
    STANDALONE_LABELS_KD_MODE,
)


def objective_metrics(
    config: Mapping[str, object], totals: torch.Tensor | None
) -> dict[str, object]:
    """Report actual update-mean objective terms; keep pooled WMSE separately in label_loss."""
    if config.get("objective_mode") is None or totals is None:
        return {}
    values = totals.detach().cpu().tolist()
    rho0, rho1, correlation_loss, wmse = values[:4]
    kd_wmse = values[4] if len(values) > 5 else 0.0
    count = values[5] if len(values) > 5 else values[4]
    if count <= 0:
        return {}
    correlation_loss, wmse = correlation_loss / count, wmse / count
    treatment = config["objective_mode"] == MIXTURE_MODE
    wmse_weight, correlation_weight = (0.5, 0.5) if treatment else (1.0, 0.0)
    result = {
        "objective_mode": config["objective_mode"],
        "objective_wmse": wmse,
        "objective_correlation_loss": correlation_loss,
        "wmse_objective_weight": wmse_weight,
        "correlation_objective_weight": correlation_weight,
        "total_loss": wmse_weight * wmse + correlation_weight * correlation_loss,
        "objective_aggregation": "mean of actual optimizer-window objectives",
        "label_loss_is_curvature_scaled": False,
    }
    if config.get("supervised_target") is not None:
        result["supervised_target"] = config["supervised_target"]
        result["objective_wmse_scope"] = "selected target numerator / both-target denominator"
    if config.get("objective_mode") == STANDALONE_LABELS_KD_MODE:
        kd_weight = float(config["kd_weight"])
        result.update(
            objective_label_wmse=wmse,
            objective_kd_wmse=kd_wmse / count,
            objective_kd_weight=kd_weight,
            objective_shared_denominator="sum_abs_clipped_label_targets",
            total_loss=wmse + kd_weight * kd_wmse / count,
        )
    if treatment:
        result.update(
            microbatch_correlation_surrogate_t0=rho0 / count,
            microbatch_correlation_surrogate_t1=rho1 / count,
            correlation_prediction_std_floor=0.01,
            correlation_scope="attenuated global-microbatch correlation averaged within each actual accumulation window",
        )
    return result
