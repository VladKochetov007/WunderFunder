from __future__ import annotations

from collections.abc import Mapping

import torch

from wnn.training.sequence_ensemble_config import MIXTURE_MODE


def objective_metrics(
    config: Mapping[str, object], totals: torch.Tensor | None
) -> dict[str, object]:
    """Report actual update-mean objective terms; keep pooled WMSE separately in label_loss."""
    if config.get("objective_mode") is None or totals is None:
        return {}
    rho0, rho1, correlation_loss, wmse, count = totals.detach().cpu().tolist()
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
    if treatment:
        result.update(
            microbatch_correlation_surrogate_t0=rho0 / count,
            microbatch_correlation_surrogate_t1=rho1 / count,
            correlation_prediction_std_floor=0.01,
            correlation_scope="attenuated global-microbatch correlation averaged within each actual accumulation window",
        )
    return result
