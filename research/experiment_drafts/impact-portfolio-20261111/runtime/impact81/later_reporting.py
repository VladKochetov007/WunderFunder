"""Window-normalized auxiliary telemetry for the last four frozen mechanisms."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# JSON telemetry is heterogeneous and interpreted only through the variant contract.
RAW_WIDTH = 48
WIDTH = RAW_WIDTH + 2
VARIANTS = (
    "future_heads",
    "gru_plain",
    "perturbation_control",
    "state_perturbation",
    "nll_transfer",
    "clip_event",
)


def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Use the actual positive mass and a finite divisor for empty cells."""
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(denominator > 0, numerator / safe, torch.zeros_like(numerator))


def window_totals(raw: Tensor, label_mass: Tensor, variant: str) -> Tensor:
    """Keep sufficient sums and the actual globally normalized objective per update."""
    if raw.shape != (RAW_WIDTH,) or variant not in VARIANTS:
        raise ValueError("later auxiliary telemetry shape or variant mismatch")
    del label_mass  # These objectives have their own window-global denominators.
    result = torch.zeros(WIDTH, dtype=torch.float64, device=raw.device)
    if variant == "future_heads":
        numerator, mass = raw[:6].reshape(3, 2).sum(-1), raw[6:12].reshape(3, 2).sum(-1)
        result[0] = 0.1 * _ratio(numerator, mass).mean()
    elif variant in ("state_perturbation", "nll_transfer", "clip_event"):
        coefficient = 0.01 / (0.01**2 + 1e-8) if variant == "state_perturbation" else 0.05
        result[0] = coefficient * _ratio(raw[:2].sum(), raw[6:8].sum())
    result[1:-1] = raw
    result[-1] = 1
    return result


def metrics(totals: Tensor, variant: str) -> dict[str, Any]:
    """Distinguish optimized update means from pooled descriptive diagnostics."""
    if totals.shape != (WIDTH,) or float(totals[-1]) <= 0:
        raise ValueError("later auxiliary summary requires a completed update")
    raw = totals[1:-1].detach()
    result: dict[str, Any] = {
        "variant": variant,
        "weighted_auxiliary_loss": float(totals[0] / totals[-1]),
        "completed_windows": int(totals[-1]),
    }
    if variant == "future_heads":
        numerator, mass = raw[:6].reshape(3, 2), raw[6:12].reshape(3, 2)
        result.update(
            {
                "offsets": [1, 16, 64],
                "pooled_wmse_by_horizon": _ratio(numerator.sum(-1), mass.sum(-1)).cpu().tolist(),
                "pooled_wmse_by_horizon_target": _ratio(numerator, mass).cpu().tolist(),
                "weight_mass_by_horizon_target": mass.cpu().tolist(),
                "eligible_counts_by_horizon_target": raw[12:18].reshape(3, 2).cpu().tolist(),
            }
        )
    elif variant in ("state_perturbation", "nll_transfer", "clip_event"):
        result.update(
            {
                "pooled_auxiliary_mean_by_target": _ratio(raw[:2], raw[6:8]).cpu().tolist(),
                "eligible_counts_by_target": raw[12:14].cpu().tolist(),
            }
        )
        if variant == "state_perturbation":
            result["offsets_in_perturbed_suffix"] = [1, 8, 32]
            result["squared_difference_by_offset_target"] = (
                _ratio(raw[18:24], raw[24:30]).reshape(3, 2).cpu().tolist()
            )
        elif variant == "nll_transfer":
            result["mean_log_variance_by_target"] = _ratio(raw[18:20], raw[6:8]).cpu().tolist()
            result["lower_bound_fraction_by_target"] = _ratio(raw[20:22], raw[6:8]).cpu().tolist()
            result["upper_bound_fraction_by_target"] = _ratio(raw[22:24], raw[6:8]).cpu().tolist()
        else:
            confusion = raw[24:42].reshape(2, 3, 3)
            result["confusion_true_pred_by_target"] = confusion.cpu().tolist()
            result["pooled_ce_by_true_class_target"] = (
                _ratio(raw[42:48].reshape(2, 3), confusion.sum(-1)).cpu().tolist()
            )
    return result
