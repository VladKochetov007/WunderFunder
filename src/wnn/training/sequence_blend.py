from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from wnn.splits import CampaignSplit

if TYPE_CHECKING:
    from wnn.training.sequence_reference_cache import ReferenceCache


def validate_blend_config(config: dict[str, Any]) -> None:
    """Validate the opt-in frozen-reference objective without changing old recipes."""
    if "blend_objective_alpha" not in config:
        return
    alpha = config["blend_objective_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)):
        raise ValueError("blend_objective_alpha must be numeric")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("blend_objective_alpha must be finite and in [0, 1]")
    if alpha == 0:
        config.pop("blend_objective_alpha")
        return
    if config["model"]["architecture"] != "transformer":
        raise ValueError("blend objective currently requires a Transformer")
    if config["kd_weight"] != 0 or config["hint_weight"] != 0 or config["model"]["hint_dim"] != 0:
        raise ValueError("blend objective cannot also use KD or hidden hints")


def fit_reference(
    config: dict[str, Any],
    path: Path | None,
    teacher_cache: Path | None,
    split: CampaignSplit,
    members: list[tuple[str, int]],
) -> ReferenceCache | None:
    """Require an explicit fitting reference only for the composite-loss treatment."""
    from wnn.training.sequence_reference_cache import ReferenceCache

    needed = config.get("blend_objective_alpha", 0) > 0
    if needed != (path is not None):
        raise ValueError("blend objective requires --reference-cache; controls must omit it")
    if path is not None and teacher_cache is not None:
        raise ValueError("reference cache and KD teacher cache are mutually exclusive")
    return ReferenceCache(path, split, members, membership="fit") if path is not None else None


def compose_prediction(
    prediction: torch.Tensor, reference: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Mix un-clipped FP32 head and frozen reference values before official clipping."""
    return (1.0 - alpha) * reference.detach().float() + alpha * prediction.float()


def blend_loss_metrics(config: Mapping[str, object], label_loss: float) -> dict[str, float | bool]:
    """Expose unscaled composite WMSE separately from the curvature-matched objective."""
    alpha = float(config.get("blend_objective_alpha", 0.0))
    if alpha <= 0:
        return {}
    return {
        "blend_objective_alpha": alpha,
        "composite_wmse": label_loss * alpha**2,
        "label_loss_is_curvature_scaled": True,
    }
