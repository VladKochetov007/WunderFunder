from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce

from wnn.training.sequence_distributed import DistributedContext, all_ranks_true

EPS = 1e-8


def _sum(values: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Differentiate global SUM on every rank; DDP's later averaging cancels its replication."""
    return all_reduce(values, op=dist.ReduceOp.SUM) if context.process_group else values


def weighted_microbatch_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    context: DistributedContext,
    prediction_std_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two centered FP32 global-microbatch correlations and their mean loss.

    Predictions and targets are clipped to +/-2 after selected raw finite checks;
    weights are abs(clipped target). Detached global minimum anchors make exact
    constants remain exact and avoid cancellation in weighted means. Both mean
    and centered-moment SUM collectives retain autograd, including empty ranks.
    A globally empty row mask contributes zero loss. Constant targets have zero
    correlation gradient. Raw mode also assigns zero gradient at constant p;
    the positive-floor surrogate has a finite useful gradient there. This is not
    full-dataset Pearson or Pearson pooled over an accumulation window. A positive
    prediction_std_floor attenuates rho by min(1,std_p/floor); it is a declared
    training surrogate in clipped-mixture prediction units, never the grader.
    """
    if prediction.shape != target.shape or target.shape[-1] != 2 or mask.shape != target.shape[:-1]:
        raise ValueError("weighted correlation expects (...,2) predictions/targets and row mask")
    if mask.dtype != torch.bool:
        raise TypeError("weighted correlation requires a boolean row mask")
    if prediction_std_floor not in (0.0, 0.01):
        raise ValueError(
            "only raw known-answer Pearson or frozen0.01-std training surrogate is supported"
        )
    finite = bool(torch.isfinite(prediction[mask]).all() and torch.isfinite(target[mask]).all())
    if not all_ranks_true(finite, context):
        raise FloatingPointError("nonfinite selected raw prediction/target in correlation")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        selected = mask[..., None]
        y = torch.where(selected, target.detach().float(), 0).clamp(-2, 2).reshape(-1, 2)
        p = torch.where(selected, prediction.float(), 0).clamp(-2, 2).reshape(-1, 2)
        weight = y.abs()
        positive = weight > 0
        infinity = torch.full_like(y, float("inf"))
        anchors = (
            torch.stack(
                (
                    torch.where(positive, y, infinity).amin(0),
                    torch.where(positive, p.detach(), infinity).amin(0),
                )
            )
            if len(y)
            else torch.full((2, 2), float("inf"), dtype=torch.float32, device=y.device)
        )
        if context.process_group:
            dist.all_reduce(anchors, op=dist.ReduceOp.MIN)
        anchors = torch.where(torch.isfinite(anchors), anchors, 0)
        ys, ps = y - anchors[0], p - anchors[1]
        first = _sum(
            torch.stack(
                (
                    weight.sum(0),
                    (weight * ys).sum(0),
                    (weight * ps).sum(0),
                    mask.float().sum().expand(2),
                )
            ),
            context,
        )
        mass = first[0]
        safe_mass = mass.clamp_min(EPS)
        yc, pc = ys - first[1] / safe_mass, ps - first[2] / safe_mass
        centered = _sum(
            torch.stack(
                (
                    (weight * yc.square()).sum(0),
                    (weight * pc.square()).sum(0),
                    (weight * yc * pc).sum(0),
                )
            ),
            context,
        )
        vy, vp, covariance = centered / safe_mass
        usable = (mass >= EPS) & (vy > EPS**2)
        if prediction_std_floor == 0:
            usable = usable & (vp > EPS**2)
        denominator = (
            vy.clamp_min(EPS**2).sqrt() * vp.clamp_min(max(EPS, prediction_std_floor) ** 2).sqrt()
        )
        correlation = torch.where(usable, (covariance / denominator).clamp(-1, 1), 0)
        loss = torch.where(first[3] > 0, 1 - correlation, 0).mean()
    return correlation, loss
