from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def per_time_batch_norm(x: torch.Tensor, norm: nn.BatchNorm1d) -> torch.Tensor:
    """Normalize a chunk across batch rows independently at every time position.

    The running moments are the algebraic composition of the original T sequential
    BatchNorm updates. Floating-point reduction order changes, so receiver parity
    uses tolerances; this is not a bitwise-equivalence claim. In particular this
    never treats time as additional batch members during training.
    """
    if x.ndim != 3 or x.shape[-1] != norm.num_features:
        raise ValueError("block BatchNorm requires [batch, time, configured features]")
    if x.shape[1] == 0:
        return x
    use_batch = norm.training or norm.running_mean is None or norm.running_var is None
    if not use_batch:
        # Flattening is safe only in eval mode with already-frozen running moments.
        return F.batch_norm(
            x.reshape(-1, x.shape[-1]),
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            eps=norm.eps,
        ).reshape_as(x)
    if x.shape[0] < 2:
        raise ValueError("training-time BatchNorm requires batch size >= 2")
    variance, mean = torch.var_mean(x, dim=0, correction=0)
    result = (x - mean.unsqueeze(0)) * torch.rsqrt(variance.unsqueeze(0) + norm.eps)
    if norm.affine:
        result = result * norm.weight + norm.bias
    if norm.training and norm.track_running_stats:
        _update_running_moments(norm, mean, variance * (x.shape[0] / (x.shape[0] - 1)))
    return result


@torch.no_grad()
def _update_running_moments(
    norm: nn.BatchNorm1d, means: torch.Tensor, unbiased_variances: torch.Tensor
) -> None:
    """Fold sequential EMA or cumulative-average updates without device synchronization."""
    if norm.running_mean is None or norm.running_var is None or norm.num_batches_tracked is None:
        raise ValueError("tracked BatchNorm must retain all three running-statistic buffers")
    count = means.shape[0]
    means = means.detach().to(norm.running_mean.dtype)
    unbiased_variances = unbiased_variances.detach().to(norm.running_var.dtype)
    if norm.momentum is None:
        previous_count = norm.num_batches_tracked.to(norm.running_mean.dtype)
        denominator = previous_count + count
        norm.running_mean.copy_((norm.running_mean * previous_count + means.sum(0)) / denominator)
        norm.running_var.copy_(
            (norm.running_var * previous_count + unbiased_variances.sum(0)) / denominator
        )
    else:
        decay = 1.0 - norm.momentum
        powers = torch.arange(count - 1, -1, -1, device=means.device, dtype=means.dtype)
        weights = norm.momentum * torch.pow(decay, powers)
        norm.running_mean.mul_(decay**count).add_((means * weights[:, None]).sum(0))
        norm.running_var.mul_(decay**count).add_((unbiased_variances * weights[:, None]).sum(0))
    norm.num_batches_tracked.add_(count)
