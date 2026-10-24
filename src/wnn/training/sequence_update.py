"""Strict-FP32 CUDA update boundary with isolated optimizer timing."""

from __future__ import annotations

import torch

from wnn.training.sequence_distributed import DistributedContext, all_ranks_true


def finish_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    gradient_clip: float,
) -> tuple[torch.Tensor, float]:
    """Agree on finite gradients, measure preclip norm, step, and validate parameters."""
    gradient_flags = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradients_finite = bool(torch.stack(gradient_flags).all()) if gradient_flags else False
    if not all_ranks_true(gradients_finite, context):
        raise FloatingPointError("nonfinite gradients detected consistently across torchrun ranks")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=False
    )
    if not all_ranks_true(bool(torch.isfinite(norm)), context):
        raise FloatingPointError("nonfinite preclip gradient norm detected across torchrun ranks")
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    optimizer.step()
    finished.record()
    scheduler.step()
    parameter_flags = [torch.isfinite(parameter).all() for parameter in model.parameters()]
    parameters_finite = bool(torch.stack(parameter_flags).all()) if parameter_flags else False
    if not all_ranks_true(parameters_finite, context):
        raise FloatingPointError("nonfinite model parameter detected after optimizer update")
    optimizer.zero_grad(set_to_none=True)
    return norm.detach(), started.elapsed_time(finished) / 1000.0
