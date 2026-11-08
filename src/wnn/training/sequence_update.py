"""Strict-FP32 CUDA update boundary with isolated optimizer timing."""

from __future__ import annotations

import time

import torch

from wnn.training.sequence_distributed import DistributedContext, all_ranks_true
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_update_telemetry import (
    capture_optimizer_update,
    finalize_optimizer_update,
    optimizer_gradient_statistics,
)


def finish_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    gradient_clip: float,
    *,
    update_step: int = 1,
    diagnostics_interval: int = 0,
) -> tuple[torch.Tensor, float, dict[str, object] | None]:
    """Agree on finite gradients, measure preclip norm, step, and validate parameters."""
    gradient_flags = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradients_finite = bool(torch.stack(gradient_flags).all()) if gradient_flags else False
    if not all_ranks_true(gradients_finite, context):
        raise FloatingPointError("nonfinite gradients detected consistently across torchrun ranks")
    snapshot = (
        capture_optimizer_update(optimizer, update_step, diagnostics_interval)
        if diagnostics_interval
        else None
    )
    preclip_stats = optimizer_gradient_statistics(optimizer) if snapshot is not None else {}
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=False
    )
    if not all_ranks_true(bool(torch.isfinite(norm)), context):
        raise FloatingPointError("nonfinite preclip gradient norm detected across torchrun ranks")
    postclip_stats = optimizer_gradient_statistics(optimizer) if snapshot is not None else {}
    applied_learning_rates = optimizer_learning_rates(optimizer) if snapshot is not None else {}
    gpu_timer = context.device.type == "cuda"
    started = torch.cuda.Event(enable_timing=True) if gpu_timer else None
    finished = torch.cuda.Event(enable_timing=True) if gpu_timer else None
    if started is not None:
        started.record()
    else:
        optimizer_started = time.perf_counter()
    optimizer.step()
    if finished is not None:
        finished.record()
        optimizer_seconds = 0.0
    else:
        optimizer_seconds = time.perf_counter() - optimizer_started
    optimizer_update = None
    if snapshot is not None:
        optimizer_update = finalize_optimizer_update(
            snapshot,
            optimizer,
            applied_learning_rates,
            preclip_stats,
            postclip_stats,
            float(norm.detach().item()),
        )
    scheduler.step()
    parameter_flags = [torch.isfinite(parameter).all() for parameter in model.parameters()]
    parameters_finite = bool(torch.stack(parameter_flags).all()) if parameter_flags else False
    if not all_ranks_true(parameters_finite, context):
        raise FloatingPointError("nonfinite model parameter detected after optimizer update")
    optimizer.zero_grad(set_to_none=True)
    if started is not None and finished is not None:
        optimizer_seconds = started.elapsed_time(finished) / 1000.0
    return norm.detach(), optimizer_seconds, optimizer_update
