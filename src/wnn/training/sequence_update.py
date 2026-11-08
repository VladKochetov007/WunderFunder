"""Strict-FP32 CUDA update boundary with isolated optimizer timing."""

from __future__ import annotations

import torch

from wnn.training.sequence_distributed import DistributedContext, all_ranks_true
from wnn.training.sequence_update_telemetry import (
    OptimizerUpdateSnapshot,
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
    telemetry_snapshot: OptimizerUpdateSnapshot | None = None,
) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, dict[str, object]]:
    """Agree on finite gradients, measure preclip norm, step, and validate parameters."""
    gradient_flags = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradients_finite = bool(torch.stack(gradient_flags).all()) if gradient_flags else False
    if not all_ranks_true(gradients_finite, context):
        raise FloatingPointError("nonfinite gradients detected consistently across torchrun ranks")
    preclip_gradient_stats = (
        optimizer_gradient_statistics(optimizer) if telemetry_snapshot is not None else None
    )
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=False
    )
    if not all_ranks_true(bool(torch.isfinite(norm)), context):
        raise FloatingPointError("nonfinite preclip gradient norm detected across torchrun ranks")
    postclip_gradient_stats = (
        optimizer_gradient_statistics(optimizer) if telemetry_snapshot is not None else None
    )
    applied_learning_rates = (
        {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        if telemetry_snapshot is not None
        else None
    )
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
    diagnostics = (
        finalize_optimizer_update(
            telemetry_snapshot,
            optimizer,
            applied_learning_rates,
            preclip_gradient_stats,
            postclip_gradient_stats,
            float(norm.detach().item()),
        )
        if telemetry_snapshot is not None
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    elapsed = started.elapsed_time(finished) / 1000.0
    if diagnostics is None:
        return norm.detach(), elapsed
    return norm.detach(), elapsed, diagnostics


def finish_sampled_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    gradient_clip: float,
    update_step: int,
    sampling_interval: int,
) -> tuple[torch.Tensor, float, dict[str, object] | None]:
    """Take an opt-in snapshot on sampled steps and preserve the ordinary update path otherwise."""
    snapshot = capture_optimizer_update(optimizer, update_step, sampling_interval)
    if snapshot is None:
        norm, elapsed = finish_update(model, optimizer, scheduler, context, gradient_clip)
        return norm, elapsed, None
    norm, elapsed, diagnostics = finish_update(
        model,
        optimizer,
        scheduler,
        context,
        gradient_clip,
        telemetry_snapshot=snapshot,
    )
    return norm, elapsed, diagnostics
