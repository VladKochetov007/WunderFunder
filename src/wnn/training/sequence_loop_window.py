from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    global_sum,
)
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_monitor_diagnostics import DiagnosticProbe
from wnn.training.sequence_objectives import (
    detached_sums,
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_precision import backward_context, forward_context


def backward_window(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    teacher: torch.Tensor,
    hint_target: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
    intervals: list[tuple[int, int]],
    real_sequence_count: int,
    denominators: torch.Tensor,
    state: Any,
    config: dict[str, object],
    context: DistributedContext,
    diagnostics: DiagnosticProbe | None = None,
) -> tuple[Any, torch.Tensor]:
    """Forward and backpropagate one pre-normalized TBPTT accumulation window."""
    local_stats = torch.zeros(6, dtype=torch.float64, device=context.device)
    for index, (start, stop) in enumerate(intervals):
        final_microchunk = index == len(intervals) - 1
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not final_microchunk
            else nullcontext()
        )
        attention_backend = str(config["model"].get("attention_backend", "math"))
        attention_context = (
            sdpa_kernel(SDPBackend.MATH) if attention_backend == "math" else nullcontext()
        )
        mode = str(config.get("precision_mode", "fp32"))
        with sync_context, attention_context:
            with forward_context(mode, context.device.type):
                prediction, hint_prediction, next_state = model(x[:, start:stop], state)
            if diagnostics is not None:
                diagnostics.observe_prediction(prediction, mask[:, start:stop])
            state = detach_state(next_state)
            prediction = prediction.float()
            hint_prediction = hint_prediction.float()
            numerators = objective_numerators(
                prediction,
                hint_prediction,
                target[:, start:stop],
                teacher[:, start:stop],
                hint_target[:, start:stop],
                mask[:, start:stop],
            )
            loss = scaled_objective(
                numerators,
                denominators,
                context.world_size,
                float(config["kd_weight"]),
                float(config["hint_weight"]),
            )
            check_finite_loss(loss, context)
            with backward_context(mode, context.device.type):
                loss.backward()
        local_stats[:5] += detached_sums(numerators, target[:, start:stop], mask[:, start:stop])
        local_stats[5] += real_sequence_count * (stop - start)
    return state, local_stats


def window_denominators_global(
    target: torch.Tensor,
    mask: torch.Tensor,
    hints: torch.Tensor,
    intervals: list[tuple[int, int]],
    context: DistributedContext,
) -> torch.Tensor:
    """Reduce both objective denominators across ranks before any window backward."""
    local = torch.zeros(2, dtype=torch.float32, device=context.device)
    for start, stop in intervals:
        local += window_denominators(
            target[:, start:stop], mask[:, start:stop], hints[:, start:stop]
        )
    global_denominators = global_sum(local, context)
    if not all_ranks_true(bool(torch.isfinite(global_denominators).all()), context):
        raise FloatingPointError("nonfinite objective denominator detected across torchrun ranks")
    return global_denominators


def check_finite_loss(loss: torch.Tensor, context: DistributedContext) -> None:
    """Check loss finiteness consistently across all distributed ranks."""
    local_finite = bool(torch.isfinite(loss.detach()).all())
    if not all_ranks_true(local_finite, context):
        raise FloatingPointError("nonfinite objective detected consistently across torchrun ranks")


def window_record(
    stats: torch.Tensor,
    denominators: torch.Tensor,
    rank: int,
    epoch: int,
    step: int,
    learning_rate: float,
    gradient_norm: torch.Tensor,
    elapsed: float,
    peak_memory: int,
    gradient_clip: float,
    kd_weight: float,
    hint_weight: float,
) -> dict[str, object]:
    """Convert global sufficient sums and this rank's diagnostics into a JSON row."""
    label_denom = float(denominators[0].item())
    hint_denom = float(denominators[1].item())
    label_loss = float(stats[0].item()) / label_denom if label_denom > 0 else 0.0
    kd_loss = float(stats[1].item()) / label_denom if label_denom > 0 else 0.0
    hint_loss = float(stats[2].item()) / hint_denom if hint_denom > 0 else 0.0
    gradient_value = float(gradient_norm.item())
    return {
        "rank": rank,
        "epoch": epoch,
        "step": step,
        "label_loss": label_loss,
        "kd_loss": kd_loss,
        "hint_loss": hint_loss,
        "total_loss": label_loss + kd_weight * kd_loss + hint_weight * hint_loss,
        "valid_rows": int(stats[3].item()),
        "forwarded_rows": int(stats[4].item()),
        "real_rows": int(stats[5].item()),
        "label_weight": label_denom,
        "hint_elements": hint_denom,
        "learning_rate": learning_rate,
        "gradient_norm_preclip": gradient_value,
        "gradient_clipped": int(gradient_value > gradient_clip),
        "gradient_clip_threshold": gradient_clip,
        "window_seconds": elapsed,
        "valid_rows_per_second": float(stats[3].item()) / max(elapsed, 1e-12),
        "forwarded_rows_per_second": float(stats[4].item()) / max(elapsed, 1e-12),
        "peak_memory_bytes": peak_memory,
    }
