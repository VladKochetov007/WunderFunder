"""Loss-specific microchunk work; reuse canonical precision, DDP and primary objective."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel

from impact81.auxiliary_objectives import (
    deep_supervision_scaled_loss,
    deep_supervision_wmse_numerators,
    pairwise_objective_sums,
    pairwise_scaled_loss,
    pairwise_window_denominators,
)
from wnn.training.sequence_distributed import DistributedContext, all_ranks_true, global_sum
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_objectives import detached_sums, objective_numerators, scaled_objective
from wnn.training.sequence_precision import backward_context, forward_context

# Explicit model state and validated canonical config have heterogeneous runtime types.


def auxiliary_denominators(
    target: Tensor,
    mask: Tensor,
    intervals: list[tuple[int, int]],
    variant: str,
    context: DistributedContext,
) -> Tensor:
    """Reduce pair masses across the complete accumulation window before any backward."""
    local = torch.zeros(2, dtype=torch.float32, device=context.device)
    if variant == "pairwise_ranking":
        for start, stop in intervals:
            local += pairwise_window_denominators(target[:, start:stop], mask[:, start:stop])
    result = global_sum(local, context)
    if not all_ranks_true(bool(torch.isfinite(result).all()), context):
        raise FloatingPointError("nonfinite global auxiliary denominator")
    return result


def backward_window(
    model: nn.Module,
    batch: dict[str, Tensor],
    intervals: list[tuple[int, int]],
    denominators: Tensor,
    auxiliary_mass: Tensor,
    state: Any,
    variant: str,
    real_sequences: int,
    config: dict[str, Any],
    context: DistributedContext,
) -> tuple[Any, Tensor, Tensor]:
    """Backpropagate primary plus declared auxiliary losses with unchanged TBPTT cadence."""
    local = torch.zeros(6, dtype=torch.float64, device=context.device)
    raw_auxiliary = torch.zeros(13, dtype=torch.float64, device=context.device)
    mode = str(config["precision_mode"])
    for index, (start, stop) in enumerate(intervals):
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index < len(intervals) - 1
            else nullcontext()
        )
        attention = (
            sdpa_kernel(SDPBackend.MATH)
            if config["model"]["attention_backend"] == "math"
            else nullcontext()
        )
        target, mask = batch["y"][:, start:stop], batch["mask"][:, start:stop]
        hints = batch["hints"][:, start:stop]
        with sync, attention:
            with forward_context(mode, context.device.type):
                prediction, auxiliary, next_state = model(batch["x"][:, start:stop], state)
            state = detach_state(next_state)
            prediction = prediction.float()
            numerators = objective_numerators(
                prediction,
                hints,
                target,
                batch["teacher"][:, start:stop],
                hints,
                mask,
                objective_mode=config["objective_mode"],
                context=context,
            )
            loss = scaled_objective(
                numerators, denominators, context.world_size, 0.0, 0.0, microchunks=len(intervals)
            )
            if variant == "deep_supervision":
                deep = deep_supervision_wmse_numerators(auxiliary.float().unbind(-2), target, mask)
                loss = loss + deep_supervision_scaled_loss(
                    deep, denominators[0], context.world_size
                )
                raw_auxiliary[:3] += deep.detach().double()
            elif variant == "pairwise_ranking":
                pair = pairwise_objective_sums(prediction, target, mask)
                loss = loss + pairwise_scaled_loss(pair, auxiliary_mass, context.world_size)
                raw_auxiliary[3:5] += pair.loss_numerator.detach().double()
                raw_auxiliary[5:7] += pair.weight_denominator.detach().double()
                raw_auxiliary[7:9] += pair.eligible_count.detach().double()
                raw_auxiliary[9:11] += pair.concordance_mass.detach().double()
                raw_auxiliary[11:13] += pair.tie_count.detach().double()
            if not all_ranks_true(bool(torch.isfinite(loss.detach()).all()), context):
                raise FloatingPointError("nonfinite primary/auxiliary loss on a rank")
            with backward_context(mode, context.device.type):
                loss.backward()
        local[:5] += detached_sums(numerators, target, mask)
        local[5] += real_sequences * (stop - start)
    return state, local, raw_auxiliary
