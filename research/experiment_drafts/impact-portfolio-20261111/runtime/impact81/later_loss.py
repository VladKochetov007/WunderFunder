"""Loss-only future labels and GRU objectives; no label is passed to a model."""

from __future__ import annotations

import torch
from torch import Tensor

from impact81.later_reporting import RAW_WIDTH


def perturbation_active(start: int, stop: int) -> bool:
    """Select every fourth fixed1000-row chunk, with a complete128+32-row prefix."""
    return stop - start >= 160 and start % 1000 == 0 and (start // 1000 + 1) % 4 == 0


def local_denominators(
    target: Tensor, mask: Tensor, intervals: list[tuple[int, int]], variant: str
) -> Tensor:
    """Sum each declared auxiliary denominator over the whole update window."""
    if variant == "future_heads":
        from impact81.future_objective import future_window_denominators

        result = torch.zeros(3, device=target.device, dtype=torch.float32)
        for start, stop in intervals:
            result += future_window_denominators(target, mask, start, stop).sum(-1)
        return result
    result = torch.zeros(2, device=target.device, dtype=torch.float32)
    for start, stop in intervals:
        if variant == "state_perturbation" and perturbation_active(start, stop):
            result += mask[:, start + 128 : start + 160].sum().float().expand(2)
        elif variant in ("nll_transfer", "clip_event"):
            selected = mask[:, start:stop, None] & torch.isfinite(target[:, start:stop])
            result += selected.sum((0, 1)).float()
    return result


def contribution(
    prediction: Tensor,
    auxiliary: Tensor,
    target: Tensor,
    mask: Tensor,
    start: int,
    stop: int,
    global_mass: Tensor,
    world_size: int,
    variant: str,
) -> tuple[Tensor, Tensor]:
    """Return the differentiable scaled auxiliary loss and detached sufficient sums."""
    raw = torch.zeros(RAW_WIDTH, dtype=torch.float64, device=prediction.device)
    if variant == "future_heads":
        from impact81.future_objective import future_objective_sums, future_scaled_loss

        sums = future_objective_sums(auxiliary.float(), target, mask, start, stop)
        raw[:6] = sums.loss_numerator.detach().flatten().double()
        raw[6:12] = sums.weight_mass.detach().flatten().double()
        raw[12:18] = sums.eligible_count.detach().flatten().double()
        return future_scaled_loss(sums, global_mass, world_size), raw
    if variant in ("gru_plain", "perturbation_control"):
        return prediction.sum() * 0.0, raw
    from impact81.gru_objectives import (
        clip_event_objective_sums,
        gru_scaled_loss,
        nll_objective_sums,
        perturbation_objective_sums,
    )

    labels, selected = target[:, start:stop], mask[:, start:stop]
    if variant == "state_perturbation":
        active = perturbation_active(start, stop)
        expected = 32 if active else 0
        if auxiliary.shape != (prediction.shape[0], expected, 2):
            raise ValueError("perturbation branch does not match the frozen chunk schedule")
        suffix_mask = selected[:, 128:160] if active else selected[:, :0]
        sums = perturbation_objective_sums(auxiliary.float(), suffix_mask)
        if active:
            for index, position in enumerate((0, 7, 31)):
                valid = suffix_mask[:, position, None].expand(-1, 2)
                values = torch.where(valid, auxiliary[:, position].float(), 0.0)
                raw[18 + 2 * index : 20 + 2 * index] = values.detach().double().square().sum(0)
                raw[24 + 2 * index : 26 + 2 * index] = valid.double().sum(0)
    elif variant == "nll_transfer":
        sums = nll_objective_sums(prediction, auxiliary.float(), labels, selected)
        raw[18:20] = sums["log_variance_sum"].detach().double()
        raw[20:22] = sums["lower_bound_count"].detach().double()
        raw[22:24] = sums["upper_bound_count"].detach().double()
    elif variant == "clip_event":
        sums = clip_event_objective_sums(auxiliary.float(), labels, selected)
        raw[24:42] = sums["confusion"].detach().flatten().double()
        raw[42:48] = sums["ce_by_class"].detach().flatten().double()
    else:
        raise ValueError("unsupported later auxiliary objective")
    raw[:2] = sums["loss_numerator"].detach().double()
    raw[6:8] = sums["count"].detach().double()
    raw[12:14] = sums["count"].detach().double()
    return gru_scaled_loss(sums, global_mass, world_size, variant), raw
