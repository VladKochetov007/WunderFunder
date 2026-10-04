from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

HiddenState = Tensor | tuple[Tensor, ...] | None
Forward = Callable[[slice, HiddenState], tuple[Tensor, HiddenState]]


class DistilledLoss(NamedTuple):
    """Coefficients and output columns of the supported distilled GRU objective."""

    alpha: float
    bootstrap_horizon: int
    ema: float
    ema_metric_weight: bool
    structural: float
    structural_start: int
    spread_bins: int
    next_event: float
    next_start: int


def ema_decay_per_update(decay: float, update_rows: int, reference_rows: int) -> float:
    """Keep legacy decay unless an explicit exposure reference is selected."""
    if not 0 <= decay <= 1 or update_rows <= 0 or reference_rows < 0:
        raise ValueError("invalid EMA decay or row clock")
    return decay if reference_rows == 0 else decay ** (update_rows / reference_rows)


def detach_hidden(state: HiddenState) -> HiddenState:
    """Cut gradient credit without resetting the carried recurrent values."""
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(value.detach() for value in state)
    return state.detach()


def blended_target(
    labels: Tensor,
    teacher: Tensor,
    next_change: Tensor | None,
    rows: slice,
    alpha: float,
    horizon: int,
) -> Tensor:
    """Reproduce strict stopped bootstrap; target weights still use real labels."""
    label = labels[:, rows].float().clamp(-2, 2)
    if horizon:
        if next_change is None:
            raise ValueError("stopped bootstrap requires next-label-change indices")
        ahead = torch.arange(rows.start, rows.stop, device=label.device)[None] + horizon
        alive = (next_change[:, rows] > ahead) & (ahead < teacher.shape[1])
        indices = ahead.clamp(max=teacher.shape[1] - 1).expand(label.shape[0], -1)
        later = teacher.gather(1, indices[..., None].expand(-1, -1, 2)).float().clamp(-2, 2)
        label = torch.where(alive[..., None], later, label)
    return (1 - alpha) * label + alpha * teacher[:, rows].float().clamp(-2, 2)


def shuffled_masses(weights: Tensor, update_rows: int, seed: int) -> Tensor:
    """Permute whole-update mass factors within the same batch using a private RNG."""
    masses = torch.stack(
        [
            weights[:, start : start + update_rows].sum().detach()
            for start in range(0, weights.shape[1], update_rows)
        ]
    )
    order = np.random.default_rng(seed).permutation(len(masses))
    return masses[torch.as_tensor(order, device=masses.device)]


def detach_boundary_offset(micro_rows: int, detach_rows: int, seed: int, step: int) -> int:
    """Draw a reproducible phase without changing rows, update windows or process RNG."""
    if micro_rows <= 0 or detach_rows <= micro_rows or detach_rows % micro_rows:
        raise ValueError("jitter requires at least two microchunks per detach interval")
    generator = np.random.default_rng(np.random.SeedSequence([seed, step]))
    return int(generator.integers(detach_rows // micro_rows)) * micro_rows


def distilled_macro_losses(
    *,
    forward: Forward,
    ema_forward: Forward | None,
    state: HiddenState,
    ema_state: HiddenState,
    labels: Tensor,
    teacher: Tensor,
    row_weights: Tensor,
    next_change: Tensor | None,
    structural_labels: tuple[Tensor, Tensor, Tensor] | None,
    next_labels: tuple[Tensor, Tensor, Tensor] | None,
    rows: slice,
    micro_rows: int,
    detach_rows: int,
    config: DistilledLoss,
    denominator: str = "chunk",
    reference_mass: float = 0.0,
    donor_mass: Tensor | None = None,
    detach_offset_rows: int = 0,
) -> Iterator[tuple[Tensor, HiddenState, HiddenState]]:
    """Yield losses ready for backward, pooling each divisor over one update.

    Consume and backpropagate each yield before requesting the next. Parameters
    remain frozen until the iterator is exhausted; one optimizer/clip/scheduler/
    EMA advance follows the macrostep. Short detach spans release their graphs.
    EMA uses its actual incoming state and weights throughout the same macrostep.
    ``chunk`` means this update's full row mass, preserving the legacy objective
    when micro, detach and update intervals coincide. ``fixed`` substitutes a
    frozen TRAIN-estimated mass for the MAIN divisor only. ``shuffled`` applies
    another update's mass/reference factor to the original normalized main loss.
    """
    width = rows.stop - rows.start
    if (
        width <= 0
        or micro_rows <= 0
        or detach_rows <= 0
        or detach_rows % micro_rows
        or width % detach_rows
    ):
        raise ValueError("micro/detach/update intervals must divide one another")
    if denominator not in {"chunk", "fixed", "shuffled"}:
        raise ValueError("unsupported distilled denominator")
    if not 0 <= detach_offset_rows < detach_rows or detach_offset_rows % micro_rows:
        raise ValueError(
            "detach phase must be a microchunk-aligned offset within the detach interval"
        )
    if denominator != "chunk" and (reference_mass <= 0 or not np.isfinite(reference_mass)):
        raise ValueError("fixed/shuffled normalization requires a finite positive reference mass")
    if denominator == "shuffled" and donor_mass is None:
        raise ValueError("shuffled normalization requires a bound donor mass")
    clipped = labels[:, rows].float().clamp(-2, 2)
    main_weights = clipped.abs() * row_weights[:, rows, None]
    main_mass = main_weights.sum().clamp_min(1e-8)
    main_divisor = reference_mass if denominator == "fixed" else main_mass
    main_scale = donor_mass / reference_mass if denominator == "shuffled" else 1.0
    aux_weights = row_weights[:, rows].float()
    structural_mass = aux_weights.sum().clamp_min(1e-8)
    consistency_weights = aux_weights[..., None]
    if config.ema_metric_weight:
        consistency_weights = consistency_weights * clipped.abs()
    consistency_mass = consistency_weights.sum().clamp_min(1e-8)
    next_mass = None if next_labels is None else next_labels[2][:, rows].sum().clamp_min(1e-8)
    target = blended_target(
        labels, teacher, next_change, rows, config.alpha, config.bootstrap_horizon
    )
    partial = None
    for start in range(rows.start, rows.stop, micro_rows):
        window = slice(start, start + micro_rows)
        local = slice(start - rows.start, start - rows.start + micro_rows)
        output, state = forward(window, state)
        prediction = output.float()
        loss = (
            main_scale
            * (main_weights[:, local] * (prediction[..., :2] - target[:, local]).square()).sum()
            / main_divisor
        )
        if config.ema:
            if ema_forward is None:
                raise ValueError("EMA consistency requires an EMA forward function")
            with torch.no_grad():
                averaged, ema_state = ema_forward(window, ema_state)
            ema_state = detach_hidden(ema_state)
            scale = 2.0 if config.ema_metric_weight else 1.0
            loss = (
                loss
                + config.ema
                * scale
                * (
                    consistency_weights[:, local]
                    * (prediction[..., :2] - averaged[..., :2].float()).square()
                ).sum()
                / consistency_mass
            )
        if config.structural:
            if structural_labels is None:
                raise ValueError("structural loss requires its three label arrays")
            first = config.structural_start
            sizes = (config.spread_bins, 3, 4)
            ce = 0.0
            for size, label in zip(sizes, structural_labels, strict=True):
                ce = ce + F.cross_entropy(
                    prediction[..., first : first + size].reshape(-1, size),
                    label[:, window].reshape(-1),
                    reduction="none",
                )
                first += size
            loss = (
                loss
                + config.structural
                * (ce * aux_weights[:, local].reshape(-1)).sum()
                / structural_mass
            )
        if config.next_event:
            if next_labels is None:
                raise ValueError("next-event loss requires row/sign/weight arrays")
            next_rows, next_up, next_weights = next_labels
            offset = config.next_start
            mse = (prediction[..., offset] - (next_rows[:, window] - 3.0) / 2.0).square()
            bce = F.binary_cross_entropy_with_logits(
                prediction[..., offset + 1], next_up[:, window], reduction="none"
            )
            loss = (
                loss + config.next_event * ((mse + bce) * next_weights[:, window]).sum() / next_mass
            )
        partial = loss if partial is None else partial + loss
        if (
            start + micro_rows - rows.start - detach_offset_rows
        ) % detach_rows == 0 or start + micro_rows == rows.stop:
            state = detach_hidden(state)
            yield partial, state, ema_state
            partial = None
