from __future__ import annotations

import numpy as np
import torch


def origin_windows(
    sequence_rows: int,
    update_rows: int,
    micro_rows: int,
    mode: str,
    seed: int,
    epoch: int,
    batch: int,
) -> list[slice]:
    """Partition all rows while preserving update count and summed causal-prefix work.

    Random boundaries move in opposite-sign pairs. Every accepted partition is
    strictly increasing and microchunk aligned, and the sum of window origins
    equals the fixed partition exactly. Prefix replay therefore visits the same
    number of rows in both arms, including zero-loss prefixes.
    """
    if mode not in ("off", "prefix-fixed", "prefix-random"):
        raise ValueError("unknown origin view")
    if sequence_rows <= 0 or update_rows <= 0 or micro_rows <= 0:
        raise ValueError("row intervals must be positive")
    if mode == "off":
        return [
            slice(start, min(start + update_rows, sequence_rows))
            for start in range(0, sequence_rows, update_rows)
        ]
    if sequence_rows % update_rows or update_rows % micro_rows:
        raise ValueError("prefix views require update and micro intervals dividing sequence rows")
    boundaries = np.arange(0, sequence_rows + 1, update_rows)
    if mode == "prefix-random":
        if len(boundaries) < 4 or update_rows <= micro_rows or seed < 0:
            raise ValueError("random origins require >=3 updates and >=2 microchunks per update")
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, batch]))
        radius = update_rows // micro_rows - 1
        for left in range(1, (len(boundaries) - 1) // 2 + 1):
            right = len(boundaries) - 1 - left
            if left >= right:
                break
            delta = int(rng.integers(-radius, radius + 1)) * micro_rows
            proposed = boundaries.copy()
            proposed[left] += delta
            proposed[right] -= delta
            if (np.diff(proposed) > 0).all():
                boundaries = proposed
    return [
        slice(int(first), int(last))
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


@torch.no_grad()
def update_ema_for_rows(
    averaged: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
    rows: int,
    reference_rows: int,
) -> None:
    """Temporarily apply a variable-width row clock; keep checkpoint configuration fixed."""
    if not reference_rows:
        averaged.update_parameters(model)
        return
    effective = decay ** (rows / reference_rows)
    if hasattr(averaged, "decay"):
        original = averaged.decay
        try:
            averaged.decay = effective
            averaged.update_parameters(model)
        finally:
            averaged.decay = original
    else:
        original = averaged.multi_avg_fn
        try:
            averaged.multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(effective)
            averaged.update_parameters(model)
        finally:
            averaged.multi_avg_fn = original
