"""Causal time-warp augmentation for training views (research flag --aug-warp).

A warped view repeats or drops only fully static rows: rows whose 104 price/volume channels equal the previous
row. Event order and every non-static row are untouched, so each kept row's label remains the label of the row it
copies. Warping happens inside one sequence, so it never crosses a component boundary, and the first WARMUP rows
are never changed.
"""

from __future__ import annotations

import torch

from wnn.data import WARMUP

PRICE_VOLUME_CHANNELS = 104


def parse_warp(spec: str) -> tuple[float, float, float]:
    """Parse ``P,LO,HI`` with 0 < P <= 1 and 0 < LO <= HI."""
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("--aug-warp expects P,LO,HI")
    prob, low, high = parts
    if not 0 < prob <= 1 or not 0 < low <= high:
        raise ValueError("--aug-warp needs 0 < P <= 1 and 0 < LO <= HI")
    return prob, low, high


def static_rows(features: torch.Tensor) -> torch.Tensor:
    """Rows after the warm-up whose price/volume channels did not change from the previous row."""
    n = features.shape[0]
    quiet = torch.zeros(n, dtype=torch.bool)
    if n > WARMUP:
        quiet[WARMUP:] = (features[WARMUP:, :PRICE_VOLUME_CHANNELS]
                          == features[WARMUP - 1:-1, :PRICE_VOLUME_CHANNELS]).all(dim=1)
    return quiet


def warp_index(features: torch.Tensor, factor: float) -> tuple[torch.Tensor, int]:
    """Index map and kept length for a warp by ``factor`` (expected copies of each static row).

    Static rows get floor(factor) copies plus one more with probability frac(factor); all other rows keep one.
    The map is truncated to the original length, or padded with the last row, and the kept length is returned
    so the padding can be excluded from every loss.
    """
    n = features.shape[0]
    quiet = static_rows(features)
    base = int(factor)
    counts = torch.ones(n, dtype=torch.long)
    extra = (torch.rand(int(quiet.sum())) < (factor - base)).long()
    counts[quiet] = base + extra
    index = torch.repeat_interleave(torch.arange(n), counts)
    kept = min(int(index.numel()), n)
    index = index[:n]
    if index.numel() < n:
        index = torch.cat([index, index[-1:].expand(n - index.numel())])
    return index, kept


def warp_rows(result: tuple, warp: tuple[float, float, float]) -> tuple:
    """Warp one sequence tuple (features, targets, scored, ...) and append the kept length."""
    prob, low, high = warp
    n = result[0].shape[0]
    if torch.rand(()) >= prob:
        return result + (torch.tensor(n),)
    factor = low + (high - low) * float(torch.rand(()))
    index, kept = warp_index(result[0], factor)
    warped = []
    for j, part in enumerate(result):
        part = part[index]
        if j == 2:  # is_scored: padded rows are never scored
            part = part.clone()
            part[kept:] = False
        warped.append(part)
    return tuple(warped) + (torch.tensor(kept),)
