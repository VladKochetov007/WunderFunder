"""Causal i1 lead block: i1 raw values held at the most recent i0 book event.

Reconstructs the 52-input "i1 lead block" of the team's served tied-lead members, identified
from their normalization fingerprint (research/campaign_2026-10-25/class3_reframe_r1/
FTT_LEAD_BLOCK_FINGERPRINT_R1.md). An i0 book event is a row whose raw columns 0:22 differ
from the previous row. Each output row carries raw columns 52:104 from the latest such
event at or before it; before the first event it carries the sequence's first row. Together
with the current i1 columns, the network can read how far i1 moved while i0's book was still.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

MODE = "i1_held_i0book"
I0_BOOK = slice(0, 22)
I1_BLOCK = slice(52, 104)
WIDTH = I1_BLOCK.stop - I1_BLOCK.start
FEATURE_NAMES = tuple(f"i1_held_i0book_{index:02d}" for index in range(WIDTH))


def features_numpy(raw: np.ndarray) -> np.ndarray:
    """Return (B,T,52) held i1 values for finite float32 raw input of shape (B,T,112)."""
    batch_size, rows, _ = raw.shape
    positions = np.arange(rows, dtype=np.int64)[None, :]
    event = np.zeros((batch_size, rows), dtype=np.bool_)
    if rows > 1:
        event[:, 1:] = (raw[:, 1:, I0_BOOK] != raw[:, :-1, I0_BOOK]).any(axis=-1)
    source = np.where(event, positions, 0)
    np.maximum.accumulate(source, axis=1, out=source)
    return np.take_along_axis(raw[..., I1_BLOCK], source[..., None], axis=1).astype(np.float32)


def features_torch(raw: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Match ``features_numpy`` on a full (B,T,112) float32 tensor."""
    import torch

    batch_size, rows, _ = raw.shape
    positions = torch.arange(rows, device=raw.device).view(1, rows).expand(batch_size, rows)
    event = torch.zeros((batch_size, rows), dtype=torch.bool, device=raw.device)
    if rows > 1:
        event[:, 1:] = (raw[:, 1:, I0_BOOK] != raw[:, :-1, I0_BOOK]).any(dim=-1)
    source = torch.cummax(torch.where(event, positions, torch.zeros_like(positions)), dim=1).values
    held = torch.gather(raw[..., I1_BLOCK], 1, source[..., None].expand(-1, -1, WIDTH))
    if output is None:
        return held
    if output.shape != held.shape or output.dtype != raw.dtype or output.device != raw.device:
        raise ValueError("i1 lead output buffer must match the input batch, dtype and device")
    output.copy_(held)
    return output


class StreamingI1LeadBlock:
    """O(1) per-row twin of ``features_numpy`` with an explicit sequence reset."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the previous row and held snapshot at a sequence boundary."""
        self.previous_i0: np.ndarray | None = None
        self.held = np.zeros(WIDTH, dtype=np.float32)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Consume one finite raw row of width 112 and return the held i1 snapshot."""
        if self.previous_i0 is None or np.any(row[I0_BOOK] != self.previous_i0):
            self.held = row[I1_BLOCK].astype(np.float32, copy=True)
        self.previous_i0 = row[I0_BOOK].copy()
        return self.held.copy()
