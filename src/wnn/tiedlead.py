"""Causal tiedlead feature contract shared by training and recurrent models."""

from __future__ import annotations

import torch
from torch import Tensor

from wnn.data import N_FEATURES
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES

LEAD_WIDTH = 52
LEAD_I1_START = 52
LEAD_I1_STOP = 104
TIEDLEAD_INPUT_WIDTH = N_FEATURES + N_EVENT_FEATURES + LEAD_WIDTH
TIEDLEAD_SCHEMA = "raw112-event99-lead52-v1"
_I0_PRICE_EVENT_INDEX = FEATURE_NAMES.index("ev_i0px")


def tiedlead_i1_at_last_i0_price_event(raw: Tensor, events: Tensor) -> Tensor:
    """Gather the latest i1 block at or before each row's last i0 price event.

    Row zero is treated as an event, matching the historical ``RS_LEAD=1``
    transform. Call this once on each complete sequence, before TBPTT slicing,
    so event history remains continuous across chunk boundaries.
    """
    if raw.ndim != 3 or raw.shape[-1] != N_FEATURES:
        raise ValueError("tiedlead raw input must have shape [B,T,112]")
    if events.shape != (*raw.shape[:2], N_EVENT_FEATURES):
        raise ValueError("tiedlead event features must match [B,T,99]")
    batch, rows, _ = raw.shape
    if rows < 1:
        raise ValueError("tiedlead sequences must contain at least one row")

    event_flags = events[..., _I0_PRICE_EVENT_INDEX] > 0.5
    event_flags = event_flags.clone()
    event_flags[:, 0] = True
    row_indices = torch.arange(rows, device=raw.device).expand(batch, -1)
    last_event = torch.cummax(
        torch.where(event_flags, row_indices, torch.zeros_like(row_indices)), dim=1
    ).values
    return torch.gather(
        raw[..., LEAD_I1_START:LEAD_I1_STOP],
        1,
        last_event[..., None].expand(-1, -1, LEAD_WIDTH),
    )
