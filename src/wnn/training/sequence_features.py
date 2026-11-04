"""Causal input-feature assembly for the canonical full-sequence runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES, event_features, event_features_torch

InputFeatureMode = Literal["raw112", "event211"]
RAW_INPUT_MODE: InputFeatureMode = "raw112"
EVENT_INPUT_MODE: InputFeatureMode = "event211"
RAW_INPUT_SCHEMA = "raw112-v1"
EVENT_INPUT_SCHEMA = "raw112-event211-v1"


def sequence_input_width(mode: str = RAW_INPUT_MODE) -> int:
    """Return the configured row width for a supported sequence input mode."""
    if mode == RAW_INPUT_MODE:
        return N_FEATURES
    if mode == EVENT_INPUT_MODE:
        return N_FEATURES + N_EVENT_FEATURES
    raise ValueError(f"unsupported sequence input feature mode: {mode}")


def sequence_input_schema(mode: str = RAW_INPUT_MODE) -> str:
    """Return a stable provenance name for a causal sequence input mode."""
    sequence_input_width(mode)
    return RAW_INPUT_SCHEMA if mode == RAW_INPUT_MODE else EVENT_INPUT_SCHEMA


def sequence_inputs_from_raw(
    raw: np.ndarray | Tensor, mode: str = RAW_INPUT_MODE
) -> np.ndarray | Tensor:
    """Append deterministic full-sequence causal event features when explicitly requested.

    Event features are calculated once from each complete 112-column sequence,
    before the trainer slices it into TBPTT chunks. Calling this function once per
    sequence also resets its event clock and held-event state at the sequence boundary.
    """
    width = sequence_input_width(mode)
    if isinstance(raw, Tensor):
        if raw.ndim not in (2, 3) or raw.shape[-1] != N_FEATURES:
            raise ValueError("raw sequence tensor must have shape [T,112] or [B,T,112]")
        if mode == RAW_INPUT_MODE:
            return raw
        single_sequence = raw.ndim == 2
        batched = raw.unsqueeze(0) if single_sequence else raw
        events = event_features_torch(batched)
        result = torch.cat((batched, events), dim=-1)
        return result.squeeze(0) if single_sequence else result
    if isinstance(raw, np.ndarray):
        if raw.ndim not in (2, 3) or raw.shape[-1] != N_FEATURES:
            raise ValueError("raw sequence array must have shape [T,112] or [B,T,112]")
        if mode == RAW_INPUT_MODE:
            return raw
        events = event_features(raw)
        result = np.concatenate((raw.astype(np.float32, copy=False), events), axis=-1)
        if result.shape[-1] != width:
            raise RuntimeError("causal event feature assembly returned an unexpected width")
        return result
    raise TypeError("raw sequence input must be a NumPy array or torch tensor")


def numpy_sequence_reader(
    reader: Callable[[str, int], np.ndarray], mode: str
) -> Callable[[str, int], np.ndarray]:
    """Wrap a raw fitting-member reader with the declared deterministic input transform."""
    sequence_input_width(mode)

    def read(source: str, group: int) -> np.ndarray:
        """Read one complete sequence and assemble its causal configured features."""
        raw = reader(source, group)
        values = sequence_inputs_from_raw(raw, mode)
        if not isinstance(values, np.ndarray):
            raise TypeError("normalization feature assembly must return a NumPy array")
        return values

    return read
