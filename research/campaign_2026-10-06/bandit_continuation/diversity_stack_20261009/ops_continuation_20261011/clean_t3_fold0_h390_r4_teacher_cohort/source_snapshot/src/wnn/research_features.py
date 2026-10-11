from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

ENDPOINT = np.float32(-5.199337482452393)
RESEARCH_FEATURE_MODES = ("none", "volume28", "volume28_current", "flow_v2")
# flow_v2 (research arm from the 2026-10-10 critic review): second-instrument volume deltas, trade-stream deltas, per-block event ages,
# per-block update intensity over 16/64/256 rows and signed windowed price changes. Torch/training-time only until a native port exists.
_FLOW_BLOCKS = {"i0px": (0, 22), "i0vol": (22, 44), "i0trade": (44, 52), "i1px": (52, 74), "i1vol": (74, 96), "i1trade": (96, 104), "extra": (104, 112)}
_FLOW_PRICE_COLS = (0, 11, 52, 63)
_FLOW_WINDOWS = (16, 64, 256)
FLOW_V2_NAMES = (tuple(f"i1dvol_{j}" for j in range(22)) + tuple(f"tradeD_{j}" for j in range(16)) + tuple(f"age_{b}" for b in _FLOW_BLOCKS)
                 + tuple(f"rate{w}_{b}" for w in _FLOW_WINDOWS for b in _FLOW_BLOCKS)
                 + tuple(f"pxchg{w}_{c}" for w in (16, 64) for c in _FLOW_PRICE_COLS))
VOLUME28_NAMES = tuple(f"dVrun_{index}" for index in range(22)) + (
    "pathV_first11",
    "pathV_last11",
    "nchgV_first11",
    "nchgV_last11",
    "netV_first11",
    "netV_last11",
)


def research_feature_names(mode: str) -> tuple[str, ...]:
    """Return a fixed feature order; anonymous volume halves imply no physical axes."""
    if mode not in RESEARCH_FEATURE_MODES:
        raise ValueError(f"Unknown research feature mode: {mode}")
    if mode == "none":
        return ()
    if mode == "volume28":
        return VOLUME28_NAMES
    if mode == "flow_v2":
        return FLOW_V2_NAMES
    return tuple(f"currentV_{index}" for index in range(22)) + (
        "current_step_path_first11",
        "current_step_path_last11",
        "current_step_change_first11",
        "current_step_change_last11",
        "current_norm_first11",
        "current_norm_last11",
    )


def _raw_array(raw: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(raw, dtype=np.float32)
    squeeze = array.ndim == 2
    if squeeze:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] != 112 or array.shape[1] < 1:
        raise ValueError("Expected nonempty (T,112) or (B,T,112) raw features")
    if not np.isfinite(array).all():
        raise ValueError("Raw features must be finite")
    return array, squeeze


def research_features_numpy(raw: np.ndarray, mode: str = "none") -> np.ndarray:
    """Compute causal full-sequence inputs; run-start movement is excluded from sums."""
    names = research_feature_names(mode)
    x, squeeze = _raw_array(raw)
    if not names:
        output = np.empty((*x.shape[:2], 0), dtype=np.float32)
        return output[0] if squeeze else output
    volume = x[..., 22:44]
    delta = np.diff(volume, axis=1, prepend=volume[:, :1])
    path = np.stack((np.abs(delta[..., :11]).sum(-1), np.abs(delta[..., 11:]).sum(-1)), -1)
    changes = np.stack(((delta[..., :11] != 0).any(-1), (delta[..., 11:] != 0).any(-1)), -1)
    if mode == "volume28":
        index = np.broadcast_to(np.arange(x.shape[1]), x.shape[:2])
        start = np.maximum.accumulate(np.where(x[..., 110] == ENDPOINT, index, 0), axis=1)
        volume = volume - np.take_along_axis(volume, start[..., None], axis=1)
        total = np.cumsum(np.concatenate((path, changes), -1), axis=1, dtype=np.float64)
        sums = total - np.take_along_axis(total, start[..., None], axis=1)
    else:
        sums = np.concatenate((path, changes), -1)
    net = np.stack((np.abs(volume[..., :11]).sum(-1), np.abs(volume[..., 11:]).sum(-1)), -1)
    output = np.concatenate((volume, sums, net), axis=-1).astype(np.float32)
    return output[0] if squeeze else output


def research_features_torch(raw: torch.Tensor, mode: str = "none") -> torch.Tensor:
    """Match NumPy on full float32 raw sequences before autocast or TBPTT slicing."""
    import torch

    names = research_feature_names(mode)
    if raw.ndim != 3 or raw.shape[-1] != 112 or raw.shape[1] < 1:
        raise ValueError("Expected nonempty (B,T,112) raw features")
    if raw.dtype != torch.float32:
        raise ValueError("Extract endpoint features from original float32 inputs before autocast")
    if not names:
        return raw.new_empty((*raw.shape[:2], 0))
    if mode == "flow_v2":
        return _flow_v2_torch(raw)
    volume = raw[..., 22:44]
    delta = torch.diff(volume, dim=1, prepend=volume[:, :1])
    path = torch.stack((delta[..., :11].abs().sum(-1), delta[..., 11:].abs().sum(-1)), -1)
    changes = torch.stack(((delta[..., :11] != 0).any(-1), (delta[..., 11:] != 0).any(-1)), -1)
    if mode == "volume28":
        index = torch.arange(raw.shape[1], device=raw.device).expand(raw.shape[:2])
        start = torch.cummax(torch.where(raw[..., 110] == float(ENDPOINT), index, 0), dim=1).values
        volume = volume - torch.gather(volume, 1, start[..., None].expand(*start.shape, 22))
        total = torch.cumsum(
            torch.cat((path, changes.to(raw.dtype)), -1), dim=1, dtype=torch.float64
        )
        sums = total - torch.gather(total, 1, start[..., None].expand(*start.shape, 4))
    else:
        sums = torch.cat((path, changes.to(raw.dtype)), -1)
    net = torch.stack((volume[..., :11].abs().sum(-1), volume[..., 11:].abs().sum(-1)), -1)
    return torch.cat((volume, sums.to(raw.dtype), net), -1)


def _flow_v2_torch(raw: torch.Tensor) -> torch.Tensor:
    """Causal full-sequence flow features (every value uses rows <= t only)."""
    import torch

    b, t, _ = raw.shape
    delta = torch.diff(raw, dim=1, prepend=raw[:, :1])
    parts = [delta[..., 74:96], delta[..., 44:52], delta[..., 96:104]]
    changed = [(delta[..., lo:hi] != 0).any(-1) for lo, hi in _FLOW_BLOCKS.values()]
    index = torch.arange(t, device=raw.device).expand(b, t)
    ages = []
    for event in changed:
        last = torch.cummax(torch.where(event, index, torch.zeros_like(index)), dim=1).values
        ages.append(torch.log1p((index - last).to(raw.dtype).clamp(max=10_000.0)))
    parts.append(torch.stack(ages, -1))
    counts = torch.cumsum(torch.stack(changed, -1).to(torch.float64), dim=1)
    for w in _FLOW_WINDOWS:
        lagged = torch.cat([torch.zeros_like(counts[:, :w]), counts[:, :-w]], dim=1) if t > w else torch.zeros_like(counts)
        parts.append(((counts - lagged) / w).to(raw.dtype))
    price = delta[..., list(_FLOW_PRICE_COLS)].to(torch.float64)
    total = torch.cumsum(price, dim=1)
    for w in (16, 64):
        lagged = torch.cat([torch.zeros_like(total[:, :w]), total[:, :-w]], dim=1) if t > w else torch.zeros_like(total)
        parts.append((total - lagged).to(raw.dtype))
    return torch.cat(parts, dim=-1)


class StreamingResearchFeatures:
    """Keep O(1) per-sequence state matching the volume28 training transform."""

    def __init__(self, mode: str = "none") -> None:
        self.mode = mode
        self.names = research_feature_names(mode)
        self.reset()

    def reset(self) -> None:
        """Clear every volume accumulator at a sequence boundary."""
        self.previous: np.ndarray | None = None
        self.origin: np.ndarray | None = None
        self.sums = np.zeros(4, dtype=np.float64)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Read current/past raw input only and return owned float32 output."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (112,) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite float32 raw row of width112")
        if not self.names:
            return np.empty(0, dtype=np.float32)
        volume = raw[22:44]
        delta = np.zeros(22, dtype=np.float32) if self.previous is None else volume - self.previous
        path = np.array([np.abs(delta[:11]).sum(), np.abs(delta[11:]).sum()], dtype=np.float32)
        changes = np.array([(delta[:11] != 0).any(), (delta[11:] != 0).any()], dtype=np.float32)
        if self.origin is None or raw[110] == ENDPOINT:
            self.origin = volume.copy()
            self.sums.fill(0)
        else:
            self.sums += np.concatenate((path, changes))
        current = volume - self.origin if self.mode == "volume28" else volume
        sums = self.sums if self.mode == "volume28" else np.concatenate((path, changes))
        net = np.array([np.abs(current[:11]).sum(), np.abs(current[11:]).sum()])
        self.previous = volume.copy()
        return np.concatenate((current, sums, net)).astype(np.float32)
