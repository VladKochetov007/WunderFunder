"""Causal event features derived from consecutive rows.

Three implementations must agree exactly: ``event_features`` (numpy, batched),
``event_features_torch`` (GPU training) and ``StreamingEventFeatures`` (one row
at a time, as in the submission callback).

Findings behind the design (research/notes): targets change only on rows where
both i0 bid and ask price columns update; trades show as transient trade-column
changes; many columns hold between events, so recency and last-event deltas
carry information a GRU would otherwise have to memorize.
"""

from __future__ import annotations

import numpy as np

from wnn.data import I0, I1

_I0_PX = slice(I0.bid_px.start, I0.ask_px.stop)  # 22
_I0_VOL = slice(I0.bid_vol.start, I0.ask_vol.stop)  # 22
_I1_PX = slice(I1.bid_px.start, I1.ask_px.stop)  # 22
_COUNTER_CAP = 10_000.0

FEATURE_NAMES = (
    [f"d_i0px{j}" for j in range(22)]
    + [f"d_i0vol{j}" for j in range(22)]
    + [f"d_i1px{j}" for j in range(22)]
    + [
        "ev_i0px",
        "ev_i0bidvol",
        "ev_i0askvol",
        "ev_i0trade_px",
        "ev_i0trade_vol",
        "ev_i1px",
        "ev_i1vol",
        "ev_i1trade",
    ]
    + ["since_i0px", "since_i0trade", "since_i1px"]
    + [f"last_d_i0px{j}" for j in range(22)]
)
N_EVENT_FEATURES = len(FEATURE_NAMES)


def _flags(prev, cur):
    """Event flags from two rows (arrays with a trailing 112 axis). Returns list of bool arrays."""
    ne = prev != cur
    return [
        ne[..., I0.bid_px.start] & ne[..., I0.ask_px.start],
        ne[..., I0.bid_vol].any(-1),
        ne[..., I0.ask_vol].any(-1),
        ne[..., I0.trade_px].any(-1),
        ne[..., I0.trade_vol].any(-1),
        ne[..., I1.bid_px.start] | ne[..., I1.ask_px.start],
        ne[..., I1.bid_vol.start : I1.ask_vol.stop].any(-1),
        ne[..., I1.trade_vol].any(-1),
    ]


def event_features(x: np.ndarray) -> np.ndarray:
    """x: (T, 112) or (B, T, 112) float32 -> (..., T, N_EVENT_FEATURES) float32."""
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    b, t, _ = x.shape
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    d = x - prev
    flags = np.stack(_flags(prev, x), axis=-1)  # (B, T, 8); row 0 is all False
    idx = np.broadcast_to(np.arange(t), (b, t))

    def last_index(event):
        return np.maximum.accumulate(np.where(event, idx, 0), axis=1)

    last_px = last_index(flags[..., 0])
    last_trade = last_index(flags[..., 4])
    last_i1 = last_index(flags[..., 5])
    since = np.stack([idx - last_px, idx - last_trade, idx - last_i1], axis=-1)
    since = np.log1p(np.minimum(since, _COUNTER_CAP)).astype(np.float32)
    held = np.take_along_axis(d[..., _I0_PX], last_px[..., None], axis=1)
    out = np.concatenate(
        [d[..., _I0_PX], d[..., _I0_VOL], d[..., _I1_PX], flags.astype(np.float32), since, held],
        axis=-1,
    )
    return out[0] if squeeze else out


def event_features_torch(x):
    """Torch twin of ``event_features`` for (B, T, 112) tensors on any device."""
    import torch

    b, t, _ = x.shape
    prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    d = x - prev
    flags = torch.stack(_flags(prev, x), dim=-1)
    idx = torch.arange(t, device=x.device).expand(b, t)
    zero = torch.zeros_like(idx)

    def last_index(event):
        return torch.cummax(torch.where(event, idx, zero), dim=1).values

    last_px = last_index(flags[..., 0])
    last_trade = last_index(flags[..., 4])
    last_i1 = last_index(flags[..., 5])
    since = torch.stack([idx - last_px, idx - last_trade, idx - last_i1], dim=-1)
    since = torch.log1p(torch.clamp(since.to(x.dtype), max=_COUNTER_CAP))
    held = torch.gather(d[..., _I0_PX], 1, last_px[..., None].expand(b, t, 22))
    return torch.cat(
        [d[..., _I0_PX], d[..., _I0_VOL], d[..., _I1_PX], flags.to(x.dtype), since, held], dim=-1
    )


class StreamingEventFeatures:
    """Per-row feature state for the submission callback."""

    def __init__(self):
        self.out = np.zeros(N_EVENT_FEATURES, dtype=np.float32)
        self.reset()

    def reset(self):
        self.prev = None
        self.step = 0
        self.last = np.zeros(3, dtype=np.int64)
        self.held = np.zeros(22, dtype=np.float32)

    def update(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32)
        prev = row if self.prev is None else self.prev
        d = row - prev
        flags = _flags(prev, row)
        step = self.step
        for k, flag_index in enumerate((0, 4, 5)):
            if flags[flag_index]:
                self.last[k] = step
        if flags[0]:
            self.held = d[_I0_PX].copy()
        out = self.out
        out[0:22] = d[_I0_PX]
        out[22:44] = d[_I0_VOL]
        out[44:66] = d[_I1_PX]
        out[66:74] = [float(f) for f in flags]
        out[74:77] = np.log1p(np.minimum(step - self.last, _COUNTER_CAP))
        out[77:99] = self.held
        self.prev = row.copy()
        self.step += 1
        return out


def online_zscore_torch(x, eps: float = 1e-4):
    """Causal per-sequence running z-scores: stats at row t use rows <= t only. x: (B, T, F)."""
    import torch

    n = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
    mean = torch.cumsum(x, 1) / n
    var = torch.cumsum(x * x, 1).div_(n).sub_(mean * mean).clamp_min_(0.0)
    return (x - mean).div_(var.add_(eps).sqrt_())
