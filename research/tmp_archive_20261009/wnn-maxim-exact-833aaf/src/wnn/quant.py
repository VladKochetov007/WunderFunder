"""Order-flow features accumulated over the current quiet interval (since the last i0 price event).

Microstructure motivation (Cont, Kukanov & Stoikov 2014, order flow imbalance; queue-reactive models):
between two price moves of a large-tick book the next move is decided by which best queue depletes
first, i.e. by net inflow/outflow at the touch and by signed trading since the last move. A GRU sees
per-row changes, but has to integrate them exactly over a variable-length window; these features give
that integral directly.

Columns follow utils.FEATURE_COLUMNS (volume codes are identifiers, not depth levels).
  vol_since[22]   i0 volume codes now minus at the last i0 price event (net queue change proxy)
  dv_since[4]     sum of the four i0 trade-volume codes over trade rows since the last price event
  dp_since[4]     sum of the four i0 trade-price codes over the same trade rows
  n_trades        log1p(trade rows since the last price event)

A trade row is a row where the trade-volume codes changed and are not all equal (the no-trade state
repeats one neutral code in all four columns). Everything at row t uses rows <= t only.
"""

from __future__ import annotations

import numpy as np

from wnn.data import I0, I1

N_QUANT = 22 + 4 + 4 + 1  # per instrument block
QUANT_NAMES = ([f"q_vol_since{j}" for j in range(22)] + [f"q_dv_since{j}" for j in range(4)]
               + [f"q_dp_since{j}" for j in range(4)] + ["q_n_trades"])
_VOL = slice(I0.bid_vol.start, I0.ask_vol.stop)


def _block(raw, prev, last, inst):
    """Flow of instrument `inst` accumulated since the last i0 price event (index `last`)."""
    import torch

    b, t, _ = raw.shape
    dv = raw[..., inst.trade_vol]
    dp = raw[..., inst.trade_px]
    changed = (dv != prev[..., inst.trade_vol]).any(-1)
    trade = changed & ((dv.amax(-1) - dv.amin(-1)) > 1e-6)
    vol = raw[..., inst.bid_vol.start : inst.ask_vol.stop]
    vol_since = vol - torch.gather(vol, 1, last[..., None].expand(b, t, 22))
    tr = trade.to(raw.dtype)[..., None]
    flow = torch.cat([dv * tr, dp * tr, tr], dim=-1).double()  # float64 cumsum: exact segment differences
    cs = torch.cumsum(flow, dim=1)
    seg = (cs - torch.gather(cs, 1, last[..., None].expand(b, t, flow.shape[-1]))).to(raw.dtype)
    return torch.cat([vol_since, seg[..., :8], torch.log1p(seg[..., 8:9].clamp_min(0))], dim=-1)


def quant_features_torch(raw, with_i1: bool = False):
    """raw: (B, T, 112) -> (B, T, N_QUANT) i0 block, plus the i1 block (cross-impact) if with_i1."""
    import torch

    b, t, _ = raw.shape
    prev = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
    px_event = (prev[..., I0.bid_px.start] != raw[..., I0.bid_px.start]) & (
        prev[..., I0.ask_px.start] != raw[..., I0.ask_px.start])
    idx = torch.arange(t, device=raw.device).expand(b, t)
    last = torch.cummax(torch.where(px_event, idx, torch.zeros_like(idx)), dim=1).values
    blocks = [_block(raw, prev, last, I0)] + ([_block(raw, prev, last, I1)] if with_i1 else [])
    return torch.cat(blocks, dim=-1)


def quant_features(x: np.ndarray) -> np.ndarray:
    """NumPy reference for one sequence (T, 112) -> (T, N_QUANT) float32 (tests and streaming parity)."""
    import torch

    with torch.no_grad():
        return quant_features_torch(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()


class StreamingQuantFeatures:
    """Row-by-row twin of quant_features_torch for the submission callback."""

    def __init__(self) -> None:
        self.out = np.zeros(N_QUANT, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.prev = None
        self.vol_at_event = None
        self.acc = np.zeros(9, dtype=np.float64)

    def update(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32)
        prev = row if self.prev is None else self.prev
        if self.vol_at_event is None:
            self.vol_at_event = row[_VOL].copy()
        dv, dp = row[I0.trade_vol], row[I0.trade_px]
        if (dv != prev[I0.trade_vol]).any() and dv.max() - dv.min() > 1e-6:
            self.acc[0:4] += dv
            self.acc[4:8] += dp
            self.acc[8] += 1.0
        # the window starts after the event row: a trade on the event row itself is not counted
        if prev[I0.bid_px.start] != row[I0.bid_px.start] and prev[I0.ask_px.start] != row[I0.ask_px.start]:
            self.vol_at_event = row[_VOL].copy()
            self.acc[:] = 0.0
        out = self.out
        out[0:22] = row[_VOL] - self.vol_at_event
        out[22:30] = self.acc[0:8]
        out[30] = np.log1p(self.acc[8])
        self.prev = row.copy()
        return out
