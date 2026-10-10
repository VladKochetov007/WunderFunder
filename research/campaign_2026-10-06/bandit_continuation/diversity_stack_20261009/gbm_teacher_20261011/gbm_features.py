"""Causal windowed tabular features for the GBM teacher-family test (H-347). Independent implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from wnn.data import I0, I1  # noqa: E402
from wnn.features import event_features  # noqa: E402

WINDOWS = (3, 13, 21)
LAGS = (1, 3, 13)
EPS = 1e-6


def _csum(a):
    return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0, dtype=np.float64)], axis=0)


def roll_mean_std(a, w):
    """Trailing window mean/std; the first rows use the available prefix (causal)."""
    t = len(a)
    c1, c2 = _csum(a), _csum(a.astype(np.float64) ** 2)
    hi = np.arange(1, t + 1)
    lo = np.maximum(hi - w, 0)
    n = (hi - lo)[:, None] if a.ndim > 1 else (hi - lo)
    m = (c1[hi] - c1[lo]) / n
    v = np.maximum((c2[hi] - c2[lo]) / n - m * m, 0)
    return m.astype(np.float32), np.sqrt(v).astype(np.float32)


def roll_sum(a, w):
    c = _csum(a)
    hi = np.arange(1, len(a) + 1)
    return (c[hi] - c[np.maximum(hi - w, 0)]).astype(np.float32)


def ema(a, w):
    alpha = 2.0 / (w + 1)
    zi = (1 - alpha) * a[0:1]  # seeded by first observation
    return lfilter([alpha], [1, -(1 - alpha)], a, axis=0, zi=zi * np.ones((1,) + a.shape[1:]))[0].astype(np.float32)


def roll_rank(a, w):
    """Fraction of the trailing window (incl. current, front-padded with first value) below the current value."""
    from numpy.lib.stride_tricks import sliding_window_view

    out = np.empty(a.shape, dtype=np.float32)
    for j in range(a.shape[1]):
        col = np.concatenate([np.full(w - 1, a[0, j]), a[:, j]])
        win = sliding_window_view(col, w)
        out[:, j] = ((win < a[:, j:j + 1]).sum(1) + 0.5 * (win == a[:, j:j + 1]).sum(1)) / w
    return out


def _skew_kurt(m):
    mu = m.mean(1, keepdims=True)
    d = m - mu
    s2 = (d ** 2).mean(1) + EPS
    return (d ** 3).mean(1) / s2 ** 1.5, (d ** 4).mean(1) / s2 ** 2 - 3.0


def _rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + EPS)


def series_block(x):
    """Group-level base series (T, S) from anonymous price/volume groups."""
    names, cols = [], []
    def add(n, v):
        names.append(n); cols.append(np.asarray(v, dtype=np.float64))
    mids = []
    for tag, ins in (("i0", I0), ("i1", I1)):
        bp, ap = x[:, ins.bid_px].mean(1), x[:, ins.ask_px].mean(1)
        bv, av = x[:, ins.bid_vol].sum(1), x[:, ins.ask_vol].sum(1)
        mid = 0.5 * (bp + ap); mids.append(mid)
        add(f"{tag}_midpx", mid); add(f"{tag}_spreadpx", ap - bp)
        add(f"{tag}_bidpx0mean", bp); add(f"{tag}_askpx0mean", ap)
        add(f"{tag}_volimb", (bv - av) / (np.abs(bv) + np.abs(av) + EPS)); add(f"{tag}_bidvol", bv); add(f"{tag}_askvol", av)
        add(f"{tag}_tradepx", x[:, ins.trade_px].mean(1)); add(f"{tag}_tradevol", x[:, ins.trade_vol].sum(1))
        add(f"{tag}_bidpx_range", x[:, ins.bid_px].max(1) - x[:, ins.bid_px].min(1))
        add(f"{tag}_askpx_range", x[:, ins.ask_px].max(1) - x[:, ins.ask_px].min(1))
    add("mid_diff", mids[0] - mids[1])
    for k in range(8):
        add(f"a{k}", x[:, 104 + k])
    return names, np.stack(cols, 1)


def cross_block(x):
    """Cross-sectional shape statistics across the anonymous columns of a group, per row."""
    names, cols = [], []
    for tag, ins in (("i0", I0), ("i1", I1)):
        for gname, sl in (("bidvol", ins.bid_vol), ("askvol", ins.ask_vol), ("bidpx", ins.bid_px), ("askpx", ins.ask_px)):
            sk, ku = _skew_kurt(x[:, sl].astype(np.float64))
            names += [f"cs_{tag}_{gname}_skew", f"cs_{tag}_{gname}_kurt"]; cols += [sk, ku]
    v0 = np.concatenate([x[:, I0.bid_vol], x[:, I0.ask_vol]], 1).astype(np.float64)
    v1 = np.concatenate([x[:, I1.bid_vol], x[:, I1.ask_vol]], 1).astype(np.float64)
    p0 = np.concatenate([x[:, I0.bid_px], x[:, I0.ask_px]], 1).astype(np.float64)
    p1 = np.concatenate([x[:, I1.bid_px], x[:, I1.ask_px]], 1).astype(np.float64)
    names += ["cs_corr_vol01", "cs_corr_px01", "cs_corr_bid_ask_vol0", "cs_corr_bid_ask_vol1"]
    cols += [_rowcorr(v0, v1), _rowcorr(p0, p1), _rowcorr(x[:, I0.bid_vol].astype(np.float64), x[:, I0.ask_vol].astype(np.float64)),
             _rowcorr(x[:, I1.bid_vol].astype(np.float64), x[:, I1.ask_vol].astype(np.float64))]
    return names, np.stack(cols, 1)


def _build_names():
    return None


def sequence_features(x):
    """x: (T,112) float32 -> (names, (T,F) float32). Strictly causal (row t uses rows <= t)."""
    x = np.asarray(x, dtype=np.float32)
    t = len(x)
    ev = event_features(x)  # (T, 99)
    names = [f"raw{j}" for j in range(112)] + [f"ev{j}" for j in range(ev.shape[1])]
    blocks = [x, ev]
    sn, s = series_block(x)
    # lag differences
    for l in LAGS:
        lagged = np.concatenate([np.repeat(s[:1], l, 0), s[:-l]], 0)
        blocks.append((s - lagged).astype(np.float32)); names += [f"{n}_dlag{l}" for n in sn]
    for w in WINDOWS:
        m, sd = roll_mean_std(s, w)
        e = ema(s, w)
        r = roll_rank(s, w)
        blocks += [m, sd, e, r, (s - m).astype(np.float32)]
        names += [f"{n}_{k}{w}" for k in ("rmean", "rstd", "ema", "rank", "dev") for n in sn]
    # event activity clocks / counts
    flags = ev[:, 66:74]
    for w in WINDOWS:
        blocks.append(roll_sum(flags, w)); names += [f"evflag{k}_cnt{w}" for k in range(8)]
        px = np.stack([ev[:, 0:22].sum(1), ev[:, 44:66].sum(1), np.abs(ev[:, 0:22]).sum(1), np.abs(ev[:, 22:44]).sum(1)], 1)
        blocks.append(roll_sum(px, w)); names += [f"evsum{k}_{w}" for k in ("d0px", "d1px", "abs0px", "abs0vol")]
    cn, c = cross_block(x)
    c = c.astype(np.float32)
    blocks.append(c); names += cn
    for w in (3, 21):
        m, sd = roll_mean_std(c, w)
        blocks.append(m); names += [f"{n}_rmean{w}" for n in cn]
    out = np.concatenate([np.asarray(b, dtype=np.float32) for b in blocks], 1)
    assert out.shape == (t, len(names)), (out.shape, len(names))
    return names, np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
