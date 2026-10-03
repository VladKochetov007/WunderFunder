"""Shared helpers for the fractal EDA (streaming loaders, metric, scaling estimators)."""
import numpy as np, pyarrow.parquet as pq
from numba import njit

ROOT = '/home/vlad/development/wunder/wnn_connectome_starterpack/'
TRAIN = ROOT + 'datasets/train.parquet'
VALID = ROOT + 'datasets/valid.parquet'
OUT = ROOT + 'research/campaign_2026-10-03/eda/fractal/'
FEAT = pq.ParquetFile(VALID).schema_arrow.names[4:116]
IDX = {n: i for i, n in enumerate(FEAT)}


def read_seq(path, g, cols):
    """One sequence (row group) -> dict of float32 numpy arrays (no pandas)."""
    t = pq.ParquetFile(path).read_row_group(g, columns=cols)
    return {c: t.column(c).to_numpy().astype(np.float32) if c != 'is_scored' else t.column(c).to_numpy()
            for c in cols}


def wp(y, p, w=None):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    my = (w * y).sum() / w.sum(); mp = (w * p).sum() / w.sum()
    return (w * (y - my) * (p - mp)).sum() / np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum())


def acf(x, maxlag):
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:maxlag + 1]
    return a / a[0]


def ema(x, span):
    """Causal EMA with alpha = 2/(span+1)."""
    return _ema(np.ascontiguousarray(x, dtype=np.float64), 2.0 / (span + 1.0))


@njit(cache=True)
def _ema(x, a):
    out = np.empty_like(x); m = x[0]
    for i in range(len(x)):
        m += a * (x[i] - m); out[i] = m
    return out


@njit(cache=True)
def dfa(x, scales, order):
    """DFA fluctuation F(s) of increments x (profile = cumsum of demeaned x), non-overlapping windows, both ends."""
    n = len(x); prof = np.cumsum(x - x.mean()); F = np.full(len(scales), np.nan)
    for k in range(len(scales)):
        s = scales[k]; nw = n // s
        if nw < 4:
            continue
        t = np.arange(s).astype(np.float64)
        V = np.empty((s, order + 1))
        for j in range(order + 1):
            V[:, j] = t ** j
        # projection onto polynomial basis via QR-free normal equations
        G = V.T @ V; Gi = np.linalg.inv(G)
        tot = 0.0; cnt = 0
        for rev in range(2):
            for w in range(nw):
                st = w * s if rev == 0 else n - (w + 1) * s
                seg = prof[st:st + s]
                c = Gi @ (V.T @ seg)
                r = seg - V @ c
                tot += (r * r).mean(); cnt += 1
        F[k] = np.sqrt(tot / cnt)
    return F


@njit(cache=True)
def mfdfa(x, scales, qs, order):
    """MFDFA F_q(s); returns array [len(qs), len(scales)]."""
    n = len(x); prof = np.cumsum(x - x.mean()); F = np.full((len(qs), len(scales)), np.nan)
    for k in range(len(scales)):
        s = scales[k]; nw = n // s
        if nw < 4:
            continue
        t = np.arange(s).astype(np.float64)
        V = np.empty((s, order + 1))
        for j in range(order + 1):
            V[:, j] = t ** j
        Gi = np.linalg.inv(V.T @ V)
        f2 = np.empty(2 * nw)
        i = 0
        for rev in range(2):
            for w in range(nw):
                st = w * s if rev == 0 else n - (w + 1) * s
                seg = prof[st:st + s]
                r = seg - V @ (Gi @ (V.T @ seg))
                f2[i] = max((r * r).mean(), 1e-30); i += 1
        for qi in range(len(qs)):
            q = qs[qi]
            if q == 0:
                F[qi, k] = np.exp(0.5 * np.log(f2).mean())
            else:
                F[qi, k] = ((f2 ** (q / 2)).mean()) ** (1.0 / q)
    return F


def slope(s, F, lo, hi):
    m = (s >= lo) & (s <= hi) & np.isfinite(F) & (F > 0)
    return np.polyfit(np.log(s[m]), np.log(F[m]), 1)[0] if m.sum() >= 2 else np.nan


def scales_dyadic(lo=4, hi=4096, per_oct=2):
    s = np.unique(np.round(2 ** np.arange(np.log2(lo), np.log2(hi) + 1e-9, 1 / per_oct)).astype(np.int64))
    return s


@njit(cache=True)
def _win_f2(prof, s, out):
    """Per-window DFA-1 residual variance via prefix sums; windows from both ends. Returns count."""
    n = len(prof); nw = n // s
    S1 = np.zeros(n + 1); S2 = np.zeros(n + 1); ST = np.zeros(n + 1)
    for i in range(n):
        S1[i + 1] = S1[i] + prof[i]; S2[i + 1] = S2[i] + prof[i] * prof[i]; ST[i + 1] = ST[i] + i * prof[i]
    tm = (s - 1) / 2.0; vt = (s * s - 1) / 12.0
    c = 0
    for rev in range(2):
        for w in range(nw):
            a = w * s if rev == 0 else n - (w + 1) * s
            sy = S1[a + s] - S1[a]; syy = S2[a + s] - S2[a]; sty = ST[a + s] - ST[a] - a * sy
            my = sy / s; vy = syy / s - my * my; cty = sty / s - tm * my
            r = vy - cty * cty / vt
            out[c] = r if r > 1e-30 else 1e-30; c += 1
    return c


@njit(cache=True)
def dfa1(x, scales):
    prof = np.cumsum(x - x.mean()); F = np.full(len(scales), np.nan); buf = np.empty(2 * len(x) + 2)
    for k in range(len(scales)):
        if len(x) // scales[k] < 4:
            continue
        c = _win_f2(prof, scales[k], buf); F[k] = np.sqrt(buf[:c].mean())
    return F


@njit(cache=True)
def mfdfa1(x, scales, qs):
    prof = np.cumsum(x - x.mean()); F = np.full((len(qs), len(scales)), np.nan); buf = np.empty(2 * len(x) + 2)
    for k in range(len(scales)):
        if len(x) // scales[k] < 4:
            continue
        c = _win_f2(prof, scales[k], buf); f2 = buf[:c]
        for qi in range(len(qs)):
            q = qs[qi]
            if q == 0:
                F[qi, k] = np.exp(0.5 * np.log(f2).mean())
            else:
                F[qi, k] = ((f2 ** (q / 2)).mean()) ** (1.0 / q)
    return F
