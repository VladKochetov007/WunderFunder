"""Causal filter-bank features. All filters are one-sided (use rows <= t only)."""
import numpy as np
import numba as nb
import sio

INP = [c for c in sio.columns() if c not in ('seq_ix', 'step_in_seq', 'need_prediction', 'is_scored', 't0', 't1')]
GROUPS = {
    'i0p': [c for c in INP if c.startswith('i0_p')], 'i0v': [c for c in INP if c.startswith('i0_v')],
    'i0d': [c for c in INP if c.startswith('i0_d')], 'i1p': [c for c in INP if c.startswith('i1_p')],
    'i1v': [c for c in INP if c.startswith('i1_v')], 'i1d': [c for c in INP if c.startswith('i1_d')],
    'a': [c for c in INP if c.startswith('a')],
}
SENT = -5.0


@nb.njit(cache=True)
def ema_clock(X, tick, alpha):
    """EMA that advances only on rows where tick is True (event clock); held constant otherwise."""
    n, f = X.shape
    out = np.empty_like(X)
    m = X[0].copy()
    for t in range(n):
        if tick[t]:
            for j in range(f):
                m[j] += alpha * (X[t, j] - m[j])
        out[t] = m
    return out


@nb.njit(cache=True)
def since(tick):
    out = np.empty(len(tick), np.float32); c = 0.0
    for i in range(len(tick)):
        c = 0.0 if tick[i] else c + 1.0
        out[i] = c
    return out


@nb.njit(cache=True)
def count_ema(tick, alpha):
    out = np.empty(len(tick), np.float32); m = 0.0
    for i in range(len(tick)):
        m += alpha * ((1.0 if tick[i] else 0.0) - m)
        out[i] = m
    return out


def raw(d, cols=INP):
    X = np.stack([d[c] for c in cols], 1).astype(np.float32)
    return X


def clean(X, cols=INP):
    """Sentinel (-5.2) in a5..a7 -> 0 plus indicator columns."""
    ind = []
    for j, c in enumerate(cols):
        if c in ('a5', 'a6', 'a7'):
            s = X[:, j] < SENT
            ind.append(s.astype(np.float32))
            X[:, j] = np.where(s, 0.0, X[:, j])
    return X, (np.stack(ind, 1) if ind else np.zeros((len(X), 0), np.float32))


def clocks(d):
    P0 = np.stack([d[f'i0_p{k}'] for k in range(22)], 1)
    e0p = np.r_[True, np.any(np.diff(P0, axis=0) != 0, 1)]
    V0 = np.stack([d[f'i0_v{k}'] for k in range(22)], 1)
    e0v = np.r_[True, np.any(np.diff(V0, axis=0) != 0, 1)]
    P1 = np.stack([d[f'i1_p{k}'] for k in range(22)], 1)
    V1 = np.stack([d[f'i1_v{k}'] for k in range(22)], 1)
    e1 = np.r_[True, np.any(np.diff(P1, axis=0) != 0, 1) | np.any(np.diff(V1, axis=0) != 0, 1)]
    return e0p, e0p | e0v, e1


def bank(d, taus=(4, 32, 256), clock='ev', cols=INP):
    """Levels + band-pass differences of EMAs on the chosen clock. Returns (F, names)."""
    X, ind = clean(raw(d, cols), cols)
    e0p, e0, e1 = clocks(d)
    tick = e0p if clock == 'ev' else np.ones(len(X), bool)
    feats, names = [X], [f'{c}|lvl' for c in cols]
    prev = X
    for tau in taus:
        E = ema_clock(X, tick, 1.0 / tau)
        feats.append(prev - E); names += [f'{c}|band<{tau}' for c in cols]
        prev = E
    clk = np.stack([np.log1p(since(e0p)), np.log1p(since(e0)), np.log1p(since(e1)),
                    count_ema(e0p, 1 / 64), count_ema(e0, 1 / 64), count_ema(e1, 1 / 64)], 1)
    feats += [ind, clk]
    names += [f'sent_{k}' for k in range(ind.shape[1])] + ['log_since_i0p', 'log_since_i0', 'log_since_i1', 'rate_i0p', 'rate_i0', 'rate_i1']
    return np.concatenate(feats, 1).astype(np.float32), names


def wcorr(p, y):
    """Competition-style score: Pearson weighted by |y|, both clipped to [-2, 2]."""
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    sw = w.sum(); mp = (w * p).sum() / sw; my = (w * y).sum() / sw
    cov = (w * (p - mp) * (y - my)).sum(); vp = (w * (p - mp) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    return cov / np.sqrt(vp * vy + 1e-30)
