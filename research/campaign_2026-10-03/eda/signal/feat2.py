"""Extra causal features: event-time FIR taps of i0 prices (cycle phase) and exact-recurrence (limit-cycle) indicators."""
import numpy as np
import numba as nb
import feat

TAP_COLS = [f'i0_p{k}' for k in range(22)] + [f'i0_dp{k}' for k in range(4)]
LAGS = (1, 2, 3, 4, 6)


@nb.njit(cache=True)
def _runlen(eq):
    out = np.zeros(len(eq), np.float32); c = 0.0
    for i in range(len(eq)):
        c = c + 1.0 if eq[i] else 0.0
        out[i] = c
    return out


def event_taps(d):
    """x at the k-th previous i0-price event (held between events) and recurrence indicators."""
    P = np.stack([d[f'i0_p{k}'] for k in range(22)], 1)
    e0p = np.r_[True, np.any(np.diff(P, axis=0) != 0, 1)]
    ev = np.where(e0p)[0]
    evid = np.searchsorted(ev, np.arange(len(P)), side='right') - 1  # index of last event <= t
    X = np.stack([d[c] for c in TAP_COLS], 1).astype(np.float32)
    Xe = X[ev]
    _, sid = np.unique(P[ev], axis=0, return_inverse=True); sid = sid.ravel()
    feats, names = [], []
    for k in LAGS:
        lag = np.clip(evid - k, 0, None)
        feats.append(Xe[lag]); names += [f'{c}|ev-{k}' for c in TAP_COLS]
    rec = []
    for k in (2, 3, 4, 6, 12):
        eq = np.r_[np.zeros(min(k, len(sid)), bool), sid[k:] == sid[:-k]]
        rl = _runlen(eq)
        rec += [eq[evid].astype(np.float32), np.log1p(rl[evid])]
        names += [f'rec{k}', f'rec{k}_run']
    # number of distinct states among the last 12 events (low = cycling)
    nd = np.array([len(set(sid[max(0, i - 11):i + 1])) for i in range(len(sid))], np.float32)
    rec.append(nd[evid] / 12.0); names.append('distinct12')
    feats.append(np.stack(rec, 1))
    return np.concatenate(feats, 1), names


def bank2(d):
    F, n = feat.bank(d)
    G, m = event_taps(d)
    return np.concatenate([F, G], 1), n + m
