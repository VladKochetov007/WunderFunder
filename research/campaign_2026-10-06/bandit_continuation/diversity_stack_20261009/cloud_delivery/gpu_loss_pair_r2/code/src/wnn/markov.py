"""Label-free in-context Markov statistics of i0 book-state transitions.

At each joint i0 price event the book moves from state ``s`` to ``s'``. Using only
transitions completed earlier in the same sequence, estimate for the current
context the probability that the next event moves the bid (column p0) and the
ask (column p11) up or down. Values are held on rows between events.

Column ranks are global monotone transforms (H-003), so the sign of a column
difference within a sequence is the sign of the underlying price move.
"""

from __future__ import annotations

import numpy as np

N_MARKOV_FEATURES = (
    12  # 2 context orders x (count, bid up, bid down, ask up, ask down, same-as-prev)
)
_PRIOR = 1.0


def _state_ids(prices: np.ndarray) -> np.ndarray:
    _, ids = np.unique(prices, axis=0, return_inverse=True)
    return ids.ravel()


def markov_features(x: np.ndarray) -> np.ndarray:
    """x: (T, 112) -> (T, N_MARKOV_FEATURES) float32, causal."""
    t = len(x)
    ev = np.r_[True, (np.diff(x[:, 0]) != 0) & (np.diff(x[:, 11]) != 0)]
    idx = np.flatnonzero(ev)
    st = _state_ids(x[idx, 0:22])
    bid = x[idx, 0]
    ask = x[idx, 11]
    n = len(idx)
    out_ev = np.zeros((n, N_MARKOV_FEATURES), dtype=np.float32)
    # Online dictionaries: context -> [count, bid_up, bid_down, ask_up, ask_down, returns_to_prev]
    tables = ({}, {})
    for k in range(n):
        contexts = (st[k], (st[k - 1] if k > 0 else -1, st[k]))
        for order, ctx in enumerate(contexts):
            c = tables[order].get(ctx)
            base = order * 6
            if c is None:
                out_ev[k, base] = 0.0
                out_ev[k, base + 1 : base + 6] = 0.0
            else:
                total = c[0] + 2 * _PRIOR
                out_ev[k, base] = np.log1p(c[0])
                out_ev[k, base + 1 : base + 6] = (np.asarray(c[1:]) + _PRIOR) / total - 0.5
        # Transition k-1 -> k is now complete: update contexts that were current at k-1.
        if k > 0:
            j = k - 1
            upd = (
                1.0,
                float(bid[k] > bid[j]),
                float(bid[k] < bid[j]),
                float(ask[k] > ask[j]),
                float(ask[k] < ask[j]),
                float(j > 0 and st[k] == st[j - 1]),
            )
            prev_contexts = (st[j], (st[j - 1] if j > 0 else -1, st[j]))
            for order, ctx in enumerate(prev_contexts):
                c = tables[order].setdefault(ctx, [0.0] * 6)
                for q in range(6):
                    c[q] += upd[q]
    pos = np.searchsorted(idx, np.arange(t), side="right") - 1
    return out_ev[pos]
