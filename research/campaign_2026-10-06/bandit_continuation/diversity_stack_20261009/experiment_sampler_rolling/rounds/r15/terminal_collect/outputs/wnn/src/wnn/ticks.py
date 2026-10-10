"""Exact tick-path reconstruction and label-inversion tools for large-tick sequences.

Book-side encodings are global deterministic codes of the book shape, so a
transition between two codes always means the same quote move (research
2026-09-25 EDA). The move of each code pair is learned from labelled sequences
as the dominant target jump: at a book event the target changes by -d(ask) for
t0 and +d(bid) for t1 while the future reference is unchanged, which holds for
~98% of events. The table then reconstructs ask/bid paths in ticks from
features alone (causal, no targets needed at inference).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numba as nb
import numpy as np

CLASS3_QUANTUM = 0.65687


def codes(x) -> np.ndarray:
    """Integer id per row for the full 22-column i0 price code (ids are per call)."""
    _, inv = np.unique(np.round(x[:, 0:22], 4), axis=0, return_inverse=True)
    return inv.ravel()


def _keys(x):
    r = np.round(x[:, 0:22], 4)
    return [tuple(v) for v in r]


def learn_moves(sequences, quantum=CLASS3_QUANTUM) -> dict:
    """sequences: iterable of (features, targets). Returns {(code_prev, code_next): (d_ask, d_bid)}."""
    trans = defaultdict(Counter)
    for x, y in sequences:
        q = np.rint(y / quantum).astype(np.int64)
        ev = np.flatnonzero((x[1:, 0:22] != x[:-1, 0:22]).any(1)) + 1
        if len(ev) == 0:
            continue
        k = _keys(x[np.r_[ev - 1, ev]])
        n = len(ev)
        for i, e in enumerate(ev):
            trans[(k[i], k[n + i])][tuple(q[e] - q[e - 1])] += 1
    return {key: (-int(c.most_common(1)[0][0][0]), int(c.most_common(1)[0][0][1])) for key, c in trans.items()}


def learn_spreads(sequences, quantum=CLASS3_QUANTUM) -> dict:
    """Modal spread in ticks per book code, read as -(t0 + t1) (exact when both trades share an exit)."""
    table = defaultdict(Counter)
    for x, y in sequences:
        s = -np.rint((y[:, 0] + y[:, 1]) / quantum).astype(np.int64)
        k = _keys(x)
        for key, v in zip(k[::7], s[::7]):
            table[key][int(v)] += 1
    return {key: c.most_common(1)[0][0] for key, c in table.items()}


def reconstruct(x, moves, spreads=None) -> tuple[np.ndarray, np.ndarray, int]:
    """Ask and bid paths in ticks and the count of unknown transitions.

    With a per-code spread table the bid is ask minus the tabled spread (drift then only
    affects the level); without one the modal spread is anchored at one tick.
    """
    n = len(x)
    da = np.zeros(n, np.int64)
    db = np.zeros(n, np.int64)
    unknown = 0
    ev = np.flatnonzero((x[1:, 0:22] != x[:-1, 0:22]).any(1)) + 1
    if len(ev):
        k = _keys(x[np.r_[ev - 1, ev]])
        m = len(ev)
        for i, e in enumerate(ev):
            mv = moves.get((k[i], k[m + i]))
            if mv is None:
                unknown += 1
                continue
            da[e], db[e] = mv
    a = np.cumsum(da)
    b = np.cumsum(db)
    if spreads:
        # Spread is a function of the book code: take it from the table where known and
        # integrate bid moves only through codes missing from the table.
        k = _keys(x)
        want = np.array([spreads.get(key, -1) for key in k], np.int64)
        for i in range(n):
            if want[i] >= 0:
                b[i] = a[i] - want[i]
            elif i > 0:
                b[i] = b[i - 1] + db[i]
            else:
                b[i] = a[i] - 1
    else:
        spread = a - b
        b += int(np.bincount(spread - spread.min()).argmax() + spread.min()) - 1
    return a, b, unknown


@nb.njit(cache=True)
def consistent_exits(a, b, t0, t1, rows, max_lag, stop):
    """For each entry row s, the first and last exit row j in (s, s+max_lag] where a joint
    stop-loss/horizon exit reproduces both targets (in ticks): long entered at a[s], short at b[s],
    each stopped at -stop (value floored at -stop), otherwise marked to b[j] / a[j] at exit j.
    Returns (first, last, count); -1 when no row is consistent.
    """
    n = len(a)
    first = np.full(len(rows), -1, np.int64)
    last = np.full(len(rows), -1, np.int64)
    count = np.zeros(len(rows), np.int64)
    for k in range(len(rows)):
        s = rows[k]
        ea, eb = a[s], b[s]
        stopped0 = False
        stopped1 = False
        for j in range(s + 1, min(n, s + max_lag + 1)):
            v0 = b[j] - ea
            v1 = eb - a[j]
            if not stopped0 and v0 <= -stop:
                stopped0 = True
            if not stopped1 and v1 <= -stop:
                stopped1 = True
            ok0 = (t0[k] == -stop) if stopped0 else (v0 == t0[k])
            ok1 = (t1[k] == -stop) if stopped1 else (v1 == t1[k])
            if ok0 and ok1:
                if first[k] < 0:
                    first[k] = j
                last[k] = j
                count[k] += 1
            if stopped0 and stopped1:
                break
    return first, last, count
