"""Role-swap augmentation for class-3 sequences: i1 becomes the target instrument.

Labels are rebuilt causally-irrelevant (training only) from i1's future book events with the
class-3 label law (scripts/online_c3.rebuilt_labels on the reconstructed tick path); rows whose
label window contains a transition missing from the class-3 move table are flagged unreliable.
"""
from __future__ import annotations

import numpy as np

from wnn.ticks import _keys, reconstruct


def _law_labels(a, b):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from online_c3 import rebuilt_labels

    return rebuilt_labels(a, b)


def labels_with_mask(p, moves, spreads):
    """p: (n, 22) i-th instrument price code. Returns labels (n, 2) float32 and reliable-row mask."""
    n = len(p)
    a, b, _ = reconstruct(p, moves, spreads)
    lab, known = _law_labels(a, b)
    ev = np.flatnonzero((p[1:] != p[:-1]).any(1)) + 1
    unk = np.zeros(n, bool)
    if len(ev):
        k = _keys(p[np.r_[ev - 1, ev]])
        m = len(ev)
        unk[ev] = [(k[i], k[m + i]) not in moves for i in range(m)]
    cu = np.cumsum(unk)
    end = np.minimum(known, n - 1)
    ok = (known < n) & (cu[end] - cu == 0)
    return lab.astype(np.float32), ok


def swap_instruments(x):
    xs = x.copy()
    xs[:, 0:52] = x[:, 52:104]
    xs[:, 52:104] = x[:, 0:52]
    return xs
