"""Realistic online recalibration for class 3 with labels rebuilt from the F rule.

    scripts/limited .venv/bin/python scripts/online_c3.py ab18_wmse_struct
Class 3 labels are rebuilt causally: at a book event e, F = round(mean mid over the next 4
book events) becomes known at the 4th next event; the target of rows from e up to the next
event is (F - ask, bid - F) mapped through the target law. A running per-sequence
|y|-weighted fit y ~ a + b p (prior a=0, b=1, strength lam) uses only labels available by
the current row. Reports class-3 and overall global WP on devB+holdout.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile
from wnn.gmetric import wp_from_sums, wp_sums
from wnn.splits import dev_halves, valid_row_groups
from wnn.ticks import CLASS3_QUANTUM as Q
from wnn.ticks import reconstruct

ROOT = Path(__file__).resolve().parents[1]
LAMS = (3.0, 10.0, 30.0, 100.0)


def law(F, b, a):
    sp = a - b
    u = F - b
    t0 = np.where(u < 0, -(sp + 2), u - sp)
    t1 = np.where(u > sp, -(sp + 2), -u)
    return np.stack([t0, t1], -1) * Q


def rebuilt_labels(a, b):
    """Rebuilt targets and the row at which each becomes known (n if never)."""
    n = len(a)
    mid = (a + b) / 2.0
    ev = np.r_[0, np.flatnonzero((np.diff(a) != 0) | (np.diff(b) != 0)) + 1]
    cm = np.r_[0.0, np.cumsum(mid[ev])]
    lab = np.zeros((n, 2))
    known = np.full(n, n, np.int64)
    for k, e in enumerate(ev):
        end = ev[k + 1] if k + 1 < len(ev) else n
        if k + 4 < len(ev):
            F = np.rint((cm[k + 5] - cm[k + 1]) / 4)
            lab[e:end] = law(np.array([F]), np.array([b[e]]), np.array([a[e]]))[0]
            known[e:end] = ev[k + 4]
    return lab, known


def main():
    name = sys.argv[1]
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    rg, cls = d["row_group"], d["market_class"]
    path = ROOT / "datasets" / "valid.parquet"
    _, hold = valid_row_groups(path)
    _, dev_b = dev_halves(path)
    keep = np.flatnonzero(np.isin(rg, np.r_[np.asarray(dev_b), np.asarray(hold)]))
    pred = np.load(ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
    moves, spreads = pickle.load(open(ROOT / "cache" / "class3_moves_full.pkl", "rb"))
    sf = SequenceFile(path)
    acc = {}
    lab_ok = lab_n = 0

    def add(key, k, s):
        acc[(key, k)] = acc.get((key, k), 0) + s

    for i in keep:
        y = d["y"][i].astype(np.float64)
        m = d["mask"][i]
        p = np.asarray(pred[i], dtype=np.float64)
        c = int(cls[i])
        for k in range(2):
            add(("all", "none", 0), k, wp_sums(y[:, k], p[:, k], m))
        if c != 3:
            for lam in LAMS:
                for var in ("level", "affine"):
                    for k in range(2):
                        add(("all", var, lam), k, wp_sums(y[:, k], p[:, k], m))
            continue
        for k in range(2):
            add(("c3", "none", 0), k, wp_sums(y[:, k], p[:, k], m))
        x = sf.read(int(rg[i]), targets=False, mask=False).features
        a, b, _ = reconstruct(x, moves, spreads)
        lab, known = rebuilt_labels(a, b)
        sel = (known < len(a)) & (np.arange(len(a)) >= 99)
        lab_ok += (np.abs(lab[sel] - y[sel]) < 1e-3).all(1).sum()
        lab_n += sel.sum()
        # running sums over labelled rows ordered by availability time
        order = np.argsort(known, kind="stable")
        kn = known[order]
        for k in range(2):
            yl = np.clip(lab[order, k], -2, 2)
            pl_ = p[order, k]
            w = np.abs(yl) * (kn < len(a))
            cs = [np.r_[0.0, np.cumsum(v)] for v in (w, w * pl_, w * yl, w * pl_ * pl_, w * pl_ * yl)]
            idx = np.searchsorted(kn, np.arange(len(a)), side="right")  # labels known by row s
            S = [c_[idx] for c_ in cs]
            for lam in LAMS:
                lvl = p[:, k] + (S[2] - S[1]) / (S[0] + lam)
                a11, a12, a22 = S[0] + lam, S[1], S[3] + lam
                r1, r2 = S[2], S[4] + lam
                det = a11 * a22 - a12 * a12
                aa = (r1 * a22 - a12 * r2) / det
                bb = (a11 * r2 - a12 * r1) / det
                aff = aa + bb * p[:, k]
                for var, q in (("level", lvl), ("affine", aff)):
                    add(("c3", var, lam), k, wp_sums(y[:, k], q, m))
                    add(("all", var, lam), k, wp_sums(y[:, k], q, m))
    print(f"rebuilt label exact match (class 3, labelled rows): {lab_ok / lab_n:.4f}")
    keys = sorted({kk for kk, _ in acc})
    for key in keys:
        r = [float(wp_from_sums(acc[(key, k)])) for k in range(2)]
        print(key, round(0.5 * (r[0] + r[1]), 4))


if __name__ == "__main__":
    main()
