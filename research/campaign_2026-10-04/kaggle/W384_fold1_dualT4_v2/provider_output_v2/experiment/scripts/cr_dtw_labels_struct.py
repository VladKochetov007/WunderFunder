"""Label structure in event time and elapsed-time drift calibration (reads cr_dtw_labels_build2.py output).

(a) Drift coefficient per elapsed bucket of t - e_k: p' = p + g_b (p_t - p_{e_k}), g_b per (class, bucket) fit by
    coordinate ascent on three folds, applied to the fourth.
(b) Exit level X = D/q + mid (ticks; D = (t0 - t1)/2, the target's future direction; interior t0 + t1 = -spread so
    t0 = X - ask, t1 = bid - X). Event-time persistence of X (labels, i.e. training-time information only) and how
    the model's implied level Xhat = p_D/q + mid moves at events where X does / does not move.

    scripts/limited .venv/bin/python scripts/cr_dtw_labels_struct.py
"""

from __future__ import annotations

import json

import numpy as np
from cr_dtw_labels_common import OUT, PUBLIC_MIX, wp6

QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
GRID = np.round(np.arange(-0.6, 0.6001, 0.02), 3)


def corr_b(cs, g):
    """(..., B, 2, 10) -> (..., 2, 6) summed over buckets with per-bucket g (broadcast on the last axis of g)."""
    g = np.asarray(g, dtype=np.float64)[..., None]
    s = cs[..., :6].copy()
    s[..., 2] += g * cs[..., 6]
    s[..., 4] += 2 * g * cs[..., 7] + g * g * cs[..., 8]
    s[..., 5] += g * cs[..., 9]
    return s.sum(-3)


def fit_g(cs):
    """cs (B, 2, 10) pooled over training folds of one class -> per-bucket g by two coordinate-ascent sweeps."""
    nb = cs.shape[0]
    g = np.zeros(nb)
    for _ in range(2):
        for b in range(nb):
            best, bv = g[b], -9.0
            for v in GRID:
                g2 = g.copy()
                g2[b] = v
                sc = wp6(corr_b(cs, g2)).mean()
                if sc > bv:
                    best, bv = v, sc
            g[b] = best
    return g


def main():
    z = np.load(OUT / "tables2.npz")
    t = {k: z[k] for k in z.files}
    res = {}
    E = t["elapsed_cells"]  # (class, fold, bucket, 2, 10)
    edges = t["edges"].tolist()
    G = np.zeros((4, 4, E.shape[2]))
    for c in range(4):
        for k in range(4):
            G[c, k] = fit_g(E[c, [f for f in range(4) if f != k]].sum(0))
    new = corr_b(E, G)  # (4, 4, 2, 6)
    ref = E[..., :6].sum(2)
    share = np.bincount(t["cls_of_seq"], minlength=4) / len(t["cls_of_seq"])
    out = {}
    for name, cw in (("pooled", np.ones(4)), ("public_mix", PUBLIC_MIX / share)):
        f = lambda a: float(wp6((a * cw[:, None, None, None]).sum((0, 1))).mean())
        out[name] = f(new) - f(ref)
        out[name + "_fold"] = [float(wp6((new * cw[:, None, None, None])[:, k].sum(0)).mean()
                                     - wp6((ref * cw[:, None, None, None])[:, k].sum(0)).mean()) for k in range(4)]
    out["class"] = [float(wp6(new[c].sum(0)).mean() - wp6(ref[c].sum(0)).mean()) for c in range(4)]
    out["g_by_class_bucket_fold_mean"] = np.round(G.mean(1), 3).tolist()
    out["g_by_class_bucket_fold_sd"] = np.round(G.std(1), 3).tolist()
    out["bucket_lower_edges"] = edges
    out["bucket_weight_share"] = np.round(E[..., 0].sum((1, 3)) / E[..., 0].sum((1, 2, 3))[:, None], 4).tolist()
    res["elapsed_drift"] = out
    print(json.dumps(out, default=float))

    # (b) exit-level persistence
    seq, mid, dseg = t["ev_seq"], t["ev_mid"].astype(np.float64), t["ev_dseg"]
    y, p, ln = t["ev_y"].astype(np.float64), t["ev_p"].astype(np.float64), t["ev_length"].astype(np.float64)
    cls = t["cls_of_seq"][seq]
    lab = {}
    for c in range(4):
        q = QUANTA[c]
        s = np.flatnonzero(cls == c)
        X = (y[s, 0] - y[s, 1]) / 2 / q + mid[s]
        Xh = (np.clip(p[s, 0], -2, 2) - np.clip(p[s, 1], -2, 2)) / 2 / q + mid[s]
        Dy = (y[s, 0] - y[s, 1]) / 2 / q
        nxt = (seq[s][1:] == seq[s][:-1]) & (dseg[s][1:] == dseg[s][:-1]) & (dseg[s][1:] >= 0) \
            & np.isfinite(X[1:]) & np.isfinite(X[:-1]) & (np.abs(y[s][1:]).max(1) < 2) & (np.abs(y[s][:-1]).max(1) < 2)
        dX, dXh, dm = (X[1:] - X[:-1])[nxt], (Xh[1:] - Xh[:-1])[nxt], (mid[s][1:] - mid[s][:-1])[nxt]
        wl = ln[s][1:][nxt]
        same = np.abs(dX) < 0.25
        # run length of constant X in events
        brk = np.r_[True, ~same]
        runs = np.diff(np.r_[np.flatnonzero(brk), len(brk)])
        lab[c] = {"n_pairs": int(nxt.sum()),
                  "P(X unchanged) events": float(same.mean()),
                  "P(X unchanged) row-weighted": float((wl * same).sum() / wl.sum()),
                  "mean X-run length (events)": float(runs.mean()),
                  "P(X unchanged | mid moved)": float(same[dm != 0].mean()),
                  "sd dX | moved": float(dX[~same].std()), "sd dXhat | X same": float(dXh[same].std()),
                  "sd dXhat | X moved": float(dXh[~same].std()),
                  "corr(dXhat, dX)": float(np.corrcoef(dXh, dX)[0, 1]),
                  "slope dD_hat on dmid | X same (ideal -1)": float(np.polyfit(dm[same], (dXh - dm)[same], 1)[0]) if same.any() else None,
                  "P(|D|<=1 tick)": float((np.abs(Dy) <= 1).mean())}
    res["exit_level"] = lab
    print(json.dumps(lab, indent=1, default=float))
    (OUT / "struct.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
