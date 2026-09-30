"""Pooled WP of an OOF blend when one market class's predictions are scaled by s (pure chains, folds 0-1).

    .venv/bin/python scripts/class_scale.py MEMBERS [--cls 3]

Also prints each class's share of |y| weight, pooled covariance and prediction variance: the quantities an LB
scale probe identifies on the test.
"""

from __future__ import annotations

import argparse

import numpy as np
from fold_eval import ROOT, WARMUP, load_fold

CLIP = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("members")
    ap.add_argument("--cls", type=int, default=3)
    a = ap.parse_args()
    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    Y, P, C = [], [], []
    for k in (0, 1):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        pr = load_fold(a.members, k, rows)
        for j, i in enumerate(rows):
            m = tg["mask"][i].copy(); m[:WARMUP] = False
            Y.append(np.clip(tg["y"][i][m].astype(np.float64), -CLIP, CLIP))
            P.append(np.clip(np.asarray(pr[j][m], dtype=np.float64), -CLIP, CLIP))
            C.append(np.full(m.sum(), cls[rg[i]]))
    Y, P, C = map(np.concatenate, (Y, P, C))
    W = np.abs(Y)

    def wp(p):
        r = []
        for t in range(2):
            w, y, q = W[:, t], Y[:, t], p[:, t]
            my, mq = (w * y).sum() / w.sum(), (w * q).sum() / w.sum()
            r.append((w * (y - my) * (q - mq)).sum() / np.sqrt((w * (y - my) ** 2).sum() * (w * (q - mq) ** 2).sum()))
        return np.mean(r)

    for c in range(4):
        s = C == c
        print(f"class {c}: weight share {W[s].sum() / W.sum():.3f}  cov share {(W[s] * Y[s] * P[s]).sum() / (W * Y * P).sum():.3f}  "
              f"pred-var share {(W[s] * P[s] ** 2).sum() / (W * P * P).sum():.3f}  y-var share {(W[s] * Y[s] ** 2).sum() / (W * Y * Y).sum():.3f}  "
              f"wp {wp(np.where((C == c)[:, None], P, 0)):.4f}")
    base = wp(P)
    print(f"base {base:.5f}")
    for sc in (0.0, 0.5, 0.7, 0.85, 1.15, 1.4, 2.0):
        q = P.copy(); q[C == a.cls] *= sc
        print(f"class {a.cls} x{sc}: {wp(q):.5f} ({wp(q) - base:+.5f})")


if __name__ == "__main__":
    main()
