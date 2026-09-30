"""Cross-fitted blend weights (global or per market class) for a set of OOF members, exact pooled metric.

    .venv/bin/python -u scripts/blend_weights.py m1 m2 ... [--folds 0,1] [--per-class]

For each held-out fold, weights (>= 0, summing to 1) are fitted on the other listed folds by maximising the
pooled |y|-weighted Pearson of the blend, computed analytically from per-sequence moment sums including all
pairwise member products (prediction clipping at +-2 ignored: it binds on <0.1% of rows). The fitted blend is
scored on the held-out fold and compared with the equal-weight blend of the same members.
"""

from __future__ import annotations

import argparse

import numpy as np
from fold_eval import ROOT, WARMUP
from scipy.optimize import minimize

CLIP = 2.0


def moments(y, preds, m):
    """Per sequence and target: W, Swy, Swyy, Swp_i (K), Swyp_i (K), Swp_ip_j (KxK)."""
    y = np.clip(y.astype(np.float64), -CLIP, CLIP)
    P = np.stack([np.clip(p.astype(np.float64), -CLIP, CLIP) for p in preds], -1)  # (B,T,2,K)
    w = np.abs(y) * m[..., None]
    W = w.sum(1); Sy = (w * y).sum(1); Syy = (w * y * y).sum(1)
    Sp = np.einsum("btk,btkm->bkm", w, P)
    Syp = np.einsum("btk,btkm->bkm", w * y, P)
    Spp = np.einsum("btk,btkm,btkn->bkmn", w, P, P)
    return W, Sy, Syy, Sp, Syp, Spp


def wp(mom, idx, a):
    W, Sy, Syy, Sp, Syp, Spp = (x[idx].sum(0) for x in mom)
    out = []
    for t in range(2):
        mp = Sp[t] @ a / W[t]; my = Sy[t] / W[t]
        cov = Syp[t] @ a / W[t] - my * mp
        vy = Syy[t] / W[t] - my * my
        vp = a @ Spp[t] @ a / W[t] - mp * mp
        out.append(cov / np.sqrt(vy * vp))
    return float(np.mean(out))


def wp_class(mom, idx, cls, A):
    """Blend with per-class weights A[c]: combine by scaling each sequence's member moments."""
    W, Sy, Syy, Sp, Syp, Spp = (x[idx] for x in mom)
    a = A[cls[idx]]  # (n, K)
    Sp_b = np.einsum("bkm,bm->bk", Sp, a); Syp_b = np.einsum("bkm,bm->bk", Syp, a)
    Spp_b = np.einsum("bkmn,bm,bn->bk", Spp, a, a)
    Wt, Syt, Syyt, Spt, Sypt, Sppt = W.sum(0), Sy.sum(0), Syy.sum(0), Sp_b.sum(0), Syp_b.sum(0), Spp_b.sum(0)
    r = []
    for t in range(2):
        my, mp = Syt[t] / Wt[t], Spt[t] / Wt[t]
        r.append((Sypt[t] / Wt[t] - my * mp) / np.sqrt((Syyt[t] / Wt[t] - my * my) * (Sppt[t] / Wt[t] - mp * mp)))
    return float(np.mean(r))


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--per-class", action="store_true")
    a = ap.parse_args()
    K = len(a.members)
    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    folds = [int(f) for f in a.folds.split(",")]
    parts, F, C = [], [], []
    for k in folds:
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        m = tg["mask"][rows].copy(); m[:, :WARMUP] = False
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows] for r in a.members]
        for s in range(0, len(rows), 64):  # chunk to bound memory
            parts.append(moments(tg["y"][rows[s:s + 64]], [p[s:s + 64] for p in preds], m[s:s + 64]))
        F.append(np.full(len(rows), k)); C.append(classes[rg[rows]])
    mom = tuple(np.concatenate([p[i] for p in parts]) for i in range(6))
    F, C = np.concatenate(F), np.concatenate(C)
    eq = np.full(K, 1.0 / K)
    tot_fit = tot_eq = 0.0
    for k in folds:
        tr, te = np.flatnonzero(F != k), np.flatnonzero(F == k)
        if a.per_class:
            def loss(z):
                return -wp_class(mom, tr, C, softmax(z.reshape(4, K)))
            z = minimize(loss, np.zeros(4 * K), method="Powell", options={"maxiter": 4000}).x
            A = softmax(z.reshape(4, K))
            fit, base = wp_class(mom, te, C, A), wp(mom, te, eq)
            wdesc = np.round(A, 2).tolist()
        else:
            z = minimize(lambda z: -wp(mom, tr, softmax(z)), np.zeros(K), method="Powell").x
            wv = softmax(z); fit, base = wp(mom, te, wv), wp(mom, te, eq); wdesc = np.round(wv, 3).tolist()
        print(f"held-out fold {k}: fitted {fit:.5f} vs equal {base:.5f} ({fit - base:+.5f}); weights {wdesc}", flush=True)
        tot_fit += fit - base
    print(f"mean held-out gain over equal weights: {tot_fit / len(folds):+.5f}")


if __name__ == "__main__":
    main()
