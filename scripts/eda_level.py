"""EDA: where does the oracle per-sequence level gain come from?

    .venv/bin/python -u scripts/eda_level.py [--members ft256H+ft128H+ft128s4H]

Decomposes targets and predictions into the half-sum S = (t0 + t1) / 2 (spread-like, mostly known from the
book) and the half-difference D = (t0 - t1) / 2 (future move). For each component separately it replaces the
prediction's per-sequence weighted mean with the target's per-sequence weighted mean (oracle level) and
reports pooled WP. Then it checks causal level estimates: expanding-window means of the model's own
predictions (no labels) do not help by construction, so instead it reports how well the per-sequence target
level is explained by the per-sequence prediction level (R^2) and by class, split by component.
"""

from __future__ import annotations

import argparse

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums

CLIP = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="ft256H+ft128H+ft128s4H")
    a = ap.parse_args()
    tg = np.load(ROOT / "cache" / "valid_targets.npz")
    rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    Y, P, M, C = [], [], [], []
    for k in range(4):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows].astype(np.float32)
                 for r in a.members.split("+")]
        Y.append(tg["y"][rows]); P.append(sum(preds) / len(preds))
        m = tg["mask"][rows].copy(); m[:, :WARMUP] = False; M.append(m); C.append(classes[rg[rows]])
    Y, P, M, C = map(np.concatenate, (Y, P, M, C))
    base = pooled(sums(Y, P, M).sum(0))
    print(f"baseline {base:.5f}", flush=True)

    y = np.clip(Y.astype(np.float64), -CLIP, CLIP)
    p = np.clip(P.astype(np.float64), -CLIP, CLIP)
    w = np.abs(y).sum(-1) * M  # metric weight mass per row (both targets)
    W = w.sum(1)
    ys, yd = (y[..., 0] + y[..., 1]) / 2, (y[..., 0] - y[..., 1]) / 2
    ps, pd = (p[..., 0] + p[..., 1]) / 2, (p[..., 0] - p[..., 1]) / 2

    def seq_mean(z):
        return (w * z).sum(1) / np.maximum(W, 1e-9)

    lvl = {"S": (seq_mean(ys), seq_mean(ps)), "D": (seq_mean(yd), seq_mean(pd))}
    for comp in ("S", "D", "both"):
        s2, d2 = ps.copy(), pd.copy()
        if comp in ("S", "both"):
            s2 += (lvl["S"][0] - lvl["S"][1])[:, None]
        if comp in ("D", "both"):
            d2 += (lvl["D"][0] - lvl["D"][1])[:, None]
        q = np.stack([s2 + d2, s2 - d2], -1).astype(np.float32)
        print(f"oracle level on {comp:4s}: {pooled(sums(Y, q, M).sum(0)):.5f}", flush=True)

    for comp, (ty, tp) in lvl.items():
        ok = W > 0
        r2 = np.corrcoef(ty[ok], tp[ok])[0, 1] ** 2
        print(f"{comp}: per-sequence target level sd {ty[ok].std():.4f}, prediction level sd {tp[ok].std():.4f}, "
              f"R^2(pred level -> target level) {r2:.3f}", flush=True)
        for c in range(4):
            sel = ok & (C == c)
            print(f"   class {c}: target level mean {ty[sel].mean():+.4f} sd {ty[sel].std():.4f} | pred level mean "
                  f"{tp[sel].mean():+.4f} sd {tp[sel].std():.4f} | R^2 {np.corrcoef(ty[sel], tp[sel])[0, 1] ** 2:.3f}",
                  flush=True)

    # share of weighted target variance that is between-sequence (level) vs within-sequence, per component
    for comp, z in (("S", ys), ("D", yd)):
        mu = (w * z).sum() / w.sum()
        tot = (w * (z - mu) ** 2).sum()
        between = (W * (seq_mean(z) - mu) ** 2).sum()
        print(f"{comp}: between-sequence share of weighted variance {between / tot:.3f}", flush=True)


if __name__ == "__main__":
    main()
