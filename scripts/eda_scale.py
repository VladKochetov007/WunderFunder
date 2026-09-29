"""EDA: how much of the pooled metric is about the relative scale of predictions across classes/sequences?

    .venv/bin/python scripts/eda_scale.py [--members ft256H+ft128H+ft128s4H]

Uses out-of-fold predictions of the given ensemble on the pure chains of folds 0-3.
1. Baseline pooled WP.
2. Per-class output scale, cross-fitted over folds (fit on three folds, apply to the fourth): causal and
   deployable, since the class is known by row 99.
3. Oracle per-sequence linear recalibration (p -> a_b p + c_b with a_b, c_b the weighted LS fit on the
   sequence's own targets): an upper bound on any per-sequence reliability scaling.
4. Oracle per-sequence scale only (a_b >= 0, no shift).
5. Which sequence statistics predict the oracle scale (Spearman with label-free sequence features).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums

CLIP = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="ft256H+ft128H+ft128s4H")
    ap.add_argument("--out", default="research/artifacts/quant_probes/eda_scale.json")
    a = ap.parse_args()
    tg = np.load(ROOT / "cache" / "valid_targets.npz")
    rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    Y, P, M, C, F, R = [], [], [], [], [], []
    for k in range(4):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows].astype(np.float32)
                 for r in a.members.split("+")]
        Y.append(tg["y"][rows]); P.append(sum(preds) / len(preds))
        m = tg["mask"][rows].copy(); m[:, :WARMUP] = False; M.append(m)
        C.append(classes[rg[rows]]); F.append(np.full(len(rows), k)); R.append(rg[rows])
    Y, P, M, C, F, R = map(np.concatenate, (Y, P, M, C, F, R))
    res = {"members": a.members, "sequences": int(len(Y))}
    base = pooled(sums(Y, P, M).sum(0))
    res["baseline"] = base
    print(f"sequences {len(Y)}, baseline pooled WP {base:.5f}")

    # 2. per-class scale, cross-fitted. Scaling predictions by s scales the sufficient statistics
    # analytically (Swp, Swyp by s; Swpp by s^2); clipping at +-2 binds on <0.1% of predictions.
    S = sums(Y, P, M)  # (B, 2, 6)
    def scaled(idx, sc):
        t = S[idx].copy()
        f = sc[C[idx]][:, None]
        t[..., 2] *= f; t[..., 5] *= f; t[..., 4] *= f * f
        return t.sum(0)
    grid = np.linspace(0.4, 1.8, 57)
    fitted, parts = {}, []
    for k in range(4):
        tr = np.flatnonzero(F != k)
        sc = np.ones(4)
        for _ in range(3):
            for c in range(4):
                vals = []
                for g in grid:
                    trial = sc.copy(); trial[c] = g
                    vals.append(pooled(scaled(tr, trial)))
                sc[c] = grid[int(np.argmax(vals))]
        fitted[k] = [round(float(x), 3) for x in sc]
        parts.append(scaled(np.flatnonzero(F == k), sc))
    res["class_scale_crossfit"] = pooled(sum(parts))
    res["class_scales_by_heldout_fold"] = fitted
    print(f"per-class scale (cross-fitted): {res['class_scale_crossfit']:.5f} ({res['class_scale_crossfit'] - base:+.5f}); scales {fitted}", flush=True)

    # 3/4. oracle per-sequence recalibration
    yc = np.clip(Y.astype(np.float64), -CLIP, CLIP)
    pc = np.clip(P.astype(np.float64), -CLIP, CLIP)
    Po, Ps = pc.copy(), pc.copy()
    a_scale = np.zeros((len(Y), 2))
    for b in range(len(Y)):
        for t in range(2):
            m = M[b]
            if m.sum() < 3:
                continue
            w = np.abs(yc[b, m, t]); x = pc[b, m, t]; y = yc[b, m, t]
            W = w.sum()
            if W < 1e-8:
                continue
            mx, my = (w * x).sum() / W, (w * y).sum() / W
            vx = (w * (x - mx) ** 2).sum() / W
            cov = (w * (x - mx) * (y - my)).sum() / W
            slope = cov / vx if vx > 1e-12 else 0.0
            Po[b, :, t] = slope * (pc[b, :, t] - mx) + my
            s0 = max(0.0, (w * x * y).sum() / max((w * x * x).sum(), 1e-12))
            Ps[b, :, t] = s0 * pc[b, :, t]
            a_scale[b, t] = s0
    res["oracle_seq_linear"] = pooled(sums(Y, Po, M).sum(0))
    res["oracle_seq_scale"] = pooled(sums(Y, Ps, M).sum(0))
    print(f"oracle per-sequence linear recalibration: {res['oracle_seq_linear']:.5f}; scale only: {res['oracle_seq_scale']:.5f}")

    # 5. label-free sequence statistics vs oracle scale
    from scipy.stats import spearmanr

    from wnn.data import SequenceFile

    f = SequenceFile(ROOT / "datasets" / "valid.parquet")
    feats = {"pred_abs_mean": [], "pred_std": [], "px_event_rate": [], "trade_rate": [], "a5_mean": [], "a2_mean": [],
             "a3_mean": [], "a6_mean": [], "class": []}
    for b, g in enumerate(R):
        x = f.read(int(g), targets=False, mask=False).features
        px = ((x[1:, 0] != x[:-1, 0]) & (x[1:, 11] != x[:-1, 11])).mean()
        tr = (x[1:, 48:52] != x[:-1, 48:52]).any(1).mean()
        feats["pred_abs_mean"].append(float(np.abs(pc[b, WARMUP:]).mean()))
        feats["pred_std"].append(float(pc[b, WARMUP:].std()))
        feats["px_event_rate"].append(float(px)); feats["trade_rate"].append(float(tr))
        for col, j in (("a5_mean", 109), ("a2_mean", 106), ("a3_mean", 107), ("a6_mean", 110)):
            feats[col].append(float(x[:, j].mean()))
        feats["class"].append(int(C[b]))
    target = a_scale.mean(1)
    res["oracle_scale_quantiles"] = np.quantile(target, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    res["spearman_with_oracle_scale"] = {k: float(spearmanr(v_, target).correlation) for k, v_ in feats.items()}
    print("oracle scale quantiles", np.round(res["oracle_scale_quantiles"], 3))
    print("spearman(feature, oracle scale):", {k: round(v_, 3) for k, v_ in res["spearman_with_oracle_scale"].items()})
    (ROOT / a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
