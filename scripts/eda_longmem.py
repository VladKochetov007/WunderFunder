"""EDA: do long-memory (>> GRU memory) causal features explain the ensemble's residual on the future move?

    .venv/bin/python -u scripts/eda_longmem.py [--members ft256H+ft128H+ft128s4H] [--every 5]

Residual = D_target - D_pred with D = (t0 - t1) / 2, on scored rows of pure chains of folds 0-3 (OOF).
Candidate features (all causal, per sequence):
  expanding z-score of the i0 / i1 mid-price code, deviations from EMAs with half-lives 500/2000/8000 rows,
  changes over the last 1000 / 5000 rows, and the same for a2/a3/a5/a6 (activity clocks).
Reports the |y|-weighted correlation of each feature with the residual, by market class, and a cross-fitted
(fit on 3 folds, test on the 4th) ridge of all features -> residual, as the pooled WP change when the fitted
correction is added to D (a quick upper-ish estimate of what a model input could gain).
"""

from __future__ import annotations

import argparse

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums

from wnn.data import SequenceFile

CLIP = 2.0


def ema_dev(x: np.ndarray, hl: float) -> np.ndarray:
    a = 1.0 - 0.5 ** (1.0 / hl)
    out = np.empty_like(x)
    m = x[0]
    for t in range(len(x)):
        m = m + a * (x[t] - m)
        out[t] = x[t] - m
    return out


def features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    import numba  # noqa: F401  (ema loop is small enough in numpy for 20k rows x ~10 series)

    cols, names = [], []
    series = {"i0mid": (x[:, 0] + x[:, 11]) / 2, "i1mid": (x[:, 52] + x[:, 63]) / 2,
              "a2": x[:, 106], "a3": x[:, 107], "a5": x[:, 109], "a6": x[:, 110]}
    n = np.arange(1, len(x) + 1)
    for name, s in series.items():
        s = s.astype(np.float64)
        mu = np.cumsum(s) / n
        var = np.maximum(np.cumsum(s * s) / n - mu * mu, 0)
        cols.append((s - mu) / np.sqrt(var + 1e-4)); names.append(f"{name}_z")
        for hl in (500, 2000, 8000):
            cols.append(ema_dev(s, hl)); names.append(f"{name}_dev{hl}")
        if name.endswith("mid"):
            for L in (1000, 5000):
                d = np.zeros_like(s); d[L:] = s[L:] - s[:-L]
                cols.append(d); names.append(f"{name}_chg{L}")
    return np.stack(cols, -1), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="ft256H+ft128H+ft128s4H")
    ap.add_argument("--every", type=int, default=5, help="subsample scored rows for the screen")
    a = ap.parse_args()
    tg = np.load(ROOT / "cache" / "valid_targets.npz")
    rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    sf = SequenceFile(ROOT / "datasets" / "valid.parquet")
    X, R, Wt, CL, FO = [], [], [], [], []
    seq_rows = []
    names = None
    for k in range(4):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows].astype(np.float32)
                 for r in a.members.split("+")]
        P = sum(preds) / len(preds)
        for j, i in enumerate(rows):
            m = tg["mask"][i].copy(); m[:WARMUP] = False
            idx = np.flatnonzero(m)[:: a.every]
            if len(idx) == 0:
                continue
            y = np.clip(tg["y"][i].astype(np.float64), -CLIP, CLIP)
            p = np.clip(P[j].astype(np.float64), -CLIP, CLIP)
            f, names = features(sf.read(int(rg[i]), targets=False, mask=False).features)
            res = (y[idx, 0] - y[idx, 1]) / 2 - (p[idx, 0] - p[idx, 1]) / 2
            X.append(f[idx]); R.append(res); Wt.append(np.abs(y[idx]).sum(1))
            CL.append(np.full(len(idx), classes[rg[i]])); FO.append(np.full(len(idx), k))
        print(f"fold {k} loaded", flush=True)
    X, R, Wt, CL, FO = map(np.concatenate, (X, R, Wt, CL, FO))
    X = np.nan_to_num(X)

    def wcorr(a_, b_, w_):
        ma, mb = (w_ * a_).sum() / w_.sum(), (w_ * b_).sum() / w_.sum()
        return ((w_ * (a_ - ma) * (b_ - mb)).sum() / np.sqrt((w_ * (a_ - ma) ** 2).sum() * (w_ * (b_ - mb) ** 2).sum()))

    print(f"rows {len(R)}; weighted corr of feature with residual (classes 0..3, all):", flush=True)
    for j, nm in enumerate(names):
        cs = [wcorr(X[CL == c, j], R[CL == c], Wt[CL == c]) for c in range(4)]
        print(f"  {nm:14s} " + " ".join(f"{c:+.3f}" for c in cs) + f"   all {wcorr(X[:, j], R, Wt):+.3f}", flush=True)

    # cross-fitted ridge per class: how much residual variance is linearly explainable out of fold
    for c in range(4):
        num = den = 0.0
        for k in range(4):
            tr, te = (CL == c) & (FO != k), (CL == c) & (FO == k)
            if te.sum() == 0:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = (X[tr] - mu) / sd; w = Wt[tr]
            beta = np.linalg.solve(A.T @ (A * w[:, None]) + 10.0 * np.eye(A.shape[1]), A.T @ (w * R[tr]))
            pred = ((X[te] - mu) / sd) @ beta
            num += (Wt[te] * (R[te] - pred) ** 2).sum(); den += (Wt[te] * R[te] ** 2).sum()
        print(f"class {c}: cross-fitted ridge explains {1 - num / den:+.4f} of weighted residual MSE", flush=True)


if __name__ == "__main__":
    main()
