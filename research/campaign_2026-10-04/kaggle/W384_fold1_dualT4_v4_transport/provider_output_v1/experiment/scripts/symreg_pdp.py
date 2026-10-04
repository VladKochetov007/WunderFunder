"""Binned partial-dependence tables for symbolic direction terms (for plotting in the EDA notebook).

    scripts/limited .venv/bin/python scripts/symreg_pdp.py
For each class c and each term of cache/symreg_direction_c<c>.pkl, on held-out chain folds 2-3 scored
rows: decile bins of the term value -> |y|-weighted mean target direction D = t0 - t1, the GRU's mean
predicted direction, and bin weight. Writes research/artifacts/symreg/pdp_direction.parquet.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

from wnn.symreg import predict_terms

ROOT = Path(__file__).resolve().parents[1]


def main():
    h = np.load(ROOT / "cache" / "heur_valid.npz")
    sc = h["scored"]
    seq, step = h["seq"][sc], h["step"][sc]
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    cls, rg = d["market_class"], d["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    vm = cf["split"] == "valid"
    fm = dict(zip(cf["row_group"][vm].tolist(), cf["fold"][vm].tolist()))
    fold = np.array([fm[int(g)] for g in rg])[seq]
    idx = np.flatnonzero(sc)
    P = np.load(ROOT / "cache" / "pred_ab24_eps003_valid.npy", mmap_mode="r")
    rows_out = []
    for c in range(4):
        rows = np.flatnonzero((cls[seq] == c) & (fold >= 2))
        X = np.concatenate([h["X"][idx[rows]], h["raw"][idx[rows]]], 1)
        res = pickle.load(open(ROOT / "cache" / f"symreg_direction_c{c}.pkl", "rb"))
        T = predict_terms(res, X)
        y = np.clip(d["y"][seq[rows], step[rows]].astype(np.float64), -2, 2)
        D = y[:, 0] - y[:, 1]
        w = 0.5 * (np.abs(y[:, 0]) + np.abs(y[:, 1]))
        p = np.clip(np.asarray(P[seq[rows], step[rows]], dtype=np.float64), -2, 2)
        g = p[:, 0] - p[:, 1]
        for j, term in enumerate(res.terms):
            v = T[:, j]
            edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
            b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
            for k in range(len(edges) - 1):
                m = b == k
                if m.sum() < 50:
                    continue
                rows_out.append({"cls": c, "term": j + 1, "expr": term.expr, "bin": k, "x": float(np.median(v[m])),
                                 "target_D": float((w[m] * D[m]).sum() / w[m].sum()), "gru_D": float((w[m] * g[m]).sum() / w[m].sum()),
                                 "weight": float(w[m].sum())})
    pl.DataFrame(rows_out).write_parquet(ROOT / "research" / "artifacts" / "symreg" / "pdp_direction.parquet")
    print(len(rows_out), "bins")


if __name__ == "__main__":
    main()
