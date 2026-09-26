"""Adversarial validation (V4) and time-order probe (V5) on per-sequence statistics.

    scripts/limited .venv/bin/python scripts/adversarial_validation.py
Within each market class, a gradient-boosting classifier separates train from validation
sequences using feature-only per-sequence statistics (no targets, no mask). Reports cross-
validated AUC per class and the most discriminative statistics, then looks for one slowly
drifting coordinate: the first principal component of standardized statistics and its
AUC for the train/valid split (a later valid period would sit at one end).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]


def main():
    tr = pl.read_parquet(ROOT / "cache" / "seqstats_train.parquet").with_columns(split=pl.lit(0))
    va = pl.read_parquet(ROOT / "cache" / "seqstats_valid.parquet").with_columns(split=pl.lit(1))
    df = pl.concat([tr, va], how="diagonal_relaxed")
    banned = ("t0_", "t1_", "S_", "D_", "t01_", "scored_frac", "ev_target", "row_group", "seq_ix", "split", "cls")
    feats = [c for c in df.columns if not any(c.startswith(b) for b in banned) and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64)]
    out = {"n_features": len(feats), "by_class": {}}
    for k in range(4):
        sub = df.filter(pl.col("cls") == k)
        X = sub.select(feats).to_numpy().astype(np.float64)
        y = sub["split"].to_numpy()
        oof = np.zeros(len(y))
        imps = np.zeros(len(feats))
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=f).fit(X[a], y[a])
            oof[b] = m.predict_proba(X[b])[:, 1]
        auc = roc_auc_score(y, oof)
        # single-feature AUCs as an importance proxy robust to correlation
        single = np.array([abs(roc_auc_score(y, np.nan_to_num(X[:, j])) - 0.5) + 0.5 for j in range(len(feats))])
        top = np.argsort(-single)[:8]
        Z = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
        pc = PCA(3).fit_transform(np.nan_to_num(Z))
        pc_auc = [round(abs(roc_auc_score(y, pc[:, i]) - 0.5) + 0.5, 4) for i in range(3)]
        out["by_class"][k] = {"n_train": int((y == 0).sum()), "n_valid": int((y == 1).sum()), "cv_auc": round(auc, 4),
                              "top_single_auc": [(feats[j], round(float(single[j]), 4)) for j in top], "pc_auc": pc_auc}
        print(k, json.dumps(out["by_class"][k]), flush=True)
    (ROOT / "research" / "artifacts" / "adversarial_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
