"""Residual-lift probe: does a candidate causal feature set add information beyond a model?

    scripts/limited .venv/bin/python scripts/lift_probe.py --groups since_,rate_ [--model ab20_gru128_e2]
    scripts/limited .venv/bin/python scripts/lift_probe.py --npy extra.npy --name myidea

On scored validation rows (cache/heur_valid.npz), per market class and target, fit a
|y|-weighted ridge of the model residual (clip(y) - clip(p)) on the candidate features
(standardized, plus their products with the model prediction) using devA sequences, apply
the correction on devB + holdout, and report the global WP change with a paired sequence
bootstrap. ``--npy`` takes a float32 array aligned to the scored rows of heur_valid (in
file order). Results append to research/artifacts/lift_probe.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.gmetric import wp_from_sums, wp_sums
from wnn.splits import dev_halves

ROOT = Path(__file__).resolve().parents[1]


def seq_sums(y, p, seq, n_seq):
    """(n_seq, 2, 6) per-sequence WP sums for bootstrap."""
    out = np.zeros((n_seq, 2, 6))
    for k in range(2):
        yk = np.clip(y[:, k], -2, 2)
        pk = np.clip(p[:, k], -2, 2)
        w = np.abs(yk)
        for col, v in enumerate((w, w * yk, w * pk, w * yk * yk, w * pk * pk, w * yk * pk)):
            out[:, k, col] = np.bincount(seq, weights=v, minlength=n_seq)
    return out


def wp_of(s):
    return float(np.mean([wp_from_sums(s[:, k].sum(0)) for k in range(2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="", help="comma-separated heuristic name prefixes, 'all', or 'raw'")
    ap.add_argument("--npy", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--model", default="ab20_gru128_e2")
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--no-interactions", action="store_true")
    ap.add_argument("--gbm", action="store_true", help="nonlinear residual model (HistGradientBoosting) instead of ridge")
    ap.add_argument("--with-raw", action="store_true", help="append the 112 raw columns to the candidate features")
    ap.add_argument("--grouping", choices=["chain", "dev"], default="chain",
                    help="chain: fit on chain folds 0-1, evaluate on 2-3 (no neighbour leakage); dev: devA -> devB+holdout")
    a = ap.parse_args()
    h = np.load(ROOT / "cache" / "heur_valid.npz")
    sc = h["scored"]
    seq, step = h["seq"][sc], h["step"][sc]
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    cls, rg = d["market_class"], d["row_group"]
    y = d["y"][seq, step].astype(np.float64)
    p = np.asarray(np.load(ROOT / "cache" / f"pred_{a.model}_valid.npy", mmap_mode="r")[seq, step], dtype=np.float64)
    names = list(h["names"])
    cols = []
    if a.npy:
        F = np.load(a.npy).astype(np.float32)
        label = a.name or Path(a.npy).stem
    else:
        prefixes = [g for g in a.groups.split(",") if g]
        if prefixes == ["raw"]:
            F = h["raw"][sc].astype(np.float32)
        else:
            cols = [i for i, n in enumerate(names) if prefixes == ["all"] or any(n.startswith(g) for g in prefixes)]
            F = h["X"][sc][:, cols].astype(np.float32)
        label = a.name or a.groups
    if a.with_raw:
        F = np.concatenate([F, h["raw"][sc]], 1).astype(np.float32, copy=False)
        label += "+raw"
    if a.gbm:
        label += "|gbm"
    if a.grouping == "chain":
        cf = np.load(ROOT / "cache" / "chain_folds.npz")
        vmask = cf["split"] == "valid"
        fold_map = dict(zip(cf["row_group"][vmask].tolist(), cf["fold"][vmask].tolist()))
        seq_fold = np.array([fold_map[int(g)] for g in rg])
        fit_rows = (seq_fold < 2)[seq]
    else:
        dev_a, _ = dev_halves(ROOT / "datasets" / "valid.parquet")
        fit_rows = np.isin(rg, np.asarray(dev_a))[seq]
    sample = F[np.flatnonzero(fit_rows)[::7]]
    lo, hi = np.percentile(sample, 0.5, axis=0), np.percentile(sample, 99.5, axis=0)
    np.clip(F, lo, hi, out=F)
    mu, sd = sample.mean(0), sample.std(0)
    del sample
    F -= mu.astype(np.float32)
    F /= np.where(sd > 1e-9, sd, 1.0).astype(np.float32)
    q = p.copy()
    c_seq = cls[seq]
    for c in range(4):
        rows = c_seq == c
        for k in range(2):
            yk, pk = np.clip(y[:, k], -2, 2), np.clip(p[:, k], -2, 2)
            if a.gbm:
                from sklearn.ensemble import HistGradientBoostingRegressor

                fr = fit_rows[rows]
                Dg = np.concatenate([F[rows], pk[rows, None], p[rows]], 1)
                idx = np.flatnonzero(fr)
                idx = np.random.default_rng(c * 2 + k).choice(idx, min(len(idx), 250_000), replace=False)
                gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                                    l2_regularization=1.0)
                gbm.fit(Dg[idx], (yk[rows] - pk[rows])[idx], sample_weight=np.abs(yk[rows])[idx])
                q[rows, k] = p[rows, k] + gbm.predict(Dg)
                continue
            D = F[rows] if a.no_interactions else np.concatenate([F[rows], F[rows] * pk[rows, None]], 1)
            D = np.concatenate([np.ones((rows.sum(), 1)), D], 1)
            w = np.abs(yk[rows])
            fr = fit_rows[rows]
            A = D[fr] * np.sqrt(w[fr])[:, None]
            b = (yk[rows] - pk[rows])[fr] * np.sqrt(w[fr])
            reg = a.ridge * np.eye(D.shape[1])
            reg[0, 0] = 0.0
            beta = np.linalg.solve(A.T @ A + reg * w[fr].sum() / 1e4, A.T @ b)
            q[rows, k] = p[rows, k] + D @ beta
    ev = ~fit_rows
    n_seq = len(cls)
    s0 = seq_sums(y[ev], p[ev], seq[ev], n_seq)
    s1 = seq_sums(y[ev], q[ev], seq[ev], n_seq)
    keep = np.flatnonzero(s0[:, 0, 0] > 0)
    base, new = wp_of(s0[keep]), wp_of(s1[keep])
    rng = np.random.default_rng(0)
    boots = [wp_of(s1[b]) - wp_of(s0[b]) for b in (rng.choice(keep, len(keep)) for _ in range(200))]
    by_class = {}
    for c in range(4):
        kc = keep[cls[keep] == c]
        by_class[c] = round(wp_of(s1[kc]) - wp_of(s0[kc]), 4)
    out = {"probe": label, "model": a.model, "n_features": int(F.shape[1]), "eval_wp_model": round(base, 4),
           "eval_wp_corrected": round(new, 4), "delta": round(new - base, 4), "delta_sd": round(float(np.std(boots)), 4),
           "delta_by_class": by_class, "ridge": a.ridge, "interactions": not a.no_interactions, "grouping": a.grouping}
    print(json.dumps(out))
    with open(ROOT / "research" / "artifacts" / "lift_probe.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
