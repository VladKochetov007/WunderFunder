"""Symbolic-regression explanation tasks on the heuristic dataset (cache/heur_valid.npz).

    scripts/limited .venv/bin/python scripts/symreg_explain.py TASK [--cls K] [--terms 6]
Tasks: direction (D = t0 - t1), residual (targets minus a cached model's predictions),
mask (is_scored), level (S = t0 + t1), target0/target1 (raw targets).
Fit on chain folds 0-1 of validation, validate on chain folds 2-3 (no stream neighbours across). Base features are the
heuristic library plus the 112 raw columns. Writes research/artifacts/symreg/<task>[_cK].json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn import symreg
from wnn.data import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--cls", type=int, default=-1)
    ap.add_argument("--terms", type=int, default=6)
    ap.add_argument("--top-m", type=int, default=50)
    ap.add_argument("--fit-rows", type=int, default=150_000)
    ap.add_argument("--model", default="ab20_gru128_e2")
    a = ap.parse_args()
    h = np.load(ROOT / "cache" / "heur_valid.npz")
    seq, step, scored = h["seq"], h["step"], h["scored"]
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    cls, rg = d["market_class"], d["row_group"]
    # chain-grouped halves (cache/chain_folds.npz): fit on chain folds 0-1, validate on 2-3, so no
    # validation sequence has a stream neighbour in the fit half
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    vmask = cf["split"] == "valid"
    fold_map = dict(zip(cf["row_group"][vmask].tolist(), cf["fold"][vmask].tolist()))
    in_a = (np.array([fold_map[int(g)] for g in rg]) < 2)[seq]
    y2 = d["y"][seq, step].astype(np.float64)
    yc = np.clip(y2, -2, 2)
    sel = np.ones(len(seq), bool) if a.cls < 0 else (cls[seq] == a.cls)
    if a.task == "mask":
        target = scored.astype(np.float64)
        weight = np.where(scored, 1.0, 1.0 / 0.03)
    else:
        sel &= scored
        if a.task == "direction":
            target, weight = yc[:, 0] - yc[:, 1], 0.5 * (np.abs(yc[:, 0]) + np.abs(yc[:, 1]))
        elif a.task == "level":
            target, weight = yc[:, 0] + yc[:, 1], 0.5 * (np.abs(yc[:, 0]) + np.abs(yc[:, 1]))
        elif a.task in ("target0", "target1"):
            k = int(a.task[-1])
            target, weight = yc[:, k], np.abs(yc[:, k])
        elif a.task == "residual":
            p = np.load(ROOT / "cache" / f"pred_{a.model}_valid.npy", mmap_mode="r")
            pk = np.clip(np.asarray(p[seq, step], dtype=np.float64), -2, 2)
            target = (yc[:, 0] - pk[:, 0]) - (yc[:, 1] - pk[:, 1])  # residual of the direction component
            weight = 0.5 * (np.abs(yc[:, 0]) + np.abs(yc[:, 1]))
        else:
            raise SystemExit(f"unknown task {a.task}")
    names = [f"h:{n}" for n in h["names"]] + [f"r:{c}" for c in FEATURE_COLUMNS]
    fit_idx = np.flatnonzero(sel & in_a)
    val_idx = np.flatnonzero(sel & ~in_a)
    rng = np.random.default_rng(0)
    fit_idx = np.sort(rng.choice(fit_idx, min(a.fit_rows, len(fit_idx)), replace=False))
    val_idx = np.sort(rng.choice(val_idx, min(2 * a.fit_rows, len(val_idx)), replace=False))
    X = np.concatenate([h["X"][fit_idx], h["raw"][fit_idx]], 1)
    Xv = np.concatenate([h["X"][val_idx], h["raw"][val_idx]], 1)
    print(f"task {a.task} cls {a.cls}: fit rows {len(fit_idx)} val rows {len(val_idx)} features {X.shape[1]}", flush=True)
    r = symreg.fit(X, names, target[fit_idx], weight[fit_idx], Xv, target[val_idx], weight[val_idx], n_terms=a.terms, top_m=a.top_m)
    out = {"task": a.task, "cls": a.cls, "fit_rows": int(len(fit_idx)), "val_rows": int(len(val_idx)),
           "terms": [t.expr for t in r.terms], "screen_corr": [t.score for t in r.terms],
           "cum_wp_train": r.wp_train, "cum_wp_val": r.wp_val, "coef": r.coef.tolist()}
    name = a.task + (f"_c{a.cls}" if a.cls >= 0 else "") + (f"_{a.model}" if a.task == "residual" else "")
    (ROOT / "research" / "artifacts" / "symreg" / f"{name}.json").write_text(json.dumps(out, indent=1))
    import pickle

    pickle.dump(r, open(ROOT / "cache" / f"symreg_{name}.pkl", "wb"))


if __name__ == "__main__":
    main()
