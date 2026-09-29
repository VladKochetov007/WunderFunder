"""Seed-ensemble curve: pooled WP of the average of K same-recipe members, K = 1..N, on pure fold chains.

    .venv/bin/python scripts/seed_curve.py ref refs4 refs5 refs6 [...] --folds 0,1

For each K the score is averaged over every K-subset of the given runs (so the curve does not depend on
which seeds come first), with the min/max over subsets. Exact competition scorer on pure held-out chains
(scripts/fold_eval.py). Tells how much each extra member is worth, to set the ensemble size that the
inference budget should buy.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tg = np.load(ROOT / "cache" / "valid_targets.npz")
    rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    folds = [int(f) for f in a.folds.split(",")]
    data = {}
    for k in folds:
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        m = tg["mask"][rows].copy()
        m[:, :WARMUP] = False
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows].astype(np.float32)
                 for r in a.runs]
        data[k] = (tg["y"][rows], m, preds)
    out = {}
    for size in range(1, len(a.runs) + 1):
        scores = []
        for combo in itertools.combinations(range(len(a.runs)), size):
            total = 0
            for k in folds:
                y, m, preds = data[k]
                p = sum(preds[i] for i in combo) / size
                total = total + sums(y, p, m).sum(0)
            scores.append(pooled(total))
        out[size] = {"mean": float(np.mean(scores)), "min": float(np.min(scores)), "max": float(np.max(scores)),
                     "subsets": len(scores)}
        gain = "" if size == 1 else f"  gain vs K-1 {out[size]['mean'] - out[size - 1]['mean']:+.5f}"
        print(f"K={size}: mean {out[size]['mean']:.5f} (min {out[size]['min']:.5f}, max {out[size]['max']:.5f}, "
              f"{len(scores)} subsets){gain}", flush=True)
    if a.out:
        (ROOT / a.out).write_text(json.dumps({"runs": a.runs, "folds": folds, "curve": out}, indent=1))


if __name__ == "__main__":
    main()
