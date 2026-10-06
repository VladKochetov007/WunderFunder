"""Class-routed combination of two runs vs a reference, exact pooled WP on pure fold chains.

    .venv/bin/python scripts/route_eval.py ROUTED BASE REFERENCE --classes 3 --folds 0,1

Sequences of the listed market classes take ROUTED predictions, the rest take BASE. The market class is
known causally by row 99 in deployment (class head, E-258 notes), so routing by the labelled class is a
faithful analog. Reports the paired delta vs REFERENCE with a chain bootstrap, per fold.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from fold_eval import ROOT, WARMUP, load_fold, pooled, sums


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("routed"); ap.add_argument("base"); ap.add_argument("reference")
    ap.add_argument("--classes", default="3"); ap.add_argument("--folds", default="0,1")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    route = {int(c) for c in a.classes.split(",")}
    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), int(c), bool(p)) for g, f, c, p in zip(cf["row_group"][v], cf["fold"][v], cf["chain"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    sc, sr, ch, fo = [], [], [], []
    for k in [int(f) for f in a.folds.split(",")]:
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)][0] == k and info[int(g)][2]])
        y = tg["y"][rows]; m = tg["mask"][rows].copy(); m[:, :WARMUP] = False
        use = np.isin(classes[rg[rows]], list(route))
        pr = load_fold(a.routed, k, rows)
        pb = load_fold(a.base, k, rows)
        p = np.where(use[:, None, None], pr, pb)
        sc.append(sums(y, p, m)); sr.append(sums(y, load_fold(a.reference, k, rows), m))
        ch += [info[int(rg[i])][1] for i in rows]; fo += [k] * len(rows)
    sc, sr, ch, fo = np.concatenate(sc), np.concatenate(sr), np.array(ch), np.array(fo)
    d = lambda idx: pooled(sc[idx].sum(0)) - pooled(sr[idx].sum(0))  # noqa: E731
    groups = {}
    for i, key in enumerate(zip(fo, ch)):
        groups.setdefault(key, []).append(i)
    keys = list(groups); rng = np.random.default_rng(0); boots = []
    for _ in range(a.boot):
        pick = []
        for k in np.unique(fo):
            ks = [key for key in keys if key[0] == k]
            for j in rng.integers(0, len(ks), len(ks)):
                pick += groups[ks[j]]
        boots.append(d(np.asarray(pick)))
    print(json.dumps({"routed": a.routed, "classes": sorted(route), "base": a.base, "reference": a.reference,
                      "delta": d(np.arange(len(sc))), "by_fold": {int(k): d(np.flatnonzero(fo == k)) for k in np.unique(fo)},
                      "boot_sd": float(np.std(boots)), "p_le_0": float(np.mean(np.asarray(boots) <= 0))}))


if __name__ == "__main__":
    main()
