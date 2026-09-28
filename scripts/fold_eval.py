"""Paired out-of-fold comparison of two runs with the exact scorer on chain-grouped folds.

    .venv/bin/python scripts/fold_eval.py CANDIDATE REFERENCE --folds 0,1 [--boot 1000] [--all]

A run name R means predictions cache/pred_R_f{k}_valid.npy (written by predict_run.py for the fold
model R_f{k}); only the held-out fold's sequences are scored. By default only *pure* sequences are
used (cache/chain_folds.npz: validation chains without a train member, so no time-adjacent training
neighbour). The score is the competition's pooled |clip(y)|-weighted Pearson over need_prediction AND
is_scored rows, predictions clipped to +-2, mean of t0/t1. The paired delta's uncertainty is a chain
bootstrap (whole chains resampled within each fold). Also reports per-class deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLIP = 2.0
WARMUP = 99


def sums(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-sequence sufficient statistics (B, 2 targets, 6): W, Swy, Swp, Swyy, Swpp, Swyp."""
    y = np.clip(y.astype(np.float64), -CLIP, CLIP)
    p = np.clip(p.astype(np.float64), -CLIP, CLIP)
    w = np.abs(y) * m[..., None]
    return np.stack([w.sum(1), (w * y).sum(1), (w * p).sum(1), (w * y * y).sum(1),
                     (w * p * p).sum(1), (w * y * p).sum(1)], axis=-1)


def pooled(s: np.ndarray) -> float:
    """Pooled WP from summed statistics (2, 6)."""
    W, sy, sp, syy, spp, syp = s.T
    my, mp = sy / W, sp / W
    cov = syp / W - my * mp
    vy, vp = syy / W - my * my, spp / W - mp * mp
    return float(np.mean(cov / np.sqrt(vy * vp)))


def load_fold(run: str, k: int, rows: np.ndarray) -> np.ndarray:
    return np.load(ROOT / "cache" / f"pred_{run}_f{k}_valid.npy", mmap_mode="r")[rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--all", action="store_true", help="score every fold sequence, not only pure chains")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tg = np.load(ROOT / "cache" / "valid_targets.npz")
    rg_order = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    v = cf["split"] == "valid"
    info = {int(g): (int(f), int(c), bool(pu)) for g, f, c, pu in
            zip(cf["row_group"][v], cf["fold"][v], cf["chain"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]

    stats_c, stats_r, chains, cls, fold_id = [], [], [], [], []
    for k in [int(f) for f in args.folds.split(",")]:
        rows = np.array([i for i, g in enumerate(rg_order)
                         if info[int(g)][0] == k and (args.all or info[int(g)][2])])
        y = tg["y"][rows]
        m = tg["mask"][rows].copy()
        m[:, :WARMUP] = False
        stats_c.append(sums(y, load_fold(args.candidate, k, rows), m))
        stats_r.append(sums(y, load_fold(args.reference, k, rows), m))
        chains += [info[int(rg_order[i])][1] for i in rows]
        cls += [int(classes[rg_order[i]]) for i in rows]
        fold_id += [k] * len(rows)
    sc, sr = np.concatenate(stats_c), np.concatenate(stats_r)
    chains, cls, fold_id = np.array(chains), np.array(cls), np.array(fold_id)

    def delta(idx: np.ndarray) -> float:
        return pooled(sc[idx].sum(0)) - pooled(sr[idx].sum(0))

    everything = np.arange(len(sc))
    res = {"candidate": args.candidate, "reference": args.reference, "sequences": int(len(sc)),
           "pure_only": not args.all,
           "candidate_wp": pooled(sc.sum(0)), "reference_wp": pooled(sr.sum(0)), "delta": delta(everything),
           "delta_by_fold": {int(k): delta(np.flatnonzero(fold_id == k)) for k in np.unique(fold_id)},
           "delta_by_class": {int(c): delta(np.flatnonzero(cls == c)) for c in np.unique(cls)}}
    rng = np.random.default_rng(0)
    groups = {}
    for i, (k, ch) in enumerate(zip(fold_id, chains)):
        groups.setdefault((k, ch), []).append(i)
    keys = list(groups)
    by_fold = {k: [key for key in keys if key[0] == k] for k in np.unique(fold_id)}
    boots = []
    for _ in range(args.boot):
        pick = []
        for k, ks in by_fold.items():
            for j in rng.integers(0, len(ks), len(ks)):
                pick += groups[ks[j]]
        boots.append(delta(np.asarray(pick)))
    res["delta_boot_sd"] = float(np.std(boots))
    res["p_delta_le_0"] = float(np.mean(np.asarray(boots) <= 0))
    print(json.dumps(res, indent=1))
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
