"""Paired deltas on train-like vs train-unlike validation sequences (4 chain folds, pure chains).

    .venv/bin/python -u scripts/shift_slices.py CAND REF [CAND REF ...]

Train-unlikeness is the adversarial train-vs-valid score (cache/adv_valid_score.npy, within-class
classifier on label-free sequence statistics). Within each market class, sequences are split into
terciles of that score. If the public test is further from train than validation (as the fine-tune and
width uploads suggest: public gains > fold gains), the train-unlike tercile is the better local proxy.
"""

from __future__ import annotations

import sys

import numpy as np
from fold_eval import ROOT, WARMUP, load_fold, pooled, sums


def main() -> None:
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    tg = np.load(ROOT / "cache" / "valid_targets.npz")
    rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    adv = np.load(ROOT / "cache" / "adv_valid_score.npy")  # indexed like valid_targets row order
    tercile = np.full(len(rg), -1)
    for c in range(4):
        sel = np.flatnonzero(classes[rg] == c)
        q = np.quantile(adv[sel], [1 / 3, 2 / 3])
        tercile[sel] = np.digitize(adv[sel], q)  # 0 train-like ... 2 train-unlike
    for cand, ref in pairs:
        S_c, S_r, T = [], [], []
        for k in range(4):
            rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
            y = tg["y"][rows]; m = tg["mask"][rows].copy(); m[:, :WARMUP] = False
            S_c.append(sums(y, load_fold(cand, k, rows), m)); S_r.append(sums(y, load_fold(ref, k, rows), m))
            T.append(tercile[rows])
        S_c, S_r, T = map(np.concatenate, (S_c, S_r, T))
        out = [f"{cand} vs {ref}: all {pooled(S_c.sum(0)) - pooled(S_r.sum(0)):+.5f}"]
        for t, name in ((0, "train-like"), (1, "middle"), (2, "train-unlike")):
            s = T == t
            out.append(f"{name} {pooled(S_c[s].sum(0)) - pooled(S_r[s].sum(0)):+.5f}")
        s = T >= 1
        out.append(f"unlike 2/3 {pooled(S_c[s].sum(0)) - pooled(S_r[s].sum(0)):+.5f}")
        print(" | ".join(out), flush=True)


if __name__ == "__main__":
    main()
