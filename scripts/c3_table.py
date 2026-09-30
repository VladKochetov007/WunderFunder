"""Class-3 context table: how far can a discrete lookup E_w[D | context] go, and does it add to the GRU?

    .venv/bin/python -u scripts/c3_table.py [--train-seqs 1500] [--members ft256dH+ft256ds4H]

Context per row (all causal): the full i0 price-code tuple (a global book-state code), log2 bucket of rows
since the last i0 price event, the sign of the last price move (from the bid code change), and whether the
current row is a trade row. The table is fitted on class-3 TRAIN sequences (|y|-weighted mean of the future
move D = (t0 - t1)/2, shrunk toward the parent context with a pseudo-count), then evaluated on class-3 pure
validation sequences of folds 0-1: within-class weighted correlation of the table alone, of the GRU ensemble
(OOF), and of GRU + beta * table for a few beta (the best beta is chosen on fold 0 and applied to fold 1).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from fold_eval import ROOT, WARMUP

from wnn.data import SequenceFile

CLIP = 2.0


def contexts(x: np.ndarray) -> list[tuple]:
    codes = [tuple(r) for r in np.round(x[:, 0:22], 4)]
    ev = np.r_[False, (x[1:, 0] != x[:-1, 0]) & (x[1:, 11] != x[:-1, 11])]
    move = np.r_[0.0, np.sign(x[1:, 0] - x[:-1, 0])]
    trade = np.r_[False, (x[1:, 48:52] != x[:-1, 48:52]).any(1)]
    out, last, sgn = [], -10**9, 0.0
    for t in range(len(x)):
        if ev[t]:
            last, sgn = t, move[t]
        since = t - last
        b = min(int(np.log2(since + 1)), 12)
        out.append((codes[t], b, sgn, bool(trade[t])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seqs", type=int, default=1500)
    ap.add_argument("--members", default="ft256dH+ft256ds4H")
    ap.add_argument("--prior", type=float, default=50.0)
    a = ap.parse_args()
    cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")
    tr_f = SequenceFile(ROOT / "datasets" / "train.parquet")
    c3 = np.flatnonzero(cls["train"] == 3)
    pick = np.random.default_rng(0).choice(c3, min(a.train_seqs, len(c3)), replace=False)
    full, coarse = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    for n, g in enumerate(pick):
        s = tr_f.read(int(g), mask=False)
        y = np.clip(s.targets.astype(np.float64), -CLIP, CLIP)
        d = (y[:, 0] - y[:, 1]) / 2
        w = np.abs(y).sum(1)
        for t, c in enumerate(contexts(s.features)):
            if t < WARMUP:
                continue
            full[c][0] += w[t] * d[t]; full[c][1] += w[t]
            k = c[1:]
            coarse[k][0] += w[t] * d[t]; coarse[k][1] += w[t]
        if n % 300 == 0:
            print(f"train {n}/{len(pick)} contexts {len(full)}", flush=True)

    def lookup(c):
        cs, cw = coarse.get(c[1:], (0.0, 0.0))
        prior = cs / cw if cw > 0 else 0.0
        fs, fw = full.get(c, (0.0, 0.0))
        return (fs + a.prior * prior) / (fw + a.prior)

    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    va_f = SequenceFile(ROOT / "datasets" / "valid.parquet")
    res = {}
    for k in (0, 1):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True) and cls["valid"][g] == 3])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows] for r in a.members.split("+")]
        P = sum(np.asarray(p, dtype=np.float64) for p in preds) / len(preds)
        D, G, T, W = [], [], [], []
        for j, i in enumerate(rows):
            m = tg["mask"][i].copy(); m[:WARMUP] = False
            if not m.any():
                continue
            y = np.clip(tg["y"][i].astype(np.float64), -CLIP, CLIP)
            ctx = contexts(va_f.read(int(rg[i]), targets=False, mask=False).features)
            idx = np.flatnonzero(m)
            D.append((y[idx, 0] - y[idx, 1]) / 2); W.append(np.abs(y[idx]).sum(1))
            G.append((np.clip(P[j, idx, 0], -CLIP, CLIP) - np.clip(P[j, idx, 1], -CLIP, CLIP)) / 2)
            T.append(np.array([lookup(ctx[t]) for t in idx]))
        res[k] = tuple(map(np.concatenate, (D, G, T, W)))
        print(f"fold {k} loaded ({len(rows)} class-3 sequences)", flush=True)

    def wcorr(a_, b_, w_):
        ma, mb = (w_ * a_).sum() / w_.sum(), (w_ * b_).sum() / w_.sum()
        return (w_ * (a_ - ma) * (b_ - mb)).sum() / np.sqrt((w_ * (a_ - ma) ** 2).sum() * (w_ * (b_ - mb) ** 2).sum())

    betas = [0.0, 0.1, 0.25, 0.5, 1.0]
    for k in (0, 1):
        D, G, T, W = res[k]
        print(f"fold {k}: corr(D, table) {wcorr(D, T, W):.4f} | corr(D, GRU) {wcorr(D, G, W):.4f} | corr(GRU, table) "
              f"{wcorr(G, T, W):.4f} | GRU+b*table: " + " ".join(f"b{b} {wcorr(D, G + b * T, W):.4f}" for b in betas),
              flush=True)


if __name__ == "__main__":
    main()
