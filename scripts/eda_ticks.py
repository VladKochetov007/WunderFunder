"""EDA: does the decoded class-3 PRICE LEVEL (in ticks) carry information the GRU misses?

    .venv/bin/python -u scripts/eda_ticks.py [--fit-seqs 1500] [--members ft256dH+ft256ds4H]

Class-3 price codes describe the book SHAPE (~8 states), not the price level, so earlier long-memory probes
never saw the level. wnn.ticks learns the code-transition -> tick-move table from labelled train sequences and
reconstructs ask/bid paths in ticks causally. Features on the decoded mid (ticks): expanding z-score,
distance to rolling max/min over 500/2000/8000 rows (support/resistance), change over 100/500/2000 rows,
fraction of mid moves that reversed the previous move over the last 64 moves. Target: residual of the future
move D on scored rows of class-3 pure validation sequences (folds 0-1). Reports weighted correlations and a
cross-fitted (fold 0 <-> fold 1) ridge.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from fold_eval import ROOT, WARMUP

from wnn.data import SequenceFile
from wnn.ticks import learn_moves, learn_spreads, reconstruct

CLIP = 2.0


def roll(x: np.ndarray, w: int, fn) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.concatenate([np.full(w - 1, x[0]), x])
    return fn(sliding_window_view(pad, w), axis=1)


def features(mid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = np.arange(1, len(mid) + 1)
    mu = np.cumsum(mid) / n
    var = np.maximum(np.cumsum(mid * mid) / n - mu * mu, 0)
    cols, names = [(mid - mu) / np.sqrt(var + 1.0)], ["mid_z"]
    for w in (500, 2000, 8000):
        cols += [mid - roll(mid, w, np.max), mid - roll(mid, w, np.min)]
        names += [f"from_max{w}", f"from_min{w}"]
    for L in (100, 500, 2000):
        d = np.zeros_like(mid); d[L:] = mid[L:] - mid[:-L]
        cols.append(d); names.append(f"chg{L}")
    mv = np.diff(mid, prepend=mid[0]); idx = np.flatnonzero(mv != 0)
    rev = np.zeros(len(mid))
    if len(idx) > 2:
        s = np.sign(mv[idx]); r = (s[1:] != s[:-1]).astype(float)
        cr = np.concatenate([[0.0], np.cumsum(r)])
        k = np.arange(len(idx)); lo = np.maximum(0, k - 64)
        frac = np.where(k > 0, (cr[k] - cr[lo]) / np.maximum(k - lo, 1), 0.5)  # reversals among moves (lo, k]
        ev_id = np.searchsorted(idx, np.arange(len(mid)), side="right") - 1
        rev = np.where(ev_id >= 0, frac[np.maximum(ev_id, 0)], 0.5)
    cols.append(rev); names.append("reversal_frac64")
    return np.stack(cols, -1), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-seqs", type=int, default=1500)
    ap.add_argument("--members", default="ft256dH+ft256ds4H")
    a = ap.parse_args()
    cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")
    tables = ROOT / "cache" / "class3_moves_full.pkl"
    if not tables.exists():
        tr = SequenceFile(ROOT / "datasets" / "train.parquet")
        pick = np.random.default_rng(0).choice(np.flatnonzero(cls["train"] == 3), a.fit_seqs, replace=False)
        seqs = [(s.features, s.targets) for s in (tr.read(int(g), mask=False) for g in pick)]
        moves, spreads = learn_moves(seqs), learn_spreads(seqs)
        tables.write_bytes(pickle.dumps((moves, spreads)))
        print(f"tables: {len(moves)} transitions, {len(spreads)} spread codes", flush=True)
    moves, spreads = pickle.loads(tables.read_bytes())

    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    va = SequenceFile(ROOT / "datasets" / "valid.parquet")
    data = {}
    unknown_total = events_total = 0
    names = None
    for k in (0, 1):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True) and cls["valid"][g] == 3])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows] for r in a.members.split("+")]
        P = sum(np.asarray(p, dtype=np.float64) for p in preds) / len(preds)
        X, R, W = [], [], []
        for j, i in enumerate(rows):
            m = tg["mask"][i].copy(); m[:WARMUP] = False
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            x = va.read(int(rg[i]), targets=False, mask=False).features
            ask, bid, unk = reconstruct(x, moves, spreads)
            unknown_total += unk; events_total += int(((x[1:, 0:22] != x[:-1, 0:22]).any(1)).sum())
            f, names = features((ask + bid).astype(np.float64) / 2)
            y = np.clip(tg["y"][i].astype(np.float64), -CLIP, CLIP)
            p = np.clip(P[j], -CLIP, CLIP)
            R.append((y[idx, 0] - y[idx, 1]) / 2 - (p[idx, 0] - p[idx, 1]) / 2)
            X.append(f[idx]); W.append(np.abs(y[idx]).sum(1))
        data[k] = tuple(map(np.concatenate, (X, R, W)))
        print(f"fold {k}: {len(rows)} class-3 sequences, {len(data[k][1])} scored rows", flush=True)
    print(f"unknown transitions {unknown_total} of {events_total} book events", flush=True)

    def wcorr(a_, b_, w_):
        ma, mb = (w_ * a_).sum() / w_.sum(), (w_ * b_).sum() / w_.sum()
        return (w_ * (a_ - ma) * (b_ - mb)).sum() / np.sqrt((w_ * (a_ - ma) ** 2).sum() * (w_ * (b_ - mb) ** 2).sum())

    for j, nm in enumerate(names):
        print(f"  {nm:16s} " + " ".join(f"f{k} {wcorr(data[k][0][:, j], data[k][1], data[k][2]):+.3f}" for k in (0, 1)),
              flush=True)
    for k in (0, 1):
        Xtr, Rtr, Wtr = data[1 - k]; Xte, Rte, Wte = data[k]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = (Xtr - mu) / sd
        beta = np.linalg.solve(A.T @ (A * Wtr[:, None]) + 10 * np.eye(A.shape[1]), A.T @ (Wtr * Rtr))
        pred = ((Xte - mu) / sd) @ beta
        print(f"cross-fitted ridge on fold {k}: explains {1 - (Wte * (Rte - pred) ** 2).sum() / (Wte * Rte ** 2).sum():+.4f} "
              f"of weighted residual MSE; corr {wcorr(pred, Rte, Wte):+.3f}", flush=True)


if __name__ == "__main__":
    main()
