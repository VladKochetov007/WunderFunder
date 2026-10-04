"""Price-impact-split trade-flow features for class 3 (opus-micro H1), train-fitted tables.

    scripts/limited .venv/bin/python scripts/micro_features.py
1. Side table on train class-3 sequences: for each i0 trade-price code (dp0..dp3 rounded to 1e-3),
   the mean sign of the next decoded mid move within 50 rows after trades with that code
   (mid moves decoded with the train-fitted class-3 code grammar, cache/class3_moves_full.pkl).
2. Causal features on every class-3 validation row: signed trade flow split by whether the trade
   moved the mid (at its row or the next row): EMAs at half-lives 4/16/64 for non-price-changing
   (MO0) and price-changing (MO1) trades, their contrast, and the MO0 flow since the last mid move.
   A trade at row t is classified at row t+1 (causal).
3. Writes cache/micro_class3_scored.npy aligned to the scored rows of cache/heur_valid.npz
   (zeros outside class 3) for scripts/lift_probe.py --npy.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile
from wnn.ticks import _keys

ROOT = Path(__file__).resolve().parents[1]
HL = (4, 16, 64)
NAMES = [f"mo{k}_hl{h}" for k in (0, 1) for h in HL] + ["contrast_hl16", "mo0_since_move"]
_MOVES = None
_SIDE = None


def _moves():
    global _MOVES
    if _MOVES is None:
        _MOVES = pickle.load(open(ROOT / "cache" / "class3_moves_full.pkl", "rb"))[0]
    return _MOVES


def decode_mid(x):
    px = x[:, 0:22]
    ev = np.flatnonzero((px[1:] != px[:-1]).any(1)) + 1
    dm = np.zeros(len(x))
    if len(ev):
        k = _keys(x[np.r_[ev - 1, ev]])
        n = len(ev)
        mv = _moves()
        for i, e in enumerate(ev):
            v = mv.get((k[i], k[n + i]))
            if v is not None:
                dm[e] = (v[0] + v[1]) / 2.0
    return dm


def trade_rows(x):
    rest = x[:, 48:52]
    return np.flatnonzero((np.abs(rest - np.median(rest, 0)) > 1e-3).any(1))


def _side_acc(g):
    x = SequenceFile(ROOT / "datasets" / "train.parquet").read(g, targets=False, mask=False).features
    dm = decode_mid(x)
    nz = np.flatnonzero(dm != 0)
    acc = defaultdict(lambda: [0.0, 0])
    for t in trade_rows(x):
        k = np.searchsorted(nz, t)
        if k < len(nz) and nz[k] - t <= 50:
            key = tuple(np.round(x[t, 44:48], 3))
            acc[key][0] += float(np.sign(dm[nz[k]]))
            acc[key][1] += 1
    return dict(acc)


def features(x, side):
    T = len(x)
    dm = decode_mid(x)
    is_tr = np.zeros(T, bool)
    is_tr[trade_rows(x)] = True
    al = {h: 1 - 0.5 ** (1 / h) for h in HL}
    e = {(k, h): 0.0 for k in (0, 1) for h in HL}
    out = np.zeros((T, len(NAMES)), np.float32)
    pending = None
    since0 = 0.0
    for t in range(T):
        if dm[t] != 0:
            since0 = 0.0
        if pending is not None:
            s, tt = pending
            k = 1 if (dm[t] != 0 or dm[tt] != 0) else 0
            for h in HL:
                e[(k, h)] += al[h] * (s - e[(k, h)])
            if k == 0:
                since0 += s
            pending = None
        if is_tr[t]:
            pending = (side.get(tuple(np.round(x[t, 44:48], 3)), 0.0), t)
        c = 0
        for k in (0, 1):
            for h in HL:
                out[t, c] = e[(k, h)]
                c += 1
        out[t, c] = e[(0, 16)] - e[(1, 16)]
        out[t, c + 1] = since0
    return out


def _init(side):
    global _SIDE
    _SIDE = side


def _valid_feats(args):
    i, g = args
    x = SequenceFile(ROOT / "datasets" / "valid.parquet").read(g, targets=False, mask=False).features
    return i, features(x, _SIDE)


def main():
    workers = int(os.environ.get("WNN_THREADS", "6"))
    m3 = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["train"]
    tg = [int(g) for g in np.flatnonzero(m3 == 3) if g != 528][:300]
    acc = defaultdict(lambda: [0.0, 0])
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_side_acc, tg, chunksize=4):
            for k, (s, n) in part.items():
                acc[k][0] += s
                acc[k][1] += n
    side = {k: s / n for k, (s, n) in acc.items() if n >= 5}
    print("side codes", len(side), "strong", sum(abs(v) > 0.5 for v in side.values()), flush=True)
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    cls, rg = d["market_class"], d["row_group"]
    h = np.load(ROOT / "cache" / "heur_valid.npz")
    sc = h["scored"]
    seq, step = h["seq"][sc], h["step"][sc]
    out = np.zeros((len(seq), len(NAMES)), np.float32)
    jobs = [(int(i), int(rg[i])) for i in np.flatnonzero(cls == 3)]
    order = np.argsort(seq, kind="stable")
    starts = np.searchsorted(seq[order], np.arange(len(cls)))
    ends = np.searchsorted(seq[order], np.arange(len(cls)), side="right")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(side,)) as ex:
        for i, f in ex.map(_valid_feats, jobs, chunksize=4):
            rows = order[starts[i]:ends[i]]
            out[rows] = f[step[rows]]
    np.save(ROOT / "cache" / "micro_class3_scored.npy", out)
    print("saved", out.shape, NAMES)


if __name__ == "__main__":
    main()
