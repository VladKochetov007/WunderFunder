"""Surprise miner: causal long-horizon mid drift (ticks) from the TRAIN move table, per scored row.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_drift.py OUT.npz

Variant of cr_surprise_fspace.py: instead of smoothing, writes mid displacement over trailing windows
(rows) and since the last unknown transition, plus quote-move counts, to test whether a sequence-scale
trend that a GRU may forget explains the out-of-fold residual.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
PX0 = [f"i0_p{i}" for i in range(22)]
WINDOWS = (20, 100, 500, 2000, 8000)


def main() -> None:
    out_path = Path(sys.argv[1])
    with np.load(ROOT / "cache/valid_targets.npz") as z:
        mask_all, cls_all = z["mask"], z["market_class"]
    fold_of = np.full(len(cls_all), -1)
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        fold_of[man["evaluation"]["valid"]] = k
    preds = {(s, k): np.load(ROOT / f"cache/pred_c20260929_ft_since_s{s}_f{k}_valid.npy", mmap_mode="r")
             for s in (3, 4) for k in range(4)}
    tab = np.load(ROOT / "research/campaign_2026-10-01/data_eval/target_law_tables.npz")
    mult = tab["mult"]
    mk = tab["move_prev"].astype(np.uint64) * np.uint64(1000003) ^ tab["move_next"].astype(np.uint64)
    order = np.argsort(mk)
    mk, md = mk[order], tab["move_d"][order]
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    acc: dict[str, list] = {}
    for g in range(len(cls_all)):
        t = pf.read_row_group(g, columns=PX0, use_threads=False)
        px = np.column_stack([t.column(c).to_numpy() for c in PX0])
        q = np.rint(px.astype(np.float64) * 1e4).astype(np.int64)
        with np.errstate(over="ignore"):
            h = (q * mult[None]).sum(1)
            h ^= h >> np.int64(29)
        n = len(h)
        ch = np.flatnonzero(h[1:] != h[:-1]) + 1
        key = h[ch - 1].astype(np.uint64) * np.uint64(1000003) ^ h[ch].astype(np.uint64)
        pos = np.minimum(np.searchsorted(mk, key), len(mk) - 1)
        hit = mk[pos] == key
        da = np.zeros(n)
        db = np.zeros(n)
        reset = np.zeros(n, bool)
        reset[0] = True
        da[ch[hit]] = md[pos[hit], 0]
        db[ch[hit]] = md[pos[hit], 1]
        reset[ch[~hit]] = True
        A, B = np.cumsum(da), np.cumsum(db)
        k = fold_of[g]
        cls = int(cls_all[g])
        qu = QUANTA[cls]
        p = 0.5 * (preds[(3, k)][g].astype(np.float64) + preds[(4, k)][g]) / qu
        F0, F1 = p[:, 0] + A, B - p[:, 1]
        sel = np.flatnonzero(mask_all[g])
        d = {"seq": np.full(len(sel), g, np.int16)}
        since_reset = np.arange(n) - np.maximum.accumulate(np.where(reset, np.arange(n), 0))
        d["since_reset"] = since_reset[sel].astype(np.int32)
        mid = 0.5 * (A + B)
        seg = np.cumsum(reset)
        nmove = np.cumsum((da != 0) | (db != 0))
        idx = np.arange(n)
        seg_start = np.maximum.accumulate(np.where(reset, idx, 0))
        for w in WINDOWS:
            j = np.maximum(idx - w, seg_start)
            d[f"drift_w{w}"] = (mid - mid[j])[sel].astype(np.float32)
            d[f"moves_w{w}"] = (nmove - nmove[j])[sel].astype(np.float32)
        d["drift_seg"] = (mid - mid[seg_start])[sel].astype(np.float32)
        d["spread_path"] = (A - B)[sel].astype(np.float32)
        for kk, v in d.items():
            acc.setdefault(kk, []).append(v)
        if g % 300 == 0:
            print(g, flush=True)
    np.savez(out_path, **{kk: np.concatenate(v) for kk, v in acc.items()})


if __name__ == "__main__":
    main()
