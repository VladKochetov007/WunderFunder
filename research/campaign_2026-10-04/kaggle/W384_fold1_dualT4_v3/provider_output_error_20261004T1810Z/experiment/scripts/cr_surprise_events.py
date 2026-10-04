"""Surprise miner: event (E-row) and run-level table for the since pair, clean out-of-fold.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_events.py OUT.npz

Per E row t >= 100 (a6 at its lower endpoint): targets and FT-pair predictions at t-1 and t (ticks of the
class quantum), the TRAIN-fitted move table shift (d_ask, d_bid) of the code transition (unknown -> 99),
the scored count and length of the run that starts at t, and the prediction at the run's last row.
Targets/mask are diagnostics only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = np.float32(-5.199337482452393)
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
PX0 = [f"i0_p{i}" for i in range(22)]


def main() -> None:
    out_path = Path(sys.argv[1])
    with np.load(ROOT / "cache/valid_targets.npz") as z:
        y_all, mask_all, cls_all = z["y"], z["mask"], z["market_class"]
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
        t = pf.read_row_group(g, columns=PX0 + ["a6"], use_threads=False)
        px = np.column_stack([t.column(c).to_numpy() for c in PX0])
        e = t.column("a6").to_numpy().astype(np.float32) == ENDPOINT
        e[0] = False
        q = np.rint(px.astype(np.float64) * 1e4).astype(np.int64)
        with np.errstate(over="ignore"):
            h = (q * mult[None]).sum(1)
            h ^= h >> np.int64(29)
        k = fold_of[g]
        p = 0.5 * (preds[(3, k)][g].astype(np.float64) + preds[(4, k)][g])
        cls = int(cls_all[g])
        qu = QUANTA[cls]
        y = y_all[g].astype(np.float64) / qu
        p = p / qu
        m = mask_all[g]
        ev = np.flatnonzero(e)
        ev = ev[ev >= 100]
        if len(ev) == 0:
            continue
        key = h[ev - 1].astype(np.uint64) * np.uint64(1000003) ^ h[ev].astype(np.uint64)
        pos = np.minimum(np.searchsorted(mk, key), len(mk) - 1)
        hit = mk[pos] == key
        da = np.where(hit, md[pos, 0], 99)
        db = np.where(hit, md[pos, 1], 99)
        nxt = np.r_[ev[1:], len(e)]
        cm = np.r_[0, np.cumsum(m)]
        d = {
            "seq": np.full(len(ev), g, np.int16), "t": ev.astype(np.int16), "cls": np.full(len(ev), cls, np.int8),
            "y_prev0": y[ev - 1, 0], "y_prev1": y[ev - 1, 1], "y0": y[ev, 0], "y1": y[ev, 1],
            "p_prev0": p[ev - 1, 0], "p_prev1": p[ev - 1, 1], "p0": p[ev, 0], "p1": p[ev, 1],
            "p_end0": p[nxt - 1, 0], "p_end1": p[nxt - 1, 1],
            "p_next0": p[np.minimum(nxt, len(e) - 1), 0], "p_next1": p[np.minimum(nxt, len(e) - 1), 1],
            "da": da.astype(np.int8), "db": db.astype(np.int8),
            "run_len": (nxt - ev).astype(np.int32), "run_scored": (cm[nxt] - cm[ev]).astype(np.int32),
            "scored": m[ev],
        }
        for kk, v in d.items():
            acc.setdefault(kk, []).append(v)
        if g % 300 == 0:
            print(g, flush=True)
    np.savez(out_path, **{kk: np.concatenate(v) for kk, v in acc.items()})


if __name__ == "__main__":
    main()
