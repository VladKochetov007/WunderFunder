"""Second pass on valid: event-level labels/predictions (label-structure stats) and elapsed-bucket drift sums.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_labels_build2.py OUT.npz

Per event e_k (all events, not only scored): decoded mid (ticks, NaN when the decoder path is broken), y at e_k
(t0, t1, raw), parent OOF p at e_k and at the segment's last row, scored flag of e_k, segment length.
Per (class, fold, elapsed bucket of t - e_k, target): the ten |y|-weighted drift sums of cr_dtw_labels_build.py for
the parent model, so the drift coefficient can be fit per elapsed bucket.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cr_mature_common import ENDPOINT, PRICE, Decoder  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

MODEL = "c20261001_d256_s3_f{k}"
EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])  # bucket b: EDGES[b] <= t - e_k < EDGES[b+1]
_G: dict = {}


def _init(fold_of):
    z = np.load(ROOT / "cache/valid_targets.npz")
    _G["y"], _G["mask"], _G["cls"] = z["y"], z["mask"], z["market_class"]
    _G["fold"] = fold_of
    _G["pred"] = {k: np.load(ROOT / f"cache/pred_{MODEL.format(k=k)}_valid.npy", mmap_mode="r") for k in range(4)}
    _G["pf"] = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    _G["dec"] = Decoder()


def seq_tables(i: int):
    t = _G["pf"].read_row_group(i, columns=PRICE + ["a6"], use_threads=False)
    x = np.column_stack([t.column(c).to_numpy() for c in PRICE]).astype(np.float32)
    a6 = t.column("a6").to_numpy().astype(np.float32)
    d = _G["dec"].decode(x, a6)
    n = len(a6)
    e = a6 == ENDPOINT
    e[0] = False
    ev = np.flatnonzero(e)
    if len(ev) < 2:
        return None
    mid = np.where(d["seg"][ev] >= 0, (d["ask"][ev] + d["bid"][ev]) / 2.0, np.nan)
    segid = np.where(d["seg"][ev] >= 0, d["seg"][ev], -1)
    y = _G["y"][i].astype(np.float32)
    f = _G["fold"][i]
    p = np.asarray(_G["pred"][f][i], dtype=np.float32)
    ends = np.r_[ev[1:], n]
    m = _G["mask"][i].copy()
    m[:99] = False
    ev_tab = dict(seq=np.full(len(ev), i, np.int32), start=ev.astype(np.int32), length=(ends - ev).astype(np.int32),
                  mid=mid.astype(np.float32), dseg=segid.astype(np.int32), y=y[ev], p=p[ev], pend=p[ends - 1],
                  scored=m[ev])
    # elapsed-bucket drift sums
    sid = np.cumsum(e) - 1
    rows = np.flatnonzero(m & (sid >= 0))
    el = rows - ev[sid[rows]]
    b = np.searchsorted(EDGES, el, side="right") - 1
    yc = np.clip(y[rows].astype(np.float64), -2, 2)
    pc = np.clip(p[rows].astype(np.float64), -2, 2)
    dc = pc - np.clip(p[ev[sid[rows]]].astype(np.float64), -2, 2)
    w = np.abs(yc)
    sums = np.zeros((len(EDGES), 2, 10))
    for j in range(2):
        ww, yj, pj, dj = w[:, j], yc[:, j], pc[:, j], dc[:, j]
        for s, v in enumerate((ww, ww * yj, ww * pj, ww * yj * yj, ww * pj * pj, ww * yj * pj,
                               ww * dj, ww * pj * dj, ww * dj * dj, ww * yj * dj)):
            sums[:, j, s] = np.bincount(b, weights=v, minlength=len(EDGES))
    return ev_tab, sums, int(_G["cls"][i]), int(f)


def main():
    out = Path(sys.argv[1])
    fold_of = np.full(1873, -1, np.int64)
    for k in range(4):
        s = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json", ROOT)
        fold_of[np.asarray(sorted(s.evaluation_groups))] = k
    parts = []
    cellsums = np.zeros((4, 4, len(EDGES), 2, 10))
    with Pool(2, initializer=_init, initargs=(fold_of,)) as pool:
        for r in pool.imap(seq_tables, range(1873), chunksize=8):
            if r is None:
                continue
            a, s, c, f = r
            parts.append(a)
            cellsums[c, f] += s
    res = {"ev_" + k: np.concatenate([r[k] for r in parts]) for k in parts[0]}
    z = np.load(ROOT / "cache/valid_targets.npz")
    res.update(cls_of_seq=z["market_class"], fold_of_seq=fold_of, elapsed_cells=cellsums, edges=EDGES)
    np.savez(out, **res)
    print("events", len(res["ev_seq"]))


if __name__ == "__main__":
    main()
