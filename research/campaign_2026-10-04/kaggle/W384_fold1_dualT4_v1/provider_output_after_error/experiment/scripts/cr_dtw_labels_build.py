"""Event-clock tables for the DTW label probes: per-event path features and per-segment metric sums on valid.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_labels_build.py OUT.npz

Events: rows where a6 sits at its float32 lower endpoint (quote-move marker, feature only). Segment k = rows
[e_k, e_{k+1}); targets are constant on it (checked). Per event (feature only, causal at e_k):
  dask, dbid  quote change in ticks since the previous event (TRAIN-fit decoder of scripts/cr_mature_common.py;
              0 and flagged when the decoder path is broken), spread ticks at e_k,
  gap         log1p(e_k - e_{k-1}), ntr = log1p(trade-update rows in (e_{k-1}, e_k]), n1 = log1p(i1 code changes).
Per segment and model m in (parent, head-FT), target j: |clip y|-weighted sums over scored rows >= 99 of
  1, y, p, y^2, p^2, y p   and the within-segment drift d = p_t - p_{e_k}:  d, p d, d^2, y d.
Out-of-fold panel: sequence i takes its predictions from fold f(i)'s files (only those rows are OOF).
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

MODELS = ("c20261001_d256_s3_f{k}", "c20261001_ft_d256_s3_f{k}h")
TRADE = [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)]
I1 = [f"i1_p{i}" for i in range(22)]
NS = 10  # sums per (model, target)

_G: dict = {}


def _init(fold_of):
    z = np.load(ROOT / "cache/valid_targets.npz")
    _G["y"], _G["mask"] = z["y"], z["mask"]
    _G["fold"] = fold_of
    _G["pred"] = {(m, k): np.load(ROOT / f"cache/pred_{name.format(k=k)}_valid.npy", mmap_mode="r")
                  for m, name in enumerate(MODELS) for k in range(4)}
    _G["pf"] = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    _G["dec"] = Decoder()


def seq_tables(i: int):
    t = _G["pf"].read_row_group(i, columns=PRICE + TRADE + I1 + ["a6"], use_threads=False)
    x = np.column_stack([t.column(c).to_numpy() for c in PRICE]).astype(np.float32)
    tr = np.column_stack([t.column(c).to_numpy() for c in TRADE]).astype(np.float32)
    i1 = np.column_stack([t.column(c).to_numpy() for c in I1]).astype(np.float32)
    a6 = t.column("a6").to_numpy().astype(np.float32)
    d = _G["dec"].decode(x, a6)
    n = len(a6)
    e = a6 == ENDPOINT
    e[0] = False
    ev = np.flatnonzero(e)
    if len(ev) < 2:
        return None
    trow = np.zeros(n, np.int64)
    trow[1:] = (tr[1:] != tr[:-1]).any(1)
    i1row = np.zeros(n, np.int64)
    i1row[1:] = (i1[1:] != i1[:-1]).any(1)
    ctr, ci1 = np.cumsum(trow), np.cumsum(i1row)
    prev = np.r_[0, ev[:-1]]
    ok = (d["seg"][ev] >= 0) & (d["seg"][prev] == d["seg"][ev])
    ok[0] = False
    dask = np.where(ok, d["ask"][ev] - d["ask"][prev], 0)
    dbid = np.where(ok, d["bid"][ev] - d["bid"][prev], 0)
    feat = np.column_stack([dask, dbid, np.maximum(d["spread"][ev], 0), np.log1p(ev - prev),
                            np.log1p(ctr[ev] - ctr[prev]), np.log1p(ci1[ev] - ci1[prev]), ok]).astype(np.float32)

    y = np.clip(_G["y"][i].astype(np.float64), -2, 2)
    m = _G["mask"][i].copy()
    m[:99] = False
    f = _G["fold"][i]
    P = [np.clip(np.asarray(_G["pred"][(mm, f)][i], dtype=np.float64), -2, 2) for mm in range(len(MODELS))]
    seg_id = np.cumsum(e) - 1  # -1 before first event
    ends = np.r_[ev[1:], n]
    # check targets are constant on segments
    chg = np.zeros(n, bool)
    chg[1:] = (y[1:] != y[:-1]).any(1)
    nonconst = int((chg & ~e).sum())
    sel = m & (seg_id >= 0)
    rows = np.flatnonzero(sel)
    sid = seg_id[rows]
    nseg = len(ev)
    sums = np.zeros((nseg, len(MODELS), 2, NS))
    w = np.abs(y[rows])
    yy = y[rows]
    for mm, p in enumerate(P):
        pr = p[rows]
        dr = pr - p[ev[sid]]
        for j in range(2):
            ww, yj, pj, dj = w[:, j], yy[:, j], pr[:, j], dr[:, j]
            for s, v in enumerate((ww, ww * yj, ww * pj, ww * yj * yj, ww * pj * pj, ww * yj * pj,
                                   ww * dj, ww * pj * dj, ww * dj * dj, ww * yj * dj)):
                sums[:, mm, j, s] = np.bincount(sid, weights=v, minlength=nseg)
    nsc = np.bincount(sid, minlength=nseg)
    first = np.full(nseg, -1, np.int64)
    first[sid[::-1]] = rows[::-1]
    pstart = np.stack([p[ev] for p in P], 1).astype(np.float32)  # (nseg, model, 2)
    k = np.flatnonzero(nsc > 0)
    return dict(feat=feat, seq=np.full(nseg, i, np.int32), start=ev.astype(np.int32),
                length=(ends - ev).astype(np.int32)), \
        dict(sidx=k, sums=sums[k].astype(np.float32), nsc=nsc[k].astype(np.int32), first=first[k].astype(np.int32),
             ystart=y[ev[k]].astype(np.float32), pstart=pstart[k]), nonconst


def main():
    out = Path(sys.argv[1])
    fold_of = np.full(1873, -1, np.int64)
    for k in range(4):
        s = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json", ROOT)
        fold_of[np.asarray(sorted(s.evaluation_groups))] = k
    assert (fold_of >= 0).all()
    evp, sgp, off, nonconst = [], [], 0, 0
    with Pool(2, initializer=_init, initargs=(fold_of,)) as pool:
        for r in pool.imap(seq_tables, range(1873), chunksize=8):
            if r is None:
                continue
            a, b, nc = r
            b["sidx"] = b["sidx"] + off
            off += len(a["seq"])
            evp.append(a)
            sgp.append(b)
            nonconst += nc
    res = {"ev_" + k: np.concatenate([r[k] for r in evp]) for k in evp[0]}
    res.update({"sg_" + k: np.concatenate([r[k] for r in sgp]) for k in sgp[0]})
    z = np.load(ROOT / "cache/valid_targets.npz")
    res["cls_of_seq"] = z["market_class"]
    res["fold_of_seq"] = fold_of
    np.savez(out, **res)
    print("events", off, "scored segments", len(res["sg_sidx"]), "non-constant target rows inside segments", nonconst)


if __name__ == "__main__":
    main()
