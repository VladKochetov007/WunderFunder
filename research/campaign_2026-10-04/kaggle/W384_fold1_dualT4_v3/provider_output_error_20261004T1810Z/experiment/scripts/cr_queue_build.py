"""Queue-race falsifier: class-3 row tables with current-row features and causal in-run history features.

    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_queue_build.py train OUT.npz [STRIDE]
    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_queue_build.py valid OUT.npz

A run starts at a quote-move row (a6 at its float32 lower endpoint). Every history feature at row t uses rows <= t
of the same sequence and is O(1)-state computable online (running sums reset at each run start, values latched at
the run start / last change). train: class-3 sequences not chain-linked to any valid sequence, every STRIDE-th row
from 99. valid: all scored class-3 rows. Columns prefixed nc_ (rows until the next quote move) are diagnostics only.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import connected_components  # noqa: E402

ENDPOINT = np.float32(-5.199337482452393)
FEAT = ([f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)] + [f"i0_dp{i}" for i in range(4)]
        + [f"i0_dv{i}" for i in range(4)] + [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]
        + [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)] + [f"a{i}" for i in range(8)])
HIST = (["h_since_E", "h_prev_run", "h_nE_256", "h_nE_2048"]
        + [f"h_dVrun{i}" for i in range(22)] + ["h_pathVb", "h_pathVa", "h_nchgVb", "h_nchgVa", "h_netVb", "h_netVa"]
        + [f"h_dVatE{i}" for i in range(22)] + [f"h_dPatE{i}" for i in range(22)] + ["h_dPbE", "h_dPaE"]
        + [f"h_dPrun{i}" for i in range(22)] + ["h_ncc_run", "h_ntr_run"]
        + [f"h_sumdp{i}" for i in range(4)] + [f"h_sumdv{i}" for i in range(4)] + [f"h_sumdpdv{i}" for i in range(4)]
        + [f"h_lastdVb{i}" for i in range(11)] + [f"h_lastdVa{i}" for i in range(11)]
        + ["h_since_Vb", "h_since_Va", "h_since_tr", "h_npx1_run", "h_ntr1_run"])


def latest(flag: np.ndarray) -> np.ndarray:
    """Index of the latest row <= t where flag holds (0 if none)."""
    idx = np.arange(len(flag))
    return np.maximum.accumulate(np.where(flag, idx, 0))


def run_sum(x: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Sum of x over rows (rs, t] (run start row excluded)."""
    c = np.cumsum(x, axis=0, dtype=np.float64)
    return c - c[rs]


_FILES: dict[str, pq.ParquetFile] = {}


def seq_features(g: int, split: str, stride: int, sel_mask: np.ndarray | None):
    if split not in _FILES:
        _FILES[split] = pq.ParquetFile(ROOT / f"datasets/{split}.parquet")
    pf = _FILES[split]
    t = pf.read_row_group(g, columns=FEAT + ["t0", "t1"], use_threads=False)
    X = np.column_stack([t.column(c).to_numpy() for c in FEAT]).astype(np.float32)
    y = np.column_stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()]).astype(np.float32)
    n = len(X)
    P, V, DP, DV = X[:, 0:22], X[:, 22:44], X[:, 44:48], X[:, 48:52]
    P1, DP1, DV1 = X[:, 52:74], X[:, 96:100], X[:, 100:104]
    e = X[:, 110] == ENDPOINT
    e[0] = False
    idx = np.arange(n)
    rs = latest(e)
    prev_e = np.zeros(n, np.int64)
    eidx = np.flatnonzero(e)
    if len(eidx) > 1:
        pr = np.zeros(n, np.int64)
        pr[eidx[1:]] = eidx[:-1]
        prev_e = pr[rs]
    ce = np.cumsum(e)
    dV = np.zeros_like(V)
    dV[1:] = V[1:] - V[:-1]
    dP = np.zeros_like(P)
    dP[1:] = P[1:] - P[:-1]
    chb = (dV[:, :11] != 0).any(1)
    cha = (dV[:, 11:] != 0).any(1)
    cc = (dP != 0).any(1) & ~e
    tr = np.zeros(n, bool)
    tr[1:] = ((DP[1:] != DP[:-1]) | (DV[1:] != DV[:-1])).any(1)
    px1 = np.zeros(n, bool)
    px1[1:] = (P1[1:] != P1[:-1]).any(1)
    tr1 = np.zeros(n, bool)
    tr1[1:] = ((DP1[1:] != DP1[:-1]) | (DV1[1:] != DV1[:-1])).any(1)
    lb, la, lt = latest(chb), latest(cha), latest(tr)
    trf = tr[:, None].astype(np.float32)
    H = [
        np.log1p(idx - rs), np.log1p(np.where(rs > 0, rs - prev_e, 0)),
        ce - np.r_[np.zeros(256), ce[:-256]], ce - np.r_[np.zeros(2048), ce[:-2048]],
    ]
    dvrun = V - V[rs]
    H += list(dvrun.T)
    adv = np.abs(dV)
    H += [run_sum(adv[:, :11].sum(1), rs), run_sum(adv[:, 11:].sum(1), rs), run_sum(chb, rs), run_sum(cha, rs),
          np.abs(dvrun[:, :11]).sum(1), np.abs(dvrun[:, 11:]).sum(1)]
    H += list(dV[rs].T) + list(dP[rs].T)
    H += [np.abs(dP[rs, :11]).sum(1), np.abs(dP[rs, 11:]).sum(1)]
    H += list((P - P[rs]).T)
    H += [run_sum(cc, rs), run_sum(tr, rs)]
    H += list(run_sum(DP * trf, rs).T) + list(run_sum(DV * trf, rs).T) + list(run_sum(DP * DV * trf, rs).T)
    H += list(dV[lb, :11].T) + list(dV[la, 11:].T)
    H += [np.log1p(idx - lb), np.log1p(idx - la), np.log1p(idx - lt), run_sum(px1, rs), run_sum(tr1, rs)]
    Hm = np.column_stack(H).astype(np.float32)
    nxt = np.minimum.accumulate(np.where(e, idx, n)[::-1])[::-1]
    # rows strictly after t until the next quote move (diagnostic only)
    nxt_after = np.r_[nxt[1:], n]
    until = nxt_after - idx
    if sel_mask is None:
        rows = np.arange(99, n, stride)
    else:
        rows = np.flatnonzero(sel_mask)
    return (np.full(len(rows), g, np.int32), rows.astype(np.int32), X[rows], Hm[rows], y[rows],
            until[rows].astype(np.int32), e[rows])


def _job(a):
    return seq_features(*a)


def main() -> None:
    split, out = sys.argv[1], Path(sys.argv[2])
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    ch = np.load(ROOT / "cache/chains.npz")
    comp = connected_components(ch["next"])
    comp_of = {(s, int(r)): int(c) for s, r, c in zip(ch["split"], ch["row_group"], comp, strict=True)}
    if split == "train":
        z = np.load(ROOT / "cache/train_targets.npz")
        vcomps = {c for (s, _), c in comp_of.items() if s == "valid"}
        groups = [int(g) for g, c in zip(z["row_group"], z["market_class"], strict=True)
                  if c == 3 and comp_of[("train", int(g))] not in vcomps]
        jobs = [(g, "train", stride, None) for g in groups]
    else:
        z = np.load(ROOT / "cache/valid_targets.npz")
        groups = [int(g) for g, c in zip(z["row_group"], z["market_class"], strict=True) if c == 3]
        mask = z["mask"]
        jobs = [(g, "valid", stride, mask[g]) for g in groups]
    print(split, "sequences", len(jobs), flush=True)
    parts = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap(_job, jobs, chunksize=4)):
            parts.append(r)
            if i % 500 == 0:
                print(i, flush=True)
    names = ["seq", "t", "X", "H", "y", "nc_until_E", "E"]
    res = {k: np.concatenate([p[i] for p in parts]) for i, k in enumerate(names)}
    res["comp"] = np.array([comp_of[(split, int(g))] for g in res["seq"]], np.int32)
    res["hist_names"] = np.array(HIST)
    assert res["H"].shape[1] == len(HIST), (res["H"].shape, len(HIST))
    np.savez(out, **res)
    print("rows", len(res["seq"]), "H", res["H"].shape, flush=True)


if __name__ == "__main__":
    main()
