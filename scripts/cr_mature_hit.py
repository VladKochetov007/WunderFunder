"""cr_mature step 1b: law-agnostic lower bound on the determination lag of class-3 run labels (TRAIN only).

For run k with entry quotes (A_k, B_k) and tick label (t0, t1), the exit levels are X0 = A_k + t0 (long) and
X1 = B_k - t1 (short). Every label level observed so far is a realised future quote (the long winner's X0 is a
future bid never exceeded before it is first touched). A label cannot be a function of the decoded path before
its levels have been touched, so j_lb(k) = max(first i >= 1 with X0 in {A, B}_{k+i}, same for X1) lower-bounds
the determination lag. Reports the |y|-weighted distribution of j_lb by run kind (interior / outside) and the
weight share whose j_lb exceeds a lag N (labels that are certainly not computable at lag N).

    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_mature_hit.py [n_seq]
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_mature_common import QUANTA, ROOT, Decoder, read_group  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-01/creative/cr_mature"
HMAX = 512
_DEC = None


@nb.njit(cache=True)
def first_hits(A, B, S, X0, X1, hmax):
    n = len(A)
    out = np.full(n, -1, np.int64)
    for k in range(n):
        if S[k] < 0:
            continue
        h0 = -1
        h1 = -1
        for i in range(1, hmax + 1):
            q = k + i
            if q >= n or S[q] != S[k]:
                break
            if h0 < 0 and (A[q] == X0[k] or B[q] == X0[k]):
                h0 = i
            if h1 < 0 and (A[q] == X1[k] or B[q] == X1[k]):
                h1 = i
            if h0 > 0 and h1 > 0:
                break
        if h0 > 0 and h1 > 0:
            out[k] = max(h0, h1)
        elif q >= n or S[q] != S[k]:
            out[k] = -2  # censored by sequence end or a broken path
        else:
            out[k] = hmax + 1
    return out


def process(group):
    global _DEC
    if _DEC is None:
        _DEC = Decoder()
    x, a6, y = read_group(pq.ParquetFile(ROOT / "datasets/train.parquet"), group)
    d = _DEC.decode(x, a6)
    rows = np.flatnonzero(d["E"])
    rows = rows[rows >= 99]
    A, B, S = d["ask"][rows], d["bid"][rows], d["seg"][rows]
    lab = np.rint(y[rows] / QUANTA[3]).astype(np.int64)
    X0, X1 = A + lab[:, 0], B - lab[:, 1]
    jlb = first_hits(A, B, S, X0, X1, HMAX)
    n_rows = np.diff(np.r_[rows, len(a6)])
    w = n_rows * np.minimum(np.abs(lab * QUANTA[3]), 2.0).sum(1)
    kind = np.where((lab[:, 0] <= 0) & (lab[:, 1] <= 0), 0, 1)
    return jlb, w, kind


def main() -> None:
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    z = np.load(ROOT / "cache/train_targets.npz")
    groups = [int(g) for g, c in zip(z["row_group"], z["market_class"], strict=True) if c == 3]
    groups = sorted(groups, key=lambda g: hashlib.sha256(f"cr-mature-hit:{g}".encode()).hexdigest())[:n_seq]
    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(process, groups, chunksize=8))
    jlb = np.concatenate([p[0] for p in parts])
    w = np.concatenate([p[1] for p in parts]).astype(np.float64)
    kind = np.concatenate([p[2] for p in parts])
    known = jlb != -2
    rep = {"n_seq": len(groups), "runs": int(len(jlb)), "censored_w_share": float(w[~known].sum() / w.sum()),
           "never_within_hmax_w_share": float(w[jlb == HMAX + 1].sum() / w[known].sum())}
    grid = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256, 512)
    for name, sel in (("all", known), ("interior", known & (kind == 0)), ("outside", known & (kind == 1))):
        ws = w[sel].sum()
        rep[name] = {"weight_share": float(ws / w[known].sum()),
                     "share_w_jlb_gt": {str(g): float(w[sel & (jlb > g)].sum() / ws) for g in grid},
                     "quantiles_w": {}}
        o = np.argsort(jlb[sel])
        cw = np.cumsum(w[sel][o]) / ws
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            rep[name]["quantiles_w"][str(q)] = int(jlb[sel][o][np.searchsorted(cw, q)])
        print(name, json.dumps(rep[name]), flush=True)
    (OUT / "hit_lag_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
