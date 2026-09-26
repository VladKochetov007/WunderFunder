"""Physicist H-P2: does i1 price displacement since the last i0 mid move predict the next i0 move (class 3)?

    WNN_THREADS=4 scripts/limited .venv/bin/python scripts/probe_i1_lead.py [N_SEQ]
For every class-3 validation sequence: decode i0 and i1 ticks with the train-fitted class-3 code
grammar, then for scored rows (step >= 99) collect lead = i1 mid displacement since the last i0 mid
change (causal), s = sign of the next i0 mid move, D = GRU direction (ab30_eps_aux, train-only, so
out of sample on valid). Moment matrices of [1, lead, s, D] per rows-to-move bucket x chain half x
density tercile, row level and event level (one row per long stretch midpoint).
"""

from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BUCKETS = ((1, 8), (9, 32), (33, 128), (129, 10**9))
_SF = _MV = _P = None


def _init():
    global _SF, _MV, _P
    from wnn.data import SequenceFile
    _SF = SequenceFile(ROOT / "datasets" / "valid.parquet")
    _MV = pickle.load(open(ROOT / "cache" / "class3_moves_full.pkl", "rb"))
    _P = np.load(ROOT / "cache" / "pred_ab30_eps_aux_valid.npy", mmap_mode="r")


def _one(args):
    i, g = args
    from wnn.ticks import reconstruct
    s = _SF.read(int(g))
    x = s.features
    m = s.scored.copy()
    m[:99] = False
    a, b, _ = reconstruct(x, *_MV)
    x1 = x.copy()
    x1[:, 0:22] = x[:, 52:74]
    a1, b1, _ = reconstruct(x1, *_MV)
    mid0, mid1 = (a + b).astype(float), (a1 + b1).astype(float)
    T = len(x)
    mch = np.flatnonzero(np.r_[False, np.diff(mid0) != 0])
    t = np.arange(T)
    k = np.searchsorted(mch, t, side="right")  # next change index in mch
    has = k < len(mch)
    nxt = np.where(has, mch[np.minimum(k, len(mch) - 1)], T)
    d1 = nxt - t
    sgn = np.zeros(T)
    sgn[has] = np.sign(mid0[nxt[has]] - mid0[nxt[has] - 1])
    last = k - 1
    base = np.where(last >= 0, mid1[mch[np.maximum(last, 0)]], mid1[0])
    lead = mid1 - base
    p = np.asarray(_P[i], dtype=np.float64)
    D = p[:, 0] - p[:, 1]
    out = np.zeros((2, len(BUCKETS), 4, 4))
    for bi, (lo, hi) in enumerate(BUCKETS):
        q = m & has & (d1 >= lo) & (d1 <= hi)
        if q.any():
            Z = np.c_[np.ones(q.sum()), lead[q], sgn[q], D[q]]
            out[0, bi] = Z.T @ Z
    # event level: midpoint row of every stretch between i0 changes, if scored-eligible
    starts = np.r_[0, mch]
    ends = np.r_[mch, T]
    mids = (starts + ends) // 2
    ok = (mids >= 99) & (ends < T) & m[np.minimum(mids, T - 1)]
    for bi, (lo, hi) in enumerate(BUCKETS):
        sel = mids[ok & (ends - mids >= lo) & (ends - mids <= hi)]
        if len(sel):
            Z = np.c_[np.ones(len(sel)), lead[sel], sgn[sel], D[sel]]
            out[1, bi] = Z.T @ Z
    return i, out


def pcorr(M):
    n = M[0, 0]
    if n < 30:
        return np.nan, np.nan, 0
    mu = M[0, 1:] / n
    C = M[1:, 1:] / n - np.outer(mu, mu)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    r_ls = R[0, 1]
    # partial corr(lead, s | D)
    r_p = (R[0, 1] - R[0, 2] * R[1, 2]) / np.sqrt((1 - R[0, 2] ** 2) * (1 - R[1, 2] ** 2))
    return r_ls, r_p, int(n)


def main():
    import decide
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    cls, rg, mask = d["market_class"], d["row_group"], d["mask"]
    idx = np.flatnonzero(cls == 3)
    if len(sys.argv) > 1:
        idx = idx[: int(sys.argv[1])]
    _, fold = decide._chains()
    rate = mask.mean(1)
    q = np.quantile(rate[cls == 3], [1 / 3, 2 / 3])
    terc = np.digitize(rate, q)
    res = {}
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "4")), initializer=_init) as ex:
        for i, out in ex.map(_one, [(int(i), int(rg[i])) for i in idx], chunksize=4):
            res[i] = out
    np.savez(ROOT / "cache" / "probe_i1_lead.npz", idx=np.array(list(res)), M=np.stack(list(res.values())))
    names = ["1-8", "9-32", "33-128", "129+"]
    for lev, lname in ((0, "row"), (1, "event")):
        print(f"== {lname} level: corr(lead, next i0 move sign) | partial given GRU direction | n")
        for bi, bn in enumerate(names):
            cells = []
            for tag, sel in (("all", lambda j: True), ("half01", lambda j: fold[j] < 2), ("half23", lambda j: fold[j] >= 2),
                             ("sparse", lambda j: terc[j] == 0), ("mid", lambda j: terc[j] == 1), ("dense", lambda j: terc[j] == 2)):
                M = sum(res[j][lev, bi] for j in res if sel(j))
                r, rp, n = pcorr(M) if not np.isscalar(M) else (np.nan, np.nan, 0)
                se = 1 / np.sqrt(max(n, 1))
                cells.append(f"{tag} {r:+.3f}/{rp:+.3f} (n {n}, se {se:.3f})")
            print(f"  rows-to-move {bn:>7}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
