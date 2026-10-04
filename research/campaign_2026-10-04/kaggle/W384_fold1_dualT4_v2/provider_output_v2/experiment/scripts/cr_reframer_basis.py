"""R12 probe: does a cross-instrument error-correction term (i1 mid - i0 mid vs its causal EWMA) predict class-3 moves?

For every class-3 validation sequence: decode i0 and i1 tick paths with the train-fitted class-3 code grammar
(``cache/class3_moves_full.pkl``), build basis = mid1 - mid0, reset at every unknown i1 transition, and form causal
error-correction terms ec_h = basis - EWMA_h(basis) (h = 64, 512, 4096 rows). Also the short "lead" (i1 mid
displacement since the last i0 mid change, physicist F5). Moments of [1, lead, ec64, ec512, ec4096, s, Dg, Dy] are
accumulated on scored rows by rows-to-next-i0-mid-move bucket and by fold half, where s = sign of the next i0 mid
move, Dg = (p0 - p1)/2 of the sequence's own clean-fold OOS ft mix, Dy = (t0 - t1)/2. Nothing is fitted.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_reframer_basis.py OUT.json [N_SEQ]
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BUCKETS = ((1, 8), (9, 32), (33, 128), (129, 10**9))
HALFLIVES = (64, 512, 4096)
NAMES = ["lead", "ec64", "ec512", "ec4096", "r0_64", "r0_512", "r1_64", "r1_512", "s", "Dg", "Dy"]
FEATS = ["lead", "ec64", "ec512", "ec4096", "r1_64", "r1_512", "r0_64", "r0_512"]
I0_CONTROLS = ["Dg", "r0_64", "r0_512"]
_SF = _MV = None


def _init():
    global _SF, _MV
    from wnn.data import SequenceFile
    _SF = SequenceFile(ROOT / "datasets" / "valid.parquet")
    with (ROOT / "cache" / "class3_moves_full.pkl").open("rb") as source:
        _MV = pickle.load(source)


def ewma_reset(x, seg, h):
    """Causal EWMA (previous rows only) restarting at every new segment id."""
    a = 1 - 0.5 ** (1 / h)
    out = np.empty_like(x)
    m = x[0]
    for t in range(len(x)):
        if t == 0 or seg[t] != seg[t - 1]:
            m = x[t]
        out[t] = m
        m += a * (x[t] - m)
    return out


def _one(args):
    i, g, fold = args
    from wnn.ticks import reconstruct
    s = _SF.read(int(g))
    x, y = s.features, s.targets
    m = s.scored.copy()
    m[:99] = False
    a, b, _ = reconstruct(x, *_MV)
    x1 = x.copy()
    x1[:, 0:22] = x[:, 52:74]
    a1, b1, _ = reconstruct(x1, *_MV)
    mid0, mid1 = (a + b) / 2.0, (a1 + b1) / 2.0
    T = len(x)
    ev1 = np.flatnonzero((x1[1:, :22] != x1[:-1, :22]).any(1)) + 1
    codes = [tuple(r) for r in np.round(x1[np.r_[ev1 - 1, ev1], :22], 4)]
    n1 = len(ev1)
    unk = np.zeros(T, np.int64)
    unk[ev1] = [(c0, c1) not in _MV[0] for c0, c1 in zip(codes[:n1], codes[n1:], strict=True)]
    seg = np.cumsum(unk)
    basis = mid1 - mid0
    ec = [basis - ewma_reset(basis, seg, h) for h in HALFLIVES]
    zero = np.zeros(T, np.int64)
    r0 = [mid0 - ewma_reset(mid0, zero, h) for h in (64, 512)]
    r1 = [mid1 - ewma_reset(mid1, seg, h) for h in (64, 512)]
    mch = np.flatnonzero(np.r_[False, np.diff(mid0) != 0])
    t = np.arange(T)
    k = np.searchsorted(mch, t, side="right")
    has = k < len(mch)
    nxt = np.where(has, mch[np.minimum(k, len(mch) - 1)], T)
    d1 = nxt - t
    sgn = np.zeros(T)
    sgn[has] = np.sign(mid0[nxt[has]] - mid0[nxt[has] - 1])
    last = k - 1
    base_row = np.where(last >= 0, mch[np.maximum(last, 0)], 0)
    lead = mid1 - mid1[base_row]
    lead_ok = seg == seg[base_row]
    seg_start = np.maximum.accumulate(np.where(np.r_[True, seg[1:] != seg[:-1]], t, 0))
    age_ok = (t - seg_start) >= 64
    preds = [np.load(ROOT / "cache" / f"pred_c20260929_ft_since_s{sd}_f{fold}_valid.npy", mmap_mode="r") for sd in (3, 4)]
    p = sum(0.5 * np.asarray(pr[i], np.float64) for pr in preds)
    Dg = (p[:, 0] - p[:, 1]) / 2
    yc = np.clip(y.astype(np.float64), -2, 2)
    Dy = (yc[:, 0] - yc[:, 1]) / 2
    Z_all = np.c_[np.ones(T), lead, *ec, *r0, *r1, sgn, Dg, Dy]
    out = np.zeros((2, len(BUCKETS), Z_all.shape[1], Z_all.shape[1]))
    for bi, (lo, hi) in enumerate(BUCKETS):
        q = m & has & lead_ok & age_ok & (d1 >= lo) & (d1 <= hi)
        if q.any():
            Z = Z_all[q]
            out[0, bi] = Z.T @ Z
    starts, ends = np.r_[0, mch], np.r_[mch, T]
    mids = (starts + ends) // 2
    ok = (mids >= 99) & (ends < T)
    mids_c = np.minimum(mids, T - 1)
    ok &= m[mids_c] & lead_ok[mids_c] & age_ok[mids_c]
    for bi, (lo, hi) in enumerate(BUCKETS):
        sel = mids[ok & (ends - mids >= lo) & (ends - mids <= hi)]
        if len(sel):
            Z = Z_all[sel]
            out[1, bi] = Z.T @ Z
    unk_share = float(unk[ev1].mean()) if n1 else 0.0
    return i, fold, out, unk_share


def corr_table(M):
    n = M[0, 0]
    if n < 30:
        return None
    mu = M[0, 1:] / n
    C = M[1:, 1:] / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(C), 1e-18, None))
    R = C / np.outer(sd, sd)
    ix = {k: j for j, k in enumerate(NAMES)}
    res = {"n": int(n)}

    def partial(f, tgt, controls):
        sub = [ix[f], ix[tgt]] + [ix[c] for c in controls if c != f]
        P = np.linalg.pinv(R[np.ix_(sub, sub)])
        return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))

    for f in FEATS:
        for tgt in ("s", "Dy"):
            res[f"{f}~{tgt}"] = round(float(R[ix[f], ix[tgt]]), 4)
            res[f"{f}~{tgt}|Dg"] = round(partial(f, tgt, ["Dg"]), 4)
            if not f.startswith("r0"):
                res[f"{f}~{tgt}|Dg,r0"] = round(partial(f, tgt, I0_CONTROLS), 4)
    res["Dg~s"] = round(float(R[ix["Dg"], ix["s"]]), 4)
    res["Dg~Dy"] = round(float(R[ix["Dg"], ix["Dy"]]), 4)
    return res


def main():
    out_path = Path(sys.argv[1])
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    cls, rg = d["market_class"], d["row_group"]
    fold_of = {}
    for f in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json").read_text())
        for g in man["evaluation"]["valid"]:
            fold_of[int(g)] = f
    idx = [i for i in np.flatnonzero(cls == 3) if int(rg[i]) in fold_of]
    if len(sys.argv) > 2:
        idx = idx[: int(sys.argv[2])]
    res, unk = {}, []
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "2")), initializer=_init) as ex:
        for i, f, out, u in ex.map(_one, [(int(i), int(rg[i]), fold_of[int(rg[i])]) for i in idx], chunksize=4):
            res[i] = (f, out)
            unk.append(u)
    names = ["1-8", "9-32", "33-128", "129+"]
    report = {"n_sequences": len(res), "i1_unknown_transition_share_mean": float(np.mean(unk)), "levels": {}}
    for lev, lname in ((0, "row"), (1, "event")):
        report["levels"][lname] = {}
        for bi, bn in enumerate(names):
            cell = {}
            for tag, sel in (("all", lambda f: True), ("folds01", lambda f: f < 2), ("folds23", lambda f: f >= 2)):
                M = sum(out[lev, bi] for f, out in res.values() if sel(f))
                cell[tag] = corr_table(M)
            report["levels"][lname][bn] = cell
            print(lname, bn, json.dumps(cell["all"]), flush=True)
    out_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
