"""cr_mature step 1: determination lag of class-3 run labels from the feature-decoded quote path (TRAIN only).

A run k is the block of rows from event E_k up to E_{k+1} (labels are constant on it, H1). Its label is the tick
pair (t0, t1). The key K_j(k) is the entry spread plus the decoded (d_ask, d_bid) moves of the next j events
(variant "path") or that plus log2 inter-event row-gap buckets (variant "path_gap"). A mode table K_j -> label is
fitted on fit-side components and graded on held-out components: coverage, exact share among covered runs
(|y|-weighted over scored rows and unweighted), and per run the earliest j after which the table label equals the
true label for every larger j in the grid (the determination lag). Tables are saved for the VALID feedback probe.

    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_mature_lag.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_mature_common import QUANTA, ROOT, Decoder, read_group  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-01/creative/cr_mature"
JS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32)
P = np.int64(1_000_003)
MASK63 = np.int64(0x7FFFFFFFFFFFFFFF)
SUPPORT = 3
MAX_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
_DEC = None


def move_code(da, db):
    return (np.clip(da, -15, 15) + 16) * 33 + (np.clip(db, -15, 15) + 16)


def event_keys(A, B, rows, seg, n_rows, with_gap: bool):
    """Keys K_j for every event k (rows = E rows); -1 where the path up to k+j is broken or censored."""
    n = len(A)
    code = np.zeros(n, np.int64)
    code[1:] = move_code(np.diff(A), np.diff(B))
    if with_gap:
        gap = np.diff(np.r_[rows, n_rows])  # gap[k] = rows from E_k to E_{k+1} (run length)
        g = np.minimum(np.log2(np.maximum(gap, 1)).astype(np.int64), 12)
        code[1:] = code[1:] * 13 + g[:-1]  # move into E_i combined with the length of run i-1
    ok_step = np.zeros(n, bool)
    ok_step[1:] = (seg[1:] == seg[:-1]) & (seg[1:] >= 0)
    keys = np.full((len(JS), n), -1, np.int64)
    h = (A - B + 50).astype(np.int64)
    valid = seg >= 0
    jmax = max(JS)
    with np.errstate(over="ignore"):
        for j in range(jmax + 1):
            if j > 0:
                nxt = np.full(n, 0, np.int64)
                nxt[: n - j] = code[j:]
                st = np.zeros(n, bool)
                st[: n - j] = ok_step[j:]
                h = (h * P + nxt) & MASK63
                valid &= st
            if j in JS:
                keys[JS.index(j)] = np.where(valid, h, -1)
    return keys


def summary_keys(A, B, seg):
    """Compact keys: entry spread and, over the next j events relative to the entry ask, the running max/min of
    ask and bid plus the latest ask and bid (clipped to +-8 ticks)."""
    n = len(A)
    ok_step = np.zeros(n, bool)
    ok_step[1:] = (seg[1:] == seg[:-1]) & (seg[1:] >= 0)
    keys = np.full((len(JS), n), -1, np.int64)
    valid = seg >= 0
    s0 = np.clip(A - B, 0, 9)
    big = np.int64(10**6)
    amax = np.full(n, -big)
    amin = np.full(n, big)
    bmax = np.full(n, -big)
    bmin = np.full(n, big)
    la = np.zeros(n, np.int64)
    lb = np.zeros(n, np.int64)
    for j in range(max(JS) + 1):
        if j > 0:
            a = np.zeros(n, np.int64)
            b = np.zeros(n, np.int64)
            a[: n - j] = A[j:] - A[: n - j]
            b[: n - j] = B[j:] - A[: n - j]
            st = np.zeros(n, bool)
            st[: n - j] = ok_step[j:]
            valid &= st
            amax, amin = np.maximum(amax, a), np.minimum(amin, a)
            bmax, bmin = np.maximum(bmax, b), np.minimum(bmin, b)
            la, lb = a, b
        if j in JS:
            if j == 0:
                h = s0
            else:
                h = s0
                for v in (amax, amin, bmax, bmin, la, lb):
                    h = h * 17 + np.clip(v, -8, 8) + 8
            keys[JS.index(j)] = np.where(valid, h, -1)
    return keys


def process(args):
    group, path, labels = args
    global _DEC
    if _DEC is None:
        _DEC = Decoder()
    src = pq.ParquetFile(path)
    x, a6, y = read_group(src, group, labels=labels)
    d = _DEC.decode(x, a6)
    rows = np.flatnonzero(d["E"])
    A, B, seg = d["ask"][rows], d["bid"][rows], d["seg"][rows]
    out = {"rows": rows, "A": A, "B": B, "seg": seg,
           "kp": event_keys(A, B, rows, seg, len(a6), False), "ks": summary_keys(A, B, seg)}
    if labels:
        out["lab"] = np.rint(y[rows] / QUANTA[3]).astype(np.int64)
    return group, out


def label_code(lab):
    return (np.clip(lab[:, 0], -40, 40) + 50) * 101 + (np.clip(lab[:, 1], -40, 40) + 50)


def fit_table(keys, lab, support=SUPPORT):
    """Mode label per key with support >= SUPPORT. Returns sorted keys, mode labels, support, purity."""
    ok = keys >= 0
    k, l = keys[ok], lab[ok]
    pair = np.stack([k, l], 1)
    uniq, cnt = np.unique(pair, axis=0, return_counts=True)
    # for each key take the max-count label
    order = np.lexsort((-cnt, uniq[:, 0]))
    uniq, cnt = uniq[order], cnt[order]
    first = np.r_[True, uniq[1:, 0] != uniq[:-1, 0]]
    tot = np.add.reduceat(cnt, np.flatnonzero(first))
    tk, tl, tc = uniq[first, 0], uniq[first, 1], cnt[first]
    keep = tot >= support
    return tk[keep], tl[keep], tot[keep], tc[keep] / tot[keep]


def lookup(tk, tl, keys):
    pos = np.searchsorted(tk, keys)
    pos = np.minimum(pos, len(tk) - 1)
    hit = (keys >= 0) & (tk[pos] == keys)
    return np.where(hit, tl[pos], -1), hit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "cache/train_targets.npz")
    rg, cls = z["row_group"].astype(int), z["market_class"].astype(int)
    man = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
    comp = {s["row_group"]: s["component"] for s in man["sequences"]["train"]}
    groups = [int(g) for g, c in zip(rg, cls, strict=True) if c == 3 and int(g) in comp]
    groups = sorted(groups, key=lambda g: hashlib.sha256(f"cr-mature-seq:{g}".encode()).hexdigest())[:MAX_SEQ]
    side = {g: int(hashlib.sha256(f"cr-mature:{comp[g]}".encode()).hexdigest(), 16) % 10 < 6 for g in groups}
    path = str(ROOT / "datasets/train.parquet")
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for i, (g, o) in enumerate(ex.map(process, [(g, path, True) for g in groups], chunksize=8)):
            # run weights: scored rows of the run times |clip(t*q)| summed over both targets
            m = np.arange(20_000) >= 99  # TRAIN has no scoring mask: count rows past warm-up
            rows = o["rows"]
            cnt = np.add.reduceat(m.astype(np.int64), rows) if len(rows) else np.zeros(0, np.int64)
            w = np.minimum(np.abs(o["lab"] * QUANTA[3]), 2.0).sum(1)
            o["w"] = cnt * w
            del o["A"], o["B"], o["seg"]
            res[g] = o
            if i % 500 == 0:
                print("read", i, len(groups), flush=True)
    fit = [g for g in groups if side[g]]
    test = [g for g in groups if not side[g]]
    report = {"n_fit_seq": len(fit), "n_test_seq": len(test), "js": list(JS), "support": SUPPORT, "variants": {}}
    for var in ("kp", "ks"):
        lab_f = np.concatenate([label_code(res[g]["lab"]) for g in fit])
        lab_t = np.concatenate([label_code(res[g]["lab"]) for g in test])
        w_t = np.concatenate([res[g]["w"] for g in test]).astype(np.float64)
        rec = np.full((len(JS), len(lab_t)), -1, np.int64)
        rows_out = []
        tables = {}
        for ji, j in enumerate(JS):
            kf = np.concatenate([res[g][var][ji] for g in fit])
            kt = np.concatenate([res[g][var][ji] for g in test])
            tk, tl, tc, pur = fit_table(kf, lab_f)
            tables[j] = (tk, tl, tc)
            pred, hit = lookup(tk, tl, kt)
            rec[ji] = pred
            valid = kt >= 0
            ex = hit & (pred == lab_t)
            row = {"j": j, "table_keys": len(tk), "valid_share_w": float(w_t[valid].sum() / w_t.sum()),
                   "coverage_w": float(w_t[hit].sum() / w_t[valid].sum()),
                   "exact_of_covered_w": float(w_t[ex].sum() / max(w_t[hit].sum(), 1e-12)),
                   "exact_of_covered_runs": float(ex.sum() / max(hit.sum(), 1)),
                   "exact_of_valid_w": float(w_t[ex].sum() / w_t[valid].sum()),
                   "fit_purity_mean_w": float(np.average(pur, weights=tc))}
            rows_out.append(row)
            print(var, row, flush=True)
        # earliest stable j: rec == lab for all j' >= j in the grid (only runs covered at every j)
        cov_all = (rec >= 0).all(0)
        good = rec == lab_t[None]
        stable = np.flip(np.cumprod(np.flip(good, 0), 0), 0).astype(bool)  # stable[ji] = good for all >= ji
        first = np.where(stable.any(0), stable.argmax(0), len(JS))
        sel = cov_all
        dist = {}
        for ji in range(len(JS) + 1):
            name = str(JS[ji]) if ji < len(JS) else "never"
            on = sel & (first == ji)
            dist[name] = {"runs": int(on.sum()), "share_w": float(w_t[on].sum() / max(w_t[sel].sum(), 1e-12))}
        cum = np.cumsum([dist[k]["share_w"] for k in dist])
        report["variants"][var] = {"by_j": rows_out, "first_stable_j": dist,
                                   "cum_share_w": dict(zip(dist.keys(), cum.tolist())),
                                   "runs_covered_all_j": int(sel.sum()), "w_covered_all_j": float(w_t[sel].sum() / w_t.sum())}
        print(var, "first stable j dist", {k: round(v["share_w"], 4) for k, v in dist.items()}, flush=True)
        np.savez_compressed(OUT / f"lag_tables_{var}.npz",
                            **{f"k{j}": t[0] for j, t in tables.items()}, **{f"l{j}": t[1] for j, t in tables.items()},
                            **{f"c{j}": t[2] for j, t in tables.items()})
    (OUT / "lag_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", OUT / "lag_report.json")


if __name__ == "__main__":
    main()
