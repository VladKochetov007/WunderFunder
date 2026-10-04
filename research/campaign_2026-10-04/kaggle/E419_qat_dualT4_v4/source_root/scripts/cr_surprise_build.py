"""Surprise miner: per-scored-row forensic table of clean out-of-fold predictions and causal event state.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_build.py OUT.npz

Each valid sequence takes predictions from the fold whose evaluation_groups contain it (4 clean folds).
Models: ft = two-stage FT pair (s3+s4)/2, par = parents, hd = head-FT. Feature columns are read from
valid.parquet; E = a6 at its float32 lower endpoint (quote-move marker). Columns flagged nc_ are
non-causal (diagnostics only). Mask/target fields are diagnostics, never model inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = np.float32(-5.199337482452393)
FAMILIES = {"ft": "pred_c20260929_ft_since_s{s}_f{k}_valid.npy",
            "hd": "pred_c20260929_ft_since_s{s}_f{k}h_valid.npy",
            "par": "pred_c20260929_since_s{s}_f{k}_valid.npy"}
PX0 = [f"i0_p{i}" for i in range(22)]
V0 = [f"i0_v{i}" for i in range(22)]
DV0 = [f"i0_dv{i}" for i in range(4)]
DP0 = [f"i0_dp{i}" for i in range(4)]
PX1 = [f"i1_p{i}" for i in range(22)]
V1 = [f"i1_v{i}" for i in range(22)]
DV1 = [f"i1_dv{i}" for i in range(4)]
AX = [f"a{i}" for i in range(8)]
COLS = PX0 + V0 + DV0 + DP0 + PX1 + V1 + DV1 + AX


def changed(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), bool)
    out[1:] = (x[1:] != x[:-1]).any(1) if x.ndim == 2 else x[1:] != x[:-1]
    return out


def since(flag: np.ndarray) -> np.ndarray:
    idx = np.arange(len(flag))
    return idx - np.maximum.accumulate(np.where(flag, idx, 0))


def until(flag: np.ndarray) -> np.ndarray:
    n = len(flag)
    idx = np.arange(n)
    nxt = np.minimum.accumulate(np.where(flag, idx, n)[::-1])[::-1]
    return nxt - idx  # n - idx when no later event


def in_run_count(flag: np.ndarray, run: np.ndarray) -> np.ndarray:
    c = np.cumsum(flag)
    start = np.zeros(run.max() + 1, np.int64)
    first = np.r_[0, np.flatnonzero(run[1:] != run[:-1]) + 1]
    start[run[first]] = c[first] - flag[first]
    return c - start[run]


def main() -> None:
    out_path = Path(sys.argv[1])
    with np.load(ROOT / "cache/valid_targets.npz") as z:
        y_all, mask_all = z["y"], z["mask"]
        rg_all, cls_all = z["row_group"], z["market_class"]
    assert (rg_all == np.arange(len(rg_all))).all()
    fold_of = np.full(len(rg_all), -1)
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        fold_of[man["evaluation"]["valid"]] = k
    assert (fold_of >= 0).all()
    preds = {(fam, s, k): np.load(ROOT / "cache" / pat.format(s=s, k=k), mmap_mode="r")
             for fam, pat in FAMILIES.items() for s in (3, 4) for k in range(4)}
    tab = np.load(ROOT / "research/campaign_2026-10-01/data_eval/target_law_tables.npz")
    mult = tab["mult"]
    order = np.argsort(tab["spread_code"])
    sp_codes, sp_ticks = tab["spread_code"][order], tab["spread_ticks"][order]
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    rows: dict[str, list] = {}
    seq_stats: list[dict] = []
    mask_by_since = np.zeros((4, 16, 2))  # class, log2 since_E bin, [rows, scored]
    for g in range(len(rg_all)):
        t = pf.read_row_group(g, columns=COLS, use_threads=False)
        col = {c: t.column(c).to_numpy() for c in COLS}
        px0 = np.column_stack([col[c] for c in PX0])
        e = col["a6"].astype(np.float32) == ENDPOINT
        e[0] = False
        cc = changed(px0)
        q = np.rint(px0.astype(np.float64) * 1e4).astype(np.int64)
        with np.errstate(over="ignore"):
            h = (q * mult[None]).sum(1)
            h ^= h >> np.int64(29)
        pos = np.minimum(np.searchsorted(sp_codes, h), len(sp_codes) - 1)
        spread = np.where(sp_codes[pos] == h, sp_ticks[pos], -1).astype(np.int16)
        tr0 = changed(np.column_stack([col[c] for c in DV0]))
        vol0 = changed(np.column_stack([col[c] for c in V0]))
        px1 = changed(np.column_stack([col[c] for c in PX1]))
        tr1 = changed(np.column_stack([col[c] for c in DV1]))
        run = np.cumsum(e)
        s_e = since(e)
        u_e = until(e)
        m = mask_all[g].copy()
        n = len(m)
        cls = int(cls_all[g])
        b = np.minimum(np.floor(np.log2(s_e[99:] + 1)).astype(int), 15)
        np.add.at(mask_by_since[cls, :, 0], b, 1)
        np.add.at(mask_by_since[cls, :, 1], b, m[99:])
        # mask runs
        ms = np.r_[False, m, False].astype(np.int8)
        starts = np.flatnonzero(np.diff(ms) == 1)
        ends = np.flatnonzero(np.diff(ms) == -1)
        mrun_len = np.zeros(n, np.int32)
        mrun_pos = np.zeros(n, np.int32)
        for a, z_ in zip(starts, ends, strict=True):
            mrun_len[a:z_] = z_ - a
            mrun_pos[a:z_] = np.arange(z_ - a)
        sel = np.flatnonzero(m)
        k = fold_of[g]
        d = {
            "seq": np.full(len(sel), g, np.int16), "t": sel.astype(np.int16), "cls": np.full(len(sel), cls, np.int8),
            "y0": y_all[g, sel, 0], "y1": y_all[g, sel, 1],
            "E": e[sel], "since_E": s_e[sel].astype(np.int32), "run": run[sel].astype(np.int32),
            "nc_until_E": u_e[sel].astype(np.int32),
            "since_cc": since(cc)[sel].astype(np.int32), "since_tr0": since(tr0)[sel].astype(np.int32),
            "since_vol0": since(vol0)[sel].astype(np.int32), "since_px1": since(px1)[sel].astype(np.int32),
            "since_tr1": since(tr1)[sel].astype(np.int32),
            "run_cc": in_run_count(cc & ~e, run)[sel].astype(np.int32),
            "run_tr0": in_run_count(tr0, run)[sel].astype(np.int32),
            "run_px1": in_run_count(px1, run)[sel].astype(np.int32),
            "run_vol0": in_run_count(vol0, run)[sel].astype(np.int32),
            "spread": spread[sel], "nc_mrun_len": mrun_len[sel], "nc_mrun_pos": mrun_pos[sel],
        }
        for a in AX:
            d[a] = col[a][sel].astype(np.float32)
        for fam in FAMILIES:
            p = 0.5 * (preds[(fam, 3, k)][g, sel].astype(np.float64) + preds[(fam, 4, k)][g, sel])
            d[f"{fam}0"], d[f"{fam}1"] = p[:, 0].astype(np.float32), p[:, 1].astype(np.float32)
            d[f"{fam}s3_0"] = np.asarray(preds[(fam, 3, k)][g, sel, 0], np.float32) if fam == "ft" else None
            d[f"{fam}s4_0"] = np.asarray(preds[(fam, 4, k)][g, sel, 0], np.float32) if fam == "ft" else None
        d = {kk: v for kk, v in d.items() if v is not None}
        d["ft_s3_1"] = np.asarray(preds[("ft", 3, k)][g, sel, 1], np.float32)
        d["ft_s4_1"] = np.asarray(preds[("ft", 4, k)][g, sel, 1], np.float32)
        for kk, v in d.items():
            rows.setdefault(kk, []).append(v)
        seq_stats.append({
            "seq": g, "cls": cls, "fold": int(k), "nE": int(e[99:].sum()), "ncc": int(cc[99:].sum()),
            "ntr0": int(tr0[99:].sum()), "npx1": int(px1[99:].sum()), "ntr1": int(tr1[99:].sum()),
            "nvol0": int(vol0[99:].sum()), "nE_warm": int(e[:99].sum()), "ncc_warm": int(cc[:99].sum()),
            "ntr0_warm": int(tr0[:99].sum()), "npx1_warm": int(px1[:99].sum()),
            "nscored": int(m.sum()), "nmruns": int(len(starts)),
            "spread_unknown": float((spread[99:] < 0).mean()),
            "spread_mean": float(spread[99:][spread[99:] >= 0].mean()) if (spread[99:] >= 0).any() else -1.0,
        })
        if g % 200 == 0:
            print(g, flush=True)
    out = {kk: np.concatenate(v) for kk, v in rows.items()}
    keys = list(seq_stats[0])
    for kk in keys:
        out["seq_" + kk] = np.array([s[kk] for s in seq_stats])
    out["mask_by_since"] = mask_by_since
    np.savez(out_path, **out)
    print("rows", len(out["seq"]))


if __name__ == "__main__":
    main()
