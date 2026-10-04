"""Scout falsifier F3b: event-exact matured-residual correction (upper bound for online learning on matured labels).

Segments = rows between consecutive a6 lower-endpoint events. Row r's target depends on the next N events after
r, so at row t the newest matured segment is the one that ended N events ago. Feature at t: |y|-weighted mean
residual (y - beta*p) of the newest matured segment, plus an EMA over matured segments. True y stands in for
feature-computed labels (upper bound). Correction p + kappa*feature, kappa chosen on the other folds.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_scout_matured_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import wp_from_sums  # noqa: E402

MEMBERS = ("c20260929_ft_since_s3_f{k}", "c20260929_ft_since_s4_f{k}")
N_MATURE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
EMA = 0.2
KAPPAS = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0])


def matured_features(y, p, ev, beta):
    """Per row: newest matured segment residual and EMA of matured segment residuals (one target, one sequence)."""
    seg = np.cumsum(ev)  # segment id; an event row opens a new segment
    yc = np.clip(y, -2, 2)
    w = np.abs(yc)
    r = yc - beta * np.clip(p, -2, 2)
    n_seg = seg[-1] + 1
    ws = np.bincount(seg, weights=w, minlength=n_seg)
    rs = np.bincount(seg, weights=w * r, minlength=n_seg)
    mean_r = np.where(ws > 0, rs / np.maximum(ws, 1e-12), np.nan)
    ema = np.zeros(n_seg)
    last = np.zeros(n_seg)
    e, l = 0.0, 0.0
    for s in range(n_seg):
        if not np.isnan(mean_r[s]):
            e = (1 - EMA) * e + EMA * mean_r[s]
            l = mean_r[s]
        ema[s], last[s] = e, l
    src = seg - N_MATURE
    ok = src >= 0
    f_last = np.where(ok, last[np.maximum(src, 0)], 0.0)
    f_ema = np.where(ok, ema[np.maximum(src, 0)], 0.0)
    return f_last, f_ema


def wsums(y, q):
    y, q = np.clip(y, -2, 2), np.clip(q, -2, 2)
    w = np.abs(y)
    return np.array([w.sum(), (w * y).sum(), (w * q).sum(), (w * y * y).sum(), (w * q * q).sum(), (w * y * q).sum()])


def main() -> None:
    z = np.load(ROOT / "cache/valid_targets.npz")
    row_group, cls_all = z["row_group"], z["market_class"]
    f = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    pos = {int(g): i for i, g in enumerate(row_group)}
    per_fold = {}
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        idx = np.array(sorted(pos[int(g)] for g in man["evaluation"]["valid"]))
        p = sum(np.load(ROOT / f"cache/pred_{n.format(k=k)}_valid.npy", mmap_mode="r")[idx].astype(np.float64)
                for n in MEMBERS) / len(MEMBERS)
        y = z["y"][idx].astype(np.float64)
        m = z["mask"][idx]
        cls = cls_all[idx]
        a6 = np.stack([f.read_row_group(int(row_group[i]), columns=["a6"]).column("a6").to_numpy() for i in idx])
        ev = np.isclose(a6, -5.1993, atol=1e-3)
        # per-class beta on this fold is label-using but only rescales the residual definition; kappa is cross-fitted
        betas = {}
        for c in range(4):
            for j in range(2):
                sel = (cls == c)[:, None] & m
                yy, pp = np.clip(y[..., j][sel], -2, 2), np.clip(p[..., j][sel], -2, 2)
                w = np.abs(yy)
                betas[c, j] = (w * yy * pp).sum() / (w * pp * pp).sum()
        F = np.zeros((2,) + p.shape)
        for i in range(len(idx)):
            for j in range(2):
                fl, fe = matured_features(y[i, :, j], p[i, :, j], ev[i], betas[cls[i], j])
                F[0, i, :, j], F[1, i, :, j] = fl, fe
        # sums per class, kappa, feature kind
        S = np.zeros((2, 4, len(KAPPAS), 2, 6))
        corr = np.zeros((2, 4, 2))
        for c in range(4):
            sel = (cls == c)[:, None] & m
            for j in range(2):
                yy, pp = y[..., j][sel], p[..., j][sel]
                for kind in range(2):
                    ff = F[kind][..., j][sel]
                    rr = np.clip(yy, -2, 2) - betas[c, j] * np.clip(pp, -2, 2)
                    w = np.abs(np.clip(yy, -2, 2))
                    mr, mf = np.average(rr, weights=w), np.average(ff, weights=w)
                    corr[kind, c, j] = np.average((rr - mr) * (ff - mf), weights=w) / np.sqrt(
                        np.average((rr - mr) ** 2, weights=w) * np.average((ff - mf) ** 2, weights=w) + 1e-30)
                    for a, kap in enumerate(KAPPAS):
                        S[kind, c, a, j] = wsums(yy, pp + kap * ff)
        per_fold[k] = {"S": S, "corr": corr}
        print(k, "corr(resid, matured feature) [last, ema] x class x target:", np.round(corr, 3).tolist(), flush=True)
    # cross-fitted kappa per class and feature kind; deltas on held fold vs kappa=0
    k0 = list(KAPPAS).index(0.0)
    out = {"n_mature": N_MATURE, "ema": EMA, "kappas": KAPPAS.tolist(), "folds": {}}
    for k in range(4):
        res = {}
        for kind, name in enumerate(("last", "ema")):
            base_s = per_fold[k]["S"][kind][:, k0].sum(0)  # [2,6] summed over classes
            base = float(wp_from_sums(base_s).mean())
            deltas = {}
            for c in range(4):
                other = sum(np.array([float(wp_from_sums(per_fold[j]["S"][kind][:, k0].sum(0)
                                                          - per_fold[j]["S"][kind][c, k0]
                                                          + per_fold[j]["S"][kind][c, a]).mean())
                                      for a in range(len(KAPPAS))]) for j in range(4) if j != k)
                a = int(np.argmax(other))
                held = base_s - per_fold[k]["S"][kind][c, k0] + per_fold[k]["S"][kind][c, a]
                deltas[c] = {"kappa": float(KAPPAS[a]), "delta": float(wp_from_sums(held).mean()) - base}
            res[name] = {"base": base, "per_class": deltas,
                         "corr": per_fold[k]["corr"][kind].tolist()}
        out["folds"][k] = res
        print(k, json.dumps({n: {c: (d["kappa"], round(d["delta"], 5)) for c, d in r["per_class"].items()}
                             for n, r in res.items()}))
    out_path = ROOT / f"research/campaign_2026-10-01/creative/cr_scout/probe_matured_n{N_MATURE}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
