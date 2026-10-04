"""Scout falsifier F4: WP-deficit ledger over causal case cells, with cross-fitted per-cell scale overlays.

Cells = class x rows-since-last-event x step-in-sequence x causal event-rate bucket (all label-free; events are
rows where a6 sits at its lower endpoint). For each fold and cell we keep the six |y|-weighted raw sums, so any
per-cell scale alpha is exact and instant. Factors are chosen on the other three folds and applied to the held fold.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_scout_cell_ledger.py
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

OUT = ROOT / "research/campaign_2026-10-01/creative/cr_scout/probe_cells.json"
MEMBERS = ("c20260929_ft_since_s3_f{k}", "c20260929_ft_since_s4_f{k}")
SINCE_EDGES = np.array([1, 4, 16, 64, 256])  # rows since last event: 0 | 1-3 | 4-15 | 16-63 | 64-255 | 256+
STEP_EDGES = np.array([1000, 5000])  # 99-999 | 1000-4999 | 5000+
RATE_WIN = 2000
ALPHAS = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6])


def cell_ids(a6: np.ndarray, cls: int, rate_edges: np.ndarray) -> np.ndarray:
    ev = np.isclose(a6, A6_LO)
    idx = np.arange(len(a6))
    last = np.maximum.accumulate(np.where(ev, idx, -1))
    since = np.where(last >= 0, idx - last, 10**6)
    c = np.cumsum(ev)
    lag = np.r_[np.zeros(RATE_WIN, dtype=c.dtype), c[:-RATE_WIN]]
    rate = (c - lag) / np.minimum(idx + 1, RATE_WIN)
    s = np.searchsorted(SINCE_EDGES, since, side="right")
    st = np.searchsorted(STEP_EDGES, idx, side="right")
    r = np.searchsorted(rate_edges, rate, side="right")
    return ((cls * 6 + s) * 3 + st) * 3 + r


A6_LO = None


def main() -> None:
    global A6_LO
    z = np.load(ROOT / "cache/valid_targets.npz")
    row_group, cls_all = z["row_group"], z["market_class"]
    f = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    a6 = np.stack([f.read_row_group(int(g), columns=["a6"]).column("a6").to_numpy() for g in row_group])
    A6_LO = float(a6.min())
    # label-free event-rate edges (tertiles over all valid rows)
    ev = np.isclose(a6, A6_LO)
    c = np.cumsum(ev, axis=1)
    lag = np.concatenate([np.zeros((len(c), RATE_WIN), c.dtype), c[:, :-RATE_WIN]], axis=1)
    rate = (c - lag) / np.minimum(np.arange(c.shape[1]) + 1, RATE_WIN)
    rate_edges = np.quantile(rate[:, 99:], [1 / 3, 2 / 3])
    cells = np.stack([cell_ids(a6[i], int(cls_all[i]), rate_edges) for i in range(len(a6))])
    n_cells = 4 * 6 * 3 * 3
    pos = {int(g): i for i, g in enumerate(row_group)}
    S = np.zeros((4, n_cells, 2, 6))
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        idx = np.array(sorted(pos[int(g)] for g in man["evaluation"]["valid"]))
        p = sum(np.load(ROOT / f"cache/pred_{n.format(k=k)}_valid.npy", mmap_mode="r")[idx].astype(np.float64)
                for n in MEMBERS) / len(MEMBERS)
        y = np.clip(z["y"][idx].astype(np.float64), -2, 2)
        p = np.clip(p, -2, 2)
        m = z["mask"][idx]
        cc = cells[idx][m]
        for j in range(2):
            yy, pp = y[..., j][m], p[..., j][m]
            w = np.abs(yy)
            for col, v in enumerate((w, w * yy, w * pp, w * yy * yy, w * pp * pp, w * yy * pp)):
                S[k, :, j, col] = np.bincount(cc, weights=v, minlength=n_cells)

    def score(sums):  # sums: [cells, 2, 6]
        return float(wp_from_sums(sums.sum(0)).mean())

    def scaled(sums, cell, a):
        t = sums.copy()
        t[cell, :, 2] *= a
        t[cell, :, 4] *= a * a
        t[cell, :, 5] *= a
        return t

    base = [score(S[k]) for k in range(4)]
    ledger = []
    for cell in range(n_cells):
        wshare = S[:, cell, :, 0].sum() / S[:, :, :, 0].sum()
        if wshare < 0.005:
            continue
        tot = S[:, cell].sum(0)
        corr = wp_from_sums(tot).mean()
        deltas = np.array([[score(scaled(S[k], cell, a)) - base[k] for a in ALPHAS] for k in range(4)])
        held = []
        for k in range(4):
            best = ALPHAS[np.argmax(np.delete(deltas, k, axis=0).sum(0))]
            held.append(float(deltas[k, list(ALPHAS).index(best)]))
        cls, rest = divmod(cell, 54)
        s, rest = divmod(rest, 9)
        st, r = divmod(rest, 3)
        ledger.append({"cell": cell, "class": cls, "since_bucket": s, "step_bucket": st, "rate_bucket": r,
                       "weight_share": float(wshare), "wp_in_cell": float(corr),
                       "crossfit_delta_per_fold": held, "crossfit_delta_mean": float(np.mean(held))})
    ledger.sort(key=lambda d: -d["crossfit_delta_mean"])
    # joint cross-fitted overlay: per held fold, greedily pick alphas on the other folds (coordinate ascent, 2 sweeps)
    joint = []
    for k in range(4):
        others = [j for j in range(4) if j != k]
        alpha = np.ones(n_cells)
        for _ in range(2):
            for cell in range(n_cells):
                if S[others, cell, :, 0].sum() <= 0:
                    continue
                best, best_v = alpha[cell], -1e9
                for a in ALPHAS:
                    trial = alpha.copy()
                    trial[cell] = a
                    v = sum(score(apply(S[j], trial)) for j in others)
                    if v > best_v:
                        best, best_v = a, v
                alpha[cell] = best
        joint.append(score(apply(S[k], alpha)) - base[k])
    out = {"base": base, "joint_crossfit_delta": joint, "rate_edges": rate_edges.tolist(), "ledger": ledger}
    OUT.write_text(json.dumps(out, indent=1))
    print("base", np.round(base, 5), "joint cross-fitted per-cell scale delta", np.round(joint, 5))
    print("class since step rate  w%   wp    held-out deltas")
    for d in ledger[:12] + ledger[-5:]:
        print(d["class"], d["since_bucket"], d["step_bucket"], d["rate_bucket"], f"{100 * d['weight_share']:5.1f}",
              f"{d['wp_in_cell']:.3f}", np.round(d["crossfit_delta_per_fold"], 5))
    by_class = {}
    for d in ledger:
        by_class.setdefault(d["class"], []).append(d)
    for c, ds in sorted(by_class.items()):
        print("class", c, "lowest-wp heavy cells:",
              [(d["since_bucket"], d["step_bucket"], d["rate_bucket"], round(100 * d["weight_share"], 1),
                round(d["wp_in_cell"], 3)) for d in sorted(ds, key=lambda d: d["wp_in_cell"])[:4]])


def apply(sums, alpha):
    t = sums.copy()
    t[:, :, 2] *= alpha[:, None]
    t[:, :, 4] *= (alpha**2)[:, None]
    t[:, :, 5] *= alpha[:, None]
    return t


if __name__ == "__main__":
    main()
