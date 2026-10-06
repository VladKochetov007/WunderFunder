"""Full i0/i1 volume snapshot on the long-quiet cell, against the frozen champion.

Touch-volume depletion already failed. This asks whether the rest of the book
shape still correlates with the champion's residual. Same split, alpha 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

from wave1_anatomy import ROOT, _changed, _since_last, _solve_ridge, _split_bit

OUT = Path(__file__).resolve().parent
VOL_NAMES = [f"i0_v{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]
COLS = ["i0_p0", "i0_p11", *VOL_NAMES]


def _wp(y: np.ndarray, p: np.ndarray) -> float:
    acc = GlobalAccumulator()
    acc.add(y[:, None, :], p[:, None, :], np.ones(y.shape[0], dtype=bool)[:, None])
    return float(acc.result()["wp"])


def main() -> None:
    cache = ROOT / "cache"
    with np.load(cache / "valid_targets.npz") as valid:
        y_all = valid["y"]
        mask = valid["mask"]
        seq_ix = valid["seq_ix"]
        row_group = valid["row_group"]
    parent = np.load(cache / "pred_c20261004_combo_s4_final_fp32_valid.npy", mmap_mode="r")
    head = np.load(cache / "pred_c20261004_combo_head_s4_final_fp32_valid.npy", mmap_mode="r")
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in ("fx", "fy", "fp", "ex", "ey", "ep")}
    for npz_index in range(y_all.shape[0]):
        if is_holdout(int(seq_ix[npz_index])):
            continue
        table = pf.read_row_group(int(row_group[npz_index]), columns=COLS, use_threads=False)
        p0 = table.column("i0_p0").to_numpy()
        p11 = table.column("i0_p11").to_numpy()
        or_ev = _changed(p0) | _changed(p11)
        age, _last = _since_last(or_ev)
        quiet = (~or_ev) & (age >= 32) & mask[npz_index]
        if not quiet.any():
            continue
        vols = np.column_stack([table.column(name).to_numpy() for name in VOL_NAMES])[quiet]
        yy = y_all[npz_index][quiet]
        pred = 0.5 * (
            parent[npz_index][quiet].astype(np.float64) + head[npz_index][quiet].astype(np.float64)
        )
        prefix = "f" if _split_bit(int(seq_ix[npz_index]), "wave1b") == 0 else "e"
        buckets[prefix + "x"].append(vols.astype(np.float32))
        buckets[prefix + "y"].append(yy.astype(np.float32))
        buckets[prefix + "p"].append(pred.astype(np.float32))
    x_fit = np.concatenate(buckets["fx"]).astype(np.float64)
    y_fit = np.concatenate(buckets["fy"]).astype(np.float64)
    p_fit = np.concatenate(buckets["fp"]).astype(np.float64)
    x_ev = np.concatenate(buckets["ex"]).astype(np.float64)
    y_ev = np.concatenate(buckets["ey"]).astype(np.float64)
    p_ev = np.concatenate(buckets["ep"]).astype(np.float64)
    w = np.abs(np.clip(y_fit, -2, 2)).mean(axis=1)
    mean = (x_fit * w[:, None]).sum(axis=0) / w.sum()
    var = ((x_fit - mean) ** 2 * w[:, None]).sum(axis=0) / w.sum()
    scale = np.sqrt(np.maximum(var, 1e-8))
    x_fit = (x_fit - mean) / scale
    x_ev = (x_ev - mean) / scale
    sw = np.sqrt(np.maximum(w, 0.0))
    design = np.column_stack([np.ones(x_fit.shape[0]), x_fit, p_fit]) * sw[:, None]
    beta = _solve_ridge(design.T @ design, design.T @ (y_fit * sw[:, None]), 1.0)
    pred = np.clip(
        np.column_stack([np.ones(x_ev.shape[0]), x_ev, p_ev]) @ beta, -2, 2
    ).astype(np.float32)
    incumbent = _wp(y_ev.astype(np.float32), p_ev.astype(np.float32))
    corrected = _wp(y_ev.astype(np.float32), pred)
    # Top residual correlations on the eval cell.
    residual = y_ev - p_ev
    we = np.abs(np.clip(y_ev, -2, 2)).mean(axis=1)
    sw_e = we / we.sum()
    ranked = []
    for j, name in enumerate(VOL_NAMES):
        col = x_ev[:, j]
        vals = []
        for t in range(2):
            a = col - np.sum(sw_e * col)
            b = residual[:, t] - np.sum(sw_e * residual[:, t])
            denom = np.sqrt(np.sum(sw_e * a * a) * np.sum(sw_e * b * b))
            vals.append(float(np.sum(sw_e * a * b) / denom) if denom > 1e-12 else 0.0)
        ranked.append((max(abs(vals[0]), abs(vals[1])), name, vals[0], vals[1]))
    ranked.sort(reverse=True)
    payload = {
        "cell": "quiet age>=32, 44 i0 volume columns + 44 i1 volume columns",
        "fit_rows": int(x_fit.shape[0]),
        "eval_rows": int(x_ev.shape[0]),
        "incumbent_wp": incumbent,
        "book_plus_incumbent_wp": corrected,
        "delta_vs_incumbent": corrected - incumbent,
        "top_residual_correlations": [
            {"feature": name, "t0": t0, "t1": t1} for _score, name, t0, t1 in ranked[:8]
        ],
    }
    (OUT / "wave1_book.json").write_text(json.dumps(payload, indent=2))
    print("BOOK_DONE", payload["delta_vs_incumbent"], ranked[0], flush=True)


if __name__ == "__main__":
    main()
