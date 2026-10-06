"""Read-only independent holdout scan (2026-10-05 second-pair-of-eyes review).

Recomputes pooled global weighted Pearson, per-class WP and a public-mix
weighted proxy on the 368-sequence holdout domain for a set of cached final
model predictions. No fitting, no uploads, no writes except stdout.

Usage:
    OMP_NUM_THREADS=4 python3 research/lb/indep_holdout_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import is_holdout  # noqa: E402

PUB143 = np.array([0.183, 0.160, 0.198, 0.459])


def load(name: str, idx: np.ndarray) -> np.ndarray:
    path = ROOT / "cache" / f"pred_{name}_valid.npy"
    return np.asarray(np.load(path, mmap_mode="r")[idx], dtype=np.float32)


def class_stats(pred, y, mask, classes):
    out = np.zeros((4, 2, 6))
    for c in range(4):
        sel = classes == c
        for t in range(2):
            yy = np.clip(y[sel][..., t], -2, 2)[mask[sel]].astype(np.float64)
            pp = np.clip(pred[sel][..., t], -2, 2)[mask[sel]].astype(np.float64)
            w = np.abs(yy)
            out[c, t] = [
                w.sum(), (w * yy).sum(), (w * pp).sum(),
                (w * yy * yy).sum(), (w * pp * pp).sum(), (w * yy * pp).sum(),
            ]
    return out


def wp(stats, class_factor=None):
    cf = np.ones(4) if class_factor is None else class_factor
    s = (stats * cf[:, None, None]).sum(0)
    r = []
    for t in range(2):
        n, sy, sp, syy, spp, syp = s[t]
        cov = syp / n - sy / n * sp / n
        r.append(cov / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)))
    return float(np.mean(r))


def main() -> None:
    vt = np.load(ROOT / "cache" / "valid_targets.npz", mmap_mode="r")
    idx = np.flatnonzero([is_holdout(int(s)) for s in vt["seq_ix"]])
    y = np.asarray(vt["y"][idx])
    mask = np.asarray(vt["mask"][idx])
    classes = np.asarray(vt["market_class"][idx])
    share = np.bincount(classes, minlength=4) / len(classes)
    mix = PUB143 / np.maximum(share, 1e-12)

    half = 0.5
    complete = {
        "combo_s4": lambda: half * load("c20261004_combo_s4_final_fp32", idx)
        + half * load("c20261004_combo_head_s4_final_fp32", idx),
        "ab85_ema6s4_gclpall": lambda: load("ab85_final_ema6_s4_gclpall", idx),
        "ab85_ema6s3_gclpall": lambda: load("ab85_final_ema6_s3_gclpall", idx),
        "ab85_onorm_s5_gclpall": lambda: load("ab85_final_onorm_s5_gclpall", idx),
        "ab83_ema6s4_gclp": lambda: load("ab83_final_ema6_s4_gclp", idx),
    }
    preds = {k: f() for k, f in complete.items()}
    preds["s4+0.5*ab85_ema6s4_gclpall"] = (
        0.5 * preds["combo_s4"] + 0.5 * preds["ab85_ema6s4_gclpall"]
    )
    for name, p in preds.items():
        st = class_stats(p, y, mask, classes)
        per_class = [round(wp(st[c : c + 1]), 4) for c in range(4)]
        print(
            f"{name:26s} holdout {wp(st):.5f}  publicmix {wp(st, mix):.5f}  "
            f"cls {per_class}"
        )
    print("holdout class share", np.round(share, 4).tolist(), "public mix", PUB143.tolist())


if __name__ == "__main__":
    main()
