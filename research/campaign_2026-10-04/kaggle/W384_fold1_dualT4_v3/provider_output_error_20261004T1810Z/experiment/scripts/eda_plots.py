"""Exploratory figures from cache/seqstats_valid.parquet and cached targets.

    scripts/limited .venv/bin/python scripts/eda_plots.py
Writes PNGs to research/visual/eda/. Classes are faceted (small multiples) so
each panel carries one series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from wnn.gmetric import global_wp
from wnn.metric import block_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "visual" / "eda"
BLUE, INK, MUTED, ORANGE = "#2a78d6", "#0b0b0b", "#8a8984", "#eb6834"
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def arcs(df):
    cols = ["i0_p0", "i0_p11", "i0_p4", "i0_v0", "i0_dp0", "i0_dv0", "i1_p0", "a0", "a5", "a7"]
    fig, ax = plt.subplots(len(cols), 4, figsize=(13, 2.3 * len(cols)), sharex="row", sharey="row")
    cls = df["cls"].to_numpy()
    for i, c in enumerate(cols):
        mu, sd = df[f"{c}_mean"].to_numpy(), df[f"{c}_std"].to_numpy()
        a, b = df[f"{c}_a"].to_numpy(), df[f"{c}_b"].to_numpy()
        # most common global pair of top values for the theory arc
        va, vb = np.median(a), np.median(b)
        lo, hi = min(va, vb), max(va, vb)
        g = np.linspace(lo, hi, 200)
        for k in range(4):
            s = cls == k
            ax[i, k].scatter(mu[s], sd[s], s=4, color=BLUE, alpha=0.5, lw=0)
            ax[i, k].plot(g, np.sqrt(np.maximum((g - lo) * (hi - g), 0)), color=INK, lw=1, ls="--")
            ax[i, k].set_title(f"{c}  class {k}", fontsize=8)
    fig.suptitle("Per-sequence mean vs std; dashed = two-state arc sqrt((mu-a)(b-mu)) from median top-2 values", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "01_arcs.png", dpi=110)


def joint(y, m, cls):
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
    r = np.arange(-7, 8)
    for k in range(4):
        s = cls == k
        yy = np.rint(y[s][:, 99:] / QUANTA[k]).astype(int).reshape(-1, 2)[::5]
        for row, sel in enumerate([np.ones(len(yy), bool), m[s][:, 99:].reshape(-1)[::5]]):
            h, _, _ = np.histogram2d(yy[sel, 0], yy[sel, 1], bins=[np.arange(-7.5, 8), np.arange(-7.5, 8)])
            h = h / h.sum()
            a = ax[row, k]
            a.imshow(np.log10(h.T + 1e-5), origin="lower", extent=[-7.5, 7.5, -7.5, 7.5], cmap="Blues", vmin=-5, vmax=-0.3)
            a.plot(r, -r - 1, color=ORANGE, lw=0.8, ls="--")
            a.set_title(f"class {k} {'all rows' if row == 0 else 'scored rows'}", fontsize=8)
            a.set_xlabel("t0 / quantum")
            a.set_ylabel("t1 / quantum")
            a.grid(False)
    fig.suptitle("Joint target distribution in quanta (log share); dashed: t0 + t1 = -1", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "02_target_joint.png", dpi=110)


def level(df):
    lv = ((df["t0_wmean_sc"] + df["t1_wmean_sc"]) / 2).to_numpy()
    cls = df["cls"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for k in range(4):
        s = cls == k
        ax[0].hist(lv[s], bins=40, histtype="step", lw=1.4, color=[BLUE, ORANGE, "#1baf7a", "#4a3aa7"][k], label=f"class {k}")
    ax[0].set_xlabel("per-seq |y|-weighted scored target level ((t0+t1)/2)")
    ax[0].legend(frameon=False)
    ax[1].scatter(np.log10(df["n_states_i0"].to_numpy()), lv, s=4, color=BLUE, alpha=0.5, lw=0)
    ax[1].set_xlabel("log10 distinct i0 book states in sequence")
    ax[1].set_ylabel("level")
    ax[2].scatter(df["ev_i0_both"].to_numpy(), lv, s=4, color=BLUE, alpha=0.5, lw=0)
    ax[2].set_xlabel("joint i0 bid+ask change rate")
    fig.tight_layout()
    fig.savefig(OUT / "03_level.png", dpi=110)


def scale_curve(y, m):
    p = np.load(ROOT / "cache" / "pred_baseline_valid.npy")
    scales = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    g = [global_wp(y, s * p, m)["wp"] for s in scales]
    b = [np.nanmean(block_scores(y, s * p, m).mean(1)) for s in scales]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(scales, g, color=BLUE, lw=2, marker="o", ms=5, label="global WP")
    ax.plot(scales, b, color=ORANGE, lw=2, marker="o", ms=5, label="per-sequence WP")
    ax.set_xscale("log")
    ax.set_xlabel("multiplier on baseline predictions")
    ax.set_ylabel("validation WP")
    ax.legend(frameon=False)
    ax.set_title("Clip at +-2 punishes the baseline's scale", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "04_scale_curve.png", dpi=110)
    return dict(zip(scales.tolist(), g)), dict(zip(scales.tolist(), b))


def main():
    df = pl.read_parquet(ROOT / "cache" / "seqstats_valid.parquet").sort("row_group")
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    y, m, cls = d["y"], d["mask"], d["market_class"]
    arcs(df)
    joint(y, m, cls)
    level(df)
    print(scale_curve(y, m))


if __name__ == "__main__":
    main()
