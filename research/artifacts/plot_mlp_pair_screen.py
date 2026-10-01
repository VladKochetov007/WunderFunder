"""Render the fold-0 MLP partner replacement screen from its JSON result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "research/artifacts/ensemble_scan_mlp_f0.json"
OUTPUT_DIR = ROOT / "artifacts/plots"


def main() -> None:
    result = json.loads(RESULT_PATH.read_text())
    screen = result["sets"]["mlp_second"]
    labels = ["λ = 0", "λ = 0.02", "λ = 0.05"]
    keys = ["lam0", "lam0.02", "lam0.05"]
    deltas = np.array([screen[key]["delta"] for key in keys]) * 1_000
    errors = np.array([screen[key]["sd"] for key in keys]) * 1_000

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    x = np.arange(len(labels))
    ax.axhline(0, color="#4b5563", linewidth=1.1, zorder=1)
    ax.errorbar(
        x,
        deltas,
        yerr=errors,
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.6,
        color="#0072B2",
        ecolor="#56B4E9",
        zorder=3,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Δ global weighted Pearson (×10⁻³)")
    ax.set_title("MLP as the second ensemble member")
    ax.text(
        0.02,
        0.97,
        f"Fold 0 · {screen['n']} pure-chain sequences · paired bootstrap SD",
        transform=ax.transAxes,
        va="top",
        color="#374151",
        fontsize=9,
    )
    ax.text(
        0.02,
        0.04,
        "Equal-weight replacement: ab68 + MLP vs ab68 + ab73\n"
        f"Train-mix λ = 0 delta: {screen['train_mix_lam0_delta'] * 1_000:+.3f} ×10⁻³",
        transform=ax.transAxes,
        color="#374151",
        fontsize=8.5,
        va="bottom",
    )
    ax.set_ylim(-3.5, 3.5)
    ax.margins(x=0.22)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "mlp_pair_screen.png")
    fig.savefig(OUTPUT_DIR / "mlp_pair_screen.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
