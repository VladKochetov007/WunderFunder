from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def draw(eda: dict, ridge: dict, output: Path) -> None:
    """Plot measured exploratory results without smoothing or fitted trend lines."""
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    variants = ridge["variants"]
    scored = [item for item in variants if item["status"] == "scored"]
    labels = [item["variant"].replace("raw_", "+ ").replace("_", " ") for item in scored]
    for target, offset, color in (("t0", -0.15, "#2878b5"), ("t1", 0.15, "#e37b32")):
        values = [item["official_metric"][target] for item in scored]
        axes[0].barh(
            np.arange(len(scored)) + offset, values, height=0.28, label=target, color=color
        )
    axes[0].set_yticks(np.arange(len(scored)), labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("Official clipped weighted Pearson")
    axes[0].set_title("Fixed ridge probes: 6 exposed evaluation components")
    axes[0].legend()
    horizon = eda["target_operator"]["horizons_rows"]
    pooled = eda["target_operator"]["pooled_fit_components"]
    for family, style in (("i0_p", "-"), ("i0_dp", "--")):
        for target, color in ((0, "#2878b5"), (1, "#e37b32")):
            values = [
                pooled[f"{family}_mean_delta_h{h}"][f"target_{target}"][
                    "magnitude_weighted_pearson"
                ]
                for h in horizon
            ]
            axes[1].plot(
                horizon,
                values,
                marker="o",
                linestyle=style,
                color=color,
                label=f"{family}, t{target}",
            )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(horizon, [str(item) for item in horizon])
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Prespecified future displacement horizon (rows)")
    axes[1].set_ylabel("Diagnostic association (not official WP)")
    axes[1].set_title("Fit-only label/operator association; no identified horizon")
    axes[1].legend()
    fig.suptitle("Bounded idea screen: 32 fit / 16 evaluation sequences; no public-score inference")
    for suffix in ("png", "svg"):
        path = output / f"screens.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Render a standalone compact figure from source-bound result JSON files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", type=Path, required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.eda.read_bytes()), json.loads(args.ridge.read_bytes()), args.output)


if __name__ == "__main__":
    main()
