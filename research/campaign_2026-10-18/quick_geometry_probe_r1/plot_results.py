"""Plot the measured quick-probe training and conditional score contrasts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PACKET = Path(__file__).resolve().parent
PATTERN = re.compile(r"^ep 0 batch (\d+)/\d+ loss ([\d.e+-]+)")


def main() -> None:
    """Save one self-contained evidence figure after the paired grade lands."""
    result = json.loads((PACKET / "comparison.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for arm, color in (("feature", "#2671c6"), ("control", "#d16d26")):
        points = [
            match.groups()
            for line in (PACKET / f"{arm}.train.log").read_text().splitlines()
            if (match := PATTERN.match(line))
        ]
        x = np.asarray([int(point[0]) for point in points]) * 10
        y = np.asarray([float(point[1]) for point in points])
        axes[0].plot(x, y, alpha=0.35, color=color)
        if len(y) >= 5:
            axes[0].plot(
                x[4:],
                np.convolve(y, np.ones(5) / 5, mode="valid"),
                color=color,
                label=f"{arm}: 5-point moving mean",
            )
    axes[0].set(
        xlabel="Completed optimizer updates (logged batches)",
        ylabel="Training loss, including auxiliary terms",
        title="One short warm continuation per arm",
    )
    axes[0].legend(fontsize=8)
    labels = {
        "feature_vs_control": "Features − zero control",
        "feature_vs_parent": "Features − parent",
        "control_vs_parent": "Zero control − parent",
    }
    for index, (key, label) in enumerate(labels.items()):
        contrast = result["comparisons"][key]
        center = contrast["mean_diff"]
        lower, upper = contrast["ci95"]
        axes[1].errorbar(
            center,
            index,
            xerr=[[center - lower], [upper - center]],
            fmt="o",
            capsize=4,
            label=label,
        )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(
        yticks=range(3),
        yticklabels=list(labels.values()),
        xlabel="Official WP difference",
        title="Paired component 95% intervals",
    )
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle(
        f"Exploratory F0, seed 3: {result['sequences']} sequences / {result['components']} components\nIntervals condition on these fixed fits; no seed or public-transfer uncertainty",
        fontsize=11,
    )
    figure.savefig(PACKET / "quick_probe_results.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
