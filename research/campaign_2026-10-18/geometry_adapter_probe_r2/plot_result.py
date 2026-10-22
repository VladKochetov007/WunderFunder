"""Plot the completed frozen-adapter comparison without a transfer claim."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKET = Path(__file__).resolve().parent


def main() -> None:
    """Save official scores and the conditional component interval together."""
    result = json.loads((PACKET / "comparison.json").read_text())["adapter_vs_parent"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(
        ["Fixed parent", "New-column adapter"],
        [result["reference_wp"], result["candidate_wp"]],
        color=["#777777", "#2671c6"],
    )
    axes[0].set(
        ylim=(0.678, 0.683),
        ylabel="Official WP",
        title="Exposed F0: 468 sequences / 141 components",
    )
    mean = result["mean_diff"]
    lower, upper = result["ci95"]
    axes[1].errorbar(mean, 0, xerr=[[mean - lower], [upper - mean]], fmt="o", capsize=5)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(
        yticks=[], xlabel="Adapter minus parent WP", title="Fixed-fit component 95% interval"
    )
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle(
        "Old weights preserved bitwise; appended weights learned\nOne seed, adaptive followup; interval excludes seed and public-transfer uncertainty",
        fontsize=10,
    )
    figure.savefig(PACKET / "adapter_result.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
