"""Render every fixed mechanism contrast from the immutable score artifact."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Keep point estimates and percentile intervals visually distinct."""
    here = Path(__file__).parent
    result = json.loads((here / "RESULT.json").read_text())
    output = here / "eda"
    output.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 7))
    for index, (_name, contrast) in enumerate(result["contrasts"].items()):
        axis.hlines(index, *contrast["ci95"], color="#3973ac")
        axis.plot(contrast["mean_diff"], index, "o", color="#3973ac")
    axis.set_yticks(np.arange(14), list(result["contrasts"]), fontsize=7)
    axis.axvline(0, color="gray", linestyle="--")
    axis.invert_yaxis()
    axis.set_xlabel("Pooled WP difference; conditional component 95% interval")
    axis.set_title("All 14 fixed contrasts: one fit seed, exposed small-model panel")
    fig.tight_layout()
    fig.savefig(output / "all_mechanism_contrasts.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
