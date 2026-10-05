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
    labels = []
    names = {
        "gelu": "GELU",
        "gelu_plus_sine_01": "GELU+sine",
        "gelu_plus_linear_01": "GELU+linear",
        "scaled_gelu_11": "1.1*GELU",
        "jax_tanh_gelu_plus_sine_01": "Approx GELU+sine",
    }
    for key in result["contrasts"]:
        candidate, reference = key.split("_minus_")
        candidate_name, optimizer = candidate.split("__")
        reference_name, _ = reference.split("__")
        optimizer_label = "AdamW" if optimizer == "adamw" else "Muon"
        labels.append(f"{names[candidate_name]} − {names[reference_name]} / {optimizer_label}")
    axis.set_yticks(np.arange(14), labels, fontsize=9)
    axis.axvline(0, color="gray", linestyle="--")
    axis.invert_yaxis()
    axis.set_xlabel("Pooled WP difference; conditional component 95% interval")
    axis.set_title("All 14 fixed contrasts: one fit seed, exposed small-model panel")
    fig.tight_layout()
    fig.savefig(output / "all_mechanism_contrasts.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
