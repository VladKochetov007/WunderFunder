"""Plot complete fixed-step training telemetry; no evaluation data is read."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Save coefficient trajectories for every learned cell and both layers."""
    here = Path(__file__).resolve().parent
    receipt = json.loads((here / "FIT_RECEIPT.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, seed in enumerate(receipt["seeds"]):
        for j, optimizer in enumerate(receipt["optimizers"]):
            axis = axes[i, j]
            for arm in receipt["arms"]:
                if arm["seed"] != seed or arm["optimizer"] != optimizer:
                    continue
                if not arm["condition"].startswith("learned"):
                    continue
                rows = arm["telemetry"]
                steps = [row["update_index_zero_based"] for row in rows]
                coefficients = np.asarray([row["effective_alpha"] for row in rows])
                for layer in range(2):
                    axis.plot(
                        steps,
                        coefficients[:, layer],
                        marker="o",
                        linestyle="-" if layer == 0 else "--",
                        label=f"{arm['condition']}, layer {layer}",
                    )
            axis.axhline(0, color="#aaaaaa", linewidth=0.6)
            axis.axhline(0.1, color="#aaaaaa", linewidth=0.6)
            axis.set(title=f"Seed {seed}, {optimizer}", ylim=(-0.015, 0.115))
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Zero-based update index (coefficient after update)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Effective coefficient, 0.2 tanh(raw_alpha)")
    fig.suptitle("E479: all learned-shape training trajectories; no test feedback")
    fig.tight_layout()
    output = here / "eda"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "coefficient_trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
