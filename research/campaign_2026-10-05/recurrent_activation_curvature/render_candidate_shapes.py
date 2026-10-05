"""Plot registered candidate maps and analytic derivatives, without loading data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Save shapes on the complete possible tanh-candidate domain."""
    u = np.linspace(-1.0, 1.0, 2001)
    maps = {
        "Baseline": (u, np.ones_like(u)),
        "Sine correction": (u + 0.1 * np.sin(np.pi * u), 1 + 0.1 * np.pi * np.cos(np.pi * u)),
        "Matched-center cubic": (
            u + 0.1 * np.pi * u * (1 - u**2),
            1 + 0.1 * np.pi * (1 - 3 * u**2),
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label, (candidate, derivative) in maps.items():
        axes[0].plot(u, candidate, label=label)
        axes[1].plot(u, candidate - u, label=label)
        axes[2].plot(u, derivative, label=label)
    for axis, title, ylabel in zip(
        axes,
        ("Candidate map", "Correction to tanh", "Local slope redistribution"),
        ("n(u)", "n(u) − u", "dn/du"),
        strict=True,
    ):
        axis.set(title=title, xlabel="u = tanh(v), necessarily in [−1, 1]", ylabel=ylabel)
        axis.axvline(0, color="#aaaaaa", linewidth=0.6)
        axis.grid(alpha=0.2)
    axes[2].axhline(0, color="#aaaaaa", linewidth=0.6)
    axes[0].legend(fontsize=8)
    fig.suptitle("E481 registered shapes: matched center and endpoints; endpoint slopes differ")
    fig.tight_layout()
    output = Path(__file__).resolve().parent / "eda"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "candidate_maps_and_slopes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
