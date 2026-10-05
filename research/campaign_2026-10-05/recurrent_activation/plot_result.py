"""Display all preregistered recurrent contrasts without selecting winners."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Separate within-backend scientific comparisons from historical controls."""
    here = Path(__file__).parent
    result = json.loads((here / "RESULT.json").read_text())
    fig, axis = plt.subplots(figsize=(11, 6))
    labels = []
    for index, (key, item) in enumerate(result["contrasts"].items()):
        color = "#3973ac" if index < 7 else "#999999"
        axis.hlines(index, *item["ci95"], color=color)
        axis.plot(item["mean_diff"], index, "o", color=color)
        candidate, reference = key.split("_minus_")
        labels.append(f"{candidate} − {reference}".replace("muon_adamw", "Muon"))
    axis.set_yticks(range(len(labels)), labels, fontsize=9)
    axis.axvline(0, color="gray", linestyle="--")
    axis.invert_yaxis()
    axis.set_xlabel("Pooled WP difference; conditional component 95% interval")
    axis.set_title("Bounded recurrent candidate × optimizer: one exposed fit seed")
    fig.tight_layout()
    output = here / "eda"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "all_recurrent_contrasts.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
