from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PublicObservation:
    """One observed public submission runtime and score."""

    candidate: str
    recipe: str
    runtime_seconds: int
    runtime_display: str
    score: float
    portal_id: str
    submission_filename: str


def read_observations(csv_path: Path) -> list[PublicObservation]:
    """Read and validate the fixed three-row public comparison input."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = ["A", "B", "C"]
    if [row["candidate"] for row in rows] != expected:
        raise ValueError("the plot requires exactly the frozen A, B, C observations")
    observations = [
        PublicObservation(
            candidate=row["candidate"],
            recipe=row["recipe"],
            runtime_seconds=int(row["public_runtime_seconds"]),
            runtime_display=row["public_runtime_display"],
            score=float(row["public_score"]),
            portal_id=row["portal_submission_id"],
            submission_filename=row["submission_filename"],
        )
        for row in rows
    ]
    if len(observations) != 3 or len({item.portal_id for item in observations}) != 3:
        raise ValueError("expected three distinct public submissions")
    return observations


def render_plot(csv_path: Path, output_dir: Path) -> None:
    """Render the descriptive public runtime-versus-score scatter plot."""
    observations = read_observations(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"A": "#2563eb", "B": "#d97706", "C": "#15803d"}
    offsets = {"A": (10, -48), "B": (-35, -55), "C": (-205, 34)}
    figure, axis = plt.subplots(figsize=(10, 5.8), layout="constrained")
    axis.set_xlim(20, 40)
    axis.set_ylim(0.6650, 0.6677)
    axis.set_xticks([20, 25, 30, 35, 40])
    axis.set_yticks([0.6650, 0.6655, 0.6660, 0.6665, 0.6670, 0.6675])
    axis.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_xlabel("Observed public execution time (minutes)")
    axis.set_ylabel("Observed public score (zoomed vertical scale)")
    axis.set_title(
        "Public score versus execution time",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )
    axis.text(
        0.0,
        1.015,
        "Three native-Q8 submissions · same competition CPU environment",
        transform=axis.transAxes,
        color="#475569",
        fontsize=10,
        va="bottom",
    )
    display_recipes = {
        "A": "H390 s3/s4 (50:50)",
        "B": "H390 s3 + Z4 (50:50)",
        "C": "H390 s3/s4/Z4 (25/25/50)",
    }
    for item in observations:
        x_value = item.runtime_seconds / 60
        label = (
            f"{item.candidate} · {display_recipes[item.candidate]}\n"
            f"{item.runtime_display} | {item.score:.4f} · {item.portal_id}"
        )
        axis.scatter(
            x_value,
            item.score,
            s=100,
            color=colors[item.candidate],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        axis.annotate(
            label,
            xy=(x_value, item.score),
            xytext=offsets[item.candidate],
            textcoords="offset points",
            ha="left" if item.candidate == "A" else "right",
            va="center",
            fontsize=9,
            color="#0f172a",
            arrowprops={"arrowstyle": "-", "color": "#64748b", "linewidth": 0.8},
            annotation_clip=False,
        )
    axis.text(
        0.5,
        -0.18,
        "Only three observed submissions; blend composition changes with runtime. "
        "Descriptive association, not causal evidence or a forecast.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )
    figure.savefig(output_dir / "PUBLIC_RUNTIME_VS_SCORE_R1.png", dpi=200)
    figure.savefig(output_dir / "PUBLIC_RUNTIME_VS_SCORE_R1.svg")
    plt.close(figure)


def main() -> None:
    """Render the frozen public-observation plot from its adjacent CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_plot(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
