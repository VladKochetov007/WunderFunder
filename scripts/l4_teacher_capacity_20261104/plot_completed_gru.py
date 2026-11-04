"""Plot the six completed teachers from their captured native histories and logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKET = (
    Path(__file__).resolve().parents[2] / "research/experiment_drafts/l4-teacher-capacity-20261104"
)
COLORS = {"gru256x2_control": "#2468ad", "gru384x2": "#dd7722", "gru512x1": "#238b60"}


def save_figure(figure: plt.Figure, name: str, output: Path) -> None:
    """Retain both directly viewable and vector scientific figures."""
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=160)
    plt.close(figure)


def main() -> None:
    """Plot measured values; seed ranges are descriptive, never confidence intervals."""
    source = PACKET / "GRU_CURVES_COMPLETE_R1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    output = PACKET / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    epochs = np.arange(1, 5)
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    for row, (configuration, color) in enumerate(COLORS.items()):
        members = [run for run in data["runs"] if run["configuration_id"] == configuration]
        assert len(members) == 2 and all(len(run["history"]) == 4 for run in members)
        for column, (train_key, valid_key, title) in enumerate(
            (("train_eval_wmse", "valid_wmse", "WMSE"), ("train_eval_wp", "weighted_pearson", "WP"))
        ):
            axis = axes[row, column]
            for key, label, style, line_color in (
                (train_key, "Train diagnostic (32 sequences)", "--", "#777777"),
                (valid_key, "Official valid (1,873 sequences)", "-", color),
            ):
                values = np.asarray([[entry[key] for entry in run["history"]] for run in members])
                axis.plot(
                    epochs, values.mean(axis=0), style, color=line_color, marker="o", label=label
                )
                axis.fill_between(
                    epochs, values.min(axis=0), values.max(axis=0), color=line_color, alpha=0.14
                )
            axis.axvline(4, color="#333333", linestyle=":", linewidth=1)
            axis.set(title=f"{configuration}: {title}", xlabel="Completed epoch", xticks=epochs)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.legend(fontsize=8)
    figure.suptitle(
        "Measured train / valid curves — mean and range of two seeds\nTrain: all rows after warmup; valid: is_scored. Masks differ; ranges are not CIs.",
        fontsize=12,
    )
    save_figure(figure, "gru_train_valid", output)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for configuration, color in COLORS.items():
        members = [run for run in data["runs"] if run["configuration_id"] == configuration]
        for quantile, style, label in ((1, "-", "median"), (2, "--", "95th percentile")):
            values = np.asarray(
                [
                    [entry["total_q05_q50_q95"][quantile] for entry in run["gradient_summaries"]]
                    for run in members
                ]
            )
            axes[0].plot(
                epochs,
                values.mean(axis=0),
                style,
                marker="o",
                color=color,
                label=f"{configuration}: {label}",
            )
        for axis, key, multiplier in (
            (axes[1], "clip_rate", 100),
            (axes[2], "family_sq_share_mean", 100),
        ):
            values = (
                np.asarray(
                    [
                        [
                            entry[key]["head"] if isinstance(entry[key], dict) else entry[key]
                            for entry in run["gradient_summaries"]
                        ]
                        for run in members
                    ]
                )
                * multiplier
            )
            axis.plot(epochs, values.mean(axis=0), marker="o", color=color, label=configuration)
            axis.fill_between(
                epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=0.14
            )
    axes[0].axhline(1, color="#333333", linestyle=":", label="Clipping threshold = 1")
    for axis, title, ylabel in zip(
        axes,
        (
            "Pre-clipping global gradient norm",
            "Updates subject to clipping",
            "Head share of squared gradient norm",
        ),
        ("L2 norm", "% of updates", "%"),
        strict=True,
    ):
        axis.set(title=title, xlabel="Completed epoch", ylabel=ylabel, xticks=epochs)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle(
        "Complete-epoch gradient diagnostics; mean of seed summaries, no significance claim",
        fontsize=11,
    )
    save_figure(figure, "gru_gradient_diagnostics", output)
    receipt = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "figures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.glob("gru_*.*"))
        },
        "scope": "Six measured completed teacher fits; no new fits, no synthetic scientific data",
    }
    (output / "GRU_PLOT_RECEIPT_R1.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
