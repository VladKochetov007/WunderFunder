"""Reproduce visual summaries and selected raw-support diagnostics from the EDA scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from wnn.moment_geometry.analysis import prefix_errors, robust_arc


def plot_overview(output: Path) -> None:
    """Render every feature and six visually selected candidate examples."""
    with np.load(output / "moments.npz", allow_pickle=False) as source:
        moments = {key: source[key] for key in source.files}
    for page in range(4):
        fig, axes = plt.subplots(7, 4, figsize=(12, 17), layout="constrained")
        for column, axis in enumerate(axes.flat, start=page * 28):
            for split, color in (("train", "#2f6cba"), ("valid", "#dd7b25")):
                axis.scatter(
                    moments[f"{split}_mean"][:, -1, column],
                    np.sqrt(moments[f"{split}_var"][:, -1, column]),
                    s=5,
                    alpha=0.5,
                    c=color,
                )
            axis.set_title(str(moments["columns"][column]), fontsize=9)
            axis.tick_params(labelsize=6)
            axis.grid(alpha=0.2)
        fig.suptitle(
            f"Raw mean(x)/population std(y), columns{page * 28}..{page * 28 + 27}; blue=train orange=valid",
            fontsize=10,
        )
        fig.savefig(output / f"all_columns_{page + 1}.png", dpi=125)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    rows = []
    for axis, name in zip(
        axes.flat, ("i0_p0", "i0_p3", "i1_p0", "i1_p3", "i1_p9", "i0_dv2"), strict=True
    ):
        column = list(moments["columns"]).index(name)
        arc = robust_arc(moments["train_mean"][:, -1, column], moments["train_var"][:, -1, column])
        beta, intercept = arc["beta"], arc["intercept"]
        for split, color in (("train", "tab:blue"), ("valid", "tab:orange")):
            mean, variance = (
                moments[f"{split}_mean"][:, -1, column],
                moments[f"{split}_var"][:, -1, column],
            )
            axis.scatter(mean, np.sqrt(variance), s=10, alpha=0.6, label=split, color=color)
            arc[f"{split}_fraction"] = float(
                np.mean(np.abs(variance + mean**2 - beta * mean - intercept) < 0.002)
            )
        radius2 = intercept + beta**2 / 4
        if radius2 > 0:
            values = np.linspace(beta / 2 - np.sqrt(radius2), beta / 2 + np.sqrt(radius2), 300)
            axis.plot(
                values,
                np.sqrt(np.maximum(beta * values + intercept - values**2, 0)),
                c="black",
                lw=1,
            )
        rows.append({"column": name, **arc})
        axis.set(title=name, xlabel="Sequence mean", ylabel="Population std")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output / "candidate_arcs.png", dpi=150)
    plt.close(fig)
    (output / "robust_arcs.json").write_text(json.dumps(rows, indent=2) + "\n")


def inspect_support(root: Path, output: Path) -> None:
    """Read six adaptively selected valid sequences and record their actual raw support."""
    with np.load(output / "moments.npz", allow_pickle=False) as source:
        moments = {key: source[key] for key in source.files}
    column = list(moments["columns"]).index("i0_p0")
    arc = robust_arc(moments["train_mean"][:, -1, column], moments["train_var"][:, -1, column])
    mean, variance = moments["valid_mean"][:, -1, column], moments["valid_var"][:, -1, column]
    error = variance + mean**2 - arc["beta"] * mean - arc["intercept"]
    chosen = [*np.argsort(np.abs(error))[:3], *np.argsort(error)[-3:]]
    reader = pq.ParquetFile(root / "datasets/valid.parquet")
    rows, traces = [], []
    for position in chosen:
        group = int(moments["valid_groups"][position])
        values = reader.read_row_group(group, columns=["i0_p0"], use_threads=False)[
            "i0_p0"
        ].to_numpy()
        support, counts = np.unique(values, return_counts=True)
        order = np.argsort(counts)[::-1]
        rows.append(
            {
                "group": group,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "arc_residual": float(error[position]),
                "unique": len(support),
                "top_values": support[order[:8]].tolist(),
                "top_counts": counts[order[:8]].tolist(),
                "top2_fraction": float(counts[order[:2]].sum() / len(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
        traces.append(values)
    reader.close()
    (output / "support_probe.json").write_text(
        json.dumps(
            {"arc": arc, "rows": rows, "prefix_errors": prefix_errors(moments, column)}, indent=2
        )
        + "\n"
    )
    np.savez_compressed(
        output / "example_traces.npz",
        groups=moments["valid_groups"][chosen],
        values=np.stack(traces),
    )


def plot_support(output: Path) -> None:
    """Show actual raw traces and prefix paths for near-arc and cloud examples."""
    with np.load(output / "example_traces.npz", allow_pickle=False) as source:
        groups, values = source["groups"], source["values"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout="constrained")
    for index, axis in enumerate(axes.flat):
        axis.plot(np.arange(20000), values[index], lw=0.5)
        axis.set(
            title=f"valid group {groups[index]}: {'near arc' if index < 3 else 'cloud'}",
            xlabel="Step",
            ylabel="Raw i0_p0",
        )
        axis.grid(alpha=0.2)
    fig.savefig(output / "support_traces.png", dpi=140)
    plt.close(fig)
    with np.load(output / "moments.npz", allow_pickle=False) as source:
        moments = {key: source[key] for key in source.files}
    column = list(moments["columns"]).index("i0_p0")
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for group in groups:
        position = int(np.flatnonzero(moments["valid_groups"] == group)[0])
        mean = moments["valid_mean"][position, :, column]
        std = np.sqrt(moments["valid_var"][position, :, column])
        axis.plot(mean, std, marker="o", lw=1, label=str(group))
        axis.annotate("32", (mean[0], std[0]), fontsize=7)
        axis.annotate("20k", (mean[-1], std[-1]), fontsize=7)
    axis.set(
        xlabel="Observed prefix mean",
        ylabel="Observed prefix population std",
        title="Prefix estimates move as more of the sequence is observed",
    )
    axis.grid(alpha=0.2)
    axis.legend(title="valid group", fontsize=7)
    fig.savefig(output / "prefix_paths.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Regenerate descriptive artifacts in a new output snapshot, using a completed scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support", action="store_true")
    args = parser.parse_args()
    plot_overview(args.output)
    if args.support:
        inspect_support(args.root, args.output)
    if (args.output / "example_traces.npz").exists():
        plot_support(args.output)


if __name__ == "__main__":
    main()
