"""Collect complete pilot evidence and evaluate only preregistered fixed mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wnn.gmetric import GlobalAccumulator

MIXTURES = (("control", "dictionary"), ("control", "combined"), ("scaled", "dictionary"))


def fixed_mixture(
    data: Path, directory: Path, members: tuple[str, str], seed: int
) -> dict[str, object]:
    """Stream a fixed half/half mixture without fitting coefficients or staging full predictions."""
    targets = np.load(data / "eval_y.npy", mmap_mode="r")
    mask = np.load(data / "eval_mask.npy", mmap_mode="r")
    folds = np.load(data / "eval_fold.npy", mmap_mode="r")
    predictions = [
        np.load(directory / "predictions" / f"{arm}_seed{seed}_evalpred.npy", mmap_mode="r")
        for arm in members
    ]
    accumulators = {key: GlobalAccumulator() for key in ("pooled", "0", "1")}
    for index in range(len(targets)):
        mixed = 0.5 * predictions[0][index] + 0.5 * predictions[1][index]
        accumulators["pooled"].add(targets[index], mixed, mask[index])
        accumulators[str(int(folds[index]))].add(targets[index], mixed, mask[index])
    return {
        "members": list(members),
        "weights": [0.5, 0.5],
        "seed": seed,
        "scores": {key: accumulator.result() for key, accumulator in accumulators.items()},
    }


def collect(directory: Path, prefix: str, output: Path) -> list[dict[str, object]]:
    """Copy small result/progress receipts and hash separately stored model/prediction payloads."""
    rows = json.loads((directory / "results.json").read_text())
    (output / f"{prefix}_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    logs = []
    for path in sorted((directory / "logs").glob("*.jsonl")):
        arm, seed = path.stem.split("_seed")
        logs.extend(
            {"arm": arm, "seed": int(seed), **json.loads(line)}
            for line in path.read_text().splitlines()
        )
    (output / f"{prefix}_progress.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in logs)
    )
    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
        files[str(path.relative_to(directory))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    (output / f"{prefix}_artifacts.json").write_text(
        json.dumps({"storage": str(directory), "files": files}, indent=2) + "\n"
    )
    return rows


def summarize(
    rows: list[dict[str, object]], mixtures: list[dict[str, object]]
) -> dict[str, object]:
    """Report all seeds, mean/SD and matched contrasts without selecting a winning seed."""
    arms = {
        arm: sorted([row for row in rows if row["arm"] == arm], key=lambda row: row["seed"])
        for arm in ("control", "scaled", "dictionary", "combined")
    }
    summary = {}
    for arm, endpoints in arms.items():
        values = np.asarray([row["evaluation"]["wp"] for row in endpoints])
        summary[arm] = {
            "mean_wp": float(values.mean()),
            "seed_sd": float(values.std(ddof=1)),
            "seeds": [row["seed"] for row in endpoints],
            "wp": values.tolist(),
            "fold_means": {
                str(fold): float(
                    np.mean([row["evaluation_by_fold"][str(fold)]["wp"] for row in endpoints])
                )
                for fold in (0, 1)
            },
        }
    contrasts = {}
    for candidate, control in (
        ("scaled", "control"),
        ("dictionary", "control"),
        ("combined", "control"),
        ("combined", "scaled"),
    ):
        differences = np.asarray(summary[candidate]["wp"]) - np.asarray(summary[control]["wp"])
        contrasts[f"{candidate}-{control}"] = {
            "deltas": differences.tolist(),
            "mean": float(differences.mean()),
            "seed_sd": float(differences.std(ddof=1)),
        }
    interaction = np.asarray(contrasts["combined-scaled"]["deltas"]) - np.asarray(
        contrasts["dictionary-control"]["deltas"]
    )
    return {
        "arms": summary,
        "paired_contrasts": contrasts,
        "interaction": {"deltas": interaction.tolist(), "mean": float(interaction.mean())},
        "fixed_mixtures": mixtures,
        "limits": "Three fitting seeds, same small fitting pool, two exposed evaluation slices; seed variation is not component uncertainty. No production, distillation or public-transfer claim.",
    }


def plot_results(output: Path) -> None:
    """Render paired-seed deltas, fold heterogeneity and observed training curves."""
    import matplotlib.pyplot as plt

    p = output
    d = json.loads((p / "discrete_summary_r2.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = {"scaled": "tab:blue", "dictionary": "tab:orange", "combined": "tab:green"}
    for axis, (shape, r) in zip(axes, d.items(), strict=True):
        for j, arm in enumerate(["scaled", "dictionary", "combined"]):
            values = r["paired_contrasts"][arm + "-control"]["deltas"]
            axis.scatter(np.full(3, j), values, label=arm, c=colors[arm], s=40)
            axis.hlines(np.mean(values), j - 0.2, j + 0.2, color=colors[arm], lw=3)
        axis.axhline(0, c="black", lw=0.8)
        axis.set_xticks([0, 1, 2], ["moments + scale", "dictionary", "all features"])
        axis.set(
            title=shape, xlabel="Treatment", ylabel="Paired pooled WP delta vs matched control"
        )
        axis.grid(alpha=0.2)
    fig.savefig(p / "paired_deltas_r2.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, (shape, r) in zip(axes, d.items(), strict=True):
        for arm in ["control", "scaled", "dictionary", "combined"]:
            v = r["arms"][arm]
            axis.plot([0, 1], [v["fold_means"][str(f)] for f in [0, 1]], marker="o", label=arm)
        axis.set_xticks([0, 1], ["exposed fold0", "exposed fold1"])
        axis.set(title=shape, ylabel="Mean official WP over3fitting seeds")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.savefig(p / "fold_heterogeneity_r2.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for j, shape in enumerate(d):
        rows = [
            json.loads(line) for line in (p / f"r2_{shape}_progress.jsonl").read_text().splitlines()
        ]
        for arm in ["control", "scaled", "dictionary", "combined"]:
            subset = [r for r in rows if r["arm"] == arm]
            steps = sorted({r["step"] for r in subset})
            for i, metric in enumerate(["loss", "preclip_grad_norm"]):
                axes[i, j].plot(
                    steps,
                    [np.mean([r[metric] for r in subset if r["step"] == s]) for s in steps],
                    label=arm,
                )
                axes[i, j].set(title=f"{shape}: {metric}", xlabel="Optimizer update", ylabel=metric)
                axes[i, j].grid(alpha=0.2)
                axes[i, j].legend(fontsize=7)
    fig.savefig(p / "training_progress_r2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Collect the completed two-capacity comparison and its preregistered half mixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for shape in ("h32_l1", "h128_l2"):
        directory = args.data / f"pilot_{shape}"
        rows = collect(directory, f"r2_{shape}", args.output)
        if len(rows) != 12:
            raise ValueError("incomplete planned endpoint set")
        mixtures = [
            fixed_mixture(args.data, directory, members, seed)
            for members in MIXTURES
            for seed in (3, 7, 11)
        ]
        summary[shape] = summarize(rows, mixtures)
    (args.output / "discrete_summary_r2.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plots:
        plot_results(args.output)


if __name__ == "__main__":
    main()
