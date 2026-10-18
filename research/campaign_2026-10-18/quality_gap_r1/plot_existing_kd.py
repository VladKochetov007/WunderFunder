"""Plot immutable historical KD trajectory summaries without reading arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Render two panels from twelve hash-pinned aggregate rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if len(raw) > 20000 or hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise ValueError("aggregate input identity mismatch")
    document = json.loads(raw)
    rows = document["rows"]
    if len(rows) != 12:
        raise ValueError("expected twelve existing aggregate rows")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True)
    for axis, subset, title in zip(
        axes,
        ("fit_valid_in_sample", "eval_exposed"),
        ("Fitted valid groups (in-sample)", "Excluded F0 groups (previously inspected)"),
        strict=True,
    ):
        for arm, label, color in (
            ("alpha0", "Labels only continuation", "#c65b43"),
            ("alpha09", "Teacher target continuation (alpha = 0.9)", "#267d99"),
        ):
            selected = sorted(
                (row for row in rows if row["arm"] == arm and row["subset"] == subset),
                key=lambda row: row["epochs"],
            )
            if [row["epochs"] for row in selected] != [2, 4, 6]:
                raise ValueError("unexpected trajectory points")
            axis.plot(
                [row["epochs"] for row in selected],
                [row["wp"] for row in selected],
                marker="o",
                linewidth=2.2,
                color=color,
                label=label,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Additional training epochs from the shared parent")
        axis.set_ylabel("Official weighted Pearson")
        axis.set_xticks([2, 4, 6])
        axis.grid(alpha=0.2)
    fig.suptitle("Historical warm-parent comparison: fit improves while evaluation diverges")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.085,
        "One seed; parent already distilled; existing grades, no new fit or public-score forecast.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    outputs = {}
    for suffix in ("png", "svg"):
        path = args.output_dir / f"existing_kd_trajectory.{suffix}"
        fig.savefig(path, dpi=170, facecolor="white")
        data = path.read_bytes()
        outputs[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    plt.close(fig)
    receipt = {
        "input_sha256": args.input_sha256,
        "source_grade_sha256": document["source_sha256"],
        "outputs": outputs,
        "new_fit_or_prediction": False,
    }
    with (args.output_dir / "PLOT_RECEIPT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
