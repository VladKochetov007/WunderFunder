"""Build unsmoothed same-weight train/validation curves and provenance tables."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from stage import OPERATORS, PREVIOUS, WAVES, read, sha

LABELS = {
    "control": "Control: LR1e-4 / drop0 / WD.01",
    "joint": "Joint: drop.1 / WD.1",
    "dropout_only": "Dropout only: .1",
    "decay_only": "Weight decay only: .1",
    "lr_low": "LR5e-5",
    "lr_high": "LR2e-4",
}
COLORS = dict(
    zip(LABELS, ["#333333", "#9467bd", "#0072b2", "#d55e00", "#009e73", "#cc79a7"], strict=True)
)


def collect(root: Path) -> list[dict]:
    """Verify each complete prediction artifact before plotting its metrics."""
    sys.path.insert(0, str(OPERATORS))
    from verify import verify_result

    contract = read(root / "CONTRACT.json")
    source = Path(contract["source_root"])
    records = []
    parents = {item["path"]: item for item in read(root / "REUSED_RESULTS.json")}
    sources = [(PREVIOUS, None)] + [(root / wave, wave) for wave in WAVES]
    for folder, wave in sources:
        plan_file = folder / ("plan-selected.json" if wave is None else "plan-profile.json")
        plan = read(plan_file)
        names = {j["name"]: j for j in plan["jobs"]}
        for path in sorted((folder / "cohort-b128").glob("*/validation/epoch-*/RESULT.json")):
            result = read(path)
            if result.get("status") != "complete":
                continue
            try:
                if wave is None and parents[str(path)]["sha256"] != sha(path):
                    raise ValueError("reused result changed")
                verify_result(path, source)
            except (OSError, ValueError, KeyError) as error:
                (root / "METRIC_VERIFICATION_FAILED.json").write_text(
                    json.dumps({"path": str(path), "error": str(error)}) + "\n"
                )
                raise
            name = result["job"]
            arm = (
                ("control" if name.startswith("ctl") else "joint")
                if wave is None
                else names[name]["scientific_arm"]
            )
            valid = result["heldout"]["metrics"]
            records.append(
                {
                    "arm": arm,
                    "fold": result["fold"],
                    "epoch": result["completed_epochs"],
                    "train_probe_wmse": result["train_probe"]["metrics"][
                        "weighted_mse_all_after_warmup"
                    ],
                    "validation_wmse": valid["weighted_mse_all_after_warmup"],
                    "validation_wp": valid["official_wp"],
                    "validation_wp_t0": valid["official_wp_t0"],
                    "validation_wp_t1": valid["official_wp_t1"],
                    "result_path": str(path),
                    "result_sha256": sha(path),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "config_sha256": result["config_sha256"],
                    "split_sha256": result["split_sha256"],
                    "source_sha256": result["source_sha256"],
                    "reused": wave is None,
                }
            )
    return records


def selection(records: list[dict]) -> dict:
    """Select one complete common epoch per arm and report paired development deltas."""
    chosen = {}
    for arm in LABELS:
        cells = {(r["fold"], r["epoch"]): r for r in records if r["arm"] == arm}
        if len(cells) != 10:
            continue
        epoch = max(
            range(1, 6), key=lambda e: (sum(cells[(f, e)]["validation_wp"] for f in [0, 2]), -e)
        )
        values = {str(f): cells[(f, epoch)]["validation_wp"] for f in [0, 2]}
        chosen[arm] = {"epoch": epoch, "wp_by_fold": values, "mean_wp": sum(values.values()) / 2}
    if "control" in chosen:
        for item in chosen.values():
            item["paired_deltas"] = {
                f: v - chosen["control"]["wp_by_fold"][f] for f, v in item["wp_by_fold"].items()
            }
            item["mean_delta"] = item["mean_wp"] - chosen["control"]["mean_wp"]
    return {
        "selection": chosen,
        "independent_confirmation": False,
        "rule": "Each arm: common epoch maximizing mean F0/F2 inner WP, earliest tie.",
        "caution": "Development selection; one seed per fold. No significance/public-transfer claim.",
    }


def render(root: Path) -> list[dict]:
    """Export real metric coordinates and standalone PNG/SVG figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = collect(root)
    output = root / "plots"
    output.mkdir(exist_ok=True)
    (output / "coordinates.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "coordinates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "selection.json").write_text(json.dumps(selection(records), indent=2) + "\n")
    groups = {
        "regularization": ["control", "joint", "dropout_only", "decay_only"],
        "learning_rate": ["control", "lr_low", "lr_high"],
    }
    with plt.rc_context({"font.size": 10, "svg.hashsalt": "compact-factor-sweep-v1"}):
        for group, arms in groups.items():
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
            for column, fold in enumerate([0, 2]):
                for arm in arms:
                    points = sorted(
                        (r for r in records if r["arm"] == arm and r["fold"] == fold),
                        key=lambda r: r["epoch"],
                    )
                    if not points:
                        continue
                    epochs = [r["epoch"] for r in points]
                    color = COLORS[arm]
                    axes[0, column].plot(
                        epochs,
                        [r["train_probe_wmse"] for r in points],
                        "--o",
                        color=color,
                        alpha=0.75,
                    )
                    axes[0, column].plot(
                        epochs,
                        [r["validation_wmse"] for r in points],
                        "-o",
                        color=color,
                        label=LABELS[arm],
                    )
                    axes[1, column].plot(
                        epochs, [r["validation_wp"] for r in points], "-o", color=color
                    )
                axes[0, column].set_title(f"F{fold}: weighted MSE (lower is better)")
                axes[0, column].set_ylabel("WMSE: dashed train probe / solid validation")
                axes[1, column].set_title(f"F{fold}: validation WP (higher is better)")
                axes[1, column].set_ylabel("Official-mask inner WP")
                axes[1, column].set_xlabel("Completed epoch")
                for row in range(2):
                    axes[row, column].set_xticks(range(1, 6))
                    axes[row, column].grid(alpha=0.2)
                axes[0, column].legend(fontsize=8)
            fig.suptitle(f"Compact Transformer 113.5M — {group.replace('_', ' ')}")
            fig.text(
                0.5,
                0.015,
                "Train = fixed64 fitting-sequence probe; val = full inner holdout. Same-weight eval, rows >=99. Development splits; no smoothing.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.045, 1, 0.95))
            fig.savefig(output / f"{group}.png", dpi=180)
            fig.savefig(output / f"{group}.svg", metadata={"Date": None})
            plt.close(fig)
    return records
