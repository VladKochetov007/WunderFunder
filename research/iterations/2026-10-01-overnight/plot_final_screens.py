"""Render already recorded comparisons; no predictions, targets or new statistics."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wunder-final-screens-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ITEMS = [
    ("AUX off − ordinary H2", "TRAIN-AUX02-primary"),
    ("Transition T − frequency F", "TRANSHEAD-01-primary"),
    ("Ensemble heads − ordinary heads", "ENSHEAD-01-primary"),
    ("Scale control − ordinary heads", "ENSHEAD-01-scale-primary"),
    ("SAM − matched parent", "SAM-seed3-paired"),
    ("Event memory − current-state control", "CTX-joint-memory-vs-current"),
    ("Q3 INT8 − fixed-three GPU", "Q3-quant-vs-baseline3"),
    ("Scalar embedding − matched parent", "SCALARPLE01-seed3-paired"),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    records = []
    for label, identifier in ITEMS:
        path = HERE / "evaluations" / (identifier + ".json")
        data = json.loads(path.read_text())
        records.append((label, data, path))
    fig, (pooled, folds) = plt.subplots(1, 2, figsize=(13.2, 6.1),
                                      sharey=True, gridspec_kw={"width_ratios": [1.7, 1]})
    y = np.arange(len(records))
    for i, (label, data, path) in enumerate(records):
        delta = data["delta"] * 1000
        low, high = np.asarray(data["delta_boot_ci95"]) * 1000
        pooled.plot([low, high], [i, i], color="#8795a5", lw=3, solid_capstyle="round")
        pooled.scatter(delta, i, color="#153e63", s=45, zorder=3)
        folds.scatter(data["delta_by_fold"]["0"] * 1000, i - .12,
                      color="#167d8d", marker="o", s=42)
        folds.scatter(data["delta_by_fold"]["1"] * 1000, i + .12,
                      color="#c8792e", marker="s", s=39)
    pooled.set_yticks(y, [record[0] for record in records])
    pooled.invert_yaxis()
    pooled.set_title("Pooled change and conditional 95% chain interval", fontsize=11, loc="left")
    folds.set_title("Point changes by held-out fold", fontsize=11, loc="left")
    for ax in (pooled, folds):
        ax.axvline(0, color="#202c38", lw=1)
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Δ WP × 1,000   (right = higher)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    folds.scatter([], [], color="#167d8d", marker="o", label="Fold 0")
    folds.scatter([], [], color="#c8792e", marker="s", label="Fold 1")
    folds.legend(loc="lower left", bbox_to_anchor=(0, 1.07),
                 frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Eight completed research contrasts · 52 fits · 36 recorded readouts",
                 fontsize=14, x=.03, ha="left", weight="bold")
    fig.text(.03, .045,
             "Comparators differ as named. Intervals condition on fitted models and observed chains; "
             "all folds were exposed.\nNo new forecasting recipe passed its predeclared promotion gates. "
             "No seed/selection/hidden-domain uncertainty is estimated here.",
             fontsize=9, color="#465261", ha="left")
    fig.subplots_adjust(left=.285, right=.98, top=.86, bottom=.19, wspace=.17)
    fig.savefig(HERE / "final-research-screens.png", dpi=180)
    fig.savefig(HERE / "final-research-screens.svg")
    plt.close(fig)
    receipt = {"script_sha256": digest(__file__), "new_WP_queries": 0,
               "new_bootstrap_or_scores": False,
               "sources": {str(path.relative_to(HERE)): digest(path) for _, _, path in records},
               "outputs": {name: digest(HERE / name) for name in
                           ["final-research-screens.png", "final-research-screens.svg"]}}
    (HERE / "final-research-screens-inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
