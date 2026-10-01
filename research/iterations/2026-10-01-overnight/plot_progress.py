"""Plot existing frozen reports only; performs no model or metric evaluation."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CONTRASTS = [
    ("TRAIN-AUX02-primary", "Aux off / matched H2"),
    ("TRANSHEAD-01-primary", "Transition history / frequency control"),
    ("ENSHEAD-01-primary", "Ensemble-trained heads / matched H2"),
    ("ENSHEAD-01-scale-primary", "Ensemble-trained scales / matched H2"),
    ("SAM-seed3-paired", "SAM parent / matched parent + H2"),
]


def main():
    reports = []
    source_hashes = {}
    for name, label in CONTRASTS:
        path = HERE / "evaluations" / (name + ".json")
        reports.append((label, json.loads(path.read_text())))
        source_hashes[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    scale = 10000
    for i, (label, result) in enumerate(reports):
        point = result["delta"] * scale
        lo, hi = np.array(result["delta_boot_ci95"]) * scale
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", color="#243b53",
                    capsize=4, lw=1.5, label="Pooled + conditional 95% interval" if i == 0 else None)
        ax.scatter(result["delta_by_fold"]["0"] * scale, i-.14, marker="|", s=130,
                   color="#d97706", label="Fold 0" if i == 0 else None)
        ax.scatter(result["delta_by_fold"]["1"] * scale, i+.14, marker="|", s=130,
                   color="#0f766e", label="Fold 1" if i == 0 else None)
    ax.axvline(0, color="#64748b", lw=1)
    ax.axvline(3, color="#b91c1c", ls="--", lw=1,
               label="Predeclared pooled spending gate (+0.0003)")
    ax.set_yticks(range(len(reports)), [label for label, _ in reports])
    ax.invert_yaxis()
    ax.set_xlabel("Change in weighted Pearson (× 10⁻⁴); larger is better")
    ax.grid(axis="x", alpha=.18)
    ax.set_title("First research batch: no recipe passed its promotion gate", loc="left", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(.47, -.19), ncol=2, frameon=False, fontsize=9)
    fig.text(.02, .025,
             "All folds exposed. Intervals resample whole chains conditional on fitted models; they exclude seed, selection and domain-shift uncertainty.\n"
             "Comparators differ as labelled; effects are not additive. A pooled gate also requires the predefined fold/seed and serving checks.",
             fontsize=8.5, color="#475569")
    fig.subplots_adjust(left=.39, right=.98, top=.86, bottom=.27)
    for suffix in ("png", "svg"):
        fig.savefig(HERE / ("completed-screens." + suffix), dpi=180)
    receipt = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "source_reports_sha256": source_hashes, "new_WP_readouts": 0,
               "scope": "Visualization of five completed predefined contrasts; not an exhaustive experiment table",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / "completed-screens.inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
