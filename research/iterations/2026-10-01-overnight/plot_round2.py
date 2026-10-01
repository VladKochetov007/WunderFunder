"""Visualize completed frozen reports; no predictions or targets are loaded."""
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
    ("CTX-joint-memory-vs-current", "Event memory / current-state control"),
    ("CTX-joint-memory-vs-plain", "Event memory / plain fixed three"),
    ("Q3-quant-vs-baseline3", "INT8 trio / plain fixed three"),
    ("Q3-quant-vs-fp16", "INT8 trio / same trio in FP16"),
]


def main():
    reports, sources = [], {}
    for name, label in CONTRASTS:
        path = HERE / "evaluations" / (name + ".json")
        reports.append((label, json.loads(path.read_text())))
        sources[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7), gridspec_kw={"height_ratios": [3, 1.5]})
    for panel, (ax, subset) in enumerate(zip(axes, [reports[:3], reports[3:]])):
        for i, (label, result) in enumerate(subset):
            value = result["delta"] * 10000
            low, high = np.array(result["delta_boot_ci95"]) * 10000
            ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt="o", color="#243b53",
                        capsize=4, lw=1.5, label="Pooled + conditional 95% interval" if i == 0 else None)
            ax.scatter(result["delta_by_fold"]["0"] * 10000, i-.12, marker="|", s=130,
                       color="#d97706", label="Fold 0" if i == 0 else None)
            ax.scatter(result["delta_by_fold"]["1"] * 10000, i+.12, marker="|", s=130,
                       color="#0f766e", label="Fold 1" if i == 0 else None)
        ax.set_yticks(range(len(subset)), [label for label, _ in subset])
        ax.invert_yaxis()
        ax.set_ylim(len(subset)-.5, -.5)
        ax.axvline(0, color="#64748b", lw=1)
        ax.grid(axis="x", alpha=.18)
        ax.set_xlabel("Change in weighted Pearson (× 10⁻⁴); larger is better")
        gate = 3 if panel == 0 else -1
        ax.axvline(gate, color="#b91c1c", ls="--", lw=1)
    axes[0].set_title("Useful-gain gate: pooled ≥ +0.0003 and both folds ≥ 0 — no candidate passed",
                      loc="left", fontsize=11, pad=14)
    axes[1].set_title("Quantization-loss gate: pooled and each fold ≥ −0.0001 — passed",
                      loc="left", fontsize=11, pad=14)
    axes[1].set_xlim(-1.15, .2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.56, .07), ncol=3,
               frameon=False, fontsize=9)
    fig.text(.025, .025,
             "All folds exposed. Intervals resample whole chains conditional on fitted models; they exclude seed, selection and hidden-domain uncertainty.\n"
             "Q3 serving runtime is qualified only on the tested VNNI path. The plain fixed-three comparator uses existing GPU predictions.",
             fontsize=8.5, color="#475569")
    fig.subplots_adjust(left=.34, right=.98, top=.93, bottom=.22, hspace=.75)
    for extension in ("png", "svg"):
        fig.savefig(HERE / ("round2-screens." + extension), dpi=180)
    receipt = {"generated_utc": datetime.now(timezone.utc).isoformat(), "new_WP_readouts": 0,
               "source_reports_sha256": sources,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Four predeclared completed contrasts; secondary seed diagnostics remain in full reports"}
    (HERE / "round2-screens.inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
