"""Check anonymized-column algebra and plot raw target mechanics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent


def main() -> None:
    """Audit within-row contrasts and target increments without fitting future."""
    cohort = json.loads((OUT / "cohort.json").read_text())["cohort"]
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    records = []
    figure, axes = plt.subplots(4, 2, figsize=(12, 10))
    plotted = set()
    for node in cohort:
        table = pf.read_row_group(
            node["row_group"], columns=["t0", "t1", "i0_p0", "i0_p11"], use_threads=False
        )
        y = np.column_stack([table[c].to_numpy() for c in ("t0", "t1")]).astype(float)
        p = np.column_stack([table[c].to_numpy() for c in ("i0_p0", "i0_p11")]).astype(float)
        contrast = np.column_stack([y[:, 0] + y[:, 1], y[:, 0] - y[:, 1]])
        deltay, deltap = np.diff(y, axis=0), np.diff(p, axis=0)
        varied = np.any(deltap != 0, axis=1)
        # Labels only appear on the response side; this measures algebra, not inference.
        coefficient = np.linalg.lstsq(deltap[varied], deltay[varied], rcond=None)[0]
        residual = deltay[varied] - deltap[varied] @ coefficient
        denom = np.maximum(np.sum(deltay[varied] ** 2, axis=0), 1e-12)
        identity = np.column_stack([np.ones(len(y)), p])
        row_coefficient = np.linalg.lstsq(identity[::20], contrast[::20], rcond=None)[0]
        row_residual = contrast - identity @ row_coefficient
        records.append(
            {
                **node,
                "increment_coefficient_p0_p11_to_t0_t1": coefficient.tolist(),
                "increment_explained_uncentered": (
                    1 - np.sum(residual**2, axis=0) / denom
                ).tolist(),
                "increment_exact_to_1e_minus5_fraction": (np.abs(residual) < 1e-5).mean(0).tolist(),
                "sum_difference_std": contrast.std(0).tolist(),
                "sum_difference_unique": [
                    len(np.unique(contrast[:, k].round(5))) for k in range(2)
                ],
                "current_p0_p11_explained_centered_sum_difference": (
                    1
                    - np.sum(row_residual**2, axis=0)
                    / np.maximum(np.sum((contrast - contrast.mean(0)) ** 2, axis=0), 1e-12)
                ).tolist(),
                "same_target_increment_fraction": float(
                    (np.abs(deltay[:, 0] - deltay[:, 1]) < 1e-5).mean()
                ),
                "opposite_target_increment_fraction": float(
                    (np.abs(deltay[:, 0] + deltay[:, 1]) < 1e-5).mean()
                ),
            }
        )
        cls = node["label_class_diagnostic_only"]
        if cls not in plotted:
            axes[cls, 0].plot(y[:1000, 0], label="t0", lw=1)
            axes[cls, 0].plot(y[:1000, 1], label="t1", lw=1)
            axes[cls, 0].set_title(f"Class {cls}, raw first 1000 rows, rg {node['row_group']}")
            axes[cls, 0].legend()
            axes[cls, 1].scatter(contrast[::20, 0], contrast[::20, 1], s=3, alpha=0.3)
            axes[cls, 1].set_xlabel("t0 + t1")
            axes[cls, 1].set_ylabel("t0 - t1")
            plotted.add(cls)
    (OUT / "algebra_evidence.json").write_text(json.dumps(records, indent=2))
    (OUT / "eda").mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(OUT / "eda/target_mechanics.png", dpi=140)


if __name__ == "__main__":
    main()
