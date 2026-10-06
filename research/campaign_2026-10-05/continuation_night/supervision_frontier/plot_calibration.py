from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_calibration(report_path: Path, output_dir: Path) -> None:
    """Plot recorded fit-only masses without modifying calibration evidence."""
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    observations = report["update_masses"]
    masses = np.asarray(observations["values"], dtype="<f8")
    if (
        masses.ndim != 1
        or masses.size != observations["count"]
        or not masses.size
        or not np.isfinite(masses).all()
        or (masses < 0).any()
        or hashlib.sha256(masses.tobytes()).hexdigest() != observations["sha256"]
    ):
        raise ValueError("Recorded calibration mass identity is invalid")
    reference = float(masses.mean())
    if reference <= 0 or not np.isclose(reference, report["reference_mass"]):
        raise ValueError("Recorded calibration reference does not match observations")
    output_dir.mkdir(parents=True, exist_ok=False)
    relative_mass = masses / reference
    index = np.arange(1, masses.size + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(index, relative_mass, linewidth=0.6, color="#2878a5")
    axes[0].set(xlabel="Recorded update order", ylabel="Mass / fit-pass mean")
    axes[1].hist(relative_mass, bins=40, color="#2878a5")
    axes[1].set(xlabel="Mass / fit-pass mean", ylabel="Update count")
    axes[2].plot(index, np.cumsum(relative_mass) / index, color="#ad632c")
    axes[2].set(xlabel="Recorded update order", ylabel="Cumulative mean / final mean")
    for axis in (axes[0], axes[2]):
        axis.axhline(1, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axvline(1, color="gray", linewidth=0.7, linestyle="--")
    for axis in axes:
        axis.grid(alpha=0.15)
    fig.suptitle("E496 measured fit-only denominator masses; no predictive-score claim")
    fig.savefig(output_dir / "denominator_masses.png", dpi=180)
    plt.close(fig)
    summary = {
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "observations_sha256": observations["sha256"],
        "updates": int(masses.size),
        "reference_mass": reference,
        "coefficient_of_variation": float(masses.std() / reference),
        "relative_mass_quantiles": np.quantile(
            relative_mass, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
        ).tolist(),
        "claim_boundary": "descriptive fit-only update masses; no model gain",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    plot_calibration(args.report, args.output_dir)


if __name__ == "__main__":
    main()
