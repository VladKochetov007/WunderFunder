"""Render the independently verified E487 fixed-gradient diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    """Read saved proof and scores only; never load data or rerun an optimizer."""
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    proof = json.loads(
        (root / "research/campaign_2026-10-05/checks/e487_saved_delta_reproduction.json").read_text()
    )
    result_path = here / "gpu_probe_v2/result.json"
    if proof["status"] != "all_checked_values_match":
        raise ValueError("independent arithmetic proof has not passed")
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != proof["result_sha256"]:
        raise ValueError("verified result changed")
    result = json.loads(result_path.read_text())
    gates = result["crossed_replay_arithmetic"]["strict_native_custom_update_gates"]
    names = list(gates["bf16_source"]["metrics"])
    if names != list(gates["fp32_diagnostic"]["metrics"]):
        raise ValueError("precision-arm metric inventories differ")
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    for precision, offset, color, label in (
        ("bf16_source", -0.12, "#cc6b28", "BF16 orthogonalization"),
        ("fp32_diagnostic", 0.12, "#237956", "FP32 orthogonalization"),
    ):
        rows = [gates[precision]["metrics"][name] for name in names]
        axes[0].scatter(
            [row["relative_error_rms"] * 100 for row in rows],
            positions + offset,
            color=color,
            label=label,
        )
        axes[1].scatter(
            [row["max_abs_error"] for row in rows], positions + offset, color=color
        )
    axes[0].axvline(2, color="#a32c32", linestyle="--", label="Registered limit")
    axes[1].axvline(2e-5, color="#a32c32", linestyle="--")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yticks(positions, [name.replace("matrix:", "").replace("family:", "") for name in names])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Native/custom update RMS discrepancy (%)")
    axes[1].set_xlabel("Maximum absolute update discrepancy")
    axes[0].legend(loc="lower right", fontsize=9)
    figure.suptitle("E487: orthogonalization precision changes one-step parity\nFit-only fixed batch; not a training or public-score comparison")
    output = here / "eda/verified_v4"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "all_matrix_and_family_update_gates.png", dpi=180)
    plt.close(figure)
    print("Rendered all seven registered update groups for both precision paths.")


if __name__ == "__main__":
    main()
