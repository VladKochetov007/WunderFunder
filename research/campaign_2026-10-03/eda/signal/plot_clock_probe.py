from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from clock_probe import scores

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Plot measured scores and clock-control diagnostics from saved arrays."""
    os.environ["MPLCONFIGDIR"] = str(HERE / "mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((HERE / "clock_probe_results.json").read_text())
    saved = np.load(HERE / "clock_probe_stats.npz")
    names = list(saved["configs"])
    pooled = np.array([[result["configs"][c][t] for t in ("t0", "t1")] for c in names])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(names))
    for j, target in enumerate(("t0", "t1")):
        axes[0].bar(positions + (j - 0.5) * 0.34, pooled[:, j], 0.34, label=target)
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("Pooled clipped target-weighted correlation")
    axes[0].set_title("Frozen fast-band clock comparison")
    axes[0].legend()
    individual = scores(saved["stats"]).mean(-1)
    delta = individual[:, 1] - individual[:, 0]
    axes[1].scatter(np.arange(len(delta)), delta, s=16, alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvline(31.5, color="grey", ls="--")
    axes[1].set_xlabel("Validation sequence in frozen sampling order")
    axes[1].set_ylabel("Within-sequence mean WP: own minus shared")
    axes[1].set_title("Diagnostic only; not an independent-row test")
    fig.suptitle("64 validation sequences; prior EDA exposed; no baseline comparison")
    fig.tight_layout()
    fig.savefig(HERE / "fig_clock_probe.png", dpi=130)
    plt.close(fig)

    activity = saved["activity"]
    groups = list(saved["groups"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rate = activity[:, 0]
    off_shared = activity[:, 1]
    axes[0].bar(np.arange(len(groups)), np.mean(off_shared, axis=0))
    axes[0].set_xticks(np.arange(len(groups)), groups)
    axes[0].set_ylabel("Fraction of rows: group change without i0_p change")
    axes[0].set_title("Innovations absent from shared clock")
    for j, group in enumerate(groups):
        axes[1].scatter(activity[:, 2, j], rate[:, j], s=12, alpha=0.45, label=group)
    axes[1].plot([0, 1], [0, 1], color="black", lw=0.8)
    axes[1].set_xlabel("Event rate estimated from rows 1..98")
    axes[1].set_ylabel("Observed event rate over rows 99..19999")
    axes[1].set_title("Periodic control matches prefix rates approximately")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_clock_activity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
