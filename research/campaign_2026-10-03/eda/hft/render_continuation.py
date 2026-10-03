from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OWN = Path(__file__).resolve().parent
QUANTA = [0.12543, 0.24133, 0.46854, 0.65687]


def render_all() -> list[Path]:
    """Render small saved-result figures; never reread raw datasets or fit."""
    table = pd.read_csv(OWN / "quote_identity_v1.csv")
    h2 = json.loads((OWN / "causal_sign_scale_summary_v1.json").read_text())
    h3 = json.loads((OWN / "spread_ablation_summary_v1.json").read_text())
    figures = []
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    x = np.arange(len(table))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(x, table.negative_spread_fraction * 100, color="tab:red")
    axes[0].set_ylabel("Negative sum proxy (%)")
    axes[0].set_title("H1: exact price identities fail on own-unseen sequences")
    for axis, observed, control, title in [
        (axes[1], "state_purity", "state_control_mean", "Repeated current states"),
        (
            axes[2],
            "transition_purity",
            "transition_control_mean",
            "Repeated transitions",
        ),
    ]:
        axis.plot(x, table[observed], "o-", label="Observed modal agreement")
        axis.plot(x, table[control], "s--", label="Matched shuffle (20 draws)")
        axis.axhline(
            0.99, color="black", linestyle=":", label="Frozen exactness threshold"
        )
        axis.set_ylabel(title + "\nagreement")
        axis.set_ylim(0, 1.04)
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(alpha=0.2)
    axes[2].set_xticks(
        x,
        [f"{g}\nq={q:.3f}" for g, q in zip(table.row_group, table.q)],
        rotation=65,
        ha="right",
        fontsize=8,
    )
    axes[2].set_xlabel("Validation row group and target quantum (diagnostics only)")
    fig.tight_layout()
    path = OWN / "fig06_identity_falsifier_v1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(path)

    names = ["spread", "momentum20", "reversion20", "momentum100", "reversion100"]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for offset, scale, color in [
        (-0.18, "estimated", "tab:blue"),
        (0.18, "fixed", "tab:orange"),
    ]:
        values = [h2["pooled_wp"][f"{name}_{scale}"]["mean"] for name in names]
        axes[0].bar(
            positions + offset,
            values,
            width=0.36,
            color=color,
            label=f"{scale} quantum scale",
        )
    axes[0].set_xticks(positions, names, rotation=25, ha="right")
    axes[0].set_title("H2: pooled clipped WP; 38,294 scored rows")
    axes[0].set_ylabel("Mean target WP")
    axes[0].legend(fontsize=9)
    positions = np.arange(4)
    for offset, name, color in [
        (-0.18, "momentum100_estimated", "tab:blue"),
        (0.18, "spread_estimated", "tab:green"),
    ]:
        values = [h2["wp_by_diagnostic_regime"][str(r)][name]["mean"] for r in range(4)]
        axes[1].bar(positions + offset, values, width=0.36, color=color, label=name)
    axes[1].set_xticks(positions, [f"q={q:.3f}" for q in QUANTA])
    axes[1].set_title("Diagnostic strata: largest quantum reverses sign")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.7)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = OWN / "fig07_causal_sign_scale_v1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(path)

    names = ["current_spread", "regime_only", "warmup_spread"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, (name, color) in enumerate(zip(names, colors)):
        axes[0].bar(index, h3["pooled_wp"][name]["mean"], color=color)
        strata = sorted(h3["wp_by_diagnostic_regime"], key=int)
        values = [h3["wp_by_diagnostic_regime"][r][name]["mean"] for r in strata]
        axes[1].bar(
            np.arange(len(strata)) + (index - 1) * 0.25,
            values,
            width=0.25,
            color=color,
            label=name,
        )
    axes[0].set_xticks(range(3), names, rotation=20, ha="right")
    axes[0].set_ylabel("Mean target WP")
    axes[0].set_title(f"H3: new 12-sequence cohort, {h3['scored_rows']:,} scored rows")
    axes[1].set_xticks(range(len(strata)), [f"q={QUANTA[int(r)]:.3f}" for r in strata])
    axes[1].set_title("Current book versus causal constants")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.7)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = OWN / "fig08_spread_ablation_v1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(path)
    return figures


if __name__ == "__main__":
    render_all()
