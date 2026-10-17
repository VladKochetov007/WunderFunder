"""Plot the local F0 R3 block fits and inference-only alpha screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main() -> None:
    """Render the grade-only comparison without opening prediction payloads."""
    evidence_dir = Path(__file__).resolve().parent
    grade_path = evidence_dir / "geometry_r3_grade.json"
    grade: dict[str, Any] = json.loads(grade_path.read_text(encoding="utf-8"))
    scores = {row["name"]: float(row["wp"]) for row in grade["ranking"]}
    parent = scores["parent"]

    labels = [
        "Parent\nreference",
        "R2 full\nfeature fit",
        "R3 moments\nfit",
        "R3 dictionary\nfit",
        "α = −0.50\ninference only",
        "α = −0.25\ninference only",
        "α = +0.25\ninference only",
        "α = +0.50\ninference only",
    ]
    names = [
        "parent",
        "r2_full_geometry",
        "r3_moments_only",
        "r3_dictionary_only",
        "r2_scale_m0p50",
        "r2_scale_m0p25",
        "r2_scale_p0p25",
        "r2_scale_p0p50",
    ]
    deltas = [scores[name] - parent for name in names]
    fit_color = "#31688e"
    alpha_color = "#c98216"
    parent_color = "#59636e"
    selected_color = "#087f5b"

    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.axhline(0.0, color="#30343b", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(3.5, color="#aab0b6", linewidth=1.0, linestyle=":", zorder=1)

    for idx, delta in enumerate(deltas):
        color = parent_color if idx == 0 else fit_color if idx < 4 else alpha_color
        marker = "s" if idx == 0 else "o"
        size = 78 if idx == 0 else 66
        ax.scatter(idx, delta, color=color, marker=marker, s=size, zorder=3)
        offset = 8 if delta >= 0 else -14
        ax.annotate(
            f"{delta:+.5f}",
            (idx, delta),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8.5,
            color="#20252b",
        )

    parent_ci = grade["top_variant_vs_fixed_references"]["r2_scale_m0p50_minus_parent"]["ci95"]
    selected_delta = deltas[4]
    ax.errorbar(
        4,
        selected_delta,
        yerr=[[selected_delta - parent_ci[0]], [parent_ci[1] - selected_delta]],
        fmt="*",
        markersize=15,
        markerfacecolor=selected_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        ecolor=selected_color,
        elinewidth=1.8,
        capsize=4,
        zorder=5,
    )

    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylabel("WP difference from fixed parent", fontsize=10)
    ax.set_ylim(-0.00095, 0.00052)
    ax.set_xlim(-0.55, 7.55)
    ax.grid(axis="y", color="#dfe3e8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Local F0 geometry variants: score difference from the fixed parent",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax.text(
        0.0,
        1.01,
        "468 previously exposed groups · one F0/seed3 lineage · unchanged global_wp_clip2_v1",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4f5964",
    )
    legend = [
        Line2D([], [], marker="s", linestyle="None", color=parent_color, label="fixed parent"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=fit_color,
            label="separately fitted R2/R3 endpoints",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=alpha_color,
            label="inference-only R2 α aliases",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            color=selected_color,
            markersize=12,
            label="α = −0.50; conditional 95% component CI",
        ),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8.5)
    fig.text(
        0.5,
        0.015,
        "The α variants reuse one R2 fit; the R3 block fits reuse the same parent and seed. "
        "Only α = −0.50 has an interval shown: a 2,000-draw component bootstrap conditional on fixed predictions. "
        "It crosses zero and is not adjusted for α selection, fitting-seed/fold uncertainty, or public transfer.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4f5964",
        wrap=True,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.29)
    fig.savefig(evidence_dir / "geometry_r3_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
