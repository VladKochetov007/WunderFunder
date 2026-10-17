"""Render the fixed output-policy audit without fitting or selecting policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    """Plot observed deltas and conditional chain-bootstrap intervals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    policies = [
        ("hold_i0_and", "Держать до AND-события"),
        ("hold_i0_or", "Держать до OR-события"),
        ("cadence2", "Обновлять каждые 2 строки"),
        ("cadence4", "Каждые 4 строки"),
        ("cadence8", "Каждые 8 строк"),
        ("blend_half_and", "50% исходный + 50% AND-held"),
        ("ema_raw_tau4", "EMA α = 1/4"),
        ("ema_raw_tau16", "EMA α = 1/16"),
        ("ema_raw_tau64", "EMA α = 1/64"),
    ]
    labels = ["t384 / seed 3", "t384 / seed 4", "tied-lead / seed 3"]
    colors = ["#176b9d", "#3b926a", "#bc6434"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.7), gridspec_kw={"width_ratios": [3, 1]})
    fig.subplots_adjust(left=0.25, right=0.97, top=0.78, bottom=0.22, wspace=0.25)
    for model_index, model in enumerate(data["models"]):
        base = model["scenarios"]["baseline"]["global"]["wp"]
        intervals = data["paired_chain_bootstrap"]["intervals_by_model_index"][str(model_index)]
        for axis_index, entries in enumerate((policies, [("oracle_clipped_label_run", "Oracle")])):
            observed, lower, upper = [], [], []
            for name, _ in entries:
                value = model["scenarios"][name]["global"]["wp"] - base
                interval = intervals[name]["wp"]
                observed.append(value)
                lower.append(value - interval["q025"])
                upper.append(interval["q975"] - value)
            positions = np.arange(len(entries)) + (model_index - 1) * 0.18
            axes[axis_index].errorbar(
                observed,
                positions,
                xerr=[lower, upper],
                fmt="o",
                ms=5,
                capsize=3,
                lw=1.2,
                color=colors[model_index],
                label=labels[model_index],
            )
    for axis in axes:
        axis.grid(axis="x", color="#e1e5e8", lw=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlabel("Изменение WP к своему исходному массиву")
    axes[0].axvline(0, color="#505860", lw=1)
    axes[0].set_yticks(np.arange(len(policies)), [label for _, label in policies])
    axes[0].set_ylim(len(policies) - 0.5, -0.5)
    axes[0].set_xlim(-0.18, 0.005)
    axes[0].set_title("Причинные преобразования выхода", loc="left", fontsize=12, pad=18)
    axes[1].set_yticks([])
    axes[1].set_ylim(0.8, -0.8)
    axes[1].set_xlim(0.020, 0.040)
    axes[1].set_xticks([0.02, 0.03, 0.04])
    axes[1].set_title("Oracle: недоступен онлайн", fontsize=12, pad=18)
    axes[1].set_xlabel("Ретроспективный ΔWP")
    fig.suptitle(
        "Постоянная метка не оправдывает постоянный прогноз",
        x=0.25,
        y=0.96,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.25,
        0.89,
        "467 последовательностей · 846 654 оцениваемые строки · один exposed fold",
        color="#535f67",
        fontsize=11,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.62, 0.065),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.25,
        0.025,
        "Полосы: условные 95% интервалы, 512 парных bootstrap по 143 цепочкам.\n"
        "Скрытые состояния не менялись. Clipped-input EMA также отрицательны; полная матрица в отчёте.",
        fontsize=9,
        color="#535f67",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = hashlib.sha256(args.results.read_bytes()).hexdigest()
    for suffix in (".png", ".svg"):
        fig.savefig(
            args.output.with_suffix(suffix),
            dpi=170,
            metadata={"Description": "Source RESULTS.json SHA-256: " + receipt},
        )
        if suffix == ".svg":
            svg_path = args.output.with_suffix(suffix)
            svg_path.write_text(
                "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
