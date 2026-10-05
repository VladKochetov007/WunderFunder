"""Render all registered E474 arms and contrasts from the sealed grader result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Save the complete result table and a shared-scale figure."""
    result_path = HERE / "scores/twelve_cpu_v1/score_result.json"
    result = json.loads(result_path.read_text())
    arms = ("base", "sine_embed", "slope_embed", "sine_head", "slope_head", "sine_both")
    output = HERE / "eda"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, seed in zip(axes, result["seed_order"], strict=True):
        scores = [result["arm_scores"][f"seed{seed}_{arm}"]["wp"] for arm in arms]
        axis.bar(range(len(arms)), scores, color=["#526478", *["#3384a8"] * 5])
        axis.set_xticks(range(len(arms)), [arm.replace("_", "\n") for arm in arms])
        axis.set_title(f"Seed {seed}; fixed AdamW")
        axis.axhline(scores[0], color="#526478", linestyle="--", linewidth=1)
        axis.set_ylim(0.4, 0.52)
        for index, value in enumerate(scores):
            axis.text(index, value + 0.001, f"{value:.4f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Official global WP; 16 exposed components")
    fig.suptitle("E474: activation placement, all 12 fixed fits")
    fig.tight_layout()
    fig.savefig(output / "all_placement_arms.png", dpi=160)
    plt.close(fig)
    lines = [
        "# E474: localized activation changes on a two-layer GRU",
        "",
        "All12 fixed fits and independent CPU scoring completed with no new OOM. "
        "The score is official global_wp_clip2_v1 on16 previously exposed diagnostic "
        "components (26,990 selected rows); this is neither the strong submission "
        "baseline nor a public expectation.",
        "",
        "Sine in the embedding changes global WP by −0.004506 and +0.002476 "
        "across the two seeds. Sine in the head changes it by −0.000570 and "
        "−0.000072. Thus the earlier GRU32 sine advantage does not transfer "
        "uniformly to this topology/placement/optimizer. Sine versus matched "
        "linear slope in the embedding remains positive at both seeds, but that "
        "relative comparison does not imply improvement over ordinary GELU.",
        "",
        "| Seed | Arm | Global WP | t0 | t1 |",
        "|---|---|---:|---:|---:|",
    ]
    for value in result["arm_scores"].values():
        lines.append(
            f"| {value['seed']} | {value['arm']} | {value['wp']:.7f} | "
            f"{value['t0']:.7f} | {value['t1']:.7f} |"
        )
    lines += [
        "",
        "| Seed | Registered contrast | Global WP delta | Conditional95% interval |",
        "|---|---|---:|---|",
    ]
    for seed, contrasts in result["paired_contrasts_by_seed"].items():
        for name, value in contrasts.items():
            lo, hi = value["bootstrap"]["wp"]["ci95"]
            delta = value["point_delta"]["wp"]
            lines.append(f"| {seed} | {name} | {delta:+.7f} | [{lo:+.7f}, {hi:+.7f}] |")
    lines += [
        "",
        "All14 contrasts are retained, including the joint placement interaction. "
        "Intervals resample the same components across all arms and condition on "
        "fitted models; they do not include seed/search uncertainty or correct "
        "multiplicity. No winner was used to alter the registered fits.",
        "",
        "The independent grader reconstructs each model and recomputes fit-only "
        "normalization from sealed training rows. It checks runner receipts, raw "
        "selected targets/masks and finite predictions before official clipping. "
        "See scores/twelve_cpu_v1/score_result.json for exact identities and "
        "eda/all_placement_arms.png for all arms. Admission-only errors and the "
        "earlier probe artifact error are preserved separately; no failed fit "
        "was counted as completed.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
