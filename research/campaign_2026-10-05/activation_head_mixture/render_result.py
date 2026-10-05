"""Render every fixed E477 head/dropout arm and registered contrast."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Build a report and shared-scale figure from the independent result."""
    result = json.loads((HERE / "eda/twelve_v6b/RESULT.json").read_text())
    conditions = (
        "gelu_p0",
        "gelu_p01",
        "channel_split_p0",
        "channel_split_p01",
        "same_channel_mix_p0",
        "same_channel_mix_p01",
    )
    labels = (
        "GELU\np=0",
        "GELU\np=.1",
        "Split\np=0",
        "Split\np=.1",
        "Mixture\np=0",
        "Mixture\np=.1",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, seed in zip(axes, (271828, 271829), strict=True):
        values = [result["arms"][f"seed{seed}_{arm}"]["global_wp"] for arm in conditions]
        axis.bar(
            range(6),
            values,
            color=["#526478", "#8091a0", "#3384a8", "#6bacc4", "#548552", "#8bad89"],
        )
        axis.set_xticks(range(6), labels)
        axis.axhline(values[0], color="#526478", linestyle="--", linewidth=1)
        axis.set(title=f"Seed {seed}; fixed AdamW", ylim=(0.4, 0.53))
        for index, value in enumerate(values):
            axis.text(index, value + 0.001, f"{value:.4f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Official global WP; 16 exposed components")
    fig.suptitle("E477: all 12 activation-mixture and head-dropout fits")
    fig.tight_layout()
    (HERE / "eda").mkdir(exist_ok=True)
    fig.savefig(HERE / "eda/all_head_mixture_arms.png", dpi=160)
    plt.close(fig)
    lines = [
        "# E477: mixed head activations and dropout",
        "",
        "All12 full fits and independent CPU grading completed without a new OOM. "
        "The half-width GELU/LeakyReLU head changes global WP by −0.017395 and "
        "+0.006980 across the two seeds without dropout. The same-channel mixture "
        "changes it by −0.005748 and +0.005261. Thus head-mixture effects are "
        "seed-dependent at this scale; neither a universal advantage nor a whole-direction "
        "rejection is supported.",
        "",
        "The mixture-by-dropout interaction is +0.006146 and +0.000753; the "
        "two conditional intervals differ. This is an interaction comparison, "
        "not necessarily improvement over the matched GELU head. Every arm uses "
        "H64 two-layer61058-parameter topology, the same24 fit/16 exposed test "
        "components,474 updates and seed-specific common initialization/order. "
        "The head mixture is fixed, not learned routing.",
        "",
        "| Seed | Activation | Dropout | Global WP | t0 | t1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for value in result["arms"].values():
        lines.append(
            f"| {value['seed']} | {value['activation']} | {value['head_dropout']} | "
            f"{value['global_wp']:.7f} | {value['target_wp']['t0']:.7f} | "
            f"{value['target_wp']['t1']:.7f} |"
        )
    lines += [
        "",
        "| Contrast | Family | Global WP delta | Conditional95% interval |",
        "|---|---|---:|---|",
    ]
    for name, value in result["contrasts"].items():
        low, high = value["ci95"]
        lines.append(
            f"| {name} | {value['family']} | {value['mean_diff']:+.7f} | "
            f"[{low:+.7f}, {high:+.7f}] |"
        )
    lines += [
        "",
        "All14 primary and4 secondary contrasts are retained. The joint interactions "
        "use identical numeric-sorted component draws across their four arms. "
        "Intervals condition on fitted models and do not include seed/search "
        "uncertainty or multiplicity correction. No public score is predicted.",
        "",
        "The grader independently reconstructs the heads, recomputes training-only "
        "normalization and validates actual fit/runner/source/checkpoint receipts. "
        "Raw-finite checks precede clipping. Two grader failures (cohort-role "
        "comparison and export-path metadata) and unexecuted corrections remain "
        "in separate incident notes; no model fit was rerun or discarded. "
        "Successful grade attempt: e477-head-mixture-cpu-test-grade-v6b, "
        "167.5934s. Raw result and paired export identity: eda/twelve_v6b/RESULT.json. "
        "Independent export reproduction is recorded separately when complete.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
