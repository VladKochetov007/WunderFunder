"""Render every E478 scaling arm and its registered paired contrast."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Create a full-table report and a shared-scale figure without selecting arms."""
    result = json.loads((HERE / "eda/six_v1/RESULT.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, seed in zip(axes, (271828, 271829, 271830), strict=True):
        values = [result["arms"][f"seed{seed}_alpha{alpha:.1f}"]["global_wp"] for alpha in (0, 0.1)]
        axis.bar((0, 1), values, color=("#526478", "#3384a8"))
        axis.set_xticks((0, 1), ("alpha 0", "alpha +0.1"))
        axis.set_title(f"Seed {seed}")
        axis.set_ylim(0.46, 0.54)
        for index, value in enumerate(values):
            axis.text(index, value + 0.001, f"{value:.4f}", ha="center")
    axes[0].set_ylabel("Official global WP; 16 exposed components")
    fig.suptitle("E478: bounded candidate, two-layer H64, fixed Muon recipe")
    fig.tight_layout()
    fig.savefig(HERE / "eda/all_scaling_arms.png", dpi=160)
    plt.close(fig)
    lines = [
        "# E478: bounded candidate scaling to the two-layer topology",
        "",
        "Allsix fixed Muon fits and the independent CPU grader completed with "
        "no new OOM. The larger topology retains a positive point-estimate "
        "gain at allthree seeds: +0.0027025, +0.0106748 and +0.0044788. "
        "Two conditional component intervals exclude zero; the third crosses it. "
        "This supports further testing, without proving a public gain.",
        "",
        "The modification is n=u+0.1*sin(pi*u), u=tanh(v), in both GRU layers. "
        "The alpha-zero custom-backend control shares the same FP32 policy, "
        "61,058 parameters, base initialization, fit-only preprocessing, "
        "shuffled order and474 updates. The24 training and16 diagnostic test "
        "components are distinct but previously exposed. The submission is a "
        "much larger distilled/fine-tuned predictor; these are not its scores.",
        "",
        "| Seed | Alpha | Global WP | t0 | t1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in result["arms"].values():
        lines.append(
            f"| {value['seed']} | {value['alpha']:.1f} | {value['global_wp']:.7f} | "
            f"{value['target_wp']['t0']:.7f} | {value['target_wp']['t1']:.7f} |"
        )
    lines += [
        "",
        "| Registered contrast | Global WP delta | Conditional95% interval |",
        "|---|---:|---|",
    ]
    for name, value in result["contrasts"].items():
        lo, hi = value["ci95"]
        lines.append(f"| {name} | {value['mean_diff']:+.7f} | [{lo:+.7f}, {hi:+.7f}] |")
    lines += [
        "",
        "The grader independently reconstructs reset-after hidden-bias dynamics, "
        "carries [2,B,64] state across chunks, resets per sequence, checks all "
        "sealed sources/inputs, runtime/artifact receipts and raw finite selected "
        "targets/predictions. It recomputes external normalization from exactly "
        "the24 fit groups and verifies the checkpoint's internal buffers remain "
        "identity. A wrong pre-execution buffer assumption was caught and fixed "
        "before the scorer seal; no scientific fit was altered.",
        "",
        "Allsix official scores and allthree paired comparisons are retained. "
        "Component intervals condition on fitted models, not seed or search "
        "uncertainty; there is no cross-seed pooled CI. Independent export reproduction "
        "passed for allsix scores and allthree 2,000-draw contrasts. See eda/six_v1/RESULT.json and "
        "eda/all_scaling_arms.png. Fit wall time285.9948s; independent CPU "
        "grade135.1611s. Fit peak0.398GB GPU/1.609GB RAM; CPU grade0.598GB RAM.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
