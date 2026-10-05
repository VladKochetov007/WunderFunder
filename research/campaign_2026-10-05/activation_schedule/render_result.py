"""Render the complete E476 activation/optimizer/schedule factorial."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Save all scores and every registered contrast, without selecting a winner."""
    result = json.loads((HERE / "eda/twelve_v1/scores.json").read_text())
    schedules = ("constant_peak", "warmcos_47_427", "constant_mean_warmcos")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, optimizer in zip(axes, ("adamw", "muon_adamw"), strict=True):
        for offset, activation, color in (
            (-0.18, "base", "#526478"),
            (0.18, "sine_embed", "#3384a8"),
        ):
            values = [
                result["arm_scores"][f"{activation}__{optimizer}__{schedule}"]["wp"]
                for schedule in schedules
            ]
            positions = [index + offset for index in range(3)]
            axis.bar(positions, values, width=0.36, color=color, label=activation)
            for position, value in zip(positions, values, strict=True):
                axis.text(position, value + 0.001, f"{value:.4f}", ha="center", fontsize=9)
        axis.set_xticks(
            range(3), ("Constant peak\nLR .001", "47 warm + 427 cosine", "Constant mean\nLR .00055")
        )
        axis.set_title(optimizer)
        axis.set_ylim(0.42, 0.54)
        axis.legend()
    axes[0].set_ylabel("Official global WP; 16 exposed components")
    fig.suptitle("E476: all 12 fixed fits; one seed, 474 updates")
    fig.tight_layout()
    fig.savefig(HERE / "eda/all_schedule_arms.png", dpi=160)
    plt.close(fig)
    lines = [
        "# E476: learning-rate shape changes the activation comparison",
        "",
        "All12 fixed fits and independent CPU scoring finished without a new "
        "OOM. Warmup/cosine beats the constant-at-equal-mean control by "
        "+0.067328 (GELU/AdamW) and +0.072700 (sine embedding/AdamW); both "
        "conditional component intervals are positive. The corresponding "
        "Muon gains are +0.015065/+0.016691, with intervals crossing zero. "
        "These are one-seed exposed-panel results, not public gains.",
        "",
        "The activation ranking depends on the LR protocol. Sine embedding "
        "loses to GELU under constant peak but improves under warmup/cosine. "
        "The activation-by-schedule interactions are +0.005372 (AdamW) and "
        "+0.001626 (Muon), both with positive conditional intervals. This "
        "is evidence against selecting activation independently of the "
        "optimizer schedule; independent seed replication is still needed.",
        "",
        "Both the warmup/cosine and constant-mean traces sum to0.2607. "
        "This matches arithmetic LR area, not actual optimizer update area. "
        "All recipes have61,058 parameters, common initialization/order, "
        "24 fit components and16 previously exposed diagnostic components. "
        "The strong public model uses different capacity/data/training. "
        "The large AdamW difference warrants reproduction, not extrapolation.",
        "",
        "| Registered arm | Global WP | t0 | t1 |",
        "|---|---:|---:|---:|",
    ]
    for name, value in result["arm_scores"].items():
        lines.append(f"| {name} | {value['wp']:.7f} | {value['t0']:.7f} | {value['t1']:.7f} |")
    lines += [
        "",
        "| Registered contrast | Global WP delta | Conditional95% interval |",
        "|---|---:|---|",
    ]
    for name, value in {**result["pair_contrasts"], **result["interactions"]}.items():
        lo, hi = value["ci95"]
        lines.append(f"| {name} | {value['mean_diff']:+.7f} | [{lo:+.7f}, {hi:+.7f}] |")
    lines += [
        "",
        "All20 pairs and2 jointly resampled interactions are retained, with "
        "no seed pooling or multiplicity correction. The independent scorer "
        "checks actual runner/spec/runtime/artifact receipts, raw selected "
        "target masks, finite predictions before clipping, exact LR receipts, "
        "fit-only normalization and checkpoint identity. A larger-memory "
        "CPU-helper admission was queued until E478 scoring drained; it "
        "then completed127.2383s, peak0.965GB RAM. No recipe was changed.",
        "",
        "Source review and additional independent export reproduction are "
        "recorded separately. Raw official result: eda/twelve_v1/scores.json. "
        "Figure: eda/all_schedule_arms.png. Fixed gradient/update telemetry "
        "is in the fit receipt and is descriptive, never a test-based tuner.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
