"""Render every recurrent candidate recipe across all three fixed seeds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Plot all recipes and write tables directly from the verified grader."""
    here = Path(__file__).parent
    result = json.loads((here / "THREE_SEED_ACTIVATION_RESULT.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, optimizer in zip(axes, ("adamw", "muon_adamw"), strict=True):
        for seed in result["seeds"]:
            values = [
                result["attempts_and_scores"][str(seed)]["scores"][f"{alpha}__{optimizer}"]["wp"]
                for alpha in ("m10", "zero", "p10")
            ]
            axis.plot((-0.1, 0, 0.1), values, marker="o", label=f"seed {seed}")
        axis.set_xticks((-0.1, 0, 0.1))
        axis.set_xlabel("Bounded recurrent candidate alpha")
        axis.set_title(optimizer)
        axis.legend()
    axes[0].set_ylabel("Official pooled WP; exposed diagnostic panel")
    fig.suptitle("All 18 recurrent activation fits; no public-score forecast")
    fig.tight_layout()
    output = here / "eda"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "all_recurrent_seeds.png", dpi=170)
    plt.close(fig)
    lines = [
        "# E471/E475: all recurrent activation recipes across three seeds",
        "",
        "All18 fixed fits completed with successful teardown and no OOM. "
        "The independently written grader validated seals/artifacts, exact "
        "raw selected targets/masks, finite predictions and official global/target scores.",
        "",
        "| Seed | Recipe | Global WP | t0 | t1 |",
        "|---|---|---:|---:|---:|",
    ]
    for seed in result["seeds"]:
        for arm, score in result["attempts_and_scores"][str(seed)]["scores"].items():
            lines.append(
                f"| {seed} | {arm} | {score['wp']:.7f} | {score['t0']:.7f} | {score['t1']:.7f} |"
            )
    lines += [
        "",
        "| Fixed contrast | WP delta | Conditional95% component interval |",
        "|---|---:|---|",
    ]
    for name, item in result["within_seed_contrasts"].items():
        low, high = item["ci95"]
        lines.append(f"| {name} | {item['mean_diff']:+.7f} | [{low:+.7f}, {high:+.7f}] |")
    lines += [
        "",
        "Positive alpha improves Muon at allthree seeds: +0.0061420, "
        "+0.0152788 and+0.0115503, with positive conditional component intervals. "
        "Negative alpha hurts Muon at allthree. AdamW effects vary in sign. "
        "The original apparent negative-alpha/AdamW advantage does not reproduce "
        "consistently. Both signs, all optimizers and all21 comparisons are retained.",
        "",
        "These are common-FP32 GRU32 fits on the same exposed16components. "
        "No pooled CI, multiplicity correction, fresh-generalization or public "
        "prediction follows. The next discriminating test is transfer to the "
        "two-layer incumbent topology at larger width, with an alpha-zero "
        "custom-backend control and fixed seeds. Historical default-precision "
        "backend diagnostics remain separate in the original E471 report.",
        "",
        "See THREE_SEED_ACTIVATION_RESULT.json for exact identities and "
        "eda/all_recurrent_seeds.png for the complete matrix.",
    ]
    report = here / "THREE_SEED_REPORT.md"
    if report.exists():
        raise FileExistsError("refusing to overwrite the result report")
    report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
