"""Render frozen E484 V2 grade outputs without recomputing any scores."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = ROOT / "research/campaign_2026-10-05/activation_context_mixture"
EDA = HERE / "eda/twelve_v1"
SCORES = EDA / "scores.json"
EXPORT = EDA / "paired_components.npz"
GRADE_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e484-context-mixture-independent-cpu-grade-v2.json"
)
FIT_ATTEMPT = (
    ROOT / "research/campaign_2026-10-04/runner/attempts/e484-context-mixture-twelve-fit-v2.json"
)
REPRODUCTION = (
    ROOT / "research/campaign_2026-10-05/checks/context_activation_export_reproduction.json"
)
REPORT = HERE / "REPORT.md"
SEEDS = (271_834, 271_835)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix", "projection_mix", "self_mix")
CONTRASTS = (
    "recurrent_minus_static",
    "recurrent_minus_projection",
    "recurrent_minus_self",
    "static_minus_fixed",
    "recurrent_minus_fixed",
    "projection_minus_fixed",
    "self_minus_fixed",
    "fixed_minus_gelu",
)
ARM_LABELS = {
    "gelu": "GELU",
    "fixed_mix": "Fixed 50/50",
    "static_mix": "Learned static",
    "recurrent_mix": "Recurrent context",
    "projection_mix": "Projection context",
    "self_mix": "Head-self context",
}
CONTRAST_LABELS = {
    "recurrent_minus_static": "Recurrent − static",
    "recurrent_minus_projection": "Recurrent − projection",
    "recurrent_minus_self": "Recurrent − self",
    "static_minus_fixed": "Static − fixed",
    "recurrent_minus_fixed": "Recurrent − fixed",
    "projection_minus_fixed": "Projection − fixed",
    "self_minus_fixed": "Self − fixed",
    "fixed_minus_gelu": "Fixed mixture − GELU",
}
PRIMARY = set(CONTRASTS[:3])
EXPECTED_RESULT_SHA256 = "6a4862dd07ef28040e6aefd5ccbe4c68d266acee9cd3b999312ecc6131d13545"
EXPECTED_EXPORT_SHA256 = "cfd0c4807249ae37e641fe329e8227a6c110990dc22f9c2fd4c94dfa2de81c81"
EXPECTED_GRADE_ATTEMPT_SHA256 = "9f302e6ad2e18fc3a68932f830bddb7cdcf515e0484a7615e81beab080380623"
EXPECTED_FIT_ATTEMPT_SHA256 = "1fa8d74b94b37f76dc0b2bd8410919a40bff7ade234d36131334f88628ad8bf2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = json.loads(SCORES.read_text())
    grade = json.loads(GRADE_ATTEMPT.read_text())
    fit = json.loads(FIT_ATTEMPT.read_text())
    reproduction = json.loads(REPRODUCTION.read_text())
    require(sha256_file(SCORES) == EXPECTED_RESULT_SHA256, "E484 score JSON hash changed")
    require(sha256_file(EXPORT) == EXPECTED_EXPORT_SHA256, "E484 raw paired export hash changed")
    require(
        sha256_file(GRADE_ATTEMPT) == EXPECTED_GRADE_ATTEMPT_SHA256,
        "E484 official grade attempt hash changed",
    )
    require(
        sha256_file(FIT_ATTEMPT) == EXPECTED_FIT_ATTEMPT_SHA256, "E484 fit attempt hash changed"
    )
    require(
        grade.get("id") == "e484-context-mixture-independent-cpu-grade-v2"
        and grade.get("status") == "finished"
        and grade.get("exit_code") == 0
        and grade.get("teardown_verified") is True
        and grade.get("peak_observed_vram_bytes") == 0
        and grade.get("unit_properties", {}).get("Result") == "success"
        and grade.get("unit_properties", {}).get("ExecMainStatus") in (0, "0"),
        "E484 grade runner receipt is not a clean completed CPU grade",
    )
    require(
        fit.get("id") == "e484-context-mixture-twelve-fit-v2"
        and fit.get("status") == "finished"
        and fit.get("exit_code") == 0
        and fit.get("teardown_verified") is True,
        "E484 fit runner receipt is not clean",
    )
    for record, label in ((grade, "grade"), (fit, "fit")):
        before = record.get("initial_memory_events", {})
        after = record.get("final_memory_events", {})
        for key in ("oom", "oom_kill", "oom_group_kill"):
            require(
                isinstance(before.get(key), int) and before[key] == after.get(key),
                f"E484 {label} OOM event counter changed: {key}",
            )
    artifacts = grade.get("artifacts", [])
    require(
        [item.get("path") for item in artifacts]
        == [
            "research/campaign_2026-10-05/activation_context_mixture/eda/twelve_v1/scores.json",
            "research/campaign_2026-10-05/activation_context_mixture/eda/twelve_v1/paired_components.npz",
        ]
        and artifacts[0].get("sha256") == EXPECTED_RESULT_SHA256
        and artifacts[1].get("sha256") == EXPECTED_EXPORT_SHA256,
        "E484 grade attempt artifact list differs",
    )
    require(
        result.get("fit_cells") == 12
        and result.get("contrast_count") == 16
        and len(result.get("arm_scores", {})) == 12
        and len(result.get("registered_contrasts", {})) == 16
        and len(result.get("registered_target_contrasts", {})) == 16
        and result.get("no_seed_pooling") is True,
        "E484 result matrix is incomplete",
    )
    require(
        reproduction.get("study") == "E484"
        and reproduction.get("status") == "all_checked_values_match"
        and reproduction.get("arms") == 12
        and reproduction.get("global_contrasts") == 16
        and reproduction.get("target_contrasts") == 32
        and reproduction.get("result_sha256") == EXPECTED_RESULT_SHA256
        and reproduction.get("export_sha256") == EXPECTED_EXPORT_SHA256,
        "independent E484 arithmetic reproduction does not bind this result",
    )
    return result, grade, fit, reproduction


def fmt(value: float) -> str:
    return f"{value:+.6f}"


def build_report(
    result: dict[str, Any],
    grade: dict[str, Any],
    fit: dict[str, Any],
    reproduction: dict[str, Any],
) -> str:
    lines = [
        "# E484 causal context-mixture grade",
        "",
        "This report displays the frozen independent V2 grade for 12 registered fits, 16 global contrasts, and 32 target-specific contrasts. It keeps both seeds separate and retains all six arms. The raw-export reproducer matched all 12 arms and all 48 registered contrasts with maximum absolute error "
        f"**{reproduction['maximum_absolute_error']:.3e}**. It checks arithmetic on saved exports; it is not a second inference run.",
        "",
        "This is a two-seed, one-epoch, H64 exposed-panel diagnostic over the same 16 previously exposed components. It does not establish an activation mechanism or public score improvement. The grader names the anchor contrast as fixed mixture minus GELU; signs below follow candidate minus reference.",
        "",
        "## Run receipts",
        "",
        f"The 12 fits finished in {fit['elapsed_seconds']:.2f}s (peak worker RAM {fit['peak_ram_bytes']:,} bytes; observed GPU peak {fit['peak_observed_vram_bytes']:,} bytes). The independent CPU grade finished in {grade['elapsed_seconds']:.3f}s with peak RAM {grade['peak_ram_bytes']:,} bytes, zero worker and parent OOM-counter deltas, no GPU use, and verified teardown. Each fit used 474 updates. No public forecast follows from this panel.",
        "",
        f"The root raw-export check matched {reproduction['arms']} arms, {reproduction['global_contrasts']} global contrasts, and {reproduction['target_contrasts']} target contrasts (result SHA {reproduction['result_sha256']}; export SHA {reproduction['export_sha256']}).",
        "",
        "## All 12 arm scores",
        "",
        "| Seed | Arm | Global WP | Target 0 | Target 1 | Updates |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for arm in ARMS:
            row = result["arm_scores"][f"seed{seed}__{arm}"]
            lines.append(
                f"| {seed} | {ARM_LABELS[arm]} | {row['wp']:.6f} | {row['t0']:.6f} | {row['t1']:.6f} | {row['updates']} |"
            )
    lines.extend(
        [
            "",
            "![All twelve arm scores for global WP and both targets](eda/twelve_v1/all12_arm_scores.png)",
            "",
            "## All 16 global contrasts",
            "",
            "| Seed | Contrast | Δ Global WP | Paired component 95% CI |",
            "|---:|---|---:|---:|",
        ]
    )
    for seed in SEEDS:
        for contrast in CONTRASTS:
            row = result["registered_contrasts"][f"seed{seed}__{contrast}"]
            lo, hi = row["ci95"]
            lines.append(
                f"| {seed} | {CONTRAST_LABELS[contrast]} | {fmt(row['point_diff'])} | [{fmt(lo)}, {fmt(hi)}] |"
            )
    lines.extend(
        [
            "",
            "![All sixteen global paired contrasts](eda/twelve_v1/all16_global_contrasts.png)",
            "",
            "## All 32 target-specific contrasts",
            "",
            "| Seed | Contrast | Target | Δ WP | Paired component 95% CI |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for seed in SEEDS:
        for contrast in CONTRASTS:
            row = result["registered_target_contrasts"][f"seed{seed}__{contrast}"]
            for target, label in (("t0", "Target 0"), ("t1", "Target 1")):
                metric = row["targets"][target]
                lo, hi = metric["ci95"]
                lines.append(
                    f"| {seed} | {CONTRAST_LABELS[contrast]} | {label} | {fmt(metric['point_diff'])} | [{fmt(lo)}, {fmt(hi)}] |"
                )
    lines.extend(
        [
            "",
            "![All thirty-two target-specific paired contrasts](eda/twelve_v1/all32_target_contrasts.png)",
            "",
            "## What this panel shows",
            "",
            "The three primary recurrent-context comparisons against static, projection, and self-context controls remain seed-specific. Recurrent-minus-static intervals cross zero in both seeds. Recurrent-minus-self reverses sign across seeds: seed 271834 is −0.000129 (CI [−0.000246, −0.000058]), while seed 271835 is +0.000140 (CI [+0.000005, +0.000248]). The direction is not stable across seeds.",
            "",
            "Fixed-mixture-minus-GELU global point estimates are +0.002845 and +0.018051, but both paired intervals include zero. Target-specific results are heterogeneous; the full 32-row table preserves that variation rather than collapsing targets or selecting an arm.",
            "",
            "This report is descriptive for these 12 fits and the previously exposed panel. It does not select a winner, demonstrate that one context contains more information, or imply a public-score gain.",
            "",
            "## Provenance",
            "",
            f"- Score JSON SHA256: {sha256_file(SCORES)}",
            f"- Paired export SHA256: {sha256_file(EXPORT)}",
            f"- Official CPU grade attempt SHA256: {sha256_file(GRADE_ATTEMPT)}",
            f"- Completed fit attempt SHA256: {sha256_file(FIT_ATTEMPT)}",
            f"- Independent arithmetic receipt SHA256: {sha256_file(REPRODUCTION)}",
            "",
            "The renderer validates and displays frozen machine outputs. It does not call the grader, inspect checkpoints, or recompute scores, contrasts, or bootstrap statistics.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_arm_plot(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metrics = (("wp", "Global WP"), ("t0", "Target 0 WP"), ("t1", "Target 1 WP"))
    colors = ("#3569a8", "#d27635")
    x = np.arange(len(ARMS))
    for axis, (key, title) in zip(axes, metrics, strict=True):
        for offset, seed in enumerate(SEEDS):
            values = [result["arm_scores"][f"seed{seed}__{arm}"][key] for arm in ARMS]
            axis.plot(x, values, marker="o", linewidth=1.6, color=colors[offset], label=str(seed))
        axis.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS], rotation=45, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel("WP score")
    axes[0].legend(title="Seed", frameon=False)
    fig.suptitle("E484: all six arms, both seeds")
    fig.tight_layout()
    fig.savefig(EDA / "all12_arm_scores.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_global_contrasts(result: dict[str, Any]) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CONTRASTS))
    offsets = (-0.12, 0.12)
    colors = ("#3569a8", "#d27635")
    for seed_idx, seed in enumerate(SEEDS):
        rows = [result["registered_contrasts"][f"seed{seed}__{contrast}"] for contrast in CONTRASTS]
        point = np.array([row["point_diff"] for row in rows])
        low = np.array([row["ci95"][0] for row in rows])
        high = np.array([row["ci95"][1] for row in rows])
        axis.errorbar(
            x + offsets[seed_idx],
            point,
            yerr=np.vstack((point - low, high - point)),
            fmt="o",
            capsize=3,
            color=colors[seed_idx],
            label=str(seed),
        )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(x, [CONTRAST_LABELS[item] for item in CONTRASTS], rotation=38, ha="right")
    axis.set_ylabel("Candidate − reference global WP")
    axis.set_title("All 16 registered global paired contrasts (95% component bootstrap CI)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Seed", frameon=False)
    fig.tight_layout()
    fig.savefig(EDA / "all16_global_contrasts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_target_contrasts(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    x = np.arange(len(CONTRASTS))
    offsets = (-0.12, 0.12)
    colors = ("#3569a8", "#d27635")
    for axis, target, target_label in zip(
        axes, ("t0", "t1"), ("Target 0", "Target 1"), strict=True
    ):
        for seed_idx, seed in enumerate(SEEDS):
            rows = [
                result["registered_target_contrasts"][f"seed{seed}__{contrast}"]["targets"][target]
                for contrast in CONTRASTS
            ]
            point = np.array([row["point_diff"] for row in rows])
            low = np.array([row["ci95"][0] for row in rows])
            high = np.array([row["ci95"][1] for row in rows])
            axis.errorbar(
                x + offsets[seed_idx],
                point,
                yerr=np.vstack((point - low, high - point)),
                fmt="o",
                capsize=3,
                color=colors[seed_idx],
                label=str(seed),
            )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_ylabel(f"{target_label} Δ WP")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Seed", frameon=False)
    axes[0].set_title(
        "All 32 registered target-specific paired contrasts (95% component bootstrap CI)"
    )
    axes[-1].set_xticks(x, [CONTRAST_LABELS[item] for item in CONTRASTS], rotation=38, ha="right")
    fig.tight_layout()
    fig.savefig(EDA / "all32_target_contrasts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csvs(result: dict[str, Any]) -> None:
    with (EDA / "all12_arm_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "arm", "global_wp", "target0_wp", "target1_wp", "updates"))
        for seed in SEEDS:
            for arm in ARMS:
                row = result["arm_scores"][f"seed{seed}__{arm}"]
                writer.writerow(
                    (seed, ARM_LABELS[arm], row["wp"], row["t0"], row["t1"], row["updates"])
                )
    with (EDA / "all16_global_contrasts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "contrast", "candidate_minus_reference", "ci95_low", "ci95_high"))
        for seed in SEEDS:
            for contrast in CONTRASTS:
                row = result["registered_contrasts"][f"seed{seed}__{contrast}"]
                writer.writerow((seed, CONTRAST_LABELS[contrast], row["point_diff"], *row["ci95"]))
    with (EDA / "all32_target_contrasts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("seed", "contrast", "target", "candidate_minus_reference", "ci95_low", "ci95_high")
        )
        for seed in SEEDS:
            for contrast in CONTRASTS:
                targets = result["registered_target_contrasts"][f"seed{seed}__{contrast}"][
                    "targets"
                ]
                for target, label in (("t0", "Target 0"), ("t1", "Target 1")):
                    row = targets[target]
                    writer.writerow(
                        (seed, CONTRAST_LABELS[contrast], label, row["point_diff"], *row["ci95"])
                    )


def main() -> None:
    result, grade, fit, reproduction = read_inputs()
    EDA.mkdir(parents=True, exist_ok=True)
    render_arm_plot(result)
    render_global_contrasts(result)
    render_target_contrasts(result)
    write_csvs(result)
    REPORT.write_text(build_report(result, grade, fit, reproduction))
    print(f"wrote {REPORT}")
    print(f"score_sha256={sha256_file(SCORES)}")
    print(f"report_sha256={sha256_file(REPORT)}")


if __name__ == "__main__":
    main()
