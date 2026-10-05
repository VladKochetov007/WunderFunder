"""Render E483 figures/report from the frozen grade and attempt receipts."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ACTOR = ROOT / "research/campaign_2026-10-05/activation_momentum"
SCORES_PATH = HERE / "scores.json"
MANIFEST_PATH = ACTOR / "runs/eighteen_v1/fit_manifest.json"
FIT_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e483-isolated-muon-momentum-eighteen-fit-v1.json"
)
GRADE_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e483-momentum-independent-cpu-grade-v3.json"
)
REPROOF_PATH = (
    ROOT / "research/campaign_2026-10-05/checks/momentum_activation_export_reproduction.json"
)

SEEDS = (271831, 271832, 271833)
SHAPES = ("baseline", "sine")
BETAS = (0.8, 0.9, 0.95)
FAMILIES = ("inp", "gru", "head")


def sha256(path: Path) -> str:
    """Hash an input receipt with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_receipts() -> tuple[dict, dict, dict, dict, dict]:
    """Load existing grade and run receipts; do not score or infer anything."""
    scores = json.loads(SCORES_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    fit_attempt = json.loads(FIT_ATTEMPT.read_text())
    grade_attempt = json.loads(GRADE_ATTEMPT.read_text())
    repro = json.loads(REPROOF_PATH.read_text())
    if scores["arm_count"] != 18 or scores["contrast_count"] != 27:
        raise ValueError("unexpected E483 grade matrix size")
    if {int(seed) for seed in scores["contrasts_by_seed"]} != set(SEEDS):
        raise ValueError("E483 grade does not contain the registered three seeds")
    if len(manifest["arms"]) != 18 or manifest["test_rows_loaded"] is not False:
        raise ValueError("E483 fit manifest is incomplete or touched test rows")
    if sha256(SCORES_PATH) != repro["result_sha256"]:
        raise ValueError("root raw-export receipt does not bind this independent grade")
    if (
        repro["status"] != "all_checked_values_match"
        or repro["arms"] != 18
        or repro["global_contrasts"] != 27
    ):
        raise ValueError("root export reproduction did not check the full E483 result")
    if repro["export_sha256"] != scores["selected_exports_sha256"]:
        raise ValueError("root raw-export proof is bound to another selected-export file")
    if fit_attempt["status"] != "finished" or not fit_attempt["teardown_verified"]:
        raise ValueError("E483 fit attempt is not successful with teardown")
    if any(
        fit_attempt["memory_events"].get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")
    ):
        raise ValueError("E483 fit attempt reports a worker OOM")
    if grade_attempt["status"] != "finished" or not grade_attempt["teardown_verified"]:
        raise ValueError("E483 independent grade attempt is not complete")
    if any(
        grade_attempt["memory_events"].get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")
    ):
        raise ValueError("E483 grade attempt reports an OOM")
    if set(manifest["seeds"]) != set(SEEDS) or tuple(manifest["shapes"]) != SHAPES:
        raise ValueError("fit manifest differs from the registered seed/shape scope")
    if set(scores["arm_scores"]) != {
        f"seed{seed}__{shape}__beta{beta}" for seed in SEEDS for shape in SHAPES for beta in BETAS
    }:
        raise ValueError("E483 grade arm inventory is not exactly the registered matrix")
    return scores, manifest, fit_attempt, grade_attempt, repro


def arm_plot(scores: dict) -> None:
    """Plot all eighteen arm scores for each metric in three panels."""
    metrics = (("wp", "Global WP"), ("t0", "Target 0 WP"), ("t1", "Target 1 WP"))
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True)
    x = np.arange(len(BETAS))
    colors = {"baseline": "#677481", "sine": "#147B8B"}
    linestyles = {271831: "-", 271832: "--", 271833: ":"}
    seed_markers = {271831: "o", 271832: "s", 271833: "^"}
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for seed in SEEDS:
            for shape in SHAPES:
                values = [
                    scores["arm_scores"][f"seed{seed}__{shape}__beta{beta}"][metric]
                    for beta in BETAS
                ]
                ax.plot(
                    x,
                    values,
                    color=colors[shape],
                    linestyle=linestyles[seed],
                    marker=seed_markers[seed],
                    label=f"{seed} {shape}",
                    linewidth=1.2,
                    markersize=4,
                )
        ax.set_title(title, loc="left")
        ax.set_ylabel("WP")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.2, 2.2)
        ax.set_xticks(x, [str(beta) for beta in BETAS])
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Muon recurrent-matrix β₁")
    fig.suptitle("E483: all 18 independent-grade arm scores", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(HERE / "all18_arm_scores.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def contrast_plot(scores: dict) -> None:
    """Plot all 27 seed-specific registered contrast intervals."""
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 14), sharex=True)
    for ax, seed in zip(axes, SEEDS, strict=True):
        items = list(scores["contrasts_by_seed"][str(seed)].items())
        y = np.arange(len(items))
        for idx, (_name, item) in enumerate(items):
            lo, hi = item["ci95"]
            family = item["family"]
            color = {
                "shape_at_momentum": "#167D8D",
                "momentum_within_shape": "#C16C32",
                "shape_by_momentum_interaction": "#7655A1",
            }[family]
            ax.errorbar(
                item["mean_diff"],
                idx,
                xerr=[[item["mean_diff"] - lo], [hi - item["mean_diff"]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=4,
            )
        ax.axvline(0.0, color="#222222", linewidth=0.9)
        ax.set_yticks(y, [name.replace("_", " ") for name, _ in items], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Seed {seed}: all nine contrasts", loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Paired-component change in global WP (95% bootstrap interval)")
    fig.suptitle("E483: all 27 registered contrasts, kept separate by seed", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(HERE / "all27_contrasts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_telemetry(manifest: dict) -> list[dict]:
    """Average the four preregistered telemetry snapshots separately per arm."""
    result = []
    for arm in manifest["arms"]:
        if len(arm["telemetry"]) != 4:
            raise ValueError("an E483 arm is missing a fixed telemetry sample")
        row = {
            "seed": arm["seed"],
            "shape": arm["shape"],
            "muon_beta1": arm["muon_beta1"],
            "clipped_updates_of_474": arm["clipped_updates"],
        }
        for family in FAMILIES:
            row[f"mean_preclip_gradient_rms_{family}"] = float(
                np.mean([sample["preclip_gradient_rms"][family] for sample in arm["telemetry"]])
            )
            row[f"mean_parameter_update_rms_{family}"] = float(
                np.mean([sample["parameter_update_rms"][family] for sample in arm["telemetry"]])
            )
        result.append(row)
    return result


def telemetry_plot(rows: list[dict]) -> None:
    """Plot sampled mean parameter-update RMS by module for all 18 arms."""
    ordered = sorted(rows, key=lambda row: (row["seed"], row["shape"], row["muon_beta1"]))
    labels = [f"{row['seed']} {row['shape']} β{row['muon_beta1']}" for row in ordered]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(15, 5.7))
    for family, color, marker in (
        ("inp", "#5B8E7D", "o"),
        ("gru", "#C16C32", "s"),
        ("head", "#5177A5", "^"),
    ):
        values = [row[f"mean_parameter_update_rms_{family}"] for row in ordered]
        ax.plot(x, values, marker=marker, color=color, label=family, linewidth=1.1)
    ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Mean parameter-update RMS")
    ax.set_title("E483 source telemetry: mean of four fixed sampled updates per fit")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, title="Parameter family")
    fig.tight_layout()
    fig.savefig(HERE / "source_telemetry_means.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_report(
    scores: dict,
    manifest: dict,
    fit_attempt: dict,
    grade_attempt: dict,
    repro: dict,
    telemetry: list[dict],
) -> None:
    """Write a receipt-driven report without calculating predictions or grades."""
    sine_rows = []
    for seed in SEEDS:
        for beta in BETAS:
            name = f"sine_minus_baseline_beta{beta}"
            item = scores["contrasts_by_seed"][str(seed)][name]
            sine_rows.append(
                f"| {seed} | {beta:.2f} | {item['mean_diff']:+.6f} | "
                f"[{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] |"
            )
    score_rows = []
    for _key, item in scores["arm_scores"].items():
        score_rows.append(
            f"| {item['seed']} | {item['shape']} | {item['muon_beta1']:.2f} | "
            f"{item['wp']:.6f} | {item['t0']:.6f} | {item['t1']:.6f} | "
            f"{item['clipped_updates']}/474 |"
        )
    contrast_rows = []
    for seed in SEEDS:
        for name, item in scores["contrasts_by_seed"][str(seed)].items():
            contrast_rows.append(
                f"| {seed} | {name.replace('_', ' ')} | {item['mean_diff']:+.6f} | "
                f"[{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] | {item['family']} |"
            )
    telemetry_rows = []
    for row in sorted(telemetry, key=lambda x: (x["seed"], x["shape"], x["muon_beta1"])):
        telemetry_rows.append(
            f"| {row['seed']} | {row['shape']} | {row['muon_beta1']:.2f} | "
            f"{row['clipped_updates_of_474']} | "
            + " | ".join(
                f"{row[f'mean_preclip_gradient_rms_{family}']:.3e} / "
                f"{row[f'mean_parameter_update_rms_{family}']:.3e}"
                for family in FAMILIES
            )
            + " |"
        )
    best_rows = []
    for seed in SEEDS:
        for shape in SHAPES:
            ranked = sorted(
                BETAS,
                key=lambda beta: scores["arm_scores"][f"seed{seed}__{shape}__beta{beta}"]["wp"],
            )
            best_rows.append(f"| {seed} | {shape} | {ranked[-1]:.2f} |")
    interaction_rows = []
    for seed in SEEDS:
        for name in (
            "shape_by_momentum_interaction_beta0.8_vs0.9",
            "shape_by_momentum_interaction_beta0.95_vs0.9",
        ):
            item = scores["contrasts_by_seed"][str(seed)][name]
            interaction_rows.append(
                f"| {seed} | {name.replace('_', ' ')} | {item['mean_diff']:+.6f} |"
            )
    report = f"""# E483 recurrent Muon momentum: independent V3 grade

The independent grader reports **18/18** fixed fits and **27/27** registered contrasts on the same 16 previously exposed components, with the same 24 fit-role components for each model. It keeps the three fit seeds separate. This is a bounded H64, one-epoch exposed-panel experiment; it does not establish a mechanism or forecast the public score.

Root's separate raw-export arithmetic receipt matches all 18 arm scores and 27 contrasts with maximum absolute error **{repro["maximum_absolute_error"]:.3e}**. That checks arithmetic on saved exports; it is not a second grader or fresh inference. The recorded public best remains **0.6660**.

## Run receipts

The full-fit attempt finished with exit status zero and verified teardown in {fit_attempt["elapsed_seconds"]:.2f} seconds. Peak worker RAM was {fit_attempt["peak_ram_bytes"]:,} bytes and peak observed physical GPU use was {fit_attempt["peak_observed_vram_bytes"]:,} bytes. Worker and parent OOM counters did not increase. The independent CPU grade finished in {grade_attempt["elapsed_seconds"]:.6f} seconds at {grade_attempt["peak_ram_bytes"]:,} bytes peak RAM, with no worker OOM and verified teardown. The fit manifest records 474 updates per arm and `test_rows_loaded: false` during fitting.

## All 18 arm scores

| Seed | Shape | Muon β₁ | Global WP | Target 0 | Target 1 | Clipped updates |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(score_rows)}

![All eighteen seed/shape/momentum scores](eda/eighteen_v3/all18_arm_scores.png)

## Sine-minus-baseline at each momentum

All nine seed-specific sine-minus-baseline intervals are positive. This consistency applies to this fixed exposed panel and these three seeds only.

| Seed | Muon β₁ | Δ global WP | Paired-component 95% CI |
|---:|---:|---:|---:|
{chr(10).join(sine_rows)}

## All 27 registered contrasts

Intervals are paired bootstrap intervals over the same 16 numeric-sorted components; no seed pooling is performed. Momentum rankings and interaction signs vary by seed.

| Seed | Contrast | Δ global WP | Paired-component 95% CI | Family |
|---:|---|---:|---:|---|
{chr(10).join(contrast_rows)}

![All 27 seed-specific contrasts](eda/eighteen_v3/all27_contrasts.png)

Best point-estimate β₁ varies across seed and shape:

| Seed | Shape | Highest point-estimate β₁ |
|---:|---|---:|
{chr(10).join(best_rows)}

The two preregistered shape-by-momentum interaction point estimates also change sign:

| Seed | Contrast | Difference-in-differences Δ global WP |
|---:|---|---:|
{chr(10).join(interaction_rows)}

These changes do not support a universal best momentum value. No optimizer-mechanism or public-score claim follows from this panel.

## Sampled source telemetry means

The values below average each arm's four preregistered telemetry snapshots (updates 0, 46, 236 and 473). Each cell is `mean preclip gradient RMS / mean parameter-update RMS`; `inp`, `gru`, and `head` are the source actor's parameter families. These four samples do not describe the full 474-update trajectory.

| Seed | Shape | β₁ | Clipped updates | Input | GRU | Head |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(telemetry_rows)}

![Sampled mean parameter-update RMS by source family](eda/eighteen_v3/source_telemetry_means.png)

The independent score/export are `eda/eighteen_v3/scores.json` and `selected_exports.npz`; root's arithmetic proof is `checks/momentum_activation_export_reproduction.json`. This report displays those receipts and the fit manifest telemetry; it does not rerun scoring.
"""
    (ACTOR / "REPORT.md").write_text(report)

    out = HERE / "source_telemetry_means.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(telemetry[0]))
        writer.writeheader()
        writer.writerows(telemetry)


def main() -> None:
    scores, manifest, fit_attempt, grade_attempt, repro = load_receipts()
    telemetry = summarize_telemetry(manifest)
    if len(telemetry) != 18:
        raise ValueError("telemetry summary omitted an arm")
    arm_plot(scores)
    contrast_plot(scores)
    telemetry_plot(telemetry)
    render_report(scores, manifest, fit_attempt, grade_attempt, repro, telemetry)
    print("Rendered E483 report and figures from saved independent receipts.")


if __name__ == "__main__":
    main()
