"""Render frozen E488 score JSON and root arithmetic receipt only.

This module never opens the parquet dataset, checkpoints, or selected-export
NPZ. The independent scorer and root reproducer produce all numeric values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (271_839, 271_840, 271_841)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix")
EPOCHS = (1, 3)
CONTRASTS = (
    "recurrent_minus_static",
    "recurrent_minus_fixed",
    "static_minus_fixed",
    "fixed_minus_gelu",
)
ARM_LABELS = {
    "gelu": "GELU",
    "fixed_mix": "Fixed mixture",
    "static_mix": "Learned static",
    "recurrent_mix": "Recurrent context",
}
CONTRAST_LABELS = {
    "recurrent_minus_static": "Recurrent − static",
    "recurrent_minus_fixed": "Recurrent − fixed",
    "static_minus_fixed": "Static − fixed",
    "fixed_minus_gelu": "Fixed mixture − GELU",
}
COLORS = {
    "gelu": "#555555",
    "fixed_mix": "#4c78a8",
    "static_mix": "#f58518",
    "recurrent_mix": "#54a24b",
}


def sha256_file(path: Path) -> str:
    """Hash a saved result or receipt using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_arm_keys() -> set[str]:
    return {
        f"seed{seed}__{arm}__epoch{epoch}" for seed in SEEDS for arm in ARMS for epoch in EPOCHS
    }


def expected_contrast_keys() -> set[str]:
    keys = {
        f"seed{seed}__epoch{epoch}__{contrast}"
        for seed in SEEDS
        for epoch in EPOCHS
        for contrast in CONTRASTS
    }
    keys.update(f"seed{seed}__duration_interaction" for seed in SEEDS)
    return keys


def validate_numeric_summary(
    scores: dict[str, Any], reproduction: dict[str, Any], score_sha256: str, export_sha256: str
) -> None:
    """Validate complete registered E488 result matrices and arithmetic binding."""
    require(scores.get("metric_id") == "global_wp_clip2_v1", "E488 metric differs")
    require(scores.get("trajectory_count") == 12, "E488 trajectory count differs")
    require(scores.get("snapshot_count") == 24, "E488 snapshot count differs")
    require(
        scores.get("registered_global_contrast_count") == 27, "E488 global contrast count differs"
    )
    require(
        scores.get("registered_target_contrast_count") == 27, "E488 target contrast count differs"
    )
    require(scores.get("no_seed_pooling") is True, "E488 result pooled seeds")
    require(
        scores.get("test_labels_used_for_fitting_or_selection") is False,
        "E488 result crosses fit boundary",
    )
    bootstrap = scores.get("bootstrap", {})
    require(bootstrap.get("draws") == 2000, "E488 bootstrap draw count differs")
    require(bootstrap.get("seed") == 271839, "E488 bootstrap seed differs")
    require(
        bootstrap.get("multiplicities_shared_across_all_24_snapshots_and_contrasts") is True,
        "E488 bootstrap does not share component draws",
    )

    arms = scores.get("arm_scores", {})
    global_rows = scores.get("registered_global_contrasts", {})
    target_rows = scores.get("registered_target_contrasts", {})
    require(set(arms) == expected_arm_keys(), "E488 arm snapshot inventory is incomplete")
    require(
        set(global_rows) == expected_contrast_keys(), "E488 global contrast inventory is incomplete"
    )
    require(
        set(target_rows) == expected_contrast_keys(), "E488 target contrast inventory is incomplete"
    )

    for key, row in arms.items():
        expected = key.split("__")
        require(row.get("seed") == int(expected[0][4:]), f"E488 seed differs in {key}")
        require(row.get("arm") == expected[1], f"E488 arm differs in {key}")
        require(row.get("epoch") == int(expected[2][5:]), f"E488 epoch differs in {key}")
        require(row.get("updates") == row["epoch"] * 474, f"E488 updates differ in {key}")
        require(row.get("metric_id") == "global_wp_clip2_v1", f"E488 metric differs in {key}")
        require(row.get("selected_rows", 0) > 0, f"E488 selected row count is empty in {key}")
        for field in ("wp", "t0", "t1"):
            finite_number(row.get(field), f"{key}.{field}")

    for key, row in global_rows.items():
        require(row.get("seed") in SEEDS, f"unknown E488 seed in global contrast {key}")
        if key.endswith("duration_interaction"):
            require(row.get("epoch") is None, f"duration interaction has an epoch in {key}")
        else:
            require(row.get("epoch") in EPOCHS, f"unknown E488 epoch in {key}")
        validate_summary_row(row, key)

    for key, row in target_rows.items():
        require(row.get("seed") in SEEDS, f"unknown E488 seed in target contrast {key}")
        require(
            set(row.get("targets", {})) == {"t0", "t1"},
            f"missing E488 target contrast fields in {key}",
        )
        for target, summary in row["targets"].items():
            validate_summary_row(summary, f"{key}.{target}")

    require(reproduction.get("study") == "E488", "root reproduction is not E488")
    require(
        reproduction.get("status") == "all_checked_values_match",
        "root arithmetic check did not pass",
    )
    require(
        reproduction.get("result_sha256") == score_sha256, "root check does not bind E488 scores"
    )
    require(
        reproduction.get("export_sha256") == export_sha256, "root check does not bind E488 export"
    )
    require(reproduction.get("arms") == 24, "root check arm count differs")
    require(reproduction.get("global_contrasts") == 27, "root check global contrast count differs")
    target_count = reproduction.get("target_contrasts")
    require(target_count in (27, 54), "root check target contrast count differs")
    if "maximum_absolute_error" in reproduction:
        arithmetic_error = finite_number(
            reproduction["maximum_absolute_error"], "root maximum arithmetic error"
        )
        require(arithmetic_error <= 1e-10, "root arithmetic reproduction exceeds tolerance")


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} is not finite numeric data")
    return float(value)


def validate_summary_row(row: dict[str, Any], name: str) -> None:
    finite_number(row.get("point_diff"), f"{name}.point_diff")
    interval = row.get("ci95")
    require(isinstance(interval, list) and len(interval) == 2, f"{name} interval is missing")
    finite_number(interval[0], f"{name}.ci95[0]")
    finite_number(interval[1], f"{name}.ci95[1]")
    require(interval[0] <= interval[1], f"{name} interval is reversed")
    require(row.get("bootstrap_count") == 2000, f"{name} bootstrap count differs")
    require(row.get("bootstrap_seed") == 271839, f"{name} bootstrap seed differs")
    require(row.get("components") == 16, f"{name} component count differs")


def validate_attempt(record: dict[str, Any], label: str) -> None:
    require(record.get("status") == "finished", f"E488 {label} attempt did not finish")
    require(record.get("exit_code") == 0, f"E488 {label} attempt exit code is not zero")
    require(record.get("teardown_verified") is True, f"E488 {label} teardown is not verified")
    require(record.get("signals", []) == [], f"E488 {label} received a signal")
    before = record.get("initial_memory_events", {})
    after = record.get("final_memory_events", {})
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if key in before or key in after:
            require(before.get(key) == after.get(key), f"E488 {label} OOM counter changed: {key}")
    unit = record.get("unit_properties", {})
    if unit:
        require(unit.get("Result") == "success", f"E488 {label} unit result differs")
        require(unit.get("ExecMainStatus") in (0, "0"), f"E488 {label} unit exit status differs")


def read_inputs(
    score_path: Path,
    proof_path: Path,
    export_path: Path,
    fit_attempt_path: Path,
    grade_attempt_path: Path,
    reproduction_path: Path,
) -> tuple[dict[str, Any], ...]:
    """Read saved JSON and hashes only; never deserialize selected prediction arrays."""
    for path in (
        score_path,
        proof_path,
        export_path,
        fit_attempt_path,
        grade_attempt_path,
        reproduction_path,
    ):
        require(path.is_file(), f"required saved grade evidence is missing: {path}")
    scores = json.loads(score_path.read_text())
    proof = json.loads(proof_path.read_text())
    fit_attempt = json.loads(fit_attempt_path.read_text())
    grade_attempt = json.loads(grade_attempt_path.read_text())
    reproduction = json.loads(reproduction_path.read_text())
    score_sha = sha256_file(score_path)
    proof_sha = sha256_file(proof_path)
    export_sha = sha256_file(export_path)
    fit_sha = sha256_file(fit_attempt_path)
    validate_attempt(fit_attempt, "fit")
    validate_attempt(grade_attempt, "grade")
    require(scores.get("proof_sha256") == proof_sha, "E488 score JSON proof hash mismatch")
    require(
        scores.get("selected_export_sha256") == export_sha, "E488 score JSON export hash mismatch"
    )
    require(
        scores.get("selected_export_bytes") == export_path.stat().st_size,
        "E488 export byte count mismatch",
    )
    require(
        proof.get("status") == "all_fit_only_evidence_passed_test_rows_unread",
        "E488 score proof is not fit-only verified",
    )
    require(proof.get("fit_attempt_sha256") == fit_sha, "E488 proof does not bind fit attempt")
    require(proof.get("snapshot_count") == 24, "E488 proof snapshot count differs")
    grade_artifacts = grade_attempt.get("artifacts", [])
    artifact_receipts = {item.get("sha256"): item for item in grade_artifacts}
    score_receipt = artifact_receipts.get(score_sha)
    export_receipt = artifact_receipts.get(export_sha)
    require(score_receipt is not None, "E488 grade attempt does not bind score JSON")
    require(export_receipt is not None, "E488 grade attempt does not bind selected export")
    expected_score_path = score_path.resolve().relative_to(ROOT).as_posix()
    expected_export_path = export_path.resolve().relative_to(ROOT).as_posix()
    require(
        score_receipt.get("path") == expected_score_path, "E488 attempt score artifact path differs"
    )
    require(
        export_receipt.get("path") == expected_export_path,
        "E488 attempt export artifact path differs",
    )
    require(
        score_receipt.get("size") == score_path.stat().st_size, "E488 score artifact size differs"
    )
    require(
        export_receipt.get("size") == export_path.stat().st_size,
        "E488 export artifact size differs",
    )
    validate_numeric_summary(scores, reproduction, score_sha, export_sha)
    return (
        scores,
        proof,
        fit_attempt,
        grade_attempt,
        reproduction,
        {
            "score_sha256": score_sha,
            "proof_sha256": proof_sha,
            "selected_export_sha256": export_sha,
            "fit_attempt_sha256": fit_sha,
            "grade_attempt_sha256": sha256_file(grade_attempt_path),
            "reproduction_sha256": sha256_file(reproduction_path),
        },
    )


def render_arm_scores(scores: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True)
    for axis, metric, title in zip(
        axes, ("wp", "t0", "t1"), ("Global WP", "Target 0 WP", "Target 1 WP"), strict=True
    ):
        for seed_index, seed in enumerate(SEEDS):
            x = np.asarray(EPOCHS, dtype=float) + (seed_index - 1) * 0.035
            for arm in ARMS:
                means = [
                    scores["arm_scores"][f"seed{seed}__{arm}__epoch{epoch}"][metric]
                    for epoch in EPOCHS
                ]
                axis.plot(
                    x,
                    means,
                    marker="o",
                    color=COLORS[arm],
                    alpha=0.35 + 0.2 * seed_index,
                    linestyle=("-", "--", ":")[seed_index],
                    label=f"{seed} · {ARM_LABELS[arm]}",
                )
        axis.set_title(title)
        axis.set_xticks(EPOCHS)
        axis.set_xlabel("Snapshot after epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("global_wp_clip2_v1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("E488 score trajectories · all registered snapshots")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_global_contrasts(scores: dict[str, Any], output: Path) -> None:
    rows = []
    for seed in SEEDS:
        for epoch in EPOCHS:
            for contrast in CONTRASTS:
                key = f"seed{seed}__epoch{epoch}__{contrast}"
                rows.append(
                    (key, seed, epoch, contrast, scores["registered_global_contrasts"][key])
                )
        key = f"seed{seed}__duration_interaction"
        rows.append(
            (key, seed, 4, "duration_interaction", scores["registered_global_contrasts"][key])
        )
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for axis, seed in zip(axes, SEEDS, strict=True):
        seed_rows = [row for row in rows if row[1] == seed]
        positions = np.arange(len(seed_rows))
        for position, (_, _, _epoch, _contrast, summary) in zip(positions, seed_rows, strict=True):
            point = summary["point_diff"]
            low, high = summary["ci95"]
            color = "#54a24b" if point >= 0 else "#e45756"
            axis.errorbar(
                point,
                position,
                xerr=[[point - low], [high - point]],
                fmt="o",
                color=color,
                capsize=2,
                markersize=4,
            )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(
            positions,
            [
                f"E{epoch}: {CONTRAST_LABELS.get(name, 'Duration interaction')}"
                for _, _, epoch, name, _ in seed_rows
            ],
        )
        axis.set_title(f"Seed {seed}")
        axis.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Paired component-bootstrap effect on global WP (95% interval)")
    fig.suptitle("E488 · all 27 global contrasts, seeds shown separately")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_target_contrasts(scores: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 15), sharex=True)
    for axis, target in zip(axes, ("t0", "t1"), strict=True):
        rows = []
        for seed in SEEDS:
            for epoch in EPOCHS:
                for contrast in CONTRASTS:
                    key = f"seed{seed}__epoch{epoch}__{contrast}"
                    rows.append(
                        (
                            seed,
                            epoch,
                            contrast,
                            scores["registered_target_contrasts"][key]["targets"][target],
                        )
                    )
            key = f"seed{seed}__duration_interaction"
            rows.append(
                (
                    seed,
                    4,
                    "duration_interaction",
                    scores["registered_target_contrasts"][key]["targets"][target],
                )
            )
        positions = np.arange(len(rows))
        for position, (seed, _epoch, _contrast, summary) in zip(positions, rows, strict=True):
            point = summary["point_diff"]
            low, high = summary["ci95"]
            color = (
                "#4c78a8" if seed == SEEDS[0] else ("#f58518" if seed == SEEDS[1] else "#54a24b")
            )
            axis.errorbar(
                point,
                position,
                xerr=[[point - low], [high - point]],
                fmt="o",
                color=color,
                capsize=2,
                markersize=3,
            )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(
            positions,
            [
                f"{seed} · E{epoch}: {CONTRAST_LABELS.get(name, 'Duration interaction')}"
                for seed, epoch, name, _ in rows
            ],
            fontsize=7,
        )
        axis.set_title(f"{target.upper()} · all 27 contrast families across seeds and epochs")
        axis.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Paired component-bootstrap effect (95% interval)")
    fig.suptitle("E488 · all 54 target-specific contrasts")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csvs(scores: dict[str, Any], output: Path) -> None:
    with (output / "all24_arm_snapshots.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "seed",
                "arm",
                "epoch",
                "updates",
                "global_wp",
                "target0_wp",
                "target1_wp",
                "selected_rows",
            )
        )
        for seed in SEEDS:
            for epoch in EPOCHS:
                for arm in ARMS:
                    row = scores["arm_scores"][f"seed{seed}__{arm}__epoch{epoch}"]
                    writer.writerow(
                        (
                            seed,
                            arm,
                            epoch,
                            row["updates"],
                            row["wp"],
                            row["t0"],
                            row["t1"],
                            row["selected_rows"],
                        )
                    )
    with (output / "all27_global_contrasts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "seed",
                "epoch_or_interaction",
                "contrast",
                "point_diff",
                "ci95_low",
                "ci95_high",
                "bootstrap_sd",
                "p_le_0",
                "draws",
                "bootstrap_seed",
            )
        )
        for key in sorted(scores["registered_global_contrasts"]):
            row = scores["registered_global_contrasts"][key]
            writer.writerow(
                (
                    row["seed"],
                    "duration_interaction"
                    if key.endswith("duration_interaction")
                    else row["epoch"],
                    row["contrast"],
                    row["point_diff"],
                    *row["ci95"],
                    row["bootstrap_sd"],
                    row["p_le_0"],
                    row["bootstrap_count"],
                    row["bootstrap_seed"],
                )
            )
    with (output / "all54_target_contrasts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "seed",
                "epoch_or_interaction",
                "contrast",
                "target",
                "point_diff",
                "ci95_low",
                "ci95_high",
                "bootstrap_sd",
                "p_le_0",
                "draws",
                "bootstrap_seed",
            )
        )
        for key in sorted(scores["registered_target_contrasts"]):
            record = scores["registered_target_contrasts"][key]
            for target in ("t0", "t1"):
                row = record["targets"][target]
                writer.writerow(
                    (
                        record["seed"],
                        "duration_interaction"
                        if key.endswith("duration_interaction")
                        else record["epoch"],
                        record["contrast"],
                        target,
                        row["point_diff"],
                        *row["ci95"],
                        row["bootstrap_sd"],
                        row["p_le_0"],
                        row["bootstrap_count"],
                        row["bootstrap_seed"],
                    )
                )


def build_report(
    scores: dict[str, Any],
    fit: dict[str, Any],
    grade: dict[str, Any],
    reproduction: dict[str, Any],
    hashes: dict[str, str],
) -> str:
    lines = [
        "# E488: context-mixture effects across training duration",
        "",
        "This report displays all 24 saved snapshots from 12 continuous three-epoch fits, with all 27 registered global contrasts and 54 target-specific contrast values. The score is `global_wp_clip2_v1` on the same 16 historically exposed evaluation components. Seeds remain separate; no seed pooling or checkpoint selection is performed here.",
        "",
        "The report is descriptive for this fixed panel. Component-bootstrap intervals condition on the fitted checkpoints and do not include run-to-run seed variation, search uncertainty, or multiplicity correction. It does not establish that a context gate improves generalization or predict a public score.",
        "",
        "## Receipts",
        "",
        f"The 12 fits completed in {fit['elapsed_seconds']:.2f}s; fit peak RAM was {fit.get('peak_ram_bytes', 'unreported')} bytes and peak observed GPU memory was {fit.get('peak_observed_vram_bytes', 'unreported')} bytes. The independent grade completed in {grade['elapsed_seconds']:.2f}s with {grade.get('peak_ram_bytes', 'unreported')} bytes peak RAM. The root saved-export arithmetic reproduction matched 24 snapshots, 27 global contrasts, and 54 target values; maximum absolute difference was {reproduction.get('maximum_absolute_error', float('nan')):.3e}.",
        "",
        f"Bound score SHA-256: `{hashes['score_sha256']}`. Selected-export SHA-256: `{hashes['selected_export_sha256']}`. Independent root receipt SHA-256: `{hashes['reproduction_sha256']}`.",
        "",
        "## All 24 arm snapshots",
        "",
        "![All arm scores](all24_arm_snapshots.png)",
        "",
        "| Seed | Epoch | Arm | Global WP | Target 0 | Target 1 | Updates |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for epoch in EPOCHS:
            for arm in ARMS:
                row = scores["arm_scores"][f"seed{seed}__{arm}__epoch{epoch}"]
                lines.append(
                    f"| {seed} | {epoch} | {ARM_LABELS[arm]} | {row['wp']:.6f} | {row['t0']:.6f} | {row['t1']:.6f} | {row['updates']} |"
                )
    lines.extend(
        [
            "",
            "## All 27 global contrasts",
            "",
            "![All global contrasts](all27_global_contrasts.png)",
            "",
            "| Seed | Epoch / contrast | Contrast | Difference | Paired component 95% interval |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for key in sorted(scores["registered_global_contrasts"]):
        row = scores["registered_global_contrasts"][key]
        epoch = (
            "duration interaction" if key.endswith("duration_interaction") else str(row["epoch"])
        )
        lines.append(
            f"| {row['seed']} | {epoch} | {row['contrast']} | {row['point_diff']:+.6f} | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## All 54 target-specific contrasts",
            "",
            "![All target contrasts](all54_target_contrasts.png)",
            "",
            "| Seed | Epoch / contrast | Contrast | Target | Difference | Paired component 95% interval |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for key in sorted(scores["registered_target_contrasts"]):
        record = scores["registered_target_contrasts"][key]
        epoch = (
            "duration interaction" if key.endswith("duration_interaction") else str(record["epoch"])
        )
        for target in ("t0", "t1"):
            row = record["targets"][target]
            lines.append(
                f"| {record['seed']} | {epoch} | {record['contrast']} | {target} | {row['point_diff']:+.6f} | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The registered question is whether the recurrent-minus-static contrast changes from epoch 1 to epoch 3 for this architecture, optimizer, data recipe, and exposed component panel. A positive duration interaction would support a duration-dependent effect in this experiment; it would not show that context provides new information or that the result transfers to a larger teacher/EMA model. The four model arms and both snapshots per arm are retained regardless of direction.",
            "",
            "The independent renderer reads saved score/receipt JSON and hashes the selected export without opening its NPZ contents, checkpoints, or parquet rows. It recomputes no score or bootstrap statistic.",
            "",
        ]
    )
    return "\n".join(lines)


def render(
    score_path: Path,
    proof_path: Path,
    export_path: Path,
    fit_attempt_path: Path,
    grade_attempt_path: Path,
    reproduction_path: Path,
    output_dir: Path,
    report_path: Path,
) -> None:
    """Validate existing receipts and render only saved grade values."""
    inputs = read_inputs(
        score_path, proof_path, export_path, fit_attempt_path, grade_attempt_path, reproduction_path
    )
    scores, _proof, fit_attempt, grade_attempt, reproduction, hashes = inputs
    if output_dir.exists() and any(
        (output_dir / name).exists()
        for name in (
            "all24_arm_snapshots.png",
            "all27_global_contrasts.png",
            "all54_target_contrasts.png",
            "all24_arm_snapshots.csv",
            "all27_global_contrasts.csv",
            "all54_target_contrasts.csv",
        )
    ):
        raise FileExistsError("refusing to overwrite E488 renderer outputs")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite E488 report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_arm_scores(scores, output_dir / "all24_arm_snapshots.png")
    render_global_contrasts(scores, output_dir / "all27_global_contrasts.png")
    render_target_contrasts(scores, output_dir / "all54_target_contrasts.png")
    write_csvs(scores, output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(scores, fit_attempt, grade_attempt, reproduction, hashes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--fit-attempt", type=Path, required=True)
    parser.add_argument("--grade-attempt", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    render(
        args.scores,
        args.proof,
        args.export,
        args.fit_attempt,
        args.grade_attempt,
        args.reproduction,
        args.output_dir,
        args.report,
    )


if __name__ == "__main__":
    main()
