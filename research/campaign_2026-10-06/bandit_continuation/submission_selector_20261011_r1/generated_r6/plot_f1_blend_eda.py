from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_METRICS_SHA256 = "e030f8914aafa7bb172824577552d059a6e101ea7fc5c1ab97b1c2b9d33ff1f5"
H384_SEEDS = (
    "vt_f1_muonall_h384_e3",
    "vt_f1_muonall_h384_e3_s4",
    "vt_f1_muonall_h384_e3_s5",
)
H256 = "vt_f1_dsq_dm_t3_a9_e6"
H391 = "vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2"
MODEL_LABELS = ("H256", "H391", "H384 s3", "H384 s4", "H384 s5")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one small, pinned EDA input or artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], value)


def as_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def score_by_name(rows: object, name: str, score_key: str) -> float:
    for raw_row in as_list(rows, "mixture rows"):
        row = as_dict(raw_row, "mixture row")
        if row.get("candidate") == name:
            score = as_dict(row[score_key], score_key)
            return float(score["wp"])
    raise ValueError(f"candidate {name!r} is missing")


def correlation_matrix(rows: object, target: int) -> np.ndarray:
    for raw_row in as_list(rows, "correlation rows"):
        row = as_dict(raw_row, "correlation row")
        if row.get("target") == target:
            raw_matrix = as_list(row["matrix"], "correlation matrix")
            return np.asarray(raw_matrix, dtype=np.float64)
    raise ValueError(f"target {target} correlation matrix is missing")


def build_figure(metrics: dict[str, object], output_dir: Path) -> dict[str, float]:
    singles = as_dict(metrics["singles_full_fold"], "singles_full_fold")
    single_scores = {seed: float(as_dict(singles[seed], seed)["wp"]) for seed in H384_SEEDS}
    mean_single = mean(single_scores.values())

    fixed_full = as_list(metrics["fixed_mixtures_full_fold"], "fixed_mixtures_full_fold")
    pair_names = ("h384_s3_s4_equal", "h384_s3_s5_equal", "h384_s4_s5_equal")
    mean_pair = mean(score_by_name(fixed_full, name, "full_fold") for name in pair_names)
    trio = score_by_name(fixed_full, "h384_s3_s4_s5_equal", "full_fold")
    student_trio = score_by_name(fixed_full, "h256_half_h384_trio_half", "full_fold")
    stages = (mean_single, mean_pair, trio, student_trio)
    deltas_milli = [(score - mean_single) * 1000.0 for score in stages]
    increments_milli = [
        (mean_pair - mean_single) * 1000.0,
        (trio - mean_pair) * 1000.0,
        (student_trio - trio) * 1000.0,
    ]

    correlations = as_dict(
        metrics["weighted_correlations_and_output_scales"],
        "weighted_correlations_and_output_scales",
    )
    raw_rows = correlations["weighted_residual_corr_by_target"]
    aligned = as_dict(
        correlations["calibration_only_zero_intercept_scale_alignment"],
        "calibration_only_zero_intercept_scale_alignment",
    )
    raw_matrices = [correlation_matrix(raw_rows, target) for target in (0, 1)]
    aligned_matrices = [
        correlation_matrix(aligned["weighted_residual_corr_by_target"], target) for target in (0, 1)
    ]

    complement = as_list(
        metrics["fixed_mixtures_component_complement"],
        "fixed_mixtures_component_complement",
    )
    fitted = as_list(
        metrics["fitted_mixtures_component_complement"],
        "fitted_mixtures_component_complement",
    )
    base_wp = score_by_name(complement, "h256_half_h384_trio_half", "score")
    complement_rows = {
        "H256 + H391 equal (fixed)": score_by_name(complement, "h256_h391_equal", "score"),
        "H384 trio (fixed)": score_by_name(complement, "h384_s3_s4_s5_equal", "score"),
        "Inverse output-SD (fit)": score_by_name(
            fitted, "inverse_output_sd_calibration_split", "score"
        ),
        "80% covariance shrink (fit)": score_by_name(
            fitted, "residual_covariance_nonnegative_80pct_equal_shrink", "score"
        ),
        "Global output scale (fit)": score_by_name(
            fitted, "h256_half_h384_trio_half_targetwise_calibrated_global_scale", "score"
        ),
    }
    complement_deltas = {
        name: (score - base_wp) * 1_000_000.0 for name, score in complement_rows.items()
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.2, 6.4),
        gridspec_kw={"width_ratios": (1.15, 1.35, 1.5)},
        constrained_layout=False,
    )
    fig.suptitle(
        "Output-mixture evidence from exposed F1 selector predictions",
        x=0.02,
        ha="left",
        fontsize=15,
        weight="bold",
    )

    axis = axes[0]
    x = np.arange(4)
    axis.bar(x, deltas_milli, color=("#9aabb9", "#7099b7", "#477fa8", "#7b8c58"))
    axis.plot(x, deltas_milli, color="#283b4b", marker="o", linewidth=1.5, zorder=4)
    axis.set_xticks(
        x,
        (
            "single mean\n.670680",
            "pair mean\n.672075",
            "H384 trio\n.672542",
            "H256 + trio\n.672640",
        ),
    )
    axis.set_ylabel("gain over mean single (milli-WP)")
    axis.set_title(
        "Fixed F1 mixture ladder\nFull exposed fold · 457 groups",
        loc="left",
        fontsize=10,
        weight="bold",
    )
    axis.set_ylim(0.0, 2.4)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    for index, value in enumerate(deltas_milli):
        axis.annotate(
            f"{value:.3f}",
            (index, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    # Compare the two scopes side by side without implying a paired effect.
    x_groups = np.array([0.0, 1.0])
    widths = (-0.18, 0.18)
    raw_color = "#3976a8"
    aligned_color = "#d97732"
    for group_index, (_label, pairs) in enumerate(
        (
            ("H256 ↔ H384", [(0, index) for index in (2, 3, 4)]),
            ("H384 seed pairs", list(combinations((2, 3, 4), 2))),
        )
    ):
        for scope_index, (matrices, color, scope_label) in enumerate(
            (
                (raw_matrices, raw_color, "Raw F1"),
                (aligned_matrices, aligned_color, "Scale-aligned complement"),
            )
        ):
            values = [float(matrix[left, right]) for matrix in matrices for left, right in pairs]
            jitter = np.linspace(-0.045, 0.045, num=len(values))
            xpos = group_index + widths[scope_index] + jitter
            axes[1].scatter(
                xpos,
                values,
                color=color,
                s=27,
                alpha=0.75,
                label=scope_label if group_index == 0 else None,
                zorder=3,
            )
            average = mean(values)
            axes[1].hlines(
                average,
                group_index + widths[scope_index] - 0.12,
                group_index + widths[scope_index] + 0.12,
                color=color,
                linewidth=2.2,
            )
            axes[1].annotate(
                f"{average:.4f}",
                (group_index + widths[scope_index], average),
                xytext=(0, 7 if scope_index == 0 else -13),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=color,
            )
    axes[1].set_xticks(x_groups, ["H256 ↔ H384", "H384 seed pairs"])
    axes[1].set_ylim(0.975, 1.002)
    axes[1].set_ylabel("mean of targetwise weighted residual correlations")
    axes[1].set_title(
        "Residual similarity\nRaw F1 (blue) vs scale-aligned complement (orange)",
        loc="left",
        fontsize=10,
        weight="bold",
    )
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    axis = axes[2]
    candidate_names = tuple(complement_deltas)
    delta_values = [complement_deltas[name] for name in candidate_names]
    colors = ("#7b8c58", "#8b9cab", "#739e9a", "#739e9a", "#a18851")
    y = np.arange(len(candidate_names))
    axis.barh(y, delta_values, color=colors, height=0.62)
    axis.axvline(0.0, color="#333333", linewidth=1)
    scale_boot = as_dict(
        metrics["calibrated_overall_scale_vs_fixed_base_component_bootstrap"],
        "calibrated_overall_scale_vs_fixed_base_component_bootstrap",
    )
    bootstrap = as_dict(scale_boot["bootstrap"], "scale bootstrap")
    ci = cast(list[float], bootstrap["ci95"])
    scale_index = candidate_names.index("Global output scale (fit)")
    scale_delta = complement_deltas[candidate_names[scale_index]]
    axis.errorbar(
        scale_delta,
        scale_index,
        xerr=np.array([[scale_delta - ci[0] * 1_000_000.0], [ci[1] * 1_000_000.0 - scale_delta]]),
        fmt="o",
        color="#222222",
        capsize=3,
        markersize=4,
        zorder=4,
    )
    candidate_labels = [f"{name}\nWP {complement_rows[name]:.9f}" for name in candidate_names]
    axis.set_yticks(y, candidate_labels)
    axis.invert_yaxis()
    axis.set_xlim(-520.0, 420.0)
    axis.set_xlabel("Δ global WP vs fixed H256 + H384 trio (µWP)")
    axis.set_title(
        f"Component-complement grades\n52 components / 138 groups · base WP {base_wp:.9f}",
        loc="left",
        fontsize=10,
        weight="bold",
    )
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    for index, delta in enumerate(delta_values):
        if delta < -100.0:
            axis.text(
                delta + 8.0,
                index,
                f"{delta:+.3f} µWP",
                va="center",
                ha="left",
                fontsize=7,
                color="#25313a",
            )
        elif delta > 100.0:
            axis.text(
                delta - 8.0,
                index,
                f"{delta:+.3f} µWP",
                va="center",
                ha="right",
                fontsize=7,
                color="#25313a",
            )
        else:
            axis.text(12.0, index, f"{delta:+.3f} µWP", va="center", fontsize=7)

    fig.text(
        0.015,
        0.02,
        "F1 is exposed selector evidence, not fresh CV or a public-score estimate. The H384 trio and its three pairs reuse the same n=3 training seeds. "
        "Scale-aligned correlations fit targetwise slopes on 97 calibration components / 319 groups and measure residuals on a distinct 52-component complement; they are not paired before/after estimates. "
        "Fixed equal weights remain preferred: the calibrated global-scale delta is effectively zero.",
        fontsize=8,
        color="#444444",
        wrap=True,
    )
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.91), w_pad=2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "F1_BLEND_OUTPUT_EDA_R1.svg"
    png_path = output_dir / "F1_BLEND_OUTPUT_EDA_R1.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight", metadata={"Date": None})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8"
    )
    fig.savefig(png_path, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_h384_single_wp": mean_single,
        "mean_h384_pair_wp": mean_pair,
        "h384_trio_wp": trio,
        "h256_half_plus_h384_trio_half_wp": student_trio,
        "single_mean_to_pair_mean_milli_wp": increments_milli[0],
        "pair_mean_to_trio_milli_wp": increments_milli[1],
        "trio_to_student_plus_trio_milli_wp": increments_milli[2],
        "fixed_complement_base_wp": base_wp,
        "calibrated_global_scale_delta_wp": float(bootstrap["mean_diff"]),
    }


def main() -> None:
    default_metrics = Path(__file__).with_name("F1_SELECTOR_METRICS_R1.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=default_metrics)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    metrics_sha256 = sha256_file(args.metrics)
    if metrics_sha256 != EXPECTED_METRICS_SHA256:
        raise ValueError(f"metrics digest mismatch: {metrics_sha256} != {EXPECTED_METRICS_SHA256}")
    raw_metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics = as_dict(raw_metrics, "metrics")
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "wnn-f1-blend-eda-r1"})
    values = build_figure(metrics, args.output_dir)
    script_path = Path(__file__).resolve()
    receipt = {
        "schema": "wnn.f1_blend_eda_receipt.v1",
        "metrics_file": args.metrics.name,
        "metrics_sha256": metrics_sha256,
        "generator_file": script_path.name,
        "generator_sha256": sha256_file(script_path),
        "matplotlib_version": matplotlib.__version__,
        "scope": "exposed F1 selector output EDA; not fresh CV or public validation",
        "derived_values": values,
        "artifacts": {
            "svg": {
                "path": "F1_BLEND_OUTPUT_EDA_R1.svg",
                "sha256": sha256_file(args.output_dir / "F1_BLEND_OUTPUT_EDA_R1.svg"),
            },
            "png": {
                "path": "F1_BLEND_OUTPUT_EDA_R1.png",
                "sha256": sha256_file(args.output_dir / "F1_BLEND_OUTPUT_EDA_R1.png"),
            },
        },
    }
    receipt_path = args.output_dir / "F1_BLEND_OUTPUT_EDA_R1_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
