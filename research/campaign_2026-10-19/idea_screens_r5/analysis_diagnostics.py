"""Fixed-bin, outcome-descriptive denominator diagnostics for R5."""

from __future__ import annotations

import numpy as np

from .analysis_fit import MomentFit

DECODERS = ("conditional_ratio", "constant_denominator")
TARGETS = ("t0", "t1")
HIST_EDGES = (-np.inf, 0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf)
HIST_EDGE_LABELS: tuple[str | float, ...] = ("-inf", 0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0, "inf")


def empty_diagnostic() -> dict:
    """Create fixed-bin counts for one role, component, and target."""
    return {
        "selected_rows": 0,
        "ratio_bhat_count": 0,
        "ratio_bhat_sum": 0.0,
        "ratio_bhat_sumsq": 0.0,
        "ratio_bhat_min": None,
        "ratio_bhat_max": None,
        "ratio_bhat_histogram": [0] * (len(HIST_EDGES) - 1),
        "counts": {
            "B_hat_below_epsilon": 0,
            "B_hat_negative": 0,
            "A_plus_negative": 0,
            "A_minus_negative": 0,
            "abs_D_hat_gt_2max_B_epsilon": 0,
        },
        "raw_prediction_abs_gt_2": {decoder: 0 for decoder in DECODERS},
    }


def update_diagnostic(
    summary: dict,
    decoded: dict[str, np.ndarray],
    mask: np.ndarray,
    fit: MomentFit,
    target_index: int,
) -> None:
    """Accumulate per-target fixed-bin counts over selected original-mask rows."""
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return
    b_hat = decoded["B_hat"][selected, target_index]
    normalized_b = b_hat / fit.mu_b[target_index]
    histogram = np.histogram(normalized_b, bins=HIST_EDGES)[0]
    summary["selected_rows"] += int(selected.sum())
    summary["ratio_bhat_count"] += len(normalized_b)
    summary["ratio_bhat_sum"] += float(normalized_b.sum())
    summary["ratio_bhat_sumsq"] += float(normalized_b @ normalized_b)
    local_min, local_max = float(normalized_b.min()), float(normalized_b.max())
    summary["ratio_bhat_min"] = (
        local_min
        if summary["ratio_bhat_min"] is None
        else min(summary["ratio_bhat_min"], local_min)
    )
    summary["ratio_bhat_max"] = (
        local_max
        if summary["ratio_bhat_max"] is None
        else max(summary["ratio_bhat_max"], local_max)
    )
    summary["ratio_bhat_histogram"] = [
        before + int(after)
        for before, after in zip(summary["ratio_bhat_histogram"], histogram, strict=True)
    ]
    epsilon = fit.epsilon[target_index]
    conditions = {
        "B_hat_below_epsilon": b_hat < epsilon,
        "B_hat_negative": b_hat < 0.0,
        "A_plus_negative": decoded["A_plus"][selected, target_index] < 0.0,
        "A_minus_negative": decoded["A_minus"][selected, target_index] < 0.0,
        "abs_D_hat_gt_2max_B_epsilon": np.abs(decoded["D_hat"][selected, target_index])
        > 2.0 * np.maximum(b_hat, epsilon),
    }
    for name, flags in conditions.items():
        summary["counts"][name] += int(flags.sum())
    for decoder in DECODERS:
        summary["raw_prediction_abs_gt_2"][decoder] += int(
            (np.abs(decoded[decoder][selected, target_index]) > 2.0).sum()
        )


def finish_diagnostic(summary: dict) -> dict:
    """Emit fixed edges, counts, pooled means/stds and selected-row fractions."""
    rows = summary["selected_rows"]
    count = summary["ratio_bhat_count"]
    if count:
        mean = summary["ratio_bhat_sum"] / count
        variance = max(0.0, summary["ratio_bhat_sumsq"] / count - mean * mean)
        distribution = {
            "count": count,
            "min": summary["ratio_bhat_min"],
            "max": summary["ratio_bhat_max"],
            "mean": mean,
            "std": float(np.sqrt(variance)),
        }
    else:
        distribution = {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    summary["B_hat_over_muB"] = {
        "edges": list(HIST_EDGE_LABELS),
        "histogram_counts": summary["ratio_bhat_histogram"],
        **distribution,
    }
    summary["fractions"] = {
        key: (value / rows if rows else None) for key, value in summary["counts"].items()
    }
    summary["raw_prediction_abs_gt_2_fractions"] = {
        key: (value / rows if rows else None)
        for key, value in summary["raw_prediction_abs_gt_2"].items()
    }
    for key in (
        "ratio_bhat_count",
        "ratio_bhat_sum",
        "ratio_bhat_sumsq",
        "ratio_bhat_min",
        "ratio_bhat_max",
        "ratio_bhat_histogram",
    ):
        summary.pop(key)
    return summary
