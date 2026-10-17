"""Describe pooled score moments using saved fixed-prediction sufficient sums.

No labels, features, fitted coefficients or model weights are loaded. This is
an exact descriptive decomposition on the already selected exposed panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def decompose(sums: np.ndarray) -> dict[str, float]:
    """Split global centered moments into within- and between-sequence terms."""
    sums = sums[sums[:, 0] > 0]
    weight, wy, wp, wyy, wpp, wyp = sums.T
    total_weight = weight.sum()
    mean_y, mean_p = wy.sum() / total_weight, wp.sum() / total_weight
    seq_y, seq_p = wy / weight, wp / weight
    within_y = float(np.sum(wyy - wy * seq_y))
    within_p = float(np.sum(wpp - wp * seq_p))
    within_cov = float(np.sum(wyp - wy * seq_p))
    between_y = float(np.sum(weight * (seq_y - mean_y) ** 2))
    between_p = float(np.sum(weight * (seq_p - mean_p) ** 2))
    between_cov = float(np.sum(weight * (seq_y - mean_y) * (seq_p - mean_p)))
    total_y = within_y + between_y
    total_p = within_p + between_p
    total_cov = within_cov + between_cov
    denominator = np.sqrt(max(total_y * total_p, 0.0))
    if denominator == 0:
        raise ValueError("degenerate target or prediction variance")
    direct_cov = float(wyp.sum() - wy.sum() * wp.sum() / total_weight)
    if not np.isclose(total_cov, direct_cov, rtol=1e-10, atol=1e-7):
        raise AssertionError("within/between covariance identity failed")
    return {
        "target_weight": float(total_weight),
        "pooled_wp_reconstructed": float(total_cov / denominator),
        "within_covariance_over_global_denominator": float(within_cov / denominator),
        "between_covariance_over_global_denominator": float(between_cov / denominator),
        "within_covariance_sum": within_cov,
        "between_covariance_sum": between_cov,
        "within_target_variance_sum": within_y,
        "between_target_variance_sum": between_y,
        "within_prediction_variance_sum": within_p,
        "between_prediction_variance_sum": between_p,
        "between_fraction_target_variance": float(between_y / total_y),
        "between_fraction_prediction_variance": float(between_p / total_p),
    }


def main() -> None:
    """Bind input hashes, verify pooled scores, and save descriptive results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sums", type=Path)
    parser.add_argument("grade", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    grade = json.loads(args.grade.read_text())
    sums_hash = hashlib.sha256(args.sums.read_bytes()).hexdigest()
    expected_hash = grade["per_sequence_sufficient_statistics"]["sha256"]
    if sums_hash != expected_hash:
        raise ValueError("sufficient-statistic input differs from the grade receipt")
    with np.load(args.sums, allow_pickle=False) as archive:
        sums = archive["sums"]
        models = archive["model_names"].tolist()
        scenarios = archive["scenario_names"].tolist()
        statistic_names = archive["statistic_names"].tolist()
        if statistic_names != ["w", "wy", "wp", "wyy", "wpp", "wyp"]:
            raise ValueError("unexpected sufficient-statistic semantics")
    rows = []
    grade_models = {record["cache_name"]: record for record in grade["models"]}
    largest_score_difference = 0.0
    for model_index, model in enumerate(models):
        for scenario_index, scenario in enumerate(scenarios):
            targets = [
                decompose(sums[model_index, scenario_index, :, component]) for component in range(2)
            ]
            expected = grade_models[model]["scenarios"][scenario]["global"]
            for component, name in enumerate(("t0", "t1")):
                difference = abs(targets[component]["pooled_wp_reconstructed"] - expected[name])
                largest_score_difference = max(largest_score_difference, difference)
                if difference > 1e-8:
                    raise AssertionError("raw-moment reconstruction differs from canonical grade")
            rows.append({"model": model, "scenario": scenario, "targets": targets})
    output = {
        "schema": "wunder.sequence-moment-decomposition.v1",
        "sums_sha256": sums_hash,
        "grade_sha256": hashlib.sha256(args.grade.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "max_absolute_wp_difference_from_canonical_centered_grade": largest_score_difference,
        "interpretation": (
            "Within and between covariance numerators share the full global denominator; "
            "their normalized terms sum to pooled WP. They are not separately computed "
            "subgroup correlations, causal effects, or removable error budgets. "
            "Centering on a true per-sequence target mean would be nondeployable."
        ),
        "fitting": False,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
