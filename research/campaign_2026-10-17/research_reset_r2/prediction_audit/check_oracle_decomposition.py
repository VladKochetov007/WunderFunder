#!/usr/bin/env python3
"""Check oracle variance/covariance identities using direct canonical moments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_bytes = args.results.read_bytes()
    results = json.loads(source_bytes)
    report: dict[str, object] = {
        "schema": "wunder.prediction-audit.oracle-decomposition-check.v1",
        "results_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "between_variance_source": "direct clipped-label-run oracle per-target weighted_prediction_variance_sum in canonical results",
        "within_variance_source": "selected clipped baseline residual variance from oracle_variance_checks",
        "tolerance_relative": 1e-8,
        "models": [],
    }
    models = report["models"]
    assert isinstance(models, list)
    for model_index, model in enumerate(results["models"]):
        baseline = model["scenarios"]["baseline"]["global"]
        oracle = model["scenarios"]["oracle_clipped_label_run"]["global"]
        prior = results["oracle_variance_checks"][model_index]
        target_checks = []
        for target_index, target_name in enumerate(("t0", "t1")):
            base_stats = baseline["targets"][target_index]
            oracle_stats = oracle["targets"][target_index]
            total = float(base_stats["weighted_prediction_variance_sum"])
            direct_between = float(oracle_stats["weighted_prediction_variance_sum"])
            within = float(prior["targets"][target_index]["within_run_prediction_variance_sum"])
            partition_error = abs(within + direct_between - total) / max(total, 1e-12)
            base_wp = float(baseline[target_name])
            oracle_wp = float(oracle[target_name])
            expected_wp = (
                base_wp / (direct_between / total) ** 0.5
                if total > 0 and direct_between > 0
                else None
            )
            covariance_delta = float(
                oracle_stats["weighted_covariance_numerator"]
                - base_stats["weighted_covariance_numerator"]
            )
            covariance_relative_error = abs(covariance_delta) / max(
                abs(float(base_stats["weighted_covariance_numerator"])), 1e-12
            )
            identity_error = abs(oracle_wp - expected_wp) if expected_wp is not None else None
            target_checks.append(
                {
                    "target": target_name,
                    "within_run_prediction_variance_sum": within,
                    "direct_between_run_mean_prediction_variance_sum": direct_between,
                    "baseline_total_prediction_variance_sum": total,
                    "variance_partition_relative_error": partition_error,
                    "fraction_prediction_variance_within_runs": within / total
                    if total > 0
                    else None,
                    "weighted_covariance_numerator_delta_oracle_minus_base": covariance_delta,
                    "weighted_covariance_relative_error": covariance_relative_error,
                    "baseline_wp": base_wp,
                    "oracle_wp": oracle_wp,
                    "identity_expected_wp_from_direct_variance": expected_wp,
                    "identity_wp_abs_error": identity_error,
                    "passes_1e-8_relative_partition_and_covariance": (
                        partition_error <= 1e-8 and covariance_relative_error <= 1e-8
                    ),
                }
            )
        models.append({"cache_name": model["cache_name"], "targets": target_checks})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
