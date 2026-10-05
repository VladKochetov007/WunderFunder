"""Reproduce E483 from raw exports with separate shifted-sum arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reproduce_activation_exports import assert_near, component_statistics, draws, file_hash, scores

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Check every target score, paired contrast, and saved bootstrap draw."""
    directory = HERE / "activation_momentum/eda/eighteen_v3"
    result_path = directory / "scores.json"
    export_path = directory / "selected_exports.npz"
    result = json.loads(result_path.read_text())
    error = 0.0
    points, boot = {}, {}
    with np.load(export_path, allow_pickle=False) as export:
        names = export["arm_names"].tolist()
        ids = export["component_ids"]
        counts = export["selected_row_counts"]
        if len(names) != 18 or len(np.unique(ids)) != 16:
            raise ValueError("incomplete arm/component matrix")
        if not np.array_equal(export["selected_masks_by_component"].sum(axis=1), counts):
            raise ValueError("mask selection/count mismatch")
        multiplicities = draws(271828, len(ids))
        if not np.array_equal(multiplicities, export["bootstrap_multiplicities"]):
            raise ValueError("regenerated bootstrap draws differ")
        for index, name in enumerate(names):
            statistics = component_statistics(
                export["targets_raw_selected"],
                export["predictions_raw_selected_by_arm"][index],
                counts,
            )[np.argsort(ids, kind="stable")]
            target_scores = scores(statistics, np.ones((1, len(ids))))[0]
            arm = result["arm_scores"][name]
            error = max(error, assert_near(target_scores, [arm["t0"], arm["t1"]]))
            points[name] = float(target_scores.mean())
            error = max(error, assert_near(points[name], arm["wp"]))
            boot[name] = scores(statistics, multiplicities).mean(axis=1)
            error = max(error, assert_near(boot[name], export["arm_bootstrap_wp"][index]))
        contrast_indices = {name: index for index, name in enumerate(export["contrast_names"])}
        for seed, contrasts in result["contrasts_by_seed"].items():
            for name, record in contrasts.items():

                def full_key(short: str, fit_seed: str = seed) -> str:
                    shape, beta = short.rsplit("_beta", 1)
                    return f"seed{fit_seed}__{shape}__beta{beta}"

                if record["family"] == "shape_by_momentum_interaction":
                    beta = record["muon_beta1"]
                    terms = [
                        (full_key(f"sine_beta{beta}"), 1),
                        (full_key(f"baseline_beta{beta}"), -1),
                        (full_key("sine_beta0.9"), -1),
                        (full_key("baseline_beta0.9"), 1),
                    ]
                else:
                    terms = [
                        (full_key(record["candidate"]), 1),
                        (full_key(record["reference"]), -1),
                    ]
                point = sum(sign * points[arm] for arm, sign in terms)
                sampled = sum(sign * boot[arm] for arm, sign in terms)
                error = max(error, assert_near(point, record["mean_diff"]))
                error = max(
                    error, assert_near(np.quantile(sampled, [0.025, 0.975]), record["ci95"])
                )
                error = max(error, assert_near(sampled.std(ddof=1), record["bootstrap_sd"]))
                error = max(error, assert_near(float((sampled <= 0).mean()), record["p_le_0"]))
                key = f"seed{seed}__{name}"
                error = max(
                    error,
                    assert_near(sampled, export["contrast_bootstrap_diffs"][contrast_indices[key]]),
                )
    receipt = {
        "study": "E483",
        "status": "all_checked_values_match",
        "arms": len(points),
        "global_contrasts": len(contrast_indices),
        "maximum_absolute_error": error,
        "source_sha256": file_hash(Path(__file__)),
        "method_source_sha256": file_hash(HERE / "reproduce_activation_exports.py"),
        "result_sha256": file_hash(result_path),
        "export_sha256": file_hash(export_path),
        "scope": "independent arithmetic on saved exports; no inference, refit, or public confirmation",
    }
    (HERE / "checks/momentum_activation_export_reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
