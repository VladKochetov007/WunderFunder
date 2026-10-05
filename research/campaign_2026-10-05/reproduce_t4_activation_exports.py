"""Reproduce all E482 scores with independent shifted raw-sum arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reproduce_activation_exports import assert_near, component_statistics, draws, file_hash, scores

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Validate all six target scores, ten contrasts and saved bootstrap draws."""
    directory = HERE / "activation_t4_replication_v2/eda/six_v2"
    result_path, export_path = directory / "scores.json", directory / "selected_exports.npz"
    result = json.loads(result_path.read_text())
    points, boot = {}, {}
    error = 0.0
    with np.load(export_path, allow_pickle=False) as export:
        names, ids, counts = (
            export["arm_names"].tolist(),
            export["component_ids"],
            export["selected_row_counts"],
        )
        if len(names) != 6 or len(np.unique(ids)) != 16 or len(result["contrasts"]) != 10:
            raise ValueError("incomplete E482 matrix")
        if not np.array_equal(export["selected_mask"].sum(axis=1), counts):
            raise ValueError("mask selection/count mismatch")
        multiplicities = draws(271828, len(ids))
        if not np.array_equal(multiplicities, export["component_bootstrap_multiplicities"]):
            raise ValueError("regenerated bootstrap draws differ")
        for index, name in enumerate(names):
            statistics = component_statistics(
                export["targets_raw"], export["predictions_raw"][index], counts
            )[np.argsort(ids, kind="stable")]
            targets = scores(statistics, np.ones((1, len(ids))))[0]
            arm = result["arm_scores"][name]
            error = max(error, assert_near(targets, [arm["t0"], arm["t1"]]))
            points[name] = float(targets.mean())
            error = max(error, assert_near(points[name], arm["wp"]))
            boot[name] = scores(statistics, multiplicities).mean(axis=1)
            error = max(error, assert_near(boot[name], export["arm_bootstrap_wp"][index]))
        contrast_indices = {name: i for i, name in enumerate(export["contrast_names"])}
        if set(contrast_indices) != set(result["contrasts"]):
            raise ValueError("contrast export inventory mismatch")
        for name, record in result["contrasts"].items():
            if record["family"] == "shape_by_optimizer_interaction":
                terms = [
                    ("muon_adamw__cubic", 1),
                    ("muon_adamw__sine", -1),
                    ("adamw__cubic", -1),
                    ("adamw__sine", 1),
                ]
            else:
                terms = [(record["candidate"], 1), (record["reference"], -1)]
            point = sum(sign * points[arm] for arm, sign in terms)
            sampled = sum(sign * boot[arm] for arm, sign in terms)
            error = max(error, assert_near(point, record["mean_diff"]))
            error = max(error, assert_near(np.quantile(sampled, [0.025, 0.975]), record["ci95"]))
            error = max(error, assert_near(sampled.std(ddof=1), record["bootstrap_sd"]))
            error = max(error, assert_near(float((sampled <= 0).mean()), record["p_le_0"]))
            error = max(
                error,
                assert_near(sampled, export["contrast_bootstrap_diffs"][contrast_indices[name]]),
            )
    receipt = {
        "study": "E482",
        "status": "all_checked_values_match",
        "arms": len(points),
        "global_contrasts": len(contrast_indices),
        "maximum_absolute_error": error,
        "source_sha256": file_hash(Path(__file__)),
        "method_source_sha256": file_hash(HERE / "reproduce_activation_exports.py"),
        "result_sha256": file_hash(result_path),
        "export_sha256": file_hash(export_path),
        "scope": "independent arithmetic on saved exports; no inference, refit or new-market confirmation",
    }
    (HERE / "checks/t4_activation_export_reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
