"""Reproduce E484 scores and target-wise contrasts with separate raw sums."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reproduce_activation_exports import assert_near, component_statistics, draws, file_hash, scores

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Check all saved statistics against independently regenerated draws."""
    directory = HERE / "activation_context_mixture/eda/twelve_v1"
    result_path = directory / "scores.json"
    export_path = directory / "paired_components.npz"
    result = json.loads(result_path.read_text())
    error = 0.0
    points, boot = {}, {}
    with np.load(export_path, allow_pickle=False) as export:
        names = export["arm_names"].tolist()
        ids = export["component_ids"]
        counts = export["selected_row_counts"]
        mask = export["selected_mask"]
        if (
            mask.shape != (16, 20000)
            or mask.dtype != np.bool_
            or export["targets_raw"].shape != (16, 20000, 2)
            or export["predictions_raw"].shape != (12, 16, 20000, 2)
        ):
            raise ValueError("raw export shape or mask dtype mismatch")
        if len(names) != 12 or len(np.unique(ids)) != 16:
            raise ValueError("incomplete arm/component matrix")
        if not np.array_equal(mask.sum(axis=1), counts):
            raise ValueError("mask/count mismatch")
        if not np.array_equal(np.sort(ids), export["sorted_component_ids"]):
            raise ValueError("component sorting mismatch")
        multiplicities = draws(271834, len(ids))
        if not np.array_equal(multiplicities, export["component_bootstrap_multiplicities"]):
            raise ValueError("bootstrap draw mismatch")
        targets = export["targets_raw"][mask]
        for index, name in enumerate(names):
            statistics = component_statistics(
                targets, export["predictions_raw"][index][mask], counts
            )[np.argsort(ids, kind="stable")]
            points[name] = scores(statistics, np.ones((1, len(ids))))[0]
            boot[name] = scores(statistics, multiplicities)
            arm = result["arm_scores"][name]
            error = max(error, assert_near(points[name], [arm["t0"], arm["t1"]]))
            error = max(error, assert_near(points[name].mean(), arm["wp"]))
            error = max(error, assert_near(boot[name], export["arm_bootstrap_wp_by_target"][index]))
            error = max(
                error, assert_near(boot[name].mean(axis=1), export["arm_bootstrap_wp"][index])
            )
        contrast_names = export["contrast_names"].tolist()
        if len(contrast_names) != 16 or set(contrast_names) != set(result["registered_contrasts"]):
            raise ValueError("contrast inventory mismatch")
        if set(contrast_names) != set(result["registered_target_contrasts"]):
            raise ValueError("target contrast inventory mismatch")
        for index, name in enumerate(contrast_names):
            record = result["registered_contrasts"][name]
            if set(result["registered_target_contrasts"][name]["targets"]) != {"t0", "t1"}:
                raise ValueError("target summary inventory mismatch")
            prefix = f"seed{record['seed']}__"
            candidate, reference = prefix + record["candidate"], prefix + record["reference"]
            point = points[candidate] - points[reference]
            sampled = boot[candidate] - boot[reference]
            global_sampled = sampled.mean(axis=1)
            error = max(
                error, assert_near(global_sampled, export["contrast_bootstrap_diffs"][index])
            )
            error = max(
                error, assert_near(sampled, export["target_contrast_bootstrap_diffs"][index])
            )
            records = [
                record,
                *(
                    result["registered_target_contrasts"][name]["targets"][f"t{t}"]
                    for t in range(2)
                ),
            ]
            for t, summary in enumerate(records):
                value = point.mean() if t == 0 else point[t - 1]
                sampled_value = global_sampled if t == 0 else sampled[:, t - 1]
                error = max(error, assert_near(value, summary["point_diff"]))
                error = max(
                    error, assert_near(np.quantile(sampled_value, [0.025, 0.975]), summary["ci95"])
                )
                error = max(error, assert_near(sampled_value.std(ddof=1), summary["bootstrap_sd"]))
                error = max(
                    error, assert_near(float((sampled_value <= 0).mean()), summary["p_le_0"])
                )
    receipt = {
        "study": "E484",
        "status": "all_checked_values_match",
        "arms": 12,
        "global_contrasts": 16,
        "target_contrasts": 32,
        "maximum_absolute_error": error,
        "source_sha256": file_hash(Path(__file__)),
        "method_source_sha256": file_hash(HERE / "reproduce_activation_exports.py"),
        "result_sha256": file_hash(result_path),
        "export_sha256": file_hash(export_path),
        "scope": "independent arithmetic on saved exports; no inference/refit/public confirmation",
    }
    (HERE / "checks/context_activation_export_reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
