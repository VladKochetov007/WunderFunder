"""Check every E479 score and contrast with the separate shifted-sum method."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reproduce_activation_exports import (
    assert_near,
    component_statistics,
    draws,
    file_hash,
    scores,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Reproduce all sixteen arms and twenty-eight global contrasts."""
    directory = HERE / "recurrent_activation_adaptive/eda/sixteen_v2"
    result_path = directory / "RESULT.json"
    export_path = directory / "RESULT_paired_components.npz"
    result = json.loads(result_path.read_text())
    points, boot = {}, {}
    error = 0.0
    with np.load(export_path, allow_pickle=False) as export:
        names = export["registered_arm_keys"].tolist()
        ids = export["component_ids"]
        counts = export["selected_rows_per_component"]
        masks = export["score_masks_by_component"]
        if len(names) != 16 or len(np.unique(ids)) != 16:
            raise ValueError("incomplete matrix/component export")
        if not np.array_equal(masks.sum(axis=1), counts):
            raise ValueError("mask selection/count mismatch")
        for index, name in enumerate(names):
            seed = int(name.split("_")[0].removeprefix("seed"))
            statistics = component_statistics(
                export["selected_targets_raw"],
                export["selected_raw_predictions_by_arm"][index],
                counts,
            )[np.argsort(ids, kind="stable")]
            target_scores = scores(statistics, np.ones((1, len(ids))))[0]
            arm = result["arms"][name]
            error = max(
                error,
                assert_near(
                    target_scores,
                    [
                        arm["target_wp"]["t0"],
                        arm["target_wp"]["t1"],
                    ],
                ),
            )
            points[name] = float(target_scores.mean())
            error = max(error, assert_near(points[name], arm["global_wp"]))
            boot[name] = scores(statistics, draws(seed, len(ids))).mean(axis=1)
    for name, record in result["contrasts"].items():
        prefix, factor, comparison = name.split(":")
        if factor == "interaction":
            candidate, reference = comparison.split("_vs_")
            terms = [
                (f"{prefix}_muon_adamw_{candidate}", 1),
                (f"{prefix}_muon_adamw_{reference}", -1),
                (f"{prefix}_adamw_{candidate}", -1),
                (f"{prefix}_adamw_{reference}", 1),
            ]
        elif comparison == "muon_minus_adamw":
            terms = [(f"{prefix}_muon_adamw_{factor}", 1), (f"{prefix}_adamw_{factor}", -1)]
        else:
            candidate, reference = comparison.split("_minus_")
            terms = [(f"{prefix}_{factor}_{candidate}", 1), (f"{prefix}_{factor}_{reference}", -1)]
        point = sum(sign * points[arm] for arm, sign in terms)
        sampled = sum(sign * boot[arm] for arm, sign in terms)
        error = max(error, assert_near(point, record["difference"]))
        error = max(error, assert_near(np.quantile(sampled, [0.025, 0.975]), record["ci95"]))
        error = max(error, assert_near(sampled.std(ddof=1), record["bootstrap_sd"]))
        error = max(error, assert_near(float((sampled <= 0).mean()), record["p_le_0"]))
    receipt = {
        "study": "E479",
        "status": "all_checked_values_match",
        "arms": len(points),
        "global_contrasts": len(result["contrasts"]),
        "maximum_absolute_error": error,
        "source_sha256": file_hash(Path(__file__)),
        "method_source_sha256": file_hash(HERE / "reproduce_activation_exports.py"),
        "result_sha256": file_hash(result_path),
        "export_sha256": file_hash(export_path),
        "scope": "separate raw-sum method on existing exports; no inference/refit/public confirmation",
    }
    (HERE / "checks/adaptive_activation_export_reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
