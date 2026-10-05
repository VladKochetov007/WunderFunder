"""Independently reproduce E486's frozen-weight activation-swap statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reproduce_activation_exports import assert_near, component_statistics, draws, file_hash, scores

HERE = Path(__file__).resolve().parent
EXPECTED_PAIRS = {
    f"seed{seed}__beta{beta}" for seed in (271831, 271832, 271833) for beta in (0.8, 0.9, 0.95)
}
EXPECTED_CONTRASTS = {
    "D_minus_A_matched_diagonal": ("crossed_cell_difference", "sine_sine", "baseline_baseline"),
    "B_minus_A_inference_sine_on_base_fit": (
        "crossed_cell_difference",
        "baseline_sine",
        "baseline_baseline",
    ),
    "C_minus_D_remove_sine_on_sine_fit": ("crossed_cell_difference", "sine_baseline", "sine_sine"),
    "C_minus_A_train_shape_under_base_eval": (
        "crossed_cell_difference",
        "sine_baseline",
        "baseline_baseline",
    ),
    "D_minus_B_train_shape_under_sine_eval": (
        "crossed_cell_difference",
        "sine_sine",
        "baseline_sine",
    ),
    "training_by_inference_interaction": (
        "training_by_inference_difference_in_differences",
        "(D-B)",
        "(C-A)",
    ),
}


def cell_name(pair: str, alias: str) -> str:
    """Map a registered pair and shape alias to an explicit export key."""
    seed, beta = pair.split("__")
    train, evaluate = alias.split("_")
    if train not in {"baseline", "sine"} or evaluate not in {"baseline", "sine"}:
        raise ValueError("unregistered shape alias")
    return f"{seed}__train_{train}__eval_{evaluate}__{beta}"


def main() -> None:
    """Verify every cell score, registered contrast and stored resample."""
    directory = HERE / "activation_counterfactual/eda/cross_v1"
    result_path = directory / "scores.json"
    export_path = directory / "selected_exports.npz"
    receipt_path = HERE / "checks/counterfactual_activation_export_reproduction.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite arithmetic receipt: {receipt_path}")
    result = json.loads(result_path.read_text())
    if result.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError("registered metric identity mismatch")
    if (
        result.get("cell_count") != 36
        or result.get("contrast_count") != 54
        or result.get("new_fits") != 0
    ):
        raise ValueError("result experiment count mismatch")
    if set(result["contrasts_by_seed_momentum"]) != EXPECTED_PAIRS:
        raise ValueError("registered pair inventory mismatch")
    error = 0.0
    points, boot = {}, {}
    with np.load(export_path, allow_pickle=False) as export:
        names = export["cell_names"].tolist()
        ids = export["component_ids"]
        counts = export["selected_row_counts"]
        mask = export["selected_masks_by_component"]
        if len(names) != 36 or len(set(names)) != 36 or ids.shape != (16,):
            raise ValueError("incomplete cell/component matrix")
        if len(np.unique(ids)) != 16 or not np.array_equal(np.sort(ids), ids):
            raise ValueError("component ordering mismatch")
        if mask.shape != (16, 20000) or mask.dtype != np.bool_:
            raise ValueError("selected mask layout mismatch")
        if not np.array_equal(
            mask, export["need_masks_by_component"] & export["scored_masks_by_component"]
        ):
            raise ValueError("selection intersection mismatch")
        if not np.array_equal(mask.sum(axis=1), counts):
            raise ValueError("mask/count mismatch")
        if result.get("selected_rows_total") != int(counts.sum()):
            raise ValueError("result selection count mismatch")
        if not np.array_equal(result["component_ids"], ids) or not np.array_equal(
            result["row_groups"], export["row_groups"]
        ):
            raise ValueError("result/export component identity mismatch")
        if not np.array_equal(export["selected_component_index"], np.repeat(np.arange(16), counts)):
            raise ValueError("selected component identity mismatch")
        row_indices = np.concatenate([np.flatnonzero(selected) for selected in mask])
        if not np.array_equal(export["selected_row_index_within_component"], row_indices):
            raise ValueError("selected row identity mismatch")
        multiplicities = draws(271828, 16)
        if not np.array_equal(multiplicities, export["bootstrap_multiplicities"]):
            raise ValueError("bootstrap draw mismatch")
        if set(names) != set(result["cells"]):
            raise ValueError("result/export cell inventory mismatch")
        expected_cells = {
            cell_name(pair, alias)
            for pair in EXPECTED_PAIRS
            for alias in ("baseline_baseline", "baseline_sine", "sine_baseline", "sine_sine")
        }
        if set(names) != expected_cells:
            raise ValueError("registered cell inventory mismatch")
        for index, name in enumerate(names):
            statistics = component_statistics(
                export["targets_raw_selected"],
                export["predictions_raw_selected_by_cell"][index],
                counts,
            )
            target_scores = scores(statistics, np.ones((1, 16)))[0]
            cell = result["cells"][name]
            error = max(error, assert_near(target_scores, [cell["target0_wp"], cell["target1_wp"]]))
            points[name] = float(target_scores.mean())
            error = max(error, assert_near(points[name], cell["global_wp"]))
            boot[name] = scores(statistics, multiplicities).mean(axis=1)
            error = max(error, assert_near(boot[name], export["cell_bootstrap_wp"][index]))
        contrast_indices = {
            name: index for index, name in enumerate(export["contrast_names"].tolist())
        }
        observed_names = set()
        for pair, records in result["contrasts_by_seed_momentum"].items():
            if set(records) != set(EXPECTED_CONTRASTS):
                raise ValueError("registered contrast label inventory mismatch")
            for label, record in records.items():
                actual_formula = (record["family"], record["candidate"], record["reference"])
                if actual_formula != EXPECTED_CONTRASTS[label]:
                    raise ValueError("registered contrast formula mismatch")
                if record["family"] == "training_by_inference_difference_in_differences":
                    aliases = [
                        ("sine_sine", 1),
                        ("baseline_sine", -1),
                        ("sine_baseline", -1),
                        ("baseline_baseline", 1),
                    ]
                elif record["family"] == "crossed_cell_difference":
                    aliases = [(record["candidate"], 1), (record["reference"], -1)]
                else:
                    raise ValueError("unregistered contrast family")
                terms = [(cell_name(pair, alias), sign) for alias, sign in aliases]
                point = sum(sign * points[name] for name, sign in terms)
                sampled = sum(sign * boot[name] for name, sign in terms)
                error = max(error, assert_near(point, record["mean_diff"]))
                error = max(
                    error, assert_near(np.quantile(sampled, [0.025, 0.975]), record["ci95"])
                )
                error = max(error, assert_near(sampled.std(ddof=1), record["bootstrap_sd"]))
                error = max(error, assert_near(float((sampled <= 0).mean()), record["p_le_0"]))
                full = f"{pair}__{label}"
                observed_names.add(full)
                error = max(
                    error,
                    assert_near(
                        sampled, export["contrast_bootstrap_diffs"][contrast_indices[full]]
                    ),
                )
        if len(observed_names) != 54 or observed_names != set(contrast_indices):
            raise ValueError("contrast inventory mismatch")
    receipt = {
        "study": "E486",
        "status": "all_checked_values_match",
        "cells": 36,
        "global_contrasts": 54,
        "new_training_fits": 0,
        "maximum_absolute_error": error,
        "source_sha256": file_hash(Path(__file__)),
        "method_source_sha256": file_hash(HERE / "reproduce_activation_exports.py"),
        "result_sha256": file_hash(result_path),
        "export_sha256": file_hash(export_path),
        "scope": "independent arithmetic on saved exports; no inference/refit/public confirmation",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
