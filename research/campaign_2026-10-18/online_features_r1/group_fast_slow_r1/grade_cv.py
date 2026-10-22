"""Independently score component-cross-fitted innovation-feature predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from source_utils import (
    ROOT,
    load_members_and_folds,
    require_physical_scratch,
    sha256_file,
    verify_recipe_sources,
    verify_train_footer,
    write_json,
)

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.gmetric import METRIC_CLIP, METRIC_ID, GlobalAccumulator

ARMS = ("raw", "raw_plus_clock", "raw_plus_group_states")
TIME_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20261020


def _component_moments(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Compute stable weighted centered moments per target for component resampling."""
    output = np.zeros((2, 6), dtype=np.float64)
    for target_index in range(2):
        selected = np.asarray(mask, dtype=bool)
        y = np.clip(
            np.asarray(targets[:, target_index], dtype=np.float64), -METRIC_CLIP, METRIC_CLIP
        )
        p = np.clip(
            np.asarray(predictions[:, target_index], dtype=np.float64),
            -METRIC_CLIP,
            METRIC_CLIP,
        )
        if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
            raise ValueError("selected target or prediction contains a non-finite value")
        weights = np.abs(y[selected])
        positive = weights > 0.0
        if not positive.any():
            continue
        y_selected, p_selected, weights = (
            y[selected][positive],
            p[selected][positive],
            weights[positive],
        )
        total = float(weights.sum())
        # Shift first to retain precision for nearly constant component series.
        y_shifted = y_selected - y_selected[0]
        p_shifted = p_selected - p_selected[0]
        mean_y_shifted = float(weights @ y_shifted / total)
        mean_p_shifted = float(weights @ p_shifted / total)
        centered_y = y_shifted - mean_y_shifted
        centered_p = p_shifted - mean_p_shifted
        output[target_index] = (
            total,
            y_selected[0] + mean_y_shifted,
            p_selected[0] + mean_p_shifted,
            weights @ np.square(centered_y),
            weights @ np.square(centered_p),
            weights @ (centered_y * centered_p),
        )
    return output


def _scores_from_moments(moments: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Compute pooled official WP for each bootstrap draw from component moments."""
    sampled = moments[draws]
    weights = sampled[:, :, :, 0]
    total = weights.sum(axis=1)
    safe_total = np.where(total > 0.0, total, 1.0)
    mean_y = (weights * sampled[:, :, :, 1]).sum(axis=1) / safe_total
    mean_p = (weights * sampled[:, :, :, 2]).sum(axis=1) / safe_total
    delta_y = sampled[:, :, :, 1] - mean_y[:, None, :]
    delta_p = sampled[:, :, :, 2] - mean_p[:, None, :]
    variance_y = (sampled[:, :, :, 3] + weights * np.square(delta_y)).sum(axis=1)
    variance_p = (sampled[:, :, :, 4] + weights * np.square(delta_p)).sum(axis=1)
    covariance = (sampled[:, :, :, 5] + weights * delta_y * delta_p).sum(axis=1)
    denom = np.sqrt(np.maximum(variance_y * variance_p, 0.0))
    scale_y = np.sqrt(np.divide(variance_y, total, out=np.zeros_like(total), where=total > 0.0))
    scale_p = np.sqrt(np.divide(variance_p, total, out=np.zeros_like(total), where=total > 0.0))
    per_target = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 1e-16)
    per_target = np.where(
        (total > 0.0) & (scale_y > 1e-8) & (scale_p > 1e-8),
        np.clip(per_target, -1.0, 1.0),
        0.0,
    )
    return per_target.mean(axis=1)


def _paired_component_bootstrap(
    moments_by_arm: dict[str, np.ndarray], component_count: int
) -> dict[str, dict[str, float | list[float]]]:
    """Return paired component-bootstrap deltas conditional on fixed CV fits."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, component_count, size=(BOOTSTRAP_DRAWS, component_count))
    baseline = _scores_from_moments(moments_by_arm["raw"], draws)
    output: dict[str, dict[str, float | list[float]]] = {}
    for arm in ARMS[1:]:
        deltas = _scores_from_moments(moments_by_arm[arm], draws) - baseline
        output[arm] = {
            "point_delta": None,
            "component_bootstrap_95_ci": [
                float(value) for value in np.quantile(deltas, [0.025, 0.975])
            ],
        }
    return output


def run(
    root: Path,
    recipe_path: Path,
    fit_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Read held-out component labels only in the grader and score OOF predictions."""
    started = time.perf_counter()
    output = require_physical_scratch(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing grade artifacts: {output}")
    output.mkdir(parents=True)

    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    source_hashes = verify_recipe_sources(root, recipe)
    members, fold_by_component, input_hashes = load_members_and_folds(root, recipe)
    fit_receipt_path = require_physical_scratch(fit_receipt_path.resolve(strict=True))
    fit_receipt = json.loads(fit_receipt_path.read_text())
    if fit_receipt["recipe_sha256"] != hashlib.sha256(recipe_bytes).hexdigest():
        raise ValueError("fit/prediction receipt belongs to another recipe")
    expected_members = [
        {**member, "fold": fold_by_component[member["component"]]} for member in members
    ]
    if fit_receipt["members"] != expected_members:
        raise ValueError("fit/prediction receipt has different sample or fold membership")
    recorded_members_path = require_physical_scratch(
        Path(fit_receipt["members_path"]).resolve(strict=True)
    )
    if sha256_file(recorded_members_path) != fit_receipt["members_sha256"]:
        raise ValueError("fit/prediction member manifest hash differs")
    if json.loads(recorded_members_path.read_text())["members"] != expected_members:
        raise ValueError("fit/prediction member manifest differs from the frozen sample")
    model_path = require_physical_scratch(Path(fit_receipt["model_archive"]).resolve(strict=True))
    if sha256_file(model_path) != fit_receipt["model_archive_sha256"]:
        raise ValueError("cross-fit model archive hash differs")
    for arm in ARMS:
        prediction_path = require_physical_scratch(
            Path(fit_receipt["prediction_paths"][arm]).resolve(strict=True)
        )
        if sha256_file(prediction_path) != fit_receipt["prediction_sha256"][arm]:
            raise ValueError(f"prediction hash differs for arm {arm}")

    prediction_arrays = {
        arm: np.load(fit_receipt["prediction_paths"][arm], mmap_mode="r", allow_pickle=False)
        for arm in ARMS
    }
    expected_shape = (len(members), SEQUENCE_LENGTH - WARMUP, 2)
    if any(value.shape != expected_shape for value in prediction_arrays.values()):
        raise ValueError(f"OOF prediction shape differs from frozen design: {expected_shape}")

    selection = recipe["selection"]
    train_path = root / selection["data_path"]
    train_identity = verify_train_footer(train_path, selection["data_identity"])
    # The trusted metadata route is checked before the Parquet reader opens the file.
    source = SequenceFile(train_path)
    if len(source) != int(selection["row_groups"]):
        raise ValueError("train Parquet row-group count differs from the frozen recipe")

    accumulators = {arm: GlobalAccumulator() for arm in ARMS}
    fold_accumulators = {
        arm: {fold: GlobalAccumulator() for fold in range(int(recipe["cv"]["fold_count"]))}
        for arm in ARMS
    }
    bin_accumulators = {
        arm: {index: GlobalAccumulator() for index in range(len(TIME_BINS))} for arm in ARMS
    }
    bin_selected_rows = {index: 0 for index in range(len(TIME_BINS))}
    component_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    moments_by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}

    for position, member in enumerate(members):
        sequence = source.read(member["row_group"], targets=True, mask=True)
        if (
            sequence.seq_ix != member["seq_ix"]
            or sequence.targets is None
            or sequence.scored is None
        ):
            raise ValueError(f"grade identity/labels/mask unavailable for {member['row_group']}")
        targets = sequence.targets[WARMUP:]
        mask = sequence.scored[WARMUP:]
        fold = fold_by_component[member["component"]]
        for bin_index, (row_start, row_end) in enumerate(TIME_BINS):
            begin, end = row_start - WARMUP, row_end - WARMUP
            bin_selected_rows[bin_index] += int(mask[begin:end].sum())
        component_moments: dict[str, np.ndarray] = {}
        for arm in ARMS:
            predictions = np.asarray(prediction_arrays[arm][position])
            added = accumulators[arm].add(targets, predictions, mask, group=member["component"])
            fold_accumulators[arm][fold].add(targets, predictions, mask)
            component_accumulator = GlobalAccumulator()
            component_accumulator.add(targets, predictions, mask)
            component_scores[arm].append(float(component_accumulator.result()["wp"]))
            component_moments[arm] = _component_moments(targets, predictions, mask)
            for bin_index, (row_start, row_end) in enumerate(TIME_BINS):
                begin, end = row_start - WARMUP, row_end - WARMUP
                bin_accumulators[arm][bin_index].add(
                    targets[begin:end], predictions[begin:end], mask[begin:end]
                )
            if added.shape != (2, 6):
                raise RuntimeError("official metric accumulator returned an unexpected shape")
        for arm in ARMS:
            moments_by_arm[arm].append(component_moments[arm])
        del sequence

    pooled = {arm: accumulators[arm].result() for arm in ARMS}
    fold_scores = {
        arm: {
            str(fold): (
                fold_accumulators[arm][fold].result()
                if fold_accumulators[arm][fold].selected_rows
                else None
            )
            for fold in fold_accumulators[arm]
        }
        for arm in ARMS
    }
    bin_scores = {
        arm: {
            f"{TIME_BINS[index][0]}_{TIME_BINS[index][1]}": (
                bin_accumulators[arm][index].result() if bin_selected_rows[index] else None
            )
            for index in bin_accumulators[arm]
        }
        for arm in ARMS
    }
    score_deltas = {arm: float(pooled[arm]["wp"] - pooled["raw"]["wp"]) for arm in ARMS[1:]}
    moments_arrays = {arm: np.stack(values) for arm, values in moments_by_arm.items()}
    paired_intervals = _paired_component_bootstrap(moments_arrays, len(members))
    for arm in ARMS[1:]:
        paired_intervals[arm]["point_delta"] = score_deltas[arm]

    result = {
        "experiment_id": recipe["experiment_id"],
        "metric_id": METRIC_ID,
        "status": "completed_component_crossfit_incrementality_screen",
        "sample": {
            "components": len(members),
            "rows_per_component": SEQUENCE_LENGTH,
            "fit_membership": "32 source-qualified fold0.fit.train components, grouped 4-fold cross-fit",
            "out_of_fold_scoring": True,
            "previously_exposed_development_sample": True,
            "masks_used_only_by_grader": True,
        },
        "pooled_oof_scores": pooled,
        "delta_vs_raw": score_deltas,
        "paired_component_bootstrap": paired_intervals,
        "fold_scores": fold_scores,
        "position_bin_scores": bin_scores,
        "component_diagnostics": {
            arm: {
                "median_component_wp": float(np.median(values)),
                "component_win_rate_vs_raw": (
                    None
                    if arm == "raw"
                    else float(np.mean(np.asarray(values) > np.asarray(component_scores["raw"])))
                ),
            }
            for arm, values in component_scores.items()
        },
        "uncertainty_scope": "Paired component bootstrap conditions on these four fitted cross-validation partitions and the selected 32 previously exposed fit-side components; it does not estimate seed variance or clean validation transfer.",
        "claim_limit": "A positive delta indicates incremental linear signal in these fit-side components only. It does not establish recurrent endpoint gain, competition validation gain, online feedback, or public transfer.",
        "fit_predict_receipt_sha256": sha256_file(fit_receipt_path),
        "source_hashes": source_hashes,
        "sample_source_hashes": input_hashes,
        "data_identity": train_identity,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False,
    }
    result_path = output / "grade_result.json"
    write_json(result_path, result)
    grade_receipt = {
        "experiment_id": recipe["experiment_id"],
        "phase": "independent_official_metric_grade_complete",
        "status": "completed",
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "recipe_sha256": fit_receipt["recipe_sha256"],
        "source_hashes": source_hashes,
        "fit_predict_receipt_sha256": sha256_file(fit_receipt_path),
        "labels_and_masks_read": "held-out fold component labels/masks after OOF predictions were frozen",
        "pooled_scores": {arm: pooled[arm]["wp"] for arm in ARMS},
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False,
    }
    receipt_path = output / "grade_receipt.json"
    write_json(receipt_path, grade_receipt)
    print(json.dumps({"status": grade_receipt["status"], "result": str(result_path)}), flush=True)
    return grade_receipt


def parse_args() -> argparse.Namespace:
    """Parse the frozen recipe, fit receipt, and scratch output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / Path(__file__).relative_to(ROOT).parent / "recipe.json",
    )
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the label-reading stage after OOF prediction completion."""
    arguments = parse_args()
    run(
        arguments.root.resolve(),
        arguments.recipe.resolve(),
        arguments.fit_receipt,
        arguments.output,
    )


if __name__ == "__main__":
    main()
