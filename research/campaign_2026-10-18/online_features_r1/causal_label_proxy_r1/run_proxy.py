from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np
from weighted_ridge import merge_weighted_batch

from wnn.data import N_FEATURES, WARMUP, SequenceFile
from wnn.gmetric import METRIC_CLIP, METRIC_ID, GlobalAccumulator, wp_from_sums
from wnn.splits import CampaignSplit, load_campaign_split

LOG = logging.getLogger(__name__)
SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp").resolve()
BOOTSTRAP_DRAWS = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _prepare(recipe_path: Path, root: Path) -> tuple[dict[str, object], CampaignSplit, str]:
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    if recipe["metric_id"] != METRIC_ID:
        raise ValueError(f"unexpected metric: {recipe['metric_id']}")
    selection = recipe["selection"]
    manifest_path = root / str(selection["split_manifest"])
    split = load_campaign_split(manifest_path, root)
    if split.sha256 != selection["split_manifest_sha256"]:
        raise ValueError("source-qualified split manifest hash changed")
    component_of = {(side, group): component for side, group, component in split.components}
    fit_groups = [int(item["row_group"]) for item in selection["fit_groups"]]
    eval_groups = [int(item["row_group"]) for item in selection["evaluation_groups"]]
    split.assert_fit_groups(fit_groups, [])
    if not set(eval_groups).issubset(split.evaluation_groups):
        raise ValueError("evaluation groups are outside the frozen F0 membership")
    for side, items in (
        ("train", selection["fit_groups"]),
        ("valid", selection["evaluation_groups"]),
    ):
        for item in items:
            group = int(item["row_group"])
            if component_of[(side, group)] != int(item["component"]):
                raise ValueError(f"component identity changed for {side} row group {group}")
    if {int(item["component"]) for item in selection["fit_groups"]} & {
        int(item["component"]) for item in selection["evaluation_groups"]
    }:
        raise ValueError("fit and evaluation component overlap")
    return recipe, split, hashlib.sha256(recipe_bytes).hexdigest()


def _design(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    current = features[indices]
    change = current - features[indices - 1]
    return np.concatenate((current, change), axis=1).astype(np.float64, copy=False)


def _fit(recipe_path: Path, root: Path, artifact_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    recipe, split, recipe_sha = _prepare(recipe_path, root)
    out = artifact_dir.resolve()
    if not out.is_relative_to(SCRATCH_ROOT):
        raise ValueError(f"artifacts must stay under physical scratch: {SCRATCH_ROOT}")
    model_path = out / "model.npz"
    receipt_path = out / "fit_receipt.json"
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite fit artifacts")
    out.mkdir(parents=True, exist_ok=True)

    dimensions = 2 * N_FEATURES
    total_weight = np.zeros(2, dtype=np.float64)
    mean_x = np.zeros((2, dimensions), dtype=np.float64)
    mean_y = np.zeros(2, dtype=np.float64)
    centered_xx = np.zeros((2, dimensions, dimensions), dtype=np.float64)
    centered_xy = np.zeros((2, dimensions), dtype=np.float64)
    sample_count = 0
    source = SequenceFile(root / "datasets/train.parquet")
    fit_config = recipe["model"]
    stride = int(fit_config["fit_step_stride"])
    ridge_alpha = float(fit_config["ridge_alpha"])
    groups = recipe["selection"]["fit_groups"]
    for position, item in enumerate(groups, start=1):
        row_group = int(item["row_group"])
        sequence = source.read(row_group, targets=True, mask=False)
        if sequence.seq_ix != int(item["seq_ix"]) or sequence.targets is None:
            raise ValueError(f"fit sequence identity/targets mismatch at row group {row_group}")
        indices = np.arange(WARMUP + 1, sequence.features.shape[0], stride)
        design = _design(sequence.features, indices)
        targets = np.clip(sequence.targets[indices - 1], -METRIC_CLIP, METRIC_CLIP).astype(
            np.float64, copy=False
        )
        if not np.isfinite(design).all() or not np.isfinite(targets).all():
            raise ValueError(f"nonfinite fit values in row group {row_group}")
        for target_index in range(2):
            weights = np.abs(targets[:, target_index])
            (
                total_weight[target_index],
                mean_x[target_index],
                mean_y[target_index],
                centered_xx[target_index],
                centered_xy[target_index],
            ) = merge_weighted_batch(
                total_weight[target_index],
                mean_x[target_index],
                mean_y[target_index],
                centered_xx[target_index],
                centered_xy[target_index],
                design,
                targets[:, target_index],
                weights,
            )
        sample_count += int(targets.shape[0])
        LOG.info("fit source row group %d/%d complete", position, len(groups))

    means = np.zeros((2, dimensions), dtype=np.float64)
    scales = np.ones((2, dimensions), dtype=np.float64)
    coefficients = np.zeros((2, dimensions), dtype=np.float64)
    intercepts = np.zeros(2, dtype=np.float64)
    scale_summaries: list[dict[str, float | int]] = []
    for target_index in range(2):
        weight_sum = total_weight[target_index]
        if weight_sum <= 0:
            raise ValueError(f"target {target_index} has zero training weight")
        covariance = centered_xx[target_index] / weight_sum
        covariance = (covariance + covariance.T) * 0.5
        scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        normalized_covariance = covariance / np.outer(scale, scale)
        cross_covariance = centered_xy[target_index] / weight_sum
        normalized_cross = cross_covariance / scale
        solution = np.linalg.solve(
            normalized_covariance + ridge_alpha * np.eye(dimensions), normalized_cross
        )
        means[target_index] = mean_x[target_index]
        scales[target_index] = scale
        coefficients[target_index] = solution / scale
        intercepts[target_index] = (
            mean_y[target_index] - mean_x[target_index] @ coefficients[target_index]
        )
        scale_summaries.append(
            {
                "min": float(scale.min()),
                "median": float(np.median(scale)),
                "max": float(scale.max()),
                "floored_count": int(np.count_nonzero(np.diag(covariance) <= 1e-12)),
            }
        )

    with model_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            means=means,
            scales=scales,
            coefficients=coefficients,
            intercepts=intercepts,
        )
    fit_groups_sha = hashlib.sha256(
        json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = {
        "experiment_id": recipe["id"],
        "phase": "fit_complete",
        "status": "completed",
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "recipe_sha256": recipe_sha,
        "runner_sha256": _sha256(Path(__file__)),
        "split_manifest_sha256": split.sha256,
        "fit_groups_sha256": fit_groups_sha,
        "fit_row_groups": len(groups),
        "fit_components": len({int(item["component"]) for item in groups}),
        "fit_sampled_rows_per_target": sample_count,
        "fit_step_stride": stride,
        "weighted_ridge_alpha": ridge_alpha,
        "weighted_feature_scale_summary_by_target": scale_summaries,
        "moment_accumulation": "stable centered weighted batch merge across fit sequences",
        "official_metric_used_for_model_selection": False,
        "elapsed_seconds": time.monotonic() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "task_labels_used": "training labels only; one-row lag",
    }
    _write_json(receipt_path, receipt)
    return receipt


def _bootstrap_interval(group_sums: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    count = group_sums.shape[0]
    draws = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    scores = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index, draw in enumerate(draws):
        target_scores = wp_from_sums(group_sums[draw].sum(axis=0))
        scores[index] = float(np.mean(target_scores))
    return [float(value) for value in np.quantile(scores, [0.025, 0.975])]


def _grade(recipe_path: Path, root: Path, artifact_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    recipe, split, recipe_sha = _prepare(recipe_path, root)
    out = artifact_dir.resolve()
    if not out.is_relative_to(SCRATCH_ROOT):
        raise ValueError(f"artifacts must stay under physical scratch: {SCRATCH_ROOT}")
    model_path = out / "model.npz"
    predictions_path = out / "evaluation_proxy_predictions.npy"
    receipt_path = out / "grade_receipt.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing fitted proxy: {model_path}")
    if predictions_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite grade artifacts")
    with np.load(model_path, allow_pickle=False) as model_file:
        coefficients = model_file["coefficients"].copy()
        intercepts = model_file["intercepts"].copy()

    scorer = GlobalAccumulator()
    prediction_parts: list[np.ndarray] = []
    component_order: list[int] = []
    source = SequenceFile(root / "datasets/valid.parquet")
    groups = recipe["selection"]["evaluation_groups"]
    for position, item in enumerate(groups, start=1):
        row_group = int(item["row_group"])
        sequence = source.read(row_group, targets=True, mask=True)
        if sequence.seq_ix != int(item["seq_ix"]):
            raise ValueError(f"evaluation sequence identity mismatch at row group {row_group}")
        if sequence.targets is None or sequence.scored is None:
            raise ValueError(f"targets or grader mask unavailable at row group {row_group}")
        indices = np.arange(WARMUP + 1, sequence.features.shape[0])
        design = _design(sequence.features, indices)
        predictions = (design @ coefficients.T + intercepts).astype(np.float32)
        targets = sequence.targets[indices - 1]
        mask = sequence.scored[indices - 1]
        if not np.isfinite(predictions).all():
            raise ValueError(f"nonfinite proxy output in row group {row_group}")
        component = int(item["component"])
        scorer.add(targets, predictions, mask, group=component)
        prediction_parts.append(predictions)
        component_order.append(component)
        LOG.info("grade source row group %d/%d complete", position, len(groups))

    predictions = np.concatenate(prediction_parts, axis=0)
    with predictions_path.open("xb") as handle:
        np.save(handle, predictions, allow_pickle=False)
    group_sums = np.stack([scorer.groups[component] for component in component_order])
    official = scorer.result()
    conditional_interval = _bootstrap_interval(group_sums, seed=20261020)
    component_scores = [float(np.mean(wp_from_sums(sums))) for sums in group_sums]
    receipt = {
        "experiment_id": recipe["id"],
        "phase": "grade_complete",
        "status": "completed_exploratory",
        "metric_id": official["metric_id"],
        "lagged_proxy_wp": official["wp"],
        "target_wp": {"t0": official["t0"], "t1": official["t1"]},
        "zero_proxy_control_wp": 0.0,
        "paired_delta_vs_zero_control": official["wp"],
        "conditional_component_bootstrap_95_interval": conditional_interval,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": 20261020,
        "bootstrap_scope": "raw additive official metric sums resampled across the 32 distinct fixed components; conditional on one proxy fit",
        "component_scores": component_scores,
        "components": component_order,
        "evaluation_row_groups": len(groups),
        "evaluation_rows_per_sequence": int(sequence.features.shape[0] - WARMUP - 1),
        "scored_rows": scorer.selected_rows,
        "exposure": recipe["selection"]["exposure"],
        "test_targets_or_mask_used_by_predictor": False,
        "future_input_rows_used": False,
        "recipe_sha256": recipe_sha,
        "runner_sha256": _sha256(Path(__file__)),
        "model_sha256": _sha256(model_path),
        "prediction_path": str(predictions_path),
        "prediction_sha256": _sha256(predictions_path),
        "prediction_shape": list(predictions.shape),
        "prediction_dtype": str(predictions.dtype),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit or grade a causal lagged-target proxy.")
    parser.add_argument("--mode", choices=("fit", "grade"), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    action = _fit if args.mode == "fit" else _grade
    result = action(
        args.recipe.resolve(strict=True), args.root.resolve(strict=True), args.artifact_dir
    )
    LOG.info("result=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
