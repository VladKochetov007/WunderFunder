from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from disagreement_screen import E514Inputs, load_e514_inputs
from numpy.typing import NDArray
from scipy.optimize import minimize

SOURCE_ORDER = ("anchor", "ab12", "ab48", "baseline")
SOURCE_SCALE = {"anchor": 1.0, "ab12": 1.0, "ab48": 1.0, "baseline": 0.25}
BOOTSTRAP_SEED = 2_985_631_662
BOOTSTRAP_REPLICATES = 2_000
E516_RELATIVE_DIR = Path(
    "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
)


@dataclass(frozen=True)
class SimplexFit:
    """One target's weighted convex blend and fitted intercept."""

    target: int
    weights: tuple[float, ...]
    intercept: float
    calibration_weighted_mse: float
    calibration_chain_count: int


def _source_array(inputs: E514Inputs, name: str) -> NDArray[np.float32]:
    if name == "anchor":
        return inputs.anchor
    return inputs.candidate_predictions[name]


def fit_target_simplex(
    inputs: E514Inputs,
    calibration_rows: NDArray[np.integer],
    target_index: int,
    calibration_chain_count: int,
    *,
    warmup_rows: int = 99,
    clip: float = 2.0,
) -> SimplexFit:
    """Fit a target-specific weighted-MSE simplex using calibration groups."""
    if target_index not in (0, 1):
        raise ValueError("target index must be 0 or 1")
    if calibration_rows.ndim != 1 or calibration_rows.size == 0:
        raise ValueError("calibration rows must be a nonempty one-dimensional index")
    if calibration_rows.min() < 0 or calibration_rows.max() >= inputs.targets.shape[0]:
        raise ValueError("calibration row index is outside the source arrays")
    if calibration_chain_count < 1:
        raise ValueError("at least one calibration chain is required")
    if not 0 <= warmup_rows < inputs.targets.shape[1] or clip <= 0:
        raise ValueError("warmup or clip setting is invalid")

    source_count = len(SOURCE_ORDER)
    sum_weight = 0.0
    sum_x = np.zeros(source_count, dtype=np.float64)
    sum_y = 0.0
    sum_xx = np.zeros((source_count, source_count), dtype=np.float64)
    sum_xy = np.zeros(source_count, dtype=np.float64)
    row_mask = np.arange(inputs.targets.shape[1]) >= warmup_rows

    for group in calibration_rows:
        target = np.clip(inputs.targets[group, :, target_index].astype(np.float64), -clip, clip)
        weights = np.abs(target) * (inputs.is_scored[group] & row_mask)
        selected = weights > 0
        if not selected.any():
            continue
        features = np.column_stack(
            [
                np.clip(
                    _source_array(inputs, name)[group, :, target_index].astype(np.float64)
                    * SOURCE_SCALE[name],
                    -clip,
                    clip,
                )
                for name in SOURCE_ORDER
            ]
        )
        if not np.isfinite(features[selected]).all() or not np.isfinite(target[selected]).all():
            raise ValueError("selected calibration values must be finite")
        x = features[selected]
        y = target[selected]
        w = weights[selected]
        sum_weight += float(w.sum(dtype=np.float64))
        sum_x += x.T @ w
        sum_y += float(y @ w)
        sum_xx += x.T @ (x * w[:, None])
        sum_xy += x.T @ (y * w)

    if not np.isfinite(sum_weight) or sum_weight <= 0:
        raise ValueError("calibration rows contain no official metric weight")
    mean_x = sum_x / sum_weight
    mean_y = sum_y / sum_weight
    covariance_xx = (sum_xx - np.outer(sum_x, sum_x) / sum_weight) / sum_weight
    covariance_xy = (sum_xy - sum_x * mean_y) / sum_weight
    covariance_xx = 0.5 * (covariance_xx + covariance_xx.T)

    ridge = 1e-8 * max(float(np.trace(covariance_xx)) / source_count, 1e-12)
    anchor_vertex = np.zeros(source_count, dtype=np.float64)
    anchor_vertex[0] = 1.0

    def objective(weights: NDArray[np.float64]) -> float:
        residual = weights - anchor_vertex
        return float(
            0.5 * weights @ covariance_xx @ weights
            - covariance_xy @ weights
            + 0.5 * ridge * (residual @ residual)
        )

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return covariance_xx @ weights - covariance_xy + ridge * (weights - anchor_vertex)

    fit = minimize(
        objective,
        anchor_vertex.copy(),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * source_count,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(weights.sum() - 1.0),
                "jac": lambda weights: np.ones_like(weights),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"simplex optimizer failed: {fit.message}")
    weights = np.clip(fit.x, 0.0, 1.0)
    weights /= weights.sum()
    intercept = float(mean_y - mean_x @ weights)
    weighted_mse = 0.0
    for group in calibration_rows:
        target = np.clip(inputs.targets[group, :, target_index].astype(np.float64), -clip, clip)
        metric_weights = np.abs(target) * (inputs.is_scored[group] & row_mask)
        selected = metric_weights > 0
        features = np.column_stack(
            [
                np.clip(
                    _source_array(inputs, name)[group, :, target_index].astype(np.float64)
                    * SOURCE_SCALE[name],
                    -clip,
                    clip,
                )
                for name in SOURCE_ORDER
            ]
        )
        prediction = np.clip(features @ weights + intercept, -clip, clip)
        residual = target[selected] - prediction[selected]
        weighted_mse += float(np.sum(metric_weights[selected] * residual**2))
    weighted_mse /= sum_weight
    return SimplexFit(
        target=target_index,
        weights=tuple(map(float, weights)),
        intercept=intercept,
        calibration_weighted_mse=weighted_mse,
        calibration_chain_count=calibration_chain_count,
    )


def predict_groups(
    inputs: E514Inputs,
    group_rows: NDArray[np.integer],
    fits: Sequence[SimplexFit],
    *,
    clip: float = 2.0,
) -> NDArray[np.float32]:
    """Apply the two frozen target-wise fits to selected groups."""
    if len(fits) != 2 or tuple(fit.target for fit in fits) != (0, 1):
        raise ValueError("one ordered fit is required for each of the two targets")
    if group_rows.ndim != 1 or group_rows.size == 0:
        raise ValueError("prediction rows must be a nonempty one-dimensional index")
    output = np.empty((group_rows.size, inputs.targets.shape[1], 2), dtype=np.float32)
    for position, group in enumerate(group_rows):
        for target_index, fit in enumerate(fits):
            features = np.column_stack(
                [
                    np.clip(
                        _source_array(inputs, name)[group, :, target_index].astype(np.float64)
                        * SOURCE_SCALE[name],
                        -clip,
                        clip,
                    )
                    for name in SOURCE_ORDER
                ]
            )
            output[position, :, target_index] = np.clip(
                features @ np.asarray(fit.weights) + fit.intercept, -clip, clip
            ).astype(np.float32)
    return output


def experiment_identity(inputs: E514Inputs) -> dict[str, str]:
    """Bind the E-516 results to their preregistration and frozen inputs."""
    return {
        "experiment_id": "E-516",
        "hypothesis_id": "H-206",
        "bandit_arm": "arm:targetwise-weak-simplex",
        "source_manifest": (
            "research/campaign_2026-10-06/bandit_continuation/"
            "conditional_weak_portfolio_E514/SOURCE_MANIFEST_E514_V3.json"
        ),
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "chain_partition": (
            "research/campaign_2026-10-06/bandit_continuation/"
            "conditional_weak_portfolio_E514/E514_CHAIN_PARTITION.json"
        ),
        "chain_partition_sha256": inputs.chain_partition_sha256,
    }


def fit_and_crossfit(inputs: E514Inputs) -> tuple[dict[str, object], NDArray[np.float32]]:
    """Fit calibration-only weights and score every frozen outer fold once."""
    from wnn.gmetric import GlobalAccumulator, global_wp

    chain_ids = np.unique(inputs.chain_by_sequence)
    outer_fold_ids = sorted(set(inputs.chain_to_outer_fold.values()))
    if outer_fold_ids != [0, 1, 2]:
        raise ValueError("expected the three frozen outer folds 0, 1, and 2")
    oof = np.empty_like(inputs.targets, dtype=np.float32)
    warmup = np.arange(inputs.targets.shape[1]) >= 99
    metric_mask = inputs.is_scored & warmup[None, :]
    fold_records: list[dict[str, object]] = []

    for outer_fold in outer_fold_ids:
        eval_chains = [
            chain for chain in chain_ids if inputs.chain_to_outer_fold[int(chain)] == outer_fold
        ]
        calibration_chains = [
            chain for chain in chain_ids if inputs.chain_to_outer_fold[int(chain)] != outer_fold
        ]
        eval_rows = np.flatnonzero(np.isin(inputs.chain_by_sequence, eval_chains))
        calibration_rows = np.flatnonzero(np.isin(inputs.chain_by_sequence, calibration_chains))
        fits = [
            fit_target_simplex(
                inputs,
                calibration_rows,
                target_index,
                len(calibration_chains),
            )
            for target_index in (0, 1)
        ]
        oof[eval_rows] = predict_groups(inputs, eval_rows, fits)
        anchor_fold_score = global_wp(
            inputs.targets[eval_rows],
            inputs.anchor[eval_rows],
            metric_mask[eval_rows],
        )
        stack_fold_score = global_wp(
            inputs.targets[eval_rows], oof[eval_rows], metric_mask[eval_rows]
        )
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "evaluation_chains": len(eval_chains),
                "calibration_chains": len(calibration_chains),
                "anchor": anchor_fold_score,
                "simplex_stack": stack_fold_score,
                "fits": [asdict(fit) for fit in fits],
            }
        )

    anchor_accumulator = GlobalAccumulator()
    stack_accumulator = GlobalAccumulator()
    anchor_moments: dict[int, NDArray[np.float64]] = {}
    stack_moments: dict[int, NDArray[np.float64]] = {}
    for chain in chain_ids:
        rows = np.flatnonzero(inputs.chain_by_sequence == chain)
        chain_anchor = GlobalAccumulator()
        chain_stack = GlobalAccumulator()
        chain_anchor.add(inputs.targets[rows], inputs.anchor[rows], metric_mask[rows])
        chain_stack.add(inputs.targets[rows], oof[rows], metric_mask[rows])
        anchor_accumulator.add(inputs.targets[rows], inputs.anchor[rows], metric_mask[rows])
        stack_accumulator.add(inputs.targets[rows], oof[rows], metric_mask[rows])
        anchor_moments[int(chain)] = chain_anchor._moments.copy()
        stack_moments[int(chain)] = chain_stack._moments.copy()
    anchor_score = anchor_accumulator.result()
    stack_score = stack_accumulator.result()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_delta = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    fold_chains = {
        fold: np.asarray(
            [chain for chain in chain_ids if inputs.chain_to_outer_fold[int(chain)] == fold],
            dtype=np.int64,
        )
        for fold in outer_fold_ids
    }
    from wnn.gmetric import _wp_from_moments

    for replicate in range(BOOTSTRAP_REPLICATES):
        anchor_total = np.zeros((2, 6), dtype=np.float64)
        stack_total = np.zeros((2, 6), dtype=np.float64)
        for fold in outer_fold_ids:
            sampled = rng.choice(fold_chains[fold], size=fold_chains[fold].size, replace=True)
            for chain in sampled:
                anchor_total = _merge_pair(anchor_total, anchor_moments[int(chain)])
                stack_total = _merge_pair(stack_total, stack_moments[int(chain)])
        anchor_wp = 0.5 * sum(_wp_from_moments(row) for row in anchor_total)
        stack_wp = 0.5 * sum(_wp_from_moments(row) for row in stack_total)
        bootstrap_delta[replicate] = stack_wp - anchor_wp

    folds_with_weak_mass = sum(
        any(sum(fit["weights"][1:]) > 1e-6 for fit in record["fits"]) for record in fold_records
    )
    result: dict[str, object] = {
        **experiment_identity(inputs),
        "metric": "global_wp_clip2_v1",
        "data_exposure": "previously inspected fold 0; exploratory, not confirmation",
        "source_order": list(SOURCE_ORDER),
        "source_scales": SOURCE_SCALE,
        "anchor": anchor_score,
        "simplex_stack": stack_score,
        "delta_vs_anchor": float(stack_score["wp"] - anchor_score["wp"]),
        "paired_chain_bootstrap_delta_95pct": np.quantile(bootstrap_delta, [0.025, 0.975]).tolist(),
        "paired_chain_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_chain_bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_conditional_on_outer_fits": True,
        "folds": fold_records,
        "folds_with_positive_total_weak_mass": int(folds_with_weak_mass),
        "oof_prediction_data_sha256": hashlib.sha256(
            memoryview(oof).cast("B")
        ).hexdigest(),
        "new_public_gain_verified": False,
        "expected_public_gain": None,
        "public_forecast": None,
        "numerical_prior_update": None,
    }
    return result, oof


def _merge_pair(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Merge two per-target centered-moment tables with the metric's stable formula."""
    from wnn.gmetric import _merge_moments

    return np.stack([_merge_moments(left[i], right[i]) for i in range(2)])


def save_result(result: Mapping[str, object], output_path: Path) -> None:
    """Persist a deterministic, human-readable JSON result."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    project_root = Path.cwd()
    source_inputs = load_e514_inputs(project_root)
    result, _ = fit_and_crossfit(source_inputs)
    result_path = project_root / E516_RELATIVE_DIR / "E516_RESULT.json"
    save_result(result, result_path)
    print(json.dumps(result, indent=2, sort_keys=True))
