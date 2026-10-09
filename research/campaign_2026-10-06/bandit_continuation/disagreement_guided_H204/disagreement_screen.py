from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

E514_DIR = Path("research/campaign_2026-10-06/bandit_continuation/conditional_weak_portfolio_E514")
E515_DIR = Path("research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204")
EXPECTED_MANIFEST_SHA256 = "8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd"
EXPECTED_PARTITION_SHA256 = "e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2"
PREDICTION_FILES = {
    "c1_s3": "cache/pred_bd26_c1v2_s3_f0_valid.npy",
    "c1_s4": "cache/pred_bd26_c1v2_s4_f0_valid.npy",
    "ab12": "cache/pred_e514_ab12_h128_plain_fallback_valid.npy",
    "ab48": "cache/pred_e514_ab48_basehead_f0_valid.npy",
    "baseline": "cache/pred_baseline_valid.npy",
}


@dataclass(frozen=True)
class E514Inputs:
    """Verified, previously exposed group-OOF inputs for E-515."""

    targets: NDArray[np.float32]
    is_scored: NDArray[np.bool_]
    anchor: NDArray[np.float32]
    candidate_predictions: Mapping[str, NDArray[np.float32]]
    chain_by_sequence: IntArray
    chain_to_outer_fold: Mapping[int, int]
    row_groups: IntArray
    source_manifest_sha256: str
    chain_partition_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e514_inputs(root: Path) -> E514Inputs:
    """Load E-514 predictions only after replaying identity and chain gates."""
    manifest_path = root / E514_DIR / "SOURCE_MANIFEST_E514_V3.json"
    partition_path = root / E514_DIR / "E514_CHAIN_PARTITION.json"
    manifest_digest = _sha256(manifest_path)
    partition_digest = _sha256(partition_path)
    if manifest_digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("E-514 V3 source manifest digest changed")
    if partition_digest != EXPECTED_PARTITION_SHA256:
        raise ValueError("E-514 chain partition digest changed")

    manifest = json.loads(manifest_path.read_text())
    partition = json.loads(partition_path.read_text())
    if manifest["chain_partition_sha256"] != partition_digest:
        raise ValueError("manifest does not bind the frozen chain partition")
    if not manifest["ancestry_and_row_order_gate"]["passed"]:
        raise ValueError("E-514 ancestry and row-order gate is not marked passed")

    target_path = root / "cache/valid_targets.npz"
    chain_path = root / "cache/chain_folds.npz"
    target_digest = _sha256(target_path)
    if target_digest != manifest["source_hashes"]["cache/valid_targets.npz"]:
        raise ValueError("target-cache digest differs from the frozen manifest")
    if _sha256(chain_path) != manifest["source_hashes"]["cache/chain_folds.npz"]:
        raise ValueError("chain-cache digest differs from the frozen manifest")

    with np.load(target_path, allow_pickle=False) as target_cache:
        targets = target_cache["y"]
        is_scored = target_cache["mask"]
        row_groups = target_cache["row_group"].astype(np.int64, copy=True)
    if targets.shape != (1873, 20_000, 2) or is_scored.shape != targets.shape[:2]:
        raise ValueError("target cache does not match the preregistered population shape")
    row_digest = hashlib.sha256(row_groups.astype("<i8", copy=False).tobytes()).hexdigest()
    if row_digest != manifest["target_row_groups_sha256_little_endian_int64"]:
        raise ValueError("target row-group order differs from the frozen manifest")

    prediction_arrays: dict[str, NDArray[np.float32]] = {}
    for name, relative_path in PREDICTION_FILES.items():
        path = root / relative_path
        if _sha256(path) != manifest["prediction_hashes"][name]:
            raise ValueError(f"{name} prediction digest differs from the manifest")
        if manifest["source_hashes"][relative_path] != manifest["prediction_hashes"][name]:
            raise ValueError(f"{name} prediction digest is inconsistent in the manifest")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(manifest["prediction_shapes"][name])
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(f"{name} predictions have an unexpected shape or dtype")
        sidecar_path = path.with_suffix(".json")
        sidecar_relative = str(sidecar_path.relative_to(root))
        if _sha256(sidecar_path) != manifest["source_hashes"][sidecar_relative]:
            raise ValueError(f"{name} prediction sidecar digest differs from the manifest")
        sidecar = json.loads(sidecar_path.read_text())
        if list(map(int, sidecar["row_groups"])) != row_groups.tolist():
            raise ValueError(f"{name} prediction row order differs from the target cache")
        prediction_arrays[name] = array

    with np.load(chain_path, allow_pickle=False) as chain_cache:
        chain_rows = chain_cache["row_group"]
        split_labels = chain_cache["split"]
        valid_chain_mask = split_labels == "valid"
        valid_groups = chain_rows[valid_chain_mask]
        valid_positions = {int(group): i for i, group in enumerate(valid_groups)}
        if set(map(int, row_groups)) != set(valid_positions):
            raise ValueError("target groups do not exactly match valid chain groups")
        target_positions = np.asarray(
            [valid_positions[int(group)] for group in row_groups], dtype=np.int64
        )
        chain_by_sequence = chain_cache["chain"][valid_chain_mask][target_positions]
        fold_by_sequence = chain_cache["fold"][valid_chain_mask][target_positions]
        train_chain_ids = np.unique(chain_cache["chain"][split_labels == "train"])

    pure_rows = np.flatnonzero(
        (fold_by_sequence == 0) & ~np.isin(chain_by_sequence, train_chain_ids)
    )
    if pure_rows.size != partition["pure_fold0_group_count"]:
        raise ValueError("pure fold-0 group count differs from the frozen partition")
    pure_chain_ids = np.unique(chain_by_sequence[pure_rows])
    chain_to_outer_fold = {int(k): int(v) for k, v in partition["chain_to_outer_fold"].items()}
    if pure_chain_ids.size != partition["pure_fold0_chain_count"]:
        raise ValueError("pure fold-0 chain count differs from the frozen partition")
    if set(map(int, pure_chain_ids)) != set(chain_to_outer_fold):
        raise ValueError("frozen outer folds do not cover exactly the pure chains")
    fold_counts = np.bincount(
        [chain_to_outer_fold[int(chain)] for chain in pure_chain_ids], minlength=3
    )
    if not np.array_equal(fold_counts, np.full(3, 41)):
        raise ValueError("each frozen outer fold must contain 41 complete chains")
    group_counts = np.bincount(
        [chain_to_outer_fold[int(chain)] for chain in chain_by_sequence[pure_rows]],
        minlength=3,
    )
    if not np.array_equal(group_counts, np.asarray([150, 130, 152])):
        raise ValueError("frozen outer-fold group counts changed")
    if any(
        np.unique(fold_by_sequence[chain_by_sequence == chain]).size != 1
        for chain in pure_chain_ids
    ):
        raise ValueError("a connected chain crosses the source split folds")

    selected_targets = targets[pure_rows].astype(np.float32, copy=True)
    selected_mask = is_scored[pure_rows].astype(np.bool_, copy=True)
    anchor = prediction_arrays["c1_s3"][pure_rows].astype(np.float64) + prediction_arrays["c1_s4"][
        pure_rows
    ].astype(np.float64)
    anchor = (0.5 * anchor).astype(np.float32)
    candidates = {
        name: prediction_arrays[name][pure_rows].astype(np.float32, copy=True)
        for name in ("ab12", "ab48", "baseline")
    }
    if not np.isfinite(selected_targets).all() or not np.isfinite(anchor).all():
        raise ValueError("selected targets or anchor predictions contain nonfinite values")
    if not all(np.isfinite(array).all() for array in candidates.values()):
        raise ValueError("selected candidate predictions contain nonfinite values")

    return E514Inputs(
        targets=selected_targets,
        is_scored=selected_mask,
        anchor=anchor,
        candidate_predictions=candidates,
        chain_by_sequence=chain_by_sequence[pure_rows].astype(np.int64),
        chain_to_outer_fold=chain_to_outer_fold,
        row_groups=row_groups[pure_rows],
        source_manifest_sha256=manifest_digest,
        chain_partition_sha256=partition_digest,
    )


def chain_diagnostics(
    y: NDArray[np.floating],
    is_scored: NDArray[np.bool_],
    anchor: NDArray[np.floating],
    candidate_predictions: Mapping[str, NDArray[np.floating]],
    candidate_scales: Mapping[str, float],
    chain_by_sequence: NDArray[np.integer],
    chain_to_outer_fold: Mapping[int, int],
    pure_rows: NDArray[np.integer],
    *,
    warmup_rows: int = 99,
    clip: float = 2.0,
) -> list[dict[str, float | int | str]]:
    """Compute weighted error and disagreement summaries per held-out chain.

    The returned error is a weighted squared-residual diagnostic, not the
    competition's pooled weighted Pearson score or an additive contribution
    to it.
    """
    if y.ndim != 3 or anchor.shape != y.shape:
        raise ValueError("targets and anchor must have matching [group,row,target] shape")
    group_count, row_count, target_count = y.shape
    if is_scored.shape != (group_count, row_count):
        raise ValueError("scored mask shape does not match targets")
    if chain_by_sequence.shape != (group_count,):
        raise ValueError("chain membership shape does not match targets")
    if any(values.shape != y.shape for values in candidate_predictions.values()):
        raise ValueError("candidate prediction shape does not match targets")
    if set(candidate_predictions) != set(candidate_scales):
        raise ValueError("candidate names and scales differ")
    if not 0 <= warmup_rows < row_count:
        raise ValueError("warmup row count is outside the sequence")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    if pure_rows.ndim != 1 or np.unique(pure_rows).size != pure_rows.size:
        raise ValueError("pure rows must be a unique one-dimensional index")
    if pure_rows.size and (pure_rows.min() < 0 or pure_rows.max() >= group_count):
        raise ValueError("pure row index is outside the targets")
    if not candidate_predictions:
        raise ValueError("at least one candidate prediction is required")
    if any(not np.isfinite(scale) or scale <= 0 for scale in candidate_scales.values()):
        raise ValueError("candidate scales must be finite and positive")

    row_mask = np.arange(row_count) >= warmup_rows
    diagnostics: list[dict[str, float | int | str]] = []

    for chain_id in np.unique(chain_by_sequence[pure_rows]):
        rows = pure_rows[chain_by_sequence[pure_rows] == chain_id]
        if rows.size == 0:
            continue
        outer_fold = chain_to_outer_fold.get(int(chain_id))
        if outer_fold is None:
            raise ValueError(f"chain {chain_id} has no frozen outer-fold assignment")
        target_values = np.clip(y[rows].astype(np.float64, copy=False), -clip, clip)
        anchor_values = np.clip(anchor[rows].astype(np.float64, copy=False), -clip, clip)
        selected = is_scored[rows] & row_mask[None, :]

        for target_index in range(target_count):
            target = target_values[:, :, target_index]
            weights = np.abs(target) * selected
            weight_sum = float(weights.sum(dtype=np.float64))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(f"chain {chain_id} target {target_index} has no metric weight")
            residual = target - anchor_values[:, :, target_index]
            error = float(np.sum(weights * residual**2, dtype=np.float64) / weight_sum)
            diagnostics.append(
                {
                    "chain_id": int(chain_id),
                    "outer_fold": int(outer_fold),
                    "target": int(target_index),
                    "candidate": "anchor_error",
                    "disagreement": 0.0,
                    "anchor_weighted_squared_error": error,
                    "weight_mass": weight_sum,
                }
            )
            for name, predictions in candidate_predictions.items():
                candidate = np.clip(
                    predictions[rows, :, target_index].astype(np.float64, copy=False)
                    * candidate_scales[name],
                    -clip,
                    clip,
                )
                difference = candidate - anchor_values[:, :, target_index]
                disagreement = float(np.sum(weights * difference**2, dtype=np.float64) / weight_sum)
                diagnostics.append(
                    {
                        "chain_id": int(chain_id),
                        "outer_fold": int(outer_fold),
                        "target": int(target_index),
                        "candidate": name,
                        "disagreement": disagreement,
                        "anchor_weighted_squared_error": error,
                        "weight_mass": weight_sum,
                    }
                )
    return diagnostics


def average_ranks(values: NDArray[np.floating]) -> FloatArray:
    """Return one-indexed average ranks with stable handling of ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rho(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Compute Spearman correlation from average ranks."""
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("Spearman inputs must be equal-length vectors with at least 3 items")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs must be finite")
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(x_centered, y_centered) / denominator, -1.0, 1.0))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _stratified_spearman(
    disagreement: FloatArray,
    error: FloatArray,
    fold: IntArray,
) -> tuple[float, dict[int, float]]:
    fold_correlations: dict[int, float] = {}
    fisher_values: list[float] = []
    for fold_id in np.unique(fold):
        selected = fold == fold_id
        rho = spearman_rho(disagreement[selected], error[selected])
        fold_correlations[int(fold_id)] = rho
        fisher_values.append(float(np.arctanh(np.clip(rho, -0.999999, 0.999999))))
    return float(np.tanh(np.mean(fisher_values))), fold_correlations


def association_screen(
    diagnostics: Sequence[Mapping[str, float | int | str]],
    *,
    seed: int,
    bootstrap_replicates: int = 2_000,
    permutation_replicates: int = 5_000,
) -> list[dict[str, object]]:
    """Estimate chain-stratified association and matched-random error excess.

    Confidence intervals condition on the observed chain-level disagreement
    ranking and use the frozen outer fold as the resampling stratum.
    """
    candidate_names = sorted(
        {str(row["candidate"]) for row in diagnostics if row["candidate"] != "anchor_error"}
    )
    if not candidate_names:
        raise ValueError("diagnostics contain no candidate predictions")
    if bootstrap_replicates < 1 or permutation_replicates < 1 or seed < 0:
        raise ValueError("seed and resampling counts must be positive")
    target_ids = sorted({int(row["target"]) for row in diagnostics})
    chain_rows: dict[tuple[str, int], list[Mapping[str, float | int | str]]] = {}
    for candidate in candidate_names:
        for target in target_ids:
            key = (candidate, target)
            chain_rows[key] = [
                row
                for row in diagnostics
                if row["candidate"] == candidate and int(row["target"]) == target
            ]

    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for (candidate, target), rows in chain_rows.items():
        rows = sorted(rows, key=lambda row: int(row["chain_id"]))
        chain = np.asarray([int(row["chain_id"]) for row in rows], dtype=np.int64)
        fold = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
        disagreement = np.asarray([float(row["disagreement"]) for row in rows])
        error = np.asarray([float(row["anchor_weighted_squared_error"]) for row in rows])
        if np.unique(chain).size != chain.size or np.unique(fold).size != 3:
            raise ValueError("expected one row per chain and three outer folds")
        observed_rho, fold_rho = _stratified_spearman(disagreement, error, fold)
        fold_indices = [np.flatnonzero(fold == fold_id) for fold_id in np.unique(fold)]

        bootstrap_rho = np.empty(bootstrap_replicates, dtype=np.float64)
        observed_excess = 0.0
        top_labels: dict[int, NDArray[np.bool_]] = {}
        fold_top_counts: dict[int, int] = {}
        fold_observed_excess: dict[int, float] = {}
        for fold_id, indices in zip(np.unique(fold), fold_indices, strict=True):
            count = max(1, int(np.ceil(0.25 * indices.size)))
            top_local = _top_disagreement_indices(
                disagreement[indices], chain[indices], count
            )
            labels = np.zeros(indices.size, dtype=np.bool_)
            labels[top_local] = True
            top_labels[int(fold_id)] = labels
            fold_top_counts[int(fold_id)] = count
            fold_observed_excess[int(fold_id)] = float(
                error[indices][labels].mean() - error[indices][~labels].mean()
            )
        observed_excess = float(np.mean(list(fold_observed_excess.values())))

        bootstrap_excess = np.empty(bootstrap_replicates, dtype=np.float64)
        for replicate in range(bootstrap_replicates):
            rho_values: list[float] = []
            excess_values: list[float] = []
            for fold_id, indices in zip(np.unique(fold), fold_indices, strict=True):
                sampled_local = rng.integers(0, indices.size, size=indices.size)
                sampled = indices[sampled_local]
                rho = spearman_rho(disagreement[sampled], error[sampled])
                rho_values.append(float(np.arctanh(np.clip(rho, -0.999999, 0.999999))))
                high = indices[top_labels[int(fold_id)]]
                rest = indices[~top_labels[int(fold_id)]]
                sampled_high = rng.choice(high, size=high.size, replace=True)
                sampled_rest = rng.choice(rest, size=rest.size, replace=True)
                excess_values.append(float(error[sampled_high].mean() - error[sampled_rest].mean()))
            bootstrap_rho[replicate] = np.tanh(np.mean(rho_values))
            bootstrap_excess[replicate] = np.mean(excess_values)

        permutation_excess = np.empty(permutation_replicates, dtype=np.float64)
        for replicate in range(permutation_replicates):
            fold_excess: list[float] = []
            for fold_id, indices in zip(np.unique(fold), fold_indices, strict=True):
                count = fold_top_counts[int(fold_id)]
                top_local = rng.choice(indices.size, size=count, replace=False)
                labels = np.zeros(indices.size, dtype=np.bool_)
                labels[top_local] = True
                error_values = error[indices]
                fold_excess.append(
                    float(error_values[labels].mean() - error_values[~labels].mean())
                )
            permutation_excess[replicate] = np.mean(fold_excess)
        p_value = float(
            (1 + np.count_nonzero(permutation_excess >= observed_excess))
            / (permutation_replicates + 1)
        )
        rho_interval = np.quantile(bootstrap_rho, [0.025, 0.975]).tolist()
        excess_interval = np.quantile(bootstrap_excess, [0.025, 0.975]).tolist()
        output.append(
            {
                "candidate": candidate,
                "target": target,
                "chain_count": int(chain.size),
                "fold_spearman": {str(key): value for key, value in fold_rho.items()},
                "equal_fold_spearman": observed_rho,
                "equal_fold_spearman_ci95": rho_interval,
                "positive_folds": sum(value > 0 for value in fold_rho.values()),
                "fold_top_quartile_excess": {
                    str(key): value for key, value in fold_observed_excess.items()
                },
                "top_quartile_error_excess": observed_excess,
                "top_quartile_error_excess_ci95": excess_interval,
                "matched_random_one_sided_p": p_value,
            }
        )
    adjusted = holm_adjust([float(row["matched_random_one_sided_p"]) for row in output])
    for row, adjusted_p in zip(output, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p
        row["screen_gate_pass"] = bool(
            int(row["positive_folds"]) >= 2
            and float(row["equal_fold_spearman_ci95"][0]) > 0
            and float(row["top_quartile_error_excess_ci95"][0]) > 0
            and adjusted_p < 0.05
        )
    return output


def _top_disagreement_indices(
    disagreement: NDArray[np.float64], chain_ids: NDArray[np.int64], count: int
) -> NDArray[np.int64]:
    """Select a fixed top count; exact score ties break by ascending chain ID.

    Rows are normally already canonicalized by chain ID, but making the tie
    key explicit keeps the rule invariant to input row order.
    """
    if disagreement.ndim != 1 or chain_ids.ndim != 1:
        raise ValueError("disagreement and chain IDs must be one-dimensional")
    if disagreement.size != chain_ids.size or not 1 <= count <= disagreement.size:
        raise ValueError("top-count selection has inconsistent dimensions")
    if np.unique(chain_ids).size != chain_ids.size:
        raise ValueError("chain IDs must be unique within a fold")
    order = np.lexsort((chain_ids, disagreement))
    return order[-count:]
