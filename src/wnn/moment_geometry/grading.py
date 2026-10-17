from __future__ import annotations

import numpy as np

from wnn.gmetric import GlobalAccumulator, wp_from_sums, wp_sums


def _validate_arrays(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> None:
    """Validate the shared sequence-axis and official metric shapes."""
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError("targets must have shape (N, T, 2)")
    if predictions.shape != targets.shape:
        raise ValueError("predictions must match targets")
    if mask.shape != targets.shape[:2]:
        raise ValueError("mask must have shape (N, T)")
    if components.shape != (targets.shape[0],):
        raise ValueError("components must have shape (N,)")


def grade(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> dict[str, float | int | str]:
    """Stream the official pooled metric one complete sequence at a time.

    Args:
        targets: Evaluation labels shaped ``(N, T, 2)``.
        predictions: Candidate predictions with the same shape.
        mask: Official selected-row mask shaped ``(N, T)``.
        components: Source-qualified component ID for each sequence.

    Returns:
        The official pooled score and counts; it never selects a seed or arm.
    """
    _validate_arrays(targets, predictions, mask, components)
    accumulator = GlobalAccumulator()
    for index in range(targets.shape[0]):
        accumulator.add(targets[index], predictions[index], mask[index])
    result = accumulator.result()
    selected = np.asarray(mask, dtype=bool).any(axis=1)
    result["sequence_count"] = int(selected.sum())
    result["component_count"] = int(np.unique(components[selected]).size)
    return result


def grade_by_fold(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
    fold_ids: np.ndarray,
    folds: tuple[int, ...] = (0, 1),
) -> dict[str, object]:
    """Return pooled and fold-specific official scores with sequence-level masks."""
    if fold_ids.shape != (targets.shape[0],):
        raise ValueError("fold_ids must have one ID per sequence")
    pooled = grade(targets, predictions, mask, components)
    by_fold = {}
    for fold in folds:
        fold_mask = np.array(mask, dtype=bool, copy=True)
        fold_mask[fold_ids != fold] = False
        by_fold[str(fold)] = grade(targets, predictions, fold_mask, components)
    return {"pooled": pooled, "by_fold": by_fold}


def _component_raw_sums(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate legacy six-sum vectors by component for conditional resampling."""
    _validate_arrays(targets, predictions, mask, components)
    component_ids, inverse = np.unique(components, return_inverse=True)
    sums = np.zeros((component_ids.size, 2, 6), dtype=np.float64)
    for index, component_index in enumerate(inverse):
        for side in range(2):
            sums[component_index, side] += wp_sums(
                targets[index, :, side], predictions[index, :, side], mask[index]
            )
    return component_ids, sums


def _score_from_component_draw(sums: np.ndarray, draw: np.ndarray) -> float:
    """Compute a raw-sum pooled score for component multiplicities."""
    pooled = np.tensordot(draw, sums, axes=(0, 0))
    return float(np.mean(wp_from_sums(pooled)))


def paired_component_bootstrap(
    targets: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Estimate paired component resampling uncertainty for a fixed prediction pair.

    This is conditional on these predictions, the observed component membership,
    and the legacy additive raw sums. It does not capture seed or fitting variation;
    ``wp_from_sums`` makes it an approximate bootstrap for nearly constant data.
    """
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    component_ids, reference_sums = _component_raw_sums(
        targets, reference_predictions, mask, components
    )
    candidate_ids, candidate_sums = _component_raw_sums(
        targets, candidate_predictions, mask, components
    )
    if not np.array_equal(component_ids, candidate_ids):
        raise ValueError("reference and candidate components differ")
    generator = np.random.default_rng(seed)
    component_count = int(component_ids.size)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws = generator.integers(0, component_count, size=component_count)
        multiplicities = np.bincount(draws, minlength=component_count)
        differences[index] = _score_from_component_draw(
            candidate_sums, multiplicities
        ) - _score_from_component_draw(reference_sums, multiplicities)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "method": "paired component bootstrap; fixed predictions; approximate raw sums",
        "replicates": replicates,
        "seed": seed,
        "component_count": component_count,
        "mean_delta": float(differences.mean()),
        "ci_2p5": float(low),
        "ci_97p5": float(high),
    }
