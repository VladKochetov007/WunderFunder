from __future__ import annotations

import numpy as np


def merge_weighted_batch(
    total_weight: float,
    mean_features: np.ndarray,
    mean_target: float,
    centered_feature_sums: np.ndarray,
    centered_cross_sums: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray]:
    """Merge a batch into stable weighted centered moments.

    The returned second moments are unnormalized sums around the combined
    weighted means, avoiding cancellation from E[x*x] - E[x]**2.
    """
    batch_weight = float(weights.sum())
    if batch_weight <= 0.0:
        return total_weight, mean_features, mean_target, centered_feature_sums, centered_cross_sums

    batch_mean_features = weights @ features / batch_weight
    batch_mean_target = float(weights @ targets / batch_weight)
    batch_centered_features = features - batch_mean_features
    batch_centered_targets = targets - batch_mean_target
    batch_feature_sums = batch_centered_features.T @ (batch_centered_features * weights[:, None])
    batch_cross_sums = batch_centered_features.T @ (weights * batch_centered_targets)

    if total_weight <= 0.0:
        return (
            batch_weight,
            batch_mean_features,
            batch_mean_target,
            batch_feature_sums,
            batch_cross_sums,
        )

    combined_weight = total_weight + batch_weight
    feature_delta = batch_mean_features - mean_features
    target_delta = batch_mean_target - mean_target
    correction_weight = total_weight * batch_weight / combined_weight
    centered_feature_sums += batch_feature_sums + correction_weight * np.outer(
        feature_delta, feature_delta
    )
    centered_cross_sums += batch_cross_sums + correction_weight * feature_delta * target_delta
    mean_features += feature_delta * (batch_weight / combined_weight)
    mean_target += target_delta * (batch_weight / combined_weight)
    return (
        combined_weight,
        mean_features,
        mean_target,
        centered_feature_sums,
        centered_cross_sums,
    )
