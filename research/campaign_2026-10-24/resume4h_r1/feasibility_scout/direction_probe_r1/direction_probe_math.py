"""Fixed-prediction clipped-score derivatives and component moments."""

from __future__ import annotations

from collections import Counter
from typing import Any  # Moment arrays have runtime-defined endpoint/variable axes.

import numpy as np

EPS = 1e-8
RANK_RELATIVE_CUTOFF = 1e-10
NUMERICAL_ZERO_VARIANCE = 1e-16
CALIBRATION_NAMES = ("plus_one", "minus_one", "plus_parent", "minus_parent")
ALPHA_ZERO_NAMES = (
    "clipped_label",
    "clipped_base",
    "candidate_right_tangent",
    *CALIBRATION_NAMES,
    "common_A_right_tangent",
)
ALPHA_POINT_ONE_NAMES = ("clipped_label", "clipped_base", "candidate_right_tangent")


def right_clip_derivative(base: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return the right derivative of clipping at the supplied base and direction."""
    base64 = np.asarray(base, dtype=np.float64)
    direction64 = np.asarray(direction, dtype=np.float64)
    if base64.shape != direction64.shape:
        raise ValueError("clip base and direction must have identical shapes")
    result = np.zeros_like(direction64)
    interior = (base64 > -2.0) & (base64 < 2.0)
    result[interior] = direction64[interior]
    upper = base64 == 2.0
    lower = base64 == -2.0
    result[upper] = np.minimum(direction64[upper], 0.0)
    result[lower] = np.maximum(direction64[lower], 0.0)
    return result


def endpoint_statistics(
    targets: np.ndarray,
    official_mask: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    common_a: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw additive and stable centered moments for one sequence."""
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(parent, dtype=np.float64)
    q = np.asarray(candidate, dtype=np.float64)
    a = np.asarray(common_a, dtype=np.float64)
    mask = np.asarray(official_mask, dtype=bool)
    if y.shape != p.shape or q.shape != p.shape or a.shape != p.shape:
        raise ValueError("target and prediction sequence shapes differ")
    if y.ndim != 2 or y.shape[1] != 2 or mask.shape != y.shape[:1]:
        raise ValueError("expected aligned (rows, 2) values and a row mask")
    label = np.clip(y, -2.0, 2.0)
    correction = q - p
    base = p + float(alpha) * correction
    clipped_base = np.clip(base, -2.0, 2.0)
    tangent = right_clip_derivative(base, correction)
    if alpha == 0.0:
        controls = (
            right_clip_derivative(p, np.ones_like(p)),
            right_clip_derivative(p, -np.ones_like(p)),
            right_clip_derivative(p, p),
            right_clip_derivative(p, -p),
        )
        a_tangent = right_clip_derivative(p, a - p)
        variables = np.stack((label, clipped_base, tangent, *controls, a_tangent), axis=-1)
    elif alpha == 0.1:
        variables = np.stack((label, clipped_base, tangent), axis=-1)
    else:
        raise ValueError("the frozen probe uses only alpha 0 and alpha 0.1")

    dimensions = variables.shape[-1]
    packed = np.zeros((2, 1 + dimensions + dimensions * dimensions), dtype=np.float64)
    centered_packed = np.zeros_like(packed)
    for target in range(2):
        weights = np.abs(label[:, target])
        selected = mask & (weights > 0.0)
        x = variables[:, target, :][selected]
        w = weights[selected]
        if not np.isfinite(x).all() or not np.isfinite(w).all():
            raise ValueError("nonfinite selected score or tangent input")
        total = float(w.sum())
        first = w @ x
        packed[target] = np.concatenate(([total], first, (x.T @ (w[:, None] * x)).ravel()))
        if total > 0.0:
            mean = first / total
            centered = x - mean
            cross = centered.T @ (w[:, None] * centered)
            centered_packed[target] = np.concatenate(([total], mean, cross.ravel()))
    return packed, centered_packed


def clipping_diagnostics(
    targets: np.ndarray, official_mask: np.ndarray, parent: np.ndarray, endpoint: np.ndarray
) -> dict[str, Any]:
    """Count clipping-state transitions and target-weight mass for both targets."""
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(parent, dtype=np.float64)
    q = np.asarray(endpoint, dtype=np.float64)
    mask = np.asarray(official_mask, dtype=bool)
    names = ("below", "lower_boundary", "interior", "upper_boundary", "above")
    result: dict[str, Any] = {"state_order": list(names), "per_target": []}

    def states(values: np.ndarray) -> np.ndarray:
        state = np.zeros(len(values), dtype=np.int8)
        state[values < -2.0] = -2
        state[values == -2.0] = -1
        state[values == 2.0] = 1
        state[values > 2.0] = 2
        return state

    for target in range(2):
        label = np.clip(y[:, target], -2.0, 2.0)
        if not (
            np.isfinite(label[mask]).all()
            and np.isfinite(p[mask, target]).all()
            and np.isfinite(q[mask, target]).all()
        ):
            raise ValueError("nonfinite value in an officially scored clipping row")
        weights = np.abs(label[mask])
        old, new = states(p[mask, target]), states(q[mask, target])
        old_index, new_index = old + 2, new + 2
        codes = old_index * 5 + new_index
        counts = np.bincount(codes, minlength=25).reshape(5, 5)
        masses = np.bincount(codes, weights=weights, minlength=25).reshape(5, 5)

        def state_summary(
            values: np.ndarray, row_weights: np.ndarray
        ) -> dict[str, dict[str, float | int]]:
            summary: dict[str, dict[str, float | int]] = {}
            for index, name in enumerate(names):
                selected = values == index - 2
                summary[name] = {
                    "count": int(selected.sum()),
                    "official_weight_mass": float(row_weights[selected].sum()),
                }
            return summary

        saturated_old = np.isin(old, (-2, 2))
        saturated_new = np.isin(new, (-2, 2))
        entered = ~saturated_old & saturated_new
        exited = saturated_old & ~saturated_new
        result["per_target"].append(
            {
                "selected_rows": int(mask.sum()),
                "parent_states": state_summary(old, weights),
                "endpoint_states": state_summary(new, weights),
                "strict_saturation_entry": {
                    "count": int(entered.sum()),
                    "official_weight_mass": float(weights[entered].sum()),
                },
                "strict_saturation_exit": {
                    "count": int(exited.sum()),
                    "official_weight_mass": float(weights[exited].sum()),
                },
                "transition_counts": {
                    before: {after: int(counts[i, j]) for j, after in enumerate(names)}
                    for i, before in enumerate(names)
                },
                "transition_weight_mass": {
                    before: {after: float(masses[i, j]) for j, after in enumerate(names)}
                    for i, before in enumerate(names)
                },
            }
        )
    return result


def merge_clipping(total: dict[str, Any] | None, addition: dict[str, Any]) -> dict[str, Any]:
    """Add one sequence's clipping counts and masses to a cumulative record."""
    if total is None:
        return addition
    for target in range(2):
        dst, src = total["per_target"][target], addition["per_target"][target]
        dst["selected_rows"] += src["selected_rows"]
        for field in ("parent_states", "endpoint_states"):
            for state, values in dst[field].items():
                values["count"] += src[field][state]["count"]
                values["official_weight_mass"] += src[field][state]["official_weight_mass"]
        for field in ("strict_saturation_entry", "strict_saturation_exit"):
            dst[field]["count"] += src[field]["count"]
            dst[field]["official_weight_mass"] += src[field]["official_weight_mass"]
        for before, after_counts in dst["transition_counts"].items():
            for after in after_counts:
                after_counts[after] += src["transition_counts"][before][after]
                dst["transition_weight_mass"][before][after] += src["transition_weight_mass"][
                    before
                ][after]
    return total


def unpack_additive_moments(
    packed: np.ndarray, multiplicity: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Combine component tables into two covariance matrices and target weights."""
    table = np.asarray(packed, dtype=np.float64)
    if table.ndim != 3 or table.shape[1] != 2:
        raise ValueError("component table must have shape (components, 2, packed_stats)")
    counts = (
        np.ones(len(table), dtype=np.float64)
        if multiplicity is None
        else np.asarray(multiplicity, dtype=np.float64)
    )
    if counts.shape != (len(table),) or np.any(counts < 0.0):
        raise ValueError("component multiplicities must be aligned and nonnegative")
    dimensions = int(round(np.sqrt(table.shape[2] - 1)))
    if 1 + dimensions + dimensions * dimensions != table.shape[2]:
        raise ValueError("component table has a malformed packed dimension")
    aggregate = np.tensordot(counts, table, axes=(0, 0))
    covariance = np.zeros((2, dimensions, dimensions), dtype=np.float64)
    total_weights = aggregate[:, 0].copy()
    for target, total in enumerate(total_weights):
        if total <= 0.0:
            continue
        first = aggregate[target, 1 : 1 + dimensions]
        second = aggregate[target, 1 + dimensions :].reshape(dimensions, dimensions)
        mean = first / total
        centered = second / total - np.outer(mean, mean)
        covariance[target] = 0.5 * (centered + centered.T)
    return covariance, total_weights


def unpack_centered_moments(
    packed: np.ndarray, multiplicity: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Merge stable per-component centered moments and return covariance matrices."""
    table = np.asarray(packed, dtype=np.float64)
    if table.ndim != 3 or table.shape[1] != 2:
        raise ValueError("component table must have shape (components, 2, packed_stats)")
    counts = (
        np.ones(len(table), dtype=np.float64)
        if multiplicity is None
        else np.asarray(multiplicity, dtype=np.float64)
    )
    if counts.shape != (len(table),) or np.any(counts < 0.0):
        raise ValueError("component multiplicities must be aligned and nonnegative")
    dimensions = int(round(np.sqrt(table.shape[2] - 1)))
    if 1 + dimensions + dimensions * dimensions != table.shape[2]:
        raise ValueError("component table has a malformed packed dimension")
    total_weights = np.zeros(2, dtype=np.float64)
    means = np.zeros((2, dimensions), dtype=np.float64)
    centered_sums = np.zeros((2, dimensions, dimensions), dtype=np.float64)
    for component, count in zip(table, counts, strict=True):
        if count == 0.0:
            continue
        for target in range(2):
            source_weight = float(component[target, 0]) * count
            if source_weight <= 0.0:
                continue
            source_mean = component[target, 1 : 1 + dimensions]
            source_cross = (
                component[target, 1 + dimensions :].reshape(dimensions, dimensions) * count
            )
            current_weight = total_weights[target]
            if current_weight == 0.0:
                total_weights[target] = source_weight
                means[target] = source_mean
                centered_sums[target] = source_cross
                continue
            merged_weight = current_weight + source_weight
            delta = source_mean - means[target]
            centered_sums[target] += source_cross + np.outer(delta, delta) * (
                current_weight * source_weight / merged_weight
            )
            means[target] += delta * (source_weight / merged_weight)
            total_weights[target] = merged_weight
    covariance = np.zeros_like(centered_sums)
    for target, total in enumerate(total_weights):
        if total > 0.0:
            covariance[target] = centered_sums[target] / total
    return covariance, total_weights


def merge_centered_sequence(total: np.ndarray, addition: np.ndarray) -> np.ndarray:
    """Merge one sequence's stable centered moments into a component table."""
    left = np.asarray(total, dtype=np.float64)
    right = np.asarray(addition, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 2:
        raise ValueError("centered sequence moments must be aligned two-target tables")
    dimensions = int(round(np.sqrt(left.shape[1] - 1)))
    if 1 + dimensions + dimensions * dimensions != left.shape[1]:
        raise ValueError("centered sequence moments have a malformed packed dimension")
    merged = left.copy()
    for target in range(2):
        weight_left, weight_right = float(left[target, 0]), float(right[target, 0])
        if weight_right <= 0.0:
            continue
        if weight_left <= 0.0:
            merged[target] = right[target]
            continue
        total_weight = weight_left + weight_right
        mean_left = left[target, 1 : 1 + dimensions]
        mean_right = right[target, 1 : 1 + dimensions]
        cross_left = left[target, 1 + dimensions :].reshape(dimensions, dimensions)
        cross_right = right[target, 1 + dimensions :].reshape(dimensions, dimensions)
        delta = mean_right - mean_left
        mean = mean_left + delta * (weight_right / total_weight)
        cross = (
            cross_left
            + cross_right
            + np.outer(delta, delta) * (weight_left * weight_right / total_weight)
        )
        merged[target] = np.concatenate(([total_weight], mean, cross.ravel()))
    return merged


def _variance(covariance: np.ndarray, index: int) -> float:
    """Clamp roundoff-only negative variance and reject material moment failure."""
    value = float(covariance[index, index])
    if value < -1e-12 * max(1.0, float(np.max(np.abs(np.diag(covariance))))):
        raise ValueError("materially negative variance in additive moment table")
    return max(value, 0.0)


def score_derivative(
    covariance: np.ndarray, total_weight: float, direction_index: int
) -> dict[str, Any]:
    """Apply the official score's one-sided derivative, including its EPS gates."""
    var_y, var_prediction = _variance(covariance, 0), _variance(covariance, 1)
    sd_y, sd_prediction = np.sqrt(var_y), np.sqrt(var_prediction)
    gate = {
        "total_weight": float(total_weight),
        "target_sd": float(sd_y),
        "prediction_sd": float(sd_prediction),
        "epsilon": EPS,
        "weight_distance_to_epsilon": float(total_weight - EPS),
        "target_sd_distance_to_epsilon": float(sd_y - EPS),
        "prediction_sd_distance_to_epsilon": float(sd_prediction - EPS),
        "weight_gate_state": "below_eps_zero_branch"
        if total_weight < EPS
        else "equality_active"
        if total_weight == EPS
        else "above_eps_active",
        "target_sd_gate_state": "equality_zero_branch"
        if sd_y == EPS
        else "below_eps_zero_branch"
        if sd_y < EPS
        else "above_eps_active",
        "prediction_sd_gate_state": "equality_zero_branch"
        if sd_prediction == EPS
        else "below_eps_zero_branch"
        if sd_prediction < EPS
        else "above_eps_active",
    }
    if total_weight < EPS:
        return {"derivative": 0.0, "status": "weight_below_eps", "gate": gate, "linear": True}
    if sd_y <= EPS:
        status = "target_sd_exact_eps_gate_boundary" if sd_y == EPS else "target_sd_below_eps"
        return {
            "derivative": 0.0,
            "status": status,
            "right_gate": "stays_closed_target_is_fixed" if sd_y == EPS else None,
            "gate": gate,
            "linear": True,
        }
    if sd_prediction < EPS:
        return {
            "derivative": 0.0,
            "status": "prediction_sd_below_eps",
            "gate": gate,
            "linear": True,
        }
    if sd_prediction == EPS:
        variance_slope = 2.0 * float(covariance[1, direction_index])
        tangent_variance = _variance(covariance, direction_index)
        opens = variance_slope > 0.0 or (variance_slope == 0.0 and tangent_variance > 0.0)
        gate["prediction_variance_right_slope"] = variance_slope
        return {
            "derivative": None if opens else 0.0,
            "status": "prediction_sd_exact_eps_gate_opens"
            if opens
            else "prediction_sd_exact_eps_gate_stays_closed",
            "right_gate": "opens_for_positive_alpha" if opens else "stays_closed_near_zero",
            "gate": gate,
            "linear": False,
        }

    sd_product = sd_y * sd_prediction
    covariance_y_prediction = float(covariance[0, 1])
    raw_rho = covariance_y_prediction / sd_product
    raw_derivative = float(
        covariance[0, direction_index] / sd_product
        - raw_rho * covariance[1, direction_index] / var_prediction
    )
    if raw_rho > 1.0:
        derivative, status = 0.0, "upper_correlation_clipped"
    elif raw_rho == 1.0:
        derivative, status = min(raw_derivative, 0.0), "upper_correlation_boundary"
    elif raw_rho < -1.0:
        derivative, status = 0.0, "lower_correlation_clipped"
    elif raw_rho == -1.0:
        derivative, status = max(raw_derivative, 0.0), "lower_correlation_boundary"
    else:
        derivative, status = raw_derivative, "regular"
    gate.update(
        {
            "unclipped_correlation": float(raw_rho),
            "official_correlation": float(np.clip(raw_rho, -1.0, 1.0)),
            "distance_to_correlation_clip": float(1.0 - abs(raw_rho)),
            "unclipped_directional_derivative": raw_derivative,
        }
    )
    return {
        "derivative": float(derivative),
        "status": status,
        "gate": gate,
        "linear": status == "regular",
    }


def _projection(
    covariance: np.ndarray,
    total_weight: float,
    basis_indices: tuple[int, ...],
    basis_names: tuple[str, ...],
) -> dict[str, Any]:
    """Project a tangent into a standardized weighted nuisance span."""
    gram = covariance[np.ix_(basis_indices, basis_indices)]
    variances = np.diag(gram)
    scale_reference = max(float(np.max(np.abs(variances))), 1.0)
    if np.any(variances < -1e-12 * scale_reference):
        raise ValueError("materially negative nuisance variance")
    active = variances > NUMERICAL_ZERO_VARIANCE
    active_indices = tuple(index for index, keep in zip(basis_indices, active, strict=True) if keep)
    active_names = tuple(name for name, keep in zip(basis_names, active, strict=True) if keep)
    zero_names = [name for name, keep in zip(basis_names, active, strict=True) if not keep]
    if not active_indices:
        return {
            "projected_derivative": 0.0,
            "projection_norm_share": 0.0,
            "rank": 0,
            "gram_eigenvalues_descending": [],
            "eigenvalue_cutoff": 0.0,
            "zero_variance_controls": zero_names,
            "standardized_coefficients": {},
            "retained_span_orthogonality_max_abs": 0.0,
        }
    active_gram = covariance[np.ix_(active_indices, active_indices)]
    scales = np.sqrt(np.diag(active_gram))
    correlation = active_gram / np.outer(scales, scales)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (correlation + correlation.T))
    largest = max(float(eigenvalues[-1]), 0.0)
    if largest > 0.0 and float(eigenvalues[0]) < -RANK_RELATIVE_CUTOFF * largest:
        raise ValueError("materially negative eigenvalue in standardized nuisance Gram")
    cutoff = largest * RANK_RELATIVE_CUTOFF
    retained = eigenvalues > cutoff if largest > 0.0 else np.zeros_like(eigenvalues, dtype=bool)
    cross = covariance[np.asarray(active_indices), 2] / scales
    standardized_coefficients = np.zeros(len(active_indices), dtype=np.float64)
    if retained.any():
        vectors = eigenvectors[:, retained]
        standardized_coefficients = vectors @ ((vectors.T @ cross) / eigenvalues[retained])
    nuisance_scores = [
        score_derivative(covariance, total_weight, index) for index in active_indices
    ]
    if any(not item["linear"] for item in nuisance_scores):
        raise ValueError(
            "projection decomposition is undefined at a score gate or correlation kink"
        )
    score_per_scale = np.asarray([item["derivative"] for item in nuisance_scores]) / scales
    projected = float(score_per_scale @ standardized_coefficients)
    projected_variance = float(standardized_coefficients @ cross)
    tangent_variance = _variance(covariance, 2)
    norm_share = (
        None
        if tangent_variance <= NUMERICAL_ZERO_VARIANCE
        else projected_variance / tangent_variance
    )
    if norm_share is not None:
        if norm_share < -1e-8 or norm_share > 1.0 + 1e-8:
            raise ValueError("projected tangent norm is outside its geometric bounds")
        norm_share = float(np.clip(norm_share, 0.0, 1.0))
    orthogonality = (
        eigenvectors[:, retained].T @ (cross - correlation @ standardized_coefficients)
        if retained.any()
        else np.zeros(0, dtype=np.float64)
    )
    coefficients = {
        name: float(value)
        for name, value in zip(active_names, standardized_coefficients, strict=True)
    }
    return {
        "projected_derivative": projected,
        "projection_norm_share": norm_share,
        "rank": int(retained.sum()),
        "gram_eigenvalues_descending": eigenvalues[::-1].tolist(),
        "eigenvalue_cutoff": cutoff,
        "rank_rule": "lambda > lambda_max * 1e-10 on standardized weighted Gram",
        "zero_variance_controls": zero_names,
        "standardized_coefficients": coefficients,
        "retained_span_orthogonality_max_abs": float(np.max(np.abs(orthogonality)))
        if len(orthogonality)
        else 0.0,
    }


def analyze_covariance(
    covariance: np.ndarray, total_weight: float, include_a: bool
) -> dict[str, Any]:
    """Report one tangent derivative and its alpha-zero geometric projections."""
    total = score_derivative(covariance, total_weight, 2)
    result: dict[str, Any] = {
        "total": total,
        "control_derivatives": {
            name: score_derivative(covariance, total_weight, index)
            for index, name in enumerate(CALIBRATION_NAMES, start=3)
        },
        "projections": {},
    }
    if total["status"] != "regular":
        result["projection_status"] = "omitted_outside_regular_linear_score_branch"
        return result
    calibration = _projection(covariance, total_weight, (3, 4, 5, 6), CALIBRATION_NAMES)
    projections: dict[str, Any] = {"calibration": calibration}
    if include_a:
        augmented = _projection(
            covariance,
            total_weight,
            (3, 4, 5, 6, 7),
            (*CALIBRATION_NAMES, "common_A_right_tangent"),
        )
        if augmented["rank"] < calibration["rank"]:
            raise ValueError("adding the common A direction reduced numerical nuisance rank")
        projections["calibration_plus_A"] = augmented

    for projection in projections.values():
        indices = (3, 4, 5, 6) if projection is calibration else (3, 4, 5, 6, 7)
        active_names = [name for name in projection["standardized_coefficients"]]
        basis_names = (
            (*CALIBRATION_NAMES, "common_A_right_tangent")
            if len(indices) == 5
            else CALIBRATION_NAMES
        )
        active_indices = [
            index for index, name in zip(indices, basis_names, strict=True) if name in active_names
        ]
        scales = np.sqrt(np.diag(covariance[np.ix_(active_indices, active_indices)]))
        coeff = np.asarray([projection["standardized_coefficients"][name] for name in active_names])
        raw_coeff = coeff / scales if len(coeff) else np.zeros(0)
        cov_y_projected = (
            float(covariance[0, active_indices] @ raw_coeff) if len(raw_coeff) else 0.0
        )
        cov_c_projected = (
            float(covariance[1, active_indices] @ raw_coeff) if len(raw_coeff) else 0.0
        )
        y_variance, c_variance = _variance(covariance, 0), _variance(covariance, 1)
        sd_product = np.sqrt(y_variance * c_variance)
        rho = float(covariance[0, 1] / sd_product)
        projected_score = float(cov_y_projected / sd_product - rho * cov_c_projected / c_variance)
        total_value = float(total["derivative"])
        residual_y = float(covariance[0, 2] - cov_y_projected)
        residual_c = float(covariance[1, 2] - cov_c_projected)
        residual_score = float(residual_y / sd_product - rho * residual_c / c_variance)
        identity_error = projected_score + residual_score - total_value
        if abs(identity_error) > 1e-10:
            raise ValueError("projected plus residual derivative does not reconstruct total")
        projection["projected_derivative"] = projected_score
        projection["remaining_derivative"] = residual_score
        projection["reconstruction_error"] = identity_error
    result["projections"] = projections
    result["projection_status"] = "regular_linear_branch"
    return result


def summarize(values: np.ndarray) -> dict[str, Any]:
    """Summarize fixed component-bootstrap draws, retaining degenerate draws as missing."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": None, "sd": None, "ci95": [None, None], "p_le_0": None}
    return {
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "ci95": [float(value) for value in np.percentile(finite, (2.5, 97.5))],
        "p_le_0": float(np.mean(finite <= 0.0)),
        "valid_draws": int(len(finite)),
    }


def rank_histogram(ranks: list[int]) -> dict[str, int]:
    """Count component-bootstrap ranks in stable key order."""
    return {str(rank): count for rank, count in sorted(Counter(ranks).items())}
