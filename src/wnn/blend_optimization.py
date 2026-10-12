"""Bounded numerical primitives for shared-weight, two-target blends.

The module has no file, prediction-bank, model, or scoring-workflow access. Callers
must bind a small, source-qualified calibration array before passing it here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wnn.gmetric import global_wp

FloatArray = NDArray[np.float64]
_EPS = 1e-8
_CLIP = 2.0
_SLSQP_MAX_ITERATIONS = 50


@dataclass(frozen=True)
class WeightedPredictionMoments:
    """Weighted centered moments for one unclipped target and prediction pool."""

    covariance: FloatArray
    target_covariance: FloatArray
    target_variance: float
    total_weight: float


@dataclass(frozen=True)
class SolverAttempt:
    """One smooth-proxy SLSQP trajectory and official-scored endpoint."""

    start_name: str
    success: bool
    status: int
    iterations: int
    start_weights: FloatArray
    start_surrogate_score: float
    start_official_score: float
    surrogate_score: float
    official_score: float
    weights: FloatArray
    message: str


@dataclass(frozen=True)
class SharedBlendFit:
    """Best officially scored feasible weights from three proxy solver starts."""

    weights: FloatArray
    unshrunk_weights: FloatArray
    official_score: float
    unshrunk_official_score: float
    surrogate_score: float
    unshrunk_surrogate_score: float
    selected_candidate: str
    shrinkage: float
    attempts: tuple[SolverAttempt, ...]


@dataclass(frozen=True)
class _PreparedProblem:
    targets: FloatArray
    scorer_targets: NDArray[np.float32]
    predictions: NDArray[np.float32]
    weights: FloatArray


def weighted_prediction_moments(
    target: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> WeightedPredictionMoments:
    """Return C, c, and target variance for an unclipped weighted Pearson target.

    Args:
        target: Already-selected one-dimensional target values.
        predictions: Already-selected ``(rows, models)`` prediction values.
        weights: Fixed nonnegative row weights, normally ``abs(clip(y, -2, 2))``.

    Returns:
        Population covariance matrix, prediction/target covariance vector,
        target variance, and total weight. All values use float64 accumulation.

    Raises:
        ValueError: Shapes differ, values are nonfinite, or a weight is negative.
    """
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(predictions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size or w.shape != y.shape:
        raise ValueError("expected target/weights (rows,) and predictions (rows, models)")
    if x.shape[1] < 1:
        raise ValueError("at least one prediction column is required")
    if not np.isfinite(y).all() or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("moments require finite selected values")
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative")

    total = float(w.sum(dtype=np.float64))
    model_count = x.shape[1]
    if total < _EPS:
        return WeightedPredictionMoments(
            np.zeros((model_count, model_count), dtype=np.float64),
            np.zeros(model_count, dtype=np.float64),
            0.0,
            total,
        )

    positive = w > 0.0
    y_pos, x_pos, w_pos = y[positive], x[positive], w[positive]
    y_shift = y_pos - y_pos[0]
    x_shift = x_pos - x_pos[0]
    mean_y = float(np.dot(w_pos, y_shift) / total)
    mean_x = np.sum(x_shift * w_pos[:, None], axis=0, dtype=np.float64) / total
    centered_y = y_shift - mean_y
    centered_x = x_shift - mean_x
    covariance = (centered_x * w_pos[:, None]).T @ centered_x / total
    covariance = (covariance + covariance.T) * 0.5
    target_covariance = centered_x.T @ (w_pos * centered_y) / total
    target_variance = float(np.dot(w_pos, centered_y * centered_y) / total)
    return WeightedPredictionMoments(covariance, target_covariance, target_variance, total)


def unclipped_correlation_and_gradient(
    coefficients: np.ndarray, moments: WeightedPredictionMoments
) -> tuple[float, FloatArray]:
    """Evaluate one-target unclipped weighted correlation and its exact gradient."""
    a = np.asarray(coefficients, dtype=np.float64)
    model_count = moments.target_covariance.size
    if a.shape != (model_count,) or not np.isfinite(a).all():
        raise ValueError("coefficients must be a finite vector matching the moment pool")
    zero = np.zeros(model_count, dtype=np.float64)
    if moments.total_weight < _EPS or moments.target_variance <= _EPS**2:
        return 0.0, zero
    prediction_variance = float(a @ moments.covariance @ a)
    if prediction_variance <= _EPS**2:
        return 0.0, zero
    denominator = float(np.sqrt(moments.target_variance * prediction_variance))
    numerator = float(moments.target_covariance @ a)
    correlation = float(np.clip(numerator / denominator, -1.0, 1.0))
    gradient = (
        moments.target_covariance / denominator
        - correlation * (moments.covariance @ a) / prediction_variance
    )
    return correlation, np.asarray(gradient, dtype=np.float64)


def ridge_simplex_initializer(moments: WeightedPredictionMoments, ridge: float) -> FloatArray:
    """Solve ``(C + ridge I)u=c`` and normalize its nonnegative direction.

    A positive ridge is a declared regularization choice, not the exact
    single-target optimum. When the solved direction has positive entries, its
    positive part is normalized to the simplex, preserving scale invariance.
    If every entry is nonpositive, Euclidean projection is a finite fallback.
    Use ``ridge=0`` only when the covariance is invertible.
    """
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    model_count = moments.target_covariance.size
    if model_count < 1 or moments.covariance.shape != (model_count, model_count):
        raise ValueError("invalid covariance moment shape")
    regularized = moments.covariance + float(ridge) * np.eye(model_count)
    try:
        direction = np.linalg.solve(regularized, moments.target_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("singular covariance; supply a positive ridge") from error
    if not np.isfinite(direction).all():
        raise ValueError("ridge system produced nonfinite coefficients")
    positive_direction = np.maximum(direction, 0.0)
    positive_total = float(positive_direction.sum(dtype=np.float64))
    if positive_total > 0.0:
        return positive_direction / positive_total
    return _project_simplex(direction)


def clip2_wp_surrogate_and_gradient(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[float, FloatArray]:
    """Evaluate the continuous clip-2 WP surrogate and its analytic Jacobian.

    Candidate arrays are float32, but the coefficient mixture and clipping are
    evaluated in float64 to provide a useful real-valued optimization surrogate.
    Targets use the official ``abs(clip(target, -2, 2))`` weights, the selected
    mask is fixed, and the value is the mean of pooled target correlations. The
    analytic Jacobian is for this continuous surrogate; it is not a derivative of
    the official scorer's float32 quantization. At exact clip boundaries the
    chosen surrogate derivative is zero.
    """
    problem = _prepare_problem(targets, predictions, mask)
    return _surrogate_and_gradient(problem, coefficients)


def official_clip2_wp(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    """Score a float32-cast candidate mixture through the unchanged official metric.

    Mixture accumulation is float64 and then cast once to float32, matching the
    canonical candidate synthesis used by this module. The unchanged
    ``wnn.gmetric.global_wp`` applies its float32 conversion, clipping, fixed mask,
    magnitude weights, and pooled two-target score. Re-score a delivered package
    using its exact serving arithmetic before making endpoint claims.
    """
    problem = _prepare_problem(targets, predictions, mask)
    return _official_score(problem, coefficients)


def analytic_shared_initializer(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    ridge: float,
) -> FloatArray:
    """Average two feasible single-target ridge starts for shared-weight SLSQP.

    Each target-specific start solves its own unclipped, regularized covariance
    system. Their simplex average is only a deterministic initialization for the
    shared clipped objective; it is not a closed-form solution to that objective.
    """
    problem = _prepare_problem(targets, predictions, mask)
    starts = []
    for target_index in range(2):
        moments = weighted_prediction_moments(
            problem.targets[:, target_index],
            problem.predictions[:, :, target_index],
            problem.weights[:, target_index],
        )
        starts.append(ridge_simplex_initializer(moments, ridge))
    return _project_simplex(np.mean(np.stack(starts), axis=0))


def fit_shared_clip2_weights(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    current_weights: np.ndarray,
    *,
    ridge: float,
    shrink_to_equal: float = 0.0,
) -> SharedBlendFit:
    """Fit proxy gradients, then rank feasible candidates by official float32 WP.

    SLSQP follows the continuous float64 clip-2 surrogate with analytic Jacobian.
    The three starts and returned SLSQP endpoints are ranked by
    ``wnn.gmetric.global_wp`` after canonical float32 mixture synthesis. This
    bounded local optimization does not claim a global optimum. The fixed
    optional shrinkage is applied after solver selection and is not searched.

    Args:
        targets: Bound target array with shape ``(rows, 2)``.
        predictions: Bound candidate array with shape ``(rows, models, 2)``.
        mask: Fixed selected-row mask with shape ``(rows,)``.
        current_weights: Existing feasible shared simplex weights.
        ridge: Fixed regularization used only for the analytic start.
        shrink_to_equal: Predeclared shrink amount in ``[0, 1]``.

    Returns:
        Best officially scored feasible result across the three starts and their
        SLSQP endpoints, with surrogate/official diagnostics and fixed shrinkage.

    Raises:
        ValueError: Inputs, current weights, ridge, or shrinkage are invalid.
    """
    if not np.isfinite(shrink_to_equal) or not 0.0 <= shrink_to_equal <= 1.0:
        raise ValueError("shrink_to_equal must be in [0, 1]")
    problem = _prepare_problem(targets, predictions, mask)
    model_count = problem.predictions.shape[1]
    current = _validated_simplex(current_weights, model_count)
    equal = np.full(model_count, 1.0 / model_count, dtype=np.float64)
    analytic = _analytic_shared_initializer(problem, ridge)
    starts = (("equal", equal), ("current", current), ("analytic", analytic))

    # SciPy is a research-only dependency; importing this module does not require it.
    from scipy.optimize import minimize

    candidates: list[tuple[str, FloatArray, float, float]] = []
    attempts: list[SolverAttempt] = []
    constraint = {
        "type": "eq",
        "fun": lambda value: float(value.sum() - 1.0),
        "jac": lambda value: np.ones_like(value, dtype=np.float64),
    }
    for name, start in starts:
        initial_surrogate_score, _ = _surrogate_and_gradient(problem, start)
        initial_official_score = _official_score(problem, start)
        candidates.append(
            (f"{name}:start", start.copy(), initial_official_score, initial_surrogate_score)
        )

        def objective(value: np.ndarray) -> tuple[float, FloatArray]:
            score, gradient = _surrogate_and_gradient(problem, value)
            return -score, -gradient

        result = minimize(
            objective,
            start.copy(),
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * model_count,
            constraints=(constraint,),
            options={"maxiter": _SLSQP_MAX_ITERATIONS, "ftol": 1e-12},
        )
        result_weights = _project_simplex(np.asarray(result.x, dtype=np.float64))
        result_surrogate_score, _ = _surrogate_and_gradient(problem, result_weights)
        result_official_score = _official_score(problem, result_weights)
        attempts.append(
            SolverAttempt(
                name,
                bool(result.success),
                int(result.status),
                int(result.nit),
                start.copy(),
                initial_surrogate_score,
                initial_official_score,
                result_surrogate_score,
                result_official_score,
                result_weights.copy(),
                str(result.message),
            )
        )
        candidates.append(
            (f"{name}:slsqp", result_weights, result_official_score, result_surrogate_score)
        )

    selected_name, unshrunk, unshrunk_official_score, unshrunk_surrogate_score = max(
        candidates, key=lambda item: item[2]
    )
    final = (1.0 - shrink_to_equal) * unshrunk + shrink_to_equal * equal
    final = _project_simplex(final)
    final_surrogate_score, _ = _surrogate_and_gradient(problem, final)
    final_official_score = _official_score(problem, final)
    return SharedBlendFit(
        final,
        unshrunk.copy(),
        final_official_score,
        unshrunk_official_score,
        final_surrogate_score,
        unshrunk_surrogate_score,
        selected_name,
        float(shrink_to_equal),
        tuple(attempts),
    )


def _prepare_problem(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> _PreparedProblem:
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or p.ndim != 3 or p.shape[0] != y.shape[0]:
        raise ValueError("expected targets (rows, 2) and predictions (rows, models, 2)")
    if p.shape[1] < 1 or p.shape[2] != 2 or selected.shape != (y.shape[0],):
        raise ValueError("prediction/model or selected-row mask shape is invalid")
    if not selected.any():
        raise ValueError("no selected rows")
    y_selected, p_selected = y[selected], p[selected]
    if not np.isfinite(y_selected).all() or not np.isfinite(p_selected).all():
        raise ValueError("nonfinite selected target or prediction")
    clipped_targets = np.clip(y_selected, -_CLIP, _CLIP).astype(np.float64)
    return _PreparedProblem(
        clipped_targets,
        np.ascontiguousarray(y_selected),
        np.ascontiguousarray(p_selected),
        np.abs(clipped_targets),
    )


def _analytic_shared_initializer(problem: _PreparedProblem, ridge: float) -> FloatArray:
    starts = []
    for target_index in range(2):
        moments = weighted_prediction_moments(
            problem.targets[:, target_index],
            problem.predictions[:, :, target_index],
            problem.weights[:, target_index],
        )
        starts.append(ridge_simplex_initializer(moments, ridge))
    return _project_simplex(np.mean(np.stack(starts), axis=0))


def _surrogate_and_gradient(
    problem: _PreparedProblem, coefficients: np.ndarray
) -> tuple[float, FloatArray]:
    a = np.asarray(coefficients, dtype=np.float64)
    model_count = problem.predictions.shape[1]
    if a.shape != (model_count,) or not np.isfinite(a).all():
        raise ValueError("coefficients must be a finite vector matching the prediction pool")
    raw_mix = np.einsum("nmk,m->nk", problem.predictions, a, dtype=np.float64, optimize=False)
    clipped_mix = np.clip(raw_mix, -_CLIP, _CLIP)
    active = (raw_mix > -_CLIP) & (raw_mix < _CLIP)
    target_scores = []
    target_gradients = []
    for target_index in range(2):
        x = problem.predictions[:, :, target_index]
        active_x = np.where(active[:, target_index, None], x, 0.0)
        score, gradient = _weighted_correlation_gradient(
            problem.targets[:, target_index],
            clipped_mix[:, target_index],
            problem.weights[:, target_index],
            active_x,
        )
        target_scores.append(score)
        target_gradients.append(gradient)
    return float(np.mean(target_scores)), np.mean(target_gradients, axis=0)


def _official_score(problem: _PreparedProblem, coefficients: np.ndarray) -> float:
    a = np.asarray(coefficients, dtype=np.float64)
    model_count = problem.predictions.shape[1]
    if a.shape != (model_count,) or not np.isfinite(a).all():
        raise ValueError("coefficients must be a finite vector matching the prediction pool")
    raw_mixture = np.einsum("nmk,m->nk", problem.predictions, a, dtype=np.float64, optimize=False)
    candidate_predictions = raw_mixture.astype(np.float32)
    selected_mask = np.ones(problem.scorer_targets.shape[0], dtype=bool)
    result = global_wp(problem.scorer_targets, candidate_predictions, selected_mask)
    return float(result["wp"])


def _weighted_correlation_gradient(
    target: np.ndarray, prediction: np.ndarray, weights: np.ndarray, derivative: np.ndarray
) -> tuple[float, FloatArray]:
    model_count = derivative.shape[1]
    zero = np.zeros(model_count, dtype=np.float64)
    total = float(weights.sum(dtype=np.float64))
    if total < _EPS:
        return 0.0, zero

    y_shift = target - target[0]
    p_shift = prediction - prediction[0]
    mean_y = float(np.dot(weights, y_shift) / total)
    mean_p = float(np.dot(weights, p_shift) / total)
    centered_y = y_shift - mean_y
    centered_p = p_shift - mean_p
    target_variance = float(np.dot(weights, centered_y * centered_y) / total)
    prediction_variance = float(np.dot(weights, centered_p * centered_p) / total)
    if target_variance <= _EPS**2 or prediction_variance <= _EPS**2:
        return 0.0, zero

    covariance = float(np.dot(weights, centered_y * centered_p) / total)
    denominator = float(np.sqrt(target_variance * prediction_variance))
    correlation = float(np.clip(covariance / denominator, -1.0, 1.0))
    covariance_y_x = (
        np.einsum("n,nm->m", weights * centered_y, derivative, dtype=np.float64, optimize=False)
        / total
    )
    covariance_p_x = (
        np.einsum("n,nm->m", weights * centered_p, derivative, dtype=np.float64, optimize=False)
        / total
    )
    gradient = covariance_y_x / denominator - correlation * covariance_p_x / prediction_variance
    return correlation, np.asarray(gradient, dtype=np.float64)


def _validated_simplex(weights: np.ndarray, model_count: int) -> FloatArray:
    value = np.asarray(weights, dtype=np.float64)
    if (
        value.shape != (model_count,)
        or not np.isfinite(value).all()
        or np.any(value < -1e-9)
        or not np.isclose(value.sum(), 1.0, rtol=0.0, atol=1e-7)
    ):
        raise ValueError("current_weights must be a finite nonnegative simplex vector")
    return _project_simplex(value)


def _project_simplex(values: np.ndarray) -> FloatArray:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or value.size < 1 or not np.isfinite(value).all():
        raise ValueError("simplex projection requires a nonempty finite vector")
    ordered = np.sort(value)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    rank = np.arange(1, value.size + 1, dtype=np.float64)
    positive = np.flatnonzero(ordered - cumulative / rank > 0.0)
    if positive.size == 0:
        return np.full(value.size, 1.0 / value.size, dtype=np.float64)
    threshold = cumulative[positive[-1]] / (positive[-1] + 1)
    projected = np.maximum(value - threshold, 0.0)
    total = float(projected.sum(dtype=np.float64))
    if total <= 0.0:
        return np.full(value.size, 1.0 / value.size, dtype=np.float64)
    return projected / total
