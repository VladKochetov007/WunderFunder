"""FP64 sufficient-statistic objectives and exact steepest-descent paths."""

from __future__ import annotations

import numpy as np
from attrs import frozen

COEFFICIENTS = 449
RIDGE = 100.0
ITERATIONS = 128
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


@frozen
class QuadraticSystem:
    """One target's unnormalized weighted ridge quadratic."""

    hessian: np.ndarray
    rhs: np.ndarray
    label_square_sum: float
    weight_sum: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float


@frozen
class Trajectory:
    """One exact-line-search path and its compact scalar checkpoints."""

    beta: np.ndarray
    beta_star: np.ndarray
    star_beta_h_beta: float
    initial_gap: float
    initial_relative_gap: float | None
    checkpoints: tuple[dict[str, float | int | str | None], ...]
    iterations: tuple[dict[str, float | int | str | bool | None], ...]
    nonzero_steps: int
    no_op_steps: int


def build_systems(features: np.ndarray, targets: np.ndarray) -> tuple[QuadraticSystem, ...]:
    """Accumulate the two clipped-label weighted ridge quadratics in FP64."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != COEFFICIENTS or y.shape != (len(x), 2):
        raise ValueError("R14 normal equations require (N,449) features and (N,2) targets")
    if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("R14 normal equation input is empty or nonfinite")
    clipped = np.clip(y, -2.0, 2.0)
    systems: list[QuadraticSystem] = []
    for target in range(2):
        label = clipped[:, target]
        weight = np.abs(label)
        weight_sum = float(weight.sum(dtype=np.float64))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError(f"R14 target {target} has no positive fit weight")
        hessian = np.zeros((COEFFICIENTS, COEFFICIENTS), dtype=np.float64)
        rhs = np.zeros(COEFFICIENTS, dtype=np.float64)
        for start in range(0, len(x), 2_048):
            stop = min(start + 2_048, len(x))
            block = x[start:stop]
            block_weight = weight[start:stop]
            hessian += block.T @ (block * block_weight[:, None])
            rhs += block.T @ (block_weight * label[start:stop])
        hessian[np.arange(COEFFICIENTS - 1), np.arange(COEFFICIENTS - 1)] += RIDGE
        hessian = 0.5 * (hessian + hessian.T)
        square_sum = float(np.dot(weight, label * label))
        if not np.isfinite(hessian).all() or not np.isfinite(rhs).all():
            raise ValueError(f"R14 target {target} normal equations are nonfinite")
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0.0:
            raise ValueError(f"R14 target {target} Hessian is not positive definite")
        systems.append(
            QuadraticSystem(
                hessian,
                rhs,
                square_sum,
                weight_sum,
                float(eigenvalues[0]),
                float(eigenvalues[-1]),
            )
        )
    return tuple(systems)


def mapped_systems(
    systems: tuple[QuadraticSystem, ...], transform: np.ndarray
) -> tuple[QuadraticSystem, ...]:
    """Map systems by congruence, including the prescribed non-isotropic penalty."""
    s = np.asarray(transform, dtype=np.float64)
    if s.shape != (COEFFICIENTS, COEFFICIENTS) or len(systems) != 2:
        raise ValueError("R14 coordinate map/system shapes differ")
    if not np.isfinite(s).all():
        raise ValueError("R14 coordinate map is nonfinite")
    mapped: list[QuadraticSystem] = []
    for system in systems:
        hessian = s @ system.hessian @ s.T
        hessian = 0.5 * (hessian + hessian.T)
        rhs = s @ system.rhs
        if not np.isfinite(hessian).all() or not np.isfinite(rhs).all():
            raise ValueError("R14 mapped system is nonfinite")
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0.0:
            raise ValueError("R14 mapped Hessian is not positive definite")
        mapped.append(
            QuadraticSystem(
                hessian,
                rhs,
                system.label_square_sum,
                system.weight_sum,
                float(eigenvalues[0]),
                float(eigenvalues[-1]),
            )
        )
    return tuple(mapped)


def solve_optimum(system: QuadraticSystem) -> np.ndarray:
    """Solve the unique FP64 optimum of one positive-definite quadratic."""
    beta = np.linalg.solve(system.hessian, system.rhs)
    if not np.isfinite(beta).all():
        raise ValueError("R14 exact optimum is nonfinite")
    return beta


def _quadratic(system: QuadraticSystem, beta: np.ndarray) -> tuple[float, float, float]:
    beta_h_beta = float(beta @ system.hessian @ beta)
    beta_dot_b = float(beta @ system.rhs)
    risk = (beta_h_beta - 2.0 * beta_dot_b + system.label_square_sum) / system.weight_sum
    if not np.isfinite((beta_h_beta, beta_dot_b, risk)).all():
        raise ValueError("R14 checkpoint objective is nonfinite")
    return beta_h_beta, beta_dot_b, risk


def _gap(
    system: QuadraticSystem, beta: np.ndarray, beta_star: np.ndarray
) -> tuple[float, float | None]:
    delta = beta - beta_star
    delta_h_delta = float(delta @ system.hessian @ delta)
    star_beta_h_beta = float(beta_star @ system.hessian @ beta_star)
    scale = max(1.0, abs(star_beta_h_beta), abs(delta_h_delta))
    if delta_h_delta < 0.0 and delta_h_delta >= -1e-12 * scale:
        delta_h_delta = 0.0
    if delta_h_delta < 0.0 or not np.isfinite((delta_h_delta, star_beta_h_beta)).all():
        raise ValueError("R14 stable delta-H-delta diagnostic is invalid")
    if star_beta_h_beta <= 0.0:
        return delta_h_delta, None
    return delta_h_delta, delta_h_delta / star_beta_h_beta


def _checkpoint(
    iteration: int,
    system: QuadraticSystem,
    beta: np.ndarray,
    beta_star: np.ndarray,
    initial_gap: float,
    initial_relative_gap: float | None,
) -> dict[str, float | int | str | None]:
    beta_h_beta, beta_dot_b, risk = _quadratic(system, beta)
    delta_h_delta, relative_gap = _gap(system, beta, beta_star)
    star_risk = _quadratic(system, beta_star)[2]
    return {
        "iteration": iteration,
        "risk": risk,
        "star_risk": star_risk,
        "beta_h_beta": beta_h_beta,
        "beta_dot_b": beta_dot_b,
        "label_square_sum": system.label_square_sum,
        "weight_sum": system.weight_sum,
        "delta_h_delta": delta_h_delta,
        "initial_gap_energy": initial_gap,
        "initial_relative_gap": initial_relative_gap,
        "relative_gap": relative_gap,
        "gap_state": "zero_optimum_norm" if relative_gap is None else "defined",
    }


def steepest_descent(system: QuadraticSystem) -> Trajectory:
    """Run exactly 128 full-fit steepest-descent iterations with line search."""
    beta_star = solve_optimum(system)
    beta = np.zeros(COEFFICIENTS, dtype=np.float64)
    initial_gap, initial_relative_gap = _gap(system, beta, beta_star)
    star_energy = float(beta_star @ system.hessian @ beta_star)
    if not np.isclose(initial_gap, star_energy, rtol=1e-12, atol=1e-12):
        raise ValueError("R14 initial delta-H-delta must equal optimum H-energy")
    checkpoints = [_checkpoint(0, system, beta, beta_star, initial_gap, initial_relative_gap)]
    iteration_records: list[dict[str, float | int | str | bool | None]] = []
    nonzero_steps = 0
    no_op_steps = 0

    for iteration in range(1, ITERATIONS + 1):
        gradient = 2.0 * (system.hessian @ beta - system.rhs)
        gradient_norm = float(np.linalg.norm(gradient))
        finite = bool(np.isfinite(gradient).all() and np.isfinite(gradient_norm))
        if not finite:
            raise ValueError(f"R14 iteration {iteration} has a nonfinite gradient")
        gradient_squared_norm = float(gradient @ gradient)
        if gradient_norm == 0.0:
            eta, step_status = 0.0, "zero_gradient_noop"
            curvature = 0.0
        else:
            curvature = float(2.0 * (gradient @ (system.hessian @ gradient)))
            if not np.isfinite(curvature) or curvature <= 0.0:
                raise ValueError(f"R14 iteration {iteration} has invalid line curvature")
            eta = float((gradient @ gradient) / curvature)
            candidate = beta - eta * gradient
            if not np.isfinite(eta) or not np.isfinite(candidate).all():
                raise ValueError(f"R14 iteration {iteration} has a nonfinite step")
            if np.array_equal(candidate, beta):
                step_status = "rounded_step_noop"
            else:
                beta = candidate
                step_status = "nonzero_step"
        if step_status == "nonzero_step":
            nonzero_steps += 1
        else:
            no_op_steps += 1
        _, _, current_risk = _quadratic(system, beta)
        if not iteration_records:
            previous_risk = checkpoints[0]["risk"]
        else:
            previous_risk = iteration_records[-1]["risk"]
        tolerance = 1e-11 * max(1.0, abs(float(previous_risk)))
        if current_risk > float(previous_risk) + tolerance:
            raise ValueError(f"R14 iteration {iteration} increased objective beyond roundoff")
        _, relative_gap = _gap(system, beta, beta_star)
        iteration_records.append(
            {
                "iteration": iteration,
                "eta": eta,
                "gradient_norm": gradient_norm,
                "gradient_squared_norm": gradient_squared_norm,
                "line_curvature": curvature,
                "objective_scale": "unnormalized_weighted_ridge_quadratic",
                "risk": current_risk,
                "relative_gap": relative_gap,
                "finite": finite,
                "step_status": step_status,
            }
        )
        if iteration in CHECKPOINTS[1:]:
            checkpoints.append(
                _checkpoint(
                    iteration,
                    system,
                    beta,
                    beta_star,
                    initial_gap,
                    initial_relative_gap,
                )
            )

    return Trajectory(
        beta=beta,
        beta_star=beta_star,
        star_beta_h_beta=float(beta_star @ system.hessian @ beta_star),
        initial_gap=initial_gap,
        initial_relative_gap=initial_relative_gap,
        checkpoints=tuple(checkpoints),
        iterations=tuple(iteration_records),
        nonzero_steps=nonzero_steps,
        no_op_steps=no_op_steps,
    )
