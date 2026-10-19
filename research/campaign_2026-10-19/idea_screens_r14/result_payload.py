"""Compact scalar result construction for the fixed R14 quadratic probe."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np

from .linear_math import Trajectory
from .runtime_support import archive_arrays


def runtime_record() -> dict[str, object]:
    """Describe the actual CPU/Python/NumPy state without mutating BLAS settings."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "CPU NumPy FP64",
        "cuda_or_torch_imported": False,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _trajectory_summary(trajectory: Trajectory) -> dict[str, object]:
    """Expose scalar convergence evidence without coefficient vectors."""
    return {
        "star_beta_h_beta": trajectory.star_beta_h_beta,
        "initial_gap_energy": trajectory.initial_gap,
        "initial_relative_gap": trajectory.initial_relative_gap,
        "checkpoints": list(trajectory.checkpoints),
        "iterations": list(trajectory.iterations),
        "attempted_target_iterations": len(trajectory.iterations),
        "nonzero_target_steps": trajectory.nonzero_steps,
        "no_op_target_steps": trajectory.no_op_steps,
    }


def _mean_defined(values: list[float | None]) -> float | None:
    """Average a complete target set, preserving undefined gaps as null."""
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def archive_parameters(
    path: Path,
    mean: np.ndarray,
    scale: np.ndarray,
    gram: np.ndarray,
    whitener: np.ndarray,
    transform: np.ndarray,
    systems_a: tuple,
    systems_b: tuple,
    beta_a: np.ndarray,
    beta_b: np.ndarray,
    beta_star_a: np.ndarray,
    mapped_optima: list[np.ndarray],
) -> dict[str, object]:
    """Archive all coefficient, normalizer, coordinate, and quadratic arrays."""
    return archive_arrays(
        path,
        {
            "normalizer_mean": mean,
            "normalizer_scale": scale,
            "ema_impulse_gram": gram,
            "ema_whitener": whitener,
            "coordinate_transform_S": transform,
            "hessian_a": np.stack([item.hessian for item in systems_a]),
            "rhs_a": np.stack([item.rhs for item in systems_a]),
            "hessian_b": np.stack([item.hessian for item in systems_b]),
            "rhs_b": np.stack([item.rhs for item in systems_b]),
            "label_square_sums": np.asarray(
                [item.label_square_sum for item in systems_a], dtype=np.float64
            ),
            "weight_sums": np.asarray([item.weight_sum for item in systems_a], dtype=np.float64),
            "beta_a_128": beta_a,
            "beta_b_128": beta_b,
            "beta_common_optimum_a": beta_star_a,
            "beta_common_optimum_b": np.column_stack(mapped_optima),
        },
    )


def build_result(
    *,
    source_sha256: str,
    gram_error: float,
    hessian_errors: list[float],
    rhs_errors: list[float],
    direct_errors: list[float],
    optimum_errors: list[float],
    systems_a: tuple,
    systems_b: tuple,
    trajectories_a: list[Trajectory],
    trajectories_b: list[Trajectory],
    spending_gates: list[bool],
    scores: dict[str, object],
    archive: dict[str, object],
    resource_guards: list[dict[str, int | str]],
    elapsed_wall_seconds: float,
    elapsed_cpu_seconds: float,
) -> dict[str, object]:
    """Assemble fixed optimization, map, score, and archive metadata."""
    gaps_a = [item.checkpoints[-1]["relative_gap"] for item in trajectories_a]
    gaps_b = [item.checkpoints[-1]["relative_gap"] for item in trajectories_b]
    return {
        "schema": "nf19.r14.result.v1",
        "source_sha256": source_sha256,
        "question": "same-span exponential-filter coordinate conditioning under fixed exact steepest descent",
        "objective": {
            "ridge_lambda": 100.0,
            "intercept_penalty": 0.0,
            "target_clipping": [-2.0, 2.0],
            "target_weights": "absolute clipped fit target",
            "normalizer": "fit-only original scored stride-40 raw rows, FP64 mean/std, scale floor 1e-6",
            "arithmetic": "FP64 features, sufficient statistics, mapping, and optimization; scorer predictions cast to FP32",
            "line_search_scale": "unnormalized weighted ridge quadratic; gradient norm is in this scale, normalized-risk gradient divides by weight_sum, and normalized-risk eta equals eta times weight_sum",
        },
        "counts": {
            "fit_components": 64,
            "fit_stride_rows": 31_808,
            "eval_components": 32,
            "eval_scored_rows": scores["selected_rows"],
            "coefficients_per_target": 449,
            "attempted_iterations_per_arm_target": 128,
            "target_iterations_attempted": 512,
            "target_nonzero_steps": sum(
                item.nonzero_steps for item in trajectories_a + trajectories_b
            ),
            "target_noop_steps": sum(item.no_op_steps for item in trajectories_a + trajectories_b),
            "fit_rows_not_physical_epoch_exposures": True,
        },
        "coordinate_map": {
            "ema_half_lives": [8, 32, 128],
            "infinite_impulse_gram_identity_max_abs_error": gram_error,
            "mapped_hessian_max_abs_errors_by_target": hessian_errors,
            "mapped_rhs_max_abs_errors_by_target": rhs_errors,
            "native_b_vs_mapped_a_fit_prediction_max_abs_by_target": direct_errors,
            "mapped_optimum_relative_coefficient_error_by_target": optimum_errors,
            "empirical_feature_covariance_whitening": False,
        },
        "spectra": {
            "arm_a": [
                {
                    "minimum_eigenvalue": item.minimum_eigenvalue,
                    "maximum_eigenvalue": item.maximum_eigenvalue,
                    "condition_number": item.maximum_eigenvalue / item.minimum_eigenvalue,
                }
                for item in systems_a
            ],
            "arm_b": [
                {
                    "minimum_eigenvalue": item.minimum_eigenvalue,
                    "maximum_eigenvalue": item.maximum_eigenvalue,
                    "condition_number": item.maximum_eigenvalue / item.minimum_eigenvalue,
                }
                for item in systems_b
            ],
        },
        "optimization": {
            "arm_a": [_trajectory_summary(item) for item in trajectories_a],
            "arm_b": [_trajectory_summary(item) for item in trajectories_b],
            "terminal_relative_gap_by_target": {"arm_a": gaps_a, "arm_b": gaps_b},
            "mean_terminal_relative_gap": {
                "arm_a": _mean_defined(gaps_a),
                "arm_b": _mean_defined(gaps_b),
            },
            "predeclared_b_over_a_gap_le_0_8_by_target": spending_gates,
            "spending_gate": bool(all(spending_gates)),
        },
        "secondary_eval_scores": scores,
        "resource_guards": resource_guards,
        "parameter_archive": archive,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "elapsed_cpu_seconds": elapsed_cpu_seconds,
    }
