"""Fit the fixed NumPy conditional-innovation estimator and audit its gains."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .linear_ops import (
    RIDGE,
    clip_pca_roundoff,
    fit_ridge_with_intercept,
    symmetric_gram_pseudoinverse,
)

FloatArray = NDArray[np.float64]
GROUPS = {
    "i0": (0, 52),
    "i1": (52, 104),
    "tail": (104, 112),
}
BASELINE_FLOOR = 1e-12
PCA_EIGEN_FLOOR = 1e-12


class HalfRows(NamedTuple):
    """Selected past and innovation rows with original component boundaries."""

    components: tuple[int, ...]
    past: FloatArray
    delta: FloatArray
    component_slices: dict[int, tuple[int, int]]


def fit_input_transform(
    train_past: FloatArray, held_past: FloatArray
) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    """Fit raw scaling and common three-PC coordinates on training rows only."""
    if train_past.ndim != 2 or train_past.shape[1] != 112 or held_past.ndim != 2:
        raise ValueError("past matrices must have 112 columns")
    if not np.isfinite(train_past).all() or not np.isfinite(held_past).all():
        raise ValueError("past inputs must be finite")
    mean = train_past.mean(axis=0, dtype=np.float64)
    raw_std = train_past.std(axis=0, dtype=np.float64)
    scale = np.maximum(raw_std, 1e-6)
    train = (train_past - mean) / scale
    held = (held_past - mean) / scale
    pca_center = train.mean(axis=0, dtype=np.float64)
    train -= pca_center
    held -= pca_center
    covariance = (train.T @ train) / max(train.shape[0] - 1, 1)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise FloatingPointError("PCA eigensystem is nonfinite")
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    eigenvalues, tolerance = clip_pca_roundoff(eigenvalues)
    basis = eigenvectors[:, :3].copy()
    top_values = eigenvalues[:3].copy()
    for column in range(3):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    pc_scale = np.sqrt(np.maximum(top_values, PCA_EIGEN_FLOOR))
    train_pc = (train @ basis) / pc_scale
    held_pc = (held @ basis) / pc_scale
    if not np.isfinite(train_pc).all() or not np.isfinite(held_pc).all():
        raise FloatingPointError("normalized common-PC scores are nonfinite")
    state: dict[str, object] = {
        "input_mean": mean,
        "input_scale": scale,
        "input_scale_floor_count": int(np.count_nonzero(raw_std < 1e-6)),
        "pca_center": pca_center,
        "pca_basis": basis,
        "pca_eigenvalues": top_values,
        "pca_eigenvalues_all": eigenvalues.copy(),
        "pca_positive_rank_1e12": int(np.count_nonzero(eigenvalues > PCA_EIGEN_FLOOR)),
        "pca_negative_tolerance": tolerance,
        "pca_covariance_denominator": "n_fit_rows_minus_1",
    }
    return np.concatenate((train, train_pc), axis=1), np.concatenate((held, held_pc), axis=1), state


def fit_source_residualizer(
    z_train: FloatArray, source_train: FloatArray, z_held: FloatArray, source_held: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, object]]:
    """Remove the fixed destination/common-control span from one source block."""
    gram = z_train.T @ z_train
    pinv, rank = symmetric_gram_pseudoinverse(gram)
    gamma = pinv @ (z_train.T @ source_train)
    train_residual = source_train - z_train @ gamma
    held_residual = source_held - z_held @ gamma
    rms = np.sqrt(np.mean(np.square(train_residual), axis=0, dtype=np.float64))
    scale = np.maximum(rms, 1e-6)
    train_residual /= scale
    held_residual /= scale
    orthogonality = z_train.T @ train_residual
    diagnostic = {
        **rank,
        "source_residual_rms_min": float(rms.min()),
        "source_residual_rms_max": float(rms.max()),
        "source_residual_scale_floor_count": int(np.count_nonzero(rms < 1e-6)),
        "max_abs_training_orthogonality_after_scale": float(np.abs(orthogonality).max()),
    }
    if not all(np.isfinite(value).all() for value in (gamma, train_residual, held_residual, scale)):
        raise FloatingPointError("source residualization produced nonfinite values")
    return train_residual, held_residual, gamma, {**diagnostic, "source_residual_scale": scale}


def _score_pair(
    target: FloatArray,
    base_prediction: FloatArray,
    corrected_prediction: FloatArray,
    component_slices: dict[int, tuple[int, int]],
    destination: str,
) -> dict[str, object]:
    """Summarize all held rows, retaining undefined component denominators."""
    if not all(
        np.isfinite(value).all() for value in (target, base_prediction, corrected_prediction)
    ):
        raise FloatingPointError("held innovation predictions or targets are nonfinite")
    with np.errstate(over="ignore", invalid="ignore"):
        base_error = np.square(target - base_prediction)
        corrected_error = np.square(target - corrected_prediction)
        target_energy = np.square(target)
    if not all(np.isfinite(value).all() for value in (base_error, corrected_error, target_energy)):
        raise FloatingPointError("held SSE terms overflowed or became nonfinite")
    base_sse = float(base_error.sum(dtype=np.float64))
    corrected_sse = float(corrected_error.sum(dtype=np.float64))
    energy = float(target_energy.sum(dtype=np.float64))
    if not np.isfinite([base_sse, corrected_sse, energy]).all():
        raise FloatingPointError("held SSE totals overflowed or became nonfinite")
    components: list[dict[str, object]] = []
    gains: list[float] = []
    undefined = False
    for component, (start, stop) in component_slices.items():
        component_base = float(base_error[start:stop].sum(dtype=np.float64))
        component_corrected = float(corrected_error[start:stop].sum(dtype=np.float64))
        component_energy = float(target_energy[start:stop].sum(dtype=np.float64))
        gain = (
            (component_base - component_corrected) / component_base
            if component_base > BASELINE_FLOOR
            else None
        )
        if gain is not None and not math.isfinite(gain):
            raise FloatingPointError("component relative SSE reduction is nonfinite")
        if gain is None:
            undefined = True
        else:
            gains.append(float(gain))
        components.append(
            {
                "component": int(component),
                "rows": int(stop - start),
                "destination_channels": int(target.shape[1]),
                "baseline_sse": component_base,
                "corrected_sse": component_corrected,
                "standardized_delta_energy": component_energy,
                "relative_sse_reduction": gain,
            }
        )
    overall = (base_sse - corrected_sse) / base_sse if base_sse > BASELINE_FLOOR else None
    if overall is not None and not math.isfinite(overall):
        raise FloatingPointError("overall relative SSE reduction is nonfinite")
    median = (
        float(np.median(gains)) if not undefined and len(gains) == len(component_slices) else None
    )
    return {
        "destination": destination,
        "held_rows": int(target.shape[0]),
        "destination_channels": int(target.shape[1]),
        "baseline_sse": base_sse,
        "corrected_sse": corrected_sse,
        "standardized_delta_energy": energy,
        "relative_sse_reduction": overall,
        "component_median_relative_sse_reduction": median,
        "component_median_defined": median is not None,
        "component_count": len(component_slices),
        "components": components,
    }


def edge_is_stable(left_to_right: dict[str, object], right_to_left: dict[str, object]) -> bool:
    """Apply the fixed 1% per-half and positive component-median gate."""
    for direction in (left_to_right, right_to_left):
        gain = direction.get("relative_sse_reduction")
        median = direction.get("component_median_relative_sse_reduction")
        if (
            not isinstance(gain, (int, float))
            or not isinstance(median, (int, float))
            or not math.isfinite(gain)
            or not math.isfinite(median)
            or gain < 0.01
            or median <= 0.0
        ):
            return False
    return True


def fit_direction(
    train_rows: HalfRows, held_rows: HalfRows, direction: str
) -> tuple[list[dict[str, object]], dict[str, FloatArray], dict[str, object]]:
    """Fit and score all six ordered cross-group pairs for one component split."""
    if (
        train_rows.past.shape != train_rows.delta.shape
        or held_rows.past.shape != held_rows.delta.shape
    ):
        raise ValueError("past and delta matrices must have matching 112-column shapes")
    if train_rows.past.shape[0] != 15_904 or held_rows.past.shape[0] != 15_904:
        raise ValueError("each direction must contain exactly 15,904 fit and held rows")
    x_train, x_held, transform = fit_input_transform(train_rows.past, held_rows.past)
    input_scale = np.asarray(transform["input_scale"], dtype=np.float64)
    y_train = train_rows.delta / input_scale
    y_held = held_rows.delta / input_scale
    parameters: dict[str, FloatArray] = {
        f"{direction}__input_mean": np.asarray(transform["input_mean"]),
        f"{direction}__input_scale": input_scale,
        f"{direction}__pca_center": np.asarray(transform["pca_center"]),
        f"{direction}__pca_basis": np.asarray(transform["pca_basis"]),
        f"{direction}__pca_eigenvalues": np.asarray(transform["pca_eigenvalues"]),
        f"{direction}__pca_eigenvalues_all": np.asarray(transform["pca_eigenvalues_all"]),
    }
    reports: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {
        "direction": direction,
        "fit_components": list(train_rows.components),
        "held_components": list(held_rows.components),
        "fit_rows": int(x_train.shape[0]),
        "held_rows": int(x_held.shape[0]),
        "pca_top3_eigenvalues": np.asarray(transform["pca_eigenvalues"]).tolist(),
        "pca_eigenvalues_all": np.asarray(transform["pca_eigenvalues_all"]).tolist(),
        "pca_positive_rank_1e12": transform["pca_positive_rank_1e12"],
        "pca_negative_tolerance": transform["pca_negative_tolerance"],
        "pca_covariance_denominator": transform["pca_covariance_denominator"],
        "input_scale_floor_count": transform["input_scale_floor_count"],
        "raw_scale_min": float(input_scale.min()),
        "raw_scale_max": float(input_scale.max()),
    }
    for destination, (dest_start, dest_stop) in GROUPS.items():
        z_train = np.concatenate((x_train[:, dest_start:dest_stop], x_train[:, 112:115]), axis=1)
        z_held = np.concatenate((x_held[:, dest_start:dest_stop], x_held[:, 112:115]), axis=1)
        target_train, target_held = (
            y_train[:, dest_start:dest_stop],
            y_held[:, dest_start:dest_stop],
        )
        beta, z_mean, response_mean = fit_ridge_with_intercept(z_train, target_train)
        base_train = response_mean + (z_train - z_mean) @ beta
        base_held = response_mean + (z_held - z_mean) @ beta
        base_residual = target_train - base_train
        parameters[f"{direction}__{destination}__baseline_z_mean"] = z_mean
        parameters[f"{direction}__{destination}__baseline_response_mean"] = response_mean
        parameters[f"{direction}__{destination}__baseline_beta"] = beta
        for source, (src_start, src_stop) in GROUPS.items():
            if source == destination:
                continue
            source_train = x_train[:, src_start:src_stop]
            source_held = x_held[:, src_start:src_stop]
            train_residual, held_residual, gamma, residual_state = fit_source_residualizer(
                z_train, source_train, z_held, source_held
            )
            correction_gram = train_residual.T @ train_residual
            correction = np.linalg.solve(
                correction_gram + RIDGE * np.eye(correction_gram.shape[0]),
                train_residual.T @ base_residual,
            )
            corrected_held = base_held + held_residual @ correction
            edge = _score_pair(
                target_held,
                base_held,
                corrected_held,
                held_rows.component_slices,
                destination,
            )
            edge.update(
                {
                    "source": source,
                    "direction": direction,
                    "source_channels": int(src_stop - src_start),
                    "source_residual_rank": residual_state["rank"],
                    "source_residualization": {
                        key: value
                        for key, value in residual_state.items()
                        if key != "source_residual_scale"
                    },
                }
            )
            if not np.isfinite(correction).all():
                raise FloatingPointError("source correction ridge produced nonfinite coefficients")
            parameters[f"{direction}__{source}_to_{destination}__gamma"] = gamma
            parameters[f"{direction}__{source}_to_{destination}__residual_scale"] = np.asarray(
                residual_state["source_residual_scale"]
            )
            parameters[f"{direction}__{source}_to_{destination}__correction_beta"] = correction
            reports.append(edge)
    return reports, parameters, diagnostics
