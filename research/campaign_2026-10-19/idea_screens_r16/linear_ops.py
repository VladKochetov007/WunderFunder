"""Fixed FP64 linear algebra for the conditional-innovation screen."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
RIDGE = 100.0
GRAM_RELATIVE_TOLERANCE = 1e-10


def symmetric_gram_pseudoinverse(gram: FloatArray) -> tuple[FloatArray, dict[str, object]]:
    """Apply the design's symmetric Gram eigenvalue cutoff."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram matrix must be square")
    gram = (gram + gram.T) * 0.5
    values, vectors = np.linalg.eigh(gram)
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise FloatingPointError("Gram eigensystem is nonfinite")
    lambda_max = max(float(values[-1]), 0.0)
    tolerance = GRAM_RELATIVE_TOLERANCE * lambda_max
    if float(values[0]) < -tolerance:
        raise FloatingPointError("Gram matrix has a negative eigenvalue below tolerance")
    keep = values > tolerance
    inverse = np.zeros_like(values)
    inverse[keep] = 1.0 / values[keep]
    pinv = (vectors * inverse) @ vectors.T
    return pinv, {
        "rank": int(np.count_nonzero(keep)),
        "lambda_max": lambda_max,
        "relative_tolerance": GRAM_RELATIVE_TOLERANCE,
        "absolute_tolerance": tolerance,
        "minimum_eigenvalue": float(values[0]),
    }


def clip_pca_roundoff(eigenvalues: FloatArray) -> tuple[FloatArray, float]:
    """Reject material negative covariance modes and retain all positive modes."""
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("PCA eigenvalue vector must be nonempty")
    largest = max(float(eigenvalues[0]), 0.0)
    tolerance = GRAM_RELATIVE_TOLERANCE * largest
    if float(eigenvalues[-1]) < -tolerance:
        raise FloatingPointError("PCA covariance has a negative eigenvalue below tolerance")
    return np.maximum(eigenvalues, 0.0), tolerance


def fit_ridge_with_intercept(
    design: FloatArray, response: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve sum-squared-error ridge100 with an unpenalized fitted intercept."""
    if design.ndim != 2 or response.ndim != 2 or design.shape[0] != response.shape[0]:
        raise ValueError("ridge design and response row counts must agree")
    design_mean = design.mean(axis=0, dtype=np.float64)
    response_mean = response.mean(axis=0, dtype=np.float64)
    centered_design = design - design_mean
    centered_response = response - response_mean
    gram = centered_design.T @ centered_design
    beta = np.linalg.solve(
        gram + RIDGE * np.eye(gram.shape[0]), centered_design.T @ centered_response
    )
    residual = centered_response - centered_design @ beta
    if not all(np.isfinite(value).all() for value in (beta, residual, design_mean, response_mean)):
        raise FloatingPointError("ridge fit produced nonfinite values")
    return beta, design_mean, response_mean
