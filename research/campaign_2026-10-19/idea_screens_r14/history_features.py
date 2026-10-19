"""Build R14's fixed reset-local EMA history and coordinate map."""

from __future__ import annotations

import numpy as np

RAW_WIDTH = 112
HALF_LIVES = (8.0, 32.0, 128.0)
FEATURE_WIDTH = 449


def validate_raw(raw: np.ndarray) -> np.ndarray:
    """Validate and expose one finite complete sequence as FP32 input."""
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != RAW_WIDTH or values.shape[0] < 1:
        raise ValueError("expected a nonempty complete (T,112) raw sequence")
    if not np.isfinite(values).all():
        raise ValueError("raw sequence contains nonfinite values")
    return values


def fit_raw_normalizer(sequences: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute FP64 raw mean and scale from fit scored stride-40 rows only."""
    fit_components: set[int] = set()
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    total = np.zeros(RAW_WIDTH, dtype=np.float64)
    count = 0
    for sequence in sequences:
        if sequence.get("role") != "fit":
            continue
        raw = validate_raw(sequence["x"])
        scored = np.asarray(sequence["scored"], dtype=bool)
        if scored.shape != (len(raw),):
            raise ValueError("fit scored mask shape differs from its sequence")
        rows = np.arange(0, len(raw), 40, dtype=np.int64)
        rows = rows[scored[rows]]
        if rows.size == 0:
            raise ValueError("fit sequence contributes no scored stride-40 rows")
        fit_components.add(int(sequence["component"]))
        selected.append((raw, rows))
        total += np.asarray(raw[rows], dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(rows)
    if len(fit_components) != 64 or len(selected) != 64:
        raise ValueError("R14 requires one fit sequence for each of 64 components")
    if count != 31_808 or not np.isfinite(total).all():
        raise ValueError("R14 fit normalizer rows differ from the frozen 64x497 panel")
    mean = total / count
    squared_deviation = np.zeros(RAW_WIDTH, dtype=np.float64)
    for raw, rows in selected:
        centered = np.asarray(raw[rows], dtype=np.float64) - mean
        squared_deviation += np.square(centered).sum(axis=0, dtype=np.float64)
    scale = np.sqrt(squared_deviation / count)
    scale = np.maximum(scale, 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("R14 fit normalizer is nonfinite")
    return mean, scale, count


def build_a_features(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Build normalized raw plus three full-history unit-norm EMA blocks.

    States reset at row zero and advance over every row before downstream
    stride or scored-mask selection.
    """
    values = validate_raw(raw)
    center = np.asarray(mean, dtype=np.float64)
    divisor = np.asarray(scale, dtype=np.float64)
    if center.shape != (RAW_WIDTH,) or divisor.shape != (RAW_WIDTH,):
        raise ValueError("R14 normalizer must have 112 entries")
    if not np.isfinite(center).all() or not np.isfinite(divisor).all() or (divisor <= 0).any():
        raise ValueError("R14 normalizer must be finite with positive scales")

    x = np.asarray(values, dtype=np.float64)
    features = np.empty((len(x), FEATURE_WIDTH), dtype=np.float64)
    features[:, :RAW_WIDTH] = (x - center) / divisor
    decay = np.exp2(-1.0 / np.asarray(HALF_LIVES, dtype=np.float64))
    innovation_scale = np.sqrt(1.0 - decay**2)
    states = np.zeros((len(HALF_LIVES), RAW_WIDTH), dtype=np.float64)
    previous = x[0]
    for row in range(len(x)):
        if row:
            delta = (x[row] - previous) / divisor
            states = decay[:, None] * states + innovation_scale[:, None] * delta[None, :]
            previous = x[row]
        start = RAW_WIDTH
        features[row, start : 4 * RAW_WIDTH] = states.reshape(-1)
    features[:, -1] = 1.0
    if not np.isfinite(features).all():
        raise ValueError("R14 full-history feature matrix is nonfinite")
    return features


def coordinate_map() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the analytic infinite-impulse Gram, whitener, and full map S."""
    decay = np.exp2(-1.0 / np.asarray(HALF_LIVES, dtype=np.float64))
    gains = np.sqrt(1.0 - decay**2)
    gram = np.outer(gains, gains) / (1.0 - np.outer(decay, decay))
    lower = np.linalg.cholesky(gram)
    whitener = np.linalg.solve(lower, np.eye(len(HALF_LIVES), dtype=np.float64))
    gram_error = float(np.max(np.abs(whitener @ gram @ whitener.T - np.eye(3))))

    transform = np.eye(FEATURE_WIDTH, dtype=np.float64)
    transform[RAW_WIDTH : 4 * RAW_WIDTH, RAW_WIDTH : 4 * RAW_WIDTH] = np.kron(
        whitener, np.eye(RAW_WIDTH, dtype=np.float64)
    )
    if not np.isfinite(transform).all() or gram_error > 1e-12:
        raise ValueError("R14 analytic EMA coordinate map failed its Gram identity")
    return gram, whitener, transform, gram_error


def direct_map_check(
    features_a: np.ndarray,
    beta_b: np.ndarray,
    transform: np.ndarray,
    whitener: np.ndarray,
) -> float:
    """Compare native B feature predictions with their mapped A function."""
    beta_a = transform.T @ beta_b
    maximum_error = 0.0
    for start in range(0, len(features_a), 512):
        block = features_a[start : start + 512]
        features_b = block.copy()
        states = block[:, RAW_WIDTH:-1].reshape(len(block), 3, RAW_WIDTH)
        features_b[:, RAW_WIDTH:-1] = np.einsum("ij,njk->nik", whitener, states).reshape(
            len(block), -1
        )
        direct, mapped = features_b @ beta_b, block @ beta_a
        maximum_error = max(maximum_error, float(np.max(np.abs(direct - mapped))))
        del block, features_b, states, direct, mapped
    if not np.isfinite(maximum_error) or maximum_error > 1e-6:
        raise ValueError("R14 native-B and mapped-A fit predictions differ")
    return maximum_error
