"""Fixed, no-fit causal output transforms for existing prediction arrays."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SCENARIO_NAMES = (
    "baseline",
    "hold_i0_and",
    "hold_i0_or",
    "cadence2",
    "cadence4",
    "cadence8",
    "blend_half_and",
    "ema_raw_tau4",
    "ema_raw_tau16",
    "ema_raw_tau64",
    "ema_clipped_tau4",
    "ema_clipped_tau16",
    "ema_clipped_tau64",
    "oracle_clipped_label_run",
)
AGE_EDGES = np.asarray([0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 20_001], dtype=np.int32)
AGE_LABELS = (
    "[0,1)",
    "[1,2)",
    "[2,4)",
    "[4,8)",
    "[8,16)",
    "[16,32)",
    "[32,64)",
    "[64,128)",
    "[128,512)",
    "[512,20001)",
)


def event_flags(price_p0: np.ndarray, price_p11: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sequence-reset AND/OR price-code change flags."""
    if price_p0.shape != price_p11.shape or price_p0.ndim != 2:
        raise ValueError("price inputs must be equal (sequence, time) arrays")
    if not np.isfinite(price_p0).all() or not np.isfinite(price_p11).all():
        raise ValueError("selected price inputs contain nonfinite values")
    p0_changed = np.zeros(price_p0.shape, dtype=bool)
    p11_changed = np.zeros(price_p11.shape, dtype=bool)
    p0_changed[:, 1:] = price_p0[:, 1:] != price_p0[:, :-1]
    p11_changed[:, 1:] = price_p11[:, 1:] != price_p11[:, :-1]
    return p0_changed & p11_changed, p0_changed | p11_changed


def age_buckets(and_event: np.ndarray) -> np.ndarray:
    """Map rows to fixed R1 age bins since the last joint price event."""
    if and_event.ndim != 2:
        raise ValueError("event flags must have sequence and time axes")
    n_time = and_event.shape[1]
    row = np.arange(n_time, dtype=np.int32)[None, :]
    last_event = np.maximum.accumulate(np.where(and_event, row, 0), axis=1)
    age = row - last_event
    return (np.searchsorted(AGE_EDGES, age, side="right") - 1).astype(np.uint8)


def hold_until_event(prediction: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold the last emitted row output, always emitting baseline output at row zero."""
    if prediction.ndim != 3 or prediction.shape[:2] != event.shape:
        raise ValueError("expected (sequence, time, target) predictions and aligned event flags")
    n_batch, n_time, _ = prediction.shape
    emit = event.copy()
    emit[:, 0] = True
    row = np.arange(n_time, dtype=np.int32)[None, :]
    last_emit = np.maximum.accumulate(np.where(emit, row, 0), axis=1)
    held = prediction[np.arange(n_batch)[:, None], last_emit]
    return held, emit


def causal_ema(prediction: np.ndarray, tau: int) -> np.ndarray:
    """Apply a per-sequence causal EMA with the fixed coefficient alpha=1/tau."""
    if prediction.ndim != 3 or prediction.shape[1] == 0 or tau < 1:
        raise ValueError("EMA requires nonempty (sequence, time, target) predictions and tau >= 1")
    values = prediction.astype(np.float64, copy=False)
    alpha = 1.0 / float(tau)
    decay = 1.0 - alpha
    initial = np.empty((values.shape[0], 1, values.shape[2]), dtype=np.float64)
    initial[:, 0, :] = decay * values[:, 0, :]
    smoothed, _ = lfilter([alpha], [1.0, -decay], values, axis=1, zi=initial)
    return smoothed.astype(np.float32)


def clipped_label_run_oracle(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average selected clipped predictions within future-defined label runs.

    This diagnostic uses the full label sequence to define runs and is not a
    causal or deployable predictor. Each target is partitioned separately.
    Returns the oracle predictions and weighted within-run variance sums.
    """
    if target.shape != prediction.shape or target.ndim != 3 or mask.shape != target.shape[:2]:
        raise ValueError("oracle target, prediction, and mask shapes do not align")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("oracle inputs must be finite before label-run grouping")
    clipped_target = np.clip(target, -2.0, 2.0)
    clipped_prediction = np.clip(prediction, -2.0, 2.0).astype(np.float32)
    oracle = clipped_prediction.copy()
    within_variance = np.zeros(2, dtype=np.float64)
    for sequence in range(target.shape[0]):
        for component in range(2):
            labels = clipped_target[sequence, :, component]
            starts = np.empty(len(labels), dtype=bool)
            starts[0] = True
            starts[1:] = labels[1:] != labels[:-1]
            run_ids = np.cumsum(starts, dtype=np.int32) - 1
            selected = mask[sequence]
            if not selected.any():
                continue
            count = int(starts.sum())
            selected_runs = run_ids[selected]
            selected_prediction = clipped_prediction[sequence, selected, component].astype(
                np.float64
            )
            counts = np.bincount(selected_runs, minlength=count)
            sums = np.bincount(selected_runs, weights=selected_prediction, minlength=count)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            oracle[sequence, selected, component] = means[selected_runs].astype(np.float32)
            weights = np.abs(labels[selected]).astype(np.float64)
            residual = selected_prediction - means[selected_runs]
            within_variance[component] += np.sum(weights * residual * residual)
    return oracle, within_variance


def build_scenarios(
    prediction: np.ndarray,
    and_event: np.ndarray,
    or_event: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the fixed output-policy matrix without changing backbone state."""
    if (
        prediction.ndim != 3
        or prediction.shape != target.shape
        or mask.shape != prediction.shape[:2]
    ):
        raise ValueError("prediction, target, and mask arrays do not align")
    if and_event.shape != mask.shape or or_event.shape != mask.shape:
        raise ValueError("event flags do not align to prediction rows")
    and_held, _ = hold_until_event(prediction, and_event)
    or_held, _ = hold_until_event(prediction, or_event)
    outputs: dict[str, np.ndarray] = {
        "baseline": prediction,
        "hold_i0_and": and_held,
        "hold_i0_or": or_held,
    }
    row = np.arange(prediction.shape[1], dtype=np.int32)
    for period in (2, 4, 8):
        cadence = np.broadcast_to((row % period == 0)[None, :], mask.shape)
        held, _ = hold_until_event(prediction, cadence)
        outputs[f"cadence{period}"] = held
    outputs["blend_half_and"] = (
        0.5 * prediction.astype(np.float64) + 0.5 * and_held.astype(np.float64)
    ).astype(np.float32)
    clipped = np.clip(prediction, -2.0, 2.0).astype(np.float32)
    for tau in (4, 16, 64):
        outputs[f"ema_raw_tau{tau}"] = causal_ema(prediction, tau)
    for tau in (4, 16, 64):
        outputs[f"ema_clipped_tau{tau}"] = causal_ema(clipped, tau)
    outputs["oracle_clipped_label_run"], _ = clipped_label_run_oracle(target, clipped, mask)
    if tuple(outputs) != SCENARIO_NAMES:
        raise AssertionError("scenario order differs from the predeclared matrix")
    if any(value.shape != prediction.shape for value in outputs.values()):
        raise AssertionError("a fixed output policy changed prediction shape")
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("a transformed output contains nonfinite predictions")
    return outputs
