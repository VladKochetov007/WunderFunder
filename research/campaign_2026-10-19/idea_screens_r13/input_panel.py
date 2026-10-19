"""Build R13 fit and evaluation inputs from the frozen R3 panel."""

from __future__ import annotations

import numpy as np
from attrs import frozen
from idea_screens_r1.feature_views import build_view
from idea_screens_r2.views import build_view_for
from idea_screens_r7.model_math import apply_normalizer, fit_normalizer
from idea_screens_r7.panel import FIT_ROW_STRIDE, SEQUENCE_ROWS, collect_fit_rows

BASE_VIEW = "raw_signed_delta_ema_hl32"
WARMUP_VIEW = "raw_warmup99_context"
ARM_NAMES = ("zero", "current_bypass", "warmup99")
CONTEXT_START = 99


@frozen
class FitPanel:
    """Fit arrays, shared fit-only statistics, side inputs and evaluation records."""

    features_by_arm: dict[str, np.ndarray]
    targets: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    evaluation: list[dict]
    panel_stats: dict


def _normalize_raw_block(
    values: np.ndarray, timesteps: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Use the shared raw-feature statistics and preserve the exact zero warmup side."""
    normalized = (np.asarray(values, dtype=np.float32) - mean[:112]) / scale[:112]
    normalized = np.asarray(normalized, dtype=np.float32)
    normalized[timesteps < CONTEXT_START] = 0.0
    return np.ascontiguousarray(normalized)


def _selected_rows(sequence: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the R7 stride rows selected by the unchanged scored mask."""
    x = np.asarray(sequence["x"], dtype=np.float32)
    mask = np.asarray(sequence["scored"], dtype=bool)
    if x.shape != (SEQUENCE_ROWS, 112) or mask.shape != (SEQUENCE_ROWS,):
        raise ValueError("R13 requires complete 20,000-row R3 sequences")
    if mask[:CONTEXT_START].any():
        raise ValueError("R3 scored mask includes rows before the 99-row warmup")
    rows = np.arange(0, SEQUENCE_ROWS, FIT_ROW_STRIDE)
    return x, rows[mask[rows]]


def prepare_fit_panel(sequences: list[dict]) -> FitPanel:
    """Collect the fixed fit rows and normalize all three side-block arms consistently."""
    raw_fit, targets, evaluation, panel_stats = collect_fit_rows(sequences)
    mean, scale = fit_normalizer(raw_fit)
    normalized_base = apply_normalizer(raw_fit, mean, scale)
    fit_sequences = [item for item in sequences if item.get("role") == "fit"]
    current_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    for sequence in fit_sequences:
        x, rows = _selected_rows(sequence)
        context_view, _ = build_view(x, WARMUP_VIEW)
        current_parts.append(_normalize_raw_block(x[rows], rows, mean, scale))
        context_parts.append(_normalize_raw_block(context_view[rows, 112:], rows, mean, scale))
    current = np.concatenate(current_parts, axis=0)
    warmup = np.concatenate(context_parts, axis=0)
    if current.shape != (len(targets), 112) or warmup.shape != current.shape:
        raise ValueError("R13 side blocks do not align with R7 fit rows")
    if not np.array_equal(current, normalized_base[:, :112]):
        raise ValueError("current raw side block differs from the shared base normalizer")
    zero = np.zeros_like(current)
    arms = {
        "zero": np.concatenate((normalized_base, zero), axis=1),
        "current_bypass": np.concatenate((normalized_base, current), axis=1),
        "warmup99": np.concatenate((normalized_base, warmup), axis=1),
    }
    if any(values.shape != (len(targets), 336) for values in arms.values()):
        raise ValueError("R13 fit arms must each have 336 input columns")
    return FitPanel(arms, targets, mean, scale, evaluation, panel_stats)


def prepare_evaluation_rows(
    sequence: dict, mean: np.ndarray, scale: np.ndarray
) -> tuple[int, np.ndarray, dict[str, np.ndarray], np.ndarray, int]:
    """Build selected causal evaluation rows with the same fit-only normalization."""
    x = np.asarray(sequence["x"], dtype=np.float32)
    y = np.asarray(sequence["y"], dtype=np.float32)
    mask = np.asarray(sequence["scored"], dtype=bool)
    if x.shape != (SEQUENCE_ROWS, 112) or y.shape != (SEQUENCE_ROWS, 2):
        raise ValueError("R13 evaluation records must contain complete input and target arrays")
    if mask.shape != (SEQUENCE_ROWS,) or mask[:CONTEXT_START].any():
        raise ValueError("R13 evaluation mask violates the 99-row warmup boundary")
    rows = np.flatnonzero(mask)
    if not np.isfinite(y[rows]).all():
        raise ValueError("selected R13 evaluation targets must be finite")
    base = apply_normalizer(build_view_for(x, BASE_VIEW)[rows], mean, scale)
    warmup_view, _ = build_view(x, WARMUP_VIEW)
    current = _normalize_raw_block(x[rows], rows, mean, scale)
    warmup = _normalize_raw_block(warmup_view[rows, 112:], rows, mean, scale)
    sides = {"zero": np.zeros_like(current), "current_bypass": current, "warmup99": warmup}
    if any(value.shape != (len(rows), 112) for value in sides.values()):
        raise ValueError("R13 evaluation side-block shape mismatch")
    return int(sequence["component"]), base, sides, y[rows], int(len(rows))
