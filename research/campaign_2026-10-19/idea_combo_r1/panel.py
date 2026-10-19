"""Fit-only R15 feature reconstruction and deterministic norm parity checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from idea_screens_r15.panel import collect_fit_inputs, fit_features_for_arm
from idea_screens_r15.pipeline_reports import validate_loaded_panel
from idea_screens_r15.representation import DictionaryAutoencoder, fit_normalizer


def _require_equal_normalizer(name: str, actual: np.ndarray, stored: np.ndarray) -> None:
    """Fail closed if recomputed fit-only statistics differ from R15 bytes."""
    if (
        actual.shape != stored.shape
        or actual.dtype != stored.dtype
        or not np.array_equal(actual, stored)
    ):
        raise ValueError(f"recomputed fit-only {name} differs from stored R15 normalizer")


def prepare_fit_panel(
    sequences: Sequence[dict[str, object]],
    manifest: dict[str, object],
    dictionary: DictionaryAutoencoder,
    restored: dict[str, np.ndarray],
    memory_guard: Callable[[str], dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Build full-history sparse+EMA fit rows and verify both R15 normalizers."""
    counts = validate_loaded_panel(list(sequences), manifest)
    if counts["fit_components"] != 64 or counts["fit_stride_rows"] != 31_808:
        raise ValueError("combo fit panel counts differ from accepted R15 panel")
    fit_raw = collect_fit_inputs(sequences)
    raw_mean, raw_scale = fit_normalizer(fit_raw)
    _require_equal_normalizer("raw mean", raw_mean, restored["raw_input_mean"])
    _require_equal_normalizer("raw scale", raw_scale, restored["raw_input_scale"])
    features, targets, downstream_mean, downstream_scale, metadata = fit_features_for_arm(
        list(sequences),
        dictionary,
        restored["raw_input_mean"],
        restored["raw_input_scale"],
        memory_guard,
    )
    _require_equal_normalizer(
        "sparse downstream mean", downstream_mean, restored["sparse_downstream_mean"]
    )
    _require_equal_normalizer(
        "sparse downstream scale", downstream_scale, restored["sparse_downstream_scale"]
    )
    if features.shape != (31_808, 256) or targets.shape != (31_808, 2):
        raise ValueError("combo normalized fit matrix dimensions differ")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise FloatingPointError("combo fit features or targets are nonfinite")
    return (
        features,
        targets,
        {
            "raw_input_mean": raw_mean,
            "raw_input_scale": raw_scale,
            "sparse_downstream_mean": downstream_mean,
            "sparse_downstream_scale": downstream_scale,
        },
        {
            "fit_components": 64,
            "fit_rows": 31_808,
            "normalizer_exact_match": True,
            "feature_width": 256,
            "normalizer_floor": 1e-6,
            "downstream_scale_min": float(downstream_scale.min()),
            "downstream_scale_max": float(downstream_scale.max()),
            "feature_diagnostics": metadata,
        },
    )
