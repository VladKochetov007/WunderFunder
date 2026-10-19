"""Bounded R3 panel row assembly and endpoint scoring for R15."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from idea_screens_r3.probe_helpers import metric_without, raw_sums_by_group

from wnn.gmetric import GlobalAccumulator

from .representation import (
    DictionaryAutoencoder,
    R15Predictor,
    build_sequence_features,
    fit_normalizer,
)

FIT_ROWS = 31_808
EVAL_ROWS = 59_436
STRIDE = 40


def stride_rows(sequence: dict[str, object]) -> np.ndarray:
    """Return original scored rows intersected with the frozen fit stride."""
    mask = np.asarray(sequence["scored"], dtype=bool)
    rows = np.arange(0, len(mask), STRIDE, dtype=np.int64)
    return rows[mask[rows]]


def collect_fit_inputs(sequences: Sequence[dict[str, object]]) -> np.ndarray:
    """Preallocate only the exact fit raw-input matrix for dictionary training."""
    selected = [item for item in sequences if item.get("role") == "fit"]
    if len(selected) != 64:
        raise ValueError("R15 requires exactly 64 fit component sequences")
    inputs = np.empty((FIT_ROWS, 112), dtype=np.float32)
    offset = 0
    for sequence in selected:
        rows = stride_rows(sequence)
        if len(rows) != 497:
            raise ValueError("R15 expected 497 original scored stride rows per fit component")
        stop = offset + len(rows)
        inputs[offset:stop] = np.asarray(sequence["x"])[rows]
        offset = stop
    if offset != FIT_ROWS or not np.isfinite(inputs).all():
        raise ValueError("R15 fit inputs do not match the accepted panel")
    return inputs


def fit_features_for_arm(
    sequences: Sequence[dict[str, object]],
    model: DictionaryAutoencoder,
    raw_mean: np.ndarray,
    raw_scale: np.ndarray,
    memory_guard: Callable[[str], dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Stream full fit histories into one downstream fit matrix and normalize it."""
    fit_sequences = [item for item in sequences if item.get("role") == "fit"]
    features = np.empty((FIT_ROWS, 256), dtype=np.float32)
    targets = np.empty((FIT_ROWS, 2), dtype=np.float32)
    active = np.zeros(16, dtype=np.int64)
    active_by_component: dict[str, list[int]] = {}
    reconstruction_total = 0.0
    row_total = 0
    residual_identity_max = 0.0
    offset = 0
    for index, sequence in enumerate(fit_sequences):
        memory_guard(f"before_{'sparse' if model.sparse else 'dense'}_fit_history_{index}")
        full, meta = build_sequence_features(sequence["x"], raw_mean, raw_scale, model)
        rows = stride_rows(sequence)
        stop = offset + len(rows)
        features[offset:stop] = full[rows]
        targets[offset:stop] = np.asarray(sequence["y"])[rows]
        # Code usage here is input-only and covers each full fit history.
        active += np.asarray(meta["active_code_count_by_factor"], dtype=np.int64)
        active_by_component[str(int(sequence["component"]))] = np.asarray(
            meta["active_code_count_by_factor"], dtype=np.int64
        ).tolist()
        reconstruction_total += float(meta["reconstruction_mse"]) * len(full) * 112
        row_total += len(full)
        residual_identity_max = max(residual_identity_max, float(meta["residual_identity_max_abs"]))
        offset = stop
        del full, rows
        memory_guard(f"after_{'sparse' if model.sparse else 'dense'}_fit_history_{index}")
    if offset != FIT_ROWS or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("R15 downstream fit matrix is incomplete or nonfinite")
    mean, scale = fit_normalizer(features)
    for start in range(0, len(features), 2048):
        stop = min(start + 2048, len(features))
        chunk = features[start:stop]
        features[start:stop] = ((chunk.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("R15 normalized downstream fit features are nonfinite")
    return (
        features,
        targets,
        mean,
        scale,
        {
            "fit_rows": int(offset),
            "feature_width": 256,
            "normalizer_floor": 1e-6,
            "normalizer_scale_min": float(scale.min()),
            "normalizer_scale_max": float(scale.max()),
            "active_code_count_by_factor_aggregated_sequences": active.tolist(),
            "code_active_rows_by_fit_component_all_history": active_by_component,
            "all_history_rows_for_reconstruction_mse": row_total,
            "reconstruction_mse_all_fit_sequence_rows": reconstruction_total / (row_total * 112),
            "residual_identity_max_abs_all_fit_sequence_rows": residual_identity_max,
        },
    )


def _predict(model: R15Predictor, features: np.ndarray) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            x = torch.from_numpy(np.ascontiguousarray(features[start : start + 2048]))
            y = model.forward_mean(x)
            if not torch.isfinite(y).all().item():
                raise FloatingPointError("nonfinite R15 endpoint prediction")
            outputs.append(y.cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, 2), dtype=np.float32)


def score_endpoints(
    sequences: Sequence[dict[str, object]],
    models: dict[str, R15Predictor],
    dictionaries: dict[str, DictionaryAutoencoder],
    raw_means: dict[str, np.ndarray],
    raw_scales: dict[str, np.ndarray],
    feature_means: dict[str, np.ndarray],
    feature_scales: dict[str, np.ndarray],
    memory_guard: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    """Score only final dense/sparse endpoints on unchanged original eval masks."""
    pooled = {name: GlobalAccumulator() for name in models}
    components: dict[str, dict[int, GlobalAccumulator]] = {name: {} for name in models}
    component_rows: dict[str, dict[str, int]] = {name: {} for name in models}
    rows_seen = 0
    for index, sequence in enumerate(item for item in sequences if item.get("role") == "eval"):
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        component = int(sequence["component"])
        if mask.shape != (20_000,) or y.shape != (20_000, 2):
            raise ValueError("R15 evaluation arrays do not match complete sequence dimensions")
        selected_y = np.asarray(y[mask], dtype=np.float32)
        if not np.isfinite(selected_y).all():
            raise ValueError("nonfinite selected R15 evaluation targets")
        rows_seen += int(mask.sum())
        for name, model in models.items():
            memory_guard(f"before_eval_{name}_{index}")
            full, _ = build_sequence_features(
                x, raw_means[name], raw_scales[name], dictionaries[name]
            )
            selected = full[mask]
            selected[:] = (
                (selected.astype(np.float64) - feature_means[name]) / feature_scales[name]
            ).astype(np.float32)
            prediction = _predict(model, selected)
            part = GlobalAccumulator()
            valid = np.ones(len(selected_y), dtype=bool)
            if len(selected_y):
                part.add(selected_y, prediction, valid)
                pooled[name].add(selected_y, prediction, valid, group=component)
            components[name][component] = part
            component_rows[name][str(component)] = len(selected_y)
            del full, selected
            memory_guard(f"after_eval_{name}_{index}")
    if rows_seen != EVAL_ROWS:
        raise ValueError(
            f"R15 expected {EVAL_ROWS} original scored evaluation rows, got {rows_seen}"
        )
    arms: dict[str, object] = {}
    for name in models:
        scores = {
            str(key): value.result() if value.selected_rows else None
            for key, value in sorted(components[name].items())
        }
        drops = {
            str(key): metric_without(components[name], key) for key in sorted(components[name])
        }
        official_sums = raw_sums_by_group(pooled[name])
        for key in component_rows[name]:
            official_sums.setdefault(key, np.zeros((2, 6), dtype=np.float64).tolist())
        arms[name] = {
            "official_metric": pooled[name].result(),
            "component_rows": component_rows[name],
            "component_scores": scores,
            "component_drop_one_scores": drops,
            "official_clipped_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": official_sums,
        }
    return {
        "metric_id": "global_wp_clip2_v1",
        "evaluation_components": len(components["dense"]),
        "scored_rows": rows_seen,
        "component_sensitivities_are_not_independent_replications": True,
        "arms": arms,
        "primary_sparse_minus_dense_wp": arms["sparse"]["official_metric"]["wp"]
        - arms["dense"]["official_metric"]["wp"],
        "target_deltas_sparse_minus_dense": [
            arms["sparse"]["official_metric"][f"t{index}"]
            - arms["dense"]["official_metric"][f"t{index}"]
            for index in range(2)
        ],
    }
