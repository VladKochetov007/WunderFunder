"""Causal geometry and exact-level features for lazy GRU training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch

from wnn.moment_geometry.analysis import robust_arc
from wnn.moment_geometry.features import SELECTED

MODE = "causal_geometry_discrete"
SCHEMA = "causal_geometry_discrete_v1"
FEATURE_NAMES = (
    *(
        f"geometry_{name}_{index}"
        for name in ("mean", "std", "arc_residual", "atom_occupancy", "normalized")
        for index in range(4)
    ),
    *(f"level_current_{field}_{level}" for field in range(4) for level in range(4)),
    *(f"level_occupancy_{field}_{level}" for field in range(4) for level in range(4)),
    *(f"level_dominant_event_{field}" for field in range(4)),
    *(f"level_log_dwell_{field}" for field in range(4)),
)


def validate_prior(prior: Mapping[str, object]) -> dict[str, object]:
    """Validate the fitting-only constants needed by the feature transform."""
    if prior.get("schema") != SCHEMA:
        raise ValueError("geometry/discrete prior has an unsupported schema")
    vectors = {
        "beta": (4,),
        "intercept": (4,),
        "atoms": (4,),
        "vocabulary": (4, 3),
    }
    result = dict(prior)
    for name, shape in vectors.items():
        values = np.asarray(prior.get(name), dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"geometry/discrete prior {name} must be finite with shape {shape}")
        result[name] = values.tolist()
    return result


def fit_prior(raw_sequences: Iterable[np.ndarray]) -> dict[str, object]:
    """Fit arc coefficients, anchor atoms and exact-level dictionaries on fitting rows."""
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    counts: list[dict[float, int]] = [{} for _ in range(4)]
    sequence_count = 0
    row_count = 0
    for raw in raw_sequences:
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 112 or not np.isfinite(values).all():
            raise ValueError("fit geometry/discrete prior from finite (T,112) sequences")
        selected = values[:, SELECTED]
        means.append(selected.mean(axis=0, dtype=np.float64))
        variances.append(selected.var(axis=0, dtype=np.float64))
        for field in range(4):
            levels, frequencies = np.unique(selected[:, field], return_counts=True)
            field_counts = counts[field]
            for level, frequency in zip(levels, frequencies, strict=True):
                key = float(level)
                field_counts[key] = field_counts.get(key, 0) + int(frequency)
        sequence_count += 1
        row_count += len(values)
    if not sequence_count:
        raise ValueError("cannot fit geometry/discrete prior without fitting sequences")

    means_array = np.stack(means)
    variances_array = np.stack(variances)
    beta = np.empty(4, dtype=np.float32)
    intercept = np.empty(4, dtype=np.float32)
    atoms = np.empty(4, dtype=np.float32)
    vocabulary = np.empty((4, 3), dtype=np.float32)
    for field in range(4):
        arc = robust_arc(means_array[:, field], variances_array[:, field])
        beta[field] = arc["beta"]
        intercept[field] = arc["intercept"]
        levels = np.asarray(sorted(counts[field]), dtype=np.float32)
        frequencies = np.asarray([counts[field][float(value)] for value in levels])
        if len(levels) < 3:
            raise ValueError("each selected input field needs at least three fitting levels")
        top = np.argsort(frequencies, kind="stable")[-8:]
        center = arc["beta"] / 2
        endpoint = center + np.sqrt(max(arc["intercept"] + center**2, 0.0))
        atoms[field] = levels[top[np.argmin(np.abs(levels[top] - endpoint))]]
        vocabulary[field] = levels[np.argsort(frequencies, kind="stable")[-3:][::-1]]
    return validate_prior(
        {
            "schema": SCHEMA,
            "selected_raw_columns": list(SELECTED),
            "feature_order": list(FEATURE_NAMES),
            "beta": beta.tolist(),
            "intercept": intercept.tolist(),
            "arc_inlier_fraction": [
                robust_arc(means_array[:, field], variances_array[:, field])["fraction"]
                for field in range(4)
            ],
            "atoms": atoms.tolist(),
            "vocabulary": vocabulary.tolist(),
            "fit_sequences": sequence_count,
            "fit_rows": row_count,
            "frequency_tie_break": "ascending_exact_float32_level_then_stable_frequency_rank",
        }
    )


def _validate_raw(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 112 or not np.isfinite(values).all():
        raise ValueError("expected finite raw sequence with shape (T,112)")
    return values[:, SELECTED]


def features_numpy(raw: np.ndarray, prior: Mapping[str, object]) -> np.ndarray:
    """Return the fixed 60-column causal transform for one raw sequence."""
    values = _validate_raw(raw).astype(np.float64)
    checked = validate_prior(prior)
    count = np.arange(1, len(values) + 1, dtype=np.float64)[:, None]
    mean = np.cumsum(values, axis=0) / count
    second = np.cumsum(values * values, axis=0) / count
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    residual = second - np.asarray(checked["beta"]) * mean - np.asarray(checked["intercept"])
    atoms = np.asarray(checked["atoms"])
    atom_occupancy = np.cumsum(values == atoms, axis=0) / count
    normalized = np.clip((values - mean) / np.maximum(std, 0.05), -8.0, 8.0)

    vocabulary = np.asarray(checked["vocabulary"])
    known = values[:, :, None] == vocabulary[None]
    category = np.concatenate((known, ~known.any(axis=-1, keepdims=True)), axis=-1)
    occupancy = np.cumsum(category, axis=0, dtype=np.float64) / count[:, :, None]
    changed = np.ones(values.shape, dtype=bool)
    changed[1:] = values[1:] != values[:-1]
    event_count = np.cumsum(changed, axis=0)
    dominant_events = np.cumsum(changed & category[..., 0], axis=0) / event_count
    position = np.arange(len(values), dtype=np.int64)[:, None]
    last_change = np.maximum.accumulate(np.where(changed, position, 0), axis=0)
    dwell = np.log1p(np.minimum(position - last_change, 10_000))
    output = np.concatenate(
        (
            mean,
            std,
            residual,
            atom_occupancy,
            normalized,
            category.reshape(len(values), 16),
            occupancy.reshape(len(values), 16),
            dominant_events,
            dwell,
        ),
        axis=-1,
    )
    return output.astype(np.float32)


def features_torch(
    raw: torch.Tensor,
    prior: Mapping[str, object],
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return batched causal features without materializing full-dataset arrays."""
    if raw.ndim != 3 or raw.shape[1] < 1 or raw.shape[2] != 112 or raw.dtype != torch.float32:
        raise ValueError("expected float32 raw batches with shape (B,T,112)")
    expected = (*raw.shape[:2], len(FEATURE_NAMES))
    if output is not None and (
        output.shape != expected or output.dtype != raw.dtype or output.device != raw.device
    ):
        raise ValueError("feature output must match the raw batch, dtype and device")
    checked = validate_prior(prior)
    values = raw[..., list(SELECTED)].to(dtype=torch.float64)
    count = torch.arange(1, raw.shape[1] + 1, device=raw.device, dtype=torch.float64)[None, :, None]
    mean = values.cumsum(dim=1) / count
    second = values.square().cumsum(dim=1) / count
    std = (second - mean.square()).clamp_min(0).sqrt()
    beta = torch.as_tensor(checked["beta"], device=raw.device, dtype=torch.float64)
    intercept = torch.as_tensor(checked["intercept"], device=raw.device, dtype=torch.float64)
    atoms = torch.as_tensor(checked["atoms"], device=raw.device, dtype=torch.float64)
    atom_occupancy = (values == atoms).to(torch.float64).cumsum(dim=1) / count
    normalized = ((values - mean) / std.clamp_min(0.05)).clamp(-8, 8)

    vocabulary = torch.as_tensor(checked["vocabulary"], device=raw.device, dtype=torch.float64)
    known = values.unsqueeze(-1) == vocabulary[None, None]
    category = torch.cat((known, ~known.any(dim=-1, keepdim=True)), dim=-1)
    occupancy = category.to(torch.float64).cumsum(dim=1) / count.unsqueeze(-1)
    changed = torch.ones_like(values, dtype=torch.bool)
    changed[:, 1:] = values[:, 1:] != values[:, :-1]
    event_count = changed.to(torch.float64).cumsum(dim=1).clamp_min(1)
    dominant_events = (changed & category[..., 0]).to(torch.float64).cumsum(dim=1) / event_count
    positions = torch.arange(raw.shape[1], device=raw.device, dtype=torch.int64).view(1, -1, 1)
    last_change = torch.where(changed, positions, torch.zeros_like(positions)).cummax(dim=1).values
    dwell = torch.log1p((positions - last_change).clamp(0, 10_000).to(torch.float64))
    features = torch.cat(
        (
            mean,
            std,
            second - beta * mean - intercept,
            atom_occupancy,
            normalized,
            category.reshape(*raw.shape[:2], 16).to(torch.float64),
            occupancy.reshape(*raw.shape[:2], 16),
            dominant_events,
            dwell,
        ),
        dim=-1,
    ).to(torch.float32)
    if output is None:
        return features
    output.copy_(features)
    return output
