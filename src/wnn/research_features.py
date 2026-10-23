from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from wnn.tiedlead import (
    TIEDLEAD_BOOK_WIDTH,
    TIEDLEAD_INSTRUMENT_WIDTH,
    TL3_LEAD_WIDTH,
)

if TYPE_CHECKING:
    import torch

ENDPOINT = np.float32(-5.199337482452393)
MOMENT_FEATURE_MODES = ("causal_moments", "causal_innovations")
FIELD_AGE_FEATURE_MODES = ("group_age", "field_age")
TL3_LEAD_NAMES = tuple(f"tl3_lead_i1_{index}" for index in range(TL3_LEAD_WIDTH))
RESEARCH_FEATURE_MODES = (
    "none",
    "volume28",
    "volume28_current",
    *MOMENT_FEATURE_MODES,
    *FIELD_AGE_FEATURE_MODES,
    "tl3_lead",
)
MOMENT_PROJECTION_WIDTH = 16
MOMENT_FAST_HALF_LIFE = 32.0
MOMENT_SLOW_HALF_LIFE = 512.0
MOMENT_PRIOR_HALF_LIFE = 99.0
MOMENT_BLOCK_ROWS = 512
MOMENT_SAMPLE_STRIDE = 64
MOMENT_SCHEMA = "causal_moments_v1"
_MOMENT_SCALE_FLOOR = np.float32(1e-4)
VOLUME28_NAMES = tuple(f"dVrun_{index}" for index in range(22)) + (
    "pathV_first11",
    "pathV_last11",
    "nchgV_first11",
    "nchgV_last11",
    "netV_first11",
    "netV_last11",
)
FIELD_AGE_CAP = 10_000
RAW_FIELD_GROUPS = ((0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112))


def research_feature_names(mode: str) -> tuple[str, ...]:
    """Return the fixed feature order used by each research representation."""
    if mode not in RESEARCH_FEATURE_MODES:
        raise ValueError(f"Unknown research feature mode: {mode}")
    if mode == "none":
        return ()
    if mode == "volume28":
        return VOLUME28_NAMES
    if mode in MOMENT_FEATURE_MODES:
        moment_names = tuple(
            f"{clock}_{stat}_{index}"
            for clock in ("slow", "fast")
            for stat in ("mean", "log_scale")
            for index in range(MOMENT_PROJECTION_WIDTH)
        )
        if mode == "causal_innovations":
            return moment_names + tuple(
                f"innovation_{index}" for index in range(MOMENT_PROJECTION_WIDTH)
            )
        return moment_names
    if mode in FIELD_AGE_FEATURE_MODES:
        return tuple(f"{mode}_{index}" for index in range(112))
    if mode == "tl3_lead":
        return TL3_LEAD_NAMES
    return tuple(f"currentV_{index}" for index in range(22)) + (
        "current_step_path_first11",
        "current_step_path_last11",
        "current_step_change_first11",
        "current_step_change_last11",
        "current_norm_first11",
        "current_norm_last11",
    )


def research_feature_control_width(mode: str, control: str = "none") -> int:
    """Return the feature suffix width zeroed by a matched-capacity control."""
    if control == "none":
        return 0
    if control != "newest" or mode not in MOMENT_FEATURE_MODES:
        raise ValueError("the newest feature control requires a causal moment feature mode")
    if mode == "causal_moments":
        return len(research_feature_names(mode))
    return MOMENT_PROJECTION_WIDTH


def _raw_array(raw: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(raw, dtype=np.float32)
    squeeze = array.ndim == 2
    if squeeze:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] != 112 or array.shape[1] < 1:
        raise ValueError("Expected nonempty (T,112) or (B,T,112) raw features")
    if not np.isfinite(array).all():
        raise ValueError("Raw features must be finite")
    return array, squeeze


def _field_age_features_numpy(raw: np.ndarray, mode: str) -> np.ndarray:
    """Build reset-local log ages from exact raw-field changes."""
    batch_size, rows, _ = raw.shape
    output = np.empty((batch_size, rows, 112), dtype=np.float32)
    positions = np.arange(rows, dtype=np.int32)[None, :, None]
    groups = ((0, 112),) if mode == "field_age" else RAW_FIELD_GROUPS
    for start, end in groups:
        width = 112 if mode == "field_age" else 1
        changed = np.zeros((batch_size, rows, width), dtype=np.bool_)
        if rows > 1:
            delta = raw[:, 1:, start:end] != raw[:, :-1, start:end]
            changed[:, 1:] = delta if mode == "field_age" else delta.any(axis=-1, keepdims=True)
        last_change = np.where(changed, positions, 0).astype(np.int32, copy=False)
        np.maximum.accumulate(last_change, axis=1, out=last_change)
        np.subtract(positions, last_change, out=last_change)
        np.minimum(last_change, FIELD_AGE_CAP, out=last_change)
        encoded = np.log1p(last_change.astype(np.float32, copy=False))
        output[..., start:end] = encoded
    return output


def _field_age_features_torch(
    raw: torch.Tensor, mode: str, output: torch.Tensor | None = None
) -> torch.Tensor:
    """Build reset-local log ages with a bounded-width cumulative last-change scan."""
    import torch

    expected_shape = (*raw.shape[:2], 112)
    if output is None:
        output = raw.new_empty(expected_shape)
    elif output.shape != expected_shape or output.dtype != raw.dtype or output.device != raw.device:
        raise ValueError("field-age output buffer must match the input batch, dtype, and device")
    batch_size, rows, _ = raw.shape
    positions = torch.arange(rows, device=raw.device, dtype=torch.int32).view(1, rows, 1)
    groups = ((0, 112),) if mode == "field_age" else RAW_FIELD_GROUPS
    for start, end in groups:
        if rows > 1:
            changed = raw[:, 1:, start:end] != raw[:, :-1, start:end]
            if mode == "group_age":
                changed = changed.any(dim=-1, keepdim=True)
            row_zero = torch.zeros(
                (batch_size, 1, changed.shape[-1]), device=raw.device, dtype=torch.bool
            )
            changed = torch.cat((row_zero, changed), dim=1)
        else:
            changed = torch.zeros(
                (batch_size, 1, 112 if mode == "field_age" else 1),
                device=raw.device,
                dtype=torch.bool,
            )
        last_change = torch.where(changed, positions, torch.zeros_like(positions))
        last_change = torch.cummax(last_change, dim=1).values
        age = (positions - last_change).clamp_max(FIELD_AGE_CAP).to(dtype=raw.dtype)
        encoded = torch.log1p(age)
        output[..., start:end] = (
            encoded if mode == "field_age" else encoded.expand(-1, -1, end - start)
        )
    return output


def _tl3_lead_features_numpy(raw: np.ndarray) -> np.ndarray:
    """Gather the i1 raw block at each row's most recent causal i0 price event."""
    from wnn.features import FEATURE_NAMES, event_features

    event = event_features(raw)
    event_index = FEATURE_NAMES.index("ev_i0px")
    flags = event[..., event_index] > 0.5
    flags[:, 0] = True
    rows = np.broadcast_to(np.arange(raw.shape[1]), raw.shape[:2])
    last_event = np.maximum.accumulate(np.where(flags, rows, 0), axis=1)
    return np.take_along_axis(
        raw[..., TIEDLEAD_INSTRUMENT_WIDTH:TIEDLEAD_BOOK_WIDTH],
        last_event[..., None].repeat(TL3_LEAD_WIDTH, axis=-1),
        axis=1,
    )


def _tl3_lead_features_torch(
    raw: torch.Tensor, event: torch.Tensor | None, output: torch.Tensor | None
) -> torch.Tensor:
    """Gather the i1 raw block at the latest i0 price event, before chunking."""
    import torch

    from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES, event_features_torch

    if event is None:
        event = event_features_torch(raw)
    expected_event_shape = (*raw.shape[:2], N_EVENT_FEATURES)
    if (
        event.shape != expected_event_shape
        or event.dtype != raw.dtype
        or event.device != raw.device
    ):
        raise ValueError("TL3 lead event features must match the raw sequence batch")
    flags = event[..., FEATURE_NAMES.index("ev_i0px")] > 0.5
    flags = flags.clone()
    flags[:, 0] = True
    rows = torch.arange(raw.shape[1], device=raw.device).expand(raw.shape[0], -1)
    last_event = torch.cummax(torch.where(flags, rows, torch.zeros_like(rows)), dim=1).values
    lead = torch.gather(
        raw[..., TIEDLEAD_INSTRUMENT_WIDTH:TIEDLEAD_BOOK_WIDTH],
        1,
        last_event[..., None].expand(-1, -1, TL3_LEAD_WIDTH),
    )
    if output is None:
        return lead
    if output.shape != lead.shape or output.dtype != raw.dtype or output.device != raw.device:
        raise ValueError("TL3 lead output buffer must match the input batch, dtype, and device")
    output.copy_(lead)
    return output


def _moment_projection() -> np.ndarray:
    """Build a portable balanced signed projection with seven inputs per output."""
    projection = np.zeros((112, MOMENT_PROJECTION_WIDTH), dtype=np.float32)
    divisor = np.float32(math.sqrt(112 / MOMENT_PROJECTION_WIDTH))
    for channel in range(112):
        code = (channel + 1) * 0x9E3779B1
        sign = -1.0 if code.bit_count() % 2 else 1.0
        projection[channel, channel % MOMENT_PROJECTION_WIDTH] = np.float32(sign) / divisor
    return projection


MOMENT_PROJECTION = _moment_projection()


def validate_causal_moment_prior(prior: Mapping[str, object]) -> dict[str, object]:
    """Validate a checkpoint-bound fitting prior and return its normalized form."""
    if prior.get("schema") != MOMENT_SCHEMA:
        raise ValueError("causal moment prior has an unsupported schema")
    mean = np.asarray(prior.get("mean"), dtype=np.float32)
    scale = np.asarray(prior.get("scale"), dtype=np.float32)
    if mean.shape != (MOMENT_PROJECTION_WIDTH,) or scale.shape != mean.shape:
        raise ValueError("causal moment prior must contain 16 means and scales")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("causal moment prior must be finite with positive scales")
    result = dict(prior)
    result["mean"] = mean.tolist()
    result["scale"] = scale.tolist()
    return result


def fit_causal_moment_prior(raw_sequences: Iterable[np.ndarray]) -> dict[str, object]:
    """Fit projected scale and center from caller-declared fitting-only sequences.

    Rows are sampled at a fixed stride to keep the prior pass bounded. Membership
    and source hashes are attached by the trainer that supplies the sequences.
    """
    count = 0
    mean = np.zeros(MOMENT_PROJECTION_WIDTH, dtype=np.float64)
    m2 = np.zeros(MOMENT_PROJECTION_WIDTH, dtype=np.float64)
    for raw in raw_sequences:
        values, squeeze = _raw_array(raw)
        if not squeeze:
            raise ValueError("fit one source-qualified sequence at a time")
        projected = values[0, ::MOMENT_SAMPLE_STRIDE] @ MOMENT_PROJECTION
        if projected.shape[0] == 0:
            continue
        batch_count = projected.shape[0]
        batch_mean = projected.mean(axis=0, dtype=np.float64)
        batch_m2 = np.square(projected.astype(np.float64) - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        total = count + batch_count
        mean += delta * (batch_count / total)
        m2 += batch_m2 + np.square(delta) * (count * batch_count / total)
        count = total
    if count == 0:
        raise ValueError("cannot fit a causal moment prior without sampled rows")
    variance = np.maximum(m2 / count, 0.0)
    raw_scale = np.sqrt(variance)
    positive = raw_scale[raw_scale > 0]
    relative_floor = np.median(positive) * 0.01 if positive.size else 0.0
    scale_floor = np.float32(max(float(_MOMENT_SCALE_FLOOR), float(relative_floor)))
    scale = np.maximum(raw_scale, scale_floor).astype(np.float32)
    result = {
        "schema": MOMENT_SCHEMA,
        "projection": "balanced_rademacher_16_v1",
        "mean": mean.astype(np.float32).tolist(),
        "scale": scale.tolist(),
        "sample_count": count,
        "sample_stride": MOMENT_SAMPLE_STRIDE,
        "scale_floor": float(scale_floor),
        "fast_half_life_rows": MOMENT_FAST_HALF_LIFE,
        "slow_half_life_rows": MOMENT_SLOW_HALF_LIFE,
        "prior_half_life_rows": MOMENT_PRIOR_HALF_LIFE,
        "block_rows": MOMENT_BLOCK_ROWS,
    }
    return validate_causal_moment_prior(result)


def _prior_arrays(prior: Mapping[str, object] | None) -> tuple[np.ndarray, np.ndarray]:
    if prior is None:
        raise ValueError("causal moment features require a fitting-only prior")
    checked = validate_causal_moment_prior(prior)
    return np.asarray(checked["mean"], dtype=np.float32), np.asarray(
        checked["scale"], dtype=np.float32
    )


def _causal_moment_features_numpy(
    raw: np.ndarray, mode: str, prior: Mapping[str, object] | None
) -> np.ndarray:
    prior_mean, prior_scale = _prior_arrays(prior)
    prior_second = prior_mean * prior_mean + prior_scale * prior_scale
    scale_floor = np.maximum(prior_scale * np.float32(0.1), _MOMENT_SCALE_FLOOR)
    alpha_fast = np.float32(math.exp2(-1.0 / MOMENT_FAST_HALF_LIFE))
    alpha_slow = np.float32(math.exp2(-1.0 / MOMENT_SLOW_HALF_LIFE))
    output_width = len(research_feature_names(mode))
    output = np.empty((*raw.shape[:2], output_width), dtype=np.float32)
    for batch_index, sequence in enumerate(raw):
        projected = sequence @ MOMENT_PROJECTION
        slow_mean, slow_second = prior_mean.copy(), prior_second.copy()
        fast_mean, fast_second = prior_mean.copy(), prior_second.copy()
        for row_index, value in enumerate(projected):
            previous_age = row_index
            previous_prior_weight = np.float32(math.exp2(-previous_age / MOMENT_PRIOR_HALF_LIFE))
            previous_slow_mean = (
                previous_prior_weight * prior_mean
                + (np.float32(1.0) - previous_prior_weight) * slow_mean
            )
            previous_slow_second = (
                previous_prior_weight * prior_second
                + (np.float32(1.0) - previous_prior_weight) * slow_second
            )
            previous_slow_variance = np.maximum(
                previous_slow_second - previous_slow_mean * previous_slow_mean,
                scale_floor * scale_floor,
            )
            innovation = (value - previous_slow_mean) / np.sqrt(previous_slow_variance)
            slow_mean = alpha_slow * slow_mean + (np.float32(1.0) - alpha_slow) * value
            slow_second = alpha_slow * slow_second + (np.float32(1.0) - alpha_slow) * value * value
            fast_mean = alpha_fast * fast_mean + (np.float32(1.0) - alpha_fast) * value
            fast_second = alpha_fast * fast_second + (np.float32(1.0) - alpha_fast) * value * value
            prior_weight = np.float32(math.exp2(-(row_index + 1) / MOMENT_PRIOR_HALF_LIFE))
            slow_mean_shrunk = (
                prior_weight * prior_mean + (np.float32(1.0) - prior_weight) * slow_mean
            )
            slow_second_shrunk = (
                prior_weight * prior_second + (np.float32(1.0) - prior_weight) * slow_second
            )
            fast_mean_shrunk = (
                prior_weight * prior_mean + (np.float32(1.0) - prior_weight) * fast_mean
            )
            fast_second_shrunk = (
                prior_weight * prior_second + (np.float32(1.0) - prior_weight) * fast_second
            )
            slow_variance = np.maximum(
                slow_second_shrunk - slow_mean_shrunk * slow_mean_shrunk,
                scale_floor * scale_floor,
            )
            fast_variance = np.maximum(
                fast_second_shrunk - fast_mean_shrunk * fast_mean_shrunk,
                scale_floor * scale_floor,
            )
            slow_scale = np.sqrt(slow_variance)
            fast_scale = np.sqrt(fast_variance)
            features = [
                (slow_mean_shrunk - prior_mean) / prior_scale,
                np.log(slow_scale / prior_scale),
                (fast_mean_shrunk - prior_mean) / prior_scale,
                np.log(fast_scale / prior_scale),
            ]
            if mode == "causal_innovations":
                features.append(innovation)
            output[batch_index, row_index] = np.concatenate(features)
    return output


def _ew_block_torch(
    values: torch.Tensor, start_mean: torch.Tensor, start_second: torch.Tensor, half_life: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate an EW scan in bounded blocks, without rowwise device launches."""
    import torch

    # Match the float32 streaming recurrence, including its complementary weight.
    # Mixing a rounded scan base with an unrounded (1-alpha) drifts on long paths.
    alpha = float(np.float32(math.exp2(-1.0 / half_life)))
    row = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
    base = values.new_tensor(alpha)
    forward = torch.pow(base, row)
    inverse = torch.pow(base, -row)
    next_forward = torch.pow(base, row + 1.0)
    mean = next_forward[None, :, None] * start_mean[:, None, :]
    mean = mean + (1.0 - alpha) * forward[None, :, None] * torch.cumsum(
        values * inverse[None, :, None], dim=1
    )
    second = next_forward[None, :, None] * start_second[:, None, :]
    second = second + (1.0 - alpha) * forward[None, :, None] * torch.cumsum(
        values.square() * inverse[None, :, None], dim=1
    )
    return mean, second, mean[:, -1], second[:, -1]


def _causal_moment_features_torch(
    raw: torch.Tensor,
    mode: str,
    prior: Mapping[str, object],
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    import torch

    checked = validate_causal_moment_prior(prior)
    prior_mean = torch.as_tensor(checked["mean"], device=raw.device, dtype=raw.dtype)
    prior_scale = torch.as_tensor(checked["scale"], device=raw.device, dtype=raw.dtype)
    prior_second = prior_mean.square() + prior_scale.square()
    scale_floor = torch.maximum(
        prior_scale * 0.1, raw.new_full(prior_scale.shape, float(_MOMENT_SCALE_FLOOR))
    )
    projection = torch.as_tensor(MOMENT_PROJECTION, device=raw.device, dtype=raw.dtype)
    expected_shape = (*raw.shape[:2], len(research_feature_names(mode)))
    if output is None:
        output = raw.new_empty(expected_shape)
    elif output.shape != expected_shape or output.dtype != raw.dtype or output.device != raw.device:
        raise ValueError(
            "causal moment output buffer must match the input batch, dtype, and device"
        )
    slow_mean = prior_mean.expand(raw.shape[0], -1)
    slow_second = prior_second.expand(raw.shape[0], -1)
    fast_mean, fast_second = slow_mean, slow_second
    previous_slow_mean = slow_mean
    previous_slow_variance = prior_scale.square().expand_as(slow_mean)
    for start in range(0, raw.shape[1], MOMENT_BLOCK_ROWS):
        end = min(start + MOMENT_BLOCK_ROWS, raw.shape[1])
        values = torch.matmul(raw[:, start:end], projection)
        slow_values, slow_seconds, slow_mean, slow_second = _ew_block_torch(
            values, slow_mean, slow_second, MOMENT_SLOW_HALF_LIFE
        )
        fast_values, fast_seconds, fast_mean, fast_second = _ew_block_torch(
            values, fast_mean, fast_second, MOMENT_FAST_HALF_LIFE
        )
        age = torch.arange(start + 1, end + 1, device=raw.device, dtype=raw.dtype)
        prior_weight = torch.exp2(-age / MOMENT_PRIOR_HALF_LIFE)[None, :, None]
        slow_mean_shrunk = prior_weight * prior_mean + (1.0 - prior_weight) * slow_values
        slow_second_shrunk = prior_weight * prior_second + (1.0 - prior_weight) * slow_seconds
        fast_mean_shrunk = prior_weight * prior_mean + (1.0 - prior_weight) * fast_values
        fast_second_shrunk = prior_weight * prior_second + (1.0 - prior_weight) * fast_seconds
        slow_variance = torch.maximum(
            slow_second_shrunk - slow_mean_shrunk.square(), scale_floor.square()
        )
        fast_variance = torch.maximum(
            fast_second_shrunk - fast_mean_shrunk.square(), scale_floor.square()
        )
        previous_means = torch.cat((previous_slow_mean[:, None], slow_mean_shrunk[:, :-1]), dim=1)
        previous_variances = torch.cat(
            (previous_slow_variance[:, None], slow_variance[:, :-1]), dim=1
        )
        innovation = (values - previous_means) / torch.sqrt(previous_variances)
        slow_scale = torch.sqrt(slow_variance)
        fast_scale = torch.sqrt(fast_variance)
        parts = [
            (slow_mean_shrunk - prior_mean) / prior_scale,
            torch.log(slow_scale / prior_scale),
            (fast_mean_shrunk - prior_mean) / prior_scale,
            torch.log(fast_scale / prior_scale),
        ]
        if mode == "causal_innovations":
            parts.append(innovation)
        output[:, start:end] = torch.cat(parts, dim=-1)
        previous_slow_mean = slow_mean_shrunk[:, -1]
        previous_slow_variance = slow_variance[:, -1]
    return output


def initial_causal_moment_state_torch(
    prior: Mapping[str, object],
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create the fixed-size streaming state for one or more sequence starts."""
    import torch

    mean, scale = _prior_arrays(prior)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    second = mean_tensor.square() + scale_tensor.square()
    state = torch.cat(
        (mean_tensor, second, mean_tensor, second, torch.zeros(1, device=mean_tensor.device))
    )
    return state[None].expand(batch_size, -1).clone()


def causal_moment_features_torch_step(
    raw: torch.Tensor,
    state: torch.Tensor,
    mode: str,
    prior: Mapping[str, object],
    control: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one causal feature row and the updated fixed-size streaming state."""
    import torch

    if mode not in MOMENT_FEATURE_MODES:
        raise ValueError("streaming causal moments require a causal moment feature mode")
    checked = validate_causal_moment_prior(prior)
    prior_mean = torch.as_tensor(checked["mean"], device=raw.device, dtype=raw.dtype)
    prior_scale = torch.as_tensor(checked["scale"], device=raw.device, dtype=raw.dtype)
    projection = torch.as_tensor(MOMENT_PROJECTION, device=raw.device, dtype=raw.dtype)
    scale_floor = torch.maximum(
        prior_scale * 0.1, raw.new_full(prior_scale.shape, float(_MOMENT_SCALE_FLOOR))
    )
    width = MOMENT_PROJECTION_WIDTH
    slow_mean, slow_second, fast_mean, fast_second = torch.split(
        state[:, : 4 * width], width, dim=-1
    )
    age = state[:, 4 * width]
    value = torch.matmul(raw, projection)
    prior_second = prior_mean.square() + prior_scale.square()
    previous_weight = torch.pow(raw.new_tensor(2.0), -age / MOMENT_PRIOR_HALF_LIFE)[:, None]
    previous_slow_mean = previous_weight * prior_mean + (1.0 - previous_weight) * slow_mean
    previous_slow_second = previous_weight * prior_second + (1.0 - previous_weight) * slow_second
    previous_variance = torch.maximum(
        previous_slow_second - previous_slow_mean.square(), scale_floor.square()
    )
    innovation = (value - previous_slow_mean) / torch.sqrt(previous_variance)
    alpha_slow = float(np.float32(math.exp2(-1.0 / MOMENT_SLOW_HALF_LIFE)))
    alpha_fast = float(np.float32(math.exp2(-1.0 / MOMENT_FAST_HALF_LIFE)))
    slow_mean = alpha_slow * slow_mean + (1.0 - alpha_slow) * value
    slow_second = alpha_slow * slow_second + (1.0 - alpha_slow) * value.square()
    fast_mean = alpha_fast * fast_mean + (1.0 - alpha_fast) * value
    fast_second = alpha_fast * fast_second + (1.0 - alpha_fast) * value.square()
    age = age + 1.0
    prior_weight = torch.pow(raw.new_tensor(2.0), -age / MOMENT_PRIOR_HALF_LIFE)[:, None]
    slow_mean_shrunk = prior_weight * prior_mean + (1.0 - prior_weight) * slow_mean
    slow_second_shrunk = prior_weight * prior_second + (1.0 - prior_weight) * slow_second
    fast_mean_shrunk = prior_weight * prior_mean + (1.0 - prior_weight) * fast_mean
    fast_second_shrunk = prior_weight * prior_second + (1.0 - prior_weight) * fast_second
    slow_variance = torch.maximum(
        slow_second_shrunk - slow_mean_shrunk.square(), scale_floor.square()
    )
    fast_variance = torch.maximum(
        fast_second_shrunk - fast_mean_shrunk.square(), scale_floor.square()
    )
    parts = [
        (slow_mean_shrunk - prior_mean) / prior_scale,
        torch.log(torch.sqrt(slow_variance) / prior_scale),
        (fast_mean_shrunk - prior_mean) / prior_scale,
        torch.log(torch.sqrt(fast_variance) / prior_scale),
    ]
    if mode == "causal_innovations":
        parts.append(innovation)
    features = torch.cat(parts, dim=-1)
    zero_width = research_feature_control_width(mode, control)
    if zero_width:
        features = torch.cat(
            (features[:, :-zero_width], torch.zeros_like(features[:, -zero_width:])), dim=-1
        )
    next_state = torch.cat((slow_mean, slow_second, fast_mean, fast_second, age[:, None]), dim=-1)
    return features, next_state


def research_features_numpy(
    raw: np.ndarray,
    mode: str = "none",
    prior: Mapping[str, object] | None = None,
    control: str = "none",
) -> np.ndarray:
    """Compute causal full-sequence inputs; run-start movement is excluded from sums."""
    names = research_feature_names(mode)
    zero_width = research_feature_control_width(mode, control)
    x, squeeze = _raw_array(raw)
    if not names:
        output = np.empty((*x.shape[:2], 0), dtype=np.float32)
        return output[0] if squeeze else output
    if mode == "tl3_lead":
        output = _tl3_lead_features_numpy(x)
        return output[0] if squeeze else output
    if mode in MOMENT_FEATURE_MODES:
        output = _causal_moment_features_numpy(x, mode, prior)
        if zero_width:
            output[..., -zero_width:] = 0.0
        return output[0] if squeeze else output
    if mode in FIELD_AGE_FEATURE_MODES:
        output = _field_age_features_numpy(x, mode)
        return output[0] if squeeze else output
    volume = x[..., 22:44]
    delta = np.diff(volume, axis=1, prepend=volume[:, :1])
    path = np.stack((np.abs(delta[..., :11]).sum(-1), np.abs(delta[..., 11:]).sum(-1)), -1)
    changes = np.stack(((delta[..., :11] != 0).any(-1), (delta[..., 11:] != 0).any(-1)), -1)
    if mode == "volume28":
        index = np.broadcast_to(np.arange(x.shape[1]), x.shape[:2])
        start = np.maximum.accumulate(np.where(x[..., 110] == ENDPOINT, index, 0), axis=1)
        volume = volume - np.take_along_axis(volume, start[..., None], axis=1)
        total = np.cumsum(np.concatenate((path, changes), -1), axis=1, dtype=np.float64)
        sums = total - np.take_along_axis(total, start[..., None], axis=1)
    else:
        sums = np.concatenate((path, changes), -1)
    net = np.stack((np.abs(volume[..., :11]).sum(-1), np.abs(volume[..., 11:]).sum(-1)), -1)
    output = np.concatenate((volume, sums, net), axis=-1).astype(np.float32)
    return output[0] if squeeze else output


def research_features_torch(
    raw: torch.Tensor,
    mode: str = "none",
    prior: Mapping[str, object] | None = None,
    output: torch.Tensor | None = None,
    control: str = "none",
    event: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match NumPy on full float32 raw sequences before autocast or TBPTT slicing."""
    import torch

    names = research_feature_names(mode)
    zero_width = research_feature_control_width(mode, control)
    if raw.ndim != 3 or raw.shape[-1] != 112 or raw.shape[1] < 1:
        raise ValueError("Expected nonempty (B,T,112) raw features")
    if raw.dtype != torch.float32:
        raise ValueError("Extract endpoint features from original float32 inputs before autocast")
    if not names:
        return raw.new_empty((*raw.shape[:2], 0))
    if mode == "tl3_lead":
        return _tl3_lead_features_torch(raw, event, output)
    if mode in MOMENT_FEATURE_MODES:
        if prior is None:
            raise ValueError("causal moment features require a fitting-only prior")
        features = _causal_moment_features_torch(raw, mode, prior, output)
        if zero_width:
            features[..., -zero_width:].zero_()
        return features
    if mode in FIELD_AGE_FEATURE_MODES:
        return _field_age_features_torch(raw, mode, output)
    volume = raw[..., 22:44]
    delta = torch.diff(volume, dim=1, prepend=volume[:, :1])
    path = torch.stack((delta[..., :11].abs().sum(-1), delta[..., 11:].abs().sum(-1)), -1)
    changes = torch.stack(((delta[..., :11] != 0).any(-1), (delta[..., 11:] != 0).any(-1)), -1)
    if mode == "volume28":
        index = torch.arange(raw.shape[1], device=raw.device).expand(raw.shape[:2])
        start = torch.cummax(torch.where(raw[..., 110] == float(ENDPOINT), index, 0), dim=1).values
        volume = volume - torch.gather(volume, 1, start[..., None].expand(*start.shape, 22))
        total = torch.cumsum(
            torch.cat((path, changes.to(raw.dtype)), -1), dim=1, dtype=torch.float64
        )
        sums = total - torch.gather(total, 1, start[..., None].expand(*start.shape, 4))
    else:
        sums = torch.cat((path, changes.to(raw.dtype)), -1)
    net = torch.stack((volume[..., :11].abs().sum(-1), volume[..., 11:].abs().sum(-1)), -1)
    return torch.cat((volume, sums.to(raw.dtype), net), -1)


class StreamingCausalMoments:
    """O(1)-state causal projected moments and standardized innovations."""

    def __init__(self, mode: str, prior: Mapping[str, object], control: str = "none") -> None:
        if mode not in MOMENT_FEATURE_MODES:
            raise ValueError(f"Unknown causal moment feature mode: {mode}")
        self.mode = mode
        self.control = control
        self.zero_width = research_feature_control_width(mode, control)
        self.prior = validate_causal_moment_prior(prior)
        self.prior_mean, self.prior_scale = _prior_arrays(self.prior)
        self.scale_floor = np.maximum(self.prior_scale * np.float32(0.1), _MOMENT_SCALE_FLOOR)
        self.reset()

    def reset(self) -> None:
        """Restore prior state and clear the per-sequence row clock."""
        prior_second = self.prior_mean * self.prior_mean + self.prior_scale * self.prior_scale
        self.slow_mean = self.prior_mean.copy()
        self.slow_second = prior_second.copy()
        self.fast_mean = self.prior_mean.copy()
        self.fast_second = prior_second.copy()
        self.rows = 0

    def update(self, row: np.ndarray) -> np.ndarray:
        """Use the current raw row and earlier rows, then advance one moment step."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (112,) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite float32 raw row of width112")
        value = raw @ MOMENT_PROJECTION
        prior_second = self.prior_mean * self.prior_mean + self.prior_scale * self.prior_scale
        prior_weight = np.float32(math.exp2(-self.rows / MOMENT_PRIOR_HALF_LIFE))
        previous_mean = (
            prior_weight * self.prior_mean + (np.float32(1.0) - prior_weight) * self.slow_mean
        )
        previous_second = (
            prior_weight * prior_second + (np.float32(1.0) - prior_weight) * self.slow_second
        )
        previous_variance = np.maximum(
            previous_second - previous_mean * previous_mean, self.scale_floor * self.scale_floor
        )
        innovation = (value - previous_mean) / np.sqrt(previous_variance)
        alpha_slow = np.float32(math.exp2(-1.0 / MOMENT_SLOW_HALF_LIFE))
        alpha_fast = np.float32(math.exp2(-1.0 / MOMENT_FAST_HALF_LIFE))
        self.slow_mean = alpha_slow * self.slow_mean + (np.float32(1.0) - alpha_slow) * value
        self.slow_second = (
            alpha_slow * self.slow_second + (np.float32(1.0) - alpha_slow) * value * value
        )
        self.fast_mean = alpha_fast * self.fast_mean + (np.float32(1.0) - alpha_fast) * value
        self.fast_second = (
            alpha_fast * self.fast_second + (np.float32(1.0) - alpha_fast) * value * value
        )
        self.rows += 1
        prior_weight = np.float32(math.exp2(-self.rows / MOMENT_PRIOR_HALF_LIFE))
        slow_mean = (
            prior_weight * self.prior_mean + (np.float32(1.0) - prior_weight) * self.slow_mean
        )
        slow_second = (
            prior_weight * prior_second + (np.float32(1.0) - prior_weight) * self.slow_second
        )
        fast_mean = (
            prior_weight * self.prior_mean + (np.float32(1.0) - prior_weight) * self.fast_mean
        )
        fast_second = (
            prior_weight * prior_second + (np.float32(1.0) - prior_weight) * self.fast_second
        )
        slow_scale = np.sqrt(
            np.maximum(slow_second - slow_mean * slow_mean, self.scale_floor * self.scale_floor)
        )
        fast_scale = np.sqrt(
            np.maximum(fast_second - fast_mean * fast_mean, self.scale_floor * self.scale_floor)
        )
        parts = [
            (slow_mean - self.prior_mean) / self.prior_scale,
            np.log(slow_scale / self.prior_scale),
            (fast_mean - self.prior_mean) / self.prior_scale,
            np.log(fast_scale / self.prior_scale),
        ]
        if self.mode == "causal_innovations":
            parts.append(innovation)
        features = np.concatenate(parts).astype(np.float32)
        if self.zero_width:
            features[-self.zero_width :] = 0.0
        return features


class StreamingFieldAgeFeatures:
    """Track exact field or group change ages across arbitrary input chunks."""

    def __init__(self, mode: str) -> None:
        if mode not in FIELD_AGE_FEATURE_MODES:
            raise ValueError(f"Unknown field-age feature mode: {mode}")
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        """Clear the previous row and return every age to sequence-start zero."""
        self.previous: np.ndarray | None = None
        self.ages = np.zeros(112, dtype=np.int32)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Encode one current row using only exact changes from its predecessor."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (112,) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite float32 raw row of width112")
        if self.previous is None:
            self.ages.fill(0)
        elif self.mode == "field_age":
            changed = raw != self.previous
            self.ages[changed] = 0
            self.ages[~changed] = np.minimum(self.ages[~changed] + 1, FIELD_AGE_CAP)
        else:
            for start, end in RAW_FIELD_GROUPS:
                if np.any(raw[start:end] != self.previous[start:end]):
                    self.ages[start:end] = 0
                else:
                    self.ages[start:end] = np.minimum(self.ages[start:end] + 1, FIELD_AGE_CAP)
        self.previous = raw.copy()
        return np.log1p(self.ages.astype(np.float32, copy=False))

    def update_chunk(self, rows: np.ndarray) -> np.ndarray:
        """Process a row chunk while retaining state for the following chunk."""
        raw = np.asarray(rows, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 112 or not np.isfinite(raw).all():
            raise ValueError("Expected finite raw rows with shape (T,112)")
        if not len(raw):
            return np.empty((0, 112), dtype=np.float32)
        return np.stack([self.update(row) for row in raw])


class StreamingTL3Lead:
    """Retain the latest causal i0 price-event snapshot of the i1 raw block."""

    def __init__(self) -> None:
        from wnn.data import I0, I1

        self._event_columns = (I0.bid_px.start, I0.ask_px.start)
        self._i1_columns = slice(I1.bid_px.start, I1.trade_vol.stop)
        self.reset()

    def reset(self) -> None:
        """Clear the previous raw row and lead snapshot at a sequence boundary."""
        self.previous: np.ndarray | None = None
        self.lead: np.ndarray | None = None

    def update(self, row: np.ndarray) -> np.ndarray:
        """Use this row's i0 event flag, then return the current causal lead snapshot."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (112,) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite float32 raw row of width112")
        is_event = self.previous is None or all(
            raw[column] != self.previous[column] for column in self._event_columns
        )
        if is_event:
            self.lead = raw[self._i1_columns].copy()
        self.previous = raw.copy()
        if self.lead is None:
            raise RuntimeError("TL3 lead snapshot was not initialized at sequence start")
        return self.lead.copy()

    def update_chunk(self, rows: np.ndarray) -> np.ndarray:
        """Process a row chunk while carrying the event snapshot into the next chunk."""
        raw = np.asarray(rows, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 112 or not np.isfinite(raw).all():
            raise ValueError("Expected finite raw rows with shape (T,112)")
        if not len(raw):
            return np.empty((0, TL3_LEAD_WIDTH), dtype=np.float32)
        return np.stack([self.update(row) for row in raw])


class StreamingResearchFeatures:
    """Keep O(1) per-sequence state for supported causal research feature modes."""

    def __init__(
        self,
        mode: str = "none",
        prior: Mapping[str, object] | None = None,
        control: str = "none",
    ) -> None:
        self.mode = mode
        self.control = control
        research_feature_control_width(mode, control)
        self.names = research_feature_names(mode)
        self.causal_moments = (
            StreamingCausalMoments(mode, prior, control)
            if mode in MOMENT_FEATURE_MODES and prior is not None
            else None
        )
        self.field_ages = (
            StreamingFieldAgeFeatures(mode) if mode in FIELD_AGE_FEATURE_MODES else None
        )
        self.tl3_lead = StreamingTL3Lead() if mode == "tl3_lead" else None
        if mode in MOMENT_FEATURE_MODES and self.causal_moments is None:
            raise ValueError("causal moment features require a fitting-only prior")
        self.reset()

    def reset(self) -> None:
        """Clear every volume accumulator at a sequence boundary."""
        if self.causal_moments is not None:
            self.causal_moments.reset()
            return
        if self.field_ages is not None:
            self.field_ages.reset()
            return
        if self.tl3_lead is not None:
            self.tl3_lead.reset()
            return
        self.previous: np.ndarray | None = None
        self.origin: np.ndarray | None = None
        self.sums = np.zeros(4, dtype=np.float64)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Read current/past raw input only and return owned float32 output."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (112,) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite float32 raw row of width112")
        if self.causal_moments is not None:
            return self.causal_moments.update(raw)
        if self.field_ages is not None:
            return self.field_ages.update(raw)
        if self.tl3_lead is not None:
            return self.tl3_lead.update(raw)
        if not self.names:
            return np.empty(0, dtype=np.float32)
        volume = raw[22:44]
        delta = np.zeros(22, dtype=np.float32) if self.previous is None else volume - self.previous
        path = np.array([np.abs(delta[:11]).sum(), np.abs(delta[11:]).sum()], dtype=np.float32)
        changes = np.array([(delta[:11] != 0).any(), (delta[11:] != 0).any()], dtype=np.float32)
        if self.origin is None or raw[110] == ENDPOINT:
            self.origin = volume.copy()
            self.sums.fill(0)
        else:
            self.sums += np.concatenate((path, changes))
        current = volume - self.origin if self.mode == "volume28" else volume
        sums = self.sums if self.mode == "volume28" else np.concatenate((path, changes))
        net = np.array([np.abs(current[:11]).sum(), np.abs(current[11:]).sum()])
        self.previous = volume.copy()
        return np.concatenate((current, sums, net)).astype(np.float32)
