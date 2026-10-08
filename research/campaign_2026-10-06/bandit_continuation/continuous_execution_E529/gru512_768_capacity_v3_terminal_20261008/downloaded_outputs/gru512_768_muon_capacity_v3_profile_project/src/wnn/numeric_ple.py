"""Feature-wise numeric bases for the E540 matched input-map screen."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from wnn.data import N_FEATURES


class NumericFeatureMap(nn.Module):
    """Expand raw scalar channels into matched linear or piecewise-linear slots.

    The extra input channels pass through unchanged. Both modes have the same
    output shape and feed the same learned outer projection. In ``linear`` mode
    each raw scalar occupies only its bypass slot; in ``ple`` mode the fixed
    training quantiles also generate bounded cumulative ramps.
    """

    MODES = ("linear", "ple")

    def __init__(self, input_width: int, mode: str, n_intervals: int = 8) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"numeric map mode must be one of {self.MODES}")
        if input_width < N_FEATURES:
            raise ValueError("numeric maps require the raw 112 input channels")
        if n_intervals < 1:
            raise ValueError("n_intervals must be positive")
        self.input_width = input_width
        self.mode = mode
        self.n_intervals = n_intervals
        self.extra_width = input_width - N_FEATURES
        self.output_width = N_FEATURES * (n_intervals + 1) + self.extra_width
        self.register_buffer("knots", torch.zeros(N_FEATURES, n_intervals + 1, dtype=torch.float32))
        self.register_buffer("knot_counts", torch.zeros(N_FEATURES, dtype=torch.int64))
        self.register_buffer("knots_fitted", torch.tensor(False, dtype=torch.bool))
        self._knots_installed = False
        self.register_load_state_dict_post_hook(self._sync_fit_flag_after_load)

    @staticmethod
    def _sync_fit_flag_after_load(module: NumericFeatureMap, incompatible_keys: object) -> None:
        """Restore the Python guard once after checkpoint buffers are loaded."""
        del incompatible_keys
        module._knots_installed = bool(module.knots_fitted.detach().cpu().item())

    @torch.no_grad()
    def set_knots(self, knots: np.ndarray, counts: np.ndarray) -> None:
        """Install validated training-only knots into checkpointed buffers."""
        values = np.asarray(knots, dtype=np.float32)
        lengths = np.asarray(counts, dtype=np.int64)
        expected = (N_FEATURES, self.n_intervals + 1)
        if values.shape != expected or lengths.shape != (N_FEATURES,):
            raise ValueError("knot arrays have the wrong feature or interval shape")
        if not np.isfinite(values).all() or np.any(lengths < 1):
            raise ValueError("knot arrays must be finite and every feature needs one value")
        if np.any(lengths > self.n_intervals + 1):
            raise ValueError("a feature has more knots than the configured interval count")
        for feature, count in enumerate(lengths.tolist()):
            if count > 1 and not np.all(np.diff(values[feature, :count]) > 0):
                raise ValueError("active knots must be strictly increasing")
            if count < self.n_intervals + 1 and not np.all(
                values[feature, count:] == values[feature, count - 1]
            ):
                raise ValueError("unused knot slots must repeat the last active knot")
        self.knots.copy_(torch.as_tensor(values, device=self.knots.device))
        self.knot_counts.copy_(torch.as_tensor(lengths, device=self.knot_counts.device))
        self.knots_fitted.fill_(True)
        self._knots_installed = True

    def forward(self, inputs: Tensor) -> Tensor:
        """Return equal-width feature slots while preserving unbounded raw values."""
        if inputs.shape[-1] != self.input_width:
            raise ValueError(
                f"expected {self.input_width} base inputs, received {inputs.shape[-1]}"
            )
        if self.mode == "ple" and not self._knots_installed:
            raise RuntimeError("PLE knots must be fitted from the declared training membership")
        raw = inputs[..., :N_FEATURES]
        if self.mode == "linear":
            ramps = raw.new_zeros((*raw.shape, self.n_intervals))
        else:
            lower = self.knots[:, :-1].to(dtype=raw.dtype)
            upper = self.knots[:, 1:].to(dtype=raw.dtype)
            denominator = (upper - lower).clamp_min(torch.finfo(raw.dtype).eps)
            ramps = ((raw.unsqueeze(-1) - lower) / denominator).clamp(0.0, 1.0)
            active = torch.arange(self.n_intervals, device=raw.device).unsqueeze(0)
            active = active < (self.knot_counts - 1).unsqueeze(1)
            ramps = ramps * active.to(dtype=raw.dtype)
        slots = torch.cat((ramps, raw.unsqueeze(-1)), dim=-1)
        expanded = slots.reshape(*raw.shape[:-1], N_FEATURES * (self.n_intervals + 1))
        if self.extra_width == 0:
            return expanded
        return torch.cat((expanded, inputs[..., N_FEATURES:]), dim=-1)


def fit_training_quantile_knots(
    sequences: Iterable[np.ndarray],
    fit_members: Sequence[tuple[str, int]],
    mean: np.ndarray,
    std: np.ndarray,
    n_intervals: int = 8,
    samples_per_sequence: int = 128,
    max_samples: int = 65_536,
    seed: int = 20_261_008,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit bounded quantiles from explicitly supplied fitting sequences only.

    Each sequence contributes evenly spaced rows before a fixed-size reservoir
    bounds memory. The caller must provide the exact training membership and
    must not pass evaluated sequences or labels.
    """
    members = tuple((str(split), int(group)) for split, group in fit_members)
    if (
        not members
        or len(set(members)) != len(members)
        or any(split not in ("train", "valid") or group < 0 for split, group in members)
    ):
        raise ValueError("fit members must be unique train/valid row groups")
    if n_intervals < 1 or samples_per_sequence < 1 or max_samples < 1:
        raise ValueError("interval and sampling limits must be positive")
    center = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    if center.shape != (N_FEATURES,) or scale.shape != (N_FEATURES,):
        raise ValueError("training normalization must cover exactly the raw channels")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("training normalization must be finite with positive scales")

    rng = np.random.default_rng(seed)
    reservoir = np.empty((max_samples, N_FEATURES), dtype=np.float32)
    kept = 0
    candidates = 0
    observed_rows = 0
    for _, sequence in zip(members, sequences, strict=True):
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != N_FEATURES or values.shape[0] < 1:
            raise ValueError("each fitting sequence must have nonempty (rows,112) features")
        if not np.isfinite(values).all():
            raise ValueError("fitting features must follow the repository's finite sentinel policy")
        observed_rows += values.shape[0]
        count = min(samples_per_sequence, values.shape[0])
        positions = np.linspace(0, values.shape[0] - 1, count, dtype=np.int64)
        sample = (values[positions] - center) / scale
        if not np.isfinite(sample).all():
            raise ValueError("standardized fitting samples must remain finite")
        for row in sample:
            if kept < max_samples:
                reservoir[kept] = row
                kept += 1
            else:
                replacement = int(rng.integers(candidates + 1))
                if replacement < max_samples:
                    reservoir[replacement] = row
            candidates += 1
    if kept == 0:
        raise ValueError("no training rows were available for knot fitting")
    quantile_levels = np.linspace(0.0, 1.0, n_intervals + 1)
    knots = np.zeros((N_FEATURES, n_intervals + 1), dtype=np.float32)
    counts = np.ones(N_FEATURES, dtype=np.int64)
    fit_values = reservoir[:kept]
    for feature in range(N_FEATURES):
        quantiles = np.quantile(fit_values[:, feature], quantile_levels, method="linear")
        unique = np.unique(np.asarray(quantiles, dtype=np.float32))
        counts[feature] = len(unique)
        knots[feature, : len(unique)] = unique
        knots[feature, len(unique) :] = unique[-1]

    membership_bytes = json.dumps(sorted(members), separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    membership_digest = hashlib.sha256(membership_bytes).hexdigest()
    knot_digest = hashlib.sha256(
        knots.astype("<f4", copy=False).tobytes() + counts.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    metadata: dict[str, object] = {
        "version": 1,
        "coordinate_system": "raw_feature_train_mean_std",
        "n_intervals": n_intervals,
        "max_knots_per_feature": n_intervals + 1,
        "samples_per_sequence": samples_per_sequence,
        "max_sample_rows": max_samples,
        "candidate_sample_rows": candidates,
        "retained_sample_rows": kept,
        "observed_fitting_rows": observed_rows,
        "fit_member_count": len(members),
        "fit_members_sha256": membership_digest,
        "seed": seed,
        "knot_sha256": knot_digest,
        "duplicate_quantiles": "dropped per feature; constant channels use zero ramps and keep raw bypass",
        "sentinel_policy": "all feature values must be finite; finite sentinel levels are ordinary values",
    }
    return knots, counts, metadata
