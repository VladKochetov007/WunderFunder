from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from wnn.data import N_FEATURES

SENTINEL = np.float32(-5.19934)
SENTINEL_TOLERANCE = np.float32(1e-4)
VALUE_MODES = ("unchanged", "neutral", "safe")
INDICATOR_MODES = ("none", "missing", "current")
SENTINEL_ARMS = (
    ("unchanged", "none"),
    ("unchanged", "current"),
    ("unchanged", "missing"),
    ("neutral", "none"),
    ("neutral", "missing"),
    ("safe", "none"),
    ("safe", "missing"),
    ("safe", "current"),
)


def canonical_hash(value: dict) -> str:
    """Bind JSON calibration contents independently of file formatting or location."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def missing_numpy(raw: np.ndarray) -> np.ndarray:
    """Recognize the fixed sentinel with absolute tolerance only, never relative tolerance."""
    return np.abs(raw - SENTINEL) < SENTINEL_TOLERANCE


def validate_metadata(metadata: dict) -> None:
    """Reject incompatible preprocessing or altered frozen calibration constants."""
    if (
        metadata.get("version") != 1
        or metadata.get("value_mode") not in VALUE_MODES
        or metadata.get("indicators") not in INDICATOR_MODES
        or metadata.get("base_width", 0) < N_FEATURES
    ):
        raise ValueError("invalid sentinel preprocessing modes or input width")
    calibration = metadata["calibration"]
    if canonical_hash(calibration) != metadata.get("calibration_sha256"):
        raise ValueError("sentinel calibration content hash differs")
    if calibration["sentinel"] != float(SENTINEL) or calibration["absolute_tolerance"] != float(
        SENTINEL_TOLERANCE
    ):
        raise ValueError("sentinel recognition constants differ")
    for name in ("parent_mean", "parent_std", "safe_mean", "safe_std"):
        value = np.asarray(calibration[name], dtype=np.float32)
        if value.shape != (N_FEATURES,) or not np.isfinite(value).all():
            raise ValueError(f"invalid frozen sentinel array: {name}")
        if name.endswith("std") and (value <= 0).any():
            raise ValueError("sentinel standard deviations must be positive")


class SentinelPreprocessor:
    """Stateless raw-level intervention; all history features must use original raw rows."""

    def __init__(self, metadata: dict) -> None:
        validate_metadata(metadata)
        self.metadata = copy.deepcopy(metadata)
        self.value_mode = metadata["value_mode"]
        self.indicators = metadata["indicators"]
        self.constants = {
            key: np.asarray(metadata["calibration"][key], np.float32)
            for key in ("parent_mean", "parent_std", "safe_mean", "safe_std")
        }
        self._torch_constants: dict[str, dict[str, torch.Tensor]] = {}

    def numpy(self, inputs: np.ndarray) -> np.ndarray:
        """Transform a row or batch of already-built base inputs, preserving history columns."""
        values = np.asarray(inputs, dtype=np.float32)
        if values.shape[-1] != self.metadata["base_width"]:
            raise ValueError("sentinel base input width differs")
        raw = values[..., :N_FEATURES]
        missing = missing_numpy(raw)
        mean, std = self.constants["parent_mean"], self.constants["parent_std"]
        levels = raw
        if self.value_mode == "safe":
            z = (raw - self.constants["safe_mean"]) / self.constants["safe_std"]
            levels = mean + std * z
        if self.value_mode != "unchanged":
            levels = np.where(missing, mean, levels)
        output = np.concatenate((levels, values[..., N_FEATURES:]), axis=-1)
        if self.indicators != "none":
            extra = (
                missing.astype(np.float32)
                if self.indicators == "missing"
                else (levels - mean) / std
            )
            output = np.concatenate((output, extra), axis=-1)
        return np.ascontiguousarray(output, dtype=np.float32)

    def torch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Torch twin; preprocessing runs in FP32 before model autocast."""
        if inputs.dtype != torch.float32 or inputs.shape[-1] != self.metadata["base_width"]:
            raise ValueError("sentinel inputs must be FP32 with the declared base width")
        device = str(inputs.device)
        if device not in self._torch_constants:
            self._torch_constants[device] = {
                key: torch.tensor(value, device=inputs.device)
                for key, value in self.constants.items()
            }
        constants = self._torch_constants[device]
        raw = inputs[..., :N_FEATURES]
        missing = (raw - float(SENTINEL)).abs() < float(SENTINEL_TOLERANCE)
        mean, std = constants["parent_mean"], constants["parent_std"]
        levels = raw
        if self.value_mode == "safe":
            z = (raw - constants["safe_mean"]) / constants["safe_std"]
            levels = mean + std * z
        if self.value_mode != "unchanged":
            levels = torch.where(missing, mean, levels)
        output = torch.cat((levels, inputs[..., N_FEATURES:]), dim=-1)
        if self.indicators != "none":
            extra = missing.float() if self.indicators == "missing" else (levels - mean) / std
            output = torch.cat((output, extra), dim=-1)
        return output

    def update(self, inputs: np.ndarray) -> np.ndarray:
        """Process one built input row; there is no sequence-level calibration state."""
        if np.asarray(inputs).ndim != 1:
            raise ValueError("streaming sentinel preprocessing expects one row")
        return self.numpy(inputs)

    def reset(self) -> None:
        """No-op: sequence reset never changes frozen preprocessing constants."""


class SentinelInputProjection(nn.Module):
    """Keep parent projection arithmetic separate from zero-initialized new columns."""

    def __init__(self, source: nn.Linear, base_width: int) -> None:
        super().__init__()
        if source.in_features != base_width + N_FEATURES:
            raise ValueError("indicator projection must add exactly 112 columns")
        self.weight, self.bias = source.weight, source.bias
        self.in_features, self.out_features = source.in_features, source.out_features
        self.base_width = base_width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        parent = functional.linear(
            inputs[..., : self.base_width].contiguous(),
            self.weight[:, : self.base_width].contiguous(),
            self.bias,
        )
        added = functional.linear(
            inputs[..., self.base_width :].contiguous(),
            self.weight[:, self.base_width :].contiguous(),
        )
        return parent + added


def initialize_sentinel_checkpoint(
    checkpoint: dict, calibration: dict, value_mode: str, indicators: str
) -> dict:
    """Create an unfitted arm with frozen parent normalization and zero new weights."""
    base_width = checkpoint["n_in"]
    metadata = {
        "version": 1,
        "value_mode": value_mode,
        "indicators": indicators,
        "base_width": base_width,
        "calibration": copy.deepcopy(calibration),
        "calibration_sha256": canonical_hash(calibration),
    }
    validate_metadata(metadata)
    result = copy.deepcopy(checkpoint)
    state = result["model"]
    for key in ("mean", "std"):
        expected = torch.tensor(calibration[f"parent_{key}"], dtype=torch.float32)
        if not torch.equal(state[key][:N_FEATURES].cpu(), expected):
            raise ValueError("parent normalization disagrees with calibration")
    if indicators != "none":
        weight = state["inp.weight"]
        state["inp.weight"] = torch.cat(
            (weight, weight.new_zeros(weight.shape[0], N_FEATURES)), dim=1
        )
        state["mean"] = torch.cat((state["mean"], state["mean"].new_zeros(N_FEATURES)))
        state["std"] = torch.cat((state["std"], state["std"].new_ones(N_FEATURES)))
        result["n_in"] += N_FEATURES
    result["args"].update(
        sentinel_values=value_mode, sentinel_indicators=indicators, sentinel_base_width=base_width
    )
    result["sentinel_preprocessing"] = metadata
    result["provenance"] = {
        **result.get("provenance", {}),
        "sentinel_initialization": {
            "parent_sha256": calibration["parent_sha256"],
            "calibration_sha256": metadata["calibration_sha256"],
            "training_updates": 0,
            "raw_level_only": True,
        },
    }
    return result
