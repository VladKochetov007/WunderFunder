from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from wnn.nn import RNNModel

RAW_CHANNELS = (0, 3, 52, 55)
FEATURES_PER_CHANNEL = 4
VARIANCE_FLOOR = 1e-6
ARM_CODES = {"fixed": 0, "learned": 1}


class CausalLocalTrendKalman(nn.Module):
    """Independent two-state local-trend filters with learned or fixed positive Q/R."""

    def __init__(self, arm: str = "learned") -> None:
        super().__init__()
        if arm not in ARM_CODES:
            raise ValueError(f"unknown Kalman arm: {arm}")
        self.arm = arm
        self.register_buffer("channel_indices", torch.tensor(RAW_CHANNELS, dtype=torch.long))
        self.register_buffer("arm_code", torch.tensor(ARM_CODES[arm], dtype=torch.int8))
        initial = (0.01, 0.001, 0.1)
        raw = torch.tensor([math.log(math.expm1(value)) for value in initial], dtype=torch.float32)
        values = (
            raw[0].repeat(len(RAW_CHANNELS)),
            raw[1].repeat(len(RAW_CHANNELS)),
            raw[2].repeat(len(RAW_CHANNELS)),
        )
        for name, value in zip(("log_q_level", "log_q_slope", "log_r"), values, strict=True):
            if arm == "learned":
                self.register_parameter(name, nn.Parameter(value.clone()))
            else:
                self.register_buffer(name, value.clone())

    def forward(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return causal level/slope/innovation features and state for the next chunk."""
        if observations.ndim != 3 or observations.shape[1] == 0:
            raise ValueError("Kalman input must be a nonempty (batch, time, features) tensor")
        if observations.shape[-1] <= max(RAW_CHANNELS):
            raise ValueError("Kalman input does not contain the selected raw channels")
        values = observations.index_select(-1, self.channel_indices)
        batch = values.shape[0]
        if state is None:
            mean = torch.stack((values[:, 0], torch.zeros_like(values[:, 0])), dim=-1)
            p00 = values.new_ones((batch, len(RAW_CHANNELS)))
            p01 = values.new_zeros((batch, len(RAW_CHANNELS)))
            p11 = values.new_ones((batch, len(RAW_CHANNELS)))
        else:
            mean, p00, p01, p11 = state
        q_level = F.softplus(self.log_q_level).unsqueeze(0) + VARIANCE_FLOOR
        q_slope = F.softplus(self.log_q_slope).unsqueeze(0) + VARIANCE_FLOOR
        measurement = F.softplus(self.log_r).unsqueeze(0) + VARIANCE_FLOOR
        outputs: list[torch.Tensor] = []
        for value in values.unbind(dim=1):
            level, slope = mean.unbind(dim=-1)
            prior_level = level + slope
            prior_slope = slope
            prior_p00 = p00 + 2.0 * p01 + p11 + q_level
            prior_p01 = p01 + p11
            prior_p11 = p11 + q_slope
            innovation_variance = (prior_p00 + measurement).clamp_min(VARIANCE_FLOOR)
            gain_level = prior_p00 / innovation_variance
            gain_slope = prior_p01 / innovation_variance
            innovation = value - prior_level
            level = prior_level + gain_level * innovation
            slope = prior_slope + gain_slope * innovation
            complement = 1.0 - gain_level
            p00 = complement.square() * prior_p00 + gain_level.square() * measurement
            p01 = complement * (prior_p01 - gain_slope * prior_p00) + (
                gain_level * gain_slope * measurement
            )
            p11 = (
                prior_p11
                - 2.0 * gain_slope * prior_p01
                + gain_slope.square() * prior_p00
                + gain_slope.square() * measurement
            )
            mean = torch.stack((level, slope), dim=-1)
            outputs.append(
                torch.stack(
                    (
                        level,
                        slope,
                        innovation / innovation_variance.sqrt(),
                        innovation_variance.log(),
                    ),
                    dim=-1,
                )
            )
        return torch.stack(outputs, dim=1).flatten(start_dim=-2), (mean, p00, p01, p11)


class KalmanTeacherGRU(nn.Module):
    """Canonical-GRU teacher with four causal local-trend feature channels appended."""

    def __init__(self, n_in: int, n_out: int, args: dict[str, object], arm: str) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.kalman = CausalLocalTrendKalman(arm)
        self.core = RNNModel(
            n_in + len(RAW_CHANNELS) * FEATURES_PER_CHANNEL,
            int(args.get("hidden", 256)),
            int(args.get("layers", 2)),
            float(args.get("dropout", 0.0)),
            n_out=n_out,
            readout_activation=str(args.get("readout_activation", "gelu")),
            encoder_width=args.get("encoder_width"),
            recurrent_width=args.get("recurrent_width"),
            readout_width=args.get("readout_width"),
            head_layers=int(args.get("head_layers", 1)),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Normalize with fit-only global statistics, update the filter, and run the GRU."""
        normalized = (inputs - self.mean) / self.std
        core_state = None if state is None else state[0]
        filter_state = None if state is None else (state[1], state[2], state[3], state[4])
        features, next_filter_state = self.kalman(normalized, filter_state)
        output, next_core_state = self.core(torch.cat((normalized, features), dim=-1), core_state)
        return output, (next_core_state, *next_filter_state)


def make_kalman_model(
    args_dict: dict[str, object], n_in: int, n_out: int, arm: str = "learned"
) -> KalmanTeacherGRU:
    """Build the narrow dense-GRU model supported by this experiment adapter."""
    unsupported = (
        args_dict.get("arch", "gru") != "gru"
        or args_dict.get("cell", "gru") != "gru"
        or args_dict.get("enc", "linear") != "linear"
        or args_dict.get("numeric_map", "none") != "none"
        or args_dict.get("rnn_init", "default") != "default"
        or bool(args_dict.get("init", ""))
    )
    if unsupported:
        raise ValueError("Kalman adapter supports only a fresh dense plain-GRU recipe")
    return KalmanTeacherGRU(n_in, n_out, args_dict, arm)


def arm_from_state_dict(state: dict[str, torch.Tensor]) -> str:
    """Recover the fixed/learned arm identifier from a saved inference state."""
    code = int(state["kalman.arm_code"].item())
    if code not in (ARM_CODES["fixed"], ARM_CODES["learned"]):
        raise ValueError(f"unknown Kalman checkpoint arm code: {code}")
    return "learned" if code == ARM_CODES["learned"] else "fixed"


def make_kalman_checkpoint_model(
    args_dict: dict[str, object], state: dict[str, torch.Tensor], n_in: int
) -> KalmanTeacherGRU:
    """Rebuild the adapter model from checkpoint-owned arm and output dimensions."""
    output_weight = state["core.head.2.weight"]
    model = make_kalman_model(args_dict, n_in, output_weight.shape[0], arm_from_state_dict(state))
    model.load_state_dict(state)
    return model
