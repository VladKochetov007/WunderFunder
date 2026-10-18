from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.nn import RNNModel

PILOT_CHANNELS = (0, 3, 52, 55)
ALL_CHANNELS = tuple(range(112))
FEATURES_PER_CHANNEL = 4
VARIANCE_FLOOR = 1e-6
ARM_CODES = {"fixed": 0, "learned": 1, "ema": 2, "raw": 3}
FIXED_Q_LEVEL = 0.01
FIXED_Q_SLOPE = 0.001
FIXED_R = 0.1
EMA_ALPHA = (
    -(FIXED_Q_LEVEL / FIXED_R)
    + math.sqrt((FIXED_Q_LEVEL / FIXED_R) ** 2 + 4.0 * FIXED_Q_LEVEL / FIXED_R)
) / 2.0


def channels_for_scope(scope: str) -> tuple[int, ...]:
    """Resolve the explicitly versioned four-channel pilot or full input scope."""
    if scope == "pilot4":
        return PILOT_CHANNELS
    if scope == "all112":
        return ALL_CHANNELS
    raise ValueError(f"unknown Kalman channel scope: {scope}")


class CausalLocalTrendKalman(nn.Module):
    """Independent two-state local-trend filters with learned or fixed positive Q/R."""

    def __init__(self, arm: str = "learned", channels: tuple[int, ...] = PILOT_CHANNELS) -> None:
        super().__init__()
        if arm not in ("fixed", "learned"):
            raise ValueError(f"unknown Kalman arm: {arm}")
        self.arm = arm
        if not channels or tuple(sorted(set(channels))) != channels or channels[-1] >= 112:
            raise ValueError("channels must be a nonempty, sorted, unique subset of 0..111")
        self.max_channel = channels[-1]
        self.register_buffer("channel_indices", torch.tensor(channels, dtype=torch.long))
        self.register_buffer("arm_code", torch.tensor(ARM_CODES[arm], dtype=torch.int8))
        initial = (FIXED_Q_LEVEL, FIXED_Q_SLOPE, FIXED_R)
        # One scalar per state family is tied across the selected raw channels.
        values = tuple(
            torch.tensor([math.log(math.expm1(value))], dtype=torch.float32) for value in initial
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
        if observations.shape[-1] <= self.max_channel:
            raise ValueError("Kalman input does not contain the selected raw channels")
        values = observations.index_select(-1, self.channel_indices)
        batch = values.shape[0]
        if state is None:
            mean = torch.stack((values[:, 0], torch.zeros_like(values[:, 0])), dim=-1)
            p00 = values.new_ones((batch, values.shape[-1]))
            p01 = values.new_zeros((batch, values.shape[-1]))
            p11 = values.new_ones((batch, values.shape[-1]))
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


class CausalEMAFeatures(nn.Module):
    """Capacity-matched causal EMA features using the scalar KF steady-state gain."""

    def __init__(self, channels: tuple[int, ...] = PILOT_CHANNELS) -> None:
        super().__init__()
        if not channels or tuple(sorted(set(channels))) != channels or channels[-1] >= 112:
            raise ValueError("channels must be a nonempty, sorted, unique subset of 0..111")
        self.max_channel = channels[-1]
        self.register_buffer("channel_indices", torch.tensor(channels, dtype=torch.long))
        self.register_buffer("arm_code", torch.tensor(ARM_CODES["ema"], dtype=torch.int8))
        self.register_buffer("alpha", torch.tensor(EMA_ALPHA, dtype=torch.float32))

    def forward(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Emit level, slope, standardized innovation and log variance causally."""
        if observations.ndim != 3 or observations.shape[1] == 0:
            raise ValueError("EMA input must be a nonempty (batch, time, features) tensor")
        if observations.shape[-1] <= self.max_channel:
            raise ValueError("EMA input does not contain the selected raw channels")
        values = observations.index_select(-1, self.channel_indices)
        if state is None:
            level = values[:, 0]
            variance = values.new_ones(level.shape)
        else:
            level, variance = state
        alpha = self.alpha.to(dtype=values.dtype)
        outputs: list[torch.Tensor] = []
        for value in values.unbind(dim=1):
            innovation = value - level
            variance = (1.0 - alpha) * variance + alpha * innovation.square()
            level_next = level + alpha * innovation
            outputs.append(
                torch.stack(
                    (
                        level_next,
                        level_next - level,
                        innovation / variance.clamp_min(VARIANCE_FLOOR).sqrt(),
                        variance.clamp_min(VARIANCE_FLOOR).log(),
                    ),
                    dim=-1,
                )
            )
            level = level_next
        return torch.stack(outputs, dim=1).flatten(start_dim=-2), (level, variance)


class CapacityMatchedRawFeatures(nn.Module):
    """Append zero channels to match smoother arms' GRU input dimensions."""

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("channel_indices", torch.tensor(channels, dtype=torch.long))
        self.register_buffer("arm_code", torch.tensor(ARM_CODES["raw"], dtype=torch.int8))

    def forward(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the zero extension and no auxiliary recurrent state."""
        if observations.ndim != 3:
            raise ValueError("raw-control input must have shape (batch, time, features)")
        width = self.channel_indices.numel() * FEATURES_PER_CHANNEL
        return observations.new_zeros((*observations.shape[:2], width)), ()


class KalmanTeacherGRU(nn.Module):
    """Canonical-GRU adapter with a scope-matched causal feature block."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        args: dict[str, object],
        arm: str,
        channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        if arm == "raw":
            self.features = CapacityMatchedRawFeatures(channels)
        elif arm == "ema":
            self.features = CausalEMAFeatures(channels)
        else:
            self.features = CausalLocalTrendKalman(arm, channels)
        self.core = RNNModel(
            n_in + len(channels) * FEATURES_PER_CHANNEL,
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
        filter_state = None if state is None else state[1:]
        features, next_filter_state = self.features(normalized, filter_state)
        output, next_core_state = self.core(torch.cat((normalized, features), dim=-1), core_state)
        return output, (next_core_state, *next_filter_state)


def make_kalman_model(
    args_dict: dict[str, object],
    n_in: int,
    n_out: int,
    arm: str = "learned",
    channels: tuple[int, ...] = PILOT_CHANNELS,
) -> KalmanTeacherGRU:
    """Build the canonical dense-GRU adapter on unmodified raw/event inputs."""
    unsupported = (
        n_in != N_FEATURES + N_EVENT_FEATURES
        or bool(args_dict.get("raw_only", False))
        or bool(args_dict.get("online_norm", False))
        or bool(args_dict.get("heuristics", False))
        or bool(args_dict.get("prop_input", ""))
        or args_dict.get("research_features", "none") != "none"
        or args_dict.get("privileged_mode", "none") != "none"
        or bool(args_dict.get("sentinel_base_width", 0))
        or args_dict.get("arch", "gru") != "gru"
        or args_dict.get("cell", "gru") != "gru"
        or args_dict.get("enc", "linear") != "linear"
        or args_dict.get("numeric_map", "none") != "none"
        or args_dict.get("rnn_init", "default") != "default"
        or args_dict.get("head_layers", 1) != 1
        or any(
            args_dict.get(name) is not None
            for name in ("encoder_width", "recurrent_width", "readout_width")
        )
        or bool(args_dict.get("init", ""))
    )
    if unsupported:
        raise ValueError("Kalman adapter supports only a fresh dense plain-GRU recipe")
    return KalmanTeacherGRU(n_in, n_out, args_dict, arm, channels)


def arm_from_state_dict(state: dict[str, torch.Tensor]) -> str:
    """Recover the fixed/learned arm identifier from a saved inference state."""
    code = int(state["features.arm_code"].item())
    if code not in ARM_CODES.values():
        raise ValueError(f"unknown Kalman checkpoint arm code: {code}")
    return {value: key for key, value in ARM_CODES.items()}[code]


def make_kalman_checkpoint_model(
    args_dict: dict[str, object], state: dict[str, torch.Tensor], n_in: int
) -> KalmanTeacherGRU:
    """Rebuild the adapter model from checkpoint-owned arm and output dimensions."""
    output_weight = state["core.head.2.weight"]
    adapter = args_dict.get("model_adapter", {})
    channels = tuple(adapter.get("raw_channels", PILOT_CHANNELS))
    model = make_kalman_model(
        args_dict, n_in, output_weight.shape[0], arm_from_state_dict(state), channels
    )
    model.load_state_dict(state)
    return model
