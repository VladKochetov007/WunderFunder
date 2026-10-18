"""Causal numerical Transformer starter for teacher research; no trainer included."""

from __future__ import annotations

import math

import torch
from torch import nn

from wnn.causal_research_features import _affine_prefix_scan

FEATURE_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("i0_bid_price", 0, 11),
    ("i0_ask_price", 11, 22),
    ("i0_bid_volume", 22, 33),
    ("i0_ask_volume", 33, 44),
    ("i0_trade_price", 44, 48),
    ("i0_trade_volume", 48, 52),
    ("i1_bid_price", 52, 63),
    ("i1_ask_price", 63, 74),
    ("i1_bid_volume", 74, 85),
    ("i1_ask_volume", 85, 96),
    ("i1_trade_price", 96, 100),
    ("i1_trade_volume", 100, 104),
    ("additional", 104, 112),
)


class Grouped112FeatureEncoder(nn.Module):
    """Encode the documented 13 groups into one temporal token per input row."""

    def __init__(self, output_width: int, group_width: int = 16, hidden_width: int = 32) -> None:
        super().__init__()
        if min(output_width, group_width, hidden_width) <= 0:
            raise ValueError("encoder widths must be positive")
        self.group_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(end - start, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, group_width),
                nn.GELU(),
            )
            for _, start, end in FEATURE_GROUPS
        )
        self.project = nn.Linear(len(FEATURE_GROUPS) * group_width, output_width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, rows, 112)`` values to ``(batch, rows, output_width)``."""
        if values.ndim != 3 or values.shape[-1] != 112:
            raise ValueError("values must have (batch, rows, 112) shape")
        groups = [
            encoder(values[..., start:end])
            for encoder, (_, start, end) in zip(self.group_mlps, FEATURE_GROUPS, strict=True)
        ]
        return self.project(torch.cat(groups, dim=-1))


def _sinusoidal_positions(
    rows: int, width: int, offset: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create deterministic absolute positions so attention can use row order."""
    positions = torch.arange(offset, offset + rows, device=device, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
    result = torch.zeros(rows, width, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(angles)
    result[:, 1::2] = torch.cos(angles[:, : result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)


class CausalTransformerTeacher(nn.Module):
    """Grouped-input, pre-norm causal Transformer with an optional output AR head.

    Inputs are observable features only. The optional recurrence uses its own
    previous prediction; labels are never accepted by ``forward``. This is an
    FP32 research reference; normalization fitting and mixed precision are not
    part of this module.
    """

    def __init__(
        self,
        width: int = 192,
        layers: int = 4,
        heads: int = 6,
        feedforward_width: int = 768,
        dropout: float = 0.05,
        autoregressive: bool = False,
        rho_max: float = 0.99,
        initial_gate: float = 0.10,
    ) -> None:
        super().__init__()
        if min(width, layers, heads, feedforward_width) <= 0 or width % heads:
            raise ValueError("positive dimensions and width divisible by heads required")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < initial_gate < rho_max < 1.0:
            raise ValueError("require 0 < initial_gate < rho_max < 1")

        self.width = width
        self.autoregressive = autoregressive
        self.rho_max = rho_max
        self.input_encoder = Grouped112FeatureEncoder(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(width, 2)
        self.gate_head: nn.Linear | None = None
        if autoregressive:
            self.gate_head = nn.Linear(width, 2)
            nn.init.zeros_(self.gate_head.weight)
            initial_bias = math.log(initial_gate / (rho_max - initial_gate))
            nn.init.constant_(self.gate_head.bias, initial_bias)

    def forward(
        self,
        values: torch.Tensor,
        previous_output: torch.Tensor | None = None,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict each row and return the final own-output state when AR is on.

        A call without ``previous_output`` resets the AR state to zero. Pass
        the returned state to continue the output recurrence across chunks,
        and reset it at each new sequence.

        Attention has no key/value cache. Each call sees only the rows supplied
        in that call, so chunked attention is not equivalent to a full-sequence
        pass even when ``position_offset`` and the output state are carried.
        The explicit dense causal mask has quadratic attention cost; callers
        should provide bounded windows. Training, normalization fitting and
        chunk selection are intentionally outside this module.
        """
        if values.ndim != 3 or values.shape[1] == 0 or values.shape[-1] != 112:
            raise ValueError("values must have nonempty (batch, rows, 112) shape")
        if position_offset < 0:
            raise ValueError("position_offset must be nonnegative")
        batch_size, rows, _ = values.shape
        if self.autoregressive:
            if previous_output is not None and (
                previous_output.shape != (batch_size, 2)
                or previous_output.device != values.device
                or previous_output.dtype != values.dtype
            ):
                raise ValueError("previous_output must match input batch, device, and dtype")
        elif previous_output is not None:
            raise ValueError("previous_output is only valid for an autoregressive model")

        hidden = self.input_encoder(values)
        hidden = hidden + _sinusoidal_positions(
            rows, self.width, position_offset, values.device, hidden.dtype
        ).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(rows, rows, dtype=torch.bool, device=values.device), diagonal=1
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        proposal = self.output_head(hidden)
        if not self.autoregressive:
            return proposal, None

        assert self.gate_head is not None
        initial = proposal.new_zeros(batch_size, 2) if previous_output is None else previous_output
        gate = self.rho_max * torch.sigmoid(self.gate_head(hidden))
        prediction = _affine_prefix_scan(gate, (1.0 - gate) * proposal, initial)
        return prediction, prediction[:, -1]
