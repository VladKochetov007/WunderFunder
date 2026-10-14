"""Causal research wrappers for constant, first-row and prefix GRU state arms."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from attrs import frozen
from torch import nn

from wnn.nn import RNNModel

InitializationMode = Literal["zero", "constant", "first_row", "prefix"]
PREFIX_ROWS = 99


@frozen
class InitializationState:
    """Sequence-local state; passing None to the wrapper resets every field."""

    hidden: torch.Tensor
    position: int
    prefix_mean: torch.Tensor | None = None
    prefix_m2: torch.Tensor | None = None
    prefix_last: torch.Tensor | None = None

    def detach(self) -> InitializationState:
        """Detach the entire recurrent graph at an explicit TBPTT boundary."""
        return InitializationState(
            hidden=self.hidden.detach(),
            position=self.position,
            prefix_mean=None if self.prefix_mean is None else self.prefix_mean.detach(),
            prefix_m2=None if self.prefix_m2 is None else self.prefix_m2.detach(),
            prefix_last=None if self.prefix_last is None else self.prefix_last.detach(),
        )


class InitializedGRU(nn.Module):
    """Wrap a GRU backbone without changing its preprocessing or readout.

    Batches contain synchronized complete-sequence starts. Inputs are the existing
    causal feature vectors, before backbone normalization. The prefix arm processes
    every warmup row, then adds a bounded residual immediately before row 99.
    No target, scoring mask, sequence ID or future row enters the initializer.

    All learned arms start as the zero-state baseline. Setting backbone parameters'
    requires_grad to False enables initializer-only adaptation without detaching the
    recurrent graph: gradients must still pass through the frozen backbone.
    """

    def __init__(
        self,
        backbone: RNNModel,
        mode: InitializationMode,
        residual_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in ("zero", "constant", "first_row", "prefix"):
            raise ValueError(f"unknown initialization mode: {mode}")
        if not isinstance(backbone.gru, nn.GRU) or backbone.enc != "linear":
            raise ValueError("initialization experiment requires the linear-encoder GRU backbone")
        if backbone.gru.dropout != 0:
            raise ValueError("chunk/full parity requires the declared dropout-zero backbone")
        if not 0 < residual_bound < float("inf"):
            raise ValueError("residual_bound must be finite and positive")
        self.backbone = backbone
        self.mode = mode
        self.residual_bound = residual_bound
        self.layers = backbone.gru.num_layers
        self.hidden_width = backbone.gru.hidden_size
        self.feature_width = backbone.mean.numel()
        self.constant = (
            nn.Parameter(torch.zeros(self.layers, 1, self.hidden_width))
            if mode == "constant"
            else None
        )
        self.predictor: nn.Sequential | None = None
        if mode in ("first_row", "prefix"):
            width = self.feature_width * (3 if mode == "prefix" else 1)
            output = nn.Linear(64, self.layers * self.hidden_width)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            self.predictor = nn.Sequential(nn.Linear(width, 64), nn.GELU(), output)

    def _predict(self, context: torch.Tensor) -> torch.Tensor:
        if self.predictor is None:
            raise RuntimeError("this arm has no context predictor")
        values = self.residual_bound * self.predictor(context).tanh()
        return values.reshape(context.shape[0], self.layers, self.hidden_width).transpose(0, 1)

    def _start(self, inputs: torch.Tensor) -> InitializationState:
        hidden = inputs.new_zeros(self.layers, inputs.shape[0], self.hidden_width)
        if self.constant is not None:
            hidden = (self.residual_bound * self.constant.tanh()).expand_as(hidden).contiguous()
        elif self.mode == "first_row":
            first = (inputs[:, 0] - self.backbone.mean) / self.backbone.std
            hidden = self._predict(first).contiguous()
        return InitializationState(hidden=hidden, position=0)

    def _collect_prefix(
        self, inputs: torch.Tensor, state: InitializationState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = (inputs - self.backbone.mean) / self.backbone.std
        mean = state.prefix_mean
        m2 = state.prefix_m2
        if mean is None:
            mean = torch.zeros_like(normalized[:, 0])
            m2 = torch.zeros_like(mean)
        if m2 is None:
            raise ValueError("prefix moments must be reset together")
        # Identical row order and Welford operations at every chunk size.
        for offset in range(normalized.shape[1]):
            row = normalized[:, offset]
            delta = row - mean
            mean = mean + delta / (state.position + offset + 1)
            m2 = m2 + delta * (row - mean)
        return mean, m2, normalized[:, -1]

    def forward(
        self, inputs: torch.Tensor, state: InitializationState | None = None
    ) -> tuple[torch.Tensor, InitializationState]:
        """Process a nonempty causal chunk and return all state needed to continue."""
        if inputs.ndim != 3 or inputs.shape[1] == 0 or inputs.shape[2] != self.feature_width:
            raise ValueError("inputs must be nonempty [batch, rows, backbone feature width]")
        if state is None:
            state = self._start(inputs)
        if state.hidden.shape != (self.layers, inputs.shape[0], self.hidden_width):
            raise ValueError("hidden shape differs from this synchronized sequence batch")
        if state.position < 0:
            raise ValueError("sequence position must be nonnegative")
        outputs = []
        offset = 0
        while offset < inputs.shape[1]:
            stop = inputs.shape[1]
            mean, m2, last = state.prefix_mean, state.prefix_m2, state.prefix_last
            hidden = state.hidden
            if self.mode == "prefix" and state.position < PREFIX_ROWS:
                stop = min(stop, offset + PREFIX_ROWS - state.position)
                mean, m2, last = self._collect_prefix(inputs[:, offset:stop], state)
            elif self.mode == "prefix" and state.position == PREFIX_ROWS:
                if mean is None or m2 is None or last is None:
                    raise ValueError("row-99 injection requires the complete preceding prefix")
                std = (m2 / PREFIX_ROWS).clamp_min(0).sqrt()
                hidden = hidden + self._predict(torch.cat((mean, std, last), dim=-1))
                mean = m2 = last = None
            prediction, hidden = self.backbone(inputs[:, offset:stop], hidden)
            outputs.append(prediction)
            state = InitializationState(hidden, state.position + stop - offset, mean, m2, last)
            offset = stop
        return torch.cat(outputs, dim=1), state


def export_native_initializer(module: InitializedGRU) -> dict[str, object]:
    """Export FP32 initializer parameters for ``wnn_native.EnsemblePredictor``.

    The backbone remains in the ordinary model weight dictionary. This payload contains only
    initializer parameters and does not quantize or alter their values.
    """
    config: dict[str, object] = {"mode": module.mode}
    if module.mode == "zero":
        return config
    config["residual_bound"] = float(module.residual_bound)
    if module.mode == "constant":
        if module.constant is None:
            raise RuntimeError("constant initializer parameters are missing")
        config["constant"] = np.ascontiguousarray(
            module.constant.detach().cpu().numpy()[:, 0, :], dtype=np.float32
        )
        return config
    if module.predictor is None:
        raise RuntimeError("context initializer parameters are missing")
    input_layer = module.predictor[0]
    output_layer = module.predictor[2]
    if not isinstance(input_layer, nn.Linear) or not isinstance(output_layer, nn.Linear):
        raise RuntimeError("unsupported context initializer layout")
    config.update(
        {
            "predictor_input_weight": np.ascontiguousarray(
                input_layer.weight.detach().cpu().numpy(), dtype=np.float32
            ),
            "predictor_input_bias": np.ascontiguousarray(
                input_layer.bias.detach().cpu().numpy(), dtype=np.float32
            ),
            "predictor_output_weight": np.ascontiguousarray(
                output_layer.weight.detach().cpu().numpy(), dtype=np.float32
            ),
            "predictor_output_bias": np.ascontiguousarray(
                output_layer.bias.detach().cpu().numpy(), dtype=np.float32
            ),
        }
    )
    return config
