"""Activation-aware fake quantization for the recurrent GRU boundaries."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch._higher_order_ops.scan import scan

from wnn.nn import RNNModel
from wnn.training_qat import qat_weight_replacements

A8_POSITIVE_LIMIT = 127


def _gru_scan_step(
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    quantize_activations: bool,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Bind one layer's recurrent tensors in a pure scan body."""

    def step(
        previous: torch.Tensor,
        gates_from_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_input = (
            fake_quantize_activation_a8(previous) if quantize_activations else previous
        )
        gates_from_state = F.linear(recurrent_input, weight_hh, bias_hh)
        input_reset, input_update, input_new = gates_from_input.chunk(3, dim=-1)
        state_reset, state_update, state_new = gates_from_state.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + state_reset)
        update = torch.sigmoid(input_update + state_update)
        candidate = torch.tanh(input_new + reset * state_new)
        next_state = (1.0 - update) * candidate + update * previous
        return next_state, next_state.clone()

    return step


def quantize_activation_a8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native-style symmetric A8 codes and scales for each final-axis vector.

    Finite float32 input vectors use ``round_ties_even(clamp(x * (127 / maxabs)))``
    and a dequantization scale of ``maxabs / 127``, matching ``vnni::quantize``.
    All-zero vectors return zero codes and scales. Callers keep the original
    float32 recurrent state and use these codes only at matvec boundaries.
    """
    if values.dtype != torch.float32 or values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("A8 activation quantization requires nonempty float32 vectors")
    detached = values.detach()
    maximum = detached.abs().amax(dim=-1, keepdim=True)
    valid = torch.isfinite(maximum) & (maximum > 0)
    inverse = torch.where(valid, A8_POSITIVE_LIMIT / maximum, torch.zeros_like(maximum))
    products = detached * inverse
    # Rust f32::max/min ignore a NaN operand. This matters only for finite
    # subnormal vectors whose reciprocal overflows before zero entries multiply.
    lower_bound = torch.full_like(products, -float(A8_POSITIVE_LIMIT))
    upper_bound = torch.full_like(products, float(A8_POSITIVE_LIMIT))
    clamped = torch.fmin(
        torch.fmax(products, lower_bound),
        upper_bound,
    )
    rounded = torch.round(clamped)
    codes = rounded.to(torch.int8)
    scale = torch.where(valid, maximum / A8_POSITIVE_LIMIT, torch.zeros_like(maximum))
    return codes, scale.squeeze(-1)


def fake_quantize_activation_a8(values: torch.Tensor) -> torch.Tensor:
    """Quantize and dequantize final-axis vectors with an identity STE backward."""
    codes, scales = quantize_activation_a8(values)
    represented = codes.to(torch.float32) * scales.unsqueeze(-1)
    return values + (represented - values).detach()


def activation_qat_forward(
    model: RNNModel,
    inputs: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    quantize_activations: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model with a scan-based GRU and optional recurrent-boundary A8.

    Input-to-hidden terms are computed over the full sequence, then each layer
    scans its causal recurrent state. A8 is applied immediately before both
    recurrent matvecs; the stored GRU state remains float32. Any configured
    weight-QAT policy is retained through the same represented-weight helpers
    used by ``qat_forward``. Projection/head activation quantization is outside
    this recurrent-only implementation.
    """
    if inputs.dtype != torch.float32 or (state is not None and state.dtype != torch.float32):
        raise ValueError("activation-aware QAT inputs and state must remain float32")
    if not isinstance(model.gru, nn.GRU) or model.enc != "linear":
        raise ValueError("activation-aware QAT supports the plain GRU with linear input encoder")
    if inputs.ndim != 3 or inputs.shape[-1] != model.mean.numel() or inputs.shape[1] == 0:
        raise ValueError("activation-aware QAT expects nonempty (batch, time, features) inputs")
    if model.gru.bidirectional:
        raise ValueError("activation-aware QAT does not support bidirectional GRUs")
    if state is not None and state.shape != (
        model.gru.num_layers,
        inputs.shape[0],
        model.gru.hidden_size,
    ):
        raise ValueError("activation-aware QAT initial state has the wrong shape")

    with torch.autocast(inputs.device.type, enabled=False):
        normalized = (inputs - model.mean) / model.std
        projection = model.encode(normalized)
        parameters = dict(model.named_parameters())
        parameters.update(qat_weight_replacements(model))
        layer_input = projection
        final_states: list[torch.Tensor] = []

        for layer in range(model.gru.num_layers):
            weight_ih = parameters[f"gru.weight_ih_l{layer}"]
            weight_hh = parameters[f"gru.weight_hh_l{layer}"]
            bias_ih = parameters[f"gru.bias_ih_l{layer}"]
            bias_hh = parameters[f"gru.bias_hh_l{layer}"]
            layer_input_for_matvec = (
                fake_quantize_activation_a8(layer_input) if quantize_activations else layer_input
            )
            input_gates = F.linear(layer_input_for_matvec, weight_ih, bias_ih)
            initial = (
                inputs.new_zeros((inputs.shape[0], model.gru.hidden_size))
                if state is None
                else state[layer]
            )

            final, sequence = scan(
                _gru_scan_step(weight_hh, bias_hh, quantize_activations),
                initial,
                input_gates.transpose(0, 1).contiguous(),
            )
            layer_input = sequence.transpose(0, 1).contiguous()
            final_states.append(final)
            if layer + 1 < model.gru.num_layers and model.training and model.gru.dropout > 0:
                layer_input = F.dropout(layer_input, p=model.gru.dropout, training=True)

        recurrent = layer_input
        final_state = torch.stack(final_states, dim=0)
        head_input = torch.cat([recurrent, projection], dim=-1)
        first = F.linear(head_input, parameters["head.0.weight"], model.head[0].bias)
        activated = model.head[1](first)
        prediction = F.linear(activated, model.head[2].weight, model.head[2].bias)
        return prediction, final_state


class ActivationQATModel(nn.Module):
    """Module wrapper suitable for ``torch.compile`` and forward/backward profiling."""

    def __init__(self, model: RNNModel, quantize_activations: bool = True) -> None:
        super().__init__()
        self.model = model
        self.quantize_activations = quantize_activations

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one complete sequence from the model's zero recurrent state."""
        return activation_qat_forward(
            self.model,
            inputs,
            quantize_activations=self.quantize_activations,
        )
