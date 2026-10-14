"""Activation-aware fake quantization for the recurrent GRU boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch._higher_order_ops.scan import scan

from wnn.nn import RNNModel
from wnn.training_qat import qat_weight_replacements

A8_POSITIVE_LIMIT = 127
_ActivationGRUMetadata = tuple[bool, bool, int, int, float, bool]


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
            fake_quantize_activation_a8(previous, require_finite=False)
            if quantize_activations
            else previous
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


def require_finite_activation_a8(values: torch.Tensor, name: str = "activation") -> None:
    """Fail before recurrence when an A8 boundary input leaves the finite domain."""
    if values.dtype != torch.float32 or not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"A8 {name} must contain only finite float32 values")


def quantize_activation_a8(
    values: torch.Tensor, *, require_finite: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native-style symmetric A8 codes and scales for each final-axis vector.

    Finite float32 input vectors use ``round_ties_even(clamp(x * (127 / maxabs)))``
    and a dequantization scale of ``maxabs / 127``, matching ``vnni::quantize``.
    All-zero vectors return zero codes and scales. By default this helper
    rejects nonfinite input; the pure scan body disables that eager guard only
    after its caller validates the batch/state outside the compiled recurrence.
    Callers keep the original float32 recurrent state and use these codes only
    at matvec boundaries.
    """
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("A8 activation quantization requires nonempty float32 vectors")
    if require_finite:
        require_finite_activation_a8(values)
    elif values.dtype != torch.float32:
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


def fake_quantize_activation_a8(
    values: torch.Tensor, *, require_finite: bool = True
) -> torch.Tensor:
    """Quantize and dequantize final-axis vectors with an identity STE backward."""
    codes, scales = quantize_activation_a8(values, require_finite=require_finite)
    represented = codes.to(torch.float32) * scales.unsqueeze(-1)
    return values + (represented - values).detach()


def activation_qat_forward(
    model: RNNModel,
    inputs: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    quantize_activations: bool = True,
    weight_replacements: Mapping[str, torch.Tensor] | None = None,
    validate_finite: bool = True,
    parameter_map_is_complete: bool = False,
    gru_metadata: _ActivationGRUMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model with a scan-based GRU and optional recurrent-boundary A8.

    Input-to-hidden terms are computed over the full sequence, then each layer
    scans its causal recurrent state. A8 is applied immediately before both
    recurrent matvecs; the stored GRU state remains float32. Any configured
    weight-QAT policy is retained through the same represented-weight helpers
    used by ``qat_forward``. It quantizes the post-encoder vector entering the
    first GRU ``weight_ih`` matvec and each later layer's input matvec. It does
    not quantize raw model inputs, input-projection weights/native I16 math, or
    the head's hidden/projection activation blocks.
    Compiled callers pass complete model parameters with QAT replacements
    applied and static GRU metadata from the wrapper, keeping module traversal
    and GRU inspection outside Dynamo's traced body.
    """
    if inputs.dtype != torch.float32 or (state is not None and state.dtype != torch.float32):
        raise ValueError("activation-aware QAT inputs and state must remain float32")
    if validate_finite:
        require_finite_activation_a8(inputs, "input batch")
        if state is not None:
            require_finite_activation_a8(state, "initial recurrent state")
        finite_parameters = torch.stack(
            [torch.isfinite(parameter).all() for parameter in model.parameters()]
            + [torch.isfinite(model.mean).all(), torch.isfinite(model.std).all()]
        ).all()
        if not bool(finite_parameters.item()):
            raise ValueError("A8 model parameters and normalization tensors must be finite")
    if gru_metadata is None:
        gru = model.gru
        gru_metadata = (
            isinstance(gru, nn.GRU),
            model.enc == "linear",
            gru.num_layers,
            gru.hidden_size,
            float(gru.dropout),
            gru.bidirectional,
        )
    is_plain_gru, is_linear_encoder, num_layers, hidden_size, dropout, bidirectional = gru_metadata
    if not is_plain_gru or not is_linear_encoder:
        raise ValueError("activation-aware QAT supports the plain GRU with linear input encoder")
    if inputs.ndim != 3 or inputs.shape[-1] != model.mean.numel() or inputs.shape[1] == 0:
        raise ValueError("activation-aware QAT expects nonempty (batch, time, features) inputs")
    if bidirectional:
        raise ValueError("activation-aware QAT does not support bidirectional GRUs")
    if state is not None and state.shape != (
        num_layers,
        inputs.shape[0],
        hidden_size,
    ):
        raise ValueError("activation-aware QAT initial state has the wrong shape")

    with torch.autocast(inputs.device.type, enabled=False):
        normalized = (inputs - model.mean) / model.std
        projection = model.encode(normalized)
        if parameter_map_is_complete:
            if weight_replacements is None:
                raise ValueError("a complete activation-QAT parameter map is required")
            parameters = weight_replacements
        else:
            parameters = dict(model.named_parameters())
            parameters.update(
                qat_weight_replacements(model)
                if weight_replacements is None
                else weight_replacements
            )
        layer_input = projection
        final_states: list[torch.Tensor] = []

        for layer in range(num_layers):
            weight_ih = parameters[f"gru.weight_ih_l{layer}"]
            weight_hh = parameters[f"gru.weight_hh_l{layer}"]
            bias_ih = parameters[f"gru.bias_ih_l{layer}"]
            bias_hh = parameters[f"gru.bias_hh_l{layer}"]
            layer_input_for_matvec = (
                fake_quantize_activation_a8(layer_input, require_finite=False)
                if quantize_activations
                else layer_input
            )
            input_gates = F.linear(layer_input_for_matvec, weight_ih, bias_ih)
            initial = (
                inputs.new_zeros((inputs.shape[0], hidden_size)) if state is None else state[layer]
            )

            final, sequence = scan(
                _gru_scan_step(weight_hh, bias_hh, quantize_activations),
                initial,
                input_gates.transpose(0, 1).contiguous(),
            )
            layer_input = sequence.transpose(0, 1).contiguous()
            final_states.append(final)
            if layer + 1 < num_layers and model.training and dropout > 0:
                layer_input = F.dropout(layer_input, p=dropout, training=True)

        recurrent = layer_input
        final_state = torch.stack(final_states, dim=0)
        head_input = torch.cat([recurrent, projection], dim=-1)
        first = F.linear(head_input, parameters["head.0.weight"], parameters["head.0.bias"])
        activated = model.head[1](first)
        if model.head.extra is not None:
            activated = model.head[1](F.linear(
                activated,
                parameters["head.extra.weight"],
                parameters["head.extra.bias"],
            ))
        prediction = F.linear(activated, parameters["head.2.weight"], parameters["head.2.bias"])
        return prediction, final_state


class ActivationQATModel(nn.Module):
    """Module wrapper suitable for ``torch.compile`` and forward/backward profiling."""

    def __init__(self, model: RNNModel, quantize_activations: bool = True) -> None:
        super().__init__()
        gru = model.gru
        if not isinstance(gru, nn.GRU) or model.enc != "linear":
            raise ValueError(
                "activation-aware QAT supports the plain GRU with linear input encoder"
            )
        self.model = model
        self.quantize_activations = quantize_activations
        self._gru_metadata: _ActivationGRUMetadata = (
            True,
            True,
            gru.num_layers,
            gru.hidden_size,
            float(gru.dropout),
            gru.bidirectional,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        weight_replacements: Mapping[str, torch.Tensor] | None = None,
        validate_finite: bool = True,
        state: torch.Tensor | None = None,
        parameter_map_is_complete: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a chunk from the supplied causal state, or initialize from zero."""
        return activation_qat_forward(
            self.model,
            inputs,
            state,
            quantize_activations=self.quantize_activations,
            weight_replacements=weight_replacements,
            validate_finite=validate_finite,
            parameter_map_is_complete=parameter_map_is_complete,
            gru_metadata=self._gru_metadata,
        )
