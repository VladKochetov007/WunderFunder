from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def collect_recurrent_gates(
    module: nn.LSTM | nn.GRU,
    inputs: Tensor,
    state: object,
    record_gate: Callable[[str, Tensor, bool, int], None],
) -> None:
    """Recompute sampled recurrent gate distributions without changing model state."""
    hidden_rows, cell_rows = _initial_state(module, inputs, state)
    if hidden_rows is None:
        return
    with torch.no_grad():
        for timestep in range(inputs.shape[1]):
            layer_input = inputs[:, timestep]
            for layer in range(module.num_layers):
                prefix = f"_l{layer}"
                input_gates = F.linear(
                    layer_input,
                    getattr(module, f"weight_ih{prefix}"),
                    getattr(module, f"bias_ih{prefix}", None),
                )
                hidden_gates = F.linear(
                    hidden_rows[layer],
                    getattr(module, f"weight_hh{prefix}"),
                    getattr(module, f"bias_hh{prefix}", None),
                )
                gates = input_gates + hidden_gates
                if isinstance(module, nn.LSTM):
                    i, f, g, o = gates.chunk(4, dim=-1)
                    input_gate, forget_gate, candidate, output_gate = (
                        i.sigmoid(),
                        f.sigmoid(),
                        g.tanh(),
                        o.sigmoid(),
                    )
                    assert cell_rows is not None
                    cell_rows[layer] = forget_gate * cell_rows[layer] + input_gate * candidate
                    hidden_rows[layer] = output_gate * cell_rows[layer].tanh()
                    record_gate("i", input_gate, True, layer)
                    record_gate("f", forget_gate, True, layer)
                    record_gate("g", candidate, False, layer)
                    record_gate("o", output_gate, True, layer)
                else:
                    input_parts, hidden_parts = (
                        input_gates.chunk(3, dim=-1),
                        hidden_gates.chunk(3, dim=-1),
                    )
                    reset = (input_parts[0] + hidden_parts[0]).sigmoid()
                    update = (input_parts[1] + hidden_parts[1]).sigmoid()
                    candidate = (input_parts[2] + reset * hidden_parts[2]).tanh()
                    hidden_rows[layer] = candidate + update * (hidden_rows[layer] - candidate)
                    record_gate("r", reset, True, layer)
                    record_gate("z", update, True, layer)
                    record_gate("n", candidate, False, layer)
                layer_input = hidden_rows[layer]


def _initial_state(
    module: nn.LSTM | nn.GRU, inputs: Tensor, state: object
) -> tuple[list[Tensor] | None, list[Tensor] | None]:
    """Clone the recurrent initial state so diagnostic gate replay is read-only."""
    if state is None:
        hidden = inputs.new_zeros(module.num_layers, inputs.shape[0], module.hidden_size)
        cell = hidden.clone() if isinstance(module, nn.LSTM) else None
    elif isinstance(module, nn.GRU) and isinstance(state, Tensor):
        hidden = state.detach().clone()
        cell = None
    elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], Tensor):
        hidden = state[0].detach().clone()
        cell = state[1].detach().clone() if isinstance(state[1], Tensor) else None
    else:
        return None, None
    hidden_rows = [hidden[index] for index in range(module.num_layers)]
    cell_rows = [cell[index] for index in range(module.num_layers)] if cell is not None else None
    return hidden_rows, cell_rows
