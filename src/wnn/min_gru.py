from __future__ import annotations

import torch
from torch import nn


def affine_prefix_scan(retention: torch.Tensor, addition: torch.Tensor) -> torch.Tensor:
    """Scan h[t]=retention[t]*h[t-1]+addition[t] from an actual zero state.

    Pair composition is associative in real arithmetic. This parallel-doubling
    reference avoids log-space initial-state transforms and divisions by tiny
    cumulative products. It uses O(T log T) work and autograd storage; it is not
    a claim of an efficient GPU scan implementation.
    """
    if retention.shape != addition.shape or retention.ndim != 3:
        raise ValueError("scan operands must share [batch, time, hidden] shape")
    coefficients, values = retention, addition
    offset = 1
    while offset < values.shape[1]:
        next_values = torch.cat(
            (
                values[:, :offset],
                values[:, offset:] + coefficients[:, offset:] * values[:, :-offset],
            ),
            dim=1,
        )
        coefficients = torch.cat(
            (coefficients[:, :offset], coefficients[:, offset:] * coefficients[:, :-offset]),
            dim=1,
        )
        values = next_values
        offset *= 2
    return values


def positive_candidate(value: torch.Tensor) -> torch.Tensor:
    """Use the positive candidate from minGRU's log-space variant."""
    return torch.where(value >= 0, value + 0.5, torch.sigmoid(value))


class MinGRULayer(nn.Module):
    """Input-conditioned minGRU with one packed candidate/gate input matrix.

    Based on Feng et al., https://arxiv.org/html/2410.01201v3, Appendix B.
    All methods receive the actual prior hidden state, including exact zero.
    The write gate is z=sigmoid(g); retention is 1-z. The state has no role in
    gate or candidate computation. This is a research reference, unconnected
    to the competition trainer or native serving implementation.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input and hidden sizes must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_size, 2 * hidden_size)

    def _state(self, inputs: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        expected = (inputs.shape[0], self.hidden_size)
        if hidden is None:
            return inputs.new_zeros(expected)
        if (
            hidden.shape != expected
            or hidden.device != inputs.device
            or hidden.dtype != inputs.dtype
        ):
            raise ValueError("state must match the batch, hidden size, dtype and device")
        return hidden

    def _coefficients(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate, gate = self.input_projection(inputs).chunk(2, dim=-1)
        write = torch.sigmoid(gate)
        return 1 - write, write * positive_candidate(candidate)

    def step(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Process exactly one row without implicit persistent state."""
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size:
            raise ValueError("step inputs must have shape [batch, input_size]")
        previous = self._state(inputs, hidden)
        retention, addition = self._coefficients(inputs)
        return retention * previous + addition

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a causal sequence through affine composition with exact h0."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("sequence inputs must have shape [batch, time, input_size]")
        previous = self._state(inputs, hidden)
        if inputs.shape[1] == 0:
            return inputs.new_empty((inputs.shape[0], 0, self.hidden_size)), previous
        retention, addition = self._coefficients(inputs)
        # Incorporate actual h0 once, before scanning. This also preserves its
        # correct derivative at h0=0 without taking log(0) or changing h0 to g(0).
        addition = torch.cat(
            ((addition[:, 0] + retention[:, 0] * previous).unsqueeze(1), addition[:, 1:]),
            dim=1,
        )
        outputs = affine_prefix_scan(retention, addition)
        return outputs, outputs[:, -1]
