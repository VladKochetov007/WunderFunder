from __future__ import annotations

import math

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

    @torch.no_grad()
    def initialize_write_timescales_(
        self, minimum_rows: float = 2.0, maximum_rows: float = 10_000.0
    ) -> torch.Tensor:
        """Initialize only write biases with a log-spaced memory-time prior.

        This explicit experimental arm leaves default initialization unchanged.
        With zero gate input, retention exp(-1/tau) gives an e-folding time of
        tau rows. Actual input-dependent gates need a separate measurement.
        Returns the intended timescales; stored bias precision can alter them.
        """
        if not (
            math.isfinite(minimum_rows)
            and math.isfinite(maximum_rows)
            and 1 <= minimum_rows <= maximum_rows
        ):
            raise ValueError("timescales must be finite with 1 <= minimum <= maximum")
        timescales = torch.logspace(
            math.log10(minimum_rows),
            math.log10(maximum_rows),
            self.hidden_size,
            dtype=torch.float64,
            device=self.input_projection.bias.device,
        )
        # z is the write gate here, opposite PyTorch GRU's retention gate.
        bias = torch.log(torch.expm1(timescales.reciprocal()))
        self.input_projection.bias[self.hidden_size :].copy_(bias)
        return timescales

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


class MinGRUModel(nn.Module):
    """Preserve the GRU student's normalized GELU encoder and skip readout.

    Inputs are already-engineered causal features, not raw competition rows.
    Returned state is [layers, batch, recurrent_width], and None resets every
    layer. This separate model is not registered in the production trainer.
    """

    def __init__(
        self,
        n_in: int,
        hidden: int = 256,
        layers: int = 2,
        n_out: int = 2,
        encoder_width: int | None = None,
        recurrent_width: int | None = None,
        readout_width: int | None = None,
    ) -> None:
        super().__init__()
        encoder = hidden if encoder_width is None else encoder_width
        recurrent = hidden if recurrent_width is None else recurrent_width
        readout = hidden if readout_width is None else readout_width
        if min(n_in, layers, n_out, encoder, recurrent, readout) <= 0:
            raise ValueError("all model dimensions must be positive")
        self.recurrent_width = recurrent
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, encoder)
        self.cells = nn.ModuleList(
            MinGRULayer(encoder if index == 0 else recurrent, recurrent) for index in range(layers)
        )
        self.head = nn.Sequential(
            nn.Linear(recurrent + encoder, readout), nn.GELU(), nn.Linear(readout, n_out)
        )

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process complete or partial sequences with caller-owned state."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.mean.numel():
            raise ValueError("inputs must have shape [batch, time, n_in]")
        if hidden is not None and hidden.shape != (
            len(self.cells),
            inputs.shape[0],
            self.recurrent_width,
        ):
            raise ValueError("model state must have shape [layers, batch, recurrent_width]")
        projection = torch.nn.functional.gelu(self.inp((inputs - self.mean) / self.std))
        sequence = projection
        states = []
        for index, cell in enumerate(self.cells):
            sequence, state = cell(sequence, None if hidden is None else hidden[index])
            states.append(state)
        predictions = self.head(torch.cat((sequence, projection), dim=-1))
        return predictions, torch.stack(states)
