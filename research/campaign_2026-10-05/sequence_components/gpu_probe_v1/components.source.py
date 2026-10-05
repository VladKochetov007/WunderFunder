"""Small causal sequence components for the 2026-10-05 research campaign."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DiagonalSSM(nn.Module):
    """Causal diagonal stable state model with a nonlinear input embedding.

    The recurrence is ``state[t] = decay * state[t-1] + (1-decay) * embed(x[t])``.
    The bounded decay keeps each coordinate stable. ``learn_decay=False`` uses
    the same fixed log-spaced spectrum as the learned model's initialization.
    """

    def __init__(self, input_size: int, state_size: int, learn_decay: bool = True):
        super().__init__()
        if input_size < 1 or state_size < 1:
            raise ValueError("input_size and state_size must be positive")
        self.input_size = input_size
        self.state_size = state_size
        self.learn_decay = learn_decay
        self.input_embedding = nn.Linear(input_size, state_size)
        initial = torch.logspace(math.log10(0.85), math.log10(0.999), state_size)
        if learn_decay:
            low, high = self._decay_bounds()
            fraction = ((initial - low) / (high - low)).clamp(1e-5, 1 - 1e-5)
            self.decay_logits = nn.Parameter(torch.logit(fraction))
        else:
            self.register_buffer("fixed_decay", initial)
            self.register_parameter("decay_logits", None)

    @staticmethod
    def _decay_bounds() -> tuple[float, float]:
        return 0.5, 0.9999

    def _decay(self) -> Tensor:
        if self.learn_decay:
            low, high = self._decay_bounds()
            return low + (high - low) * torch.sigmoid(self.decay_logits)
        return self.fixed_decay

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Process a sequence chunk and return per-step states and final state.

        Args:
            x: Tensor shaped ``(..., time, input_size)``.
            state: Optional tensor shaped ``(..., state_size)``. Passing None
                starts a fresh sequence; pass the returned state to continue.
        """
        if x.ndim < 2 or x.shape[-1] != self.input_size:
            raise ValueError(f"x must have shape (..., time, {self.input_size})")
        leading = x.shape[:-2]
        if state is None:
            state = x.new_zeros((*leading, self.state_size))
        elif state.shape != (*leading, self.state_size):
            raise ValueError("state shape must match x's leading dimensions and state_size")

        embedded = torch.tanh(self.input_embedding(x))
        decay = self._decay().to(dtype=x.dtype, device=x.device)
        outputs: list[Tensor] = []
        for input_t in embedded.unbind(dim=-2):
            state = decay * state + (1.0 - decay) * input_t
            outputs.append(state)
        if outputs:
            return torch.stack(outputs, dim=-2), state
        return x.new_empty((*leading, 0, self.state_size)), state


class AdditiveSpline(nn.Module):
    """Zero-initialized additive piecewise-linear spline edge functions.

    This is KAN-inspired: each input coordinate has a learned univariate edge
    function and the edge outputs add. The interpolation is piecewise linear,
    not the cubic B-spline parameterization of the original KAN formulation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        knots: int = 7,
        knot_min: float = -3.0,
        knot_max: float = 3.0,
    ):
        super().__init__()
        if input_size < 1 or output_size < 1:
            raise ValueError("input_size and output_size must be positive")
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if not knot_min < knot_max:
            raise ValueError("knot_min must be smaller than knot_max")
        self.input_size = input_size
        self.output_size = output_size
        self.knots = knots
        self.knot_min = knot_min
        self.knot_max = knot_max
        self.edge_values = nn.Parameter(torch.zeros(output_size, input_size, knots))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate additive learned piecewise-linear univariate functions."""
        if x.ndim < 1 or x.shape[-1] != self.input_size:
            raise ValueError(f"x must have shape (..., {self.input_size})")
        position = (x - self.knot_min) * ((self.knots - 1) / (self.knot_max - self.knot_min))
        position = position.clamp(0.0, float(self.knots - 1))
        lower = position.floor().to(torch.long)
        upper = (lower + 1).clamp_max(self.knots - 1)
        fraction = position - lower.to(position.dtype)
        low_basis = F.one_hot(lower, num_classes=self.knots).to(dtype=x.dtype)
        high_basis = F.one_hot(upper, num_classes=self.knots).to(dtype=x.dtype)
        basis = low_basis * (1.0 - fraction.unsqueeze(-1)) + high_basis * fraction.unsqueeze(-1)
        return torch.einsum("...ik,oik->...o", basis, self.edge_values)


class KernelAttention(nn.Module):
    """Bounded recurrent positive-kernel attention with fast-weight memory.

    Queries and keys are positive softmax feature maps. The memory stores a
    leaky average of key/value outer products and key mass; division by query
    key mass makes each output a bounded weighted average of tanh values.
    """

    def __init__(self, input_size: int, key_size: int = 8, value_size: int = 8):
        super().__init__()
        if min(input_size, key_size, value_size) < 1:
            raise ValueError("input_size, key_size, and value_size must be positive")
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.query_projection = nn.Linear(input_size, key_size)
        self.key_projection = nn.Linear(input_size, key_size)
        self.value_projection = nn.Linear(input_size, value_size)
        self.decay_logit = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))

    def _decay(self) -> Tensor:
        return 0.9999 * torch.sigmoid(self.decay_logit)

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Process a sequence chunk, optionally continuing fast-weight state.

        Args:
            x: Tensor shaped ``(..., time, input_size)``.
            state: Optional ``(key_value_memory, key_mass)`` state. Passing None
                starts a fresh sequence; pass the returned pair to continue.
        """
        if x.ndim < 2 or x.shape[-1] != self.input_size:
            raise ValueError(f"x must have shape (..., time, {self.input_size})")
        leading = x.shape[:-2]
        if state is None:
            memory = x.new_zeros((*leading, self.key_size, self.value_size))
            key_mass = x.new_zeros((*leading, self.key_size))
        else:
            memory, key_mass = state
            if memory.shape != (*leading, self.key_size, self.value_size):
                raise ValueError("memory shape does not match x's leading dimensions")
            if key_mass.shape != (*leading, self.key_size):
                raise ValueError("key_mass shape does not match x's leading dimensions")

        query = torch.softmax(self.query_projection(x), dim=-1)
        key = torch.softmax(self.key_projection(x), dim=-1)
        value = torch.tanh(self.value_projection(x))
        decay = self._decay().to(dtype=x.dtype, device=x.device)
        keep, write = decay, 1.0 - decay
        outputs: list[Tensor] = []
        for query_t, key_t, value_t in zip(
            query.unbind(dim=-2), key.unbind(dim=-2), value.unbind(dim=-2), strict=True
        ):
            memory = keep * memory + write * key_t.unsqueeze(-1) * value_t.unsqueeze(-2)
            key_mass = keep * key_mass + write * key_t
            numerator = torch.matmul(query_t.unsqueeze(-2), memory).squeeze(-2)
            denominator = (query_t * key_mass).sum(dim=-1, keepdim=True).clamp_min(1e-12)
            outputs.append(numerator / denominator)
        if outputs:
            output = torch.stack(outputs, dim=-2)
        else:
            output = x.new_empty((*leading, 0, self.value_size))
        return output, (memory, key_mass)


def mlp_hidden_for_budget(input_size: int, output_size: int, parameter_budget: int) -> int:
    """Choose the largest one-hidden-layer MLP width within a parameter budget."""
    if min(input_size, output_size, parameter_budget) < 1:
        raise ValueError("sizes and parameter_budget must be positive")
    minimum = input_size + 2 * output_size + 1
    if parameter_budget < minimum:
        raise ValueError(f"parameter_budget must be at least {minimum}")
    return max(1, (parameter_budget - output_size) // (input_size + output_size + 1))


class PointwiseMLPControl(nn.Module):
    """Two-layer per-step MLP control, with no recurrent state."""

    def __init__(self, input_size: int, output_size: int, parameter_budget: int):
        super().__init__()
        hidden_size = mlp_hidden_for_budget(input_size, output_size, parameter_budget)
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the same learned map independently at every sequence position."""
        return self.network(x)
