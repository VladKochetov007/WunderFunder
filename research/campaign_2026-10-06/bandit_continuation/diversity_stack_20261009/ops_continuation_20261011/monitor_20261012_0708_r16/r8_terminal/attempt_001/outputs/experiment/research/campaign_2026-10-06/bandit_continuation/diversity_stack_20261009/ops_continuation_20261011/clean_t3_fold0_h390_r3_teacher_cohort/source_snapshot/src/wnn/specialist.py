"""Causal specialist recurrence and prefix-conditioned input adapters.

These architectures are research-only. Their tensor states carry across TBPTT
chunks, and every context used at row ``t`` is computed from rows ``<= t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

PREFIX_ROWS = 99


def causal_prefix_mean(
    x: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None = None,
    limit: int = PREFIX_ROWS,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Return a causal running mean, frozen after ``limit`` rows.

    Args:
        x: Input rows with shape ``[batch, time, features]``.
        state: Previous prefix sum and row count for chunked evaluation.
        limit: Maximum number of rows that may affect the prefix context.

    Returns:
        The inclusive running mean for each row and the updated sum/count.

    Raises:
        ValueError: If the input rank, feature width, or prefix limit is invalid.
    """
    if x.ndim != 3:
        raise ValueError("prefix input must have shape [batch, time, features]")
    if limit < 1:
        raise ValueError("prefix limit must be positive")
    batch, time, features = x.shape
    if state is None:
        prefix_sum = x.new_zeros(batch, features, dtype=torch.float32)
        prefix_count = torch.zeros(batch, device=x.device, dtype=torch.long)
    else:
        prefix_sum, prefix_count = state
        if prefix_sum.shape != (batch, features) or prefix_count.shape != (batch,):
            raise ValueError("prefix state dimensions do not match input")
        prefix_sum = prefix_sum.to(device=x.device, dtype=torch.float32)
        prefix_count = prefix_count.to(device=x.device, dtype=torch.long)

    if time == 0:
        return x.new_empty(batch, 0, features), (prefix_sum, prefix_count)

    positions = torch.arange(time, device=x.device)
    remaining = (limit - prefix_count).clamp_min(0)
    accepted = positions[None, :] < remaining[:, None]
    accepted_rows = x.float() * accepted[..., None]
    sums = prefix_sum[:, None, :] + accepted_rows.cumsum(dim=1)
    counts = (prefix_count[:, None] + positions[None, :] + 1).clamp_max(limit)
    means = sums / counts.clamp_min(1)[..., None]
    updated_sum = prefix_sum + accepted_rows.sum(dim=1)
    updated_count = (prefix_count + remaining.clamp_max(time)).clamp_max(limit)
    return means, (updated_sum, updated_count)


def _flat_hash_projection(features: int, experts: int, seed: int) -> torch.Tensor:
    """Make a fixed, deterministic projection used only by the hash control."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(features, experts, generator=generator)
    return value / math.sqrt(features)


class SpecialistGRUModel(nn.Module):
    """Shared GRU with always-updated small recurrent specialists.

    ``learned`` uses a causal top-one router with a straight-through softmax
    gradient. ``hash`` uses a frozen seeded projection of the inclusive
    first-99-row prefix. Routes are consequently sequence-fixed for scored
    rows after the 99-row warmup. ``universal`` averages all residuals.
    All expert GRUs run on every row in every arm, so expert state support and
    recurrent update count match across the routing contrasts.
    """

    def __init__(
        self,
        n_in: int,
        hidden: int = 96,
        expert_hidden: int = 32,
        experts: int = 4,
        layers: int = 1,
        n_out: int = 2,
        route: str = "learned",
        prefix_rows: int = PREFIX_ROWS,
        hash_seed: int = 1729,
    ) -> None:
        super().__init__()
        if min(n_in, hidden, expert_hidden, experts, layers, n_out, prefix_rows) <= 0:
            raise ValueError("specialist dimensions and prefix_rows must be positive")
        if route not in {"learned", "hash", "universal"}:
            raise ValueError("route must be learned, hash, or universal")
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.experts = nn.ModuleList(
            nn.GRU(hidden, expert_hidden, batch_first=True) for _ in range(experts)
        )
        self.prefix_proj = nn.Linear(n_in, max(4, min(16, hidden // 4)))
        route_width = self.prefix_proj.out_features
        self.router = nn.Sequential(
            nn.Linear(route_width, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, experts)
        )
        self.universal_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out)
        )
        self.expert_heads = nn.ModuleList(nn.Linear(expert_hidden, n_out) for _ in range(experts))
        self.register_buffer("hash_projection", _flat_hash_projection(n_in, experts, hash_seed))
        self.expert_hidden = expert_hidden
        self.num_experts = experts
        self.route = route
        self.prefix_rows = prefix_rows

    @property
    def specialist_parameter_count(self) -> int:
        """Count trainable parameters belonging to expert recurrent cells."""
        return sum(
            parameter.numel() for module in self.experts for parameter in module.parameters()
        )

    def route_indices(self, prefix: torch.Tensor) -> torch.Tensor:
        """Return routes from prefix context alone for audits and controls."""
        if prefix.ndim != 3 or prefix.shape[-1] != self.mean.numel():
            raise ValueError("route prefix must have shape [batch, time, input_features]")
        if self.route == "universal":
            return torch.full(prefix.shape[:2], -1, device=prefix.device, dtype=torch.long)
        if self.route == "hash":
            scores = torch.matmul(prefix, self.hash_projection.to(prefix.dtype))
        else:
            scores = self.router(F.gelu(self.prefix_proj(prefix)))
        return scores.argmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Predict a chunk and return flat, detach-safe recurrent state."""
        if x.ndim != 3:
            raise ValueError("specialist input must have shape [batch, time, features]")
        if state is not None and len(state) != self.num_experts + 3:
            raise ValueError("specialist state must contain shared, expert, and prefix tensors")
        normalized = (x - self.mean) / self.std
        prefix_state = None if state is None else (state[-2], state[-1])
        prefix, (prefix_sum, prefix_count) = causal_prefix_mean(
            normalized, prefix_state, self.prefix_rows
        )
        projected = F.gelu(self.inp(normalized))
        shared, shared_state = self.gru(projected, None if state is None else state[0])
        expert_values: list[torch.Tensor] = []
        next_states = [shared_state]
        for index, expert in enumerate(self.experts):
            previous = None if state is None else state[index + 1]
            value, expert_state = expert(shared, previous)
            expert_values.append(value)
            next_states.append(expert_state)

        expert_tensor = torch.stack(expert_values, dim=2)
        base = self.universal_head(torch.cat([shared, projected], dim=-1))
        expert_outputs = torch.stack(
            [head(expert_tensor[:, :, index]) for index, head in enumerate(self.expert_heads)],
            dim=2,
        )
        if self.route == "universal":
            gate = torch.full_like(expert_outputs[..., :1], 1.0 / self.num_experts)
        elif self.route == "hash":
            selected = self.route_indices(prefix)
            gate = F.one_hot(selected, num_classes=self.num_experts).to(expert_outputs.dtype)[
                ..., None
            ]
        else:
            logits = self.router(F.gelu(self.prefix_proj(prefix)))
            probabilities = torch.softmax(logits.float(), dim=-1).to(expert_outputs.dtype)
            selected = self.route_indices(prefix)
            hard = F.one_hot(selected, num_classes=self.num_experts).to(expert_outputs.dtype)
            gate = (hard - probabilities).detach() + probabilities
            gate = gate[..., None]
        residual = (expert_outputs * gate).sum(dim=2)
        next_states.extend((prefix_sum, prefix_count))
        return base + residual, tuple(next_states)


class PrefixAdapterGRUModel(nn.Module):
    """GRU with a first-99-row conditioned low-rank input adapter.

    ``conditional`` applies a prefix-conditioned low-rank additive adapter;
    ``unconditional`` supplies a learned global prefix prior to the same
    factors; and ``diagonal`` uses the same factors to gate hidden channels
    elementwise. The three arms have identical parameter counts and state
    dimensions.
    """

    MODES = ("conditional", "unconditional", "diagonal")

    def __init__(
        self,
        n_in: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        n_out: int = 2,
        rank: int = 8,
        mode: str = "conditional",
        prefix_rows: int = PREFIX_ROWS,
    ) -> None:
        super().__init__()
        if min(n_in, hidden, layers, n_out, rank, prefix_rows) <= 0:
            raise ValueError("adapter dimensions and prefix_rows must be positive")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.adapter_input = nn.Linear(n_in, rank, bias=False)
        self.adapter_context = nn.Linear(n_in, rank, bias=False)
        self.adapter_output = nn.Linear(rank, hidden, bias=False)
        self.prefix_prior = nn.Parameter(torch.zeros(n_in))
        self.adapter_scale = nn.Parameter(torch.tensor(0.1))
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out)
        )
        self.rank = rank
        self.mode = mode
        self.prefix_rows = prefix_rows

    def encode(
        self,
        normalized: torch.Tensor,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the selected adapter arm before the recurrent cell."""
        base = F.gelu(self.inp(normalized))
        context = self.prefix_prior if self.mode == "unconditional" else prefix + self.prefix_prior
        interaction = self.adapter_input(normalized) * self.adapter_context(context)
        modulation = self.adapter_output(interaction)
        if self.mode == "diagonal":
            return base * (1.0 + self.adapter_scale * torch.tanh(modulation))
        return base + self.adapter_scale * modulation

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Predict a chunk while carrying both GRU and prefix state."""
        if x.ndim != 3:
            raise ValueError("adapter input must have shape [batch, time, features]")
        if state is not None and len(state) != 3:
            raise ValueError("adapter state must contain GRU state, prefix sum, and count")
        normalized = (x - self.mean) / self.std
        prefix_state = None if state is None else (state[1], state[2])
        prefix, (prefix_sum, prefix_count) = causal_prefix_mean(
            normalized, prefix_state, self.prefix_rows
        )
        encoded = self.encode(normalized, prefix)
        recurrent, next_hidden = self.gru(encoded, None if state is None else state[0])
        output = self.head(torch.cat([recurrent, encoded], dim=-1))
        return output, (next_hidden, prefix_sum, prefix_count)
