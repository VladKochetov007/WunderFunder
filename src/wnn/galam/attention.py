from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from wnn.galam.state import AttentionCache


class RelativeTimeBias(nn.Module):
    """Map absolute timestamp distances to a learned additive bias per head."""

    def __init__(self, hidden_width: int, heads: int, timestamp_scale: float) -> None:
        """Create the Linear-Tanh-Linear bias network."""
        super().__init__()
        self.timestamp_scale = timestamp_scale
        self.hidden = nn.Linear(1, hidden_width)
        self.output = nn.Linear(hidden_width, heads)

    def forward(self, query_times: torch.Tensor, key_times: torch.Tensor) -> torch.Tensor:
        """Return bias shaped batch, heads, queries, keys."""
        distances = (query_times.unsqueeze(-1) - key_times.unsqueeze(-2)).abs()
        features = (distances / self.timestamp_scale).unsqueeze(-1).to(self.hidden.weight.dtype)
        return self.output(torch.tanh(self.hidden(features))).permute(0, 3, 1, 2)


class TemporalSelfAttention(nn.Module):
    """Causal temporal attention with QK RMSNorm, relative bias, and bounded cache."""

    def __init__(
        self, width: int, heads: int, history: int, bias_width: int, timestamp_scale: float
    ) -> None:
        """Create projected Q/K/V paths and a fixed-size causal context."""
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.history = history
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.query_norm = nn.RMSNorm(self.head_width)
        self.key_norm = nn.RMSNorm(self.head_width)
        self.relative_bias = RelativeTimeBias(bias_width, heads, timestamp_scale)
        self.output = nn.Linear(width, width)
        self._monitor_observer: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None
        ) = None

    def forward_step(
        self, x: torch.Tensor, timestamp: torch.Tensor, cache: AttentionCache | None
    ) -> tuple[torch.Tensor, AttentionCache]:
        """Attend to at most `history` total keys and return the bounded cache."""
        batch = x.shape[0]
        query = self._project(self.query(x), self.query_norm).unsqueeze(2)
        key = self._project(self.key(x), self.key_norm).unsqueeze(2)
        value = self._project(self.value(x)).unsqueeze(2)
        times = timestamp.reshape(batch, 1)
        if cache is not None:
            key = torch.cat((cache.keys, key), dim=2)
            value = torch.cat((cache.values, value), dim=2)
            key_times = torch.cat((cache.timestamps, times), dim=1)
        else:
            key_times = times
        if key.shape[2] > self.history:
            key = key[:, :, -self.history :]
            value = value[:, :, -self.history :]
            key_times = key_times[:, -self.history :]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_width)
        bias = self.relative_bias(times, key_times)
        scores = scores + bias
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, self.width)
        keep = slice(-self.history, None)
        next_cache = AttentionCache(key[:, :, keep], value[:, :, keep], key_times[:, keep])
        result = self.output(attended)
        if self._monitor_observer is not None:
            self._monitor_observer(weights, bias, times, key_times)
        return result, next_cache

    def _project(self, values: torch.Tensor, norm: nn.RMSNorm | None = None) -> torch.Tensor:
        """Split the feature dimension into heads and apply the declared QK norm."""
        batch = values.shape[0]
        heads = values.reshape(batch, self.heads, self.head_width)
        return norm(heads) if norm is not None else heads
