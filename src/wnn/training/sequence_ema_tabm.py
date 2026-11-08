"""Causal EMA summaries and a custom TabM-style rank-one ensemble regressor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wnn.training.sequence_model_helpers import (
    HintProjector,
    validate_sequence_fp32_environment,
    validate_sequence_input_shape,
)


@dataclass(frozen=True)
class EMASequenceState:
    """Causal multiscale means and the previous standardized input row."""

    means: Tensor
    previous: Tensor

    def detach(self) -> EMASequenceState:
        """Detach the state without changing its causal clock."""
        return EMASequenceState(self.means.detach(), self.previous.detach())


def causal_ema_scan(values: Tensor, decay: Tensor, initial: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate first-order EMA recurrences with a stable associative prefix scan.

    Args:
        values: Current projected inputs, shaped [batch, rows, channels].
        decay: Fixed or fitting-learned causal decay values, shaped [scales].
        initial: Previous EMA states, shaped [batch, scales, channels].

    Returns:
        Every current-inclusive EMA and its final state. No division by tiny
        decay powers or future-dependent sequence normalization is used.
    """
    scanned = (1.0 - decay)[None, None, :, None] * values[:, :, None, :]
    offset = 1
    while offset < values.shape[1]:
        carried = decay.pow(offset)[None, None, :, None] * scanned[:, :-offset]
        scanned = torch.cat((scanned[:, :offset], scanned[:, offset:] + carried), dim=1)
        offset *= 2
    positions = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype)
    powers = decay[None, :].pow(positions[:, None])
    scanned = scanned + powers[None, :, :, None] * initial[:, None]
    return scanned, scanned[:, -1].clone()


class NumericalEmbedding(nn.Module):
    """Learn independent smooth scalar embeddings using fitting labels only."""

    def __init__(self, features: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(features, embedding_dim))
        nn.init.normal_(self.weight, std=1.0)

    def forward(self, values: Tensor) -> Tensor:
        """Embed each current/causal-summary scalar independently."""
        return F.gelu(values[..., None] * self.weight + self.bias).flatten(-2)


class RankOneEnsembleLinear(nn.Module):
    """Shared dense weights with member-specific rank-one input/output factors."""

    def __init__(self, n_in: int, n_out: int, members: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.input_scale = nn.Parameter(torch.empty(members, n_in))
        self.output_scale = nn.Parameter(torch.empty(members, n_out))
        self.bias = nn.Parameter(torch.zeros(members, n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.input_scale.bernoulli_(0.5).mul_(2).sub_(1)
            self.output_scale.bernoulli_(0.5).mul_(2).sub_(1)

    def forward(self, values: Tensor) -> Tensor:
        """Apply shared weights independently to each ensemble member."""
        return F.linear(values * self.input_scale, self.weight) * self.output_scale + self.bias


class EMATabMStyleSequenceAdapter(nn.Module):
    """EMA/difference features with a parameter-efficient ensemble MLP.

    This is explicitly a temporal TabM-style adaptation. It does not import the
    official TabM package, reproduce its initialization/embedding design, or use
    its memberwise training loss: the canonical supervised loss sees the mean
    two-target prediction. Four fixed row half-lives are 32/256/2048/8192.
    """

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        hidden = int(config["hidden"])
        projection = int(config["ema_projection"])
        self.members = int(config["ensemble_members"])
        self.register_buffer("mean", torch.zeros(112))
        self.register_buffer("std", torch.ones(112))
        half_lives = torch.tensor([32.0, 256.0, 2048.0, 8192.0])
        self.register_buffer("decay", torch.exp(-math.log(2.0) / half_lives))
        self.projection = nn.Linear(112, projection)
        features = 224 + 4 * projection
        embedding_dim = int(config["numerical_embedding_dim"])
        self.numerical_embedding = NumericalEmbedding(features, embedding_dim)
        widths = [features * embedding_dim, *([hidden] * int(config["layers"])), 2]
        self.ensemble = nn.ModuleList(
            RankOneEnsembleLinear(n_in, n_out, self.members)
            for n_in, n_out in zip(widths[:-1], widths[1:], strict=True)
        )
        self.hint_projector = HintProjector(hidden, int(config["hint_dim"]), "learned")

    def forward(
        self, values: Tensor, state: EMASequenceState | None = None
    ) -> tuple[Tensor, Tensor, EMASequenceState]:
        """Advance all EMA states, emit row predictions, and preserve the last row."""
        validate_sequence_input_shape(values)
        validate_sequence_fp32_environment(self, values, "fp32")
        normalized = (values - self.mean) / self.std
        encoded = self.projection(normalized)
        initial = encoded.new_zeros((values.shape[0], 4, encoded.shape[-1]))
        previous = normalized.new_zeros((values.shape[0], 112))
        if state is not None:
            if not isinstance(state, EMASequenceState):
                raise TypeError("EMA/TabM-style model requires EMASequenceState")
            if state.means.shape != initial.shape or state.previous.shape != previous.shape:
                raise ValueError("EMA state batch/channel shape differs from this model")
            for tensor in (state.means, state.previous):
                if tensor.dtype != torch.float32 or tensor.device != values.device:
                    raise ValueError("EMA state must use the input device and float32")
            initial, previous = state.means, state.previous
        summaries, final = causal_ema_scan(encoded, self.decay, initial)
        history = torch.cat((previous[:, None], normalized[:, :-1]), dim=1)
        features = torch.cat((normalized, normalized - history, summaries.flatten(-2)), dim=-1)
        hidden = self.numerical_embedding(features).unsqueeze(-2).expand(-1, -1, self.members, -1)
        for layer in self.ensemble[:-1]:
            hidden = F.gelu(layer(hidden))
        prediction = self.ensemble[-1](hidden).mean(dim=-2)
        return (
            prediction,
            self.hint_projector(hidden.mean(dim=-2)),
            EMASequenceState(final, normalized[:, -1].clone()),
        )
