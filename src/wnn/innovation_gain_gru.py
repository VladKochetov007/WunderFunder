"""Innovation-conditioned observer adapter around the fused GRU recurrence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wnn.nn import RNNReadout


class InnovationGainGRU(nn.Module):
    """Add a causal latent predict-correct path to a standard GRU.

    The adapter predicts a latent measurement from the preceding two encoded
    rows, forms its innovation, and applies a bounded learned diagonal gain.
    The corrected latent is injected into the ordinary fused GRU input and
    readout. ``context_only`` keeps identical capacity but hides the innovation
    from the gain network, providing a matched mechanism control.

    The gain adapter is not a calibrated Bayesian filter: its latent coordinate
    system and gain are learned from the task objective. The ordinary GRU core
    still carries information across rows and trainer chunks.
    """

    MODES = ("innovation", "context_only")

    def __init__(
        self,
        n_in: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
        n_out: int = 2,
        state_width: int = 64,
        gain_conditioning: str = "innovation",
    ) -> None:
        super().__init__()
        if min(n_in, hidden, layers, n_out, state_width) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if gain_conditioning not in self.MODES:
            raise ValueError(f"gain_conditioning must be one of {self.MODES}")

        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = RNNReadout(2 * hidden, hidden, n_out, 1, nn.GELU())

        gain_hidden = max(32, hidden // 4)
        self.measurement = nn.Linear(hidden, state_width)
        self.trend_logit = nn.Parameter(torch.zeros(state_width))
        self.base_gain_logit = nn.Parameter(torch.zeros(state_width))
        self.gain_net = nn.Sequential(
            nn.Linear(2 * state_width, gain_hidden),
            nn.GELU(),
            nn.Linear(gain_hidden, state_width),
        )
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.zeros_(self.gain_net[-1].bias)

        self.correction = nn.Linear(2 * state_width, hidden)
        self.gain_readout = nn.Linear(2 * state_width, n_out)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.gain_readout.weight)
        nn.init.zeros_(self.gain_readout.bias)
        self.state_width = state_width
        self.gain_conditioning = gain_conditioning

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Run a chunk, carrying GRU state and two prior latent measurements."""
        if x.ndim != 3 or x.shape[-1] != self.mean.numel():
            raise ValueError("input must have shape [batch, time, configured_features]")
        batch, time, _ = x.shape
        if time == 0:
            raise ValueError("empty sequence chunks are not supported")
        encoded = F.gelu(self.inp((x - self.mean) / self.std))
        observation = self.measurement(encoded)
        if state is None:
            recurrent_state = None
            previous = observation.new_zeros(batch, self.state_width)
            previous_previous = torch.zeros_like(previous)
            seen = torch.zeros(batch, device=x.device, dtype=torch.long)
        else:
            if not isinstance(state, tuple) or len(state) != 4:
                raise ValueError("state must contain GRU, two observer, and row-count states")
            recurrent_state, previous, previous_previous, seen = state
            expected = (batch, self.state_width)
            if previous.shape != expected or previous_previous.shape != expected:
                raise ValueError("observer state dimensions do not match the current batch")
            if seen.shape != (batch,):
                raise ValueError("observer row-count state does not match the current batch")

        offsets = torch.arange(time, device=x.device, dtype=seen.dtype)[None, :]
        has_previous = seen[:, None] + offsets >= 1
        has_two_previous = seen[:, None] + offsets >= 2
        previous_rows = torch.cat((previous[:, None, :], observation[:, :-1]), dim=1)
        prior_previous_rows = torch.cat(
            (previous_previous[:, None, :], previous_rows[:, :-1]), dim=1
        )
        second_lag = torch.where(has_two_previous[..., None], prior_previous_rows, previous_rows)
        prior = previous_rows + torch.sigmoid(self.trend_logit) * (previous_rows - second_lag)
        prior = torch.where(has_previous[..., None], prior, observation)
        innovation = torch.where(has_previous[..., None], observation - prior, 0.0)

        gain_innovation = (
            innovation if self.gain_conditioning == "innovation" else torch.zeros_like(innovation)
        )
        gain_logits = self.gain_net(torch.cat((observation, gain_innovation), dim=-1))
        gain = torch.sigmoid(self.base_gain_logit + gain_logits)
        weighted_innovation = gain * innovation
        posterior = prior + weighted_innovation
        observer_context = torch.cat((posterior, weighted_innovation), dim=-1)

        recurrent_input = encoded + self.correction(observer_context)
        recurrent, recurrent_state = self.gru(recurrent_input, recurrent_state)
        output = self.head(torch.cat((recurrent, encoded), dim=-1))
        output = output + self.gain_readout(observer_context)

        next_previous_previous = previous if time == 1 else observation[:, -2]
        next_state = (
            recurrent_state,
            observation[:, -1],
            next_previous_previous,
            (seen + time).clamp_max(2),
        )
        return output, next_state
