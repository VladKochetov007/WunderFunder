"""Past-only event-normalized cumulative volume innovation for GRU studies."""

from __future__ import annotations

import torch
from torch import nn

from wnn.event_content import ENDPOINT
from wnn.nn import RNNModel

SURPRISE_WIDTH = 22
SURPRISE_RANK = 8


def surprise_volume_features(raw: torch.Tensor) -> torch.Tensor:
    """Build endpoint-reset cumulative volume innovations using causal inputs.

    At each row, raw volume differences are scaled by the RMS of earlier
    changed rows in the same sequence, accumulated since the latest endpoint,
    and divided by the square root of the number of changed rows in that run.
    Endpoint rows reset the run and contribute a zero innovation. No labels,
    masks, later rows, or cross-sequence state are read.

    Args:
        raw: Float32 tensor with shape ``[batch, time, >=112]``.

    Returns:
        Float32 tensor with shape ``[batch, time, 22]``.

    Raises:
        ValueError: If the tensor has the wrong shape or dtype.
    """
    if raw.ndim != 3 or raw.shape[-1] < 112 or raw.shape[1] < 1:
        raise ValueError("expected nonempty [batch,time,>=112] raw input")
    if raw.dtype != torch.float32:
        raise ValueError("surprise-volume features require original float32 inputs")

    volume = raw[..., 22:44]
    endpoint = raw[..., 110] == float(ENDPOINT)
    previous = torch.cat((volume[:, :1], volume[:, :-1]), dim=1)
    delta = volume - previous
    delta = torch.where(endpoint[..., None], torch.zeros_like(delta), delta)
    changed = delta.ne(0).any(dim=-1) & ~endpoint

    event_float = changed.to(raw.dtype)
    squared_event_size = delta.square().mean(dim=-1) * event_float
    prior_sq = torch.cumsum(squared_event_size, dim=1) - squared_event_size
    prior_count = torch.cumsum(event_float, dim=1) - event_float
    prior_rms = torch.sqrt(prior_sq / prior_count.clamp_min(1.0) + 1e-6)
    scale = torch.where(prior_count > 0, prior_rms, torch.ones_like(prior_rms))
    innovation = delta / scale[..., None]

    positions = torch.arange(raw.shape[1], device=raw.device).expand(raw.shape[:2])
    latest_endpoint = torch.cummax(
        torch.where(endpoint, positions, torch.full_like(positions, -1)), dim=1
    ).values
    cumulative = torch.cumsum(innovation, dim=1)
    event_count = torch.cumsum(event_float, dim=1)
    safe_endpoint = latest_endpoint.clamp_min(0)
    endpoint_sum = torch.gather(cumulative, 1, safe_endpoint[..., None].expand_as(cumulative))
    endpoint_count = torch.gather(event_count, 1, safe_endpoint)
    has_endpoint = latest_endpoint >= 0
    run_sum = torch.where(has_endpoint[..., None], cumulative - endpoint_sum, cumulative)
    run_count = torch.where(has_endpoint, event_count - endpoint_count, event_count)
    return run_sum / run_count.clamp_min(1.0).sqrt()[..., None]


class SurpriseVolumeGRU(RNNModel):
    """E427-shaped overlay whose rank-8 z-gate sees surprise-volume context."""

    ARMS = ("aligned", "stale", "noop")

    def __init__(
        self,
        n_in: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
        n_out: int = 23,
        *,
        arm: str,
        rank: int = SURPRISE_RANK,
    ) -> None:
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        if n_in < 112 or layers != 2 or rank != SURPRISE_RANK:
            raise ValueError("surprise-volume overlay requires >=112 inputs, 2 layers, rank 8")
        super().__init__(n_in, hidden, layers, dropout, n_out, cell="gru", enc="linear")
        self.gru = nn.GRU(
            hidden + rank,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.surprise_projection = nn.Linear(SURPRISE_WIDTH, rank)
        self.arm = arm
        self.gru.weight_ih_l0.register_hook(self._mask_fixed_gate_gradients)
        with torch.no_grad():
            self.gru.weight_ih_l0[:, hidden:].zero_()
        self.project_constraints_()

    @property
    def hidden(self) -> int:
        """Return the parent model's recurrent width."""
        return self.gru.hidden_size

    def _mask_fixed_gate_gradients(self, gradient: torch.Tensor) -> torch.Tensor:
        """Keep extra input columns out of reset and candidate gate rows."""
        masked = gradient.clone()
        masked[: self.hidden, self.hidden :] = 0
        masked[2 * self.hidden :, self.hidden :] = 0
        return masked

    def project_constraints_(self) -> None:
        """Restore exact zeros in the constrained reset/candidate coordinates."""
        with torch.no_grad():
            self.gru.weight_ih_l0[: self.hidden, self.hidden :].zero_()
            self.gru.weight_ih_l0[2 * self.hidden :, self.hidden :].zero_()

    def constraints_hold(self) -> bool:
        """Check the structural zero invariant after optimizer updates."""
        weight = self.gru.weight_ih_l0.detach()
        return bool(
            torch.count_nonzero(weight[: self.hidden, self.hidden :]).item() == 0
            and torch.count_nonzero(weight[2 * self.hidden :, self.hidden :]).item() == 0
        )

    def load_parent_state_dict(self, parent_state: dict[str, torch.Tensor]) -> None:
        """Copy a standard parent model exactly and append zero GRU columns."""
        own = self.state_dict()
        added = {"surprise_projection.weight", "surprise_projection.bias"}
        if set(parent_state) != set(own) - added:
            raise ValueError("parent state keys do not match the surprise-volume overlay")
        copied: dict[str, torch.Tensor] = {}
        for name, target in own.items():
            if name in added:
                continue
            source = parent_state[name]
            if name == "gru.weight_ih_l0":
                if source.shape != (3 * self.hidden, self.hidden):
                    raise ValueError("parent GRU layer-0 input matrix has an unexpected shape")
                expanded = target.new_zeros(target.shape)
                expanded[:, : self.hidden].copy_(source)
                copied[name] = expanded
            elif source.shape != target.shape:
                raise ValueError(f"parent tensor shape mismatch for {name}")
            else:
                copied[name] = source
        current = self.state_dict()
        copied.update({name: current[name] for name in added})
        self.load_state_dict(copied, strict=True)
        self.project_constraints_()

    @staticmethod
    def stale_features(context: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Hold the previous completed run's terminal feature after 2 endpoints."""
        if context.ndim != 3 or raw.ndim != 3 or context.shape[:2] != raw.shape[:2]:
            raise ValueError("raw and surprise context must share batch/time axes")
        batch, steps, _ = raw.shape
        positions = torch.arange(steps, device=raw.device).expand(batch, -1)
        endpoint = raw[..., 110] == float(ENDPOINT)
        latest = torch.cummax(
            torch.where(endpoint, positions, torch.full_like(positions, -1)), dim=1
        ).values
        count = torch.cumsum(endpoint.to(torch.long), dim=1)
        source = latest - 1
        safe_source = source.clamp_min(0)
        prior = torch.gather(context, 1, safe_source[..., None].expand_as(context))
        return torch.where((count >= 2)[..., None], prior, torch.zeros_like(prior))

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        rows: slice,
        state: torch.Tensor | None = None,
        *,
        full_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a chunk, using context computed from the unsliced sequence."""
        if not self.constraints_hold():
            raise RuntimeError("project surprise-volume structural zeros before forward")
        context = (
            surprise_volume_features(inputs[..., :112]) if full_context is None else full_context
        )
        if context.shape != (*inputs.shape[:2], SURPRISE_WIDTH):
            raise ValueError("full surprise context has the wrong [batch,time,22] shape")
        if self.arm == "noop":
            projected = inputs.new_zeros((*inputs[:, rows].shape[:2], SURPRISE_RANK))
        else:
            selected = (
                context
                if self.arm == "aligned"
                else self.stale_features(context, inputs[..., :112])
            )
            projected = self.surprise_projection(selected[:, rows])
        encoded = self.encode((inputs[:, rows] - self.mean) / self.std)
        recurrent, next_state = self.gru(torch.cat((encoded, projected), dim=-1), state)
        output = self.head(torch.cat((recurrent, encoded), dim=-1))
        return output, next_state

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        full_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a complete sequence and return predictions plus GRU state."""
        return self.forward_chunk(inputs, slice(None), state, full_context=full_context)
