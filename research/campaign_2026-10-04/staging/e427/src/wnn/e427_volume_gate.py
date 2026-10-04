"""E427 causal volume-conditioned update gate (campaign overlay only).

The root model/trainer are deliberately untouched.  This module keeps the
parent input projection, normalization, stacked-GRU topology, and 23-output
head, adding a rank-8 context only to first-layer GRU update-gate logits.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping

import torch
from torch import nn

from wnn.nn import RNNModel
from wnn.research_features import ENDPOINT, research_features_torch


class E427VolumeGateModel(RNNModel):
    """Parent RNNModel with aligned/stale/no-op rank-8 z-gate context."""

    ARMS = ("aligned", "stale", "noop")
    CONTEXT_WIDTH = 28

    def __init__(
        self,
        n_in: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
        n_out: int = 23,
        *,
        arm: str,
        fallback_mean: list[float] | tuple[float, ...],
        rank: int = 8,
    ) -> None:
        if arm not in self.ARMS:
            raise ValueError(f"E427 arm must be one of {self.ARMS}, got {arm!r}")
        if hidden < 1 or layers != 2 or rank != 8:
            raise ValueError("E427 requires a two-layer GRU and fixed rank 8")
        if n_out != 23:
            raise ValueError("E427 requires the original 23-output combo objective")
        if len(fallback_mean) != self.CONTEXT_WIDTH:
            raise ValueError("E427 fallback mean must contain 28 volume28 values")
        super().__init__(n_in, hidden, layers, dropout, n_out, cell="gru", enc="linear")
        if n_in < 112:
            raise ValueError("E427 parent input must preserve the 112 raw input channels")

        # nn.GRU supports a wider layer-0 input while upper layers retain hidden
        # width.  Its sequence path therefore stays on the fused PyTorch GRU.
        self.gru = nn.GRU(
            hidden + rank,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.gate_context = nn.Linear(self.CONTEXT_WIDTH, rank)
        self.register_buffer("volume_fallback", torch.tensor(fallback_mean, dtype=torch.float32))
        if not torch.isfinite(self.volume_fallback).all():
            raise ValueError("E427 fallback mean must be finite")
        self.e427_arm = arm
        self.e427_rank = rank
        self._context_ref: weakref.ReferenceType[torch.Tensor] | None = None
        self._context_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.gru.weight_ih_l0.register_hook(self._mask_forbidden_gradients)
        self.project_constraints_()

    def _mask_forbidden_gradients(self, gradient: torch.Tensor) -> torch.Tensor:
        """Gradient hygiene only; post-optimizer projection enforces the invariant."""
        masked = gradient.clone()
        masked[: self.hidden, self.hidden :] = 0
        masked[2 * self.hidden :, self.hidden :] = 0
        return masked

    @property
    def hidden(self) -> int:
        return self.gru.hidden_size

    def project_constraints_(self) -> None:
        """Restore structural r/n zeros even after Muon's matrix polar update."""
        with torch.no_grad():
            self.gru.weight_ih_l0[: self.hidden, self.hidden :].zero_()
            self.gru.weight_ih_l0[2 * self.hidden :, self.hidden :].zero_()

    def constraints_hold(self) -> bool:
        weight = self.gru.weight_ih_l0.detach()
        return bool(
            torch.count_nonzero(weight[: self.hidden, self.hidden :]).item() == 0
            and torch.count_nonzero(weight[2 * self.hidden :, self.hidden :]).item() == 0
        )

    @property
    def added_parameter_counts(self) -> dict[str, int]:
        """Report allocated and effective counts; r/n extension coordinates are fixed."""
        allocated_gru = 3 * self.hidden * self.e427_rank
        constrained_gru = 2 * self.hidden * self.e427_rank
        context_projection = self.CONTEXT_WIDTH * self.e427_rank + self.e427_rank
        return {
            "allocated_added": allocated_gru + context_projection,
            "structurally_fixed": constrained_gru,
            "effective_added": allocated_gru - constrained_gru + context_projection,
        }

    def load_parent_state_dict(self, parent_state: Mapping[str, torch.Tensor]) -> None:
        """Copy a standard parent state exactly and append zero recurrent columns."""
        own = self.state_dict()
        expected_parent = set(own) - {"gate_context.weight", "gate_context.bias", "volume_fallback"}
        if set(parent_state) != expected_parent:
            missing = sorted(expected_parent - set(parent_state))
            extra = sorted(set(parent_state) - expected_parent)
            raise ValueError(f"E427 parent state keys mismatch; missing={missing}, extra={extra}")
        copied: dict[str, torch.Tensor] = {}
        for name, target in own.items():
            if name in {"gate_context.weight", "gate_context.bias", "volume_fallback"}:
                continue
            source = parent_state[name]
            if name == "gru.weight_ih_l0":
                if source.shape != (3 * self.hidden, self.hidden):
                    raise ValueError("parent layer-0 GRU input matrix does not match E427 hidden width")
                expanded = target.new_zeros(target.shape)
                expanded[:, : self.hidden].copy_(source)
                copied[name] = expanded
            elif source.shape != target.shape:
                raise ValueError(f"parent tensor shape mismatch for {name}: {source.shape} vs {target.shape}")
            else:
                copied[name] = source
        self.load_state_dict({**copied,
                              "gate_context.weight": own["gate_context.weight"],
                              "gate_context.bias": own["gate_context.bias"],
                              "volume_fallback": own["volume_fallback"]}, strict=True)
        self.project_constraints_()

    @staticmethod
    def stale_context(
        volume28: torch.Tensor,
        raw: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Previous completed-cycle terminal context, with fit-mean warmup fallback."""
        if volume28.ndim != 3 or raw.ndim != 3 or volume28.shape[:2] != raw.shape[:2]:
            raise ValueError("E427 context and raw rows must have matching [batch,time] axes")
        if raw.shape[-1] < 112 or volume28.shape[-1] != 28 or fallback.shape != (28,):
            raise ValueError("E427 stale-context input widths must be raw>=112, context=28")
        batch, steps, _ = raw.shape
        positions = torch.arange(steps, device=raw.device, dtype=torch.long)[None].expand(batch, -1)
        endpoint = raw[..., 110] == float(ENDPOINT)
        marker_positions = torch.where(endpoint, positions, torch.full_like(positions, -1))
        latest_marker = torch.cummax(marker_positions, dim=1).values
        marker_count = torch.cumsum(endpoint.to(torch.long), dim=1)
        previous_row = (latest_marker - 1).clamp_min(0)
        indices = previous_row[..., None].expand(batch, steps, 28)
        prior_cycle = torch.gather(volume28, 1, indices)
        return torch.where((marker_count >= 2)[..., None], prior_cycle, fallback[None, None, :])

    def _full_contexts(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._context_ref is not None and self._context_ref() is inputs and self._context_cache is not None:
            return self._context_cache
        if inputs.ndim != 3 or inputs.shape[-1] != self.mean.numel():
            raise ValueError("E427 expects the full parent-preprocessed input tensor")
        raw = inputs[..., :112].float()
        if raw.shape[1] < 1:
            raise ValueError("E427 cannot process an empty sequence")
        volume28 = research_features_torch(raw, "volume28")
        stale = self.stale_context(volume28, raw, self.volume_fallback.to(raw.device))
        self._context_ref = weakref.ref(inputs)
        self._context_cache = (volume28, stale)
        return volume28, stale

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        rows: slice,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.constraints_hold():
            raise RuntimeError("E427 forbidden reset/candidate context weights were not projected")
        context_aligned, context_stale = self._full_contexts(inputs)
        context = context_aligned if self.e427_arm == "aligned" else context_stale
        encoded = self.encode((inputs[:, rows] - self.mean) / self.std)
        selected_context = context[:, rows]
        if self.e427_arm == "noop":
            selected_context = torch.zeros_like(selected_context)
        projected_context = self.gate_context(selected_context)
        recurrent_input = torch.cat((encoded, projected_context), dim=-1)
        recurrent, next_state = self.gru(recurrent_input, state)
        output = self.head(torch.cat((recurrent, encoded), dim=-1))
        return output, next_state

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_chunk(inputs, slice(None), state)
