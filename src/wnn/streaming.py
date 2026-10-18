"""One-row-per-call graph: event features and GRU with explicit state tensors.

Exported to ONNX so the submission runs a single ``session.run`` per row, like
the organizer baseline. State: previous row, step counter, last-event steps for
(i0 price, i0 trade, i1 price), held i0 price deltas, GRU hidden, first-row flag.
Must equal ``wnn.features.event_features`` + ``wnn.nn.RNNModel`` on a sequence.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from wnn.data import I0, I1
from wnn.kalman_group_features import (
    features_torch_step as kalman_group_features_torch_step,
)
from wnn.kalman_group_features import (
    initial_stream_state_torch as initial_kalman_group_state_torch,
)
from wnn.research_features import (
    KALMAN_GROUP_MODE,
    MOMENT_FEATURE_MODES,
    causal_moment_features_torch_step,
    initial_causal_moment_state_torch,
    research_feature_control_width,
    validate_causal_moment_prior,
)

_CAP = 10_000.0


def initial_state(
    model: nn.Module,
    research_features: str = "none",
    prior: Mapping[str, object] | None = None,
    control: str = "none",
) -> list[torch.Tensor]:
    """Create reset state for the event graph and optional causal moment branch."""
    research_feature_control_width(research_features, control)
    layers = model.gru.num_layers
    hidden = model.gru.hidden_size
    device, dtype = model.mean.device, model.mean.dtype
    state = [
        torch.zeros(1, 112, device=device, dtype=dtype),  # prev row
        torch.zeros(1, device=device, dtype=dtype),  # step
        torch.zeros(1, 3, device=device, dtype=dtype),  # last event step: i0 px, i0 trade, i1 px
        torch.zeros(1, 22, device=device, dtype=dtype),  # held i0 price deltas at last price event
        torch.zeros(layers, 1, hidden, device=device, dtype=dtype),  # GRU hidden
        torch.ones(1, device=device, dtype=dtype),  # first row flag
    ]
    if research_features in MOMENT_FEATURE_MODES:
        if prior is None:
            raise ValueError("causal moment streaming requires a fitting-only prior")
        state.append(initial_causal_moment_state_torch(prior, device=device))
    elif research_features == KALMAN_GROUP_MODE:
        state.append(initial_kalman_group_state_torch(device=device, dtype=torch.float64))
    elif research_features != "none":
        raise ValueError(
            f"streaming graph does not support research features {research_features!r}"
        )
    return state


class StreamingGraph(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        research_features: str = "none",
        prior: Mapping[str, object] | None = None,
        control: str = "none",
    ) -> None:
        super().__init__()
        self.model = model
        self.research_features = research_features
        self.research_feature_control = control
        research_feature_control_width(research_features, control)
        self.prior = None
        self.register_buffer("kalman_group_initial_state", torch.empty(0))
        if research_features in MOMENT_FEATURE_MODES:
            if prior is None:
                raise ValueError("causal moment streaming requires a fitting-only prior")
            self.prior = validate_causal_moment_prior(prior)
            self.register_buffer(
                "moment_initial_state", initial_causal_moment_state_torch(self.prior)
            )
        elif research_features == KALMAN_GROUP_MODE:
            self.kalman_group_initial_state = initial_kalman_group_state_torch()
        elif research_features != "none":
            raise ValueError(
                f"streaming graph does not support research features {research_features!r}"
            )

    def forward(
        self,
        row: torch.Tensor,
        prev: torch.Tensor,
        step: torch.Tensor,
        last: torch.Tensor,
        held: torch.Tensor,
        h: torch.Tensor,
        first: torch.Tensor,
        moment_state: torch.Tensor | None = None,
        kalman_group_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        prev = torch.where(first[:, None] > 0.5, row, prev)
        d = row - prev
        ne = (prev != row).to(row.dtype)
        f_px = ne[:, I0.bid_px.start] * ne[:, I0.ask_px.start]
        f_bv = ne[:, I0.bid_vol].amax(-1)
        f_av = ne[:, I0.ask_vol].amax(-1)
        f_tp = ne[:, I0.trade_px].amax(-1)
        f_tv = ne[:, I0.trade_vol].amax(-1)
        f_i1p = torch.clamp(ne[:, I1.bid_px.start] + ne[:, I1.ask_px.start], max=1.0)
        f_i1v = ne[:, I1.bid_vol.start : I1.ask_vol.stop].amax(-1)
        f_i1t = ne[:, I1.trade_vol].amax(-1)
        flags = torch.stack([f_px, f_bv, f_av, f_tp, f_tv, f_i1p, f_i1v, f_i1t], -1)
        events = torch.stack([f_px, f_tv, f_i1p], -1)
        last = torch.where(events > 0.5, step[:, None], last)
        since = torch.log1p(torch.clamp(step[:, None] - last, max=_CAP))
        px = slice(I0.bid_px.start, I0.ask_px.stop)
        held = torch.where(f_px[:, None] > 0.5, d[:, px], held)
        feats = torch.cat(
            [
                d[:, px],
                d[:, I0.bid_vol.start : I0.ask_vol.stop],
                d[:, I1.bid_px.start : I1.ask_px.stop],
                flags,
                since,
                held,
            ],
            -1,
        )
        x = torch.cat([row, feats], -1)[:, None, :]
        if self.research_features in MOMENT_FEATURE_MODES:
            if moment_state is None:
                raise ValueError("causal moment streaming state is missing")
            initial = self.moment_initial_state.expand(moment_state.shape[0], -1)
            moment_state = torch.where(first[:, None] > 0.5, initial, moment_state)
            moment_features, moment_state = causal_moment_features_torch_step(
                row,
                moment_state,
                self.research_features,
                self.prior,
                self.research_feature_control,
            )
            x = torch.cat([x, moment_features[:, None, :]], dim=-1)
        elif self.research_features == KALMAN_GROUP_MODE:
            if kalman_group_state is None:
                raise ValueError("Kalman group streaming state is missing")
            initial = self.kalman_group_initial_state.expand(kalman_group_state.shape[0], -1)
            kalman_group_state = torch.where(first[:, None] > 0.5, initial, kalman_group_state)
            kalman_features, kalman_group_state = kalman_group_features_torch_step(
                row, kalman_group_state, self.research_feature_control
            )
            x = torch.cat([x, kalman_features[:, None, :]], dim=-1)
        y, h = self.model(x, h)
        result = (y[:, 0, :2], row, step + 1, last, held, h, torch.zeros_like(first))
        if self.research_features in MOMENT_FEATURE_MODES:
            return (*result, moment_state)
        if self.research_features == KALMAN_GROUP_MODE:
            return (*result, kalman_group_state)
        return result
