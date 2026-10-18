"""Per-sequence self-supervised fast adapter for causal GRU readouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from wnn.nn import RNNModel


@dataclass(frozen=True)
class FastAdapterState:
    """Fast low-rank weights for one independent state per sequence in a batch."""

    down: torch.Tensor
    up: torch.Tensor


class CausalFastAdapter(nn.Module):
    """Meta-trainable low-rank readout updates driven only by observed inputs."""

    def __init__(
        self,
        hidden_size: int,
        feature_size: int = 112,
        rank: int = 16,
        proximal_weight: float = 0.01,
        inner_learning_rate: float = 1e-3,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, feature_size, rank) < 1:
            raise ValueError("hidden_size, feature_size, and rank must be positive")
        if proximal_weight < 0 or inner_learning_rate <= 0 or gradient_clip_norm <= 0:
            raise ValueError(
                "proximal weight must be nonnegative; update controls positive"
            )

        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.rank = rank
        self.proximal_weight = proximal_weight
        self.inner_learning_rate = inner_learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.initial_down = nn.Parameter(torch.empty(hidden_size, rank))
        self.initial_up = nn.Parameter(torch.empty(rank, hidden_size))
        nn.init.kaiming_uniform_(self.initial_down, a=5**0.5)
        nn.init.normal_(self.initial_up, mean=0.0, std=1e-3)
        self.next_input = nn.Linear(hidden_size, feature_size)

    def initial_state(
        self, batch_size: int, *, differentiable: bool
    ) -> FastAdapterState:
        """Create independent per-sequence fast weights from the learned initializer."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        down = self.initial_down.expand(batch_size, -1, -1).clone()
        up = self.initial_up.expand(batch_size, -1, -1).clone()
        if not differentiable:
            down = down.detach().requires_grad_(True)
            up = up.detach().requires_grad_(True)
        return FastAdapterState(down=down, up=up)

    def transform(
        self, recurrent: torch.Tensor, state: FastAdapterState
    ) -> torch.Tensor:
        """Apply each sequence's low-rank correction without mixing batch members."""
        if recurrent.ndim != 3 or state.down.shape[0] != recurrent.shape[0]:
            raise ValueError("recurrent states and fast-weight batch sizes must match")
        if recurrent.shape[-1] != self.hidden_size:
            raise ValueError("recurrent width does not match the adapter")
        low_rank = torch.bmm(recurrent, state.down)
        return recurrent + torch.bmm(low_rank, state.up) / self.rank

    def update(
        self,
        support_recurrent: torch.Tensor,
        standardized_observed_features: torch.Tensor,
        state: FastAdapterState,
        *,
        create_graph: bool,
    ) -> tuple[FastAdapterState, dict[str, torch.Tensor]]:
        """Take one causal input-prediction update for every sequence in the batch.

        ``support_recurrent[:, t]`` has processed input row ``t``. The decoder
        predicts the already-observed, standardized raw input at row ``t + 1``.
        The final support row has no in-block successor and is excluded; no
        query input or task target is read. The caller must apply the immutable,
        fit-only raw-feature scaler and validate finite values at data ingress.
        Keep autograd enabled around this call, including teacher rollouts; the
        update uses ``autograd.grad`` even when task inference itself is no-grad.
        """
        if support_recurrent.ndim != 3 or standardized_observed_features.ndim != 3:
            raise ValueError(
                "support tensors must have batch, time, and feature dimensions"
            )
        if support_recurrent.shape[:2] != standardized_observed_features.shape[:2]:
            raise ValueError("support hidden states and observed inputs must align")
        if support_recurrent.shape[1] < 2:
            raise ValueError("each support block needs at least two input rows")
        if standardized_observed_features.shape[-1] != self.feature_size:
            raise ValueError(
                "observed feature width does not match the auxiliary decoder"
            )
        if state.down.shape[0] != support_recurrent.shape[0]:
            raise ValueError("fast-state batch size does not match the support batch")

        predictor_hidden = self.transform(support_recurrent[:, :-1], state)
        prediction = self.next_input(predictor_hidden)
        # The caller applies the same immutable, fold-fitted scaler used by the
        # task model to raw feature columns 0..111. No online or full-sequence
        # statistics are estimated inside the adapter.
        target = standardized_observed_features[:, 1:]
        per_sequence_loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(
            (1, 2)
        )
        down_anchor = self.initial_down.unsqueeze(0)
        up_anchor = self.initial_up.unsqueeze(0)
        per_sequence_drift = (state.down - down_anchor).square().mean((1, 2))
        per_sequence_drift += (state.up - up_anchor).square().mean((1, 2))
        update_loss = per_sequence_loss + self.proximal_weight * per_sequence_drift

        gradients = torch.autograd.grad(
            update_loss.sum(),
            (state.down, state.up),
            create_graph=create_graph,
        )
        preclip_norm = torch.sqrt(
            gradients[0].square().sum((1, 2)) + gradients[1].square().sum((1, 2))
        )
        clipping_scale = (
            self.gradient_clip_norm / preclip_norm.clamp_min(1e-12)
        ).clamp(max=1.0)
        clipped_down = gradients[0] * clipping_scale[:, None, None]
        clipped_up = gradients[1] * clipping_scale[:, None, None]
        next_state = FastAdapterState(
            down=state.down - self.inner_learning_rate * clipped_down,
            up=state.up - self.inner_learning_rate * clipped_up,
        )
        if not create_graph:
            next_state = FastAdapterState(
                down=next_state.down.detach().requires_grad_(True),
                up=next_state.up.detach().requires_grad_(True),
            )

        return next_state, {
            "input_prediction_loss": per_sequence_loss.detach().mean(),
            "fast_gradient_norm_preclip": preclip_norm.detach().mean(),
            "fast_gradient_clip_fraction": (clipping_scale < 1.0)
            .float()
            .mean()
            .detach(),
            "inner_learning_rate": torch.tensor(
                self.inner_learning_rate, device=support_recurrent.device
            ),
        }


def adapted_forward(
    model: RNNModel,
    adapter: CausalFastAdapter,
    inputs: torch.Tensor,
    recurrent_state: torch.Tensor | None,
    fast_state: FastAdapterState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one causal input chunk and adapt only the task readout representation."""
    normalized = (inputs - model.mean) / model.std
    encoded = model.encode(normalized)
    recurrent, next_state = model.gru(encoded, recurrent_state)
    adapted = adapter.transform(recurrent, fast_state)
    output = model.head(torch.cat([adapted, encoded], dim=-1))
    return output, next_state, recurrent
