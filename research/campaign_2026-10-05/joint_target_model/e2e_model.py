"""Isolated causal-GRU continuation that preserves the two-output callback."""

from __future__ import annotations

import torch
from model import JointTargetHead
from torch import nn


class JointTargetContinuation(nn.Module):
    """Wrap a standard GRU parent with a paired head and two-value output."""

    def __init__(self, backbone: nn.Module, head: JointTargetHead, arm: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.arm = arm
        self.hidden = int(backbone.inp.out_features)

    def forward_aux(
        self, inputs: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return two predictions, auxiliary logits, and causal recurrent state."""
        normalized = (inputs - self.backbone.mean) / self.backbone.std
        encoded = self.backbone.encode(normalized)
        recurrent, state = self.backbone.gru(encoded, state)
        parent = self.backbone.head(torch.cat((recurrent, encoded), dim=-1))[..., :2]
        paired_states = recurrent.reshape(-1, self.hidden)[:, None, :].expand(-1, 2, -1)
        paired_parent = parent.reshape(-1, 2)[:, None, :].expand(-1, 2, -1)
        prediction, joint_logits, independent_logits = self.head(
            paired_states, paired_parent, self.arm
        )
        shape = (*parent.shape[:-1], 2)
        return (
            prediction.reshape(shape),
            joint_logits.reshape(*parent.shape[:-1], -1),
            independent_logits.reshape(*parent.shape[:-1], 2, -1),
            state,
            parent,
        )

    def forward(
        self, inputs: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Match the standard model callback: causal input/state to two outputs."""
        prediction, _, _, next_state, _ = self.forward_aux(inputs, state)
        return prediction, next_state
