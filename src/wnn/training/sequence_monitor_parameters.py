from __future__ import annotations

import torch
from torch import Tensor, nn

MAX_SAMPLED_PARAMETERS = 4_096


class SampledParameterProbe:
    """Keep a fixed bounded prefix for sampled weight/update norms without RNG use."""

    def __init__(self, model: nn.Module) -> None:
        """Snapshot at most 4,096 trainable scalar weights before a sampled update."""
        remaining = MAX_SAMPLED_PARAMETERS
        self._members: list[tuple[Tensor, Tensor]] = []
        for parameter in model.parameters():
            if parameter.requires_grad and remaining:
                # All canonical parameters are contiguous; reshape also handles adapters.
                view = parameter.detach().reshape(-1)[:remaining]
                self._members.append((parameter, view.float().clone()))
                remaining -= view.numel()

    def metrics(self) -> dict[str, Tensor]:
        """Read current weights after the update and return device-resident scalars."""
        if not self._members:
            return {}
        current = torch.cat(
            [
                parameter.detach().reshape(-1)[: before.numel()].float()
                for parameter, before in self._members
            ]
        )
        before = torch.cat([snapshot for _, snapshot in self._members])
        weight_norm = current.norm()
        update_norm = (current - before).norm()
        return {
            "optimizer/sampled_weight_l2_norm": weight_norm,
            "optimizer/sampled_update_l2_norm": update_norm,
            "optimizer/sampled_update_to_weight_ratio": update_norm / weight_norm.clamp_min(1e-12),
            "optimizer/sampled_parameter_count": current.new_tensor(current.numel()),
        }
