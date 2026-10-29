"""TabM-style parameter-efficient GRU ensemble (Gorishniy et al. 2024, arXiv 2410.24210).

k implicit members share the input projection, the GRU and the hidden readout layer. Member
``i`` multiplies the normalized inputs by its own vector ``r_i`` (TabM-mini's single adapter,
initialized with random signs) and has its own output layer (stored as ``head.2``). Training repeats every sequence
``k`` times in the batch dimension (row ``j`` belongs to member ``j % k``), so the unchanged
row-weighted objectives optimize the mean of the member losses rather than the loss of the
mean. Evaluation repeats internally and returns the member mean.

Each member is exactly a plain ``RNNModel``: ``inp(x * r_i) = (W diag r_i) x + b``, so
``member_state_dict`` folds ``r_i`` into the input projection and the native serving engine
runs the members as an ordinary fixed-coefficient blend.
"""

from __future__ import annotations

import torch
from torch import nn

from wnn.nn import RNNModel


class MemberOutput(nn.Module):
    """Per-member output layer; ``weight`` is (n_out, k, width) so ``shape[0]`` stays n_out."""

    def __init__(self, width: int, n_out: int, k: int) -> None:
        super().__init__()
        bound = width**-0.5
        self.weight = nn.Parameter(torch.empty(n_out, k, width).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_out, k).uniform_(-bound, bound))

    def forward(self, values: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        """Apply member ``members[n]``'s layer to row ``n`` of ``values`` (N, T, width)."""
        weight = self.weight[:, members]
        bias = self.bias[:, members].transpose(0, 1)
        return torch.einsum("ntw,onw->nto", values, weight) + bias[:, None, :]


class TabMGRUModel(RNNModel):
    """Plain linear-encoder GRU with k TabM-mini members; see the module docstring."""

    def __init__(
        self, n_in: int, hidden: int, layers: int, n_out: int, k: int, init: str = "signs"
    ) -> None:
        super().__init__(n_in, hidden, layers, 0.0, n_out=n_out)
        if k < 2:
            raise ValueError("a TabM ensemble needs at least two members")
        self.ensemble_k = k
        if init not in ("signs", "ones"):
            raise ValueError(f"unknown TabM adapter init: {init}")
        signs = torch.randint(0, 2, (k, n_in)).float() * 2 - 1
        # "ones": members start as the same function except their output layers.
        self.member_scale = nn.Parameter(signs if init == "signs" else torch.ones(k, n_in))
        # Keep the ``head.2.*`` names so checkpoint readers still find n_out at weight.shape[0].
        self.head._modules["2"] = MemberOutput(self.readout_width, n_out, k)

    def _members(self, rows: int, device: torch.device) -> torch.Tensor:
        if rows % self.ensemble_k:
            raise ValueError("training batch must hold whole member repeats")
        return torch.arange(rows, device=device) % self.ensemble_k

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-member rows in training; member mean on the caller's batch in evaluation."""
        repeat = not self.training
        if repeat:
            x = x.repeat_interleave(self.ensemble_k, dim=0)
        members = self._members(x.shape[0], x.device)
        xn = self._normalized_inputs(x) * self.member_scale[members][:, None, :]
        z = self.encode(xn)
        o, h = self.gru(z, h)
        first, activation, member_output = (self.head._modules[str(i)] for i in range(3))
        output = member_output(activation(first(torch.cat([o, z], dim=-1))), members)
        if repeat:
            output = output.reshape(-1, self.ensemble_k, *output.shape[1:]).mean(dim=1)
        return output, h

    @torch.no_grad()
    def member_state_dict(self, member: int) -> dict[str, torch.Tensor]:
        """Exact plain ``RNNModel`` state dict of one member (input scale folded into ``inp``)."""
        state = {
            name: value.detach().clone()
            for name, value in self.state_dict().items()
            if name != "member_scale"
        }
        state["inp.weight"] = state["inp.weight"] * self.member_scale[member][None, :]
        state["head.2.weight"] = state["head.2.weight"][:, member].contiguous()
        state["head.2.bias"] = state["head.2.bias"][:, member].contiguous()
        return state
