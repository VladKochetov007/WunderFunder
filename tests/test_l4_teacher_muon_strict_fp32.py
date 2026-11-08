from __future__ import annotations

from argparse import Namespace

import torch
from torch import nn

from wnn import optim as wnn_optim
from wnn.training_dynamics import build_optimizer


class TinyGRU(nn.Module):
    """Expose one recurrent gate family and one AdamW head for optimizer checks."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(3, 4, batch_first=True)
        self.head = nn.Linear(4, 2)


def test_legacy_muon_keeps_all_groups_and_newton_schulz_fp32(monkeypatch) -> None:
    """Strict precision must cover recurrent Muon and nonrecurrent AdamW state."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    model = TinyGRU()
    args = Namespace(
        optim="muon",
        muon_scope="rnn",
        lr=1e-3,
        wd=1e-2,
        beta2=0.999,
        momentum=0.0,
        head_lr_mult=1.0,
        zoneout=0.0,
        optim_switch="",
        fused_adam=False,
        precision="fp32",
    )
    seen: list[tuple[torch.dtype, bool]] = []
    original = wnn_optim.orthogonalize

    def record_orthogonalize(
        gradient: torch.Tensor, steps: int = 5, eps: float = 1e-7, *, strict_fp32: bool = False
    ) -> torch.Tensor:
        seen.append((gradient.dtype, strict_fp32))
        return original(gradient, steps, eps, strict_fp32=strict_fp32)

    monkeypatch.setattr(wnn_optim, "orthogonalize", record_orthogonalize)
    optimizer = build_optimizer(model, args)
    assert all(group["strict_fp32"] is True for group in optimizer.param_groups)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)

    optimizer.step()

    assert seen and all(dtype is torch.float32 and strict for dtype, strict in seen)
    floating_state = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    assert floating_state and all(value.dtype is torch.float32 for value in floating_state)
