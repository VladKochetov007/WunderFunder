"""Receiving-host BF16 streaming contracts; never executed by local preparation."""

from __future__ import annotations

import pytest
import torch

from wnn.training.sequence_models import build_sequence_model, detach_state
from wnn.training.sequence_precision import EMULATION_ENV, forward_context


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="receiving CUDA host required")
@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_bf16_streaming_keeps_fp32_state_and_updates(
    architecture: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carry two chunks through BF16 forward/backward with FP32 master updates."""
    # Pure BF16 exercises dtype boundaries without changing process-wide cuBLAS
    # policy after CUDA initialization. bf16_bf16x9 real-shape profiles run in
    # separate fresh processes with the environment fixed before import.
    monkeypatch.delenv(EMULATION_ENV, raising=False)
    model = build_sequence_model(
        {
            "architecture": architecture,
            "hidden": 64,
            "layers": 2,
            "heads": 2,
            "context_rows": 16,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "precision_mode": "bf16",
            "attention_backend": "efficient" if architecture == "transformer" else "math",
            "compile_blocks": False,
        }
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = None
    for _ in range(2):
        values = torch.randn(2, 17, 112, device="cuda", dtype=torch.float32)
        with forward_context("bf16", "cuda"):
            prediction, _, state = model(values, state)
        state = detach_state(state)
        tensors = (state,) if isinstance(state, torch.Tensor) else (*state.keys, *state.values)
        assert all(value.dtype == torch.float32 for value in tensors)
        loss = prediction.float().square().mean()
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is None
            or (parameter.grad.dtype == torch.float32 and torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(
        value.dtype == torch.float32
        for item in optimizer.state.values()
        for value in item.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
