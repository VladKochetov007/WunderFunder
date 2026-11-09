"""Prepared CUDA parity contracts for compiled efficient K16 input filtering."""

from __future__ import annotations

import os

import pytest
import torch

from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import EMULATION_ENV, backward_context, forward_context


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="receiving CUDA host required")
@pytest.mark.parametrize("precision_mode", ["bf16", "bf16_bf16x9"], ids=["bf16", "bf16_bf16x9"])
def test_compiled_input_stem(precision_mode: str) -> None:
    """Compare compiled-efficient and eager-math K16 outputs and finite gradients."""
    emulation = os.environ.get(EMULATION_ENV)
    if precision_mode == "bf16_bf16x9":
        assert emulation == "1", f"{EMULATION_ENV}=1 must be set before this test process starts"
    else:
        assert emulation in (None, "0"), f"{EMULATION_ENV} must be unset or 0 for bf16"
    config: dict[str, object] = {
        "architecture": "transformer",
        "hidden": 32,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 2,
        "context_rows": 16,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "input_stem_kernel": 16,
        "precision_mode": precision_mode,
    }
    torch.manual_seed(503)
    eager = build_sequence_model(config | {"attention_backend": "math"}).cuda().train()
    compiled = (
        build_sequence_model(config | {"attention_backend": "efficient", "compile_blocks": True})
        .cuda()
        .train()
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    with torch.no_grad():
        eager.input_stem.weight.fill_(0.01)
    compiled.load_state_dict(eager.state_dict(), strict=True)
    values = torch.randn(2, 21, 112, device="cuda", dtype=torch.float32)

    with forward_context(precision_mode, "cuda"):
        eager_prediction, _, eager_state = eager(values)
        compiled_prediction, _, compiled_state = compiled(values)
    torch.testing.assert_close(
        compiled_prediction.float(), eager_prediction.float(), rtol=2e-2, atol=2e-2
    )
    assert (
        eager_state.input_history is not None and eager_state.input_history.dtype == torch.float32
    )
    assert compiled_state.input_history is not None

    for model, prediction in ((eager, eager_prediction), (compiled, compiled_prediction)):
        model.zero_grad(set_to_none=True)
        loss = prediction.float().square().mean()
        with backward_context(precision_mode, "cuda"):
            loss.backward()
        stem_gradient = model.input_stem.weight.grad
        assert stem_gradient is not None and torch.isfinite(stem_gradient).all()
        assert torch.count_nonzero(stem_gradient) > 0
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
