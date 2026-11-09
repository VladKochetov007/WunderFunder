"""Receiving-host compiled Q/K normalization checks; prepared but not executed locally."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import backward_context, forward_context


def _config(precision_mode: str, attention_backend: str, compile_blocks: bool) -> dict[str, object]:
    return {
        "architecture": "transformer",
        "hidden": 64,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 2,
        "context_rows": 16,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "precision_mode": precision_mode,
        "attention_backend": attention_backend,
        "compile_blocks": compile_blocks,
        "qk_norm": "rms",
    }


def _chunked_prediction(model: nn.Module, values: Tensor, precision_mode: str) -> Tensor:
    predictions: list[Tensor] = []
    state = None
    for start, end in ((0, 5), (5, 12), (12, 19)):
        with forward_context(precision_mode, "cuda"):
            prediction, _, state = model(values[:, start:end], state)
        predictions.append(prediction)
    return torch.cat(predictions, dim=1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="receiving CUDA host required")
@pytest.mark.parametrize("precision_mode", ["bf16", "bf16_bf16x9"])
def test_compiled_qk_norm(precision_mode: str) -> None:
    """Compare compiled efficient attention with eager math in a fresh configured process.

    Run each parameter node in its own process with CUBLAS emulation configured
    externally before CUDA initialization; this test never mutates that setting.
    """
    torch.manual_seed(503)
    compiled = (
        build_sequence_model(_config(precision_mode, "efficient", compile_blocks=True))
        .cuda()
        .train()
    )
    eager = (
        build_sequence_model(_config(precision_mode, "math", compile_blocks=False)).cuda().train()
    )
    eager.load_state_dict(compiled.state_dict(), strict=True)

    values = torch.randn(2, 19, 112, device="cuda", dtype=torch.float32)
    compiled_input = values.clone().requires_grad_(True)
    eager_input = values.clone().requires_grad_(True)
    with forward_context(precision_mode, "cuda"):
        compiled_output, _, _ = compiled(compiled_input)
        eager_output, _, _ = eager(eager_input)
    torch.testing.assert_close(compiled_output.float(), eager_output.float(), rtol=3e-2, atol=3e-2)

    altered = values.clone()
    altered[:, 10:] += 100.0
    with torch.no_grad():
        compiled_chunked = _chunked_prediction(compiled, values, precision_mode)
        eager_chunked = _chunked_prediction(eager, values, precision_mode)
        with forward_context(precision_mode, "cuda"):
            compiled_altered, _, _ = compiled(altered)
            eager_altered, _, _ = eager(altered)
    torch.testing.assert_close(
        compiled_chunked.float(), compiled_output.detach().float(), rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        eager_chunked.float(), eager_output.detach().float(), rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        compiled_altered[:, :10].float(),
        compiled_output.detach()[:, :10].float(),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        eager_altered[:, :10].float(),
        eager_output.detach()[:, :10].float(),
        rtol=3e-2,
        atol=3e-2,
    )

    compiled_loss = compiled_output.float().square().mean()
    eager_loss = eager_output.float().square().mean()
    with backward_context(precision_mode, "cuda"):
        compiled_loss.backward()
        eager_loss.backward()
    assert compiled_input.grad is not None and torch.isfinite(compiled_input.grad).all()
    assert eager_input.grad is not None and torch.isfinite(eager_input.grad).all()
    for model in (compiled, eager):
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
