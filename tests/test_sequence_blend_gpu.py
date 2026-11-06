from __future__ import annotations

import os

import pytest
import torch

from wnn.training.sequence_models import build_sequence_model, detach_state
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_precision import configure_precision_mode, forward_context


def _forward_chunks(model: torch.nn.Module, values: torch.Tensor, mode: str) -> torch.Tensor:
    """Use the production-style detached context boundary on both execution paths."""
    state = None
    outputs = []
    for start, stop in ((0, 9), (9, 23), (23, 32)):
        with forward_context(mode):
            output, _, state = model(values[:, start:stop], state)
        outputs.append(output.float())
        state = detach_state(state)
    return torch.cat(outputs, dim=1)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["bf16", "bf16_bf16x9"])
def test_compiled_composite_objective_matches_eager_finite_outputs_and_gradients(mode: str) -> None:
    """Receiver-only check; run each precision in its own correctly configured process."""
    expected_emulation = "1" if mode == "bf16_bf16x9" else "0"
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION", "0") != expected_emulation:
        pytest.skip("run this precision in a separate matching cuBLAS-emulation process")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA BF16")
    if mode == "bf16_bf16x9" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("BF16x9 acceptance requires the allocated Blackwell device")
    configure_precision_mode(mode)
    torch.manual_seed(61711)
    config = {
        "architecture": "transformer",
        "hidden": 64,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 2,
        "context_rows": 16,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "hint_projection": "learned",
        "precision_mode": mode,
        "attention_backend": "math",
        "compile_blocks": False,
    }
    eager = build_sequence_model(config).cuda().train()
    compiled = (
        build_sequence_model(config | {"attention_backend": "efficient", "compile_blocks": True})
        .cuda()
        .train()
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    values = torch.randn(2, 32, 112, device="cuda")
    targets = torch.randn(2, 32, 2, device="cuda") * 1.2
    reference = torch.randn_like(targets) * 1.5
    mask = torch.ones(2, 32, dtype=torch.bool, device="cuda")
    mask[1] = False
    empty = torch.empty(2, 32, 0, device="cuda")
    denominators = window_denominators(targets, mask, empty)
    outputs = []
    gradients = []
    for model in (eager, compiled):
        prediction = _forward_chunks(model, values, mode)
        nums = objective_numerators(
            prediction, empty, targets, reference, empty, mask, blend_objective_alpha=0.1
        )
        loss = scaled_objective(nums, denominators, 1, 0.0, 0.0)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        outputs.append(prediction.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.03, rtol=0.08)
    relative_gradient_error = torch.linalg.vector_norm(
        gradients[0] - gradients[1]
    ) / torch.linalg.vector_norm(gradients[0]).clamp_min(1e-6)
    assert relative_gradient_error.item() < 0.08
