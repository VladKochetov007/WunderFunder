from __future__ import annotations

import copy

import pytest
import torch

from wnn.optim import MuonAdamW


def _optimizer(parameters: list[torch.nn.Parameter], batched: bool) -> MuonAdamW:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return MuonAdamW(
        [
            {"params": parameters[:2], "muon": True, "blocks": 3},
            {"params": parameters[2:5], "muon": True, "blocks": 1},
            {"params": parameters[5:]},
        ],
        lr=0.0003,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        strict_fp32=True,
        muon_batch_by_shape=batched,
    )


@pytest.mark.parametrize("restart", [False, True])
def test_shape_batching_matches_independent_muon_blocks_and_adam_state(restart: bool) -> None:
    """A packed NS axis preserves independent states, missing-grad semantics and reloads."""
    generator = torch.Generator().manual_seed(9901)
    shapes = [(12, 4), (12, 4), (8, 4), (8, 4), (4, 8), (4,)]
    original = [torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in shapes]
    packed = [torch.nn.Parameter(parameter.detach().clone()) for parameter in original]
    control, candidate = _optimizer(original, False), _optimizer(packed, True)
    assert all("muon_batch_by_shape" not in group for group in control.param_groups)
    for step in range(4):
        for index, (left, right) in enumerate(zip(original, packed, strict=True)):
            gradient = torch.randn(left.shape, generator=generator) * (
                1000 if index == 0 else 0.001
            )
            left.grad = None if step == 1 and index == 1 else gradient
            right.grad = None if left.grad is None else gradient.clone()
        control.step()
        candidate.step()
        for left, right in zip(original, packed, strict=True):
            torch.testing.assert_close(right, left, atol=2e-6, rtol=2e-5)
            assert set(candidate.state[right]) == set(control.state[left])
            for key, expected in control.state[left].items():
                actual = candidate.state[right][key]
                if isinstance(expected, torch.Tensor):
                    assert actual.dtype == torch.float32
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                else:
                    assert actual == expected
        if restart and step == 1:
            saved = copy.deepcopy(candidate.state_dict())
            candidate = _optimizer(packed, True)
            candidate.load_state_dict(saved)


def test_shape_batching_rejects_non_fp32_optimizer_state() -> None:
    """Packing does not silently cast an incompatible state into the declared precision."""
    parameters = [
        torch.nn.Parameter(torch.ones(shape))
        for shape in [(12, 4), (12, 4), (8, 4), (8, 4), (4, 8), (4,)]
    ]
    optimizer = _optimizer(parameters, True)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameters[0]]["momentum_buffer"] = torch.ones_like(
        parameters[0], dtype=torch.float16
    )
    with pytest.raises(TypeError, match="optimizer state"):
        optimizer.step()
