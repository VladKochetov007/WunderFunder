from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

MODEL_PATH = Path(__file__).with_name("model.py")
SPEC = importlib.util.spec_from_file_location("residual_memory_model", MODEL_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
ResidualMatrixMemory = module.ResidualMatrixMemory


@pytest.mark.parametrize("mode", ["delta", "additive_fixed_decay"])
def test_zero_initialized_residual_preserves_arbitrary_base_outputs(mode: str) -> None:
    torch.manual_seed(1701)
    base = torch.randn(2, 7, 3)
    inputs = torch.randn(2, 7, 5)
    memory = ResidualMatrixMemory(5, 3, key_dim=4, value_dim=6, update_mode=mode, event_index=4)
    corrected, state = memory(base, inputs)
    assert torch.equal(corrected, base)
    assert state.shape == (2, 4, 6)
    assert state.dtype == torch.float32


def test_zero_residual_preserves_base_output_dtype_for_half_inputs() -> None:
    base = torch.randn(1, 4, 2, dtype=torch.float16)
    inputs = torch.randn(1, 4, 5, dtype=torch.float16)
    memory = ResidualMatrixMemory(5, 2, key_dim=4, value_dim=3, event_index=4)
    corrected, _ = memory(base, inputs)
    assert corrected.dtype == base.dtype
    assert torch.equal(corrected, base)


@pytest.mark.parametrize("mode", ["delta", "additive_fixed_decay"])
def test_future_input_perturbation_cannot_change_earlier_outputs(mode: str) -> None:
    torch.manual_seed(812)
    base = torch.randn(1, 9, 2)
    inputs = torch.randn(1, 9, 5)
    inputs[..., 4] = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 1, 0]], dtype=torch.float32)
    memory = ResidualMatrixMemory(5, 2, key_dim=4, value_dim=3, update_mode=mode, event_index=4)
    with torch.no_grad():
        memory.residual_out.weight.normal_(0, 0.1)
    changed = inputs.clone()
    changed[:, 4:] += torch.tensor([3.0, -2.0, 1.0, 4.0, -3.0])
    original, _ = memory(base, inputs)
    perturbed, _ = memory(base, changed)
    torch.testing.assert_close(original[:, :5], perturbed[:, :5], rtol=0, atol=0)
    assert not torch.equal(original[:, 5:], perturbed[:, 5:])


@pytest.mark.parametrize("mode", ["delta", "additive_fixed_decay"])
def test_persistent_state_matches_full_sequence_across_uneven_chunks(mode: str) -> None:
    torch.manual_seed(293)
    base = torch.randn(2, 13, 2)
    inputs = torch.randn(2, 13, 7)
    inputs[..., 6] = (torch.rand(2, 13) > 0.65).float()
    memory = ResidualMatrixMemory(7, 2, key_dim=5, value_dim=4, update_mode=mode, event_index=6)
    with torch.no_grad():
        memory.residual_out.weight.normal_(0, 0.03)
    full, full_state = memory(base, inputs)
    carried = None
    pieces = []
    for start, stop in ((0, 3), (3, 8), (8, 13)):
        output, carried = memory(base[:, start:stop], inputs[:, start:stop], carried)
        pieces.append(output)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(carried, full_state, rtol=1e-6, atol=1e-6)


def test_state_detach_is_explicit_and_preserves_matrix_values() -> None:
    memory = ResidualMatrixMemory(4, 2, key_dim=3, value_dim=5, event_index=3)
    base = torch.randn(1, 6, 2, requires_grad=True)
    inputs = torch.randn(1, 6, 4, requires_grad=True)
    _, state = memory(base, inputs)
    assert state.grad_fn is not None
    detached = memory.detach_state(state)
    assert detached is not None and detached.grad_fn is None
    torch.testing.assert_close(detached, state.detach(), rtol=0, atol=0)
    assert memory.detach_state(None) is None


def test_zero_head_opens_gradients_into_memory_after_first_update() -> None:
    torch.manual_seed(905)
    memory = ResidualMatrixMemory(6, 2, key_dim=4, value_dim=3, event_index=5)
    base = torch.randn(2, 5, 2)
    inputs = torch.randn(2, 5, 6)
    inputs[..., 5] = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.float32)
    probe = torch.randn(2, 5, 2)
    first, _ = memory(base, inputs)
    (first * probe).sum().backward()
    assert memory.residual_out.weight.grad is not None
    assert memory.residual_out.weight.grad.abs().sum() > 0
    assert memory.query.weight.grad is None or memory.query.weight.grad.abs().sum() == 0

    memory.zero_grad(set_to_none=True)
    with torch.no_grad():
        memory.residual_out.weight.normal_(0, 0.05)
    second, _ = memory(base, inputs)
    (second * probe).sum().backward()
    for parameter in (
        memory.query.weight,
        memory.write_key.weight,
        memory.write_value.weight,
        memory.event_write_rate.weight,
        memory.quiet_write_rate.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 1e-8


def test_persistent_memory_is_bounded_by_key_value_dimensions() -> None:
    batch, rows, key_dim, value_dim = 3, 257, 16, 16
    memory = ResidualMatrixMemory(8, 2, key_dim=key_dim, value_dim=value_dim, event_index=7)
    base = torch.zeros(batch, rows, 2)
    inputs = torch.zeros(batch, rows, 8)
    _, state = memory(base, inputs)
    assert state.shape == (batch, key_dim, value_dim)
    assert state.numel() == batch * key_dim * value_dim
    assert state.numel() < batch * rows * rows


def test_learned_decay_is_available_only_for_additive_control() -> None:
    learned = ResidualMatrixMemory(
        3, 2, update_mode="additive_fixed_decay", learned_decay=True, event_index=2
    )
    assert learned.decay_logit is not None
    assert torch.sigmoid(learned.decay_logit).item() == pytest.approx(0.999)
    with pytest.raises(ValueError, match="only to the additive control"):
        ResidualMatrixMemory(3, 2, update_mode="delta", learned_decay=True, event_index=2)
    with pytest.raises(ValueError, match="strictly between zero and one"):
        ResidualMatrixMemory(
            3,
            2,
            update_mode="additive_fixed_decay",
            fixed_decay=1.0,
            learned_decay=True,
            event_index=2,
        )


def test_delta_and_fixed_additive_arms_have_matched_parameter_counts() -> None:
    delta = ResidualMatrixMemory(5, 2, key_dim=4, value_dim=3, update_mode="delta", event_index=4)
    additive = ResidualMatrixMemory(
        5,
        2,
        key_dim=4,
        value_dim=3,
        update_mode="additive_fixed_decay",
        event_index=4,
    )
    assert sum(parameter.numel() for parameter in delta.parameters()) == sum(
        parameter.numel() for parameter in additive.parameters()
    )
