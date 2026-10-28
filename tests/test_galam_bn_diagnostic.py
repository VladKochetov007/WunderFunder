from __future__ import annotations

import pytest
import torch
from torch import nn

from wnn.galam_bn.inputs import select_members, validate_precision_environment
from wnn.galam_bn.moments import ChannelMoments, bn_order, recalibrate, state_hashes


class ToyAdapter(nn.Module):
    """Exercise the explicit local-BN seam without constructing a GALAM model."""

    def __init__(self) -> None:
        super().__init__()
        self.galam = nn.Module()
        self.galam.local_input_norm = nn.Module()
        self.galam.local_input_norm.norm = nn.BatchNorm1d(2)
        self.readout = nn.Linear(2, 2)

    def forward(
        self, values: torch.Tensor, state: object = None
    ) -> tuple[torch.Tensor, None, None]:
        """Expose one BN invocation per timestamp, as in the original step graph."""
        del state
        outputs = [
            self.readout(self.galam.local_input_norm.norm(values[:, index]))
            for index in range(values.shape[1])
        ]
        return torch.stack(outputs, dim=1), None, None


def test_pooled_moments_account_for_between_batch_means() -> None:
    """The diagnostic pools activation population, not just average within-batch variance."""
    collector = ChannelMoments()
    norm = nn.BatchNorm1d(2)
    first = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    second = torch.tensor([[100.0, 20.0]])
    collector.observe(norm, (first,))
    collector.observe(norm, (second,))
    with torch.no_grad():
        collector.install(norm)
    joined = torch.cat((first, second)).double()
    torch.testing.assert_close(norm.running_mean.double(), joined.mean(0), atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(norm.running_var.double(), joined.var(0), atol=1e-4, rtol=1e-6)
    assert norm.num_batches_tracked.item() == 2


def test_recalibration_changes_only_bn_buffers_and_resets_sequences() -> None:
    """Frozen parameters and pre-existing eval buffers survive the entire diagnostic."""
    model = ToyAdapter().eval()
    inputs = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    before = state_hashes(model)
    reports = recalibrate(
        model, lambda: iter([inputs[:2], inputs[2:]]), 2, torch.device("cpu"), lambda: None
    )
    after = state_hashes(model)
    assert reports[0]["rows"] == 12
    assert reports[0]["calls"] == 8
    assert before["readout.weight"] == after["readout.weight"]
    assert before["readout.bias"] == after["readout.bias"]
    assert (
        before["galam.local_input_norm.norm.weight"] == after["galam.local_input_norm.norm.weight"]
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(model.galam.local_input_norm.norm.running_mean, inputs.mean((0, 1)))
    torch.testing.assert_close(
        model.galam.local_input_norm.norm.running_var, inputs.reshape(-1, 2).var(0)
    )


def test_selection_is_source_only_seeded_and_count_exact() -> None:
    """No valid member, duplicate or silent shortened calibration cohort is accepted."""
    members = [("train", group) for group in range(20)]
    assert select_members(members, 7, 1001) == select_members(list(reversed(members)), 7, 1001)
    assert len(select_members(members, 7, 1001)) == 7
    for invalid, count in (([("valid", 1)], 1), (members + members, 2), (members, 21)):
        with pytest.raises(ValueError):
            select_members(invalid, count, 1001)


def test_unknown_or_block_bn_execution_fails_closed() -> None:
    """Hooks must not silently miss a block implementation's direct BN arithmetic."""
    model = ToyAdapter()
    model.execution_backend = "block"
    with pytest.raises(ValueError, match="step backend"):
        bn_order(model)
    model.execution_backend = "step"
    model.unexpected_norm = nn.BatchNorm1d(2)
    with pytest.raises(ValueError, match="dependency graph"):
        bn_order(model)


def test_tf32_process_override_is_rejected_before_cuda() -> None:
    """Direct entrypoint policy cannot rely solely on wrapper environment settings."""
    validate_precision_environment({})
    validate_precision_environment({"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"})
    for value in ("1", "true", ""):
        with pytest.raises(ValueError, match="TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
            validate_precision_environment({"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": value})
