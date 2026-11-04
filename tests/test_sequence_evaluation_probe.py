from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_evaluation_probe import (
    _evaluate_preserving_state,
)
from wnn.training.sequence_evaluation_probe_source import (
    _select_valid_members,
    _validate_audit_record,
)


class _OneSequence:
    def __init__(self) -> None:
        self.record = {
            "x": torch.zeros(20_000, 112),
            "y": torch.zeros(20_000, 2),
            "mask": torch.arange(20_000) >= 99,
            "teacher": torch.zeros(20_000, 2),
            "hints": torch.zeros(20_000, 0),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        return self.record


class _RandomBatchNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)

    def forward(
        self, values: torch.Tensor, state: object = None
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        random.random()
        np.random.random()  # noqa: NPY002 - exercise RNG restoration in the probe
        torch.rand(1)
        hidden = self.norm(values[..., :4].transpose(1, 2)).transpose(1, 2)
        return self.head(hidden), values.new_empty((*values.shape[:2], 0)), state


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"), False)


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()  # noqa: NPY002


def _assert_rng_snapshot(expected: tuple[object, tuple[object, ...], torch.Tensor]) -> None:
    py_state, np_state, torch_state = expected
    assert random.getstate() == py_state
    current_np = np.random.get_state()  # noqa: NPY002 - compare process-global RNG restoration
    assert np_state[0] == current_np[0]
    assert np.array_equal(np_state[1], current_np[1])
    assert np_state[2:] == current_np[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


def test_valid_audit_reuses_receipt_and_checks_local_footer_identity() -> None:
    audit = {
        "schema": "tabdata.memory-and-storage-audit.v1",
        "data": {
            "valid_sha256_from_complete_transfer_receipt": "a" * 64,
            "valid_bytes": 10,
            "valid_fresh_footer_rows": 40_000,
            "valid_fresh_footer_row_groups": 2,
            "valid_fresh_full_hash": False,
            "valid_fresh_schema_includes": [
                "seq_ix",
                "step_in_seq",
                "is_scored",
                "t0",
                "t1",
                "112 input features",
            ],
        },
    }
    result = _validate_audit_record(audit, 10, 40_000, 2)
    assert result["sha256_receipt_pin"] == "a" * 64
    assert "not repeated" in result["sha256_verification"]
    with pytest.raises(ValueError, match="differs from the audit"):
        _validate_audit_record(audit, 11, 40_000, 2)


def test_valid_subset_is_bounded_and_uses_physical_membership() -> None:
    members = _select_valid_members(1_873, 384)
    assert len(members) == 384
    assert members[0] == ("valid", 0)
    assert members[-1] == ("valid", 383)
    with pytest.raises(ValueError, match="max sequences"):
        _select_valid_members(1_873, 385)
    with pytest.raises(ValueError, match="exceeds official"):
        _select_valid_members(20, 21)


def test_canonical_evaluation_restores_modes_buffers_and_all_cpu_rng() -> None:
    model = _RandomBatchNormModel().train()
    before_buffers = [value.clone() for value in (model.norm.running_mean, model.norm.running_var)]
    before_count = model.norm.num_batches_tracked.clone()
    random.seed(29)
    np.random.seed(31)  # noqa: NPY002 - deterministic test state
    torch.manual_seed(37)
    rng = _rng_snapshot()

    metrics = _evaluate_preserving_state(model, _OneSequence(), 20_000, 128, _context())

    assert metrics["valid/sequence_count"] == 1
    assert metrics["valid/selected_rows"] == 19_901
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert model.training and model.norm.training
    assert torch.equal(model.norm.running_mean, before_buffers[0])
    assert torch.equal(model.norm.running_var, before_buffers[1])
    assert torch.equal(model.norm.num_batches_tracked, before_count)
    _assert_rng_snapshot(rng)


def test_probe_restores_mode_and_rng_when_canonical_evaluation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wnn.training.sequence_evaluation_probe as probe

    model = _RandomBatchNormModel().train()
    random.seed(41)
    np.random.seed(43)  # noqa: NPY002 - deterministic test state
    torch.manual_seed(47)
    rng = _rng_snapshot()

    def fail_after_rng_use(*args: object, **kwargs: object) -> dict[str, int | float]:
        random.random()
        np.random.random()  # noqa: NPY002 - exercise exceptional restoration
        torch.rand(1)
        raise RuntimeError("bounded evaluation failure")

    monkeypatch.setattr(probe, "evaluate_dataset", fail_after_rng_use)
    with pytest.raises(RuntimeError, match="bounded evaluation failure"):
        _evaluate_preserving_state(model, _OneSequence(), 20_000, 128, _context())
    assert model.training and model.norm.training
    _assert_rng_snapshot(rng)
