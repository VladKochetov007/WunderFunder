"""Bounded discarded-device-gate reload and causal-state checks for D02."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from wnn.training_dynamics import build_optimizer, build_scheduler


def assert_nested_equal(left: Any, right: Any) -> None:
    """Check actual serialized optimizer state, including Muon momentum tensors."""
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("D02 reloaded optimizer/model tensor differs")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("D02 reloaded optimizer/model keys differ")
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("D02 reloaded optimizer/model list length differs")
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    elif left != right:
        raise ValueError("D02 reloaded optimizer/model scalar differs")


def check_reload(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    args: Any,
    checkpoint: Path,
    probe: torch.Tensor,
) -> dict[str, object]:
    """Save/reload current discarded fit state and verify reset/carry/prefix behavior."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
        },
        checkpoint,
    )
    restored = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    restored_opt = build_optimizer(restored, args)
    restored_sched = build_scheduler(restored_opt, args, sched.total_steps)
    loaded = torch.load(checkpoint, map_location=probe.device, weights_only=False)
    restored.load_state_dict(loaded["model"])
    restored_opt.load_state_dict(loaded["optimizer"])
    restored_sched.load_state_dict(loaded["scheduler"])
    assert_nested_equal(model.state_dict(), restored.state_dict())
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert_nested_equal(sched.state_dict(), restored_sched.state_dict())
    model.eval()
    restored.eval()
    midpoint = probe.shape[1] // 2
    with torch.no_grad():
        expected, _ = model(probe)
        actual, _ = restored(probe)
        repeated, _ = restored(probe)
        prefix, carry = restored(probe[:, :midpoint])
        suffix, _ = restored(probe[:, midpoint:], carry)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        torch.testing.assert_close(prefix, actual[:, :midpoint], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), actual, rtol=3e-5, atol=3e-6)
    if not torch.isfinite(actual).all():
        raise ValueError("D02 reloaded forward contains nonfinite values")
    return {
        "status": "pass",
        "device_type": probe.device.type,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "probe_shape": list(probe.shape),
        "model_optimizer_scheduler_exact": True,
        "reset_exact": True,
        "prefix_and_chunk_carry_rtol": 3e-5,
        "prefix_and_chunk_carry_atol": 3e-6,
    }
