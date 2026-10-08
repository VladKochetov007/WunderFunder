from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE / "src"))

from wnn.d02_reload import check_reload  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import MuonAdamW, orthogonalize  # noqa: E402
from wnn.training_dynamics import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    muon_precision_record,
)


def args(optimizer: str = "muon") -> SimpleNamespace:
    return SimpleNamespace(
        optim=optimizer,
        muon_scope="rnn",
        muon_ortho_dtype_policy="fp32",
        head_lr_mult=1.0,
        zoneout=0,
        momentum=0.0,
        optim_switch="",
        fused_adam=False,
        lr=0.001,
        beta2=0.999,
        wd=0.01,
        sched="onecycle",
        warmup_frac=0.05,
        precision="fp32",
    )


@pytest.mark.parametrize("loss", ["wmse", "huber"])
def test_actual_fp32_muon_updates_and_full_state_reload(tmp_path: Path, loss: str) -> None:
    torch.manual_seed(20261009)
    model = RNNModel(4, hidden=8, layers=1)
    settings = args()
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings, 220)
    initial = copy.deepcopy(model.state_dict())
    probe = torch.randn(2, 12, 4)
    labels = torch.randn(2, 12, 2).clamp(-2, 2)
    for _ in range(2):
        predictions, _ = model(probe)
        error = predictions - labels
        row_loss = (
            error.square()
            if loss == "wmse"
            else torch.where(error.abs() <= 1, 0.5 * error.square(), error.abs() - 0.5)
        )
        objective = (labels.abs() * row_loss).sum() / labels.abs().sum()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    assert any(not torch.equal(initial[key], value) for key, value in model.state_dict().items())
    assert isinstance(optimizer, MuonAdamW)
    assert len(optimizer.param_groups) == 2
    for group in optimizer.param_groups:
        assert group["orthogonalization_dtype_policy"] == "fp32"
        for parameter in group["params"]:
            assert parameter.dtype == torch.float32 and torch.isfinite(parameter).all()
            tensor_state = [
                value
                for value in optimizer.state[parameter].values()
                if isinstance(value, torch.Tensor)
            ]
            assert tensor_state and all(value.dtype == torch.float32 for value in tensor_state)
    precision = muon_precision_record(optimizer, settings, torch.device("cpu"))
    assert precision["selected_orthogonalization_dtype"] == "float32"
    assert precision["model_precision"] == "fp32"
    assert precision["scope"] == "rnn" and precision["gatewise"] == "all"
    receipt = check_reload(model, optimizer, scheduler, settings, tmp_path / "state.pt", probe)
    assert receipt["status"] == "pass" and receipt["model_optimizer_scheduler_exact"] is True


def test_adamw_behavior_is_unchanged() -> None:
    torch.manual_seed(99)
    actual = RNNModel(4, hidden=8, layers=1)
    expected = copy.deepcopy(actual)
    optimizer = build_optimizer(actual, args("adamw"))
    reference = torch.optim.AdamW(
        expected.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, fused=False
    )
    for _ in range(2):
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
            gradient = torch.randn_like(left)
            left.grad, right.grad = gradient, gradient.clone()
        optimizer.step()
        reference.step()
    for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_both_muon_groups_match_all_220_adamw_schedule_rows() -> None:
    model = RNNModel(4, hidden=8, layers=1)
    muon = build_optimizer(model, args())
    adamw = build_optimizer(copy.deepcopy(model), args("adamw"))
    ms, ads = build_scheduler(muon, args(), 220), build_scheduler(adamw, args("adamw"), 220)
    for _ in range(220):
        for group in muon.param_groups:
            assert group["lr"] == adamw.param_groups[0]["lr"]
            assert group["betas"][0] == adamw.param_groups[0]["betas"][0]
        muon.step()
        adamw.step()
        ms.step()
        ads.step()


def test_unsupported_orthogonalization_precision_fails_closed() -> None:
    with pytest.raises(ValueError, match="FP32"):
        orthogonalize(torch.ones(2, 4, 4), dtype_policy="bf16")
    with pytest.raises(ValueError, match="FP32"):
        MuonAdamW([torch.nn.Parameter(torch.ones(4))], orthogonalization_dtype_policy="bf16")
