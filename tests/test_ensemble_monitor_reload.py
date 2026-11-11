from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_ensemble_correlation import weighted_microbatch_correlation
from wnn.training.sequence_monitor_evaluation import evaluate_dataset
from wnn.training.sequence_profile_reload import _same, check_profile_reload


@pytest.mark.parametrize("scale", [0.001, 0.1])
def test_declared_std_floor_attenuates_below_floor_and_preserves_above(scale: float) -> None:
    y = torch.tensor([[[-1.0, -0.3], [-0.2, 0.7], [0.4, -0.8], [1.1, 0.4]]])
    p = (
        torch.tensor([[[-1.0, 0.5], [0.2, -0.7], [0.5, -0.2], [0.7, 0.4]]]) * scale
    ).requires_grad_()
    mask = torch.ones((1, 4), dtype=torch.bool)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    raw, _ = weighted_microbatch_correlation(p, y, mask, context)
    rho, loss = weighted_microbatch_correlation(p, y, mask, context, 0.01)
    w = y.abs()
    mean = (w * p).sum((0, 1)) / w.sum((0, 1))
    sd = ((w * (p - mean).square()).sum((0, 1)) / w.sum((0, 1))).sqrt()
    torch.testing.assert_close(rho, raw * (sd / 0.01).clamp_max(1), atol=2e-6, rtol=2e-5)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_roundtrip_strict_container_dtype_and_bytes_and_real_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model(torch.ones((2, 2))).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    opaque = torch.tensor([2, 4, 8], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: opaque.clone())
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda value, device: torch.testing.assert_close(value, opaque)
    )
    result = check_profile_reload(
        model,
        optimizer,
        scheduler,
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        tmp_path,
    )
    assert (
        result["status"] == "passed" and result["by_rank"][0]["temporary_engineering_state_removed"]
    )
    assert not list(tmp_path.glob("*.pt"))
    expected = OrderedDict(weight=torch.ones((2, 2)))
    assert not _same(dict(expected), expected)
    assert not _same(OrderedDict(weight=torch.zeros((2, 2))), expected)


def test_monitor_reports_standalone_and_mixture_from_the_same_forwards() -> None:
    class Dataset:
        """Three tiny CPU fixtures include deliberately excluded NaNs."""

        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            x = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 20 + index / 4
            y = torch.stack((x[:, 0] - 0.6, 0.5 - x[:, 1]), -1)
            mask = torch.tensor([False, True, True, True, True, True, True])
            y[0] = float("nan")
            return {
                "x": x,
                "y": y,
                "mask": mask,
                "teacher": torch.zeros((7, 2)),
                "hints": torch.empty((7, 0)),
            }

    class Model(torch.nn.Module):
        """Count every chunk forward; metrics must not run this twice."""

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self, x: torch.Tensor, state: object
        ) -> tuple[torch.Tensor, torch.Tensor, None]:
            self.calls += 1
            return x[..., :2], x[..., :0], None

    data, model = Dataset(), Model()
    result = evaluate_dataset(
        model,
        data,
        "valid",
        3,
        "fp32",
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        2,
        blend_alpha=0.1,
        report_standalone=True,
    )
    assert model.calls == 6
    assert np.isfinite(list(result.values())).all()
    assert result["valid/wp"] == result["valid/mixture/wp"]
    assert (
        result["valid/z/wp"] != result["valid/wp"] or result["valid/z/wmse"] != result["valid/loss"]
    )
    assert "valid/z/wmse/t0" in result and "valid/mixture/wmse/t1" in result
