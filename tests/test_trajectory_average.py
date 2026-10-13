from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from average_trajectory_checkpoints import mean_state


def test_linear_prediction_matches_equal_prediction_mean() -> None:
    states = [
        {
            "weight": torch.tensor([[value, -value]]),
            "mean": torch.tensor([0.0, 1.0]),
            "std": torch.ones(2),
        }
        for value in (1.0, 4.0, -2.0)
    ]
    x = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    state = mean_state(states)
    expected = sum(x @ item["weight"].T for item in states) / 3
    torch.testing.assert_close(x @ state["weight"].T, expected)
    assert torch.equal(state["mean"], states[0]["mean"])
    assert state["mean"].data_ptr() != states[0]["mean"].data_ptr()
    assert states[0]["weight"][0, 0].item() == 1.0


def test_different_normalization_cannot_be_silently_averaged() -> None:
    states = [{"mean": torch.tensor([value]), "weight": torch.ones(1)} for value in (0.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="fixed buffer differs"):
        mean_state(states)


def test_nonfinite_parameter_is_rejected() -> None:
    states = [{"weight": torch.tensor([value])} for value in (0.0, float("nan"), 1.0)]
    with pytest.raises(ValueError, match="nonfinite"):
        mean_state(states)
