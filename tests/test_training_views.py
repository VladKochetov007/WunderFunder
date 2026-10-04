from __future__ import annotations

import copy

import pytest
import torch

from tests.test_trainer_resume_integration import training
from wnn.nn import RNNModel
from wnn.training_views import origin_windows, update_ema_for_rows


@pytest.mark.parametrize("dimensions", [(8, 2, 1), (20000, 1000, 100), (24, 8, 2)])
def test_random_origins_match_exact_rows_updates_and_prefix_work(
    dimensions: tuple[int, int, int],
) -> None:
    fixed = origin_windows(*dimensions, "prefix-fixed", 79, 0, 0)
    changed = False
    state = torch.get_rng_state()
    for seed in range(100):
        windows = origin_windows(*dimensions, "prefix-random", seed, 2, 7)
        assert windows == origin_windows(*dimensions, "prefix-random", seed, 2, 7)
        assert len(windows) == len(fixed)
        assert sum(window.start for window in windows) == sum(window.start for window in fixed)
        assert [row for window in windows for row in range(window.start, window.stop)] == list(
            range(dimensions[0])
        )
        assert all((window.stop - window.start) % dimensions[2] == 0 for window in windows)
        changed |= windows != fixed
    assert changed
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["prefix-fixed", "prefix-random"])
def test_prefix_replay_preserves_causal_forward_but_cuts_prefix_gradient(mode: str) -> None:
    torch.manual_seed(6)
    model = RNNModel(3, hidden=4, layers=1, n_out=2)
    raw = torch.randn(1, 8, 3, requires_grad=True)
    reference, _ = model(raw)
    windows = origin_windows(8, 2, 1, mode, 3, 0, 0)
    predictions = []
    for window in windows:
        state = None
        with torch.no_grad():
            for row in range(window.start):
                _, state = model(raw[:, row : row + 1], state)
        output, _ = model(raw[:, window], state)
        predictions.append(output.detach())
        model.zero_grad(set_to_none=True)
        raw.grad = None
        output.sum().backward()
        assert not raw.grad[:, : window.start].count_nonzero()
        assert not raw.grad[:, window.stop :].count_nonzero()
    torch.testing.assert_close(torch.cat(predictions, 1), reference.detach(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fast", [False, True])
def test_variable_row_ema_restores_checkpoint_configuration(fast: bool) -> None:
    model = torch.nn.Linear(2, 1)
    averaged = (
        training.FastEMA(model, 0.81)
        if fast
        else torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.81)
        )
    )
    update_ema_for_rows(averaged, model, 0.81, 2, 2)
    original = copy.deepcopy(averaged.module.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    update_ema_for_rows(averaged, model, 0.81, 1, 2)
    for name, value in averaged.module.state_dict().items():
        torch.testing.assert_close(value, original[name] + 0.1)
    if fast:
        assert averaged.decay == 0.81
    # The following ordinary update must use the configured .81, not temporary .9.
    averaged.update_parameters(model)
    for name, value in averaged.module.state_dict().items():
        torch.testing.assert_close(value, original[name] + 0.1 + 0.9 * 0.19)
