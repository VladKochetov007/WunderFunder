from __future__ import annotations

import json
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from wnn.data import N_FEATURES
from wnn.metric import score
from wnn.training import evaluation, outputs
from wnn.training.data_loaders import select_train_eval_row_groups
from wnn.training.environment import TrainingEnvironment


def _manual_wmse(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(targets.astype(np.float64), -2.0, 2.0)
    weights = np.abs(clipped) * mask[..., None]
    numerator = np.sum(weights * (predictions.astype(np.float64) - clipped) ** 2, axis=(0, 1))
    denominator = weights.sum(axis=(0, 1))
    return np.divide(numerator, denominator, out=np.zeros(2), where=denominator > 0)


def _batch_loader(
    predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray, batch_size: int
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(predictions.copy()),
            torch.from_numpy(targets.copy()),
            torch.from_numpy(mask.copy()),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _assert_numpy_rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_evaluate_adds_wmse_from_the_existing_prediction_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, steps = 2, 120
    target_axis = np.linspace(-3.0, 3.0, steps, dtype=np.float32)
    targets = np.stack(
        [np.stack([target_axis, target_axis[::-1]], axis=-1) for _ in range(batch)]
    )
    predictions = np.full_like(targets, 2.75)
    predictions[1, :, 1] = -3.25
    mask = np.ones((batch, steps), dtype=bool)
    mask[:, :99] = False
    mask[0, 105] = False
    calls: list[int] = []

    def existing_predictions(
        _model: torch.nn.Module, x: torch.Tensor, _chunk: int, _device: torch.device
    ) -> np.ndarray:
        calls.append(len(x))
        return x[..., :2].cpu().numpy()

    monkeypatch.setattr(evaluation, "market_class", lambda _targets: 0)
    metrics = evaluation.evaluate(
        torch.nn.Identity(),
        _batch_loader(predictions, targets, mask, batch_size=1),
        chunk=17,
        device=torch.device("cpu"),
        environment=TrainingEnvironment(warmup=99, prediction=existing_predictions),
    )
    expected_mask = mask.copy()
    expected_mask[:, :99] = False
    expected_wmse = _manual_wmse(targets, predictions, expected_mask)
    assert calls == [1, 1]
    assert metrics["selected_rows"] == int(expected_mask.sum())
    assert metrics["wmse_t0"] == pytest.approx(expected_wmse[0])
    assert metrics["wmse_t1"] == pytest.approx(expected_wmse[1])
    assert metrics["wmse"] == pytest.approx(
        np.average(expected_wmse, weights=np.abs(np.clip(targets, -2, 2)[expected_mask]).sum(0))
    )


def test_prediction_metric_uses_only_the_main_two_heads() -> None:
    def family_forward(
        _model: torch.nn.Module,
        inputs: torch.Tensor,
        rows: slice,
        state: object,
    ) -> tuple[torch.Tensor, object]:
        start = int(rows.start or 0)
        count = min(int(rows.stop), inputs.shape[1]) - start
        heads = torch.full((inputs.shape[0], count, 10), 100.0, device=inputs.device)
        heads[..., 0] = 0.25
        heads[..., 1] = -0.5
        return heads, state

    predictions = evaluation.predict(
        torch.nn.Identity(),
        torch.zeros((1, 120, N_FEATURES)),
        chunk=50,
        device=torch.device("cpu"),
        environment=TrainingEnvironment(forward_chunk=family_forward),
    )
    assert predictions.shape == (1, 120, 2)
    np.testing.assert_array_equal(predictions[..., 0], 0.25)
    np.testing.assert_array_equal(predictions[..., 1], -0.5)


def test_train_prefix_uses_all_after_warmup_and_restores_rng_modes_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, steps = 2, 120
    axis = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    targets = np.stack([np.stack([axis, -axis], axis=-1) for _ in range(batch)])
    predictions = np.zeros_like(targets)
    no_official_train_mask = np.zeros((batch, steps), dtype=bool)
    loader = _batch_loader(predictions, targets, no_official_train_mask, batch_size=2)
    monkeypatch.setattr(evaluation, "market_class", lambda _targets: 0)

    class TinyBatchNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_norm = torch.nn.BatchNorm1d(2)

    model = TinyBatchNorm()
    model.train()
    model.batch_norm.eval()
    modes_before = [module.training for module in model.modules()]
    buffers_before = [buffer.detach().clone() for buffer in model.buffers()]

    def stateful_prediction(
        current: torch.nn.Module, x: torch.Tensor, _chunk: int, _device: torch.device
    ) -> np.ndarray:
        torch.rand(3)
        np.random.random()  # noqa: NPY002 - exercise legacy process-wide RNG preservation
        random.random()
        current.batch_norm.running_mean.add_(1.0)
        current.batch_norm.num_batches_tracked.add_(1)
        current.train()
        return x[..., :2].cpu().numpy()

    torch.manual_seed(441)
    np.random.seed(442)  # noqa: NPY002 - seed the process-wide legacy stream
    random.seed(443)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()  # noqa: NPY002 - snapshot the legacy global stream
    python_before = random.getstate()
    metrics = evaluation.evaluate_train_prefix(
        model,
        loader,
        chunk=23,
        device=torch.device("cpu"),
        environment=TrainingEnvironment(warmup=99, prediction=stateful_prediction),
    )
    all_after_warmup = np.tile(np.arange(steps) >= 99, (batch, 1))
    expected_wmse = _manual_wmse(targets, predictions, all_after_warmup)
    assert metrics["selected_rows"] == batch * (steps - 99)
    assert metrics["wmse_t0"] == pytest.approx(expected_wmse[0])
    assert metrics["wmse_t1"] == pytest.approx(expected_wmse[1])
    assert metrics["weighted_pearson"] == pytest.approx(
        score(targets, predictions, all_after_warmup)["weighted_pearson"]
    )
    assert [module.training for module in model.modules()] == modes_before
    for actual, before in zip(model.buffers(), buffers_before, strict=True):
        torch.testing.assert_close(actual, before)
    torch.testing.assert_close(torch.get_rng_state(), torch_before)
    _assert_numpy_rng_equal(
        np.random.get_state(), numpy_before  # noqa: NPY002 - verify the legacy global stream
    )
    assert random.getstate() == python_before


def test_train_eval_source_prefix_is_sorted_and_bounded() -> None:
    assert select_train_eval_row_groups(np.array([10, 3, 8, 5]), 3) == [3, 5, 8]
    with pytest.raises(ValueError, match=r"count in \[1, 32\]"):
        select_train_eval_row_groups(np.array([1, 2]), 33)
    with pytest.raises(ValueError, match="unique integer groups"):
        select_train_eval_row_groups(np.array([1, 1]), 1)
    with pytest.raises(ValueError, match="exceeds"):
        select_train_eval_row_groups(np.array([1, 2]), 3)


def test_valid_record_appends_train_eval_fields_without_repeating_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_calls: list[object] = []
    train_calls: list[object] = []
    emitted: list[tuple[str, str]] = []
    monkeypatch.setattr(outputs, "checkpoint_model", lambda model, *_args: model)
    monkeypatch.setattr(outputs, "fit_epoch_summary", lambda _run, _updates: {})
    monkeypatch.setattr(outputs, "emit", lambda kind, payload, **_kwargs: emitted.append((kind, payload)))

    def evaluate_valid(*args: object, **_kwargs: object) -> dict[str, float | int]:
        valid_calls.append(args[1])
        return {
            "weighted_pearson": 0.6,
            "t0": 0.5,
            "t1": 0.7,
            "wmse": 0.4,
            "wmse_t0": 0.3,
            "wmse_t1": 0.5,
        }

    def evaluate_train(*args: object, **_kwargs: object) -> dict[str, float | int]:
        train_calls.append(args[1])
        return {
            "weighted_pearson": 0.4,
            "t0": 0.3,
            "t1": 0.5,
            "wmse": 0.8,
            "wmse_t0": 0.7,
            "wmse_t1": 0.9,
            "selected_rows": 99,
            "blocks": 2,
        }

    monkeypatch.setattr(outputs, "evaluate", evaluate_valid)
    monkeypatch.setattr(outputs, "evaluate_train_prefix", evaluate_train)

    class Counter:
        def finish(self, _cumulative: object) -> dict[str, int]:
            return {"updates": 1, "epoch": 1}

    model = torch.nn.Linear(1, 1)
    run = SimpleNamespace(
        environment=SimpleNamespace(readout=SimpleNamespace(tables=None, tilted=False)),
        options=SimpleNamespace(
            args=SimpleNamespace(no_eval_during_fit=False, eval_ema=False, eval_source="dev"),
            device=torch.device("cpu"),
        ),
        progress=SimpleNamespace(
            cumulative_updates={},
            epoch=0,
            epoch_counter=Counter(),
            epoch_grad_stats=None,
            epoch_order_identity={"seed": 3},
            eval_model=None,
            evaluation_score=None,
            history=[],
            step_count=0,
            t0=time.time() - 1,
        ),
        optimization=SimpleNamespace(ema_model=None, swa_model=None, swa_start=0),
        network=SimpleNamespace(model=model, moments=None),
        data=SimpleNamespace(
            n_valid=1873,
            provenance={"normalization_identity": "norm", "update_diagnostics": []},
            valid_loader=object(),
            train_eval_loader=object(),
            train_eval_row_groups=[2, 7],
            train_eval_members_sha256="a" * 64,
        ),
    )
    outputs.evaluate_epoch(run)
    assert len(valid_calls) == len(train_calls) == 1
    kind, payload = emitted[0]
    record = json.loads(payload)
    assert kind == "VALID"
    assert record["valid_wmse"] == 0.4
    assert record["valid_wmse_t0"] == 0.3
    assert record["train_eval_wp"] == 0.4
    assert record["train_eval_wmse"] == 0.8
    assert record["train_eval_seconds"] >= 0
    assert record["train_eval_source"] == "train"
    assert record["train_eval_mask"] == "all_after_warmup"
    assert record["train_eval_row_groups"] == [2, 7]
    assert record["train_eval_members_sha256"] == "a" * 64
