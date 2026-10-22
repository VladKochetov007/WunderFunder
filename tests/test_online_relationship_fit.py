"""CPU contract checks for the fixed online relationship readout fit."""

from __future__ import annotations

import numpy as np
import torch

import wnn.training.online_relationship_fit as fit_component
from wnn.online_relationship import fit_online_relationship_normalizer
from wnn.training.online_relationship import OnlineRelationshipResidualHead
from wnn.training.online_relationship_fit import (
    OnlineRelationshipSequence,
    fit_online_relationship_head,
)


def _sequence(rows: int, offset: float = 0.0) -> np.ndarray:
    """Return finite changing raw features for a tiny complete-sequence fixture."""
    time = np.arange(rows, dtype=np.float32)
    raw = np.zeros((rows, 112), dtype=np.float32)
    for channel in range(52):
        raw[:, channel] = offset + (channel + 1) * 0.001 * time
        raw[:, 52 + channel] = offset + (channel + 1) * 0.002 * time
    return raw


class _UnreadEvaluationArray:
    """Fail if the evaluator inspects held-out labels or row weights."""

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("evaluation labels and row weights must remain unread")


def test_zero_output_head_returns_the_parent_exactly() -> None:
    """The initial residual is exactly zero for every parent prediction."""
    head = OnlineRelationshipResidualHead()
    features = torch.randn((11, 419), dtype=torch.float32)
    parent = torch.randn((11, 2), dtype=torch.float32)
    assert torch.equal(head(features, parent), parent)


def test_fit_uses_three_fixed_epochs_and_keeps_eval_labels_unread() -> None:
    """The CPU fit returns cp2 and ordered predictions without reading eval labels."""
    raw_one = _sequence(21)
    raw_two = _sequence(19, 0.3)
    normalizer = fit_online_relationship_normalizer([raw_one, raw_two], 1)
    fit_rows = [
        OnlineRelationshipSequence(
            raw,
            np.full((len(raw), 2), 0.4, dtype=np.float32),
            np.ones(len(raw), dtype=np.float32),
            np.full((len(raw), 2), 0.1, dtype=np.float32),
        )
        for raw in (raw_one, raw_two)
    ]
    eval_raw = _sequence(23, 0.7)
    eval_rows = [
        OnlineRelationshipSequence(
            eval_raw,
            _UnreadEvaluationArray(),
            _UnreadEvaluationArray(),
            np.full((len(eval_raw), 2), 0.2, dtype=np.float32),
        )
    ]
    fit_passes: list[int] = []
    eval_passes: list[int] = []

    def fit_factory() -> list[OnlineRelationshipSequence]:
        fit_passes.append(1)
        return fit_rows

    def eval_factory() -> list[OnlineRelationshipSequence]:
        eval_passes.append(1)
        return eval_rows

    progress: list[dict[str, object]] = []
    result = fit_online_relationship_head(
        fit_factory,
        eval_factory,
        arm="C",
        lag=1,
        seed=1001,
        parent_mean=np.zeros(211, dtype=np.float32),
        parent_std=np.ones(211, dtype=np.float32),
        relationship_normalizer=normalizer,
        device="cpu",
        progress=progress.append,
    )

    assert len(fit_passes) == 3
    assert len(eval_passes) == 1
    assert len(result.epoch_records) == 3
    assert result.metadata["fixed_checkpoint_epoch"] == 2
    assert result.metadata["checkpoint_selected"] is True
    assert result.metadata["relationship_normalizer"] == normalizer.to_dict()
    assert len(result.evaluation_predictions) == 1
    assert result.evaluation_predictions[0].shape == (len(eval_raw), 2)
    assert all(record["updates"] == 1 for record in result.epoch_records)
    assert all(record["finite_updates"] == 1 for record in result.epoch_records)
    assert all(record["fit_rows"] == len(raw_one) + len(raw_two) for record in result.epoch_records)
    assert progress == list(result.epoch_records)


def test_row_batches_cross_sequence_boundaries_and_retain_the_tail(
    monkeypatch: object,
) -> None:
    """The gatherer preserves order across sequences and emits a final partial batch."""
    monkeypatch.setattr(fit_component, "BATCH_ROWS", 8)
    first_raw = _sequence(5)
    second_raw = _sequence(7, 0.2)
    normalizer = fit_online_relationship_normalizer([_sequence(20)], 1)
    fit_rows = [
        OnlineRelationshipSequence(
            raw,
            np.full((len(raw), 2), value, dtype=np.float32),
            np.ones(len(raw), dtype=np.float32),
            np.full((len(raw), 2), value * 10, dtype=np.float32),
        )
        for raw, value in ((first_raw, 1.0), (second_raw, 2.0))
    ]
    counts = {"sequences": 0, "rows": 0, "batches": 0}
    batches = [
        tuple(value.clone() for value in batch)
        for batch in fit_component._iter_row_batches(
            lambda: fit_rows,
            "A",
            1,
            normalizer,
            torch.zeros(211),
            torch.ones(211),
            torch.device("cpu"),
            counts,
        )
    ]

    assert fit_component.BATCH_ROWS == 8
    assert [len(batch[1]) for batch in batches] == [8, 4]
    np.testing.assert_array_equal(batches[0][1][:5].numpy(), 1.0)
    np.testing.assert_array_equal(batches[0][1][5:].numpy(), 2.0)
    np.testing.assert_array_equal(batches[1][1].numpy(), 2.0)
    np.testing.assert_array_equal(batches[0][3][:5].numpy(), 10.0)
    np.testing.assert_array_equal(batches[0][3][5:].numpy(), 20.0)
    assert counts == {"sequences": 2, "rows": 12, "batches": 2}
