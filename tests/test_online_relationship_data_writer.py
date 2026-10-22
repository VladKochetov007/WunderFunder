import numpy as np
import pytest
import torch

from wnn.data import SEQUENCE_LENGTH, Sequence
from wnn.training.environment import TrainingEnvironment
from wnn.training.evaluation import predict
from wnn.training.online_relationship_parent_batch import (
    PARENT_BATCH_SEQUENCES,
    fit_sequence_targets_and_mask,
    read_parent_input_batch,
)


def test_fit_sequence_defaults_missing_score_mask_but_requires_targets() -> None:
    features = np.zeros((5, 112), dtype=np.float32)
    targets = np.ones((5, 2), dtype=np.float32)
    fit_sequence = Sequence(0, features, targets, None)

    actual_targets, score_mask = fit_sequence_targets_and_mask(fit_sequence)

    np.testing.assert_array_equal(actual_targets, targets)
    np.testing.assert_array_equal(score_mask, np.zeros(5, dtype=np.bool_))
    with pytest.raises(ValueError, match="omitted canonical targets"):
        fit_sequence_targets_and_mask(Sequence(0, features, None, None))


def test_fit_sequence_preserves_present_score_mask() -> None:
    features = np.zeros((3, 112), dtype=np.float32)
    targets = np.zeros((3, 2), dtype=np.float32)
    scored = np.asarray([False, True, False])

    _, score_mask = fit_sequence_targets_and_mask(Sequence(0, features, targets, scored))

    np.testing.assert_array_equal(score_mask, scored)


class _MemorySequenceFile:
    def __init__(self, records: dict[int, Sequence]) -> None:
        self.records = records
        self.calls: list[tuple[int, bool, bool]] = []

    def read(self, index: int, *, targets: bool, mask: bool) -> Sequence:
        self.calls.append((index, targets, mask))
        source = self.records[index]
        return Sequence(
            source.seq_ix,
            source.features,
            source.targets if targets else None,
            source.scored if mask else None,
        )


def _forward_first_two(model, inputs, sequence_slice, state):
    del model
    return inputs[:, sequence_slice, :2], state


def _make_sequence(seq_ix: int, *, score_mask: np.ndarray | None) -> Sequence:
    features = np.zeros((SEQUENCE_LENGTH, 112), dtype=np.float32)
    features[:, 0] = seq_ix
    features[:, 1] = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
    targets = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    return Sequence(seq_ix, features, targets, score_mask)


def test_eight_sequence_parent_batch_preserves_order_parity_and_read_scope() -> None:
    assert PARENT_BATCH_SEQUENCES == 8
    records = {
        split: {
            group: _make_sequence(
                100 + group,
                score_mask=(np.ones(SEQUENCE_LENGTH, dtype=np.bool_) if group == 3 else None),
            )
            for group in range(8)
            if (group % 2 == 0) == (split == "train")
        }
        for split in ("train", "valid")
    }
    readers = {split: _MemorySequenceFile(values) for split, values in records.items()}
    refs = [
        {"split": "train" if group % 2 == 0 else "valid", "row_group": group} for group in range(8)
    ]

    fitting_sequences, raw_batch = read_parent_input_batch(readers, refs, 0, fitting=True)
    assert [sequence.seq_ix for sequence in fitting_sequences] == list(range(100, 108))
    assert raw_batch.shape == (8, SEQUENCE_LENGTH, 112)
    assert all(call[1:] == (True, True) for reader in readers.values() for call in reader.calls)
    assert not fit_sequence_targets_and_mask(fitting_sequences[0])[1].any()
    np.testing.assert_array_equal(
        fit_sequence_targets_and_mask(fitting_sequences[3])[1],
        np.ones(SEQUENCE_LENGTH, dtype=np.bool_),
    )
    for reader in readers.values():
        reader.calls.clear()

    evaluation_sequences, evaluation_raw = read_parent_input_batch(readers, refs, 0, fitting=False)
    assert [sequence.seq_ix for sequence in evaluation_sequences] == list(range(100, 108))
    assert all(
        sequence.targets is None and sequence.scored is None for sequence in evaluation_sequences
    )
    assert all(call[1:] == (False, False) for reader in readers.values() for call in reader.calls)
    environment = TrainingEnvironment(forward_chunk=_forward_first_two)
    model = torch.nn.Identity()
    batched = predict(
        model,
        torch.from_numpy(evaluation_raw),
        4096,
        torch.device("cpu"),
        environment=environment,
    )
    individual = np.stack(
        [
            predict(
                model,
                torch.from_numpy(sequence.features[None]),
                4096,
                torch.device("cpu"),
                environment=environment,
            )[0]
            for sequence in evaluation_sequences
        ]
    )
    np.testing.assert_array_equal(batched, individual)
