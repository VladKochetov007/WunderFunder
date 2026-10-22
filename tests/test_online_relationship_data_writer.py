import numpy as np
import pytest

from wnn.data import Sequence
from wnn.training.online_relationship_data_writer import fit_sequence_targets_and_mask


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
