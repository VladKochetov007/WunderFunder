import numpy as np
import torch

from wnn.i1_lead_block import MODE, WIDTH, StreamingI1LeadBlock, features_numpy
from wnn.research_features import (
    StreamingResearchFeatures,
    research_feature_names,
    research_features_numpy,
    research_features_torch,
)


def _raw(seed: int = 0, batch: int = 2, rows: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(batch, rows, 112)).astype(np.float32)
    # i0 book columns change only on sparse rows, as in the data.
    hold = rng.random((batch, rows)) > 0.15
    for b in range(batch):
        for t in range(1, rows):
            if hold[b, t]:
                raw[b, t, :22] = raw[b, t - 1, :22]
    return raw


def test_held_values_follow_last_i0_book_event() -> None:
    raw = _raw()
    out = features_numpy(raw)
    for b in range(raw.shape[0]):
        last = 0
        for t in range(raw.shape[1]):
            if t and np.any(raw[b, t, :22] != raw[b, t - 1, :22]):
                last = t
            np.testing.assert_array_equal(out[b, t], raw[b, last, 52:104])


def test_numpy_torch_streaming_parity_and_causality() -> None:
    raw = _raw(1)
    expected = research_features_numpy(raw, MODE)
    got = research_features_torch(torch.from_numpy(raw), MODE).numpy()
    np.testing.assert_array_equal(expected, got)
    stream = StreamingResearchFeatures(MODE)
    for b in range(raw.shape[0]):
        stream.reset()
        rows = np.stack([stream.update(row) for row in raw[b]])
        np.testing.assert_array_equal(rows, expected[b])
    changed = raw.copy()
    changed[:, 120:] = np.random.default_rng(9).normal(size=changed[:, 120:].shape)
    np.testing.assert_array_equal(features_numpy(changed)[:, :120], expected[:, :120])


def test_zero_control_and_width() -> None:
    raw = _raw(2)
    assert len(research_feature_names(MODE)) == WIDTH == 52
    assert not research_features_numpy(raw, MODE, control="all").any()
    assert not research_features_torch(torch.from_numpy(raw), MODE, control="all").any()
    stream = StreamingResearchFeatures(MODE, control="all")
    assert not stream.update(raw[0, 0]).any()


def test_streaming_reset_restarts_snapshot() -> None:
    raw = _raw(3, batch=1)[0]
    stream = StreamingI1LeadBlock()
    for row in raw[:50]:
        stream.update(row)
    stream.reset()
    np.testing.assert_array_equal(stream.update(raw[10]), raw[10, 52:104])
