from __future__ import annotations

import numpy as np
import pytest

from wnn.data import SEQUENCE_LENGTH, SequenceFile
from wnn.features import (
    FEATURE_NAMES,
    N_EVENT_FEATURES,
    StreamingEventFeatures,
    event_features,
    event_features_torch,
)

from .conftest import DATASETS


def _synthetic(rng, t=600):
    """Piecewise-constant columns with sparse joint updates, like the real feature groups."""
    x = rng.normal(size=(t, 112)).astype(np.float32)
    hold = rng.random((t, 112)) < 0.7
    hold[0] = False
    for i in range(1, t):
        x[i, hold[i]] = x[i - 1, hold[i]]
    # Joint bid/ask price updates on a subset of rows.
    joint = rng.random(t) < 0.3
    for i in range(1, t):
        if not joint[i]:
            x[i, 0:22] = x[i - 1, 0:22]
    return x


def _prefix_invariant(fn, x, cuts):
    full = fn(x)
    return all(np.array_equal(fn(x[:k]), full[:k]) for k in cuts)


def test_names_match_width():
    assert len(FEATURE_NAMES) == N_EVENT_FEATURES
    assert event_features(_synthetic(np.random.default_rng(0))).shape[1] == N_EVENT_FEATURES


def test_batch_is_prefix_invariant_and_checker_catches_leak():
    rng = np.random.default_rng(1)
    x = _synthetic(rng)
    cuts = [1, 2, 57, 300, 599]
    assert _prefix_invariant(event_features, x, cuts)

    def leaky(z):  # uses the next row: must be rejected by the same check
        nxt = np.concatenate([z[1:], z[-1:]], axis=0)
        return event_features(z) + (nxt[:, :1] - z[:, :1])

    assert not _prefix_invariant(leaky, x, cuts)


def test_streaming_matches_batch():
    x = _synthetic(np.random.default_rng(2))
    stream = StreamingEventFeatures()
    rows = np.stack([stream.update(r).copy() for r in x])
    np.testing.assert_array_equal(rows, event_features(x))
    stream.reset()
    np.testing.assert_array_equal(stream.update(x[0]), event_features(x[:1])[0])


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    x = np.stack([_synthetic(np.random.default_rng(s)) for s in (3, 4)])
    got = event_features_torch(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, event_features(x), rtol=0, atol=1e-6)


@pytest.mark.data
def test_real_sequence_streaming_equivalence():
    seq = SequenceFile(DATASETS / "valid.parquet").read(0)
    feats = event_features(seq.features)
    assert feats.shape == (SEQUENCE_LENGTH, N_EVENT_FEATURES)
    stream = StreamingEventFeatures()
    for i in range(2000):
        np.testing.assert_array_equal(stream.update(seq.features[i]), feats[i])
    # H-001's universal target-clock assertion was falsified by the October 17
    # census. Input-feature parity is the contract; label timing is not.


def test_online_zscore_is_causal_and_matches_oracle():
    torch = pytest.importorskip("torch")
    from wnn.features import online_zscore_torch

    x = torch.from_numpy(_synthetic(np.random.default_rng(9), t=200)[None])
    z = online_zscore_torch(x)
    for t in (0, 1, 57, 199):
        prefix = x[0, : t + 1].double()
        want = (prefix[-1] - prefix.mean(0)) / torch.sqrt(prefix.var(0, unbiased=False) + 1e-4)
        torch.testing.assert_close(z[0, t].double(), want, atol=1e-4, rtol=0)
    x2 = x.clone()
    x2[:, 150:] += 7.0
    torch.testing.assert_close(online_zscore_torch(x2)[:, :150], z[:, :150])
