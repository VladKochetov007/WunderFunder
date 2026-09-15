from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wnn.features import event_features  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402
from wnn.streaming import StreamingGraph, initial_state  # noqa: E402


def _rows(t=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, 112)).astype(np.float32)
    hold = rng.random((t, 112)) < 0.7
    hold[0] = False
    for i in range(1, t):
        x[i, hold[i]] = x[i - 1, hold[i]]
    joint = rng.random(t) < 0.3
    for i in range(1, t):
        if not joint[i]:
            x[i, 0:22] = x[i - 1, 0:22]
    return x


def _model(hidden=24, layers=2):
    torch.manual_seed(0)
    m = RNNModel(112 + 99, hidden, layers).eval()
    with torch.no_grad():
        m.mean.normal_()
        m.std.uniform_(0.5, 2.0)
    return m


def test_row_stepping_matches_batch_features_and_model():
    x = _rows()
    model = _model()
    graph = StreamingGraph(model).eval()
    with torch.no_grad():
        feats = torch.from_numpy(event_features(x))
        want, _ = model(torch.cat([torch.from_numpy(x), feats], -1)[None])
        state = initial_state(model)
        got = []
        for row in x:
            y, *state = graph(torch.from_numpy(row)[None], *state)
            got.append(y[0])
    np.testing.assert_allclose(torch.stack(got).numpy(), want[0].numpy(), atol=2e-5)


def test_reset_state_reproduces_first_output():
    x = _rows(10)
    graph = StreamingGraph(_model()).eval()
    with torch.no_grad():
        state = initial_state(graph.model)
        first, *state = graph(torch.from_numpy(x[0])[None], *state)
        for row in x[1:]:
            _, *state = graph(torch.from_numpy(row)[None], *state)
        again, *_ = graph(torch.from_numpy(x[0])[None], *initial_state(graph.model))
    torch.testing.assert_close(first, again)
