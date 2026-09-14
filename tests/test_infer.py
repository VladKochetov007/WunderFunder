from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wnn.infer import NumpyRNN, export_weights  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def _model(n_in=40, hidden=24, layers=2):
    torch.manual_seed(0)
    m = RNNModel(n_in, hidden, layers).eval()
    with torch.no_grad():
        m.mean.normal_()
        m.std.uniform_(0.5, 2.0)
    return m


def test_numpy_stream_matches_torch_sequence():
    m = _model()
    x = torch.randn(1, 50, 40)
    with torch.no_grad():
        want, _ = m(x)
    rnn = NumpyRNN(export_weights(m))
    got = np.stack([rnn.step(row) for row in x[0].numpy()])
    np.testing.assert_allclose(got, want[0].numpy(), atol=2e-5)


def test_reset_restores_initial_state():
    m = _model()
    rnn = NumpyRNN(export_weights(m))
    x = np.random.default_rng(0).normal(size=(5, 40)).astype(np.float32)
    first = rnn.step(x[0]).copy()
    for row in x[1:]:
        rnn.step(row)
    rnn.reset()
    np.testing.assert_array_equal(rnn.step(x[0]), first)


def test_rust_stepper_matches_numpy_and_torch():
    native = pytest.importorskip("wnn_native")
    m = _model(n_in=40, hidden=24, layers=2)
    weights = export_weights(m)
    stepper = native.GruStepper(weights)
    rnn = NumpyRNN(weights)
    x = np.random.default_rng(1).normal(size=(60, 40)).astype(np.float32)
    got = np.stack([stepper.step(r) for r in x])
    np.testing.assert_allclose(got, np.stack([rnn.step(r) for r in x]), atol=1e-5)
    stepper.reset()
    np.testing.assert_allclose(stepper.step(x[0]), got[0], atol=0)
    with pytest.raises(ValueError, match="expected 40 inputs"):
        stepper.step(x[0, :39])
