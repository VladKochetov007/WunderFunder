from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
native = pytest.importorskip("wnn_native")

from wnn.features import event_features  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402

from .conftest import DATASETS, ROOT  # noqa: E402

RUN = ROOT / "artifacts" / "runs" / "ab9_pow2_h128" / "model.pt"


def _synthetic(t=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, 112)).astype(np.float32)
    hold = rng.random((t, 112)) < 0.7
    hold[0] = False
    for i in range(1, t):
        x[i, hold[i]] = x[i - 1, hold[i]]
    return x


def _weights(model):
    return {k: np.ascontiguousarray(v.detach().numpy(), np.float32) for k, v in model.state_dict().items()}


def _batch(model, x):
    with torch.no_grad():
        y, _ = model(torch.cat([torch.from_numpy(x), torch.from_numpy(event_features(x))], -1)[None])
    return y[0, :, :2].numpy()


@pytest.mark.parametrize("fast,atol", [(False, 2e-5), (True, 5e-4)])
def test_predictor_matches_batch_on_synthetic(fast, atol):
    torch.manual_seed(0)
    model = RNNModel(211, 32, 2).eval()
    x = _synthetic()
    p = native.Predictor(_weights(model), fast_activations=fast)
    got = np.stack([p.step(r) for r in x])
    np.testing.assert_allclose(got, _batch(model, x), atol=atol)
    p.reset()
    np.testing.assert_allclose(p.step(x[0]), got[0], atol=0)


@pytest.mark.data
@pytest.mark.skipif(not RUN.exists(), reason="trained checkpoint not present")
def test_predictor_matches_trained_checkpoint_on_real_sequence():
    from wnn.data import SequenceFile

    ck = torch.load(RUN, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = RNNModel(ck["n_in"], a["hidden"], a["layers"], 0.0).eval()
    model.load_state_dict(ck["model"])
    x = SequenceFile(Path(DATASETS) / "valid.parquet").read(0).features
    want = _batch(model, x)
    p = native.Predictor(_weights(model), fast_activations=True)
    got = np.stack([p.step(r) for r in x])
    assert np.abs(got - want).max() < 2e-3
    assert np.corrcoef(got[:, 0], want[:, 0])[0, 1] > 0.99999
