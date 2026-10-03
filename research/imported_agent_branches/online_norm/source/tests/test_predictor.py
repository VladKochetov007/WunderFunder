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


@pytest.mark.skipif("online_norm" not in getattr(native, "CAPABILITIES", ()),
                    reason="installed wnn_native predates online_norm; rebuild it (maturin develop)")
def test_online_norm_predictor_matches_batch_on_synthetic():
    from wnn.features import online_zscore

    torch.manual_seed(0)
    model = RNNModel(323, 32, 2).eval()
    x = _synthetic(3000) * np.float32(3.0)
    inputs = np.concatenate([x, event_features(x), online_zscore(x)], -1)
    with torch.no_grad():
        want, _ = model(torch.from_numpy(inputs)[None])
    p = native.Predictor(_weights(model), fast_activations=False, online_norm=True)
    got = np.stack([p.step(r) for r in x])
    np.testing.assert_allclose(got, want[0, :, :2].numpy(), atol=2e-5)
    p.reset()  # running sums restart with the sequence
    np.testing.assert_allclose(p.step(x[0]), got[0], atol=0)
    with pytest.raises(ValueError, match="online_norm=false"):
        native.Predictor(_weights(model))


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


def _thread_probe(code: str) -> int:
    """Run code in a subprocess (an abort would kill pytest itself) and return its exit code."""
    import subprocess
    import sys

    return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120).returncode


def test_predictor_survives_cross_thread_use():
    code = (
        "import threading, numpy as np, wnn_native\n"
        "import torch\n"
        "from wnn.nn import RNNModel\n"
        "m = RNNModel(211, 8, 1).eval()\n"
        "w = {k: np.ascontiguousarray(v.detach().numpy(), np.float32) for k, v in m.state_dict().items()}\n"
        "p = wnn_native.Predictor(w)\n"
        "out = []\n"
        "t = threading.Thread(target=lambda: out.append(p.step(np.zeros(112, np.float32))))\n"
        "t.start(); t.join()\n"
        "assert len(out) == 1 and out[0].shape == (2,)\n"
    )
    assert _thread_probe(code) == 0


def test_nonfinite_input_does_not_poison_state():
    torch.manual_seed(0)
    model = RNNModel(211, 16, 2).eval()
    x = _synthetic(300)
    clean = native.Predictor(_weights(model))
    dirty = native.Predictor(_weights(model))
    for i, r in enumerate(x):
        a = clean.step(r)
        bad = r.copy()
        if i == 100:
            bad[5] = np.nan
        b = dirty.step(bad)
    assert np.isfinite(b).all()
    assert np.abs(a - b).max() < 1.0  # recovers instead of emitting a fixed fallback forever
