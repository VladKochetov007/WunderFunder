"""Native enc="conv" Predictor against the PyTorch model with true causal context.

The reference is one full-sequence forward: the causal conv sees zeros only before row 0, as the
streaming engine does (training and train_rnn.predict restart it at every chunk instead).
WNN_ROOT points at the checkout holding artifacts/ and datasets/ (default: this repository).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
native = pytest.importorskip("wnn_native")
if "conv" not in getattr(native, "ENCODERS", ()):
    pytest.skip("wnn_native build predates enc=conv (rebuild native/)", allow_module_level=True)

from wnn.features import event_features  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402

from .conftest import ROOT  # noqa: E402
from .test_predictor import _synthetic  # noqa: E402

DATA_ROOT = Path(os.environ.get("WNN_ROOT", ROOT))
RUN = DATA_ROOT / "artifacts" / "runs" / "ab61_conv_gru_f0" / "model.pt"
VALID = DATA_ROOT / "datasets" / "valid.parquet"
WARMUP = 99


def _weights(model):
    w = {k: np.ascontiguousarray(v.detach().float().numpy()) for k, v in model.state_dict().items()}
    w["head.2.weight"] = np.ascontiguousarray(w["head.2.weight"][:2])
    w["head.2.bias"] = np.ascontiguousarray(w["head.2.bias"][:2])
    return w


@torch.no_grad()
def _full_sequence(model, x):
    xin = torch.from_numpy(np.concatenate([x, event_features(x)], -1))[None]
    return model(xin)[0][0, :, :2].numpy()


def _conv_model(hidden=32, k=16):
    torch.manual_seed(0)
    m = RNNModel(211, hidden, 2, enc="conv", conv_k=k).eval()
    with torch.no_grad():  # default init leaves the conv branch small; make it matter
        m.conv_dw.weight.mul_(4.0)
        m.conv_pw.weight.mul_(2.0)
    return m


@pytest.mark.parametrize("fast,atol", [(False, 2e-5), (True, 5e-4)])
@pytest.mark.parametrize("k", [1, 4, 16])
def test_conv_predictor_matches_torch_on_synthetic(fast, atol, k):
    model = _conv_model(k=k)
    x = _synthetic(400)
    p = native.Predictor(_weights(model), fast_activations=fast)
    assert p.encoder == "conv"
    got = np.stack([p.step(r) for r in x])
    np.testing.assert_allclose(got, _full_sequence(model, x), atol=atol)
    p.reset()  # clears GRU state and the conv ring
    np.testing.assert_allclose(np.stack([p.step(r) for r in x[:40]]), got[:40], atol=0)


def test_linear_weights_select_linear_encoder():
    torch.manual_seed(0)
    p = native.Predictor(_weights(RNNModel(211, 16, 1).eval()))
    assert p.encoder == "linear"


def test_unconsumed_weights_are_rejected():
    w = _weights(_conv_model())
    with pytest.raises(ValueError, match="unsupported weights.*enc_norm.weight"):
        native.Predictor({**w, "enc_norm.weight": np.ones(32, np.float32)})
    del w["conv_pw.bias"]
    with pytest.raises(ValueError, match="missing weight conv_pw.bias"):
        native.Predictor(w)


@pytest.mark.skipif(not (RUN.exists() and VALID.exists()), reason="ab61 conv checkpoint or valid.parquet missing")
@pytest.mark.parametrize("group", [0, 1, 2])
def test_conv_checkpoint_streams_like_full_sequence_torch(group):
    from wnn.data import SequenceFile

    ck = torch.load(RUN, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = RNNModel(ck["n_in"], a["hidden"], a["layers"], 0.0, n_out=ck["model"]["head.2.bias"].shape[0],
                     enc=a["enc"], conv_k=a["conv_k"]).eval()
    model.load_state_dict(ck["model"])
    x = SequenceFile(VALID).read(group, targets=False, mask=False).features
    want = _full_sequence(model, x)
    p = native.Predictor(_weights(model), fast_activations=True)  # as shipped
    got = np.stack([p.step(r) for r in x])
    assert np.abs(got - want)[WARMUP:].max() <= 1e-4
