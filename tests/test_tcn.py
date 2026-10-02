from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from wnn.nn import TCNModel  # noqa: E402
from wnn.tcn_ref import TCNStream  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn  # noqa: E402


def _model(n_in=20, channels=16, layers=6, n_out=5, mix=False, seed=0):
    torch.manual_seed(seed)
    m = TCNModel(n_in, channels, layers, n_out=n_out, mix=mix).eval()
    with torch.no_grad():
        m.mean.normal_()
        m.std.uniform_(0.5, 2.0)
    return m


def _x(b=3, t=300, n_in=20, seed=1):
    return torch.randn(b, t, n_in, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("mix", [False, True])
@pytest.mark.parametrize("chunk", [7, 50, 64, 2000])
def test_chunked_state_matches_full_pass(mix, chunk):
    m, x = _model(mix=mix), _x()
    with torch.no_grad():
        full, _ = m(x)
        state, outs = None, []
        for s in range(0, x.shape[1], chunk):
            o, state = m(x[:, s : s + chunk], state)
            state = train_rnn.detach_state(state)
            outs.append(o)
    assert [t.shape[1] for t in state] == [2**i for i in range(6)]
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=0, atol=1e-5)


def test_receptive_field_is_two_to_the_layers():
    m, x = _model(layers=6), _x(b=1)
    x2 = x.clone()
    x2[:, 100] += 3.0
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x2)
    diff = (a - b).abs().amax(-1)[0]
    assert diff[:100].max() == 0
    assert diff[100 + 63] > 1e-4
    assert diff[100 + 64 :].max() < 1e-6


@pytest.mark.parametrize(("n_in", "channels", "layers", "n_out", "mix", "t"), [
    (20, 16, 5, 5, False, 150),
    (20, 16, 5, 5, True, 150),
    (211, 128, 12, 23, False, 160),  # default size: the large dilations still read their zero history
])
def test_stream_reference_matches_torch(n_in, channels, layers, n_out, mix, t):
    m = _model(n_in, channels, layers, n_out, mix)
    x = _x(1, t, n_in)
    with torch.no_grad():
        want, _ = m(x)
    ref = TCNStream(m.state_dict())
    got = np.stack([ref.step(row) for row in x[0].numpy()])
    np.testing.assert_allclose(got, want[0].numpy(), rtol=0, atol=1e-5)
    ref.reset()
    np.testing.assert_allclose(ref.step(x[0, 0].numpy()), want[0, 0].numpy(), rtol=0, atol=1e-5)


def test_make_model_inherit_and_head_freeze():
    args = {"arch": "tcn", "hidden": 16, "layers": 4, "dropout": 0.0, "enc": "linear", "tcn_mix": 0, "freeze": ""}
    parent = train_rnn.make_model(args, 20, 7)
    assert isinstance(parent, TCNModel)
    assert parent.head[2].weight.shape[0] == 7  # predict_run reads head.2.weight
    ck = {"args": {**args, "aux_struct": 0.3}}
    ns = train_rnn.argparse.Namespace(**{f: None for f in ("hidden", "layers", "dropout", "arch", "decay_init",
                                                           "memory_arm", "cell", "enc", "tok_dim", "att_layers",
                                                           "conv_k", "tcn_mix", "event_hidden", "online_norm",
                                                           "raw_only", "heuristics", "prop_input", "aux",
                                                           "aux_struct", "aux_next", "aux_next_k", "aux_tier",
                                                           "aux_mask", "aux_sign", "aux_quote", "quote_cache_root")})
    train_rnn.inherit_init_arguments(ns, ck, [])
    child_args = {**vars(ns), "freeze": "inp,tcn"}
    child = train_rnn.make_model(child_args, 20, 7)
    child.load_state_dict(parent.state_dict())
    trainable = {n for n, p in child.named_parameters() if not n.startswith(("inp", "tcn"))}
    assert trainable == {"head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"}
    with pytest.raises(ValueError, match="trunk is inp,tcn"):
        train_rnn.make_model({**child_args, "freeze": "inp,gru"}, 20, 7)
