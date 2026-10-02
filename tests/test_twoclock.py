from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from wnn.event_content import ENDPOINT  # noqa: E402
from wnn.nn import A6, TwoClockModel  # noqa: E402
from wnn.twoclock_ref import TwoClockStream  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn  # noqa: E402


def _model(n_in=120, hidden=12, layers=2, event_hidden=8, n_out=5, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    m = TwoClockModel(n_in, hidden, layers, event_hidden, n_out=n_out).to(dtype).eval()
    with torch.no_grad():
        m.mean.normal_()
        m.std.uniform_(0.5, 2.0)
    return m


def _x(t=300, n_in=120, dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, t, n_in, generator=g)
    events = torch.rand(4, t, generator=g) < 0.2
    events[1] = False  # never an event
    events[2, 50:] = False  # long quiet tail
    events[3, :3] = True  # events from row 0, back to back
    x[..., A6] = torch.where(events, torch.tensor(ENDPOINT), x[..., A6])
    return x.to(dtype), events


def _chunked(m, x, chunk):
    state, outs = None, []
    for s in range(0, x.shape[1], chunk):
        o, state = m(x[:, s : s + chunk], state)
        state = train_rnn.detach_state(state)
        outs.append(o)
    return torch.cat(outs, 1), state


@pytest.mark.parametrize("chunk", [1, 7, 64, 300])
def test_chunked_state_matches_full_pass(chunk):
    m, (x, _) = _model(), _x()
    with torch.no_grad():
        full, end = m(x)
        got, state = _chunked(m, x, chunk)
    torch.testing.assert_close(got, full, rtol=0, atol=1e-10)
    for a, b in zip(state, end, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-10)


def test_chunked_float32_close():
    m, (x, _) = _model(dtype=torch.float32), _x(dtype=torch.float32)
    with torch.no_grad():
        full, _ = m(x)
        got, _ = _chunked(m, x, 64)
    torch.testing.assert_close(got, full, rtol=0, atol=1e-4)


def test_event_state_moves_only_on_event_rows():
    m, (x, events) = _model(), _x()
    with torch.no_grad():
        _, held, elapsed, _, _ = m.clocks(x)
    step = (held[:, 1:] - held[:, :-1]).abs().amax(-1)
    assert step[~events[:, 1:]].max() == 0
    assert step[events[:, 1:]].min() > 0
    assert held[1].abs().max() == 0  # no event: the zero initial state is carried
    assert (elapsed[events] == 0).all()
    assert elapsed[1].tolist() == list(range(1, 301))


def test_future_rows_do_not_change_past_outputs():
    m, (x, _) = _model(), _x()
    x2 = x.clone()
    x2[:, 200:] += 3.0
    x2[:, 200:, A6] = torch.where(torch.rand(4, 100) < 0.5, ENDPOINT, 0.0).to(x.dtype)
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x2)
    torch.testing.assert_close(a[:, :200], b[:, :200], rtol=0, atol=0)
    assert (a[:, 200:] - b[:, 200:]).abs().max() > 1e-3


@pytest.mark.parametrize(("n_in", "hidden", "layers", "event_hidden", "n_out"), [
    (120, 12, 2, 8, 5),
    (211, 192, 2, 128, 23),  # default size
])
def test_stream_reference_matches_torch(n_in, hidden, layers, event_hidden, n_out):
    m = _model(n_in, hidden, layers, event_hidden, n_out)
    x, _ = _x(200, n_in)
    with torch.no_grad():
        want, _ = m(x)
    ref = TwoClockStream(m.state_dict())
    for j in range(x.shape[0]):
        ref.reset()
        got = np.stack([ref.step(row) for row in x[j].numpy()])
        np.testing.assert_allclose(got, want[j].numpy(), rtol=0, atol=1e-9)


def test_stream_reference_float32_rows():
    m = _model(dtype=torch.float32)
    x, _ = _x(200, dtype=torch.float32)
    with torch.no_grad():
        want, _ = m(x)
    ref = TwoClockStream(m.state_dict())
    got = np.stack([ref.step(row) for row in x[0].numpy()])
    np.testing.assert_allclose(got, want[0].numpy(), rtol=0, atol=1e-5)


def test_head_only_freeze_covers_the_whole_trunk():
    args = {"arch": "twoclock", "hidden": 12, "layers": 2, "event_hidden": 8, "dropout": 0.0, "enc": "linear",
            "freeze": "inp,gru"}
    m = train_rnn.make_model(args, 120, 7)
    assert m.head[2].weight.shape[0] == 7  # predict_run reads head.2.weight
    frozen = tuple(args["freeze"].split(","))
    trunk = [n for n, _ in m.named_parameters() if not n.startswith("head.")]
    assert any(n.startswith("gru_event.") for n in trunk)
    assert all(n.startswith(frozen) for n in trunk)
    for partial in ("inp", "inp,gru.", "gru_event"):
        with pytest.raises(ValueError, match="whole trunk"):
            train_rnn.make_model({**args, "freeze": partial}, 120, 7)
