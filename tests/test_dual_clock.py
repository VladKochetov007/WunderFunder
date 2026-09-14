from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wnn.nn import DualClockModel  # noqa: E402


def _batch(b=3, t=240, n_in=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t, n_in, generator=g)
    events = torch.rand(b, t, generator=g) < 0.3
    events[1, :] = False  # a sequence with no events at all
    events[2, 100:] = False  # long quiet tail
    return x, events


def _model(n_in=20):
    torch.manual_seed(0)
    return DualClockModel(n_in, hidden=16, event_hidden=12).eval()


def test_prefix_invariance():
    m = _model()
    x, ev = _batch()
    with torch.no_grad():
        full, _ = m(x, ev)
        for cut in (1, 37, 150):
            part, _ = m(x[:, :cut], ev[:, :cut])
            torch.testing.assert_close(part, full[:, :cut], rtol=0, atol=1e-5)


def test_chunked_state_matches_full_pass():
    m = _model()
    x, ev = _batch()
    with torch.no_grad():
        full, _ = m(x, ev)
        state, outs = None, []
        for s in range(0, x.shape[1], 64):
            o, state = m(x[:, s : s + 64], ev[:, s : s + 64], state)
            outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=0, atol=1e-5)


def test_future_events_do_not_change_past_outputs():
    m = _model()
    x, ev = _batch()
    ev2 = ev.clone()
    ev2[:, 200:] = ~ev2[:, 200:]
    x2 = x.clone()
    x2[:, 200:] += 5.0
    with torch.no_grad():
        a, _ = m(x, ev)
        b, _ = m(x2, ev2)
    torch.testing.assert_close(a[:, :200], b[:, :200], rtol=0, atol=1e-6)
    assert (a[:, 200:] - b[:, 200:]).abs().max() > 1e-3


def test_event_state_only_moves_on_events():
    m = _model()
    x, ev = _batch()
    with torch.no_grad():
        _, (_, h_event, _) = m(x[1:2], ev[1:2])  # sequence without events
    assert torch.count_nonzero(h_event) == 0
