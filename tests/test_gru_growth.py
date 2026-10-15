"""Function-preserving recurrence and usable new-capacity checks."""

from __future__ import annotations

import pytest
import torch

from wnn.gru_growth import grow_gru_hidden, grow_gru_state
from wnn.nn import RNNModel


def _pair() -> tuple[RNNModel, RNNModel]:
    torch.manual_seed(73)
    parent = (
        RNNModel(11, hidden=5, layers=2, n_out=7, encoder_width=4, readout_width=6).double().eval()
    )
    child = (
        RNNModel(11, hidden=8, layers=2, n_out=7, encoder_width=7, readout_width=9).double().eval()
    )
    child.load_state_dict(grow_gru_state(parent.state_dict(), child.state_dict()))
    return parent, child


def test_all_outputs_and_old_hidden_survive_growth_and_chunking() -> None:
    parent, child = _pair()
    x = torch.randn(2, 17, 11, dtype=torch.float64)
    old_h = torch.randn(2, 2, 5, dtype=torch.float64)
    new_h = grow_gru_hidden(old_h, 8)
    with torch.no_grad():
        expected, expected_h = parent(x, old_h)
        actual, actual_h = child(x, new_h)
        pieces = []
        for lo, hi in ((0, 1), (1, 6), (6, 17)):
            part, new_h = child(x[:, lo:hi], new_h)
            pieces.append(part)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual_h[..., :5], expected_h, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-11, atol=1e-11)


def test_reset_and_arbitrary_new_initial_units_cannot_change_old_outputs() -> None:
    parent, child = _pair()
    x = torch.randn(1, 9, 11, dtype=torch.float64)
    h = torch.zeros(2, 1, 8, dtype=torch.float64)
    h[..., 5:] = torch.randn_like(h[..., 5:])
    with torch.no_grad():
        expected, _ = parent(x)
        actual, _ = child(x, h)
        child(torch.randn_like(x))
        reset, _ = child(x)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(reset, expected, rtol=1e-11, atol=1e-11)


def test_new_encoder_recurrence_and_readout_receive_gradients() -> None:
    _, child = _pair()
    optimizer = torch.optim.SGD(child.parameters(), lr=0.1)
    x = torch.randn(2, 13, 11, dtype=torch.float64)
    target = torch.randn(2, 13, 7, dtype=torch.float64)
    seen = {key: False for key in ("encoder", "recurrent", "readout", "connector")}
    for _ in range(4):
        optimizer.zero_grad()
        out, _ = child(x)
        (out - target).square().mean().backward()
        gradients = {
            "encoder": child.inp.weight.grad[4:],
            "recurrent": child.gru.weight_hh_l0.grad[5:8],
            "readout": child.head[0].weight.grad[6:],
            "connector": child.head[2].weight.grad[:, 6:],
        }
        for key, gradient in gradients.items():
            seen[key] |= bool(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)
        optimizer.step()
    assert all(seen.values()), seen


def test_rejects_shrink_and_unsupported_layout_without_mutating_parent() -> None:
    parent, child = _pair()
    before = {key: value.clone() for key, value in parent.state_dict().items()}
    with pytest.raises(ValueError, match="shrink"):
        grow_gru_state(child.state_dict(), parent.state_dict())
    altered = dict(child.state_dict())
    altered["unsupported"] = torch.zeros(1)
    with pytest.raises(ValueError, match="supported"):
        grow_gru_state(parent.state_dict(), altered)
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
