"""Bounded CPU invariants for the E476 source-only design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "src"))

import fit_activation_placement as e474  # noqa: E402
import fit_activation_schedule as e476  # noqa: E402

from wnn.optim import MuonAdamW  # noqa: E402


def test_lr_traces() -> None:
    traces = e476.make_lr_traces()
    assert tuple(traces) == e476.SCHEDULES
    assert all(trace.shape == (474,) for trace in traces.values())
    warmcos = traces["warmcos_47_427"]
    assert warmcos[0] == 0.0001
    assert warmcos[46] == warmcos[47] == 0.001
    assert warmcos[-1] == 0.0001
    assert np.all(np.diff(warmcos[:47]) > 0)
    assert np.all(np.diff(warmcos[47:]) < 0)
    assert not np.array_equal(traces["constant_peak"], traces["constant_mean_warmcos"])
    assert float(traces["constant_mean_warmcos"].mean()) == float(warmcos.mean())
    assert np.isclose(warmcos.sum(), traces["constant_mean_warmcos"].sum(), rtol=0, atol=1e-15)
    assert e476.e474_lr_trace().tolist() == [e474.ADAMW_LR] * 474


def test_fit_order_receipt_matches_training_traversal() -> None:
    nodes = [{"role": "fit", "row_group": index} for index in range(24)]
    order = e476.shuffled_fit_order(nodes)
    expected = list(range(24))
    np.random.default_rng(e476.SEED).shuffle(expected)
    assert order == expected
    listed = [nodes[index]["row_group"] for index in order]
    assert listed != [node["row_group"] for node in nodes]
    assert listed == [nodes[index]["row_group"] for index in e476.shuffled_fit_order(nodes)]
    wrong_roles = [*nodes[:-1], {"role": "test", "row_group": 23}]
    try:
        e476.shuffled_fit_order(wrong_roles)
    except ValueError:
        pass
    else:
        raise AssertionError("fit traversal accepted an exposed test group")


def test_parameter_layout_optimizer_scope_and_loss_mask() -> None:
    states = e474.make_seed_states()
    model = e476.model_for("base", states[e476.SEED])
    assert sum(parameter.numel() for parameter in model.parameters()) == 61_058
    adam = e476.make_optimizer("adamw", model)
    assert isinstance(adam, torch.optim.AdamW)
    muon = e476.make_optimizer("muon_adamw", model)
    assert isinstance(muon, MuonAdamW)
    recurrent = next(group for group in muon.param_groups if group.get("muon"))
    assert recurrent["blocks"] == 3
    names = {
        name
        for name, parameter in model.named_parameters()
        if any(parameter is item for item in recurrent["params"])
    }
    assert names == {
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.weight_ih_l1",
        "gru.weight_hh_l1",
    }
    # Match E470 exactly: need_prediction gates training; is_scored is absent.
    pred = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    target = torch.tensor([[[2.0, 3.0], [5.0, 9.0]]])
    need = torch.tensor([[True, False]])
    loss = e476.e470.loss_for_batch(pred, target, need)
    expected = (
        ((target[:, :1].abs()) * (pred[:, :1] - target[:, :1]).square()).sum((0, 1))
        / target[:, :1].abs().sum((0, 1))
    ).mean()
    torch.testing.assert_close(loss, expected)


def test_activation_parity_and_parameter_free_change() -> None:
    states = e474.make_seed_states()[e476.SEED]
    baseline = e476.model_for("base", states).eval()
    ref = e474.PlacementRNN("gelu", "gelu").eval()
    ref.load_state_dict(states, strict=True)
    sine = e476.model_for("sine_embed", states).eval()
    assert tuple(baseline.state_dict()) == tuple(sine.state_dict()) == tuple(ref.state_dict())
    assert sum(p.numel() for p in sine.parameters()) == e476.EXPECTED_PARAMETERS
    x = torch.randn(2, 17, 42)
    actual, actual_state = baseline(x)
    expected, expected_state = ref(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    assert torch.isfinite(sine(x)[0]).all()


def main() -> None:
    torch.set_num_threads(1)
    test_lr_traces()
    test_fit_order_receipt_matches_training_traversal()
    test_parameter_layout_optimizer_scope_and_loss_mask()
    test_activation_parity_and_parameter_free_change()
    print("E476 bounded CPU invariants passed")


if __name__ == "__main__":
    main()
