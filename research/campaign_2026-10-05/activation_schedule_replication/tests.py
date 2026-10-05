"""Bounded synthetic CPU invariants for the E480 two-seed source design."""

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
import fit_activation_schedule_replication as e480  # noqa: E402

from wnn.optim import MuonAdamW  # noqa: E402


def test_lr_traces() -> None:
    traces = e480.make_lr_traces()
    e480.verify_e476_lr_traces(traces)
    assert tuple(traces) == e480.SCHEDULES
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
    assert e480.e474_lr_trace().tolist() == [e474.ADAMW_LR] * 474


def test_fit_order_receipt_matches_training_traversal() -> None:
    nodes = [{"role": "fit", "row_group": index} for index in range(24)]
    order = e480.shuffled_fit_order(nodes, 271829)
    expected = list(range(24))
    np.random.default_rng(271829).shuffle(expected)
    assert order == expected
    listed = [nodes[index]["row_group"] for index in order]
    assert listed != [node["row_group"] for node in nodes]
    assert listed == [nodes[index]["row_group"] for index in e480.shuffled_fit_order(nodes, 271829)]
    second_order = e480.shuffled_fit_order(nodes, 271830)
    assert second_order != order
    assert [nodes[index]["row_group"] for index in second_order] != listed
    wrong_roles = [*nodes[:-1], {"role": "test", "row_group": 23}]
    try:
        e480.shuffled_fit_order(wrong_roles, 271829)
    except ValueError:
        pass
    else:
        raise AssertionError("fit traversal accepted an exposed test group")


def test_parameter_layout_optimizer_scope_and_loss_mask() -> None:
    states = e474.make_seed_states()
    for seed in e480.REPLICATION_SEEDS:
        initial = e480.make_initial_state(seed)
        assert e480.make_initial_state(seed).keys() == initial.keys()
        assert all(torch.equal(initial[key], e480.make_initial_state(seed)[key]) for key in initial)
    assert all(
        torch.equal(states[271829][key], e480.make_initial_state(271829)[key])
        for key in states[271829]
    )
    model = e480.model_for(271829, "base", states[271829])
    assert sum(parameter.numel() for parameter in model.parameters()) == 61_058
    adam = e480.make_optimizer("adamw", model)
    assert isinstance(adam, torch.optim.AdamW)
    muon = e480.make_optimizer("muon_adamw", model)
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
    loss = e480.e470.loss_for_batch(pred, target, need)
    expected = (
        ((target[:, :1].abs()) * (pred[:, :1] - target[:, :1]).square()).sum((0, 1))
        / target[:, :1].abs().sum((0, 1))
    ).mean()
    torch.testing.assert_close(loss, expected)


def test_activation_parity_and_parameter_free_change() -> None:
    ref = e474.PlacementRNN("gelu", "gelu").eval()
    for seed in e480.REPLICATION_SEEDS:
        states = e480.make_initial_state(seed)
        baseline = e480.model_for(seed, "base", states).eval()
        sine = e480.model_for(seed, "sine_embed", states).eval()
        ref.load_state_dict(states, strict=True)
        assert tuple(baseline.state_dict()) == tuple(sine.state_dict()) == tuple(ref.state_dict())
        assert all(torch.equal(value, baseline.state_dict()[key]) for key, value in states.items())
        assert all(torch.equal(value, sine.state_dict()[key]) for key, value in states.items())
        assert sum(p.numel() for p in sine.parameters()) == e480.EXPECTED_PARAMETERS
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
    print("E480 bounded synthetic CPU invariants passed")


if __name__ == "__main__":
    main()
