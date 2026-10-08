from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import train_parent_readout as runner  # noqa: E402
from wnn.conditional_readout import (  # noqa: E402
    READOUT_MODES,
    ConditionalReadoutHeads,
    primary_pair_wmse,
)


def _clone(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_parent_output_width_accepts_only_the_bound_23_or_31_contract() -> None:
    runner._validate_parent_output_count(23, 23)
    runner._validate_parent_output_count(31, 31)
    for actual, bound in ((24, 24), (23, 31), (31, 23), (23, 24)):
        with pytest.raises(ValueError):
            runner._validate_parent_output_count(actual, bound)


def test_group_ids_are_compared_in_source_qualified_namespaces() -> None:
    components = {
        ("train", 4): 10,
        ("valid", 2): 20,
        ("valid", 4): 30,
    }
    direct, linked = runner._qualified_fit_overlap((4,), (2,), (4,), components)
    assert direct == set()
    assert linked == set()

    direct, linked = runner._qualified_fit_overlap((4,), (4,), (4,), components)
    assert direct == {("valid", 4)}
    assert linked == {30}


def test_v2_selected_initial_heads_match_frozen_h272_constructor_order() -> None:
    reference_path = (
        PACKAGE_ROOT.parent / "H272_DIRECT_LABEL_READOUT_V1_RUNTIME/src/wnn/conditional_readout.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_h272_conditional_readout", reference_path)
    assert spec is not None and spec.loader is not None
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)

    torch.manual_seed(3)
    reference = reference_module.ConditionalReadoutHeads(512, 32)
    torch.manual_seed(3)
    candidate = ConditionalReadoutHeads(512, 32)
    for mode in ("hidden_only", "multiplicative"):
        expected = reference.controls[mode].state_dict()
        actual = candidate.controls[mode].state_dict()
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[name], actual[name]) for name in expected)


def test_one_synthetic_update_changes_only_two_readouts_and_preserves_parent() -> None:
    torch.manual_seed(3)

    class FrozenParent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Sequential(nn.Linear(512, 256))
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    parent = FrozenParent()
    heads = ConditionalReadoutHeads(512, 32)
    trainable = [
        parameter for mode in runner.MODES for parameter in heads.controls[mode].parameters()
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)
    parent_before = _clone(parent)
    heads_before = {mode: _clone(heads.controls[mode]) for mode in READOUT_MODES}

    base = torch.randn(1, 12, 23)
    hidden = torch.randn(1, 12, 512)
    context = torch.randn(1, 12, 99)
    target = torch.randn(1, 12, 2).clamp(-2, 2)
    label = target.clone()
    row_weight = torch.ones(1, 12)
    outputs = heads.forward_all(base, hidden, context)
    optimizer.zero_grad(set_to_none=True)
    for mode in runner.MODES:
        loss = primary_pair_wmse(outputs[mode], target, row_weight, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.controls[mode].parameters(), 1.0)
    optimizer.step()

    assert all(
        torch.equal(parent_before[name], value) for name, value in parent.state_dict().items()
    )
    for mode in runner.MODES:
        assert any(
            not torch.equal(heads_before[mode][name], value)
            for name, value in heads.controls[mode].state_dict().items()
        )
    for mode in set(READOUT_MODES) - set(runner.MODES):
        assert all(
            torch.equal(heads_before[mode][name], value)
            for name, value in heads.controls[mode].state_dict().items()
        )


def test_full_data_recipe_explicitly_disables_prediction_and_score_paths() -> None:
    recipe = runner._load_recipe(
        PACKAGE_ROOT / "recipes/H272_PARENT_READOUT_V2_FULL_DATA_20261008.json"
    )
    assert recipe["paths"]["prediction_export"] is None
    assert recipe["deployment_only"]["score_allowed"] is False
    assert recipe["deployment_only"]["fold0_grade_allowed"] is False
    assert recipe["propensity"]["run_record"]["runtime_source_bytes_attested"] is False


def test_profile_and_export_completion_gates_are_fixed_and_fail_closed() -> None:
    runner._validate_profile_plan(warmup_chunks=2, timed_chunks=8, cpu_threads=1)
    for values in ((1, 8, 1), (2, 7, 1), (2, 8, 2)):
        with pytest.raises(ValueError):
            runner._validate_profile_plan(*values)

    recipe = runner._load_recipe(
        PACKAGE_ROOT / "recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json"
    )
    epoch_updates = recipe["training"]["expected_updates_per_epoch_per_head"]
    complete = {
        "status": "complete",
        "selected_epoch": 1,
        "optimizer_completed_updates_per_head": [epoch_updates * 2] * 2,
        "epoch_update_evidence": [
            {
                "optimizer_attempts_per_head": epoch_updates,
                "completed_updates_per_head": epoch_updates,
                "amp_skipped_updates": 0,
                "finite_losses": True,
                "finite_gradients": True,
            }
            for _ in range(2)
        ],
    }
    runner._validate_fit_completion(complete, recipe)

    skipped = {
        **complete,
        "epoch_update_evidence": [dict(item) for item in complete["epoch_update_evidence"]],
    }
    skipped["epoch_update_evidence"][1]["amp_skipped_updates"] = 1
    with pytest.raises(ValueError, match="complete fixed-update evidence"):
        runner._validate_fit_completion(skipped, recipe)
