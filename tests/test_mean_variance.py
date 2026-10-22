from __future__ import annotations

import argparse

import pytest
import torch

from wnn.nn import RNNModel
from wnn.training.cli_targets import add_targets_arguments
from wnn.training.mean_variance import (
    append_log_variance_output_head,
    configuration_error,
    extend_mean_variance_state,
)


def test_optional_head_preserves_existing_readout_and_zeroes_new_rows() -> None:
    torch.manual_seed(31)
    model = RNNModel(n_in=3, hidden=4, layers=1, n_out=23)
    features = torch.randn(2, 5, 8)
    original = model.head(features).detach().clone()

    append_log_variance_output_head(model)

    expanded = model.head(features)
    assert expanded.shape == (2, 5, 25)
    assert torch.equal(expanded[..., :23], original)
    assert torch.count_nonzero(expanded[..., 23:]) == 0


def test_parent_twenty_three_row_state_extends_to_zero_initialized_twenty_five() -> None:
    torch.manual_seed(37)
    parent_model = RNNModel(n_in=3, hidden=4, layers=1, n_out=23)
    parent_state = {
        name: value.detach().clone() for name, value in parent_model.state_dict().items()
    }
    child_model = RNNModel(n_in=3, hidden=4, layers=1, n_out=23)
    append_log_variance_output_head(child_model)
    child_state = child_model.state_dict()

    extended = extend_mean_variance_state(parent_state, child_state)

    for name, parent_value in parent_state.items():
        if name == "head.2.weight" or name == "head.2.bias":
            assert torch.equal(extended[name][:23], parent_value)
            assert torch.count_nonzero(extended[name][23:]) == 0
        else:
            assert torch.equal(extended[name], parent_value)


def test_head_is_off_by_default_and_configuration_is_compatible() -> None:
    parser = argparse.ArgumentParser()
    add_targets_arguments(parser)
    args = parser.parse_args([])

    assert args.mean_variance_head == 0
    assert configuration_error(args, controlled_loss=False) is None


def test_appending_head_does_not_advance_rng() -> None:
    model = RNNModel(n_in=3, hidden=4, layers=1, n_out=23)
    rng_state = torch.get_rng_state().clone()

    append_log_variance_output_head(model)

    assert torch.equal(torch.get_rng_state(), rng_state)


def test_state_extension_rejects_other_tensor_shape_changes() -> None:
    parent_model = RNNModel(n_in=3, hidden=4, layers=1, n_out=23)
    parent_state = {
        name: value.detach().clone() for name, value in parent_model.state_dict().items()
    }
    child_model = RNNModel(n_in=3, hidden=4, layers=1, n_out=23)
    append_log_variance_output_head(child_model)
    child_state = dict(child_model.state_dict())
    child_state["inp.weight"] = child_state["inp.weight"][:, :-1]

    with pytest.raises(ValueError, match="outside output head"):
        extend_mean_variance_state(parent_state, child_state)
