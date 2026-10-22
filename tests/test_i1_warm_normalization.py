from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wnn.i1_history import I1_HISTORY_MODE, I1_INPUT_WIDTH, i1_history_input_mask_indices
from wnn.nn import RNNModel
from wnn.run_identity import tensor_identity
from wnn.training.environment import TrainingEnvironment
from wnn.training.i1_inputs import configure_i1_input_mask
from wnn.training.i1_warm_normalization import (
    PARENT_INPUT_WIDTH,
    fit_i1_appended_normalization,
    validate_i1_warm_normalization,
)
from wnn.training.inheritance import can_extend_research_features, expand_research_feature_state


def test_none_to_i1_extension_preserves_parent_outputs_across_chunks_and_resets() -> None:
    """Zero-extended i1 inputs preserve the parent function for every sequence reset."""
    torch.manual_seed(20261022)
    parent = RNNModel(PARENT_INPUT_WIDTH, hidden=8, layers=1).eval()
    child = RNNModel(I1_INPUT_WIDTH, hidden=8, layers=1).eval()
    with torch.no_grad():
        parent.mean.normal_()
        parent.std.uniform_(0.5, 2.0)
    assert can_extend_research_features("none", I1_HISTORY_MODE)
    assert not can_extend_research_features(I1_HISTORY_MODE, I1_HISTORY_MODE)
    assert not can_extend_research_features("volume28", I1_HISTORY_MODE)
    child.load_state_dict(
        expand_research_feature_state(
            parent.state_dict(), child.state_dict(), "none", I1_HISTORY_MODE, PARENT_INPUT_WIDTH
        )
    )
    assert torch.equal(child.mean[:PARENT_INPUT_WIDTH], parent.mean)
    assert torch.equal(child.std[:PARENT_INPUT_WIDTH], parent.std)
    assert torch.equal(
        child.inp.weight[:, PARENT_INPUT_WIDTH:],
        torch.zeros_like(child.inp.weight[:, PARENT_INPUT_WIDTH:]),
    )

    for seed in (3, 19):
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(2, 29, PARENT_INPUT_WIDTH, generator=generator)
        history = torch.randn(2, 29, I1_INPUT_WIDTH - PARENT_INPUT_WIDTH, generator=generator)
        full, _ = parent(base)
        extended, _ = child(torch.cat((base, history), dim=-1))
        torch.testing.assert_close(extended, full, rtol=1e-6, atol=1e-6)

        parent_state = child_state = None
        parent_chunks = []
        child_chunks = []
        extended_inputs = torch.cat((base, history), dim=-1)
        for start, stop in ((0, 7), (7, 18), (18, 29)):
            parent_piece, parent_state = parent(base[:, start:stop], parent_state)
            child_piece, child_state = child(extended_inputs[:, start:stop], child_state)
            parent_chunks.append(parent_piece)
            child_chunks.append(child_piece)
        torch.testing.assert_close(
            torch.cat(child_chunks, dim=1), torch.cat(parent_chunks, dim=1), rtol=1e-6, atol=1e-6
        )


def test_i1_control_mask_is_restored_from_child_arguments() -> None:
    """A recreated child reinstalls its semantic mask after loading checkpoint tensors."""
    torch.manual_seed(27)
    parent = RNNModel(PARENT_INPUT_WIDTH, hidden=8, layers=1).eval()
    saved_child = RNNModel(I1_INPUT_WIDTH, hidden=8, layers=1).eval()
    state = expand_research_feature_state(
        parent.state_dict(), saved_child.state_dict(), "none", I1_HISTORY_MODE, PARENT_INPUT_WIDTH
    )
    restored = RNNModel(I1_INPUT_WIDTH, hidden=8, layers=1).eval()
    restored.load_state_dict(state)
    control_none = RNNModel(I1_INPUT_WIDTH, hidden=8, layers=1).eval()
    control_none.load_state_dict(state)
    child_args = {
        "research_features": I1_HISTORY_MODE,
        "research_feature_control": "all",
        "drop_i1": False,
    }
    configure_i1_input_mask(restored, child_args, I1_INPUT_WIDTH)
    assert restored._i1_input_mask_indices == i1_history_input_mask_indices("all", False)
    assert len(restored._i1_input_mask_indices) == I1_INPUT_WIDTH - PARENT_INPUT_WIDTH

    base = torch.randn(2, 13, PARENT_INPUT_WIDTH)
    arbitrary_history = torch.randn(2, 13, I1_INPUT_WIDTH - PARENT_INPUT_WIDTH)
    parent_output, _ = parent(base)
    child_output, _ = restored(torch.cat((base, arbitrary_history), dim=-1))
    none_output, _ = control_none(torch.cat((base, arbitrary_history), dim=-1))
    torch.testing.assert_close(child_output, parent_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(child_output, none_output, rtol=0, atol=0)


class _NormalizationBuffers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.linspace(-1.0, 1.0, I1_INPUT_WIDTH))
        self.register_buffer("std", torch.linspace(0.5, 2.0, I1_INPUT_WIDTH))


def test_fit_appended_normalization_preserves_prefix_and_records_fitting_stats(monkeypatch) -> None:
    """Only canonical fitting-loader statistics replace the appended 156 buffers."""
    from wnn.training import normalization

    rows = torch.arange(3 * 21 * I1_INPUT_WIDTH, dtype=torch.float32).reshape(3, 21, I1_INPUT_WIDTH)
    loader = DataLoader(TensorDataset(rows), batch_size=1, shuffle=False)
    model = _NormalizationBuffers()
    old_mean = model.mean[:PARENT_INPUT_WIDTH].clone()
    old_std = model.std[:PARENT_INPUT_WIDTH].clone()

    def identity_inputs(values: torch.Tensor, *, environment: TrainingEnvironment) -> torch.Tensor:
        del environment
        return values

    monkeypatch.setattr(normalization, "build_inputs", identity_inputs)
    metadata = fit_i1_appended_normalization(
        model, loader, torch.device("cpu"), environment=TrainingEnvironment()
    )
    sampled = torch.cat([rows[index].reshape(-1, I1_INPUT_WIDTH)[::7] for index in range(2)])
    torch.testing.assert_close(
        model.mean[PARENT_INPUT_WIDTH:], sampled.mean(0)[PARENT_INPUT_WIDTH:]
    )
    torch.testing.assert_close(
        model.std[PARENT_INPUT_WIDTH:], sampled.std(0).clamp_min(0.001)[PARENT_INPUT_WIDTH:]
    )
    assert torch.equal(model.mean[:PARENT_INPUT_WIDTH], old_mean)
    assert torch.equal(model.std[:PARENT_INPUT_WIDTH], old_std)
    assert metadata["parent_mean_identity"] == tensor_identity(old_mean)
    assert metadata["parent_std_identity"] == tensor_identity(old_std)
    assert metadata["appended_mean_identity"] == tensor_identity(model.mean[PARENT_INPUT_WIDTH:])


def _warm_args(**updates: object) -> Namespace:
    values: dict[str, object] = {
        "i1_warm_normalization": "fit-appended",
        "init": "parent@ep1",
        "inherit_init_preprocessing": True,
        "resume_state": None,
        "research_features": I1_HISTORY_MODE,
        "research_feature_control": "none",
        "research_feature_adapter_only": False,
        "drop_i1": False,
    }
    values.update(updates)
    return Namespace(**values)


def _standard_parent_checkpoint() -> dict[str, object]:
    return {
        "args": {"arch": "gru", "cell": "gru", "enc": "linear"},
        "model": {
            "mean": torch.zeros(PARENT_INPUT_WIDTH),
            "std": torch.ones(PARENT_INPUT_WIDTH),
            "inp.weight": torch.zeros(8, PARENT_INPUT_WIDTH),
        },
    }


def test_fit_appended_normalization_accepts_only_standard_none_parent() -> None:
    """The accepted warm path derives width from stored parent state and checks its layout."""
    validate_i1_warm_normalization(_warm_args(), _standard_parent_checkpoint())


@pytest.mark.parametrize(
    ("args_updates", "checkpoint_update", "message"),
    [
        ({"init": ""}, None, "initializer checkpoint"),
        ({"inherit_init_preprocessing": False}, None, "inherited preprocessing"),
        ({"resume_state": "state.pt"}, None, "resume"),
        ({"research_feature_adapter_only": True}, None, "feature-adapter-only"),
        ({"drop_i1": True}, None, "drop-i1"),
        ({}, {"args": {"research_features": I1_HISTORY_MODE}}, "parent research_features"),
        (
            {},
            {
                "model": {
                    "mean": torch.zeros(I1_INPUT_WIDTH),
                    "std": torch.ones(I1_INPUT_WIDTH),
                    "inp.weight": torch.zeros(8, I1_INPUT_WIDTH),
                }
            },
            "standard 211-slot parent",
        ),
    ],
)
def test_fit_appended_normalization_rejects_unsupported_warm_paths(
    args_updates: dict[str, object],
    checkpoint_update: dict[str, object] | None,
    message: str,
) -> None:
    """No-init, resumed, already-expanded, or unsupported parent states fail closed."""
    checkpoint = _standard_parent_checkpoint()
    if checkpoint_update:
        checkpoint.update(checkpoint_update)
    with pytest.raises(ValueError, match=message):
        validate_i1_warm_normalization(_warm_args(**args_updates), checkpoint)


def test_default_i1_warm_normalization_is_inherit_and_requires_no_parent() -> None:
    """The opt-in has an inert default for every existing non-warm recipe."""
    from wnn.training.cli import build_parser

    args = build_parser().parse_args(["--name", "warm-normalization-default-test"])
    assert args.i1_warm_normalization == "inherit"
    validate_i1_warm_normalization(args, None)
