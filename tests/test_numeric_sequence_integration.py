"""Receiver-only checks for the shared-initialization causal numerical-input experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_numeric import StreamingNumericInputs, numeric_inputs


def _model(arm: str) -> torch.nn.Module:
    torch.manual_seed(1001)
    return build_sequence_model(
        {
            "architecture": "transformer",
            "input_encoding": arm,
            "hidden": 32,
            "layers": 2,
            "heads": 4,
            "ff_multiplier": 2,
            "context_rows": 5,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
        }
    ).eval()


def test_common_initialization_and_zero_basis_preserve_reference_output() -> None:
    """The extra branch changes neither shared parameters nor initial predictions."""
    models = [_model(arm) for arm in ("raw_events", "uniform16", "quantile16")]
    states = [dict(model.named_parameters()) for model in models]
    for name, parameter in states[0].items():
        for state in states[1:]:
            torch.testing.assert_close(parameter, state[name], atol=0, rtol=0)
    x = torch.randn(2, 9, 323)
    for model in models[1:]:
        model.numeric_basis.install_knots(torch.linspace(-2, 2, 17).expand(112, -1).clone())
        torch.testing.assert_close(model(x)[0], models[0](x)[0], atol=0, rtol=0)
    assert sum(p.numel() for p in models[1].parameters()) == sum(
        p.numel() for p in models[2].parameters()
    )


@pytest.mark.parametrize("arm", ["raw_events", "uniform16", "quantile16"])
def test_numeric_full_chunk_row_prefix_and_checkpoint_parity(arm: str, tmp_path: Path) -> None:
    """Nonzero basis weights stay causal through preprocessing and cached attention."""
    model = _model(arm)
    if model.numeric_basis is not None:
        model.numeric_basis.install_knots(torch.linspace(-2, 2, 17).expand(112, -1).clone())
        torch.nn.init.normal_(model.numeric_basis.projection.weight, std=0.01)
    raw = np.random.default_rng(7).normal(size=(13, 112)).astype(np.float32)
    raw[5, 3] = np.nan
    x = torch.from_numpy(numeric_inputs(raw)).unsqueeze(0)
    full = model(x)[0]
    changed = raw.copy()
    changed[7:] *= 10
    torch.testing.assert_close(
        full[:, :7], model(torch.from_numpy(numeric_inputs(changed)).unsqueeze(0))[0][:, :7]
    )
    parts, state = [], None
    for left, right in ((0, 2), (2, 8), (8, 13)):
        prediction, _, state = model(x[:, left:right], state)
        parts.append(prediction)
    torch.testing.assert_close(full, torch.cat(parts, dim=1))
    stream, parts, state = StreamingNumericInputs(), [], None
    for row in raw:
        prediction, _, state = model(torch.from_numpy(stream.update(row))[None, None], state)
        parts.append(prediction)
    torch.testing.assert_close(full, torch.cat(parts, dim=1))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = _model(arm)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(full, restored(x)[0])


def test_all_published_configs_are_consumable_and_fixed() -> None:
    """Every panel cell has a valid runtime schema and the intended endpoint/budget."""
    directory = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/typed-bin-feature-encoding/configs"
    )
    paths = list(directory.glob("*.json"))
    assert len(paths) == 6
    for path in paths:
        config = load_config(path)
        assert config["epochs"] == 12 and config["batch_size"] == 32
        assert config["required_world_size"] == 1 and config["schedule"] == "wsd"
        assert config["expected_fold"] in (0, 2)
        assert config["training_mask"] == "need_prediction"
