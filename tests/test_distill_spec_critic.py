"""Focused, low-cost checks derived from the frozen distillation contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from wnn.distillation import extend_family_output_head

from scripts import package_submission, train_rnn


def _small_gru_args() -> dict[str, object]:
    return {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 4,
        "layers": 1,
        "dropout": 0.0,
        "readout_activation": "gelu",
    }


def _first_shuffled_epoch(output_width: int) -> torch.Tensor:
    """Follow production model initialization with RandomSampler's first epoch."""
    torch.manual_seed(3)
    model = train_rnn.make_model(_small_gru_args(), n_in=3, n_out=output_width)
    loader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, shuffle=True, num_workers=0)
    order = torch.cat([batch[0] for batch in loader])
    del model
    return order


def test_family_head_keeps_same_seed_random_sampler_order() -> None:
    """C2/C3 matched seeds must not change row order when only the head is added."""
    c2_order = _first_shuffled_epoch(output_width=23)
    c3_order = _first_shuffled_epoch(output_width=31)

    torch.testing.assert_close(c2_order, c3_order, rtol=0, atol=0)


def test_family_head_warm_start_preserves_23_rows_and_package_uses_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training checkpoint keeps parent rows; the native interface keeps t0/t1."""
    parent = {
        "trunk": torch.arange(12, dtype=torch.float32),
        "head.2.weight": torch.arange(23 * 3, dtype=torch.float32).reshape(23, 3),
        "head.2.bias": torch.arange(23, dtype=torch.float32),
    }
    initialized = {
        "trunk": torch.zeros(12),
        "head.2.weight": torch.full((31, 3), -5.0),
        "head.2.bias": torch.full((31,), -5.0),
    }
    merged = extend_family_output_head(parent, initialized)
    torch.testing.assert_close(merged["trunk"], parent["trunk"], rtol=0, atol=0)
    torch.testing.assert_close(
        merged["head.2.weight"][:23], parent["head.2.weight"], rtol=0, atol=0
    )
    torch.testing.assert_close(merged["head.2.bias"][:23], parent["head.2.bias"], rtol=0, atol=0)

    checkpoint = {
        "args": {"readout_activation": "gelu"},
        "model": merged,
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    package_path = tmp_path / "weights.npz"
    package_submission.export("critic_student", package_path)

    with np.load(package_path, allow_pickle=False) as packaged:
        np.testing.assert_array_equal(
            packaged["head.2.weight"], parent["head.2.weight"][:2].numpy()
        )
        np.testing.assert_array_equal(packaged["head.2.bias"], parent["head.2.bias"][:2].numpy())


def test_family_disabled_model_has_legacy_23_output_shape() -> None:
    """The opt-in head is represented only by a larger requested output width."""
    torch.manual_seed(11)
    model = train_rnn.make_model(_small_gru_args(), n_in=3, n_out=23)
    with torch.inference_mode():
        output, _state = model(torch.zeros(2, 5, 3))
    assert output.shape == (2, 5, 23)
