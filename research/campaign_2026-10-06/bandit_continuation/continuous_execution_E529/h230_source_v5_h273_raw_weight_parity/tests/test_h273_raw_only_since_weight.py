from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "h273_train_rnn", SOURCE_ROOT / "scripts" / "train_rnn.py"
)
assert SPEC is not None and SPEC.loader is not None
TRAIN_RNN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN_RNN
SPEC.loader.exec_module(TRAIN_RNN)


def _weights(inputs: torch.Tensor, raw: torch.Tensor | None) -> torch.Tensor:
    batch, steps, _ = inputs.shape
    target = torch.zeros(batch, steps, 2)
    valid = torch.ones(batch, steps, dtype=torch.bool)
    scored = torch.zeros(batch, steps, dtype=torch.bool)
    args = Namespace(
        mask_weight=False,
        since_weight=32.0,
        since_floor=0.25,
        episode_weight=0.0,
        episode_floor=0.5,
        raw_only=TRAIN_RNN.RAW_ONLY,
        class_weight="",
        valid_true_mask=-1.0,
        sync_free=False,
    )
    return TRAIN_RNN.main_row_weights(
        inputs, target, valid, scored, None, None, args, raw
    )


def _raw_batch() -> torch.Tensor:
    from wnn.data import I0

    raw = torch.zeros(2, 12, TRAIN_RNN.N_FEATURES)
    raw[0, 1, I0.bid_px.start] = 1.0
    raw[0, 1, I0.ask_px.start] = 1.0
    raw[1, 3, I0.bid_px.start] = 1.0
    raw[1, 3, I0.ask_px.start] = 1.0
    return raw


def test_raw_only_preserves_since_weighting_for_identical_raw_rows() -> None:
    raw = _raw_batch()
    old_raw_only = TRAIN_RNN.RAW_ONLY
    try:
        TRAIN_RNN.RAW_ONLY = False
        full_inputs = TRAIN_RNN.build_inputs(raw)
        full_weights = _weights(full_inputs, raw)

        TRAIN_RNN.RAW_ONLY = True
        raw_inputs = TRAIN_RNN.build_inputs(raw)
        raw_weights = _weights(raw_inputs, raw)
    finally:
        TRAIN_RNN.RAW_ONLY = old_raw_only

    assert raw_inputs.shape[-1] == full_inputs.shape[-1]
    assert torch.count_nonzero(raw_inputs[..., TRAIN_RNN.N_FEATURES :]) == 0
    torch.testing.assert_close(raw_weights, full_weights, rtol=0.0, atol=0.0)
    assert torch.any(full_weights < 1.0)
    assert torch.all(full_weights >= 0.25)


def test_raw_only_since_weighting_fails_closed_without_raw_batch() -> None:
    raw = _raw_batch()
    old_raw_only = TRAIN_RNN.RAW_ONLY
    try:
        TRAIN_RNN.RAW_ONLY = True
        raw_inputs = TRAIN_RNN.build_inputs(raw)
        with pytest.raises(ValueError, match="original raw batch"):
            _weights(raw_inputs, None)
    finally:
        TRAIN_RNN.RAW_ONLY = old_raw_only
