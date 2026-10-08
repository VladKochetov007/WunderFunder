from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import train_rnn as trainer  # noqa: E402
from wnn.conditional_readout import ConditionalReadoutHeads  # noqa: E402

from wnn.data import WARMUP  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402

wnn_native = pytest.importorskip("wnn_native")


def _to_arrays(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for name, value in module.state_dict().items()
    }


def _torch_prediction(model: RNNModel, control: torch.nn.Module, raw: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(raw)[None]
    inputs = trainer.build_inputs(values)
    projected = model.encode((inputs - model.mean) / model.std)
    recurrent, _ = model.gru(projected)
    hidden = torch.cat((recurrent, projected), dim=-1)
    parent_output = model.head(hidden)
    context = inputs[..., 112:211]
    output = control(parent_output[..., :2], hidden, context)
    return output[0].detach().cpu().numpy()


def _raw_sequences() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(341)
    first = rng.normal(0.0, 0.7, size=(129, 112)).astype(np.float32)
    second = rng.normal(0.0, 0.7, size=(117, 112)).astype(np.float32)
    # Add a few stable book updates and price moves so event flags and held
    # price deltas are exercised in addition to dense synthetic variation.
    first[1:, 0:22] = first[0:1, 0:22]
    first[7, 0] += 1.0
    first[7, 11] += 1.0
    second[2:, 52:74] = second[1:2, 52:74]
    second[9, 52] += 0.5
    second[9, 63] -= 0.5
    return first, second


def _make_models() -> tuple[
    RNNModel, torch.nn.Module, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    torch.manual_seed(71)
    model = RNNModel(211, hidden=256, layers=2, dropout=0.0, n_out=23)
    model.eval()
    torch.manual_seed(3)
    heads = ConditionalReadoutHeads(512, rank=32)
    control = heads.controls["multiplicative"]
    torch.nn.init.normal_(control.output.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(control.output.bias, mean=0.0, std=0.02)
    control.eval()
    return model, control, _to_arrays(model), _to_arrays(control)


def test_native_readout_matches_torch_over_sequences_reset_and_repeated_calls() -> None:
    model, control, parent_weights, readout_weights = _make_models()
    first, second = _raw_sequences()
    predictor = wnn_native.ConditionalPredictor(
        parent_weights, readout_weights, fast_activations=False, precision="f32"
    )

    observed_first = np.stack([predictor.step(row) for row in first])
    np.testing.assert_allclose(
        observed_first, _torch_prediction(model, control, first), rtol=2e-4, atol=2e-5
    )

    predictor.reset()
    observed_second = np.stack([predictor.step(row) for row in second])
    np.testing.assert_allclose(
        observed_second, _torch_prediction(model, control, second), rtol=2e-4, atol=2e-5
    )

    def callback_outputs() -> list[np.ndarray]:
        callback = wnn_native.ConditionalPredictor(
            parent_weights, readout_weights, fast_activations=False, precision="f32"
        )
        found: list[np.ndarray] = []
        for sequence_id, rows in ((17, first), (18, second)):
            for index, row in enumerate(rows):
                result = callback.predict(
                    SimpleNamespace(seq_ix=sequence_id, state=row, need_prediction=index >= WARMUP)
                )
                if result is not None:
                    found.append(np.asarray(result))
        return found

    callback_first = callback_outputs()
    callback_second = callback_outputs()
    expected = np.concatenate(
        (
            _torch_prediction(model, control, first)[WARMUP:],
            _torch_prediction(model, control, second)[WARMUP:],
        ),
        axis=0,
    )
    np.testing.assert_allclose(np.stack(callback_first), expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(np.stack(callback_first), np.stack(callback_second))


def test_native_readout_is_causal_to_future_row_changes() -> None:
    _, _, parent_weights, readout_weights = _make_models()
    first, _ = _raw_sequences()
    changed_future = first.copy()
    changed_future[20:, 0:22] += 4.0
    left = wnn_native.ConditionalPredictor(parent_weights, readout_weights, precision="f32")
    right = wnn_native.ConditionalPredictor(parent_weights, readout_weights, precision="f32")
    left_outputs = np.stack([left.step(row) for row in first])
    right_outputs = np.stack([right.step(row) for row in changed_future])
    np.testing.assert_array_equal(left_outputs[:20], right_outputs[:20])
