from __future__ import annotations

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_trainer_resume_integration import training
from wnn.nn import RNNModel
from wnn.quantization import quantize_model, quantize_rows, represented_rows
from wnn.training_qat import (
    _represented_matrix,
    _RepresentedWeight,
    configure_qat,
    inference_state,
    invalidate_qat_cache,
    qat_forward,
)


def test_canonical_ties_subnormals_zero_rows_and_identity_ste() -> None:
    tiny = np.nextafter(np.float32(0), np.float32(1))
    values = np.array(
        [[127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], [tiny, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        np.float32,
    )
    latent = torch.tensor(values, requires_grad=True)
    represented = _represented_matrix("gru.weight_hh_l0", latent, 7)
    expected = represented_rows(*quantize_rows(values))
    np.testing.assert_array_equal(represented.numpy(), expected)
    direction = torch.arange(latent.numel(), dtype=torch.float32).reshape(latent.shape)
    (_RepresentedWeight.apply(latent, represented) * direction).sum().backward()
    torch.testing.assert_close(latent.grad, direction, rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["recurrent", "recurrent_head"])
def test_full_20000_row_qat_matches_reconstructed_checkpoint(preset: str) -> None:
    torch.manual_seed(69)
    model = RNNModel(3, hidden=4, layers=2).eval()
    raw = copy.deepcopy(model.state_dict())
    original = copy.deepcopy(model)
    configure_qat(model, preset)
    features = torch.randn(1, 20000, 3)
    with torch.no_grad():
        fake, hidden = qat_forward(model, features, None)
    assert fake.dtype == hidden.dtype == torch.float32
    assert all(torch.equal(model.state_dict()[name], value) for name, value in raw.items())
    represented, metadata = inference_state(model, "raw")
    arrays = {name: value.numpy() for name, value in raw.items()}
    expected, encoded, report = quantize_model(arrays, preset)
    for name in represented:
        np.testing.assert_array_equal(represented[name].numpy(), expected[name])
    for name, value in encoded.items():
        np.testing.assert_array_equal(metadata["encoded"][name].numpy(), value)
    reconstructed = RNNModel(3, hidden=4, layers=2).eval()
    reconstructed.load_state_dict(represented)
    with torch.no_grad():
        actual, actual_hidden = reconstructed(features)
    torch.testing.assert_close(fake, actual, rtol=0, atol=0)
    torch.testing.assert_close(hidden, actual_hidden, rtol=0, atol=0)
    for name in report["unquantized"]:
        assert torch.equal(represented[name], original.state_dict()[name])
    # A post-forward deepcopy must contain no retained non-leaf functional GRU weights.
    duplicate = copy.deepcopy(model)
    with torch.no_grad():
        copied, _ = qat_forward(duplicate, features[:, :8], None)
        original_prefix, _ = qat_forward(model, features[:, :8], None)
    torch.testing.assert_close(copied, original_prefix, rtol=0, atol=0)


def test_head_halves_have_independent_scale_and_frozen_ema_trunk_codes() -> None:
    torch.manual_seed(17)
    model = RNNModel(3, hidden=4, layers=1)
    with torch.no_grad():
        model.head[0].weight[:, 4:] *= 100
    baseline, baseline_metadata = inference_state(model, "raw")
    assert baseline_metadata is None
    configure_qat(model, "recurrent_head")
    baseline, baseline_metadata = inference_state(model, "raw")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("head."))
    averaged = training.FastEMA(model, 0.9)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output, _ = qat_forward(model, torch.randn(2, 8, 3), None)
        output.square().mean().backward()
        optimizer.step()
        invalidate_qat_cache(model)
        averaged.update_parameters(model)
        invalidate_qat_cache(averaged.module)
    effective, metadata = inference_state(averaged.module, "ema")
    for name in baseline:
        if not name.startswith("head."):
            assert torch.equal(effective[name], baseline[name])
    for name in metadata["encoded"]:
        if name.startswith("gru."):
            assert torch.equal(metadata["encoded"][name], baseline_metadata["encoded"][name])
    assert not torch.equal(
        metadata["encoded"]["head.0.weight.0.scales"], metadata["encoded"]["head.0.weight.1.scales"]
    )
    assert metadata["snapshot_source"] == "ema"
    assert not torch.equal(effective["head.2.weight"], baseline["head.2.weight"])


def test_full_model_ste_gradients_match_effective_float_model() -> None:
    torch.manual_seed(690)
    latent = RNNModel(3, hidden=4, layers=1)
    configure_qat(latent, "recurrent_head")
    effective = RNNModel(3, hidden=4, layers=1)
    effective.load_state_dict(inference_state(latent, "raw")[0])
    features = torch.randn(2, 9, 3)
    target = torch.randn(2, 9, 2)
    predicted, _ = qat_forward(latent, features, None)
    expected, _ = effective(features)
    (predicted - target).square().mean().backward()
    (expected - target).square().mean().backward()
    for (name, raw), (_, represented) in zip(
        latent.named_parameters(), effective.named_parameters(), strict=True
    ):
        assert raw.grad is not None, name
        torch.testing.assert_close(raw.grad, represented.grad, rtol=0, atol=0)


def test_no_finetune_export_keeps_bytes_and_rejects_repacking(tmp_path: Path) -> None:
    exporter = importlib.import_module("export_qat_control")
    model = RNNModel(3, hidden=4, layers=1)
    source = tmp_path / "source.pt"
    output = tmp_path / "control.pt"
    torch.save({"model": model.state_dict(), "args": {"precision": "fp16"}, "n_in": 3}, source)
    exporter.export_control(source, output, "recurrent_head")
    checkpoint = torch.load(output, weights_only=False)
    assert checkpoint["quantization"]["training_updates"] == 0
    assert checkpoint["args"]["precision"] == "fp16"  # preserve historical fit recipe
    with pytest.raises(ValueError, match="repack"):
        exporter.export_control(output, tmp_path / "again.pt", "recurrent_head")
