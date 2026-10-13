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
    _LearnedScaleWeight,
    _quantize_block_with_scales,
    _represented_matrix,
    _RepresentedWeight,
    configure_qat,
    effective_qat_state,
    inference_state,
    invalidate_qat_cache,
    project_qat_scales_,
    qat_forward,
)
from wnn.training_state import capture_training_state, restore_training_state


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


@pytest.mark.parametrize(
    "preset", ["recurrent", "recurrent_head", "recurrent_w4", "input_w4_recurrent_w8"]
)
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


def test_q4_cache_refreshes_after_parameter_version_and_explicit_invalidation() -> None:
    torch.manual_seed(417)
    model = RNNModel(3, hidden=4, layers=1).eval()
    configure_qat(model, "recurrent_w4")
    features = torch.randn(2, 8, 3)
    with torch.no_grad():
        qat_forward(model, features, None)
    name = "gru.weight_ih_l0"
    first = model._qat_cache[name]
    with torch.no_grad():
        model.gru.weight_ih_l0.add_(2.0)
        qat_forward(model, features, None)
    second = model._qat_cache[name]
    assert second is not first
    assert second[0] != first[0]
    assert not torch.equal(second[1], first[1])
    invalidate_qat_cache(model)
    assert model._qat_cache == {}
    with torch.no_grad():
        qat_forward(model, features, None)
    assert model._qat_cache[name] is not second


@pytest.mark.parametrize("preset", ["recurrent_w4", "input_w4_recurrent_w8"])
def test_learned_scale_arm_starts_at_fixed_representation_and_filters_parameters(preset: str) -> None:
    torch.manual_seed(4401)
    model = RNNModel(3, hidden=4, layers=1)
    latent = copy.deepcopy(model.state_dict())
    expected, expected_metadata = effective_qat_state(latent, preset, "raw")
    configure_qat(model, preset, learn_scales=True)

    actual, metadata = inference_state(model, "raw")
    assert set(actual) == set(latent)
    assert not any(name.startswith("_qat_scale_params.") for name in actual)
    assert metadata["scale_learning"]["version"] == 1
    assert "not a full LSQ reproduction" in metadata["scale_learning"]["reproduction_status"]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for name in expected_metadata["encoded"]:
        torch.testing.assert_close(metadata["encoded"][name], expected_metadata["encoded"][name], rtol=0, atol=0)


def test_learned_scale_gradient_uses_lsq_row_rule_and_identity_weight_ste() -> None:
    latent = torch.tensor([[-8, -2.5, -1.5, 0, 1.5, 2.5, 8]], dtype=torch.float32, requires_grad=True)
    step = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    codes_np, _, represented_np = _quantize_block_with_scales(
        latent.detach().numpy(), np.array([1.0], dtype=np.float32), 4
    )
    represented = torch.from_numpy(represented_np)
    codes = torch.from_numpy(codes_np)
    direction = torch.ones_like(latent)
    (_LearnedScaleWeight.apply(latent, step, represented, codes, 4) * direction).sum().backward()

    ratio = latent.detach().numpy()[0]
    q_codes = codes_np[0].astype(np.float32)
    derivative = np.where(ratio <= -7, -7, np.where(ratio >= 7, 7, q_codes - ratio))
    expected_scale_gradient = derivative.sum() / np.sqrt(7 * 7)
    torch.testing.assert_close(latent.grad, direction, rtol=0, atol=0)
    assert step.grad.item() == pytest.approx(float(expected_scale_gradient), abs=1e-7)


def test_learned_scales_resume_optimizer_and_handle_zero_or_nonfinite_rows(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(4402)
    model = RNNModel(3, hidden=4, layers=1)
    configure_qat(model, "recurrent_w4", learn_scales=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    inputs = torch.randn(1, 5, 3)

    optimizer.zero_grad(set_to_none=True)
    output, _ = qat_forward(model, inputs, None)
    output.square().mean().backward()
    optimizer.step()
    project_qat_scales_(model)
    invalidate_qat_cache(model)
    scheduler.step()

    state = capture_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader=loader,
        contract={"epochs": 1},
        next_epoch=1,
        progress={"step_count": 1},
    )
    checkpoint = tmp_path / "learned_scale_training_state.pt"
    torch.save(state, checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    resumed = RNNModel(3, hidden=4, layers=1)
    configure_qat(resumed, "recurrent_w4", learn_scales=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
    resumed_loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    restore_training_state(
        state,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        loader=resumed_loader,
    )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    for parameter, restored in zip(model.parameters(), resumed.parameters(), strict=True):
        assert set(optimizer.state[parameter]) == set(resumed_optimizer.state[restored])
        for key, value in optimizer.state[parameter].items():
            other = resumed_optimizer.state[restored][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(other, value, rtol=0, atol=0)
            else:
                assert other == value

    for continued, continued_optimizer in ((model, optimizer), (resumed, resumed_optimizer)):
        continued_optimizer.zero_grad(set_to_none=True)
        continued_output, _ = qat_forward(continued, inputs, None)
        continued_output.square().mean().backward()
        continued_optimizer.step()
        project_qat_scales_(continued)
        invalidate_qat_cache(continued)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    for parameter, restored in zip(model.parameters(), resumed.parameters(), strict=True):
        for key, value in optimizer.state[parameter].items():
            other = resumed_optimizer.state[restored][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(other, value, rtol=0, atol=0)
            else:
                assert other == value

    # A fully zero latent row keeps its valid zero step; once it becomes live,
    # the first represented forward installs the fixed absmax/QP initialization.
    name = "gru.weight_ih_l0"
    position = model._qat_scale_lookup[(name, 0)]
    row = 0
    with torch.no_grad():
        model.gru.weight_ih_l0[row].zero_()
        model._qat_scale_params[position][row] = 0
    invalidate_qat_cache(model)
    qat_forward(model, inputs, None)
    assert model._qat_scale_params[position][row].item() == 0
    with torch.no_grad():
        model.gru.weight_ih_l0[row, 0] = 0.25
    invalidate_qat_cache(model)
    qat_forward(model, inputs, None)
    assert model._qat_scale_params[position][row].item() > 0

    with torch.no_grad():
        model._qat_scale_params[position][row] = float("nan")
    invalidate_qat_cache(model)
    with pytest.raises(ValueError, match="invalid learned-scale"):
        qat_forward(model, inputs, None)


def test_negative_learned_step_resets_with_quantizer_min_subnormal() -> None:
    model = RNNModel(3, hidden=4, layers=1)
    configure_qat(model, "recurrent_w4", learn_scales=True)
    name = "gru.weight_ih_l0"
    position = model._qat_scale_lookup[(name, 0)]
    minimum_subnormal = np.nextafter(np.float32(0), np.float32(1))
    with torch.no_grad():
        model.gru.weight_ih_l0[0].zero_()
        model.gru.weight_ih_l0[0, 0] = float(minimum_subnormal)
        model.gru.weight_ih_l0[1].zero_()
        model._qat_scale_params[position][0] = -1.0
        model._qat_scale_params[position][1] = -1.0

    project_qat_scales_(model)

    assert model._qat_scale_params[position][0].item() == float(minimum_subnormal)
    assert model._qat_scale_params[position][1].item() == 0


def test_learned_scales_survive_ema_and_swa_model_clones() -> None:
    model = RNNModel(3, hidden=4, layers=1)
    configure_qat(model, "recurrent_w4", learn_scales=True)
    ema = training.FastEMA(model, 0.9)
    swa = torch.optim.swa_utils.AveragedModel(model)

    with torch.no_grad():
        model._qat_scale_params[0].add_(0.125)
    ema.update_parameters(model)
    swa.update_parameters(model)

    for averaged in (ema.module, swa.module):
        assert averaged._qat_learn_scales
        assert averaged._qat_scale_keys == model._qat_scale_keys
        torch.testing.assert_close(
            averaged._qat_scale_params[0], model._qat_scale_params[0], rtol=0, atol=0
        )
        _, metadata = inference_state(averaged, "ema_or_swa")
        assert metadata["scale_learning"]["version"] == 1


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
