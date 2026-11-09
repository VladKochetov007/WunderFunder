"""Prepared CPU contracts for parameter-free Transformer Q/K RMS normalization."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

from wnn.training.sequence_attention import CausalTransformerBlock, TransformerState, _rotate
from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_provenance import canonical_sha256
from wnn.training.sequence_qk_norm import QK_NORM_EPSILON, headwise_rms_normalize


def _model_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "transformer",
        "hidden": 32,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 2,
        "context_rows": 5,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "precision_mode": "fp32",
        "attention_backend": "math",
        "compile_blocks": False,
    }
    config.update(overrides)
    return config


def _write_recipe(path: Path, model: dict[str, object]) -> Path:
    model_values = dict(model)
    precision_mode = model_values.pop("precision_mode", "fp32")
    path.write_text(
        json.dumps({"precision_mode": precision_mode, "model": model_values}), encoding="utf-8"
    )
    return path


def test_none_is_canonical_and_preserves_legacy_initialization_and_arithmetic(
    tmp_path: Path,
) -> None:
    legacy = load_config(_write_recipe(tmp_path / "legacy.json", _model_config()))
    explicit_none = load_config(
        _write_recipe(tmp_path / "none.json", _model_config(qk_norm="none"))
    )
    assert "qk_norm" not in legacy["model"]
    assert "qk_norm" not in explicit_none["model"]
    assert legacy == explicit_none
    assert canonical_sha256(legacy) == canonical_sha256(explicit_none)

    torch.manual_seed(101)
    legacy_model = build_sequence_model(legacy["model"])
    legacy_rng = torch.random.get_rng_state()
    torch.manual_seed(101)
    none_model = build_sequence_model(explicit_none["model"])
    none_rng = torch.random.get_rng_state()
    assert torch.equal(legacy_rng, none_rng)
    assert legacy_model.state_dict().keys() == none_model.state_dict().keys()
    assert all(
        torch.equal(value, none_model.state_dict()[name])
        for name, value in legacy_model.state_dict().items()
    )

    values = torch.randn(2, 8, 112)
    legacy_model.eval()
    none_model.eval()
    legacy_prediction, legacy_hints, _ = legacy_model(values)
    none_prediction, none_hints, _ = none_model(values)
    torch.testing.assert_close(legacy_prediction, none_prediction, rtol=0.0, atol=0.0)
    torch.testing.assert_close(legacy_hints, none_hints, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (False, TypeError),
        (None, TypeError),
        (1, TypeError),
        ([], TypeError),
        ("cosine", ValueError),
        ("RMS", ValueError),
    ],
)
def test_invalid_qk_norm_values_are_rejected(
    tmp_path: Path, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="qk_norm"):
        load_config(_write_recipe(tmp_path / "invalid.json", _model_config(qk_norm=value)))
    with pytest.raises(error, match="qk_norm"):
        build_sequence_model(_model_config(qk_norm=value))


def test_rms_qk_norm_is_transformer_only(tmp_path: Path) -> None:
    recipe = _model_config(architecture="gru", qk_norm="rms")
    with pytest.raises(ValueError, match="Transformer"):
        load_config(_write_recipe(tmp_path / "gru.json", recipe))
    with pytest.raises(ValueError, match="Transformer"):
        build_sequence_model(recipe)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_headwise_rms_formula_dtype_zero_vectors_and_gradients(dtype: torch.dtype) -> None:
    values = torch.tensor([[[[3.0, 4.0], [0.0, 0.0]]]], dtype=dtype, requires_grad=True)
    actual = headwise_rms_normalize(values)
    expected = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + QK_NORM_EPSILON
    )
    assert actual.dtype is dtype
    torch.testing.assert_close(actual.float(), expected.to(dtype).float(), rtol=0.0, atol=0.0)
    assert torch.equal(actual[..., 1, :], torch.zeros_like(actual[..., 1, :]))
    actual.float().square().sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_qk_norm_preserves_causal_chunk_step_and_reset_equivalence() -> None:
    torch.manual_seed(211)
    model = build_sequence_model(_model_config(qk_norm="rms")).eval()
    values = torch.randn(2, 13, 112)
    full_prediction, full_hints, full_state = model(values)
    assert isinstance(full_state, TransformerState)

    altered = values.clone()
    altered[:, 7:] += 100.0
    changed_prediction, _, _ = model(altered)
    torch.testing.assert_close(full_prediction[:, :7], changed_prediction[:, :7])

    chunk_predictions: list[Tensor] = []
    chunk_hints: list[Tensor] = []
    chunk_state: TransformerState | None = None
    for start, end in ((0, 3), (3, 10), (10, 13)):
        prediction, hints, chunk_state = model(values[:, start:end], chunk_state)
        chunk_predictions.append(prediction)
        chunk_hints.append(hints)
    torch.testing.assert_close(torch.cat(chunk_predictions, dim=1), full_prediction)
    torch.testing.assert_close(torch.cat(chunk_hints, dim=1), full_hints)

    step_predictions: list[Tensor] = []
    step_state: TransformerState | None = None
    for row in range(values.shape[1]):
        prediction, _, step_state = model(values[:, row : row + 1], step_state)
        step_predictions.append(prediction)
    torch.testing.assert_close(torch.cat(step_predictions, dim=1), full_prediction)
    assert step_state is not None
    assert step_state.next_position == values.shape[1]
    assert all(key.shape[-2] <= model.context_rows - 1 for key in step_state.keys)
    assert all(key.shape[-2] <= model.context_rows - 1 for key in full_state.keys)

    prefix = values[:, :6]
    reset_rows = values[:, 6:10]
    _, _, carried_state = model(prefix)
    carried_prediction, _, continued_state = model(reset_rows, carried_state)
    reset_prediction, _, reset_state = model(reset_rows)
    isolated_prediction, _, isolated_state = model(reset_rows)
    torch.testing.assert_close(reset_prediction, isolated_prediction, rtol=0.0, atol=0.0)
    assert not torch.allclose(carried_prediction, reset_prediction)
    assert continued_state.next_position == prefix.shape[1] + reset_rows.shape[1]
    assert reset_state.next_position == reset_rows.shape[1]
    assert isolated_state.next_position == reset_rows.shape[1]


def test_cached_keys_are_normalized_rotated_once_and_old_keys_are_untouched() -> None:
    torch.manual_seed(307)
    block = CausalTransformerBlock(16, 2, 32, 0.0, qk_norm=True).eval()
    values = torch.randn(1, 1, 16)
    past_key = torch.full((1, 2, 2, 8), 2.0)
    past_value = torch.randn_like(past_key)
    captured_keys: list[Tensor] = []
    captured_values: list[Tensor] = []

    def capture_attention_key(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
        key, value = args[1], args[2]
        assert isinstance(key, Tensor) and isinstance(value, Tensor)
        captured_keys.append(key.clone())
        captured_values.append(value.clone())

    hook = block.local_attention.register_forward_pre_hook(capture_attention_key)
    try:
        _, next_key, _ = block(values, past_key, past_value, 3, 6, 0.0)
    finally:
        hook.remove()

    assert len(captured_keys) == 1
    assert len(captured_values) == 1
    assert torch.equal(captured_keys[0][..., :2, :], past_key)
    assert torch.equal(next_key[..., :2, :], past_key)
    assert torch.allclose(past_key.square().mean(dim=-1), torch.full((1, 2, 2), 4.0))

    projected_key = block.qkv(block.norm_attention(values)).chunk(3, dim=-1)[1]
    projected_key = projected_key.view(1, 1, 2, 8).transpose(1, 2)
    projected_value = block.qkv(block.norm_attention(values)).chunk(3, dim=-1)[2]
    projected_value = projected_value.view(1, 1, 2, 8).transpose(1, 2)
    expected_values = torch.cat((past_value, projected_value), dim=-2)
    torch.testing.assert_close(captured_values[0], expected_values, rtol=0.0, atol=0.0)
    normalized = headwise_rms_normalize(projected_key)
    positions = torch.tensor([3])
    expected_current = _rotate(normalized, positions, block.rotary_inverse_frequency)
    torch.testing.assert_close(next_key[..., 2:, :], expected_current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        next_key[..., 2:, :].square().mean(dim=-1),
        torch.ones((1, 2, 1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_qk_variant_has_exact_checkpoint_state_dict_roundtrip() -> None:
    torch.manual_seed(401)
    qk_model = build_sequence_model(_model_config(qk_norm="rms"))
    legacy_model = build_sequence_model(_model_config())
    assert qk_model.state_dict().keys() == legacy_model.state_dict().keys()

    checkpoint = io.BytesIO()
    torch.save({"model": qk_model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    target = build_sequence_model(_model_config(qk_norm="rms"))
    result = target.load_state_dict(restored, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert restored.keys() == qk_model.state_dict().keys()
    assert all(torch.equal(value, target.state_dict()[name]) for name, value in restored.items())
