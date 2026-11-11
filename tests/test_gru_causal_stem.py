from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from wnn.training.sequence_config import load_config
from wnn.training.sequence_gru import GRUSequenceAdapter
from wnn.training.sequence_model_helpers import GRUSequenceState, detach_state
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_predict import predict_sequence


@pytest.fixture(autouse=True)
def _single_thread_strict_fp32() -> Iterator[None]:
    """Keep this synthetic CPU contract at one thread with strict FP32 settings."""
    thread_count = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(thread_count)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _model_config(kernel: int | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "gru",
        "hidden": 8,
        "layers": 1,
        "heads": 1,
        "ff_multiplier": 2,
        "context_rows": 8,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "precision_mode": "fp32",
    }
    if kernel is not None:
        config["input_stem_kernel"] = kernel
    return config


def _set_nonzero_filter(model: GRUSequenceAdapter) -> None:
    assert model.input_stem is not None
    with torch.no_grad():
        model.input_stem.weight.zero_()
        model.input_stem.weight[:, 0, 0] = torch.linspace(0.01, 0.03, 112)
        model.input_stem.weight[:, 0, 7] = -0.02
        model.input_stem.weight[:, 0, -1] = 0.015


def test_plain_config_and_state_dict_keep_legacy_identity(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    explicit_zero = tmp_path / "explicit-zero.json"
    legacy.write_text(json.dumps({"model": {"architecture": "gru"}}), encoding="utf-8")
    explicit_zero.write_text(
        json.dumps({"model": {"architecture": "gru", "input_stem_kernel": 0}}),
        encoding="utf-8",
    )
    legacy_config = load_config(legacy)
    zero_config = load_config(explicit_zero)
    assert legacy_config == zero_config
    assert "input_stem_kernel" not in legacy_config["model"]

    torch.manual_seed(1001)
    legacy_model = build_sequence_model(_model_config())
    legacy_rng = torch.random.get_rng_state()
    torch.manual_seed(1001)
    zero_model = build_sequence_model(_model_config(0))
    assert torch.equal(legacy_rng, torch.random.get_rng_state())
    assert legacy_model.state_dict().keys() == zero_model.state_dict().keys()
    for key, value in legacy_model.state_dict().items():
        torch.testing.assert_close(value, zero_model.state_dict()[key], rtol=0, atol=0)
    assert zero_model.input_stem is None


@pytest.mark.parametrize("kernel", [1, 16, 64])
def test_stem_init_preserves_backbone_and_global_rng(kernel: int) -> None:
    torch.manual_seed(203301)
    plain = build_sequence_model(_model_config(0))
    plain_rng = torch.random.get_rng_state()
    torch.manual_seed(203301)
    stem = build_sequence_model(_model_config(kernel))
    assert torch.equal(plain_rng, torch.random.get_rng_state())
    assert isinstance(stem, GRUSequenceAdapter) and stem.input_stem is not None
    assert torch.count_nonzero(stem.input_stem.weight) == 0
    plain_state, stem_state = plain.state_dict(), stem.state_dict()
    assert set(stem_state) - set(plain_state) == {"input_stem.weight"}
    for key in plain_state:
        torch.testing.assert_close(plain_state[key], stem_state[key], rtol=0, atol=0)
    values = torch.randn(2, 7, 112)
    torch.testing.assert_close(plain(values)[0], stem(values)[0], rtol=0, atol=0)


def test_kernel16_chunk_carry_matches_whole_sequence_and_stays_bounded() -> None:
    torch.manual_seed(44)
    model = build_sequence_model(_model_config(16))
    assert isinstance(model, GRUSequenceAdapter)
    model.eval()
    _set_nonzero_filter(model)
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.5, 0.5, 112))
        model.std.copy_(torch.linspace(0.5, 1.5, 112))
    values = torch.randn(2, 37, 112)
    full_prediction, _, full_state = model(values)
    assert isinstance(full_state, GRUSequenceState)

    state = None
    output: list[Tensor] = []
    for start, stop in ((0, 2), (2, 3), (3, 11), (11, 16), (16, 37)):
        prior = state
        snapshot = None if prior is None else prior.input_history.clone()
        prediction, _, next_state = model(values[:, start:stop], state)
        assert isinstance(next_state, GRUSequenceState)
        expected = ((values[:, :stop] - model.mean) / model.std)[:, -15:]
        torch.testing.assert_close(next_state.input_history, expected, rtol=0, atol=0)
        assert next_state.input_history.numel() <= 2 * 15 * 112
        if prior is not None:
            torch.testing.assert_close(prior.input_history, snapshot, rtol=0, atol=0)
        state = detach_state(next_state)
        assert isinstance(state, GRUSequenceState)
        assert state.next_position == stop
        output.append(prediction)
    torch.testing.assert_close(torch.cat(output, dim=1), full_prediction, rtol=1e-5, atol=1e-6)
    assert isinstance(state, GRUSequenceState)
    torch.testing.assert_close(state.hidden, full_state.hidden, rtol=1e-5, atol=1e-6)


def test_kernel64_causality_chunk_reset_tbptt_and_reload(tmp_path: Path) -> None:
    torch.manual_seed(464)
    model = build_sequence_model(_model_config(64))
    assert isinstance(model, GRUSequenceAdapter)
    model.eval()
    _set_nonzero_filter(model)
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.5, 0.5, 112))
        model.std.copy_(torch.linspace(0.5, 1.5, 112))
    values = torch.randn(2, 101, 112)
    full_prediction = model(values)[0]
    altered = values.clone()
    altered[:, 39:] += 100.0
    torch.testing.assert_close(full_prediction[:, :39], model(altered)[0][:, :39], rtol=0, atol=0)

    state = None
    output: list[Tensor] = []
    for start, stop in ((0, 1), (1, 31), (31, 32), (32, 71), (71, 101)):
        prior = state
        snapshot = None if prior is None else prior.input_history.clone()
        prediction, _, next_state = model(values[:, start:stop], state)
        assert isinstance(next_state, GRUSequenceState)
        expected = ((values[:, :stop] - model.mean) / model.std)[:, -63:]
        torch.testing.assert_close(next_state.input_history, expected, rtol=0, atol=0)
        assert next_state.input_history.numel() <= 2 * 63 * 112
        if prior is not None:
            torch.testing.assert_close(prior.input_history, snapshot, rtol=0, atol=0)
        state = detach_state(next_state)
        assert isinstance(state, GRUSequenceState)
        assert state.next_position == stop
        output.append(prediction)
    torch.testing.assert_close(torch.cat(output, dim=1), full_prediction, rtol=1e-5, atol=1e-6)

    first = predict_sequence(model, values[0], 17, torch.device("cpu"))
    second_after_first = predict_sequence(model, values[1], 17, torch.device("cpu"))
    second_fresh = predict_sequence(model, values[1], 17, torch.device("cpu"))
    np.testing.assert_array_equal(second_after_first, second_fresh)
    assert first.shape == second_after_first.shape == (101, 2)

    checkpoint = tmp_path / "kernel64.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = build_sequence_model(_model_config(64))
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    reloaded.eval()
    torch.testing.assert_close(model(values)[0], reloaded(values)[0], rtol=0, atol=0)
    assert isinstance(state, GRUSequenceState)
    bad_history = replace(state, input_history=torch.zeros(2, 62, 112))
    with pytest.raises(ValueError, match="row count"):
        model(torch.randn(2, 1, 112), bad_history)
    bad_position = replace(state, next_position=True)
    with pytest.raises(ValueError, match="absolute position"):
        model(torch.randn(2, 1, 112), bad_position)


def test_gru_interlayer_dropout_train_eval_and_config_roundtrip(tmp_path: Path) -> None:
    raw = {
        "model": {
            "architecture": "gru",
            "hidden": 8,
            "layers": 2,
            "input_stem_kernel": 16,
            "dropout": 0.1,
        }
    }
    source = tmp_path / "dropout.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(source)
    assert config["model"]["dropout"] == 0.1
    normalized = tmp_path / "dropout-normalized.json"
    serialized = json.loads(json.dumps(config))
    serialized["model"].pop("precision_mode")  # runtime-only, derived from the top-level recipe
    normalized.write_text(json.dumps(serialized), encoding="utf-8")
    assert load_config(normalized) == config

    model = build_sequence_model(config["model"])
    assert isinstance(model, GRUSequenceAdapter)
    assert model.gru.dropout == 0.1
    values = torch.randn(2, 23, 112)
    model.train()
    first = model(values)[0]
    second = model(values)[0]
    assert not torch.equal(first, second)
    model.eval()
    expected = model(values)[0]
    torch.testing.assert_close(expected, model(values)[0], rtol=0, atol=0)

    transformer = tmp_path / "transformer-dropout.json"
    transformer.write_text(
        json.dumps({"model": {"architecture": "transformer", "dropout": 0.1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-GRU dropout=0"):
        load_config(transformer)
    one_layer = tmp_path / "one-layer-dropout.json"
    one_layer.write_text(
        json.dumps({"model": {"architecture": "gru", "layers": 1, "dropout": 0.1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two recurrent layers"):
        load_config(one_layer)


def test_kernel1_streams_with_legacy_tensor_state() -> None:
    torch.manual_seed(441)
    model = build_sequence_model(_model_config(1)).eval()
    assert isinstance(model, GRUSequenceAdapter) and model.input_stem is not None
    with torch.no_grad():
        model.input_stem.weight.fill_(0.03)
    values = torch.randn(2, 19, 112)
    expected, _, full_state = model(values)
    state = None
    chunks: list[Tensor] = []
    for start, stop in ((0, 1), (1, 8), (8, 19)):
        prediction, _, state = model(values[:, start:stop], state)
        assert isinstance(state, Tensor)
        chunks.append(prediction)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state, full_state, rtol=1e-5, atol=1e-6)


def test_kernel16_future_causality_batch_independence_and_predict_reset() -> None:
    torch.manual_seed(45)
    model = build_sequence_model(_model_config(16)).eval()
    assert isinstance(model, GRUSequenceAdapter)
    _set_nonzero_filter(model)
    values = torch.randn(3, 27, 112)
    prediction = model(values)[0]
    altered = values.clone()
    altered[:, 12:] += 100.0
    torch.testing.assert_close(prediction[:, :12], model(altered)[0][:, :12], rtol=0, atol=0)
    individual = torch.cat([model(values[index : index + 1])[0] for index in range(3)])
    torch.testing.assert_close(prediction, individual, rtol=1e-5, atol=1e-6)

    device = torch.device("cpu")
    first = predict_sequence(model, values[0], 5, device)
    second_after_first = predict_sequence(model, values[1], 4, device)
    second_fresh = predict_sequence(model, values[1], 4, device)
    np.testing.assert_array_equal(second_after_first, second_fresh)
    assert first.shape == second_after_first.shape == (27, 2)


def test_kernel16_tbptt_detaches_hidden_and_history_and_reloads(tmp_path: Path) -> None:
    torch.manual_seed(46)
    model = build_sequence_model(_model_config(16))
    assert isinstance(model, GRUSequenceAdapter)
    _set_nonzero_filter(model)
    first = torch.randn(1, 8, 112, requires_grad=True)
    _, _, state = model(first)
    assert isinstance(state, GRUSequenceState)
    assert state.hidden.grad_fn is not None and state.input_history.grad_fn is not None
    detached = detach_state(state)
    assert isinstance(detached, GRUSequenceState)
    assert detached.hidden.grad_fn is None and detached.input_history.grad_fn is None
    prediction, _, _ = model(torch.randn(1, 5, 112), detached)
    prediction.square().mean().backward()
    assert first.grad is None
    assert model.input_stem is not None and model.input_stem.weight.grad is not None
    assert torch.isfinite(model.input_stem.weight.grad).all()

    checkpoint = tmp_path / "stem.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = build_sequence_model(_model_config(16))
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    reloaded.eval()
    values = torch.randn(1, 9, 112)
    torch.testing.assert_close(model.eval()(values)[0], reloaded(values)[0], rtol=0, atol=0)

    bad_history = replace(detached, input_history=torch.zeros(1, 14, 112))
    with pytest.raises(ValueError, match="row count"):
        model(torch.randn(1, 1, 112), bad_history)
    bad_position = replace(detached, next_position=True)
    with pytest.raises(ValueError, match="absolute position"):
        model(torch.randn(1, 1, 112), bad_position)


@pytest.mark.parametrize("kernel", [True, 1.0, -1, 15])
def test_invalid_kernel_is_rejected(kernel: object, tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"model": {"architecture": "gru", "input_stem_kernel": kernel}}),
        encoding="utf-8",
    )
    with pytest.raises((TypeError, ValueError), match="input_stem_kernel"):
        load_config(path)


@pytest.mark.parametrize("kernel", [16, 64])
def test_nonzero_stem_kernel_survives_config_normalization(tmp_path: Path, kernel: int) -> None:
    path = tmp_path / f"kernel{kernel}.json"
    path.write_text(
        json.dumps({"model": {"architecture": "gru", "input_stem_kernel": kernel}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["model"]["input_stem_kernel"] == kernel
    assert isinstance(build_sequence_model(config["model"]), GRUSequenceAdapter)


def test_stems_are_rejected_for_transformer_models(tmp_path: Path) -> None:
    path = tmp_path / "transformer.json"
    path.write_text(
        json.dumps({"model": {"architecture": "transformer", "input_stem_kernel": 16}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only for GRU"):
        load_config(path)
