"""Bounded CPU checks for the prospective GRU and GALAM sequence adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from wnn.features import N_EVENT_FEATURES, event_features
from wnn.galam import connectome_112_to_2_config
from wnn.nn import RNNModel
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_features import (
    EVENT_INPUT_MODE,
    EVENT_INPUT_SCHEMA,
    RAW_INPUT_MODE,
    RAW_INPUT_SCHEMA,
    numpy_sequence_reader,
    sequence_input_schema,
    sequence_input_width,
    sequence_inputs_from_raw,
)
from wnn.training.sequence_fit_utils import optimizer_states_are_finite_fp32
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_normalization import (
    build_normalization_identity,
    compute_population_moments,
)
from wnn.training.sequence_optimizer import build_sequence_muon_optimizer


@pytest.fixture(autouse=True)
def _strict_fp32_environment() -> Iterator[None]:
    """Use strict-FP32 settings while restoring process-global precision flags."""
    matmul_precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    configure_strict_fp32()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _model_config(**overrides: object) -> dict[str, object]:
    """Return a small canonical GRU configuration for CPU tests."""
    return {
        "architecture": "gru",
        "input_feature_mode": EVENT_INPUT_MODE,
        "hidden": 8,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 2,
        "context_rows": 8,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "hint_projection": "learned",
        "precision_mode": "fp32",
        "attention_backend": "math",
        "compile_blocks": False,
        **overrides,
    }


def _small_galam_config() -> dict[str, object]:
    """Return a small full-LSTM GALAM for bounded optimizer tests."""
    galam = replace(
        connectome_112_to_2_config(),
        width=8,
        attention_heads=2,
        attention_history=4,
        conv_kernel_size=3,
        conv_hidden_width=8,
        relative_bias_width=4,
        feed_forward_width=16,
    )
    return {"architecture": "galam", "galam": asdict(galam) | {"variant": "schematic_lstm_full"}}


def _json_default(value: object) -> list[str]:
    """Serialize the frozen fidelity sets used by GALAM configuration records."""
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _optimizer_config(model: dict[str, object]) -> dict[str, object]:
    """Return the common strict-FP32 optimizer contract used by adapter tests."""
    return {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "muon_momentum": 0.95,
        "muon_ns_steps": 2,
        "muon_ns_eps": 1e-7,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.95,
        "adamw_eps": 1e-8,
        "model": model,
    }


def _write_recipe(path: Path, raw: dict[str, object]) -> dict[str, object]:
    """Write and load a JSON recipe through the production validator."""
    path.write_text(json.dumps(raw, default=_json_default), encoding="utf-8")
    return load_config(path)


def test_event211_assembly_matches_shared_numpy_and_torch_implementations() -> None:
    """Appending causal events is deterministic and leaves the raw path untouched."""
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(17, 112)).astype(np.float32)
    expected = np.concatenate((raw, event_features(raw)), axis=-1)
    assembled_numpy = sequence_inputs_from_raw(raw, EVENT_INPUT_MODE)
    assert isinstance(assembled_numpy, np.ndarray)
    np.testing.assert_array_equal(assembled_numpy, expected)

    assembled_torch = sequence_inputs_from_raw(torch.from_numpy(raw), EVENT_INPUT_MODE)
    assert isinstance(assembled_torch, Tensor)
    assert assembled_torch.shape == (17, sequence_input_width(EVENT_INPUT_MODE))
    torch.testing.assert_close(assembled_torch, torch.from_numpy(expected), rtol=0, atol=1e-6)
    assert sequence_input_width(EVENT_INPUT_MODE) == 112 + N_EVENT_FEATURES == 211

    unchanged = sequence_inputs_from_raw(raw, RAW_INPUT_MODE)
    assert unchanged is raw
    raw_tensor = torch.from_numpy(raw)
    assert sequence_inputs_from_raw(raw_tensor, RAW_INPUT_MODE) is raw_tensor
    assert sequence_input_schema(RAW_INPUT_MODE) == RAW_INPUT_SCHEMA
    assert sequence_input_schema(EVENT_INPUT_MODE) == EVENT_INPUT_SCHEMA


def test_event211_causality_reset_and_full_sequence_chunk_assembly() -> None:
    """Suffix changes cannot alter earlier features and each sequence starts a fresh event clock."""
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(19, 112)).astype(np.float32)
    full = sequence_inputs_from_raw(raw, EVENT_INPUT_MODE)
    altered = raw.copy()
    altered[9:] += 17.0
    altered_features = sequence_inputs_from_raw(altered, EVENT_INPUT_MODE)
    assert isinstance(full, np.ndarray) and isinstance(altered_features, np.ndarray)
    np.testing.assert_array_equal(full[:9], altered_features[:9])

    chunks = [full[:4], full[4:13], full[13:]]
    np.testing.assert_array_equal(np.concatenate(chunks), full)
    second_sequence = sequence_inputs_from_raw(raw[7:], EVENT_INPUT_MODE)
    assert isinstance(second_sequence, np.ndarray)
    np.testing.assert_array_equal(second_sequence[0, 112:], np.zeros(N_EVENT_FEATURES, np.float32))
    np.testing.assert_array_equal(full[0, 112:], np.zeros(N_EVENT_FEATURES, np.float32))


def test_event211_normalization_uses_only_transformed_fitting_members() -> None:
    """The 211-column moments are computed from train members and schema-bound."""
    rng = np.random.default_rng(29)
    fitting = {
        ("train", 3): rng.normal(size=(11, 112)).astype(np.float32),
        ("train", 8): rng.normal(loc=2.0, size=(7, 112)).astype(np.float32),
    }
    valid = rng.normal(loc=1000.0, size=(9, 112)).astype(np.float32)

    def raw_reader(source: str, group: int) -> np.ndarray:
        """Reject validation access while returning one fitting source sequence."""
        assert source == "train"
        return fitting[(source, group)]

    members = list(fitting)
    transformed = numpy_sequence_reader(raw_reader, EVENT_INPUT_MODE)
    mean, std = compute_population_moments(
        members,
        transformed,
        feature_count=sequence_input_width(EVENT_INPUT_MODE),
    )
    expected_rows = np.concatenate(
        [sequence_inputs_from_raw(fitting[member], EVENT_INPUT_MODE) for member in members]
    )
    assert isinstance(expected_rows, np.ndarray)
    np.testing.assert_allclose(mean, expected_rows.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std, expected_rows.std(axis=0).clip(0.001), rtol=1e-5, atol=1e-6)
    assert valid.shape == (9, 112)  # held-out fixture is intentionally never given to the reader

    sources = {"datasets/train.parquet": {"sha256": "train-fingerprint"}}
    identity = build_normalization_identity(
        sources,
        "a" * 64,
        members,
        "all_fit",
        211,
        input_schema=EVENT_INPUT_SCHEMA,
    )
    assert identity["input_schema"] == EVENT_INPUT_SCHEMA
    assert (
        build_normalization_identity(sources, "a" * 64, members, "all_fit", 112).get("input_schema")
        is None
    )


def test_event211_gru_matches_canonical_rnn_and_keeps_resumable_state() -> None:
    """The configured GRU remains the canonical RNNModel with a wider causal input."""
    torch.manual_seed(31)
    config = _model_config()
    model = build_sequence_model(config).eval()
    reference = RNNModel(211, hidden=8, layers=2, dropout=0.0, n_out=2, cell="gru", enc="linear")
    reference.load_state_dict(model.state_dict(), strict=True)
    reference.eval()
    values = torch.randn(2, 13, 211)

    expected, expected_state = reference(values)
    actual, hints, state = model(values)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state)
    assert hints.shape == (2, 13, 0)

    first, _, carried = model(values[:, :5])
    rest, _, final_state = model(values[:, 5:], carried)
    torch.testing.assert_close(torch.cat((first, rest), dim=1), actual)
    torch.testing.assert_close(final_state, state)


def test_sequence_config_accepts_only_declared_launch_adapters(tmp_path: Path) -> None:
    """The new Muon and monitor combinations are explicit while defaults stay unchanged."""
    event_model = {
        "architecture": "gru",
        "input_feature_mode": EVENT_INPUT_MODE,
        "hidden": 256,
        "layers": 2,
    }
    monitor = {
        "train_eval_sequences": 32,
        "diagnostics_every_steps": 25,
        "train_eval_mask": "all_after_warmup",
        "eval_batch_size": 16,
        "training_telemetry_every_steps": 10,
        "early_stopping": {"min_epoch": 5, "patience": 4, "min_delta": 0.0005},
    }
    recipe = {
        "epochs": 20,
        "batch_size": 32,
        "chunk_rows": 1000,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "optimizer": "muon",
        "schedule": "warmup_cosine_v2",
        "training_mask": "all_after_warmup",
        "normalization_scope": "all_fit",
        "checkpoint_retention": "all_epochs",
        "model": event_model,
        "monitoring": monitor,
    }
    config = _write_recipe(tmp_path / "event-gru.json", recipe)
    assert config["optimizer"] == "muon"
    assert config["model"]["input_feature_mode"] == EVENT_INPUT_MODE
    assert config["monitoring"] == monitor

    raw_gru = recipe | {"model": event_model | {"input_feature_mode": RAW_INPUT_MODE}}
    with pytest.raises(ValueError, match="Transformer models only"):
        _write_recipe(tmp_path / "raw-gru-muon.json", raw_gru)
    transformer = recipe | {
        "model": {"architecture": "transformer", "input_feature_mode": EVENT_INPUT_MODE}
    }
    transformer.pop("monitoring")
    transformer_config = _write_recipe(tmp_path / "event-transformer.json", transformer)
    assert transformer_config["model"]["input_feature_mode"] == EVENT_INPUT_MODE

    unsupported_lstm = transformer | {
        "model": {"architecture": "lstm", "input_feature_mode": EVENT_INPUT_MODE}
    }
    with pytest.raises(ValueError, match="only by GRU and Transformer adapters"):
        _write_recipe(tmp_path / "lstm-event211.json", unsupported_lstm)

    unsupported_gru_width = recipe | {
        "model": event_model | {"hidden": 128, "input_feature_mode": EVENT_INPUT_MODE}
    }
    with pytest.raises(ValueError, match="scoped to the standalone GRU256x2 launch"):
        _write_recipe(tmp_path / "gru128-event211.json", unsupported_gru_width)

    unsupported_input = transformer | {
        "model": {"architecture": "transformer", "input_feature_mode": "event212"}
    }
    with pytest.raises(ValueError, match="model.input_feature_mode must be raw112 or event211"):
        _write_recipe(tmp_path / "transformer-event212.json", unsupported_input)

    full_galam = asdict(connectome_112_to_2_config()) | {"variant": "schematic_lstm_full"}
    galam_recipe = {
        "batch_size": 2,
        "optimizer": "muon",
        "model": {"architecture": "galam", "galam": full_galam},
    }
    loaded_galam = _write_recipe(tmp_path / "galam-hybrid.json", galam_recipe)
    assert loaded_galam["model"].get("input_feature_mode", RAW_INPUT_MODE) == RAW_INPUT_MODE
    full_galam["width"] = 128
    with pytest.raises(ValueError, match="LSTM64 hybrid"):
        _write_recipe(
            tmp_path / "galam-width-muon.json",
            galam_recipe | {"model": {"architecture": "galam", "galam": full_galam}},
        )


def test_gru_muon_owns_only_gate_matrices_and_resumes_strict_fp32_state() -> None:
    """Packed gate Muon and AdamW groups cover every trainable parameter and resume exactly."""
    torch.manual_seed(37)
    model_config = _model_config()
    model = build_sequence_model(model_config).train()
    optimizer_config = _optimizer_config(model_config)
    optimizer = build_sequence_muon_optimizer(model, optimizer_config)
    by_role = {str(group["role"]): group for group in optimizer.param_groups}
    assert set(by_role) == {"muon_gru_gates", "adamw_other"}
    assert by_role["muon_gru_gates"]["blocks"] == 3
    assert all(
        name.startswith("gru.weight_ih_l") or name.startswith("gru.weight_hh_l")
        for name in by_role["muon_gru_gates"]["param_names"]
    )
    assert "inp.weight" in by_role["adamw_other"]["param_names"]
    assert any(name.endswith(".bias") for name in by_role["adamw_other"]["param_names"])
    assert by_role["adamw_other"]["weight_decay_enabled"] is True
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in owned} == {id(parameter) for parameter in trainable}

    values = torch.randn(2, 7, 211)
    prediction, _, _ = model(values)
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in trainable)
    optimizer.step()
    assert optimizer_states_are_finite_fp32(optimizer)

    resumed_model = build_sequence_model(model_config).train()
    resumed_model.load_state_dict(model.state_dict())
    resumed_optimizer = build_sequence_muon_optimizer(resumed_model, optimizer_config)
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    optimizer.zero_grad(set_to_none=True)
    resumed_optimizer.zero_grad(set_to_none=True)
    next_prediction, _, _ = model(values)
    resumed_prediction, _, _ = resumed_model(values)
    torch.testing.assert_close(resumed_prediction, next_prediction)
    next_prediction.square().mean().backward()
    resumed_prediction.square().mean().backward()
    optimizer.step()
    resumed_optimizer.step()
    for left, right in zip(model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    assert optimizer_states_are_finite_fp32(resumed_optimizer)


def test_galam_muon_excludes_lstm_and_covers_attention_ffn_with_adamw_elsewhere() -> None:
    """The full GALAM hybrid applies Muon to exactly attention/FFN matrices."""
    torch.manual_seed(41)
    model_config = _small_galam_config()
    model = build_sequence_model(model_config).train()
    optimizer_config = _optimizer_config(model_config)
    optimizer = build_sequence_muon_optimizer(model, optimizer_config)
    by_role = {str(group["role"]): group for group in optimizer.param_groups}
    assert set(by_role) == {
        "muon_attention",
        "muon_feedforward",
        "adamw_other",
    }
    muon_names = [
        name
        for role in ("muon_attention", "muon_feedforward")
        for name in by_role[role]["param_names"]
    ]
    assert any("local_attention" in name for name in muon_names)
    assert any("global_attention" in name for name in muon_names)
    assert any("feed_forward" in name for name in muon_names)
    assert not any("lstm" in name for name in muon_names)
    adamw_names = by_role["adamw_other"]["param_names"]
    lstm_names = [name for name in adamw_names if name.startswith("galam.lstm.")]
    assert lstm_names
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in owned} == {id(parameter) for parameter in trainable}

    prediction, _, _ = model(torch.randn(2, 4, 112))
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in trainable)
    optimizer.step()
    assert optimizer_states_are_finite_fp32(optimizer)


@pytest.mark.parametrize(
    "model_config", [_model_config(), _small_galam_config()], ids=["gru", "galam"]
)
def test_adapter_non_muon_update_matches_plain_adamw(model_config: dict[str, object]) -> None:
    """Every non-Muon parameter keeps the plain AdamW decay and moment update exactly."""
    torch.manual_seed(53)
    hybrid_model = build_sequence_model(model_config)
    control_model = build_sequence_model(model_config)
    control_model.load_state_dict(hybrid_model.state_dict(), strict=True)
    optimizer_config = _optimizer_config(model_config)
    hybrid = build_sequence_muon_optimizer(hybrid_model, optimizer_config)
    control = torch.optim.AdamW(
        control_model.parameters(),
        lr=float(optimizer_config["lr"]),
        betas=(
            float(optimizer_config["adamw_beta1"]),
            float(optimizer_config["adamw_beta2"]),
        ),
        eps=float(optimizer_config["adamw_eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    muon_names = {
        name for group in hybrid.param_groups if group["muon"] for name in group["param_names"]
    }
    expected_non_muon = {
        name for name, parameter in hybrid_model.named_parameters() if parameter.requires_grad
    } - muon_names
    assert expected_non_muon
    for name, parameter in hybrid_model.named_parameters():
        gradient = torch.full_like(parameter, 0.01)
        parameter.grad = gradient.clone()
        control_model.get_parameter(name).grad = gradient.clone()

    hybrid.step()
    control.step()
    for name in sorted(expected_non_muon):
        torch.testing.assert_close(
            hybrid_model.get_parameter(name),
            control_model.get_parameter(name),
            rtol=1e-6,
            atol=1e-7,
        )
