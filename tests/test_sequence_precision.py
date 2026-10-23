"""CPU tests for sequence precision scopes and identity binding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import (
    EMULATION_ENV,
    backward_context,
    checkpoint_context_fn,
    configure_precision_mode,
    forward_context,
    precision_identity,
    validate_forward_state,
    validate_process_environment,
)
from wnn.training.sequence_provenance import run_identity


@pytest.fixture(autouse=True)
def _clear_emulation_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EMULATION_ENV, raising=False)


def _small_model_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "transformer",
        "hidden": 16,
        "layers": 1,
        "heads": 2,
        "ff_multiplier": 2,
        "context_rows": 4,
        "dropout": 0.0,
        "checkpoint_blocks": True,
        "hint_dim": 0,
        "precision_mode": "fp32",
        "attention_backend": "math",
    }
    config.update(overrides)
    return config


def test_default_recipe_stays_strict_fp32(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text("{}", encoding="utf-8")
    config = load_config(path)

    assert config["precision_mode"] == "fp32"
    assert config["model"]["precision_mode"] == "fp32"
    assert config["model"]["attention_backend"] == "math"
    assert precision_identity("fp32", "math")["model_compute"] == "IEEE FP32"


@pytest.mark.parametrize("mode", ["fp32", "tf32", "bf16"])
def test_non_emulation_modes_reject_global_cublas_emulation(mode: str) -> None:
    with pytest.raises(ValueError, match=EMULATION_ENV):
        validate_process_environment(mode, {EMULATION_ENV: "1"})
    assert validate_process_environment(mode, {EMULATION_ENV: "0"}) == "0"


def test_emulation_mode_requires_explicit_process_setting() -> None:
    with pytest.raises(ValueError, match="required"):
        validate_process_environment("fp32_bf16x9", {})
    assert validate_process_environment("fp32_bf16x9", {EMULATION_ENV: "1"}) == "1"
    identity = precision_identity("fp32_bf16x9", "math", {EMULATION_ENV: "1"})
    assert "process-wide" in identity["optimizer"]
    assert identity["cublas_emulate_single_precision"] == "1"


def test_tf32_context_is_scoped_and_restores_after_exception() -> None:
    configure_precision_mode("tf32")
    with pytest.raises(RuntimeError, match="sentinel"), forward_context("tf32", "cpu"):
        validate_forward_state("tf32", "cpu")
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        raise RuntimeError("sentinel")
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    with backward_context("tf32", "cpu"):
        validate_forward_state("tf32", "cpu")


def test_bf16_checkpointed_model_keeps_fp32_parameters_grads_and_cache() -> None:
    torch.manual_seed(17)
    model = build_sequence_model(_small_model_config(precision_mode="bf16"))
    values = torch.randn(2, 5, 112, dtype=torch.float32, requires_grad=True)
    with forward_context("bf16", "cpu"):
        prediction, hints, state = model(values)
        validate_forward_state("bf16", "cpu")
    assert prediction.dtype == torch.bfloat16
    assert hints.dtype == torch.bfloat16
    assert all(key.dtype == torch.float32 for key in state.keys)
    assert all(value.dtype == torch.float32 for value in state.values)
    loss = prediction.float().square().mean() + hints.float().square().mean()
    with backward_context("bf16", "cpu"):
        loss.backward()
    assert values.grad is not None and values.grad.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in model.parameters()
    )
    first_context, recompute_context = checkpoint_context_fn("bf16", "cpu")
    assert type(first_context) is type(recompute_context)


def test_config_and_resume_identity_bind_precision_mode(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"precision_mode": "bf16"}), encoding="utf-8")
    config = load_config(path)
    assert config["model"]["precision_mode"] == "bf16"
    identity = run_identity(config, {}, {"sha256": "source", "files": {}}, {}, 1)
    assert identity["runtime"]["precision"]["mode"] == "bf16"

    path.write_text(json.dumps({"precision_mode": "tf32"}), encoding="utf-8")
    changed = load_config(path)
    changed_identity = run_identity(changed, {}, {"sha256": "source", "files": {}}, {}, 1)
    assert identity["config_sha256"] != changed_identity["config_sha256"]
    assert identity["runtime"]["precision"] != changed_identity["runtime"]["precision"]


def test_unknown_precision_and_nontransformer_attention_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"precision_mode": "fp16"}), encoding="utf-8")
    with pytest.raises(ValueError, match="precision_mode"):
        load_config(path)

    path.write_text(
        json.dumps({"model": {"architecture": "gru", "attention_backend": "efficient"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Transformer"):
        load_config(path)
