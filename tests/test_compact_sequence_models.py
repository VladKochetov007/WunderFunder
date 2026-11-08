"""Causal streaming, gradients, and frozen cohort contracts for compact models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_ema_tabm import causal_ema_scan
from wnn.training.sequence_models import build_sequence_model, detach_state


def _config(architecture: str) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": architecture,
        "hidden": 16,
        "layers": 2,
        "heads": 1,
        "ff_multiplier": 1,
        "context_rows": 1,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "precision_mode": "fp32",
        "input_feature_mode": "raw112",
    }
    if architecture == "gru_local":
        config.update(local_hidden=16, local_layers=2, heads=2, context_rows=5)
    elif architecture == "ema_tabm_style":
        config.update(ema_projection=4, ensemble_members=4, numerical_embedding_dim=2)
    elif architecture == "mamba2":
        config.update(ssm_head_dim=8, ssm_state_size=16, ssm_chunk_size=16)
    return config


@pytest.mark.parametrize("architecture", ["gru", "gru_local", "ema_tabm_style"])
def test_compact_causal_full_chunk_step_reset_and_gradients(architecture: str) -> None:
    configure_strict_fp32()
    torch.manual_seed(123)
    model = build_sequence_model(_config(architecture)).eval()
    # Nontrivial fitting-only statistics exercise normalization in both branches.
    model.mean.copy_(torch.randn(112))
    model.std.copy_(torch.rand(112) + 0.5)
    values = torch.randn(2, 17, 112)
    full, full_hints, _ = model(values)
    changed = values.clone()
    changed[:, 9:] += 100
    altered, _, _ = model(changed)
    torch.testing.assert_close(full[:, :9], altered[:, :9], atol=2e-6, rtol=2e-5)
    for boundaries in ((0, 1, 6, 8, 17), tuple(range(18))):
        predictions: list[Tensor] = []
        state = None
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            prediction, hints, state = model(values[:, start:end], state)
            predictions.append(prediction)
            assert hints.shape == (2, end - start, 0)
        torch.testing.assert_close(torch.cat(predictions, 1), full, atol=2e-6, rtol=2e-5)
    reset, _, _ = model(values)
    torch.testing.assert_close(reset, full)
    assert full_hints.shape == (2, 17, 0)
    _, _, prefix_state = model(values[:, :6])
    detached = detach_state(prefix_state)
    if architecture == "gru":
        tensors = [detached]
    elif architecture == "gru_local":
        tensors = [detached.recurrent, *detached.attention.keys, *detached.attention.values]
    else:
        tensors = [detached.means, detached.previous]
    assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in tensors)
    model.zero_grad(set_to_none=True)
    suffix, _, _ = model(values[:, 6:], detached)
    suffix.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("architecture", ["gru", "gru_local", "ema_tabm_style"])
def test_compact_chunks_preserve_attached_state_gradients(architecture: str) -> None:
    configure_strict_fp32()
    torch.manual_seed(77)
    model = build_sequence_model(_config(architecture)).eval()
    values = torch.randn(2, 9, 112)
    full, _, _ = model(values)
    full.square().sum().backward()
    reference = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    first, _, state = model(values[:, :4])
    second, _, _ = model(values[:, 4:], state)
    torch.cat((first, second), 1).square().sum().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference[name], atol=3e-5, rtol=3e-4)


def test_ema_associative_scan_matches_row_recurrence_and_initial_state_gradients() -> None:
    torch.manual_seed(73)
    values = torch.randn(2, 129, 3, requires_grad=True)
    decay = torch.tensor([0.0, 0.5, 0.99, 0.9999])
    initial = torch.randn(2, 4, 3, requires_grad=True)
    actual, final = causal_ema_scan(values, decay, initial)
    expected: list[Tensor] = []
    state = initial
    for row in values.unbind(1):
        state = decay[None, :, None] * state + (1 - decay)[None, :, None] * row[:, None]
        expected.append(state)
    torch.testing.assert_close(actual, torch.stack(expected, 1), atol=4e-6, rtol=2e-5)
    torch.testing.assert_close(final, state, atol=4e-6, rtol=2e-5)
    final.square().sum().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_eight_compact_recipes_share_exact_frozen_training_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        (root / "research/experiment_drafts/architecture-execution-20261107/l4/configs").glob(
            "*.json"
        )
    )
    assert len(paths) == 8
    cells: set[tuple[str, int]] = set()
    policies = []
    for path in paths:
        config = load_config(path)
        assert config["epochs"] == 12
        assert config["normalization_sequences"] == 10325
        assert config["normalization_scope"] == "all_fit"
        assert config["training_mask"] == "all_after_warmup"
        assert config["kd_weight"] == config["hint_weight"] == 0
        assert config["model"]["input_feature_mode"] == "raw112"
        cells.add((config["model"]["architecture"], config["seed"]))
        policies.append(
            {key: value for key, value in config.items() if key not in ("model", "seed")}
        )
    assert cells == {
        (family, seed)
        for family in ("gru", "gru_local", "mamba2", "ema_tabm_style")
        for seed in (101001, 101002)
    }
    assert all(policy == policies[0] for policy in policies)


@pytest.mark.parametrize("architecture", ["gru_local", "mamba2", "ema_tabm_style"])
def test_compact_recipe_rejects_kd_and_reduced_precision(tmp_path: Path, architecture: str) -> None:
    for override in ({"kd_weight": 0.5}, {"precision_mode": "bf16"}):
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps({"model": _config(architecture)} | override))
        with pytest.raises(ValueError):
            load_config(path)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real pinned CUDA Mamba-2")
def test_mamba2_real_backend_oracle_causality_chunk_reset_and_gradients() -> None:
    configure_strict_fp32()
    torch.manual_seed(123)
    model = build_sequence_model(_config("mamba2")).cuda().eval()
    values = torch.randn(2, 37, 112, device="cuda")
    full, _, full_state = model(values)
    predictions = []
    state = None
    for start, end in ((0, 1), (1, 7), (7, 20), (20, 37)):
        prediction, _, state = model(values[:, start:end], state)
        predictions.append(prediction)
    torch.testing.assert_close(torch.cat(predictions, 1), full, atol=1e-4, rtol=2e-4)
    changed = values.clone()
    changed[:, 20:] += 10
    altered, _, _ = model(changed)
    torch.testing.assert_close(full[:, :20], altered[:, :20], atol=1e-4, rtol=2e-4)
    reset, _, _ = model(values)
    torch.testing.assert_close(reset, full)
    # Independent official full-forward oracle for every genuine Mamba-2 layer.
    encoded = model.project((values - model.mean) / model.std)
    for block in model.layers:
        expected = encoded + block.mixer(block.norm(encoded))
        encoded, _ = block(encoded, None)
        torch.testing.assert_close(encoded, expected, atol=1e-4, rtol=2e-4)
    model.zero_grad(set_to_none=True)
    detached = detach_state(full_state)
    suffix, _, _ = model(values[:, :5], detached)
    suffix.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
