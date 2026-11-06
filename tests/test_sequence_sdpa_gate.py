"""Receiver-only regression checks for the SDPA gate experiment; not run locally."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_optimizer import build_sequence_muon_optimizer
from wnn.training.sequence_precision import forward_context


@pytest.fixture(autouse=True)
def _precision(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("CUBLAS_EMULATE_SINGLE_PRECISION", raising=False)
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    yield
    torch.set_float32_matmul_precision(previous[0])
    torch.backends.cuda.matmul.allow_tf32 = previous[1]
    torch.backends.cudnn.allow_tf32 = previous[2]


def _model_config(**overrides: object) -> dict[str, object]:
    return {
        "architecture": "transformer",
        "hidden": 24,
        "layers": 2,
        "heads": 3,
        "ff_multiplier": 4,
        "context_rows": 5,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        **overrides,
    }


def test_omitted_gate_keeps_legacy_recipe_identity(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text("{}", encoding="utf-8")
    assert "attention_gate" not in load_config(path)["model"]


def test_gate_initialization_preserves_every_shared_parameter_and_rng() -> None:
    torch.manual_seed(61)
    baseline = build_sequence_model(_model_config())
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(61)
    gated = build_sequence_model(_model_config(attention_gate="headwise"))
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, gated.state_dict()[name], rtol=0, atol=0)
    torch.manual_seed(61)
    explicit = build_sequence_model(_model_config(attention_gate="none"))
    assert set(explicit.state_dict()) == set(baseline.state_dict())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)


def test_zero_gate_matches_half_control_and_receives_gradients() -> None:
    torch.manual_seed(33)
    half = build_sequence_model(_model_config(attention_gate="constant_half"))
    torch.manual_seed(33)
    gated = build_sequence_model(_model_config(attention_gate="headwise"))
    extra = dict(gated.named_parameters()).keys() - dict(half.named_parameters()).keys()
    assert len(extra) == 2 * len(gated.blocks)
    values = torch.randn(2, 11, 112)
    expected, _, _ = half(values)
    prediction, _, _ = gated(values)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    prediction.square().sum().backward()
    parameters = dict(gated.named_parameters())
    for name in extra:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_nonconstant_gate_is_causal_and_streaming_outputs_and_gradients_agree() -> None:
    torch.manual_seed(19)
    model = build_sequence_model(_model_config(attention_gate="headwise")).eval()
    baseline_names = set(build_sequence_model(_model_config()).state_dict())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in baseline_names:
                parameter.uniform_(-0.2, 0.2)
    values = torch.randn(2, 13, 112, requires_grad=True)
    full, _, _ = model(values)
    changed = values.detach().clone()
    changed[:, 7:] += 30
    torch.testing.assert_close(full[:, :7], model(changed)[0][:, :7])
    full.square().sum().backward()
    full_grads = {name: value.grad.clone() for name, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    state = None
    chunks = []
    for start, end in ((0, 2), (2, 8), (8, 9), (9, 13)):
        prediction, _, state = model(values.detach()[:, start:end], state)
        chunks.append(prediction)
    streamed = torch.cat(chunks, dim=1)
    torch.testing.assert_close(full.detach(), streamed)
    streamed.square().sum().backward()
    for name, value in model.named_parameters():
        torch.testing.assert_close(full_grads[name], value.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(model(values.detach())[0], full.detach())


def test_gate_optimizer_parameters_have_exact_single_ownership() -> None:
    config = _model_config(attention_gate="headwise")
    model = build_sequence_model(config)
    optimizer = build_sequence_muon_optimizer(
        model,
        {
            "model": config,
            "lr": 3e-4,
            "weight_decay": 0.01,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "muon_ns_eps": 1e-7,
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.95,
            "adamw_eps": 1e-8,
        },
    )
    names = [name for group in optimizer.param_groups for name in group["param_names"]]
    assert len(names) == len(set(names)) == len(list(model.parameters()))
    assert set(names) == set(dict(model.named_parameters()))
    baseline_names = set(build_sequence_model(_model_config()).state_dict())
    for group in optimizer.param_groups:
        for name in set(group["param_names"]) - baseline_names:
            assert group["muon"] is False
            assert group["weight_decay"] == (0.0 if name.endswith(".bias") else 0.01)


@pytest.mark.parametrize("value", [True, 1, None, "unknown"])
def test_gate_rejects_invalid_values_at_both_boundaries(value: object, tmp_path: Path) -> None:
    config = _model_config(attention_gate=value)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"model": config}), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_config(path)
    with pytest.raises((ValueError, TypeError)):
        build_sequence_model(config)


@pytest.mark.parametrize("mode", ["constant_half", "headwise"])
def test_nontrivial_gate_cannot_be_silently_ignored_on_gru(mode: str) -> None:
    with pytest.raises(ValueError):
        build_sequence_model(_model_config(architecture="gru", attention_gate=mode))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["bf16", "bf16_bf16x9"])
def test_bf16_compiled_efficient_gate_matches_eager_math_on_receiver(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires the receiving CUDA host")
    if mode == "bf16_bf16x9":
        monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    config = _model_config(attention_gate="headwise", precision_mode=mode)
    reference = build_sequence_model(config).cuda().eval()
    baseline_names = set(build_sequence_model(config | {"attention_gate": "none"}).state_dict())
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name not in baseline_names:
                parameter.uniform_(-0.2, 0.2)
    candidate = (
        build_sequence_model(
            config
            | {
                "attention_backend": "efficient",
                "compile_blocks": True,
            }
        )
        .cuda()
        .eval()
    )
    candidate.load_state_dict(reference.state_dict())
    values = torch.randn(2, 13, 112, device="cuda")
    with forward_context(mode):
        expected, _, _ = reference(values)
        actual, _, state = candidate(values)
    assert all(value.dtype == torch.float32 for value in state.values)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.03, atol=0.03)
    actual.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
