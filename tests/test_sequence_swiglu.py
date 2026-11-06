"""Receiver-only checks for matched-budget SwiGLU; not executed on the author's Mac."""

from __future__ import annotations

import io
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


def _config(**overrides: object) -> dict[str, object]:
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


def test_omitted_ff_options_preserve_recipe_and_legacy_initialization(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text("{}", encoding="utf-8")
    config = load_config(path)
    assert "ff_activation" not in config["model"]
    assert "ff_hidden" not in config["model"]
    torch.manual_seed(31)
    original = build_sequence_model(_config())
    original_rng = torch.random.get_rng_state()
    torch.manual_seed(31)
    explicit = build_sequence_model(_config(ff_activation="gelu", ff_hidden=96))
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert set(original.state_dict()) == set(explicit.state_dict())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)


def test_budget_matches_matrix_parameters_and_optimizer_owns_swiglu_gate() -> None:
    gelu = build_sequence_model(_config(ff_activation="gelu", ff_hidden=96))
    swiglu_config = _config(ff_activation="swiglu", ff_hidden=64)
    swiglu = build_sequence_model(swiglu_config)
    matrix_counts = [
        sum(
            p.numel()
            for name, p in model.named_parameters()
            if "feedforward_" in name and p.ndim == 2
        )
        for model in (gelu, swiglu)
    ]
    assert matrix_counts[0] == matrix_counts[1]
    optimizer = build_sequence_muon_optimizer(
        swiglu,
        {
            "model": swiglu_config,
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
    assert len(names) == len(set(names)) == len(list(swiglu.parameters()))
    assert set(names) == set(dict(swiglu.named_parameters()))
    gate_names = {name for name in names if name.endswith("feedforward_gate.weight")}
    assert len(gate_names) == 2
    for group in optimizer.param_groups:
        if gate_names.intersection(group["param_names"]):
            assert group["muon"] is True and group["blocks"] == 1


def test_direct_legacy_model_builder_preserves_fractional_ff_multiplier() -> None:
    model = build_sequence_model(_config(ff_multiplier=2.5))
    assert all(block.feedforward_in.out_features == 60 for block in model.blocks)


def test_swiglu_causality_streaming_gradient_and_checkpoint_roundtrip() -> None:
    torch.manual_seed(19)
    config = _config(ff_activation="swiglu", ff_hidden=64, checkpoint_blocks=True)
    model = build_sequence_model(config).train()
    values = torch.randn(2, 13, 112, requires_grad=True)
    full, _, _ = model(values)
    changed = values.detach().clone()
    changed[:, 7:] -= 30
    torch.testing.assert_close(full[:, :7], model(changed)[0][:, :7])
    full.square().sum().backward()
    full_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    state = None
    chunks = []
    for start, end in ((0, 2), (2, 8), (8, 9), (9, 13)):
        prediction, _, state = model(values.detach()[:, start:end], state)
        chunks.append(prediction)
    streamed = torch.cat(chunks, dim=1)
    torch.testing.assert_close(full.detach(), streamed)
    streamed.square().sum().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(full_grads[name], parameter.grad, rtol=1e-4, atol=1e-5)
    payload = io.BytesIO()
    torch.save({"config": config, "model": model.state_dict()}, payload)
    payload.seek(0)
    checkpoint = torch.load(payload, weights_only=True)
    restored = build_sequence_model(checkpoint["config"]).eval()
    restored.load_state_dict(checkpoint["model"], strict=True)
    torch.testing.assert_close(restored(values.detach())[0], full.detach())


@pytest.mark.parametrize(
    "field,value",
    [
        ("ff_activation", True),
        ("ff_activation", "relu"),
        ("ff_activation", None),
        ("ff_hidden", True),
        ("ff_hidden", 0),
        ("ff_hidden", -1),
        ("ff_hidden", 64.0),
        ("ff_hidden", "64"),
    ],
)
def test_ff_options_reject_bad_values_at_both_boundaries(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    config = _config(**{field: value})
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"model": config}), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_config(path)
    with pytest.raises((ValueError, TypeError)):
        build_sequence_model(config)


def test_swiglu_cannot_be_silently_ignored_on_gru() -> None:
    with pytest.raises(ValueError):
        build_sequence_model(_config(architecture="gru", ff_activation="swiglu", ff_hidden=64))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["bf16", "bf16_bf16x9"])
def test_bf16_compiled_swiglu_matches_eager_math_on_receiver(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires the receiving CUDA host")
    if mode == "bf16_bf16x9":
        monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    config = _config(ff_activation="swiglu", ff_hidden=64, precision_mode=mode)
    reference = build_sequence_model(config).cuda().eval()
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
