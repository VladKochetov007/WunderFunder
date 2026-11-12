"""Small synthetic fixtures and assertions for the PR81 GRU receiver gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

ATOL = 2e-6
FEATURES = 112


def synthetic_inputs(batch: int, rows: int, device: torch.device, seed: int) -> Tensor:
    """Create repeatable raw-112 rows on CPU, then move them to the receiver device."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((batch, rows, FEATURES), generator=generator, dtype=torch.float32)
    return values.to(device=device)


def synthetic_targets(batch: int, rows: int, device: torch.device, seed: int) -> Tensor:
    """Create bounded repeatable raw labels for loss and gradient probes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((batch, rows, 2), generator=generator, dtype=torch.float32) * 1.4
    return values.to(device=device)


def install_normalization(model: nn.Module, artifact: Mapping[str, Any]) -> None:
    """Install the source-bound common FP32 mean/std into one canonical GRU."""
    mean = torch.as_tensor(artifact["mean"], dtype=torch.float32, device=model.mean.device)
    std = torch.as_tensor(artifact["std"], dtype=torch.float32, device=model.std.device)
    if mean.shape != (FEATURES,) or std.shape != (FEATURES,):
        raise ValueError("GRU receiver preprocessing must contain 112 means and scales")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("GRU receiver preprocessing contains invalid moments")
    with torch.no_grad():
        model.mean.copy_(mean)
        model.std.copy_(std)


def assert_close(actual: Tensor, expected: Tensor, label: str, atol: float = ATOL) -> float:
    """Require finite matching tensors and return their maximum absolute difference."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label} shape or dtype differs")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError(f"{label} contains nonfinite values")
    difference = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
    if difference > atol:
        raise AssertionError(f"{label} differs by {difference:.8g} (tolerance {atol:.8g})")
    return difference


def compare_sequence_state(left: Any, right: Any, label: str) -> float:
    """Compare paired GRU hidden/input-history states and their absolute row position."""
    if left.next_position != right.next_position:
        raise AssertionError(f"{label} absolute row position differs")
    return max(
        assert_close(left.hidden, right.hidden, f"{label} hidden"),
        assert_close(left.input_history, right.input_history, f"{label} input history"),
    )


def assert_head_gradients(module: nn.Module, label: str) -> dict[str, float]:
    """Require finite, nonzero gradients on every declared auxiliary head parameter."""
    gradients: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"{label}.{name} received no gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"{label}.{name} gradient is nonfinite")
        norm = float(parameter.grad.detach().norm().item())
        if norm <= 0:
            raise AssertionError(f"{label}.{name} gradient is zero")
        gradients[name] = norm
    if not gradients:
        raise AssertionError(f"{label} has no trainable auxiliary parameters")
    return gradients


def copy_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Snapshot tensor-only model state in memory for a strict reload probe."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def compare_common_state(
    candidate: nn.Module, reference: nn.Module, label: str
) -> dict[str, float]:
    """Require every canonical shared tensor to begin identically across paired variants."""
    candidate_state, reference_state = candidate.state_dict(), reference.state_dict()
    missing = set(reference_state) - set(candidate_state)
    if missing:
        raise AssertionError(f"{label} is missing canonical state: {sorted(missing)}")
    gaps = {
        name: assert_close(candidate_state[name], expected, f"{label}.{name}", atol=0.0)
        for name, expected in reference_state.items()
    }
    return gaps


def build_probe_model(
    model_config: dict[str, Any],
    variant: str,
    seed: int,
    artifact: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build one paired GRU on-device with identical backbone and normalization."""
    from impact81.gru_models import build_gru_model

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_gru_model(dict(model_config), variant, seed)
    model = model.to(device=device, dtype=torch.float32)
    install_normalization(model, artifact)
    return model


def initialization_rng_checks(
    model_config: dict[str, Any], variants: tuple[str, ...], seed: int
) -> dict[str, bool]:
    """Ensure auxiliary initialization preserves the canonical CPU/CUDA RNG streams."""
    from impact81.gru_models import build_gru_model

    reference_cpu: Tensor | None = None
    reference_cuda: list[Tensor] | None = None
    for variant in ("gru_plain", *variants):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_gru_model(dict(model_config), variant, seed)
        cpu_state = torch.get_rng_state().clone()
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
        if reference_cpu is None:
            reference_cpu, reference_cuda = cpu_state, cuda_state
        elif not torch.equal(cpu_state, reference_cpu) or any(
            not torch.equal(actual, expected)
            for actual, expected in zip(cuda_state, reference_cuda or [], strict=True)
        ):
            raise AssertionError(f"{variant} initialization changed a common RNG stream")
        del model
    return {"cpu_rng_preserved": True, "cuda_rng_preserved": True}


def _export_check(
    model_config: dict[str, Any],
    variant: str,
    seed: int,
    artifact: Mapping[str, Any],
    device: torch.device,
    checkpoint_state: dict[str, Tensor],
    plain_keys: set[str],
    expected_prediction: Tensor,
    sample: Tensor,
) -> float:
    """Strict-load then remove auxiliary tensors and compare the canonical callback."""
    from impact81.gru_models import drop_gru_auxiliary_heads
    from wnn.training.sequence_predict import predict_sequence

    loaded = build_probe_model(model_config, variant, seed, artifact, device)
    loaded.load_state_dict(checkpoint_state, strict=True)
    exported = drop_gru_auxiliary_heads(loaded)
    if set(exported.state_dict()) != plain_keys:
        raise AssertionError(f"{variant} export retained a training-only state tensor")
    prediction = torch.from_numpy(predict_sequence(exported, sample, 1000, device))
    return assert_close(prediction, expected_prediction, f"{variant} clean export parity")


def common_function_and_export_checks(
    model_config: dict[str, Any],
    variants: tuple[str, ...],
    seed: int,
    artifact: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, object]], nn.Module, Tensor, Tensor]:
    """Compare common K16 functions and strict mean-only reloads for every GRU arm."""
    from wnn.training.sequence_predict import predict_sequence

    sample = synthetic_inputs(1, 2000, device, seed + 11)[0]
    plain = build_probe_model(model_config, "gru_plain", seed, artifact, device).eval()
    plain_state = copy_state_dict(plain)
    plain_prediction = torch.from_numpy(predict_sequence(plain, sample, 1000, device))
    report: dict[str, dict[str, object]] = {}
    for variant in variants:
        candidate = build_probe_model(model_config, variant, seed, artifact, device).eval()
        shared_gaps = compare_common_state(candidate, plain, variant)
        prediction = torch.from_numpy(predict_sequence(candidate, sample, 1000, device))
        initial_gap = assert_close(
            prediction, plain_prediction, f"{variant} initial clean function"
        )
        export_gap = _export_check(
            model_config,
            variant,
            seed,
            artifact,
            device,
            copy_state_dict(candidate),
            set(plain_state),
            plain_prediction,
            sample,
        )
        report[variant] = {
            "shared_tensor_count": len(shared_gaps),
            "shared_initial_max_abs": max(shared_gaps.values(), default=0.0),
            "initial_callback_max_abs": initial_gap,
            "strict_load_mean_only_export_max_abs": export_gap,
        }
    return report, plain, sample, plain_prediction
