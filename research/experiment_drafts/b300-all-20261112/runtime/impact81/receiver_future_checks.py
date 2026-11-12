"""Receiver-only B300 probes for card07 future-label heads and objectives."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

# Check receipts and artifact/config values are heterogeneous JSON metadata.
SEED = 204101
OFFSETS = (1, 16, 64)
# Keep the synthetic efficient-SDPA probe's head width kernel-aligned; the
# production card07 model remains pinned in its native cell configuration.
MODEL_CONFIG: dict[str, Any] = {
    "architecture": "transformer",
    "hidden": 64,
    "layers": 24,
    "heads": 4,
    "ff_multiplier": 2.0,
    "context_rows": 16,
    "dropout": 0.0,
    "checkpoint_blocks": False,
    "hint_dim": 0,
    "precision_mode": "bf16_bf16x9",
    "attention_backend": "efficient",
    "compile_blocks": False,
}


def run_future_checks(
    artifact: dict[str, Any], artifactsha: str, device: torch.device
) -> dict[str, Any]:
    """Check card07 initialization, gradients, target alignment, causality, and export.

    Args:
        artifact: Shared train-only normalization/knot artifact already validated by the caller.
        artifactsha: SHA256 of that exact preprocessing artifact.
        device: The actual B300 device selected by the receiver worker.

    Returns:
        JSON-safe engineering results for the future-head module and objective.

    Raises:
        RuntimeError: If checks do not run on an admitted B300 or any invariant fails.
        ValueError: If the artifact metadata or supplied device is invalid.
    """
    _require_b300(device)
    if len(artifactsha) != 64 or any(char not in "0123456789abcdef" for char in artifactsha):
        raise ValueError("preprocessing SHA256 is malformed")
    if artifact.get("schema") != "impact81.preprocessing.v1":
        raise ValueError("unsupported card07 preprocessing artifact")

    from impact81.future_model import build_future_model, drop_future_heads
    from impact81.future_objective import FUTURE_OFFSETS, future_objective_sums, future_scaled_loss
    from impact81.receiver_future_objectives import check_future_objective_helpers
    from impact81.receiver_probes import (
        _check_sequence_behavior,
        _check_state_roundtrip,
        _max_abs,
        _predict,
    )
    from wnn.training.sequence_models import build_sequence_model

    if tuple(FUTURE_OFFSETS) != OFFSETS:
        raise RuntimeError("future objective offsets differ from the frozen card07 recipe")
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    values = torch.randn((2, 80, 112), generator=generator, dtype=torch.float32).to(device)
    torch.manual_seed(SEED)
    plain = build_sequence_model(dict(MODEL_CONFIG)).to(device=device, dtype=torch.float32).eval()
    torch.manual_seed(SEED)
    treatment = (
        build_future_model(dict(MODEL_CONFIG), SEED).to(device=device, dtype=torch.float32).eval()
    )
    _copy_normalization(plain, artifact, device)
    _copy_normalization(treatment, artifact, device)

    with torch.no_grad():
        plain_prediction, _ = _predict(plain, values)
        treatment_prediction, _ = _predict(treatment, values)
    initial_error = _max_abs(plain_prediction, treatment_prediction)
    if not torch.equal(plain_prediction, treatment_prediction):
        raise RuntimeError(
            f"future heads changed the initial canonical prediction: {initial_error}"
        )

    objective_receipt = check_future_objective_helpers(device)
    gradient_receipt = _check_future_gradients(
        treatment, values, future_objective_sums, future_scaled_loss
    )
    treatment.eval()
    behavior = _check_sequence_behavior(treatment, values)
    config = {"model": dict(MODEL_CONFIG)}

    def builder(saved_config: dict[str, Any], target_device: torch.device) -> nn.Module:
        """Rebuild a future-head model for strict state-dict restoration."""
        torch.manual_seed(SEED)
        restored = build_future_model(dict(saved_config["model"]), SEED)
        restored.to(device=target_device, dtype=torch.float32)
        return restored

    roundtrip = _check_state_roundtrip(treatment, config, builder, values, device)
    export_receipt = _check_export_pruning(
        treatment, config, builder, values, device, drop_future_heads
    )
    torch.cuda.synchronize(device)
    return {
        "variant": "future_heads",
        "device": torch.cuda.get_device_name(device),
        "precision_mode": "bf16_bf16x9",
        "initial_prediction_equal": True,
        "initial_prediction_max_abs": initial_error,
        "objective": objective_receipt,
        "future_head_gradients": gradient_receipt,
        "sequence_behavior": behavior,
        "strict_state_roundtrip": roundtrip,
        "export_pruning": export_receipt,
        "preprocessing_sha256": artifactsha,
    }


def _require_b300(device: torch.device) -> None:
    """Fail closed unless called on a Linux B300 with the pinned precision mode."""
    if sys.platform != "linux" or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("future-head receiver checks require Linux and an actual CUDA B300")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise RuntimeError("CUBLAS_EMULATE_SINGLE_PRECISION=1 is required for BF16/BF16x9")
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    if "B300" not in name.upper() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"future-head checks require an NVIDIA B300 with BF16; found {name!r}")


def _copy_normalization(model: nn.Module, artifact: dict[str, Any], device: torch.device) -> None:
    """Bind the probe model to the shared canonical train-only moments."""
    mean = torch.as_tensor(artifact.get("mean"), dtype=torch.float32, device=device)
    std = torch.as_tensor(artifact.get("std"), dtype=torch.float32, device=device)
    if mean.shape != (112,) or std.shape != (112,):
        raise ValueError("preprocessing moments must contain 112 features")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("preprocessing moments must be finite with positive scales")
    with torch.no_grad():
        model.mean.copy_(mean)
        model.std.copy_(std)


def _check_future_gradients(
    model: nn.Module,
    values: Tensor,
    sums_fn: Callable[..., Any],
    scaled_loss_fn: Callable[..., Tensor],
) -> dict[str, float | int]:
    """Require finite, connected gradients through every future prediction head."""
    from impact81.future_objective import future_window_denominators
    from wnn.training.sequence_precision import forward_context

    model.train()
    model.zero_grad(set_to_none=True)
    target = torch.randn((values.shape[0], values.shape[1], 2), device=values.device) * 0.4
    target[0, 67, 0] = torch.nan
    mask = torch.ones(target.shape[:2], dtype=torch.bool, device=values.device)
    with forward_context("bf16_bf16x9", values.device.type):
        _, auxiliary, _ = model(values)
    sums = sums_fn(auxiliary, target, mask, 0, values.shape[1])
    denominators = future_window_denominators(target, mask, 0, values.shape[1]).sum(dim=1)
    loss = scaled_loss_fn(sums, denominators, 1)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("future auxiliary loss is nonfinite")
    loss.backward()
    parameters = list(model.future_heads.parameters())
    if not parameters or any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("one or more future heads did not receive gradients")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("future head gradients are nonfinite")
    norm = torch.sqrt(sum(gradient.float().square().sum() for gradient in gradients))
    if not bool(torch.isfinite(norm) and norm > 0):
        raise RuntimeError("future-head aggregate gradient norm must be finite and positive")
    model.zero_grad(set_to_none=True)
    return {
        "head_parameter_tensors": len(parameters),
        "head_parameter_values": sum(parameter.numel() for parameter in parameters),
        "gradient_norm": float(norm.item()),
        "loss": float(loss.item()),
    }


def _check_export_pruning(
    model: nn.Module,
    config: dict[str, Any],
    builder: Callable[[dict[str, Any], torch.device], nn.Module],
    values: Tensor,
    device: torch.device,
    drop_heads: Callable[[nn.Module], None],
) -> dict[str, bool | float]:
    """Strictly reload a treatment and prove pruning leaves its serving output exact."""
    from impact81.receiver_probes import _max_abs, _predict
    from wnn.training.sequence_precision import forward_context

    checkpoint_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    exported = builder(config, device).eval()
    exported.load_state_dict(checkpoint_state, strict=True)
    with torch.no_grad():
        before, _ = _predict(exported, values)
    drop_heads(exported)
    with torch.no_grad(), forward_context("bf16_bf16x9", values.device.type):
        after, hints, _ = exported(values)
    if len(exported.future_heads) != 0:
        raise RuntimeError("export pruning left future-label heads registered")
    if tuple(hints.shape) != (values.shape[0], values.shape[1], 0):
        raise RuntimeError("pruned model must return the canonical empty hint tensor")
    error = _max_abs(before, after)
    if not torch.equal(before, after):
        raise RuntimeError(f"dropping future heads changed primary predictions: {error}")
    return {
        "strict_load_passed": True,
        "heads_removed": True,
        "prediction_equal": True,
        "max_abs_error": error,
    }
