"""Server-only B300 checks for impact81 depth-supervision and tied-depth models."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from impact81.receiver_depth_helpers import (
    GRAD_ATOL,
    GRAD_RTOL,
    _build_depth_optimizer_config,
    _check_auxiliary_gradients,
    _check_block_copies,
    _check_export_pruning,
    _check_logical_caches,
    _check_optimizer_ownership,
    _check_tied_gradient_sum,
)

# Artifact and receipt metadata are heterogeneous canonical JSON mappings.
SEED = 204101
VARIANTS = ("plain", "deep_supervision", "copied_depth", "tied_depth")
DEPTH_CONFIG: dict[str, object] = {
    "architecture": "transformer",
    "hidden": 48,
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


def _config(artifact: dict[str, Any], artifact_sha256: str, variant: str) -> dict[str, Any]:
    """Bind one receiver probe variant to the shared artifact and small model."""
    return {
        "model": dict(DEPTH_CONFIG),
        "impact81": {
            "schema": "impact81.model.v1",
            "cell": {"variant": variant, "seed": SEED},
            "preprocessing": artifact,
            "preprocessing_sha256": artifact_sha256,
        },
    }


def _build(
    builder: Callable[[dict[str, Any], torch.device], nn.Module],
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build one model with the common seed and switch it to evaluation mode."""
    torch.manual_seed(SEED)
    model = builder(config, device)
    model.eval()
    return model


def run_depth_checks(
    artifact: dict[str, Any], artifactsha: str, device: torch.device
) -> dict[str, Any]:
    """Run card04/05 semantic and optimizer checks on the receiving B300 only.

    Args:
        artifact: Shared train-only preprocessing artifact already validated by the caller.
        artifactsha: SHA256 of that exact preprocessing artifact.
        device: The actual B300 device selected by the receiver worker.

    Returns:
        A JSON-serializable summary of model, cache, gradient, optimizer, and reload checks.

    Raises:
        RuntimeError: If the environment is not an admitted B300 receiver or a check fails.
        ValueError: If the artifact digest or preprocessing structure is invalid.
    """
    if sys.platform != "linux" or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("depth receiver checks require Linux and an actual CUDA B300")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise RuntimeError("CUBLAS_EMULATE_SINGLE_PRECISION=1 is required for BF16/BF16x9")
    if len(artifactsha) != 64 or any(char not in "0123456789abcdef" for char in artifactsha):
        raise ValueError("preprocessing SHA256 is malformed")
    if artifact.get("schema") != "impact81.preprocessing.v1":
        raise ValueError("unsupported depth-check preprocessing artifact")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    device_name = torch.cuda.get_device_name(device)
    if "B300" not in device_name.upper() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"depth receiver checks require an NVIDIA B300 with BF16; found {device_name!r}"
        )

    from impact81.modeling import build_model
    from impact81.receiver_probes import (
        CHUNK_ATOL,
        CHUNK_RTOL,
        _check_sequence_behavior,
        _check_state_roundtrip,
        _max_abs,
        _predict,
    )
    from wnn.training.sequence_precision import forward_context

    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    probe = torch.randn((2, 48, 112), generator=generator, dtype=torch.float32).to(device)
    configs = {variant: _config(artifact, artifactsha, variant) for variant in VARIANTS}
    models: dict[str, nn.Module] = {}
    for variant, config in configs.items():
        torch.manual_seed(SEED)
        models[variant] = build_model(config, device).eval()

    predictions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for variant, model in models.items():
            predictions[variant], _ = _predict(model, probe)
    deep_max = _max_abs(predictions["plain"], predictions["deep_supervision"])
    if not torch.equal(predictions["plain"], predictions["deep_supervision"]):
        raise RuntimeError(f"card04 changed its initial final prediction: max_abs={deep_max}")
    depth_max = _max_abs(predictions["copied_depth"], predictions["tied_depth"])
    if not torch.equal(predictions["copied_depth"], predictions["tied_depth"]):
        raise RuntimeError(f"card05 copied/tied initial functions differ: max_abs={depth_max}")

    with torch.no_grad(), forward_context("bf16_bf16x9", device.type):
        _, deep_hints, _ = models["deep_supervision"](probe)
    if tuple(deep_hints.shape) != (probe.shape[0], probe.shape[1], 0):
        raise RuntimeError("deep-supervision evaluation must return an empty canonical hint tensor")
    deep_gradients = _check_auxiliary_gradients(models["deep_supervision"], probe)
    export_pruning = _check_export_pruning(
        models["deep_supervision"], configs["deep_supervision"], build_model, probe, device
    )

    behavior: dict[str, dict[str, Any]] = {}
    roundtrips: dict[str, dict[str, Any]] = {}
    for variant in ("deep_supervision", "copied_depth", "tied_depth"):
        model = models[variant].eval()
        behavior[variant] = _check_sequence_behavior(model, probe)
        behavior[variant]["fresh_state_reset_bitwise_equal"] = True
        roundtrips[variant] = _check_state_roundtrip(
            model, configs[variant], build_model, probe, device
        )

    block_copies = _check_block_copies(models["copied_depth"], models["tied_depth"])
    logical_caches = {
        variant: _check_logical_caches(models[variant], probe)
        for variant in ("copied_depth", "tied_depth")
    }
    optimizer_config = _build_depth_optimizer_config()
    optimizer = _check_optimizer_ownership(models["tied_depth"], optimizer_config, DEPTH_CONFIG)
    gradient_sum = _check_tied_gradient_sum(models["copied_depth"], models["tied_depth"], probe)
    torch.cuda.synchronize(device)
    return {
        "schema": "impact81.depth-receiver-checks.v1",
        "status": "passed",
        "scope": "small-model engineering check; no full profile, fit, or quality claim",
        "precision_mode": "bf16_bf16x9",
        "preprocessing_sha256": artifactsha,
        "model_config": DEPTH_CONFIG,
        "device": {
            "name": device_name,
            "index": device_index,
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "tolerances": {
            "initial_function": "bitwise equal",
            "strict_roundtrip": "bitwise equal",
            "chunk_atol": CHUNK_ATOL,
            "chunk_rtol": CHUNK_RTOL,
            "tied_gradient_atol": GRAD_ATOL,
            "tied_gradient_rtol": GRAD_RTOL,
        },
        "initial_function_max_abs": {
            "plain_vs_deep_supervision": deep_max,
            "copied_vs_tied_depth": depth_max,
        },
        "deep_supervision": {
            "auxiliary_shape": [probe.shape[0], probe.shape[1], 3, 2],
            "evaluation_hint_shape": list(deep_hints.shape),
            "head_gradients": deep_gradients,
            "strict_load_then_drop": export_pruning,
        },
        "copied_tied_depth": {
            "block_copy": block_copies,
            "logical_caches": logical_caches,
            "optimizer_ownership": optimizer,
            "gradient_sum": gradient_sum,
        },
        "sequence_behavior": behavior,
        "strict_state_roundtrips": roundtrips,
    }
