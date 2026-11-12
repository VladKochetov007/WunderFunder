"""One-GPU server-only engineering gates for the Signal Routes model variants."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import torch
from torch import nn

from signal_routes.receiver_attention import check_attention_maps
from signal_routes.receiver_ffn import check_router_contract
from signal_routes.receiver_optimizer import check_optimizer_ownership
from signal_routes.receiver_support import (
    ATOL,
    VARIANTS,
    assert_close,
    assert_shared_initial_state,
    build_probe_model,
    canonical_wmse_backward,
    check_streaming_cache,
    cpu_state_dict,
    synthetic_tensor,
)

MECHANISM_CONTRACT = "signal-routes.mechanisms.v1"
RECEIPT_SCHEMA = "signal-routes.receiver-checks.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Require a JSON object at the server receiver boundary."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _validate_request(
    config: dict[str, Any], artifact: dict[str, Any], artifact_sha256: str, device: torch.device
) -> tuple[dict[str, Any], str]:
    """Fail closed unless this is the frozen single-GPU B300 receiver contract."""
    if sys.platform != "linux" or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Signal Routes receiver checks require Linux CUDA")
    if torch.cuda.device_count() != 1 or torch.distributed.is_initialized():
        raise RuntimeError("receiver checks require one visible GPU and no process group")
    if "B300" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Signal Routes receiver checks require a B300 device")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise RuntimeError("bf16_bf16x9 checks require CUBLAS_EMULATE_SINGLE_PRECISION=1")
    if not _SHA256.fullmatch(artifact_sha256):
        raise ValueError("preprocessing SHA-256 must be lowercase hexadecimal")
    if config.get("precision_mode") != "bf16_bf16x9":
        raise ValueError("receiver config precision must remain bf16_bf16x9")
    if config.get("objective_mode") != "standalone_labels_wmse_v2":
        raise ValueError("receiver config must preserve the PR75 raw-label WMSE objective")
    expected_fit = {
        "epochs": 4,
        "batch_size": 128,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "gradient_horizon_rows": 256,
        "optimizer": "muon",
        "required_world_size": 2,
        "required_device": "B300",
        "training_mask": "all_after_warmup",
        "normalization_scope": "all_fit",
        "kd_weight": 0.0,
        "hint_weight": 0.0,
    }
    if any(config.get(key) != value for key, value in expected_fit.items()):
        raise ValueError("receiver config differs from the frozen four-epoch native recipe")
    model = _mapping(config.get("model"), "model")
    expected_model = {
        "architecture": "transformer",
        "hidden": 768,
        "layers": 24,
        "heads": 12,
        "context_rows": 1024,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "attention_backend": "efficient",
        "compile_blocks": True,
        "precision_mode": "bf16_bf16x9",
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("receiver Transformer model differs from the pinned native recipe")
    extension = _mapping(config.get("signal_routes"), "signal_routes")
    cell = _mapping(extension.get("cell"), "signal_routes.cell")
    if (
        extension.get("schema") != "signal-routes.model.v1"
        or extension.get("preprocessing_sha256") != artifact_sha256
        or cell.get("variant") not in VARIANTS
        or cell.get("seed") not in (205201, 205202)
        or config.get("seed") != cell.get("seed")
        or cell.get("id") != f"sr_{cell.get('variant')}_s{cell.get('seed')}"
    ):
        raise ValueError("receiver cell is outside the frozen Signal Routes panel")
    preprocessing = _mapping(extension.get("preprocessing"), "signal_routes.preprocessing")
    if preprocessing != artifact:
        raise ValueError("receiver preprocessing differs from the config-bound artifact")
    from signal_routes.bindings import RUNTIME_COMMIT, SPLIT_SHA256

    fit_identity = _mapping(artifact.get("fit_identity"), "preprocessing.fit_identity")
    if (
        artifact.get("schema") != "signal-routes.preprocessing.v1"
        or artifact.get("runtime_commit") != RUNTIME_COMMIT
        or fit_identity.get("split_sha256") != SPLIT_SHA256
        or not isinstance(fit_identity.get("fit_members_sha256"), str)
        or len(fit_identity["fit_members_sha256"]) != 64
    ):
        raise ValueError(
            "receiver preprocessing ancestry differs from the frozen train-only source"
        )
    mean = torch.as_tensor(artifact.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(artifact.get("std"), dtype=torch.float32)
    if mean.shape != (112,) or std.shape != (112,):
        raise ValueError("receiver preprocessing must contain 112 feature moments")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("receiver preprocessing moments must be finite with positive scales")
    from wnn.training.sequence_precision import configure_precision_mode

    configure_precision_mode("bf16_bf16x9")
    return cell, torch.cuda.get_device_name(device)


def _router_gradient_check(model: nn.Module, variant: str) -> dict[str, float | bool]:
    """Require conditional gate learning and the declared input-independent control."""
    if not variant.startswith("ffn_"):
        return {"applies": False}
    matrices = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.endswith("routing_matrix")
    ]
    biases = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.endswith("routing_bias")
    ]
    if (
        len(matrices) != 2
        or len(biases) != 2
        or any(value is None for value in (*matrices, *biases))
    ):
        raise AssertionError(f"{variant} router gradient tensors are incomplete")
    matrix_grads = [value for value in matrices if value is not None]
    bias_grads = [value for value in biases if value is not None]
    if variant == "ffn_conditional":
        if not any(float(value.abs().sum().item()) > 0 for value in matrix_grads):
            raise AssertionError("conditional model WMSE did not train its routing matrices")
    elif any(bool(value.count_nonzero()) for value in matrix_grads):
        raise AssertionError("static router matrix must receive zero gradient from zero input")
    if not any(float(value.abs().sum().item()) > 0 for value in bias_grads):
        raise AssertionError(f"{variant} router biases are not trainable")
    return {
        "applies": True,
        "matrix_gradient_l1": sum(float(value.abs().sum().item()) for value in matrix_grads),
        "bias_gradient_l1": sum(float(value.abs().sum().item()) for value in bias_grads),
    }


def _variant_checks(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, dict[str, Any]]:
    """Check matched init, strict reload, causal streaming, gradients and compile per arm."""
    from wnn.training.sequence_predict import predict_sequence

    sample = synthetic_tensor((64, 112), device, seed + 31)
    target = synthetic_tensor((1, 16, 2), device, seed + 37, scale=1.4)
    plain = build_probe_model(config, artifact, "plain", seed, device)
    plain_state = cpu_state_dict(plain)
    plain.eval()
    plain_output = torch.from_numpy(predict_sequence(plain, sample, 16, device))
    del plain
    results: dict[str, dict[str, Any]] = {}
    base_rng_cpu = torch.get_rng_state().clone()
    base_rng_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]

    for variant in VARIANTS:
        model = build_probe_model(config, artifact, variant, seed, device)
        state = cpu_state_dict(model)
        state_report = assert_shared_initial_state(state, plain_state, variant)
        if not torch.equal(torch.get_rng_state(), base_rng_cpu) or any(
            not torch.equal(actual, expected)
            for actual, expected in zip(torch.cuda.get_rng_state_all(), base_rng_cuda, strict=True)
        ):
            raise AssertionError(f"{variant} construction changed the shared initialization RNG")
        model.eval()
        output = torch.from_numpy(predict_sequence(model, sample, 16, device))
        if variant.startswith("ffn_"):
            state_report["uniform_initial_function_max_abs"] = assert_close(
                output, plain_output, f"{variant} uniform-init function", atol=ATOL
            )
            for name in state:
                if name.endswith(("routing_matrix", "routing_bias")) and bool(
                    state[name].count_nonzero()
                ):
                    raise AssertionError(f"{variant} router initialization is not zero at {name}")

        restored = build_probe_model(config, artifact, variant, seed, device)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        restored_output = torch.from_numpy(predict_sequence(restored, sample, 16, device))
        reload_gap = assert_close(output, restored_output, f"{variant} strict-reload callback")
        streaming = check_streaming_cache(model, sample, 16, 32, "bf16_bf16x9")

        changed = sample.clone()
        changed[32:] = changed[32:] + 4.0
        prefix = torch.from_numpy(predict_sequence(model, sample[:32], 16, device))
        extended = torch.from_numpy(predict_sequence(model, changed, 16, device))[:32]
        causal_gap = assert_close(prefix, extended, f"{variant} suffix-causality prefix")

        model.zero_grad(set_to_none=True)
        gradient = canonical_wmse_backward(
            model, synthetic_tensor((1, 16, 112), device, seed + 41), target, "bf16_bf16x9"
        )
        router_gradient = _router_gradient_check(model, variant)

        compiled = build_probe_model(config, artifact, variant, seed, device, compile_blocks=True)
        compiled_output = torch.from_numpy(predict_sequence(compiled, sample, 16, device))
        compile_gap = assert_close(
            output, compiled_output, f"{variant} compiled/eager callback", atol=2e-2
        )
        results[variant] = {
            "matched_initial_state": state_report,
            "strict_reload_max_abs": reload_gap,
            "causal_prefix_max_abs": causal_gap,
            "streaming_cache": streaming,
            "wmse_gradient": gradient,
            "router_gradient": router_gradient,
            "compiled_eager_max_abs": compile_gap,
        }
        del compiled, restored, model
        torch.cuda.empty_cache()
    return results


def run_receiver_checks(
    config: dict[str, Any], artifact: dict[str, Any], artifact_sha256: str, device: torch.device
) -> dict[str, Any]:
    """Run a bounded all-variant CUDA receiver gate without fitting or official scoring."""
    cell, device_name = _validate_request(config, artifact, artifact_sha256, device)
    seed = int(cell["seed"])
    variant_results = _variant_checks(config, artifact, device, seed)
    attention = check_attention_maps(device, seed + 47)
    ffn_router = check_router_contract(device, seed + 53)
    optimizer = check_optimizer_ownership(config, artifact, device, seed)
    torch.cuda.synchronize(device)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "mechanism_contract": MECHANISM_CONTRACT,
        "checked_variants": sorted(VARIANTS),
        "initiating_cell": dict(cell),
        "preprocessing_sha256": artifact_sha256,
        "precision_mode": "bf16_bf16x9",
        "device": device_name,
        "scope": "synthetic engineering checks only; no fit, profile, full-valid score or quality claim",
        "variant_checks": variant_results,
        "attention_analytic": attention,
        "ffn_router_contract": ffn_router,
        "muon_ownership": optimizer,
        "tolerance_atol": ATOL,
        "quality_score": None,
    }
