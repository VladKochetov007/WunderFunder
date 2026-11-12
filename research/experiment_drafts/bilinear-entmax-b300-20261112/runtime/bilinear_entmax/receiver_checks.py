"""One-GPU server-only engineering gates for the Bilinear Entmax model variants."""

from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from typing import Any

import torch

from bilinear_entmax.bindings import PENDING_FULL_SHAPE_ACCEPTANCE
from bilinear_entmax.receiver_attention import check_attention_contract
from bilinear_entmax.receiver_input import check_input_routing
from bilinear_entmax.receiver_optimizer import check_optimizer_ownership
from bilinear_entmax.receiver_streaming import check_sequence_reset, check_streaming_cache
from bilinear_entmax.receiver_support import (
    ATOL,
    VARIANTS,
    assert_close,
    assert_dtype_aware_close,
    assert_shared_initial_state,
    build_native_probe_model,
    build_probe_model,
    canonical_wmse_backward,
    cpu_state_dict,
    synthetic_tensor,
)

MECHANISM_CONTRACT = "bilinear-entmax.mechanisms.v1"
RECEIPT_SCHEMA = "bilinear-entmax.receiver-checks.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEEDS = (206201, 206202)
_NATIVE_SOFTMAX_ATOL = 2e-2
_NATIVE_SOFTMAX_RTOL = 2e-2


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
        raise RuntimeError("Bilinear Entmax receiver checks require Linux CUDA")
    if torch.cuda.device_count() != 1 or torch.distributed.is_initialized():
        raise RuntimeError("receiver checks require one visible GPU and no process group")
    if "B300" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Bilinear Entmax receiver checks require a B300 device")
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
    extension = _mapping(config.get("bilinear_entmax"), "bilinear_entmax")
    cell = _mapping(extension.get("cell"), "bilinear_entmax.cell")
    if (
        extension.get("schema") != "bilinear-entmax.model.v1"
        or extension.get("preprocessing_sha256") != artifact_sha256
        or cell.get("schema") != "bilinear-entmax.cell.v1"
        or cell.get("variant") not in VARIANTS
        or type(cell.get("seed")) is not int
        or cell.get("seed") not in _SEEDS
        or config.get("seed") != cell.get("seed")
        or cell.get("id") != f"be_{cell.get('variant')}_s{cell.get('seed')}"
    ):
        raise ValueError("receiver cell is outside the frozen Bilinear Entmax panel")
    preprocessing = _mapping(extension.get("preprocessing"), "bilinear_entmax.preprocessing")
    if preprocessing != artifact:
        raise ValueError("receiver preprocessing differs from the config-bound artifact")
    from bilinear_entmax.bindings import RUNTIME_COMMIT, SPLIT_SHA256

    fit_identity = _mapping(artifact.get("fit_identity"), "preprocessing.fit_identity")
    if (
        artifact.get("schema") != "bilinear-entmax.preprocessing.v1"
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


def _variant_checks(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, dict[str, Any]]:
    """Check initial state, causal streaming, reload, compilation, and main loss."""
    from wnn.training.sequence_predict import predict_sequence

    sample = synthetic_tensor((64, 112), device, seed + 31)
    target = synthetic_tensor((1, 16, 2), device, seed + 37, scale=1.4)
    native = build_native_probe_model(config, artifact, seed, device)
    native_state = cpu_state_dict(native)
    native.eval()
    native_output = torch.from_numpy(predict_sequence(native, sample, 16, device))
    results: dict[str, dict[str, Any]] = {}

    for variant in VARIANTS:
        caller_cpu_rng = torch.get_rng_state().clone()
        caller_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        model = build_probe_model(config, artifact, variant, seed, device)
        if not torch.equal(torch.get_rng_state(), caller_cpu_rng) or any(
            not torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(), caller_cuda_rng, strict=True
            )
        ):
            raise AssertionError(f"{variant} probe construction changed the caller RNG")
        state = cpu_state_dict(model)
        state_report = assert_shared_initial_state(state, native_state, variant)
        model.eval()
        output = torch.from_numpy(predict_sequence(model, sample, 16, device))
        native_parity: dict[str, float] | None = None
        if variant in ("input_linear", "input_bilinear"):
            native_parity = {
                "max_abs": assert_close(
                    output, native_output, f"{variant} zero-W native output", atol=1e-5
                )
            }
            for name in state:
                if name in ("input_projection.bilinear_output.weight",) and bool(
                    state[name].count_nonzero()
                ):
                    raise AssertionError(f"{variant} output W is not zero at initialization")
        elif variant == "attention_softmax":
            native_parity = assert_dtype_aware_close(
                output,
                native_output,
                "BF16 canonical softmax model parity",
                atol=_NATIVE_SOFTMAX_ATOL,
                rtol=_NATIVE_SOFTMAX_RTOL,
            )

        restored = build_probe_model(config, artifact, variant, seed, device)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        restored_output = torch.from_numpy(predict_sequence(restored, sample, 16, device))
        reload_gap = assert_close(output, restored_output, f"{variant} strict-reload callback")
        streaming = check_streaming_cache(model, sample, 16, 32, "bf16_bf16x9")
        reset = check_sequence_reset(model, sample, 16, "bf16_bf16x9")

        changed = sample.clone()
        changed[32:] = changed[32:] + 4.0
        prefix = torch.from_numpy(predict_sequence(model, sample[:32], 16, device))
        extended = torch.from_numpy(predict_sequence(model, changed, 16, device))[:32]
        causal_gap = assert_close(prefix, extended, f"{variant} suffix-causality prefix")

        model.zero_grad(set_to_none=True)
        gradient = canonical_wmse_backward(
            model, synthetic_tensor((1, 16, 112), device, seed + 41), target, "bf16_bf16x9"
        )
        compiled = build_probe_model(config, artifact, variant, seed, device, compile_blocks=True)
        compiled_output = torch.from_numpy(predict_sequence(compiled, sample, 16, device))
        compile_report = assert_dtype_aware_close(
            output,
            compiled_output,
            f"{variant} compiled/eager callback",
            atol=2e-2,
            rtol=2e-2,
        )
        results[variant] = {
            "matched_initial_state": state_report,
            "native_initial_output_parity": native_parity,
            "strict_reload_max_abs": reload_gap,
            "causal_prefix_max_abs": causal_gap,
            "fresh_state_reset": reset,
            "streaming_cache": streaming,
            "main_loss_gradient": gradient,
            "compiled_eager_parity": compile_report,
        }
        del compiled, restored, model
        torch.cuda.empty_cache()
    return results


def _full_shape_gate() -> dict[str, Any]:
    """Describe the separate full-shape profile and human approval required for fit."""
    return deepcopy(PENDING_FULL_SHAPE_ACCEPTANCE)


def run_receiver_checks(
    config: dict[str, Any], artifact: dict[str, Any], artifact_sha256: str, device: torch.device
) -> dict[str, Any]:
    """Run a bounded all-variant CUDA receiver gate without fitting or official scoring."""
    cell, device_name = _validate_request(config, artifact, artifact_sha256, device)
    seed = int(cell["seed"])
    variants = _variant_checks(config, artifact, device, seed)
    input_routing = check_input_routing(config, artifact, device, seed + 47)
    attention = check_attention_contract(config, artifact, device, seed + 53)
    optimizer = check_optimizer_ownership(config, artifact, device, seed + 59)
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
        "variant_checks": variants,
        "input_routing_contract": input_routing,
        "attention_contract": attention,
        "muon_ownership": optimizer,
        "full_shape_acceptance": _full_shape_gate(),
        "tolerance_atol": ATOL,
        "quality_score": None,
    }
