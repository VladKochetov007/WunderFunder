from __future__ import annotations

from typing import Any

LEGACY_STRICT_FP32 = "strict_fp32_no_autocast_no_tf32_math_sdpa"
BF16_BF16X9 = "bf16_bf16x9"
SUPPORTED_ATTENTION_BACKENDS = frozenset(("math", "efficient", "flex"))
COMPILER_RECOMPILE_LIMIT = 64


def _bf16_bf16x9_identity(attention_backend: str) -> dict[str, object]:
    """Mirror PR40's precision identity for the one accepted nonlegacy mode."""
    if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError("teacher precision names an unsupported attention backend")
    return {
        "mode": BF16_BF16X9,
        "attention_backend": attention_backend,
        "cublas_emulate_single_precision": "1",
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "activation_cache_dtype": "float32",
        "amp_scaler": False,
        "model_compute": "BF16 autocast plus process-wide cuBLAS single-precision emulation",
        "model_forward": "FP32 tensor interface with BF16 autocast and cuBLAS emulation",
        "loss": "FP32 after prediction and hint casts",
        "optimizer": "process-wide cuBLAS emulation may affect Muon operations",
    }


def _compiler_identity(enabled: bool) -> dict[str, object]:
    """Return the compiler receipt dictated by the producer config."""
    return {
        "enabled": enabled,
        "backend": "inductor" if enabled else None,
        "mode": "default" if enabled else None,
        "fullgraph": False,
        "dynamic": None,
        "recompile_limit": COMPILER_RECOMPILE_LIMIT if enabled else None,
    }


def _strict_fp32_identity(
    attention_backend: str, cublas_emulate_single_precision: object
) -> dict[str, object]:
    """Mirror the current native strict-FP32 precision receipt without loading Torch."""
    if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError("teacher precision names an unsupported attention backend")
    if cublas_emulate_single_precision not in (None, "0"):
        raise ValueError("strict FP32 cannot request cuBLAS single-precision emulation")
    return {
        "mode": "fp32",
        "attention_backend": attention_backend,
        "cublas_emulate_single_precision": cublas_emulate_single_precision,
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "activation_cache_dtype": "float32",
        "amp_scaler": False,
        "model_compute": "IEEE FP32",
        "model_forward": "strict FP32",
        "loss": "FP32",
        "optimizer": "FP32 outside model precision contexts",
    }


def validate_precision_contract(
    config: dict[str, Any], provenance: dict[str, Any], runtime: dict[str, Any]
) -> object:
    """Bind current/legacy strict-FP32 or PR40 BF16x9 attestations to the config."""
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("teacher precision recipe has no model config")
    mode = config.get("precision_mode", "fp32")
    attention_backend = model.get("attention_backend", "math")
    compile_blocks = model.get("compile_blocks", False)
    if (
        not isinstance(mode, str)
        or not isinstance(attention_backend, str)
        or type(compile_blocks) is not bool
    ):
        raise ValueError("teacher precision recipe is malformed")

    if mode == BF16_BF16X9:
        if model.get("precision_mode") != mode:
            raise ValueError("teacher runtime model precision differs from its config")
        expected = _bf16_bf16x9_identity(attention_backend)
        if (
            provenance.get("precision") != expected
            or runtime.get("precision") != expected
            or runtime.get("attention_backend") != attention_backend
        ):
            raise ValueError("teacher BF16x9 precision provenance does not match its config")
        if runtime.get("compiler") != _compiler_identity(compile_blocks):
            raise ValueError("teacher compiler provenance does not match its config")
        return expected

    if mode == "fp32" and isinstance(provenance.get("precision"), dict):
        if model.get("precision_mode", mode) != mode:
            raise ValueError("teacher runtime model precision differs from its config")
        observed = provenance["precision"]
        expected = _strict_fp32_identity(
            attention_backend, observed.get("cublas_emulate_single_precision")
        )
        if (
            observed != expected
            or runtime.get("precision") != expected
            or runtime.get("attention_backend") != attention_backend
            or runtime.get("compiler") != _compiler_identity(compile_blocks)
        ):
            raise ValueError("teacher strict-FP32 precision/compiler provenance mismatch")
        return expected

    if mode == "fp32" and provenance.get("precision") == LEGACY_STRICT_FP32:
        if (
            attention_backend != "math"
            or compile_blocks
            or "precision" in runtime
            or "compiler" in runtime
            or runtime.get("attention_backend") != "math"
        ):
            raise ValueError("legacy strict-FP32 provenance does not match its config")
        return LEGACY_STRICT_FP32

    raise ValueError("teacher precision provenance does not match a supported config")
