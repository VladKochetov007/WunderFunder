"""Explicit sequence-model precision controls with process identity binding.

The default path uses IEEE FP32. BF16 autocast is scoped to model computation;
cuBLAS emulation is process-wide and may also affect Muon matrix operations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Literal

import torch

from wnn.training.sequence_compile import COMPILER_RECOMPILE_LIMIT

PrecisionMode = Literal["fp32", "tf32", "bf16", "fp32_bf16x9", "bf16_bf16x9"]
PRECISION_MODES = ("fp32", "tf32", "bf16", "fp32_bf16x9", "bf16_bf16x9")
EMULATION_ENV = "CUBLAS_EMULATE_SINGLE_PRECISION"


def validate_process_environment(mode: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Fail closed when process-wide cuBLAS emulation disagrees with the mode."""
    if mode not in PRECISION_MODES:
        raise ValueError(f"unsupported sequence precision mode: {mode!r}")
    environment = os.environ if environ is None else environ
    value = environment.get(EMULATION_ENV)
    if mode in ("fp32_bf16x9", "bf16_bf16x9"):
        if value != "1":
            raise ValueError(f"{EMULATION_ENV}=1 is required for {mode}")
    elif value not in (None, "0"):
        raise ValueError(f"{EMULATION_ENV} must be unset or 0 for {mode}")
    return value


def configure_precision_mode(mode: str) -> None:
    """Validate process identity and reset global PyTorch precision flags to FP32."""
    validate_process_environment(mode)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _tf32_context() -> AbstractContextManager[None]:
    """Enable TF32 only for the dynamic extent of one model forward or backward."""

    class _Context:
        def __enter__(self) -> None:
            self.precision = torch.get_float32_matmul_precision()
            self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            self.cudnn_tf32 = torch.backends.cudnn.allow_tf32
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return None

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            torch.set_float32_matmul_precision(self.precision)
            torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
            torch.backends.cudnn.allow_tf32 = self.cudnn_tf32

    return _Context()


def forward_context(mode: str, device_type: str = "cuda") -> AbstractContextManager[object]:
    """Return the precision context for one model forward call."""
    if mode in ("bf16", "bf16_bf16x9"):
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if mode == "tf32":
        return _tf32_context()
    if mode in ("fp32", "fp32_bf16x9"):
        return nullcontext()
    raise ValueError(f"unsupported sequence precision mode: {mode!r}")


def backward_context(mode: str, device_type: str = "cuda") -> AbstractContextManager[object]:
    """Return a model-backward context; autocast is intentionally absent for BF16."""
    if mode == "tf32":
        return _tf32_context()
    if mode in ("fp32", "bf16", "fp32_bf16x9", "bf16_bf16x9"):
        return nullcontext()
    raise ValueError(f"unsupported sequence precision mode: {mode!r}")


def checkpoint_context_fn(
    mode: str, device_type: str = "cuda"
) -> tuple[AbstractContextManager[object], AbstractContextManager[object]]:
    """Provide matching precision contexts to non-reentrant checkpoint regions."""
    if mode in ("bf16", "bf16_bf16x9"):
        return forward_context(mode, device_type), forward_context(mode, device_type)
    if mode in ("fp32", "tf32", "fp32_bf16x9"):
        return nullcontext(), nullcontext()
    raise ValueError(f"unsupported sequence precision mode: {mode!r}")


def validate_forward_state(mode: str, device_type: str = "cuda") -> None:
    """Check that model execution is inside the selected forward precision scope."""
    autocast = torch.is_autocast_enabled(device_type)
    if mode in ("bf16", "bf16_bf16x9"):
        if not autocast or torch.get_autocast_dtype(device_type) != torch.bfloat16:
            raise RuntimeError("bf16 mode requires a bfloat16 autocast model-forward context")
        if (
            torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"
        ):
            raise RuntimeError("bf16 mode requires TF32 controls to remain disabled")
        return
    if autocast:
        raise RuntimeError(f"{mode} mode requires autocast to be disabled")
    if mode == "tf32":
        if (
            not torch.backends.cuda.matmul.allow_tf32
            or not torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "high"
        ):
            raise RuntimeError("tf32 mode requires its scoped matmul controls")
        return
    if mode in ("fp32", "fp32_bf16x9"):
        if (
            torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"
        ):
            raise RuntimeError(f"{mode} mode requires strict FP32 PyTorch matmul controls")
        return
    raise ValueError(f"unsupported sequence precision mode: {mode!r}")


def precision_identity(
    mode: str,
    attention_backend: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Describe requested arithmetic and boundaries for provenance and resume checks."""
    environment_value = validate_process_environment(mode, environ)
    if attention_backend not in ("math", "efficient", "flex"):
        raise ValueError(f"unsupported sequence attention backend: {attention_backend!r}")
    descriptions: dict[str, dict[str, object]] = {
        "fp32": {
            "model_compute": "IEEE FP32",
            "model_forward": "strict FP32",
            "loss": "FP32",
            "optimizer": "FP32 outside model precision contexts",
        },
        "tf32": {
            "model_compute": "PyTorch TF32 matmuls in scoped model forward/backward",
            "model_forward": "FP32 tensors with TF32 matmul controls",
            "loss": "FP32 outside scoped model contexts",
            "optimizer": "FP32 outside model precision contexts",
        },
        "bf16": {
            "model_compute": "CUDA BF16 autocast for model forward",
            "model_forward": "FP32 inputs/parameters with BF16 autocast operations",
            "loss": "FP32 after prediction and hint casts",
            "optimizer": "FP32 outside autocast",
        },
        "fp32_bf16x9": {
            "model_compute": "process-wide cuBLAS single-precision emulation request",
            "model_forward": "FP32 tensor interface; cuBLAS emulation may affect kernels",
            "loss": "FP32",
            "optimizer": "process-wide cuBLAS emulation may affect Muon operations",
        },
        "bf16_bf16x9": {
            "model_compute": "BF16 autocast plus process-wide cuBLAS single-precision emulation",
            "model_forward": "FP32 tensor interface with BF16 autocast and cuBLAS emulation",
            "loss": "FP32 after prediction and hint casts",
            "optimizer": "process-wide cuBLAS emulation may affect Muon operations",
        },
    }
    return {
        "mode": mode,
        "attention_backend": attention_backend,
        "cublas_emulate_single_precision": environment_value,
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "activation_cache_dtype": "float32",
        "amp_scaler": False,
        **descriptions[mode],
    }


def runtime_precision_identity(
    config: Mapping[str, object], runtime: Mapping[str, object]
) -> dict[str, object]:
    """Add selected model precision and attention settings to runtime provenance."""
    model = config.get("model")
    backend = str(model.get("attention_backend", "math")) if isinstance(model, Mapping) else "math"
    precision = precision_identity(str(config.get("precision_mode", "fp32")), backend)
    compiled = bool(model.get("compile_blocks", False)) if isinstance(model, Mapping) else False
    compiler = {
        "enabled": compiled,
        "backend": "inductor" if compiled else None,
        "mode": "default" if compiled else None,
        "fullgraph": False,
        "dynamic": None,
        "recompile_limit": COMPILER_RECOMPILE_LIMIT if compiled else None,
    }
    return dict(runtime) | {
        "attention_backend": backend,
        "precision": precision,
        "compiler": compiler,
    }
