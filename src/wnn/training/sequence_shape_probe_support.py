"""Bounded synthetic inputs, streaming updates, and finite checks for shape probes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from wnn.training.sequence_features import sequence_input_width
from wnn.training.sequence_fit_utils import optimizer_states_are_finite_fp32
from wnn.training.sequence_model_helpers import SequenceState
from wnn.training.sequence_models import detach_state

MAX_TOTAL_UPDATES, MAX_CHUNK_ROWS, MAX_BATCH_SIZE = 20, 2_000, 32
MAX_SYNTHETIC_BYTES = 256 * 1024 * 1024
MAX_ACTIVATION_PROXY = 32_000_000


def _history_rows(model_config: Mapping[str, object]) -> int:
    architecture = model_config.get("architecture")
    if architecture == "transformer":
        return max(0, int(model_config["context_rows"]) - 1)
    if architecture == "galam":
        galam = model_config.get("galam")
        if not isinstance(galam, Mapping):
            raise ValueError("GALAM model config is missing its nested shape")
        return int(galam["attention_history"])
    return 0


def _validate_shape(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[torch.device, int, int, int, int]:
    warmup, measured = args.warmup_updates, args.measure_updates
    if type(warmup) is not int or type(measured) is not int or warmup < 0 or measured < 1:
        raise ValueError("update counts must be integers with warmup >= 0 and measure >= 1")
    if warmup + measured > MAX_TOTAL_UPDATES:
        raise ValueError(f"total updates cannot exceed {MAX_TOTAL_UPDATES}")
    batch_size = config["batch_size"] if args.batch_size is None else args.batch_size
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise TypeError("validated config model must be a mapping")
    if model_config.get("architecture") == "galam" and batch_size < 2:
        raise ValueError("GALAM training-mode BatchNorm requires batch size >= 2")
    if config["precision_mode"] != "fp32":
        raise ValueError("the engineering shape probe supports strict FP32 only")
    chunk_rows = config["chunk_rows"]
    chunks_per_update = config["accumulate_chunks"]
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= MAX_CHUNK_ROWS:
        raise ValueError(f"chunk_rows must be between 1 and {MAX_CHUNK_ROWS}")
    if type(chunks_per_update) is not int or not 1 <= chunks_per_update <= 16:
        raise ValueError("accumulate_chunks must be between 1 and 16 for the bounded probe")
    feature_width = sequence_input_width(str(model_config.get("input_feature_mode", "raw112")))
    total_rows = batch_size * (chunks_per_update * chunk_rows + _history_rows(model_config))
    synthetic_bytes = total_rows * (feature_width + 2) * 4
    if synthetic_bytes > MAX_SYNTHETIC_BYTES:
        raise ValueError(
            f"synthetic input/target buffers need {synthetic_bytes} bytes; "
            f"the probe cap is {MAX_SYNTHETIC_BYTES}"
        )
    activation_proxy = (
        batch_size * chunk_rows * int(model_config["hidden"]) * int(model_config["layers"])
    )
    if activation_proxy > MAX_ACTIVATION_PROXY:
        raise ValueError(
            f"activation proxy is {activation_proxy} elements; "
            f"the probe cap is {MAX_ACTIVATION_PROXY}"
        )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    return device, int(batch_size), chunk_rows, chunks_per_update, feature_width


def _prime_attention(
    model: torch.nn.Module,
    model_config: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    chunk_rows: int,
    feature_width: int,
) -> tuple[SequenceState, int]:
    history_rows = _history_rows(model_config)
    remaining = history_rows
    state: SequenceState = None
    model.train()
    with torch.no_grad():
        while remaining:
            rows = min(chunk_rows, remaining)
            values = torch.randn(batch_size, rows, feature_width, device=device) * 0.05
            _, _, state = model(values, state)
            state = detach_state(state)
            remaining -= rows
    return state, history_rows


def _run_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    inputs: list[Tensor],
    targets: list[Tensor],
    state: SequenceState,
    gradient_clip: float,
) -> tuple[SequenceState, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    finite = torch.ones((), dtype=torch.bool, device=inputs[0].device)
    for values, target in zip(inputs, targets, strict=True):
        prediction, _, next_state = model(values, state)
        if prediction.shape != target.shape:
            raise ValueError(
                f"model returned {tuple(prediction.shape)}, expected {tuple(target.shape)}"
            )
        loss = F.mse_loss(prediction.float(), target) / len(inputs)
        finite = finite & torch.isfinite(prediction).all() & torch.isfinite(loss)
        loss.backward()
        state = detach_state(next_state)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    scheduler.step()
    return state, finite


def _check_finite(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    """Summarize gradient, parameter, and optimizer-state finiteness after synchronization."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients_present = all(parameter.grad is not None for parameter in trainable)
    gradient_flags = [
        torch.isfinite(parameter.grad).all()
        for parameter in trainable
        if parameter.grad is not None
    ]
    gradients_finite = gradients_present and bool(torch.stack(gradient_flags).all())
    parameter_flags = [torch.isfinite(parameter).all() for parameter in trainable]
    parameters_finite = bool(torch.stack(parameter_flags).all())
    optimizer_finite = optimizer_states_are_finite_fp32(optimizer)
    return {
        "trainable_parameter_count": len(trainable),
        "all_trainable_parameters_have_gradients": gradients_present,
        "gradients_finite": gradients_finite,
        "parameters_finite": parameters_finite,
        "optimizer_fp32_state_finite": optimizer_finite,
    }


def _capture_optional_trace(
    trace_path: Path,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    inputs: list[Tensor],
    targets: list[Tensor],
    state: SequenceState,
    gradient_clip: float,
) -> None:
    """Capture one explicitly requested update separately from the timing window."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        _run_update(model, optimizer, scheduler, inputs, targets, state, gradient_clip)
        prof.step()
    prof.export_chrome_trace(str(trace_path))
