"""Run a synthetic-only shape and throughput probe through the sequence runtime."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_fit_utils import make_optimizer
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_shape_probe_support import (
    _capture_optional_trace,
    _check_finite,
    _prime_attention,
    _run_update,
    _validate_shape,
)

PROBE_SEED = 20261028


def _precision_flags(mode: str) -> dict[str, object]:
    """Record and verify process flags for the declared strict-FP32 probe mode."""
    flags: dict[str, object] = {
        "mode": mode,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
    }
    flags["strict_fp32"] = (
        mode == "fp32"
        and flags["float32_matmul_precision"] == "highest"
        and flags["cuda_matmul_tf32"] is False
        and flags["cudnn_tf32"] is False
        and flags["cuda_autocast_enabled"] is False
        and flags["cpu_autocast_enabled"] is False
    )
    return flags


def source_identity(config_path: Path) -> dict[str, object]:
    """Bind the receipt to the config, probe source, and current Git revision."""
    helper_path = Path(__file__).resolve()
    repo_root = helper_path.parents[3]
    script_path = repo_root / "scripts" / "probe_sequence_shape.py"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    config_sha256 = (
        hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None
    )
    return {
        "git_revision": revision,
        "config_name": config_path.name,
        "config_sha256": config_sha256,
        "probe_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest()
        if script_path.is_file()
        else None,
        "helper_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Build canonical components and measure only a few finite synthetic updates."""
    identity = source_identity(args.config)
    config = load_config(args.config)
    device, batch_size, chunk_rows, chunks_per_update, feature_width = _validate_shape(args, config)
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise TypeError("validated config model must be a mapping")
    configure_strict_fp32()
    torch.manual_seed(PROBE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(PROBE_SEED)
    model = build_sequence_model(model_config).to(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("canonical model construction did not produce strict FP32 parameters")
    model.train()
    state, primed_rows = _prime_attention(
        model, model_config, device, batch_size, chunk_rows, feature_width
    )
    update_inputs = [
        torch.randn(batch_size, chunk_rows, feature_width, device=device) * 0.05
        for _ in range(chunks_per_update)
    ]
    update_targets = [
        torch.randn(batch_size, chunk_rows, 2, device=device) * 0.05
        for _ in range(chunks_per_update)
    ]
    total_updates = (
        args.warmup_updates + args.measure_updates + int(args.profiler_trace is not None)
    )
    optimizer, scheduler = make_optimizer(model, config, total_updates)
    gradient_clip = float(config["grad_clip"])
    finite_flags: list[Tensor] = []
    for _ in range(args.warmup_updates):
        state, finite = _run_update(
            model, optimizer, scheduler, update_inputs, update_targets, state, gradient_clip
        )
        finite_flags.append(finite)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        host_started = time.perf_counter()
        start_event.record()
    else:
        started = time.perf_counter()
        host_started = started
    for _ in range(args.measure_updates):
        state, finite = _run_update(
            model, optimizer, scheduler, update_inputs, update_targets, state, gradient_clip
        )
        finite_flags.append(finite)
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        elapsed_seconds = time.perf_counter() - started
        peak_allocated = peak_reserved = None
    host_elapsed_seconds = time.perf_counter() - host_started
    prediction_finite = bool(torch.stack(finite_flags).all())
    checks = _check_finite(model, optimizer)
    precision = _precision_flags(str(config["precision_mode"]))
    if precision["strict_fp32"] is not True:
        raise RuntimeError("sequence probe precision flags did not remain strict FP32")
    if args.profiler_trace is not None:
        _capture_optional_trace(
            args.profiler_trace,
            device,
            model,
            optimizer,
            scheduler,
            update_inputs,
            update_targets,
            state,
            gradient_clip,
        )
    rows = batch_size * chunk_rows * chunks_per_update * args.measure_updates
    rows_per_second = rows / max(elapsed_seconds, 1e-12)
    host_rows_per_second = rows / max(host_elapsed_seconds, 1e-12)
    activation_proxy = (
        batch_size * chunk_rows * int(model_config["hidden"]) * int(model_config["layers"])
    )
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        device_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    else:
        device_name, device_memory_bytes = platform.processor() or "CPU", None
    return {
        "status": "ok",
        "label": "synthetic_engineering_only",
        "quality_claim": "no_quality_claim",
        "source_identity": identity,
        "torch_version": str(torch.__version__),
        "device": {
            "type": device.type,
            "name": device_name,
            "total_memory_bytes": device_memory_bytes,
        },
        "model": dict(model_config),
        "optimizer": str(config["optimizer"]),
        "precision": precision,
        "seed": PROBE_SEED,
        "batch_size": batch_size,
        "chunk_rows": chunk_rows,
        "chunks_per_update": chunks_per_update,
        "input_width": feature_width,
        "attention_history_primed_rows": primed_rows,
        "state_semantics": "causal_state_persisted_and_detached_after_each_chunk",
        "warmup_updates": args.warmup_updates,
        "measure_updates": args.measure_updates,
        "measured_rows": rows,
        "elapsed_seconds": elapsed_seconds,
        "rows_per_second": rows_per_second,
        "host_elapsed_seconds": host_elapsed_seconds,
        "host_rows_per_second": host_rows_per_second,
        "activation_proxy_elements": activation_proxy,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "predictions_and_losses_finite": prediction_finite,
        **checks,
        "torch_profiler_enabled": args.profiler_trace is not None,
        "profiler_trace": str(args.profiler_trace) if args.profiler_trace else None,
    }
