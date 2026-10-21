from __future__ import annotations

import json
import resource
from pathlib import Path
from typing import Any

import torch

from wnn.training.profiling import profile_update_evidence
from wnn.training.reporting import emit
from wnn.training.state import TrainingRun


def _cgroup_memory_evidence() -> dict[str, Any]:
    """Read memory peak and OOM counters from the active cgroup v2 scope."""
    try:
        relative = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
        result: dict[str, Any] = {"cgroup": relative}
        for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.max"):
            path = directory / name
            if path.is_file():
                result[name.replace(".", "_")] = path.read_text().strip()
        events = directory / "memory.events"
        if events.is_file():
            result["memory_events"] = {
                key: int(value)
                for key, value in (line.split() for line in events.read_text().splitlines())
            }
        return result
    except (OSError, StopIteration, ValueError):
        return {}


def report_profile(run: TrainingRun) -> None:
    """Publish bounded profiling evidence and reject incomplete/nonfinite updates."""
    activation_profile_first_batch_seconds = run.progress.activation_profile_first_batch_seconds
    activation_profile_updates = run.progress.activation_profile_updates
    args = run.options.args
    cumulative_updates = run.progress.cumulative_updates
    device = run.options.device
    epoch_counter = run.progress.epoch_counter
    epoch_grad_stats = run.progress.epoch_grad_stats
    model = run.network.model
    profile_losses = run.progress.profile_losses
    scaler = run.optimization.scaler
    timer = run.progress.timer
    med = timer.report(args.profile_warmup, args.batch)
    emit("PROFILE", json.dumps({k: round(v, 2) for k, v in med.items()}), flush=True)
    process_peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    try:
        process_current_rss = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
    except (OSError, StopIteration, ValueError):
        process_current_rss = None
    emit(
        "PROFILE_RESOURCE_EVIDENCE",
        json.dumps(
            {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None,
                "process_peak_rss_bytes": process_peak_rss,
                "process_current_rss_bytes": process_current_rss,
                **_cgroup_memory_evidence(),
            }
        ),
        flush=True,
    )
    update_counts = epoch_counter.finish(cumulative_updates)
    loss_finite = bool(profile_losses and torch.isfinite(torch.stack(profile_losses)).all().item())
    epoch_updates = update_counts["epoch"]
    evidence = profile_update_evidence(
        epoch_updates["optimizer_attempts"],
        epoch_updates["optimizer_completed_updates"],
        epoch_updates["amp_skipped_updates"],
        len(profile_losses or []),
        loss_finite,
        scaler.is_enabled(),
        None if epoch_grad_stats is None else epoch_grad_stats.get("steps"),
        None if epoch_grad_stats is None else epoch_grad_stats.get("nonfinite_steps"),
    )
    emit("PROFILE_UPDATE_EVIDENCE", json.dumps(evidence), flush=True)
    if args.activation_backend != "none":
        import torch._dynamo.utils as dynamo_utils

        compiler_stats = {
            key: int(value)
            for key, value in dynamo_utils.counters.get("stats", {}).items()
            if key in {"unique_graphs", "calls_captured", "calls_recompiles"}
        }
        finite_parameters = bool(
            torch.stack([torch.isfinite(value).all() for value in model.parameters()]).all().item()
        )
        emit(
            "PROFILE_ACTIVATION_BACKEND_EVIDENCE",
            json.dumps(
                {
                    "backend": args.activation_backend,
                    "fullgraph_compile": True,
                    "first_batch_end_to_end_seconds": activation_profile_first_batch_seconds,
                    "first_forward_compile_seconds": getattr(
                        model, "_activation_profile_compile_timings", {}
                    ).get("forward"),
                    "first_backward_compile_seconds": getattr(
                        model, "_activation_profile_compile_timings", {}
                    ).get("backward"),
                    "compiler_stats": compiler_stats,
                    "optimizer_updates": len(activation_profile_updates),
                    "weight_qat_update_evidence": activation_profile_updates,
                    "represented_weight_fingerprints": getattr(
                        model, "_activation_profile_cache_fingerprints", []
                    ),
                    "represented_weights_changed_between_first_two_forwards": getattr(
                        model, "_activation_profile_cache_fingerprints", []
                    )[0]
                    != getattr(model, "_activation_profile_cache_fingerprints", [])[1]
                    if len(getattr(model, "_activation_profile_cache_fingerprints", [])) >= 2
                    else None,
                    "finite_parameters_after_updates": finite_parameters,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None,
                    "process_peak_rss_bytes": process_peak_rss,
                    "process_current_rss_bytes": process_current_rss,
                    "scope": "compiled recurrence; no quality evaluation",
                }
            ),
            flush=True,
        )
    if not evidence["valid_for_throughput"]:
        raise RuntimeError("profile update evidence is incomplete or nonfinite")
    return


def write_parity(run: TrainingRun) -> None:
    """Write the requested losses and final model/EMA arrays without grading."""
    args = run.options.args
    ema_model = run.optimization.ema_model
    losses_trace = run.progress.losses_trace
    model = run.network.model
    torch.save(
        {
            "losses": torch.stack(losses_trace).cpu(),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": None
            if ema_model is None
            else {k: v.detach().cpu() for k, v in ema_model.module.state_dict().items()},
        },
        args.parity_dump,
    )
    emit("PARITY_DUMP", args.parity_dump, len(losses_trace), flush=True)
    return
