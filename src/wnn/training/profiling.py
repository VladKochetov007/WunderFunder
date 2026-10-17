from __future__ import annotations

import os
import time

import numpy as np
import torch


def profile_update_evidence(
    optimizer_attempts: int,
    optimizer_completed_updates: int,
    amp_skipped_updates: int,
    loss_checks: int,
    loss_finite: bool,
    amp_enabled: bool,
    gradient_stats_steps: int | None = None,
    gradient_stats_nonfinite_steps: int | None = None,
) -> dict[str, object]:
    """Return JSON-safe profile validity evidence for the measured optimizer steps.

    Enabled GradScaler found-inf flags check AMP gradients after unscale. When
    AMP is disabled, complete pre-clip GradStats observations provide the
    gradient-finiteness check instead.
    """
    counts_consistent = (
        optimizer_attempts > 0
        and optimizer_completed_updates + amp_skipped_updates == optimizer_attempts
    )
    losses_complete = loss_checks == optimizer_attempts
    if amp_enabled and counts_consistent:
        gradients_finite = amp_skipped_updates == 0
        gradient_check_basis = "GradScaler found-inf flags after unscale, before optimizer step"
    elif (
        not amp_enabled
        and counts_consistent
        and (gradient_stats_steps == optimizer_attempts)
        and (gradient_stats_nonfinite_steps is not None)
    ):
        gradients_finite = gradient_stats_nonfinite_steps == 0
        gradient_check_basis = "complete pre-clip GradStats gradient norms"
    else:
        gradients_finite = None
        gradient_check_basis = (
            "GradScaler found-inf flags after unscale, before optimizer step"
            if amp_enabled
            else "not checked because GradScaler is disabled and GradStats is incomplete"
        )
    valid = (
        counts_consistent
        and amp_skipped_updates == 0
        and losses_complete
        and loss_finite
        and (gradients_finite is True)
    )
    return {
        "optimizer_attempts": optimizer_attempts,
        "optimizer_completed_updates": optimizer_completed_updates,
        "amp_skipped_updates": amp_skipped_updates,
        "grad_scaler_enabled": amp_enabled,
        "loss_checks": loss_checks,
        "loss_finite": loss_finite if losses_complete else None,
        "gradient_finite": gradients_finite,
        "gradient_check_basis": gradient_check_basis,
        "gradient_stats_steps": gradient_stats_steps,
        "gradient_stats_nonfinite_steps": gradient_stats_nonfinite_steps,
        "counts_consistent": counts_consistent,
        "valid_for_throughput": valid,
    }


class StageTimer:
    """Per-stage GPU times from CUDA events plus CPU wait on the loader (research profiling only).

    ``mark(name)`` records an event closing the stage that started at the previous mark; nothing is
    synchronized until ``report()``, so profiling does not add host round trips to the step.
    """

    def __init__(self, enabled: bool, warmup: int = 0) -> None:
        """Init."""
        self.enabled, self.warmup = (enabled, warmup)
        self.steps, self.cur, self.loader_wait = ([], None, [])
        self.syncs = 0
        self._warning_capture = None
        self._warnings = []

    def start(self) -> None:
        """Start."""
        if self.enabled and len(self.steps) == self.warmup:
            import warnings

            self._warning_capture = warnings.catch_warnings(record=True)
            self._warnings = self._warning_capture.__enter__()
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode(os.environ.get("WNN_SYNC_DEBUG", "warn"))
        if self.enabled:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.cur = {"_events": [("start", ev)], "_wall": time.perf_counter()}

    def mark(self, name: str) -> None:
        """Mark."""
        if self.enabled and self.cur is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.cur["_events"].append((name, ev))

    def stop(self) -> None:
        """Stop."""
        if self.enabled and self.cur is not None:
            self.steps.append(self.cur)
            self.cur = None

    def report(self, warmup: int, batch: int) -> dict[str, float]:
        """Synchronize the bounded profile and summarize stage timing and warnings."""
        if self._warning_capture is not None:
            self._warning_capture.__exit__(None, None, None)
            self._warning_capture = None
        self.syncs = sum("synchroniz" in str(record.message) for record in self._warnings)
        torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        rows = []
        for st in self.steps[warmup:]:
            ev = st["_events"]
            acc = {}
            for (_, a), (name, b) in zip(ev[:-1], ev[1:], strict=True):
                acc[name] = acc.get(name, 0.0) + a.elapsed_time(b)
            acc["gpu_total"] = ev[0][1].elapsed_time(ev[-1][1])
            rows.append(acc)
        ends = [st["_wall"] for st in self.steps[warmup:]]
        step_wall = np.median(np.diff(ends)) * 1000 if len(ends) > 2 else float("nan")
        keys = sorted({k for r in rows for k in r})
        med = {k: float(np.median([r.get(k, 0.0) for r in rows])) for k in keys}
        med["loader_wait"] = (
            float(np.median(self.loader_wait[warmup:])) * 1000 if self.loader_wait else 0.0
        )
        med["step_wall"] = float(step_wall)
        med["seq_per_s"] = batch / (step_wall / 1000) if step_wall == step_wall else float("nan")
        med["syncs_per_step"] = self.syncs / max(1, len(self.steps) - warmup)
        return med
