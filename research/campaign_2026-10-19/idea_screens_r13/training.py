"""Fixed R13 arm training using R7's canonical optimizer and minibatch update."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from idea_screens_r7.model_math import state_fingerprint
from idea_screens_r7.model_training import (
    _check_finite_model,
    _check_finite_optimizer,
    _fit_diagnostics,
    _optimizer,
    _train_epoch,
)

from .model_arms import R13MeanVarianceMLP

FIT_ROWS = 31_808
EXPECTED_EPOCHS = 12
EXPECTED_UPDATES = 756


def train_arm(
    model: R13MeanVarianceMLP,
    features: np.ndarray,
    targets: np.ndarray,
    arm: str,
    on_epoch: Callable[[dict[str, object]], None],
    memory_guard: Callable[[str], dict[str, int | str]],
) -> dict[str, object]:
    """Train one fixed arm for twelve full shuffled passes over the fit rows."""
    if arm not in ("zero", "current_bypass", "warmup99"):
        raise ValueError("unknown R13 arm")
    x_values = np.asarray(features, dtype=np.float32)
    y_values = np.asarray(targets, dtype=np.float32)
    if x_values.shape != (FIT_ROWS, 336) or y_values.shape != (FIT_ROWS, 2):
        raise ValueError("R13 training requires exactly 31,808 aligned 336-input rows")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("R13 fit arrays must be finite")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_fingerprint = state_fingerprint(model)
    memory_guard(f"{arm}_before_tensor_and_optimizer")
    x_tensor = torch.from_numpy(np.ascontiguousarray(x_values))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y_values))
    optimizer = _optimizer(model)
    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    epochs: list[dict[str, object]] = []
    for epoch in range(EXPECTED_EPOCHS):
        guard = memory_guard(f"{arm}_epoch_{epoch + 1}_start")
        epoch_wall, epoch_cpu = time.perf_counter(), time.process_time()
        update = _train_epoch(model, optimizer, x_tensor, y_tensor, "control", epoch, counters)
        fit = _fit_diagnostics(model, x_tensor, y_tensor, "control")
        record: dict[str, object] = {
            "event": "epoch_complete",
            "arm": arm,
            "epoch": epoch + 1,
            "learning_rate": 0.001,
            "attempted_updates": counters["attempted_updates"],
            "successful_updates": counters["successful_updates"],
            "nonfinite_count": counters["nonfinite_count"],
            "epoch_wall_seconds": time.perf_counter() - epoch_wall,
            "epoch_cpu_seconds": time.process_time() - epoch_cpu,
            "memory_guard": guard,
            **update,
            **fit,
        }
        epochs.append(record)
        on_epoch(record)
    _check_finite_model(model)
    _check_finite_optimizer(optimizer)
    variance_unchanged = all(
        torch.equal(initial_state[key], model.state_dict()[key])
        for key in ("logvar_head.weight", "logvar_head.bias")
    )
    if not variance_unchanged or any(
        model.get_parameter(key).grad is not None
        for key in ("logvar_head.weight", "logvar_head.bias")
    ):
        raise RuntimeError("R13 unused variance head changed or received a gradient")
    zero_side_unchanged = bool(torch.count_nonzero(model.trunk.weight[:, 224:]).item() == 0)
    if arm == "zero" and not zero_side_unchanged:
        raise RuntimeError("R13 zero side block learned nonzero weights")
    if counters != {
        "attempted_updates": EXPECTED_UPDATES,
        "successful_updates": EXPECTED_UPDATES,
        "nonfinite_count": 0,
    }:
        raise RuntimeError(f"R13 arm did not complete the fixed update budget: {counters}")
    return {
        "arm": arm,
        "initial_state_sha256": initial_fingerprint,
        "final_state_sha256": state_fingerprint(model),
        "variance_head_unchanged": variance_unchanged,
        "variance_head_grad_none": True,
        "zero_side_weights_unchanged": zero_side_unchanged if arm == "zero" else None,
        "epochs_completed": len(epochs),
        **counters,
        "row_exposures": len(x_values) * EXPECTED_EPOCHS,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "epochs": epochs,
    }
