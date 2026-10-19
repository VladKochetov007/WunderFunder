"""Fixed R7 AdamW training loop and epoch diagnostics."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from wnn.gmetric import GlobalAccumulator
from wnn.nn import weighted_mse_loss

from .model_math import (
    ADAM_EPS,
    BATCH_SIZE,
    BETAS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MeanVarianceMLP,
    bounded_gaussian_loss,
    epoch_permutation,
    state_fingerprint,
)


def _check_finite_model(model: MeanVarianceMLP) -> None:
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("nonfinite model parameter after optimizer update")


def _check_finite_optimizer(optimizer: torch.optim.AdamW) -> None:
    """Reject any nonfinite floating-point AdamW state after an update."""
    for state in optimizer.state.values():
        for value in state.values():
            if (
                isinstance(value, Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all().item()
            ):
                raise FloatingPointError("nonfinite optimizer state after update")


def _optimizer(model: MeanVarianceMLP) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )


def _fit_diagnostics(
    model: MeanVarianceMLP, features: Tensor, targets: Tensor, arm: str
) -> dict[str, float]:
    model.eval()
    valid = torch.ones(len(targets), dtype=torch.bool)
    with torch.no_grad():
        if arm == "control":
            mean = model.forward_mean(features)
            own_loss = weighted_mse_loss(mean, targets, valid)
        else:
            mean, logvar = model.forward_all(features)
            own_loss = bounded_gaussian_loss(mean, logvar, targets)
        wmse = weighted_mse_loss(mean, targets, valid)
        if not torch.isfinite(mean).all().item() or not torch.isfinite(own_loss).item():
            raise FloatingPointError("nonfinite fitting diagnostic")
        accumulator = GlobalAccumulator()
        accumulator.add(
            targets.detach().cpu().numpy(),
            mean.detach().cpu().numpy(),
            valid.numpy(),
        )
    result = accumulator.result()
    return {
        "fit_wmse": float(wmse.item()),
        "fit_own_objective": float(own_loss.item()),
        "fit_wp": float(result["wp"]),
        "fit_t0_wp": float(result["t0"]),
        "fit_t1_wp": float(result["t1"]),
    }


def _train_epoch(
    model: MeanVarianceMLP,
    optimizer: torch.optim.AdamW,
    features: Tensor,
    targets: Tensor,
    arm: str,
    epoch: int,
    counters: dict[str, int],
) -> dict[str, float | int]:
    """Perform one fixed shuffled epoch, failing on any nonfinite batch."""
    model.train()
    order = epoch_permutation(len(features), epoch)
    losses: list[float] = []
    norms: list[float] = []
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start : start + BATCH_SIZE]
        x_batch = features.index_select(0, indices)
        y_batch = targets.index_select(0, indices)
        counters["attempted_updates"] += 1
        optimizer.zero_grad(set_to_none=True)
        if arm == "control":
            mean = model.forward_mean(x_batch)
            loss = weighted_mse_loss(mean, y_batch, torch.ones(len(y_batch), dtype=torch.bool))
        else:
            mean, logvar = model.forward_all(x_batch)
            loss = bounded_gaussian_loss(mean, logvar, y_batch)
        if not torch.isfinite(loss).item():
            counters["nonfinite_count"] += 1
            raise FloatingPointError(f"nonfinite {arm} loss at epoch {epoch}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError(f"nonfinite {arm} gradient at epoch {epoch}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm).item():
            counters["nonfinite_count"] += 1
            raise FloatingPointError(f"nonfinite {arm} preclip gradient norm at epoch {epoch}")
        norms.append(float(norm.item()))
        optimizer.step()
        try:
            _check_finite_model(model)
            _check_finite_optimizer(optimizer)
        except FloatingPointError:
            counters["nonfinite_count"] += 1
            raise
        counters["successful_updates"] += 1
        losses.append(float(loss.detach().item()))
    return {
        "mean_batch_objective": float(np.mean(losses)),
        "mean_preclip_grad_norm": float(np.mean(norms)),
        "max_preclip_grad_norm": float(np.max(norms)),
        "attempted_updates_in_epoch": len(losses),
    }


def train_arm(
    model: MeanVarianceMLP,
    features: np.ndarray,
    targets: np.ndarray,
    arm: str,
    on_epoch: Callable[[dict[str, object]], None],
    memory_guard: Callable[[str], dict[str, int | str]],
    counters: dict[str, int],
) -> dict[str, object]:
    """Train one fixed R7 arm for12epochs and retain each epoch diagnostic."""
    if arm not in ("control", "heteroskedastic"):
        raise ValueError("arm must be control or heteroskedastic")
    if features.ndim != 2 or features.shape[1] != 224 or targets.shape != (len(features), 2):
        raise ValueError("invalid R7 fit feature or target shape")
    x_tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
    optimizer = _optimizer(model)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_fingerprint = state_fingerprint(model)
    arm_started_wall, arm_started_cpu = time.perf_counter(), time.process_time()
    epochs: list[dict[str, object]] = []
    for epoch in range(EPOCHS):
        guard = memory_guard(f"{arm}_epoch_{epoch + 1}_start")
        epoch_started_wall, epoch_started_cpu = time.perf_counter(), time.process_time()
        update = _train_epoch(model, optimizer, x_tensor, y_tensor, arm, epoch, counters)
        diagnostic = _fit_diagnostics(model, x_tensor, y_tensor, arm)
        record: dict[str, object] = {
            "event": "epoch_complete",
            "arm": arm,
            "epoch": epoch + 1,
            "attempted_updates": counters["attempted_updates"],
            "successful_updates": counters["successful_updates"],
            "nonfinite_count": counters["nonfinite_count"],
            "learning_rate": LEARNING_RATE,
            "epoch_wall_seconds": time.perf_counter() - epoch_started_wall,
            "epoch_cpu_seconds": time.process_time() - epoch_started_cpu,
            "memory_guard": guard,
            **update,
            **diagnostic,
        }
        epochs.append(record)
        on_epoch(record)
    _check_finite_model(model)
    final_fingerprint = state_fingerprint(model)
    variance_head_unchanged = all(
        torch.equal(initial_state[key], model.state_dict()[key])
        for key in ("logvar_head.weight", "logvar_head.bias")
    )
    if arm == "control":
        if not variance_head_unchanged:
            raise RuntimeError("control variance head changed despite being excluded from loss")
        for key in ("logvar_head.weight", "logvar_head.bias"):
            if model.get_parameter(key).grad is not None:
                raise RuntimeError("control variance head unexpectedly received a gradient")
    return {
        "arm": arm,
        "initial_state_sha256": initial_fingerprint,
        "final_state_sha256": final_fingerprint,
        "variance_head_unchanged": variance_head_unchanged,
        "epochs_completed": len(epochs),
        "attempted_updates": counters["attempted_updates"],
        "successful_updates": counters["successful_updates"],
        "nonfinite_count": counters["nonfinite_count"],
        "wall_seconds": time.perf_counter() - arm_started_wall,
        "cpu_seconds": time.process_time() - arm_started_cpu,
        "epochs": epochs,
    }
