from __future__ import annotations

import json

import torch

from wnn.propensity import PropensityNet
from wnn.propensity_progress import ProgressJournal, UpdateProgress
from wnn.training.reporting import emit


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def _optimizer_state_finite(optimizer: torch.optim.Optimizer) -> bool:
    """Reject non-finite floating point values in updated AdamW state."""
    return all(
        _finite(value)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )


def _record_nonfinite(
    *,
    stage: str,
    epoch: int,
    batch: int,
    chunk_start: int,
    attempt_id: str,
    journal: ProgressJournal,
    cumulative: UpdateProgress,
    epoch_progress: UpdateProgress,
    optimizer_step_called: bool = False,
) -> None:
    cumulative.reject_nonfinite(stage)
    epoch_progress.reject_nonfinite(stage)
    event = {
        "event": "optimizer_update_rejected",
        "epoch": epoch,
        "batch": batch,
        "chunk_start": chunk_start,
        "stage": stage,
        "optimizer_step_called": optimizer_step_called,
        "optimizer_progress": cumulative.snapshot(),
    }
    journal.append(attempt_id, event)
    emit("PROPENSITY_NONFINITE", json.dumps(event, sort_keys=True), flush=True)


def _fit_update(
    *,
    model: PropensityNet,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    target: torch.Tensor,
    hidden: torch.Tensor | None,
    chunk_start: int,
    warmup: int,
    epoch: int,
    batch: int,
    attempt_id: str,
    journal: ProgressJournal,
    cumulative: UpdateProgress,
    epoch_progress: UpdateProgress,
) -> torch.Tensor:
    learning_rates = _learning_rates(optimizer)
    for progress in (cumulative, epoch_progress):
        progress.start(learning_rates[0])
    journal.append(
        attempt_id,
        {
            "event": "optimizer_update_started",
            "epoch": epoch,
            "batch": batch,
            "chunk_start": chunk_start,
            "learning_rates": learning_rates,
            "optimizer_attempts": cumulative.optimizer_attempts,
        },
    )
    if not _finite(inputs) or not _finite(target):
        _record_nonfinite(
            stage="inputs",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
        )
        raise FloatingPointError("strict propensity input batch is not finite")
    with torch.autocast(inputs.device.type, enabled=False):
        logits, next_hidden = model(inputs, hidden)
    if not _finite(logits):
        _record_nonfinite(
            stage="logits",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
        )
        raise FloatingPointError("strict propensity logits are not finite")
    keep = (
        torch.arange(chunk_start, chunk_start + logits.shape[1], device=logits.device)
        .ge(warmup)[None]
        .expand_as(logits)
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float()[keep], target[keep])
    if not _finite(loss):
        _record_nonfinite(
            stage="loss",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
        )
        raise FloatingPointError("strict propensity loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if any(not _finite(gradient) for gradient in gradients):
        _record_nonfinite(
            stage="gradients",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
        )
        raise FloatingPointError("strict propensity gradients are not finite")
    if not gradients:
        raise RuntimeError("strict propensity update has no gradients")
    grad_norm_tensor = torch.linalg.vector_norm(
        torch.stack([gradient.norm(2) for gradient in gradients]), 2
    )
    if not _finite(grad_norm_tensor):
        _record_nonfinite(
            stage="gradient_norm",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
        )
        raise FloatingPointError("strict propensity preclip gradient norm is not finite")
    grad_norm = float(grad_norm_tensor)
    for progress in (cumulative, epoch_progress):
        progress.step_called()
    optimizer.step()
    if any(not _finite(parameter) for parameter in model.parameters()):
        _record_nonfinite(
            stage="weights",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
            optimizer_step_called=True,
        )
        raise FloatingPointError("strict propensity weights are not finite after optimizer step")
    if not _optimizer_state_finite(optimizer):
        _record_nonfinite(
            stage="optimizer_state",
            epoch=epoch,
            batch=batch,
            chunk_start=chunk_start,
            attempt_id=attempt_id,
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch_progress,
            optimizer_step_called=True,
        )
        raise FloatingPointError("strict propensity optimizer state is not finite after step")
    loss_value = float(loss.detach())
    for progress in (cumulative, epoch_progress):
        progress.complete(loss_value, grad_norm)
    event = {
        "event": "optimizer_update_completed",
        "epoch": epoch,
        "batch": batch,
        "chunk_start": chunk_start,
        "loss": loss_value,
        "learning_rates": learning_rates,
        "preclip_grad_norm": grad_norm,
        "optimizer_progress": cumulative.snapshot(),
    }
    journal.append(attempt_id, event)
    emit("PROPENSITY_UPDATE", json.dumps(event, sort_keys=True), flush=True)
    return next_hidden.detach()
