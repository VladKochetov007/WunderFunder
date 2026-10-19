"""Matched initialization, initial gradient scaling, and fixed R12 full-fit updates."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch
from idea_screens_r7.model_math import MeanVarianceMLP, state_fingerprint
from torch import Tensor

from wnn.nn import weighted_mse_loss, weighted_pooled_corr_loss

SEED = 19_051
UPDATES = 128
LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0
GAMMA_BOUNDS = (1e-4, 1e4)


def initialize_pair() -> tuple[MeanVarianceMLP, MeanVarianceMLP, dict[str, str]]:
    """Create byte-identical R7-shaped mean models from the fixed R12 seed."""
    torch.manual_seed(SEED)
    initial = MeanVarianceMLP()
    with torch.no_grad():
        initial.logvar_head.weight.zero_()
        initial.logvar_head.bias.zero_()
    models = (MeanVarianceMLP(), MeanVarianceMLP())
    state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    for model in models:
        model.load_state_dict(state)
    fingerprints = {
        "initial": state_fingerprint(initial),
        "wmse_initial": state_fingerprint(models[0]),
        "pooled_corr_initial": state_fingerprint(models[1]),
    }
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError("R12 arms do not share byte-identical initial state")
    return models[0], models[1], fingerprints


def _fit_loss(
    model: MeanVarianceMLP, features: Tensor, targets: Tensor, valid: Tensor, objective: str
) -> Tensor:
    """Apply one canonical full-fit objective to the common clipped prediction."""
    prediction = model.forward_mean(features).clamp(-2.0, 2.0)
    if objective == "wmse":
        return weighted_mse_loss(prediction, targets, valid)
    if objective == "pooled_corr":
        return weighted_pooled_corr_loss(prediction, targets, valid)
    raise ValueError(f"unknown R12 objective: {objective}")


def initial_gradient_norms(
    model: MeanVarianceMLP, features: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Measure both full-fit initial gradient norms without changing state or .grad."""
    if features.ndim != 2 or features.shape[1] != 224 or targets.shape != (len(features), 2):
        raise ValueError("expected fit-only (N,224) features and matching (N,2) targets")
    if len(features) == 0 or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("initial gradient prepass requires nonempty finite fit rows")
    before = state_fingerprint(model)
    parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("logvar_head.")
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("initial gradient prepass requires clear model .grad fields")
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
    valid = torch.ones(len(y), dtype=torch.bool)
    values: dict[str, float] = {}
    for objective in ("wmse", "pooled_corr"):
        loss = _fit_loss(model, x, y, valid, objective)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
        squared = sum(
            float(gradient.detach().double().square().sum().item()) for gradient in gradients
        )
        values[objective] = math.sqrt(squared)
    if state_fingerprint(model) != before or any(
        parameter.grad is not None for parameter in model.parameters()
    ):
        raise RuntimeError("initial gradient prepass changed parameters or .grad fields")
    if not all(math.isfinite(value) and value > 1e-12 for value in values.values()):
        raise FloatingPointError("initial gradient norms must be finite and greater than 1e-12")
    return values


def scale_gamma(norms: dict[str, float]) -> float:
    """Return the one predeclared initial-norm ratio, failing closed out of range."""
    if set(norms) != {"wmse", "pooled_corr"}:
        raise ValueError("gamma requires exactly the two R12 initial gradient norms")
    control, candidate = norms["wmse"], norms["pooled_corr"]
    if not all(math.isfinite(value) and value > 1e-12 for value in (control, candidate)):
        raise FloatingPointError("gamma inputs must be finite positive norms")
    gamma = control / candidate
    if not math.isfinite(gamma) or not GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1]:
        raise FloatingPointError("initial gradient scale ratio is outside the frozen bounds")
    return gamma


def _check_finite(model: MeanVarianceMLP, optimizer: torch.optim.Optimizer) -> None:
    """Fail if a completed update leaves nonfinite parameters or optimizer state."""
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("nonfinite R12 model parameter after update")
    for state in optimizer.state.values():
        for value in state.values():
            if (
                isinstance(value, Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all().item()
            ):
                raise FloatingPointError("nonfinite R12 optimizer state after update")


def train_arm(
    model: MeanVarianceMLP,
    features: np.ndarray,
    targets: np.ndarray,
    objective: str,
    gamma: float,
    on_update: Callable[[dict[str, object]], None],
    memory_guard: Callable[[str], dict[str, int | str]],
) -> dict[str, object]:
    """Run exactly128 full-fit AdamW updates for one fixed R12 objective arm."""
    if objective not in ("wmse", "pooled_corr") or gamma <= 0 or not math.isfinite(gamma):
        raise ValueError("invalid R12 objective or gamma")
    if features.shape != (31_808, 224) or targets.shape != (31_808, 2):
        raise ValueError("R12 training requires the complete 31,808-row fit panel")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
    valid = torch.ones(len(y), dtype=torch.bool)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    initial_fingerprint = state_fingerprint(model)
    initial_variance = tuple(value.detach().clone() for value in model.logvar_head.parameters())
    counters = {
        "attempted_updates": 0,
        "successful_updates": 0,
        "nonfinite_count": 0,
        "clip_count": 0,
    }
    norms: list[float] = []
    losses: list[float] = []
    for step in range(1, UPDATES + 1):
        if step % 16 == 1:
            memory_guard(f"{objective}_update_{step}_start")
        counters["attempted_updates"] += 1
        optimizer.zero_grad(set_to_none=True)
        loss = _fit_loss(model, x, y, valid, objective)
        if objective == "pooled_corr":
            loss = loss * gamma
        if not torch.isfinite(loss).item():
            counters["nonfinite_count"] += 1
            raise FloatingPointError(f"nonfinite {objective} loss at update {step}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.logvar_head.parameters()):
            raise RuntimeError("unused R12 variance head unexpectedly received gradients")
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError(f"nonfinite {objective} gradient at update {step}")
        preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not torch.isfinite(preclip).item():
            counters["nonfinite_count"] += 1
            raise FloatingPointError(f"nonfinite {objective} preclip norm at update {step}")
        norm = float(preclip.detach().item())
        norms.append(norm)
        counters["clip_count"] += int(norm > GRADIENT_CLIP)
        optimizer.step()
        try:
            _check_finite(model, optimizer)
        except FloatingPointError:
            counters["nonfinite_count"] += 1
            raise
        if any(
            not torch.equal(old, new.detach())
            for old, new in zip(initial_variance, model.logvar_head.parameters(), strict=True)
        ):
            raise RuntimeError("unused R12 variance head changed during mean-only training")
        counters["successful_updates"] += 1
        losses.append(float(loss.detach().item()))
        if step % 16 == 0:
            on_update(
                {
                    "event": "update_block_complete",
                    "arm": objective,
                    "step": step,
                    "loss": losses[-1],
                    "preclip_grad_norm": norm,
                    "clipped": norm > GRADIENT_CLIP,
                    "learning_rate": LEARNING_RATE,
                    **counters,
                }
            )
    if state_fingerprint(model) == initial_fingerprint:
        raise RuntimeError(f"{objective} model did not update")
    if counters["successful_updates"] != UPDATES or counters["attempted_updates"] != UPDATES:
        raise RuntimeError(f"{objective} arm did not complete exactly128 updates")
    if any(parameter.grad is not None for parameter in model.logvar_head.parameters()):
        raise RuntimeError("unused R12 variance head has a final gradient")
    return {
        **counters,
        "updates": UPDATES,
        "row_exposures": UPDATES * len(y),
        "loss_mean": float(np.mean(losses)),
        "loss_last": losses[-1],
        "preclip_norm_mean": float(np.mean(norms)),
        "preclip_norm_max": float(np.max(norms)),
        "preclip_norms": norms,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "initial_state_sha256": initial_fingerprint,
        "final_state_sha256": state_fingerprint(model),
        "variance_head_unchanged": True,
        "variance_head_grad_fields_none": True,
    }
