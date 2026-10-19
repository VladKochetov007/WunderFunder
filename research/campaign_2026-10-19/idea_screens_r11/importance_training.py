"""Fixed-step paired initialization and corrected minibatch training."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from idea_screens_r7.model_math import (
    ADAM_EPS,
    BETAS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MeanVarianceMLP,
    state_fingerprint,
)
from torch import Tensor

from .importance_math import BATCH_SIZE, DRAW_ROUNDS, correction_factors, sample_indices

INITIALIZATION_SEED = 19041
MAX_GRAD_NORM = 1.0
EXPECTED_UPDATES = 756


def initialize_pair(
    seed: int = INITIALIZATION_SEED,
) -> tuple[MeanVarianceMLP, MeanVarianceMLP, dict[str, str]]:
    """Create two byte-identical CPU models with a zero, unused variance head."""
    if seed < 0:
        raise ValueError("initialization seed must be nonnegative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = MeanVarianceMLP()
    with torch.no_grad():
        initial.logvar_head.weight.zero_()
        initial.logvar_head.bias.zero_()
    uniform, importance = MeanVarianceMLP(), MeanVarianceMLP()
    state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
    uniform.load_state_dict(state)
    importance.load_state_dict(state)
    fingerprints = {
        "initial": state_fingerprint(initial),
        "uniform_initial": state_fingerprint(uniform),
        "importance_initial": state_fingerprint(importance),
    }
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError("uniform and importance models do not share initialization")
    return uniform, importance, fingerprints


def _check_finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Reject nonfinite floating-point optimizer state after an update."""
    for state in optimizer.state.values():
        for value in state.values():
            if (
                isinstance(value, Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all()
            ):
                raise FloatingPointError("nonfinite optimizer state after update")


def _check_finite_model(model: MeanVarianceMLP) -> None:
    """Reject nonfinite model parameters after an optimizer update."""
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("nonfinite model parameter after update")


def train_arm(
    *,
    name: str,
    model: MeanVarianceMLP,
    features: Tensor,
    clipped_targets: Tensor,
    weights: Tensor,
    mu_w: float,
    probabilities: np.ndarray,
    uniforms: np.ndarray,
    on_round: Callable[[dict[str, object]], None],
    memory_guard: Callable[[str], dict[str, int | str]],
) -> dict[str, object]:
    """Train one fixed sampler arm for 756 AdamW updates with fail-closed checks."""
    if name not in ("uniform", "importance"):
        raise ValueError("sampler arm must be 'uniform' or 'importance'")
    row_count = len(features)
    if (
        features.ndim != 2
        or features.shape[1] != 224
        or clipped_targets.shape != (row_count, 2)
        or weights.shape != clipped_targets.shape
        or len(probabilities) != row_count
        or len(uniforms) != row_count * DRAW_ROUNDS
    ):
        raise ValueError("training tables, sampler, and fixed draw budget do not match")
    if any(
        tensor.device.type != "cpu" or tensor.dtype != torch.float32
        for tensor in (features, clipped_targets, weights)
    ):
        raise ValueError("R11 training tensors must be CPU FP32")
    if not np.isfinite(mu_w) or mu_w <= 0.0:
        raise ValueError("fixed mu_w must be positive and finite")

    arm_wall, arm_cpu = time.perf_counter(), time.process_time()
    row_indices = sample_indices(uniforms, probabilities)
    corrections = correction_factors(probabilities)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_fingerprint = state_fingerprint(model)
    setup_wall_seconds = time.perf_counter() - arm_wall
    setup_cpu_seconds = time.process_time() - arm_cpu
    training_wall_started, training_cpu_started = time.perf_counter(), time.process_time()
    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    rounds: list[dict[str, object]] = []

    for round_index in range(DRAW_ROUNDS):
        guard = memory_guard(f"{name}_draw_round_{round_index + 1}_start")
        round_start = round_index * row_count
        losses: list[float] = []
        norms: list[float] = []
        clipped_updates = 0
        for batch_start in range(0, row_count, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, row_count)
            sampled = row_indices[round_start + batch_start : round_start + batch_end]
            index_tensor = torch.from_numpy(sampled)
            x_batch = features.index_select(0, index_tensor)
            y_batch = clipped_targets.index_select(0, index_tensor)
            weight_batch = weights.index_select(0, index_tensor)
            correction_batch = torch.from_numpy(corrections[sampled].astype(np.float32))
            counters["attempted_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_mean(x_batch)
            row_numerator = (weight_batch * (prediction - y_batch).square()).sum(dim=1)
            loss = (row_numerator / mu_w * correction_batch).mean()
            if not torch.isfinite(loss):
                counters["nonfinite_count"] += 1
                raise FloatingPointError(
                    f"nonfinite {name} loss at update {counters['attempted_updates']}"
                )
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    counters["nonfinite_count"] += 1
                    raise FloatingPointError(
                        f"nonfinite {name} gradient at update {counters['attempted_updates']}"
                    )
            preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            if not torch.isfinite(preclip_norm):
                counters["nonfinite_count"] += 1
                raise FloatingPointError(f"nonfinite {name} preclip norm")
            norm_value = float(preclip_norm.item())
            clipped_updates += int(norm_value > MAX_GRAD_NORM)
            norms.append(norm_value)
            optimizer.step()
            try:
                _check_finite_model(model)
                _check_finite_optimizer(optimizer)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            counters["successful_updates"] += 1
            losses.append(float(loss.detach().item()))

        round_record: dict[str, object] = {
            "event": "draw_round_complete",
            "arm": name,
            "draw_round": round_index + 1,
            "attempted_updates": counters["attempted_updates"],
            "successful_updates": counters["successful_updates"],
            "nonfinite_count": counters["nonfinite_count"],
            "mean_batch_objective": float(np.mean(losses)),
            "mean_preclip_grad_norm": float(np.mean(norms)),
            "max_preclip_grad_norm": float(np.max(norms)),
            "clipped_update_count": clipped_updates,
            "updates_in_round": len(losses),
            "learning_rate": LEARNING_RATE,
            "memory_guard": guard,
        }
        rounds.append(round_record)
        on_round(round_record)

    if (
        counters["successful_updates"] != EXPECTED_UPDATES
        or counters["attempted_updates"] != EXPECTED_UPDATES
    ):
        raise RuntimeError(f"{name} arm did not complete exactly {EXPECTED_UPDATES} updates")
    variance_unchanged = all(
        torch.equal(initial_state[key], model.state_dict()[key])
        for key in ("logvar_head.weight", "logvar_head.bias")
    )
    variance_grad_free = all(
        model.get_parameter(key).grad is None for key in ("logvar_head.weight", "logvar_head.bias")
    )
    if not variance_unchanged or not variance_grad_free:
        raise RuntimeError(f"{name} unused variance head changed or received a gradient")
    return {
        "arm": name,
        "initial_state_sha256": initial_fingerprint,
        "final_state_sha256": state_fingerprint(model),
        "variance_head_unchanged": variance_unchanged,
        "variance_head_gradient_free": variance_grad_free,
        **counters,
        "draw_rounds_completed": len(rounds),
        "rounds": rounds,
        "sampler_and_optimizer_setup_wall_seconds": setup_wall_seconds,
        "sampler_and_optimizer_setup_cpu_seconds": setup_cpu_seconds,
        "training_wall_seconds": time.perf_counter() - training_wall_started,
        "training_cpu_seconds": time.process_time() - training_cpu_started,
        "wall_seconds": time.perf_counter() - arm_wall,
        "cpu_seconds": time.process_time() - arm_cpu,
    }
