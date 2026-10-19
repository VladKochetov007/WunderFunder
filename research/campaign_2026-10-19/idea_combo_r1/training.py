"""One fixed R7 Gaussian-NLL predictor on the frozen R15 sparse features."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from idea_screens_r7.model_math import bounded_gaussian_loss
from idea_screens_r7.model_training import _check_finite_model, _check_finite_optimizer, _optimizer
from idea_screens_r15.representation import R15Predictor, fingerprint
from idea_screens_r15.training import (
    BATCH_SIZE,
    CLIP_NORM,
    EPOCHS,
    _check_gradients,
    _epoch_order,
    initialize_predictor_pair,
)

EXPECTED_INITIAL_SHA256 = "bce617ce88cb42d0c2cce82f3ec479c90909adc2b9e8adcd4cb6dc1d50963405"
EXPECTED_UPDATES = 756


def initialize_candidate() -> tuple[R15Predictor, str]:
    """Create the exact R15 seed-19073 pair initializer and retain one model."""
    model, _, initial_sha = initialize_predictor_pair()
    if initial_sha != EXPECTED_INITIAL_SHA256 or fingerprint(model) != initial_sha:
        raise RuntimeError("combo predictor initialization differs from pinned R15 state")
    if any(
        parameter.dtype is not torch.float32 or parameter.device.type != "cpu"
        for parameter in model.parameters()
    ):
        raise RuntimeError("combo predictor must initialize on CPU in float32")
    return model, initial_sha


def train_candidate(
    model: R15Predictor,
    features: np.ndarray,
    targets: np.ndarray,
    on_epoch: Callable[[dict[str, object]], None],
    memory_guard: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    """Train exactly 12 seed-19074 shuffled epochs with the unchanged R7 NLL."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
    if x.shape != (31_808, 256) or y.shape != (31_808, 2):
        raise ValueError("combo supervised fit matrix dimensions differ")
    if not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
        raise ValueError("combo supervised fit matrix is nonfinite")
    optimizer = _optimizer(model)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_sha = fingerprint(model)
    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    epoch_records: list[dict[str, object]] = []
    for epoch in range(EPOCHS):
        memory = memory_guard(f"sparse_ema_nll_epoch_{epoch + 1}_start")
        epoch_wall, epoch_cpu = time.perf_counter(), time.process_time()
        model.train()
        order = _epoch_order(len(x), 19_074, epoch)
        objectives: list[float] = []
        preclip_norms: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            xb, yb = x.index_select(0, indices), y.index_select(0, indices)
            counters["attempted_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            mean, raw_logvar = model.forward_all(xb)
            loss = bounded_gaussian_loss(mean, raw_logvar, yb)
            if not torch.isfinite(loss).item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError("nonfinite combo R7 Gaussian NLL")
            loss.backward()
            try:
                _check_gradients(model)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if not torch.isfinite(norm).item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError("nonfinite combo predictor preclip gradient norm")
            optimizer.step()
            try:
                _check_finite_model(model)
                _check_finite_optimizer(optimizer)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            counters["successful_updates"] += 1
            objectives.append(float(loss.detach().item()))
            preclip_norms.append(float(norm.detach().item()))
        event: dict[str, object] = {
            "event": "epoch_complete",
            "arm": "sparse_ema_nll",
            "epoch": epoch + 1,
            **counters,
            "mean_batch_nll": float(np.mean(objectives)),
            "mean_preclip_gradient_norm": float(np.mean(preclip_norms)),
            "max_preclip_gradient_norm": float(np.max(preclip_norms)),
            "epoch_wall_seconds": time.perf_counter() - epoch_wall,
            "epoch_cpu_seconds": time.process_time() - epoch_cpu,
            "memory_guard": memory,
        }
        epoch_records.append(event)
        on_epoch(event)
    _check_finite_model(model)
    variance_unchanged = all(
        torch.equal(initial_state[key], model.state_dict()[key])
        for key in ("logvar_head.weight", "logvar_head.bias")
    )
    if counters != {
        "attempted_updates": EXPECTED_UPDATES,
        "successful_updates": EXPECTED_UPDATES,
        "nonfinite_count": 0,
    }:
        raise RuntimeError(f"combo fixed update budget differs: {counters}")
    return {
        "initial_sha256": initial_sha,
        "final_sha256": fingerprint(model),
        "variance_head_zero_initialized": bool(
            torch.count_nonzero(initial_state["logvar_head.weight"]).item() == 0
            and torch.count_nonzero(initial_state["logvar_head.bias"]).item() == 0
        ),
        "variance_head_changed_under_nll": not variance_unchanged,
        **counters,
        "epochs_completed": len(epoch_records),
        "row_exposures": int(len(x) * EPOCHS),
        "epoch_records": epoch_records,
        "wall_seconds_including_setup": time.perf_counter() - started_wall,
        "cpu_seconds_including_setup": time.process_time() - started_cpu,
    }
