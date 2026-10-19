"""Fixed reconstruction and supervised fit loops for the R15 pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from idea_screens_r7.model_training import (
    _check_finite_model,
    _check_finite_optimizer,
    _fit_diagnostics,
    _optimizer,
)
from torch import Tensor, nn

from wnn.nn import weighted_mse_loss

from .representation import DictionaryAutoencoder, R15Predictor, fingerprint

BATCH_SIZE = 512
EPOCHS = 12
CLIP_NORM = 1.0


def _epoch_order(size: int, seed: int, epoch: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator)


def _check_gradients(model: nn.Module) -> None:
    if any(
        p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()
    ):
        raise FloatingPointError("nonfinite gradient before optimizer update")


def train_dictionary(
    model: DictionaryAutoencoder,
    fit_inputs: np.ndarray,
    seed: int,
    on_epoch: Callable[[dict[str, object]], None],
    counters: dict[str, int],
) -> dict[str, object]:
    """Fit one dictionary for exactly 12 fixed AdamW reconstruction epochs."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    inputs = torch.from_numpy(np.ascontiguousarray(fit_inputs, dtype=np.float32))
    if inputs.ndim != 2 or inputs.shape[1] != 112 or not torch.isfinite(inputs).all().item():
        raise ValueError("dictionary fit inputs must be finite (N,112)")
    optimizer = _optimizer(model)
    initial_hash = fingerprint(model)
    epochs: list[dict[str, object]] = []
    for epoch in range(EPOCHS):
        model.train()
        order = _epoch_order(len(inputs), seed, epoch)
        losses: list[float] = []
        norms: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = inputs.index_select(0, order[start : start + BATCH_SIZE])
            counters["attempted_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            _, rebuilt, _ = model(batch)
            loss = torch.mean((rebuilt - batch).square())
            if not torch.isfinite(loss).item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError("nonfinite dictionary reconstruction objective")
            loss.backward()
            try:
                _check_gradients(model)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if not torch.isfinite(norm).item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError("nonfinite dictionary preclip gradient norm")
            optimizer.step()
            try:
                model.normalize_decoder_columns()
                _check_finite_model(model)
                _check_finite_optimizer(optimizer)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            counters["successful_updates"] += 1
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        event = {
            "stage": "dictionary",
            "epoch": epoch + 1,
            "attempted_updates": counters["attempted_updates"],
            "successful_updates": counters["successful_updates"],
            "nonfinite_count": counters["nonfinite_count"],
            "mean_batch_reconstruction_mse": float(np.mean(losses)),
            "mean_preclip_gradient_norm": float(np.mean(norms)),
            "max_preclip_gradient_norm": float(np.max(norms)),
        }
        epochs.append(event)
        on_epoch(event)
    norms = torch.linalg.vector_norm(model.decoder.weight.detach(), dim=0)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-6, rtol=0.0):
        raise RuntimeError("decoder columns are not unit normalized after fit")
    return {
        "initial_sha256": initial_hash,
        "final_sha256": fingerprint(model),
        "epochs": epochs,
        "attempted_updates": counters["attempted_updates"],
        "successful_updates": counters["successful_updates"],
        "nonfinite_count": counters["nonfinite_count"],
        "row_exposures": int(len(inputs) * EPOCHS),
        "fit_wall_seconds_including_tensor_and_optimizer_setup": time.perf_counter() - started_wall,
        "fit_cpu_seconds_including_tensor_and_optimizer_setup": time.process_time() - started_cpu,
        "final_decoder_column_norms": norms.cpu().numpy().astype(float).tolist(),
    }


def initialize_predictor_pair() -> tuple[R15Predictor, R15Predictor, str]:
    """Clone one seeded predictor initialization and zero the unused variance head."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(19073)
        initial = R15Predictor()
    with torch.no_grad():
        initial.logvar_head.weight.zero_()
        initial.logvar_head.bias.zero_()
    first, second = R15Predictor(), R15Predictor()
    state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
    first.load_state_dict(state)
    second.load_state_dict(state)
    fingerprints = {fingerprint(initial), fingerprint(first), fingerprint(second)}
    if len(fingerprints) != 1:
        raise RuntimeError("downstream predictors do not share their initialization")
    return first, second, fingerprints.pop()


def train_predictor(
    model: R15Predictor,
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    seed: int,
    on_epoch: Callable[[dict[str, object]], None],
    counters: dict[str, int],
) -> dict[str, object]:
    """Train one fixed mean head by canonical clipped-label weighted MSE."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    x = torch.from_numpy(np.ascontiguousarray(fit_features, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(fit_targets, dtype=np.float32))
    if x.shape != (len(y), 256) or y.shape[1:] != (2,):
        raise ValueError("invalid R15 supervised matrix dimensions")
    if not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
        raise ValueError("nonfinite supervised fit values")
    optimizer = _optimizer(model)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_hash = fingerprint(model)
    epochs: list[dict[str, object]] = []
    for epoch in range(EPOCHS):
        model.train()
        order = _epoch_order(len(x), seed, epoch)
        losses: list[float] = []
        norms: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            xb, yb = x.index_select(0, indices), y.index_select(0, indices)
            counters["attempted_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            mean = model.forward_mean(xb)
            loss = weighted_mse_loss(mean, yb, torch.ones(len(yb), dtype=torch.bool))
            if not torch.isfinite(loss).item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError("nonfinite canonical WMSE")
            loss.backward()
            try:
                _check_gradients(model)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if not torch.isfinite(norm).item():
                counters["nonfinite_count"] += 1
                raise FloatingPointError("nonfinite predictor preclip gradient norm")
            optimizer.step()
            try:
                _check_finite_model(model)
                _check_finite_optimizer(optimizer)
            except FloatingPointError:
                counters["nonfinite_count"] += 1
                raise
            counters["successful_updates"] += 1
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        diagnostic = _fit_diagnostics(model, x, y, "control")
        event = {
            "stage": "predictor",
            "epoch": epoch + 1,
            "attempted_updates": counters["attempted_updates"],
            "successful_updates": counters["successful_updates"],
            "nonfinite_count": counters["nonfinite_count"],
            "mean_batch_wmse": float(np.mean(losses)),
            "mean_preclip_gradient_norm": float(np.mean(norms)),
            "max_preclip_gradient_norm": float(np.max(norms)),
            **diagnostic,
        }
        epochs.append(event)
        on_epoch(event)
    unchanged = all(
        torch.equal(initial_state[name], model.state_dict()[name])
        for name in ("logvar_head.weight", "logvar_head.bias")
    )
    if not unchanged or any(
        model.get_parameter(name).grad is not None
        for name in ("logvar_head.weight", "logvar_head.bias")
    ):
        raise RuntimeError("unused variance head changed or received gradients")
    return {
        "initial_sha256": initial_hash,
        "final_sha256": fingerprint(model),
        "unused_variance_head_unchanged": unchanged,
        "epochs": epochs,
        "attempted_updates": counters["attempted_updates"],
        "successful_updates": counters["successful_updates"],
        "nonfinite_count": counters["nonfinite_count"],
        "row_exposures": int(len(x) * EPOCHS),
        "fit_wall_seconds_including_tensor_and_optimizer_setup": time.perf_counter() - started_wall,
        "fit_cpu_seconds_including_tensor_and_optimizer_setup": time.process_time() - started_cpu,
    }
