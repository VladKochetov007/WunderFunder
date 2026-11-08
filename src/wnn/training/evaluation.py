from __future__ import annotations

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wnn.data import market_class
from wnn.metric import block_aggregate_score, block_scores, score
from wnn.outcomes import readout
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    chunk: int,
    device: torch.device,
    *,
    environment: TrainingEnvironment | None = None,
) -> np.ndarray:
    """Predict."""
    environment = environment if environment is not None else TrainingEnvironment()
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True), environment=environment)
    state, outs = (None, [])
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast(
            device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and (not getattr(model, "_inference_fp32", False)),
        ):
            out, state = environment.forward_chunk(model, inputs, slice(s, s + chunk), state)
        outs.append(
            readout(out, *environment.readout.tables, tilted=environment.readout.tilted)
            if environment.readout.tables is not None
            else out[..., :2].float()
        )
    return torch.cat(outs, dim=1).cpu().numpy()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    chunk: int,
    device: torch.device,
    *,
    environment: TrainingEnvironment | None = None,
    mask_policy: str = "is_scored",
) -> dict[str, Any]:
    """Evaluate pooled WP and clipped-target WMSE from one prediction pass.

    ``all_after_warmup`` is for a report-only fitting-cohort diagnostic whose
    source has no official ``is_scored`` mask. Official validation keeps the
    default ``is_scored`` selection.
    """
    if mask_policy not in ("is_scored", "all_after_warmup"):
        raise ValueError("mask_policy must be is_scored or all_after_warmup")
    environment = environment if environment is not None else TrainingEnvironment()
    preds, ys, masks, classes = ([], [], [], [])
    for x, y, m in loader:
        preds.append(
            predict(model, x, chunk, device, environment=environment)
            if environment.prediction is None
            else environment.prediction(model, x, chunk, device)
        )
        ys.append(y.numpy())
        selected = (
            m.numpy().copy()
            if mask_policy == "is_scored"
            else np.ones(m.shape, dtype=bool)
        )
        selected[:, : environment.warmup] = False
        masks.append(selected)
        classes += [market_class(t) for t in y.numpy()]
    p, y, m = (np.concatenate(preds), np.concatenate(ys), np.concatenate(masks))
    res = score(y, p, m)
    res.update(_weighted_mse_metrics(y, p, m))
    legacy = (
        block_aggregate_score(y, p, m)
        if res["eligible_blocks"]
        else {"weighted_pearson": None, "t0": None, "t1": None}
    )
    res.update(
        global_t0=res["t0"],
        global_t1=res["t1"],
        per_seq_wp=legacy["weighted_pearson"],
        block_t0=legacy["t0"],
        block_t1=legacy["t1"],
    )
    res["frac_abs_pred_gt2"] = float((np.abs(p[m]) > 2).mean())
    per = block_scores(y, p, m).mean(axis=1)
    cls = np.array(classes)
    res["by_class"] = {
        int(c): score(y[cls == c], p[cls == c], m[cls == c])["weighted_pearson"]
        for c in np.unique(cls)
    }
    res["block_by_class"] = {
        int(c): round(float(np.nanmean(per[cls == c])), 4)
        if np.isfinite(per[cls == c]).any()
        else None
        for c in np.unique(cls)
    }
    return res


def evaluate_train_prefix(
    model: torch.nn.Module,
    loader: DataLoader,
    chunk: int,
    device: torch.device,
    *,
    environment: TrainingEnvironment | None = None,
) -> dict[str, Any]:
    """Score the fixed fitting prefix without changing model, buffer or RNG state."""
    environment = environment if environment is not None else TrainingEnvironment()
    with _preserve_evaluation_state(model, device):
        model.eval()
        with torch.no_grad():
            return evaluate(
                model,
                loader,
                chunk,
                device,
                environment=environment,
                mask_policy="all_after_warmup",
            )


def _weighted_mse_metrics(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Compute pooled WMSE with clipped labels, magnitude weights and raw predictions."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    if y.ndim != 3 or y.shape[-1] != 2 or p.shape != y.shape or selected.shape != y.shape[:2]:
        raise ValueError("expected (B, T, 2) targets/predictions and (B, T) mask")
    if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
        raise ValueError("nonfinite selected target or prediction")
    clipped = np.clip(y.astype(np.float64), -2.0, 2.0)
    row_weights = np.abs(clipped) * selected[..., None]
    squared_error = (p.astype(np.float64) - clipped) ** 2
    numerator = np.sum(row_weights * squared_error, axis=(0, 1), dtype=np.float64)
    denominator = np.sum(row_weights, axis=(0, 1), dtype=np.float64)
    per_target = np.divide(
        numerator,
        denominator,
        out=np.zeros(2, dtype=np.float64),
        where=denominator > 0,
    )
    total_weight = float(denominator.sum())
    wmse = 0.0 if total_weight == 0 else float(numerator.sum() / total_weight)
    return {"wmse": wmse, "wmse_t0": float(per_target[0]), "wmse_t1": float(per_target[1])}


@contextmanager
def _preserve_evaluation_state(model: torch.nn.Module, device: torch.device) -> Iterator[None]:
    """Restore global RNG streams, per-module modes and registered buffers afterward."""
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()  # noqa: NPY002 - preserve the legacy process-wide stream
    torch_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda" and torch.cuda.is_initialized()
        else None
    )
    module_modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in module_modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)  # noqa: NPY002 - restore the legacy process-wide stream
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
