"""Fixed, causal readout fit for the online i1-to-i0 relationship screen."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wnn.conditional_readout import primary_pair_wmse
from wnn.features import N_EVENT_FEATURES, event_features_torch
from wnn.online_relationship import (
    BASE_INPUT_WIDTH,
    MODEL_INPUT_WIDTH,
    OnlineRelationshipLag,
    OnlineRelationshipNormalizer,
    OnlineRelationshipState,
)
from wnn.training.checkpoints import checkpoint_selected
from wnn.training.online_relationship import OnlineRelationshipResidualHead

OnlineRelationshipArm = Literal["A", "B", "C"]
EPOCHS = 3
FIXED_EPOCH = 2
BATCH_ROWS = 8192
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
_LOG = logging.getLogger(__name__)


class OnlineRelationshipSequence(NamedTuple):
    """One ordered complete sequence from the delivery-owned immutable cache."""

    raw: np.ndarray
    target: np.ndarray
    row_weight: np.ndarray
    parent_prediction: np.ndarray


@dataclass(frozen=True)
class OnlineRelationshipFitResult:
    """Fixed-epoch weights, ordered predictions and source-normalization binding."""

    state_dict: dict[str, torch.Tensor]
    evaluation_predictions: tuple[np.ndarray, ...]
    epoch_records: tuple[dict[str, float | int], ...]
    metadata: dict[str, object]


def fit_online_relationship_head(
    fit_sequences: Callable[[], Iterable[OnlineRelationshipSequence]],
    evaluation_sequences: Callable[[], Iterable[OnlineRelationshipSequence]],
    *,
    arm: OnlineRelationshipArm,
    lag: OnlineRelationshipLag,
    seed: int,
    parent_mean: np.ndarray,
    parent_std: np.ndarray,
    relationship_normalizer: OnlineRelationshipNormalizer,
    device: str | torch.device = "cuda",
    progress: Callable[[dict[str, object]], None] | None = None,
) -> OnlineRelationshipFitResult:
    """Fit one frozen three-epoch readout and predict the ordered evaluation stream.

    The factories must replay exact source-qualified sequence order. Rows are
    gathered across complete sequences into 8,192-row batches; the final tail is
    retained. Evaluation labels and row weights are never read by this fitter.
    """
    if arm not in ("A", "B", "C") or seed < 0:
        raise ValueError("online relationship arm or seed is invalid")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("online relationship fitting requires torch default FP32")
    if relationship_normalizer.lag != lag:
        raise ValueError("relationship normalizer lag does not match the requested fit")
    mean = _normalization_vector(parent_mean, "parent mean")
    std = _normalization_vector(parent_std, "parent standard deviation")
    if np.any(std <= 0):
        raise ValueError("parent standard deviations must be positive")
    if len(mean) != BASE_INPUT_WIDTH or len(std) != BASE_INPUT_WIDTH:
        raise ValueError(f"parent normalization must have width {BASE_INPUT_WIDTH}")

    target_device = torch.device(device)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    parent_mean_tensor = torch.tensor(mean, device=target_device)
    parent_std_tensor = torch.tensor(std, device=target_device)
    head = OnlineRelationshipResidualHead().to(target_device)
    parameters = tuple(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    records: list[dict[str, float | int]] = []
    with torch.autocast(device_type=target_device.type, enabled=False):
        for epoch in range(EPOCHS):
            totals = torch.zeros(3, dtype=torch.float32, device=target_device)
            counts = {"sequences": 0, "rows": 0, "batches": 0}
            for features, target, weights, parent in _iter_row_batches(
                fit_sequences,
                arm,
                lag,
                relationship_normalizer,
                parent_mean_tensor,
                parent_std_tensor,
                target_device,
                counts,
            ):
                prediction = head(features, parent)
                loss_inputs = (prediction[None], target[None], weights[None], target[None])
                loss = primary_pair_wmse(*loss_inputs)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("online relationship loss became nonfinite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = clip_grad_norm_(parameters, GRADIENT_CLIP, error_if_nonfinite=True)
                optimizer.step()
                if not torch.stack([torch.isfinite(p).all() for p in parameters]).all().item():
                    raise FloatingPointError("online relationship parameters became nonfinite")
                totals[0] += loss.detach()
                totals[1] += gradient_norm.detach()
                totals[2] = torch.maximum(totals[2], gradient_norm.detach())
            if counts["batches"] == 0:
                raise ValueError("online relationship fitting stream is empty")
            record: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": LEARNING_RATE,
                "mean_batch_primary_pair_wmse": float(totals[0].item() / counts["batches"]),
                "mean_preclip_gradient_norm": float(totals[1].item() / counts["batches"]),
                "max_preclip_gradient_norm": float(totals[2].item()),
                "updates": counts["batches"],
                "finite_updates": counts["batches"],
                "fit_sequences": counts["sequences"],
                "fit_rows": counts["rows"],
            }
            records.append(record)
            _LOG.info(f"online_relationship_fit {json.dumps(record, sort_keys=True)}")
            if progress is not None:
                progress(record)

    if not checkpoint_selected(FIXED_EPOCH, "fixed", FIXED_EPOCH, None, float("-inf")):
        raise RuntimeError("fixed final-epoch checkpoint was not selected")
    predictions = _predict_evaluation(
        head,
        evaluation_sequences,
        arm,
        lag,
        relationship_normalizer,
        parent_mean_tensor,
        parent_std_tensor,
        target_device,
    )
    state_dict = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    metadata: dict[str, object] = {
        "arm": arm,
        "lag": lag,
        "seed": seed,
        "epochs": EPOCHS,
        "fixed_checkpoint_epoch": FIXED_EPOCH,
        "batch_rows": BATCH_ROWS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "parent_mean": mean.tolist(),
        "parent_std": std.tolist(),
        "relationship_normalizer": relationship_normalizer.to_dict(),
        "input_width": MODEL_INPUT_WIDTH,
        "checkpoint_selected": True,
    }
    return OnlineRelationshipFitResult(state_dict, predictions, tuple(records), metadata)


def _iter_row_batches(
    sequence_factory: Callable[[], Iterable[OnlineRelationshipSequence]],
    arm: OnlineRelationshipArm,
    lag: OnlineRelationshipLag,
    normalizer: OnlineRelationshipNormalizer,
    parent_mean: torch.Tensor,
    parent_std: torch.Tensor,
    device: torch.device,
    counts: dict[str, int],
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build ordered fixed-size row batches spanning sequence boundaries."""
    feature_batch = torch.empty((BATCH_ROWS, MODEL_INPUT_WIDTH), device=device)
    target_batch = torch.empty((BATCH_ROWS, 2), device=device)
    weight_batch = torch.empty(BATCH_ROWS, device=device)
    parent_batch = torch.empty((BATCH_ROWS, 2), device=device)
    filled = 0
    for item in sequence_factory():
        features = _sequence_features(
            item.raw, arm, lag, normalizer, parent_mean, parent_std, device
        )
        target_array = _finite_array(item.target, (len(item.raw), 2), "target")
        weight_array = _finite_array(item.row_weight, (len(item.raw),), "row weight")
        if np.any(weight_array < 0):
            raise ValueError("online relationship row weights must be nonnegative")
        target = torch.tensor(target_array, device=device)
        weights = torch.tensor(weight_array, device=device)
        parent = _parent_tensor(item.parent_prediction, len(item.raw), device)
        counts["sequences"] += 1
        counts["rows"] += len(item.raw)
        source_at = 0
        while source_at < len(item.raw):
            take = min(BATCH_ROWS - filled, len(item.raw) - source_at)
            batch_end = filled + take
            source_end = source_at + take
            feature_batch[filled:batch_end].copy_(features[source_at:source_end])
            target_batch[filled:batch_end].copy_(target[source_at:source_end])
            weight_batch[filled:batch_end].copy_(weights[source_at:source_end])
            parent_batch[filled:batch_end].copy_(parent[source_at:source_end])
            filled = batch_end
            source_at = source_end
            if filled == BATCH_ROWS:
                counts["batches"] += 1
                yield feature_batch, target_batch, weight_batch, parent_batch
                filled = 0
    if filled:
        counts["batches"] += 1
        yield (
            feature_batch[:filled],
            target_batch[:filled],
            weight_batch[:filled],
            parent_batch[:filled],
        )


def _predict_evaluation(
    head: OnlineRelationshipResidualHead,
    sequence_factory: Callable[[], Iterable[OnlineRelationshipSequence]],
    arm: OnlineRelationshipArm,
    lag: OnlineRelationshipLag,
    normalizer: OnlineRelationshipNormalizer,
    parent_mean: torch.Tensor,
    parent_std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    """Return one prediction array per evaluation sequence in source order."""
    predictions: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for item in sequence_factory():
            features = _sequence_features(
                item.raw, arm, lag, normalizer, parent_mean, parent_std, device
            )
            parent = _parent_tensor(item.parent_prediction, len(item.raw), device)
            prediction = head(features, parent)
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("online relationship prediction became nonfinite")
            predictions.append(prediction.cpu().numpy().copy())
    if not predictions:
        raise ValueError("online relationship evaluation stream is empty")
    return tuple(predictions)


def _sequence_features(
    raw: np.ndarray,
    arm: OnlineRelationshipArm,
    lag: OnlineRelationshipLag,
    normalizer: OnlineRelationshipNormalizer,
    parent_mean: torch.Tensor,
    parent_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Assemble canonical parent-normalized raw/event features and clipped state."""
    raw_array = _finite_array(raw, (len(raw), 112), "raw features")
    relationship = OnlineRelationshipState(arm, lag, normalizer).transform_compiled(raw_array)
    raw_tensor = torch.tensor(raw_array, device=device)
    context = event_features_torch(raw_tensor[None]).squeeze(0)
    base = torch.cat((raw_tensor, context), dim=-1)
    if base.shape[1] != BASE_INPUT_WIDTH or N_EVENT_FEATURES != 99:
        raise RuntimeError("canonical parent input layout changed")
    normalized_base = (base - parent_mean) / parent_std
    relationship_tensor = torch.tensor(relationship, device=device)
    features = torch.cat((normalized_base, relationship_tensor), dim=-1)
    if features.shape[1] != MODEL_INPUT_WIDTH:
        raise RuntimeError("online relationship model input width changed")
    return features


def _parent_tensor(values: np.ndarray, rows: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(_finite_array(values, (rows, 2), "parent prediction"), device=device)


def _finite_array(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"online relationship {name} must be finite with shape {shape}")
    return array


def _normalization_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return array.copy()
