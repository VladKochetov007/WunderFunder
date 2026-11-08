from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wnn.gmetric import GlobalAccumulator
from wnn.training.sequence_blend import compose_prediction
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import SequenceDataset, collate_sequences
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_monitor_head_metrics import HeadMetrics, TargetMoments
from wnn.training.sequence_precision import forward_context


@torch.inference_mode()
def evaluate_dataset(
    model: nn.Module,
    dataset: SequenceDataset,
    prefix: str,
    chunk_rows: int,
    precision_mode: str,
    context: DistributedContext,
    evaluation_batch_size: int = 1,
    *,
    blend_alpha: float | None = None,
    report_standalone: bool = False,
) -> dict[str, int | float]:
    """Compute primary pooled metrics and optional same-forward component diagnostics."""
    if not len(dataset) or chunk_rows <= 0 or evaluation_batch_size <= 0:
        raise ValueError("monitoring evaluation requires sequences and positive batch/chunk sizes")
    accumulator = GlobalAccumulator()
    standalone = HeadMetrics() if report_standalone else None
    reference = HeadMetrics() if report_standalone and blend_alpha is not None else None
    mixture_metrics = HeadMetrics() if report_standalone and blend_alpha is not None else None
    target_moments = TargetMoments() if report_standalone else None
    squared_error = 0.0
    target_weight = 0.0
    prediction_sum = 0.0
    prediction_square_sum = 0.0
    clipped_count = 0
    selected_values = 0
    selected_rows = 0
    for batch_start in range(0, len(dataset), evaluation_batch_size):
        batch_indices = list(
            range(batch_start, min(batch_start + evaluation_batch_size, len(dataset)))
        )
        batch = collate_sequences(dataset, batch_indices)
        inputs, targets, mask = batch["x"], batch["y"], batch["mask"]
        if inputs.ndim != 3 or targets.shape != (*inputs.shape[:2], 2):
            raise ValueError("monitoring dataset does not contain canonical full sequences")
        state = None
        prediction_chunks: list[torch.Tensor] = []
        for start in range(0, inputs.shape[1], chunk_rows):
            stop = min(start + chunk_rows, inputs.shape[1])
            values = inputs[:, start:stop].to(context.device, non_blocking=True)
            with forward_context(precision_mode, context.device.type):
                prediction, _, next_state = model(values, state)
            state = detach_state(next_state)
            prediction_chunks.append(prediction.float())
        # One host copy per sequence batch; all head diagnostics reuse this forward.
        raw_prediction_tensor = torch.cat(prediction_chunks, dim=1).cpu()
        target = targets.numpy()
        scored = mask.numpy()
        raw_prediction = raw_prediction_tensor.numpy()
        if standalone is not None:
            standalone.add(target, raw_prediction, scored)
        if target_moments is not None:
            target_moments.add(target, scored)
        reference_tensor = batch["teacher"]
        if reference is not None:
            reference.add(target, reference_tensor.numpy(), scored)
        if blend_alpha is not None:
            predicted_tensor = compose_prediction(
                raw_prediction_tensor, reference_tensor, blend_alpha
            )
        else:
            predicted_tensor = raw_prediction_tensor
        predicted = predicted_tensor.numpy()
        if mixture_metrics is not None:
            mixture_metrics.add(target, predicted, scored)
        accumulator.add(target, predicted, scored)
        clipped_target = np.where(scored[..., None], np.clip(target, -2.0, 2.0), 0)
        weight = np.abs(clipped_target) * scored[..., None]
        difference = np.where(scored[..., None], predicted, 0).astype(
            np.float64
        ) - clipped_target.astype(np.float64)
        squared_error += float((weight * np.square(difference)).sum())
        target_weight += float(weight.sum())
        selected = np.broadcast_to(scored[..., None], predicted.shape)
        chosen = predicted[selected].astype(np.float64)
        prediction_sum += float(chosen.sum())
        prediction_square_sum += float(np.square(chosen).sum())
        clipped_count += int((np.abs(chosen) > 2.0).sum())
        selected_values += len(chosen)
        selected_rows += int(scored.sum())
    if selected_values <= 0:
        raise ValueError(f"{prefix} monitoring cohort has no selected rows")
    pooled = accumulator.result()
    mean = prediction_sum / selected_values
    variance = max(0.0, prediction_square_sum / selected_values - mean * mean)
    metrics: dict[str, int | float] = {
        f"{prefix}/loss": squared_error / target_weight if target_weight > 0 else 0.0,
        f"{prefix}/wp": float(pooled["wp"]),
        f"{prefix}/wp/t0": float(pooled["t0"]),
        f"{prefix}/wp/t1": float(pooled["t1"]),
        f"{prefix}/selected_rows": selected_rows,
        f"{prefix}/sequence_count": len(dataset),
        f"{prefix}/prediction_mean": mean,
        f"{prefix}/prediction_std": variance**0.5,
        f"{prefix}/prediction_clipped_fraction": clipped_count / selected_values,
    }
    if target_moments is not None:
        metrics.update(target_moments.metrics(prefix))
    if standalone is not None:
        metrics.update(standalone.metrics(prefix))
    if reference is not None:
        metrics.update(reference.metrics(prefix, "reference"))
    if mixture_metrics is not None:
        metrics.update(mixture_metrics.metrics(prefix, "mixture"))
    return metrics
