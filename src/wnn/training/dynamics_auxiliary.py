from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

from wnn.data import N_FEATURES
from wnn.kalman_group_features import FEATURE_GROUPS
from wnn.splits import file_sha256
from wnn.training.environment import TrainingEnvironment

HORIZONS = (16, 256)
GROUP_TARGETS = 2
TARGETS_PER_HORIZON = len(FEATURE_GROUPS) * GROUP_TARGETS
DYNAMICS_TARGET_WIDTH = len(HORIZONS) * TARGETS_PER_HORIZON
LOG_EPSILON = 1e-8
TARGET_STD_FLOOR = 1e-3
HUBER_DELTA = 1.0
T = TypeVar("T")


def _block_local_window_sum(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Sum each full window without subtracting long-running FP32 prefixes.

    ``values`` has shape ``(batch, time, channels)`` and is nonnegative for
    the dynamics-energy caller. Windows are split at multiples of their own
    length: an unaligned window is the current block's suffix plus the next
    block's prefix. Both sums start inside the requested window, so an earlier
    large pulse cannot erase later small positive values.
    """
    if values.ndim != 3:
        raise ValueError("window sums need batch, time, and channel dimensions")
    if window_size < 1:
        raise ValueError("window size must be positive")
    batch_size, time_length, channels = values.shape
    window_count = time_length - window_size + 1
    if window_count <= 0:
        return values.new_empty((batch_size, 0, channels))

    block_count = (time_length + window_size - 1) // window_size
    padded_length = block_count * window_size
    if padded_length != time_length:
        values = F.pad(values, (0, 0, 0, padded_length - time_length))
    blocks = values.reshape(batch_size, block_count, window_size, channels)
    prefix = blocks.cumsum(dim=2).reshape(batch_size, padded_length, channels)
    suffix = torch.flip(
        torch.flip(blocks, dims=(2,)).cumsum(dim=2), dims=(2,)
    ).reshape(batch_size, padded_length, channels)

    starts = torch.arange(window_count, device=values.device)
    left = suffix.index_select(1, starts)
    right_indices = starts + window_size - 1
    right = prefix.index_select(1, right_indices)
    crosses_block = (starts.remainder(window_size) != 0)[None, :, None]
    right = torch.where(crosses_block, right, torch.zeros_like(right))
    return left + right


def dynamics_targets_torch(
    raw: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the two future-change summaries per raw feature group and horizon.

    Returns a ``(batch, time, 52)`` target tensor and a ``(batch, time, 2)``
    horizon-valid mask. Target values are meaningful only where the mask is true.
    """
    if raw.ndim != 3 or raw.shape[-1] < N_FEATURES:
        raise ValueError("dynamics targets need a batch of sequences with 112 raw features")
    batch_size, sequence_length = raw.shape[:2]
    if valid_rows.shape != (batch_size, sequence_length):
        raise ValueError("dynamics target row mask must match the raw sequence shape")
    if input_mean.shape != (N_FEATURES,) or input_std.shape != (N_FEATURES,):
        raise ValueError("dynamics target normalization must have 112 raw-feature values")
    if lengths is not None and lengths.shape != (batch_size,):
        raise ValueError("compressed sequence lengths must have one value per sequence")

    normalized = (raw[..., :N_FEATURES].float() - input_mean) / input_std
    changes = normalized[:, 1:] - normalized[:, :-1]
    grouped_step_energy = torch.stack(
        [changes[..., start:stop].square().mean(dim=-1) for _, start, stop in FEATURE_GROUPS],
        dim=-1,
    )
    positions = torch.arange(sequence_length, device=raw.device)
    targets_by_horizon: list[torch.Tensor] = []
    masks_by_horizon: list[torch.Tensor] = []
    for horizon in HORIZONS:
        window_count = max(sequence_length - horizon, 0)
        endpoint_change = normalized[:, horizon:] - normalized[:, :-horizon]
        energy_sum = _block_local_window_sum(grouped_step_energy, horizon)
        group_targets: list[torch.Tensor] = []
        for group_index, (_, start, stop) in enumerate(FEATURE_GROUPS):
            width = horizon * (stop - start)
            mean_change = endpoint_change[..., start:stop].sum(dim=-1) / width
            log_energy = torch.log(energy_sum[..., group_index] / horizon + LOG_EPSILON)
            group_target = torch.stack((mean_change, log_energy), dim=-1)
            if window_count < sequence_length:
                group_target = F.pad(group_target, (0, 0, 0, sequence_length - window_count))
            group_targets.append(group_target)
        targets_by_horizon.append(torch.cat(group_targets, dim=-1))
        horizon_valid = valid_rows.bool() & (positions[None, :] + horizon < sequence_length)
        if lengths is not None:
            horizon_valid = horizon_valid & (positions[None, :] + horizon < lengths[:, None])
        masks_by_horizon.append(horizon_valid)
    return torch.cat(targets_by_horizon, dim=-1), torch.stack(masks_by_horizon, dim=-1)


def dynamics_huber_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid_horizons: torch.Tensor
) -> torch.Tensor:
    """Return mean Huber loss over every valid horizon, group, and target channel."""
    if prediction.shape != target.shape or prediction.shape[-1] != DYNAMICS_TARGET_WIDTH:
        raise ValueError("dynamics predictions and targets must have matching width 52")
    if valid_horizons.shape != (*prediction.shape[:2], len(HORIZONS)):
        raise ValueError("dynamics horizon mask has an unexpected shape")
    numerator = prediction.float().new_zeros(())
    denominator = prediction.float().new_zeros(())
    for index in range(len(HORIZONS)):
        start = index * TARGETS_PER_HORIZON
        stop = start + TARGETS_PER_HORIZON
        mask = valid_horizons[..., index].float()
        error = F.huber_loss(
            prediction[..., start:stop].float(),
            target[..., start:stop].float(),
            delta=HUBER_DELTA,
            reduction="none",
        )
        numerator = numerator + (error * mask[..., None]).sum()
        denominator = denominator + mask.sum() * TARGETS_PER_HORIZON
    return numerator / denominator.clamp_min(1.0)


def fit_dynamics_target_normalization(
    model: torch.nn.Module,
    train_split: str,
    train_groups: np.ndarray,
    valid_groups: np.ndarray | None,
    warmup: int,
    *,
    environment: TrainingEnvironment | None = None,
) -> dict[str, Any]:
    """Fit target mean/std from exact fitting groups using feature-only reads."""
    environment = environment if environment is not None else TrainingEnvironment()
    target_mean = getattr(model, "dynamics_target_mean", None)
    target_std = getattr(model, "dynamics_target_std", None)
    if target_mean is None or target_std is None:
        raise ValueError("the model must be constructed with --dynamics-aux-head")

    members = {(train_split, int(group)) for group in train_groups}
    if valid_groups is not None:
        members.update(("valid", int(group)) for group in valid_groups)
    if not members:
        raise ValueError("dynamics target normalization needs at least one fitting sequence")
    ordered_members = sorted(members)
    member_hash = hashlib.sha256(
        json.dumps(ordered_members, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    center = model.mean[:N_FEATURES].detach().cpu()
    scale = model.std[:N_FEATURES].detach().cpu()
    sums = np.zeros(DYNAMICS_TARGET_WIDTH, dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    counts = np.zeros(DYNAMICS_TARGET_WIDTH, dtype=np.int64)
    files = {
        split: environment.sequence_file(environment.root / "datasets" / f"{split}.parquet")
        for split in {split for split, _ in ordered_members}
    }
    for split, group in ordered_members:
        sequence = files[split].read(group, targets=False, mask=False)
        raw = torch.from_numpy(sequence.features).unsqueeze(0)
        valid = torch.arange(raw.shape[1])[None, :] >= warmup
        target, target_mask = dynamics_targets_torch(raw, center, scale, valid)
        target_np = target.squeeze(0).numpy().astype(np.float64, copy=False)
        mask_np = target_mask.squeeze(0).numpy()
        for horizon_index in range(len(HORIZONS)):
            start = horizon_index * TARGETS_PER_HORIZON
            stop = start + TARGETS_PER_HORIZON
            values = target_np[mask_np[:, horizon_index], start:stop]
            sums[start:stop] += values.sum(axis=0)
            squared_sums[start:stop] += np.square(values).sum(axis=0)
            counts[start:stop] += len(values)

    if np.any(counts == 0):
        raise ValueError("dynamics target normalization has no valid targets for a horizon")
    mean = sums / counts
    variance = np.maximum(squared_sums / counts - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), TARGET_STD_FLOOR)
    target_mean.copy_(torch.as_tensor(mean, device=target_mean.device, dtype=target_mean.dtype))
    target_std.copy_(torch.as_tensor(std, device=target_std.device, dtype=target_std.dtype))
    group_registry = [
        {"name": name, "start": start, "stop": stop} for name, start, stop in FEATURE_GROUPS
    ]
    return {
        "version": 1,
        "groups_source": "wnn.kalman_group_features.FEATURE_GROUPS",
        "groups": group_registry,
        "horizons": list(HORIZONS),
        "target_order": "per horizon, then registry group, mean standardized change followed by log mean squared standardized change",
        "change_definition": "z[t+k]-z[t+k-1] for k=1..h; z=(raw-model.mean[:112])/model.std[:112]",
        "boundary": f"target state t is valid when t >= {warmup} and t+h < sequence length; raw fit reads are complete, unthinned sequences",
        "target_normalization": "per-output fitting-membership mean and population standard deviation, floored at 1e-3",
        "fit_members": {
            "train": sorted(int(group) for split, group in ordered_members if split == "train"),
            "valid": sorted(int(group) for split, group in ordered_members if split == "valid"),
        },
        "fit_members_sha256": member_hash,
        "target_counts": counts.tolist(),
        "input_normalization": "existing fitting-only model mean/std, raw channels 0:112",
        "fit_reader": "SequenceFile.read(targets=False, mask=False); training targets and scoring masks are not loaded",
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "module_source_sha256": file_sha256(
            environment.source_root / "src/wnn/training/dynamics_auxiliary.py"
        ),
    }


def strip_dynamics_auxiliary_state(state: Mapping[str, T]) -> dict[str, T]:
    """Remove the training-only head and normalization buffers from an export state."""
    return {
        name: value
        for name, value in state.items()
        if name != "dynamics_target_mean"
        and name != "dynamics_target_std"
        and not name.startswith("dynamics_aux_head.")
    }
