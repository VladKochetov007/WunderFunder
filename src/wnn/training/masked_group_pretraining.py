from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from wnn.data import EXTRA, I0, I1, N_FEATURES, SequenceFile
from wnn.features import N_EVENT_FEATURES
from wnn.splits import CampaignSplit
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs

_GROUP_SLICES = tuple(
    (part.start, part.stop)
    for part in (
        I0.bid_px,
        I0.ask_px,
        I0.bid_vol,
        I0.ask_vol,
        I0.trade_px,
        I0.trade_vol,
        I1.bid_px,
        I1.ask_px,
        I1.bid_vol,
        I1.ask_vol,
        I1.trade_px,
        I1.trade_vol,
        EXTRA,
    )
)
BASE_INPUT_WIDTH = N_FEATURES + N_EVENT_FEATURES
MASKED_INPUT_WIDTH = BASE_INPUT_WIDTH + N_FEATURES
_GROUP_COLUMNS = tuple(column for start, stop in _GROUP_SLICES for column in range(start, stop))
if N_FEATURES != 112 or N_EVENT_FEATURES != 99 or tuple(range(N_FEATURES)) != _GROUP_COLUMNS:
    raise RuntimeError("masked-group pretraining expects the frozen 112+99 input schema")


def pretraining_source_paths(source_root: Path) -> dict[str, Path]:
    """Return source files whose changes invalidate pretraining checkpoints."""
    relative_paths = (
        "scripts/pretrain_masked_groups.py",
        "src/wnn/data.py",
        "src/wnn/features.py",
        "src/wnn/splits.py",
        "src/wnn/nn.py",
        "src/wnn/training/environment.py",
        "src/wnn/training/inputs.py",
        "src/wnn/training/models.py",
        "src/wnn/training/masked_group_pretraining.py",
        "src/wnn/training/masked_pretraining_epoch.py",
        "src/wnn/training/masked_pretraining_loop.py",
        "src/wnn/training/masked_pretraining_reporting.py",
        "src/wnn/training/masked_pretraining_statistics.py",
    )
    return {name: source_root / name for name in relative_paths}


class FeatureOnlySequences(Dataset[torch.Tensor]):
    """Read fitting sequences without loading target or score-mask columns."""

    def __init__(
        self, members: Sequence[tuple[str, int]], root: Path, environment: TrainingEnvironment
    ) -> None:
        """Bind source-qualified row groups and lazily open each parquet source."""
        self.members = tuple(members)
        self.root = root
        self.environment = environment
        self._sources: dict[str, SequenceFile] = {}

    def __len__(self) -> int:
        """Return the number of fitting sequences."""
        return len(self.members)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read one raw feature matrix, explicitly omitting labels and masks."""
        split, row_group = self.members[index]
        if split not in self._sources:
            path = self.root / "datasets" / f"{split}.parquet"
            self._sources[split] = self.environment.sequence_file(path)
        sequence = self._sources[split].read(row_group, targets=False, mask=False)
        return torch.from_numpy(sequence.features)


def fitting_members(split: CampaignSplit) -> tuple[tuple[str, int], ...]:
    """Return only the campaign manifest's fitting row groups."""
    return tuple(("train", group) for group in split.train_groups) + tuple(
        ("valid", group) for group in split.valid_groups
    )


@torch.no_grad()
def fit_input_statistics(
    members: Sequence[tuple[str, int]],
    sample_indices: Sequence[int],
    *,
    root: Path,
    environment: TrainingEnvironment,
    row_stride: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit canonical raw-plus-event normalization on a fixed fitting-only sample."""
    if not sample_indices or row_stride <= 0:
        raise ValueError("statistics need at least one fitting sequence and a positive row stride")
    sources: dict[str, SequenceFile] = {}
    total = np.zeros(BASE_INPUT_WIDTH, dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0
    for index in sample_indices:
        split, row_group = members[index]
        if split not in sources:
            sources[split] = environment.sequence_file(root / "datasets" / f"{split}.parquet")
        raw = sources[split].read(row_group, targets=False, mask=False).features
        raw_tensor = torch.from_numpy(raw).unsqueeze(0)
        features = build_inputs(raw_tensor, environment=environment)[0, ::row_stride]
        if features.shape[-1] != BASE_INPUT_WIDTH:
            raise ValueError(
                f"expected {BASE_INPUT_WIDTH} canonical inputs, got {features.shape[-1]}"
            )
        values = features.double().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite feature-only input in {split} row group {row_group}")
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += values.shape[0]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


def sample_group_mask(
    batch_size: int,
    sequence_length: int,
    block_rows: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Hide one semantic raw-input group for each contiguous time block."""
    if batch_size <= 0 or sequence_length <= 0 or block_rows <= 0:
        raise ValueError("mask dimensions and block length must be positive")
    block_count = math.ceil(sequence_length / block_rows)
    assignments = torch.randint(
        len(_GROUP_SLICES),
        (batch_size, block_count),
        generator=generator,
        device=device,
    )
    block_ids = torch.arange(sequence_length, device=device) // block_rows
    selected = assignments[:, block_ids]
    mask = torch.zeros((batch_size, sequence_length, N_FEATURES), dtype=torch.bool, device=device)
    for group_index, (start, stop) in enumerate(_GROUP_SLICES):
        mask[..., start:stop] = (selected == group_index).unsqueeze(-1)
    return mask


def build_masked_model(
    *,
    hidden: int,
    device: torch.device,
    environment: TrainingEnvironment,
) -> nn.Module:
    """Build the canonical one-layer GRU with appended mask-indicator channels."""
    args = {
        "arch": "gru",
        "hidden": hidden,
        "encoder_width": hidden,
        "recurrent_width": hidden,
        "readout_width": hidden,
        "head_layers": 1,
        "layers": 1,
        "dropout": 0.0,
        "cell": "gru",
        "enc": "linear",
        "readout_activation": "gelu",
        "numeric_map": "none",
    }
    model = environment.make_model(args, MASKED_INPUT_WIDTH, N_FEATURES).to(device)
    return model


def install_input_statistics(model: nn.Module, mean: np.ndarray, std: np.ndarray) -> None:
    """Install fitting-only canonical statistics and leave binary mask bits unscaled."""
    if mean.shape != (BASE_INPUT_WIDTH,) or std.shape != mean.shape:
        raise ValueError("normalization arrays do not match the canonical input schema")
    with torch.no_grad():
        model.mean[:BASE_INPUT_WIDTH].copy_(torch.from_numpy(mean).to(model.mean))
        model.std[:BASE_INPUT_WIDTH].copy_(torch.from_numpy(std).to(model.std))
        model.mean[BASE_INPUT_WIDTH:].zero_()
        model.std[BASE_INPUT_WIDTH:].fill_(1.0)


def masked_model_input(
    raw: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    environment: TrainingEnvironment,
) -> torch.Tensor:
    """Rebuild causal input features after replacing the selected raw group."""
    raw_mean = mean[:N_FEATURES].view(1, 1, N_FEATURES)
    visible_raw = torch.where(mask, raw_mean, raw)
    features = build_inputs(visible_raw, environment=environment)
    return torch.cat((features, mask.to(dtype=raw.dtype)), dim=-1)


def transferred_backbone_state(
    pretrained_state: dict[str, torch.Tensor],
    cold_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy pretrained input/recurrent weights into a cold 211-input task model."""
    result = {name: value.detach().clone() for name, value in cold_state.items()}
    for name, target in result.items():
        source = pretrained_state.get(name)
        if name.startswith("head."):
            continue
        if name == "inp.weight":
            if source is None or source.ndim != 2 or source.shape[0] != target.shape[0]:
                raise ValueError("pretraining input projection is incompatible with the task model")
            if source.shape[1] != target.shape[1] + N_FEATURES:
                raise ValueError("pretraining input width must append exactly 112 mask channels")
            target.copy_(source[:, : target.shape[1]].to(target))
            continue
        if name in ("mean", "std"):
            if source is None or source.ndim != 1 or source.shape[0] < target.shape[0]:
                raise ValueError("pretraining normalization is incompatible with the task model")
            target.copy_(source[: target.shape[0]].to(target))
            continue
        if source is None or source.shape != target.shape:
            raise ValueError(f"pretraining backbone tensor {name!r} is incompatible")
        target.copy_(source.to(target))
    return result
