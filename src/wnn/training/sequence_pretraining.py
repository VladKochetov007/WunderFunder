from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.training.sequence_data import SequenceData, SequenceRecords
from wnn.training.sequence_fit_utils import SequenceItem
from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_normalization import load_or_compute_full_fit_moments
from wnn.training.sequence_pretraining_contract import (
    FUTURE_INNOVATION_CONTRACT,
    FUTURE_INNOVATION_HORIZONS,
    FUTURE_INNOVATION_WIDTH,
)

MAX_HORIZON = max(FUTURE_INNOVATION_HORIZONS)
TARGET_START = WARMUP
TARGET_STOP = SEQUENCE_LENGTH - MAX_HORIZON
TARGET_ROWS = TARGET_STOP - TARGET_START


def future_innovation_matrix(features: np.ndarray) -> np.ndarray:
    """Build concatenated causal-input difference targets for the common valid row set."""
    values = np.asarray(features)
    if values.shape != (SEQUENCE_LENGTH, N_FEATURES) or not np.isfinite(values).all():
        raise ValueError("future-innovation source must be finite raw112 full-sequence input")
    current = values[TARGET_START:TARGET_STOP].astype(np.float64, copy=False)
    targets = [
        values[TARGET_START + horizon : TARGET_STOP + horizon].astype(np.float64, copy=False)
        - current
        for horizon in FUTURE_INNOVATION_HORIZONS
    ]
    result = np.concatenate(targets, axis=-1)
    if result.shape != (TARGET_ROWS, FUTURE_INNOVATION_WIDTH) or not np.isfinite(result).all():
        raise ValueError("future-innovation targets have an invalid shape or value")
    return result


def standardized_future_innovations(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    """Standardize each horizon/feature from fitting-only moments and zero masked rows."""
    if mean.shape != (FUTURE_INNOVATION_WIDTH,) or std.shape != mean.shape:
        raise ValueError("future-innovation moments must have width 336")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("future-innovation moments must be finite with positive scales")
    target = np.zeros((SEQUENCE_LENGTH, FUTURE_INNOVATION_WIDTH), dtype=np.float32)
    target[TARGET_START:TARGET_STOP] = (
        (future_innovation_matrix(features) - mean.astype(np.float64)) / std.astype(np.float64)
    ).astype(np.float32)
    return torch.from_numpy(target)


def prepare_future_innovation_records(data: SequenceData, data_root: Path) -> dict[str, object]:
    """Fit or load train-only horizon moments and replace the fit view with feature-only records."""
    base = data.fit
    members = list(base.members)
    if not members or any(source != "train" for source, _ in members):
        raise ValueError(
            "future-input pretraining requires nonempty official train-only membership"
        )
    source_table = data.provenance.get("sources")
    if not isinstance(source_table, dict):
        raise ValueError("future-input moments require source fingerprints")
    source_names = sorted({source for source, _ in members})
    source_fingerprints = {
        f"datasets/{source}.parquet": source_table[f"datasets/{source}.parquet"]
        for source in source_names
    }
    split_sha256 = str(data.provenance["split_sha256"])
    contract = FUTURE_INNOVATION_CONTRACT | {
        "sequence_rows": SEQUENCE_LENGTH,
        "row_start": TARGET_START,
        "row_stop_exclusive": TARGET_STOP,
        "fit_members_sha256": membership_sha256(members),
        "source_fingerprints": source_fingerprints,
    }

    def read_targets(source: str, group: int) -> np.ndarray:
        """Read raw features only; labels and score masks are not requested."""
        return future_innovation_matrix(_read_features(base, source, group))

    moments = load_or_compute_full_fit_moments(
        data_root / "cache" / "sequence_future_innovations",
        source_fingerprints,
        split_sha256,
        members,
        read_targets,
        feature_count=FUTURE_INNOVATION_WIDTH,
        identity_extra=contract,
    )
    data.fit = FutureInnovationRecords(base, moments.mean, moments.std)
    data.provenance["future_input_innovations"] = contract | {
        "normalization_identity_sha256": moments.identity_sha256,
        "moments_sha256": moments.moments_sha256,
        "fit_sequence_count": len(members),
        "target_rows_per_sequence": TARGET_ROWS,
        "label_values_read": False,
        "validation_members_used": False,
    }
    return data.provenance["future_input_innovations"]


class FutureInnovationRecords(SequenceRecords):
    """Reuse source-qualified sequence storage while exposing only input-derived pretext targets."""

    def __init__(self, base: SequenceRecords, mean: np.ndarray, std: np.ndarray) -> None:
        self.members = base.members
        self.hint_dim = FUTURE_INNOVATION_WIDTH
        self.cache = None
        self.hint_members = None
        self.training_mask = "all_after_warmup"
        self.sources = base.sources
        self.positions = base.positions
        self._innovation_mean = mean.copy()
        self._innovation_std = std.copy()

    def __getitem__(self, index: int) -> SequenceItem:
        """Return raw inputs and future innovations, with labels and teachers set to zero."""
        if index == -1:
            return {
                "x": torch.zeros(SEQUENCE_LENGTH, N_FEATURES),
                "y": torch.zeros(SEQUENCE_LENGTH, 2),
                "mask": torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool),
                "teacher": torch.zeros(SEQUENCE_LENGTH, 2),
                "hints": torch.zeros(SEQUENCE_LENGTH, FUTURE_INNOVATION_WIDTH),
            }
        if not 0 <= index < len(self.members):
            raise IndexError("future-innovation sequence index is out of range")
        source, group = self.members[index]
        features = _read_features(self, source, group)
        mask = torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool)
        mask[TARGET_START:TARGET_STOP] = True
        return {
            "x": torch.from_numpy(np.asarray(features, dtype=np.float32).copy()),
            "y": torch.zeros(SEQUENCE_LENGTH, 2),
            "mask": mask,
            "teacher": torch.zeros(SEQUENCE_LENGTH, 2),
            "hints": standardized_future_innovations(
                features, self._innovation_mean, self._innovation_std
            ),
        }


def _read_features(records: SequenceRecords, source: str, group: int) -> np.ndarray:
    """Read one source-qualified sequence without opening target or scoring columns."""
    dataset = records.sources[source]
    if dataset._file is None:
        dataset._file = dataset.environment.sequence_file(dataset.path)
    return dataset._file.read(group, targets=False, mask=False).features
