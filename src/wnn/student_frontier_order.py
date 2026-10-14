"""Source-qualified fit membership for the frozen student ladder profiles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wnn.training_sampling import EpochOrderSampler

SourceRow = tuple[str, int]


def source_qualified_members(
    train_groups: Sequence[int], valid_groups: Sequence[int], positions: Sequence[int]
) -> list[SourceRow]:
    """Map train+valid dataset positions to source-local row groups.

    The source is determined only by the dataset offset. Row-group integers are
    allowed to overlap between the two parquet sources.
    """
    train_count = len(train_groups)
    total = train_count + len(valid_groups)
    members: list[SourceRow] = []
    for raw_position in positions:
        position = int(raw_position)
        if not 0 <= position < total:
            raise ValueError(f"dataset position is outside the train+valid dataset: {position}")
        if position < train_count:
            members.append(("train", int(train_groups[position])))
        else:
            members.append(("valid", int(valid_groups[position - train_count])))
    return members


def seeded_epoch_positions(size: int, seed: int, batch_size: int, epoch: int) -> list[int]:
    """Return the production sampler's full deterministic order for one epoch."""
    sampler = EpochOrderSampler(size, seed, batch_size=batch_size, drop_last=True)
    sampler.set_epoch(epoch)
    return [int(position) for position in sampler]


def source_qualified_target_positions(
    train_target_groups: Sequence[int],
    valid_target_groups: Sequence[int],
    expected_train_groups: Sequence[int],
    expected_valid_groups: Sequence[int],
) -> dict[SourceRow, int]:
    """Validate every target row in its own source namespace and return row offsets."""
    target_indices: dict[SourceRow, int] = {}
    for source, actual, expected in (
        ("train", train_target_groups, expected_train_groups),
        ("valid", valid_target_groups, expected_valid_groups),
    ):
        values = [int(group) for group in actual]
        required = [int(group) for group in expected]
        if values != required:
            raise ValueError(f"{source} target rows do not preserve the full frozen fit order")
        for offset, group in enumerate(values):
            key = (source, group)
            if key in target_indices:
                raise ValueError(f"duplicate source-qualified target row: {key}")
            target_indices[key] = offset
    expected_members = {
        *(('train', int(group)) for group in expected_train_groups),
        *(('valid', int(group)) for group in expected_valid_groups),
    }
    if target_indices.keys() != expected_members:
        raise ValueError("target cache rows do not map one-to-one to the source-qualified fit groups")
    return target_indices


def positions_sha256(positions: Sequence[int]) -> str:
    """Return a stable little-endian int64 digest of the complete sampled order."""
    import hashlib

    encoded = np.asarray(positions, dtype="<i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def members_sha256(members: Sequence[SourceRow]) -> str:
    """Return a stable digest that retains train/valid namespace identity."""
    import hashlib

    encoded = "\n".join(f"{source}:{group}" for source, group in members).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
