"""Read ordered complete sequences for batched frozen-parent inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wnn.data import SEQUENCE_LENGTH, Sequence, SequenceFile

PARENT_BATCH_SEQUENCES = 8


def read_parent_input_batch(
    files: Mapping[str, SequenceFile],
    row_refs: list[dict[str, object]],
    start: int,
    *,
    fitting: bool,
) -> tuple[list[Sequence], np.ndarray]:
    """Read up to eight complete groups in frozen order and stack only raw inputs."""
    if start < 0 or start >= len(row_refs):
        raise ValueError("parent batch start is outside the source-qualified membership")
    stop = min(start + PARENT_BATCH_SEQUENCES, len(row_refs))
    sequences = []
    for ref in row_refs[start:stop]:
        split = str(ref["split"])
        if split not in files:
            raise ValueError(f"parent batch has no open reader for split {split!r}")
        sequence = files[split].read(int(ref["row_group"]), targets=fitting, mask=fitting)
        if not fitting and (sequence.targets is not None or sequence.scored is not None):
            raise ValueError("evaluation parent reads must omit targets and score masks")
        if sequence.features.shape != (SEQUENCE_LENGTH, 112):
            raise ValueError("parent batch encountered a noncanonical sequence feature shape")
        sequences.append(sequence)
    raw_batch = np.stack([sequence.features for sequence in sequences]).astype(
        np.float32, copy=False
    )
    return sequences, raw_batch


def fit_sequence_targets_and_mask(sequence: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Require fitting labels and default an absent score mask to all false.

    This matches the canonical ``Sequences`` dataset behavior for train rows,
    where the parquet has targets but no ``is_scored`` column.
    """
    if sequence.targets is None:
        raise ValueError("fit sequence read omitted canonical targets")
    target = np.asarray(sequence.targets, dtype=np.float32)
    score_mask = (
        np.asarray(sequence.scored, dtype=np.bool_)
        if sequence.scored is not None
        else np.zeros(len(sequence.features), dtype=np.bool_)
    )
    return target, score_mask
