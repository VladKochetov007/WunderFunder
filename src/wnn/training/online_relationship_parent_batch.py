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
    """Read up to eight complete groups in order into one raw-input buffer."""
    if start < 0 or start >= len(row_refs):
        raise ValueError("parent batch start is outside the source-qualified membership")
    stop = min(start + PARENT_BATCH_SEQUENCES, len(row_refs))
    raw_batch = np.empty((stop - start, SEQUENCE_LENGTH, 112), dtype=np.float32)
    sequences = []
    for offset, ref in enumerate(row_refs[start:stop]):
        split = str(ref["split"])
        if split not in files:
            raise ValueError(f"parent batch has no open reader for split {split!r}")
        source = files[split].read(int(ref["row_group"]), targets=fitting, mask=fitting)
        if not fitting and (source.targets is not None or source.scored is not None):
            raise ValueError("evaluation parent reads must omit targets and score masks")
        if source.features.shape != (SEQUENCE_LENGTH, 112):
            raise ValueError("parent batch encountered a noncanonical sequence feature shape")
        raw_batch[offset] = source.features
        sequences.append(
            Sequence(
                source.seq_ix,
                raw_batch[offset],
                source.targets,
                source.scored,
            )
        )
        del source
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
