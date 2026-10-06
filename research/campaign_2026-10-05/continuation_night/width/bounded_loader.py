from __future__ import annotations

import mmap
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import wnn.data as data
from wnn.data import FEATURE_COLUMNS, TARGET_COLUMNS, Sequence


class BoundedSequenceFile:
    """Decode one sequence in small row batches, without retaining Arrow buffers.

    The caller still receives the exact complete float32 sequence. No rows,
    features, labels, masks, or random-number draws are changed. Parquet metadata
    is reused, while each batch iterator is released at the sequence boundary.
    Synchronous loading bounds the number of readers and queued decoded samples.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader = pq.ParquetFile(self.path, pre_buffer=False)
        names = self._reader.schema_arrow.names
        self.has_targets = "t0" in names
        self.has_mask = "is_scored" in names
        self._length = self._reader.num_row_groups
        # Resolve the active data contract when opened. This matches SequenceFile:
        # small synthetic integration fixtures patch wnn.data.SEQUENCE_LENGTH,
        # while production source remains pinned to exactly 20,000 rows.
        self.expected_rows = data.SEQUENCE_LENGTH
        if self.expected_rows <= 0:
            raise ValueError("sequence row count must be positive")

    def __len__(self) -> int:
        return self._length

    def read(self, index: int, *, targets: bool = True, mask: bool = True) -> Sequence:
        """Return unchanged dense rows, then release transient decode allocations."""
        want_targets = targets and self.has_targets
        want_mask = mask and self.has_mask
        columns = ["seq_ix", *FEATURE_COLUMNS]
        if want_targets:
            columns.extend(TARGET_COLUMNS)
        if want_mask:
            columns.append("is_scored")
        expected_rows = self.expected_rows
        x = np.empty((expected_rows, len(FEATURE_COLUMNS)), np.float32)
        y = np.empty((expected_rows, 2), np.float32) if want_targets else None
        scored = np.empty(expected_rows, bool) if want_mask else None
        offset, sequence_id = 0, None
        reader = self._reader
        metadata = reader.metadata.row_group(index)
        if metadata.num_rows != expected_rows:
            raise ValueError(f"row group {index} has {metadata.num_rows} rows")
        for batch in reader.iter_batches(
            batch_size=512,
            row_groups=[index],
            columns=columns,
            use_threads=False,
        ):
            end = offset + batch.num_rows
            if end > expected_rows:
                raise ValueError(f"decoded row group {index} exceeds {expected_rows} rows")
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            if sequence_id is None:
                sequence_id = int(ids[0])
            if not np.all(ids == sequence_id):
                raise ValueError(f"row group {index} contains multiple sequence ids")
            for column_index in range(len(FEATURE_COLUMNS)):
                x[offset:end, column_index] = batch.column(column_index + 1).to_numpy(
                    zero_copy_only=False
                )
            next_column = 1 + len(FEATURE_COLUMNS)
            if y is not None:
                for target in range(2):
                    y[offset:end, target] = batch.column(next_column + target).to_numpy(
                        zero_copy_only=False
                    )
                next_column += 2
            if scored is not None:
                scored[offset:end] = batch.column(next_column).to_numpy(zero_copy_only=False)
            offset = end
            del batch
        # Advice is restricted to byte ranges actually read from this group.
        ranges = []
        for column_index in range(metadata.num_columns):
            column = metadata.column(column_index)
            if column.path_in_schema not in columns:
                continue
            starts = [column.data_page_offset]
            if column.has_dictionary_page:
                starts.append(column.dictionary_page_offset)
            ranges.append((min(starts), column.total_compressed_size))
        pa.default_memory_pool().release_unused()
        if hasattr(os, "posix_fadvise"):
            descriptor = os.open(self.path, os.O_RDONLY)
            try:
                for start, length in ranges:
                    os.posix_fadvise(descriptor, start, length, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
        if offset != expected_rows or sequence_id is None:
            raise ValueError("decoded sequence length changed")
        return Sequence(sequence_id, x, y, scored)


def release_auxiliary_mappings(dataset: object) -> None:
    """Discard mmap residency only after target/propensity arrays were copied.

    The mappings remain valid, read-only and reusable. No write or truncation is
    performed. The reviewed trainer calls this only at the return boundary of
    its unaugmented sequence loader, after torch inputs own ordinary arrays.
    """
    arrays = [getattr(dataset, "_prop", None), getattr(dataset, "_quote", None)]
    arrays.extend(array for array, _ in (getattr(dataset, "_teach", None) or []))
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.madvise(mmap.MADV_DONTNEED)
