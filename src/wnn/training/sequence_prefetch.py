"""Single-batch CPU preparation and pinned asynchronous transfer for CUDA fits."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import torch

from wnn.training.sequence_fit_utils import SequenceDataset, SequenceItem, collate_sequences


@dataclass
class PrefetchedSequenceBatch:
    """One ordered device batch and the pinned host storage backing its copy."""

    indices: tuple[int, ...]
    batch: SequenceItem
    pinned_source: SequenceItem | None
    copy_ready: torch.cuda.Event | None
    device: torch.device

    def wait_on_current_stream(self) -> None:
        """Order model work after H2D and register device tensors on the compute stream."""
        if self.copy_ready is None:
            return
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.copy_ready)
        for tensor in self.batch.values():
            tensor.record_stream(stream)

    def release_pinned_source(self) -> None:
        """Release host storage after its asynchronous transfer has completed."""
        if self.copy_ready is not None:
            self.copy_ready.synchronize()
        self.pinned_source = None


class SequenceBatchPrefetcher(Iterator[PrefetchedSequenceBatch]):
    """Prepare exactly one next full sequence batch on a joined worker thread."""

    def __init__(
        self,
        dataset: SequenceDataset,
        indices: Sequence[int],
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Start one bounded producer; CUDA batches copy over a dedicated stream."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("prefetch batch_size must be a positive integer")
        if device.type not in ("cuda", "cpu"):
            raise ValueError("prefetch device must be CUDA or CPU")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA prefetch requires an available CUDA device")
        self._dataset = dataset
        self._indices = indices
        self._batch_size = batch_size
        self._device = device
        self._starts = iter(range(0, len(indices), batch_size))
        self._copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sequence-prefetch")
        self._future: Future[PrefetchedSequenceBatch] | None = None
        self._closed = False
        self._schedule_next()

    def __iter__(self) -> SequenceBatchPrefetcher:
        return self

    def __next__(self) -> PrefetchedSequenceBatch:
        """Return the next ordered batch and immediately start its single successor."""
        if self._closed or self._future is None:
            raise StopIteration
        future = self._future
        self._future = None
        batch = future.result()
        self._schedule_next()
        return batch

    def _schedule_next(self) -> None:
        """Submit at most one host materialization and H2D operation."""
        if self._closed or self._future is not None:
            return
        try:
            start = next(self._starts)
        except StopIteration:
            return
        batch_indices = tuple(self._indices[start : start + self._batch_size])
        self._future = self._executor.submit(self._prepare, batch_indices)

    def _prepare(self, indices: tuple[int, ...]) -> PrefetchedSequenceBatch:
        """Materialize deterministically, pin only this batch, and enqueue its copy."""
        batch = collate_sequences(self._dataset, indices)
        if self._copy_stream is None:
            return PrefetchedSequenceBatch(indices, batch, None, None, self._device)
        pinned = cast(SequenceItem, {name: tensor.pin_memory() for name, tensor in batch.items()})
        with torch.cuda.device(self._device), torch.cuda.stream(self._copy_stream):
            device_batch = cast(
                SequenceItem,
                {
                    name: tensor.to(device=self._device, non_blocking=True)
                    for name, tensor in pinned.items()
                },
            )
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        return PrefetchedSequenceBatch(indices, device_batch, pinned, event, self._device)

    def close(self) -> None:
        """Cancel queued work, join the producer, and release any unfinished H2D source."""
        if self._closed:
            return
        self._closed = True
        active_exception = sys.exc_info()[0] is not None
        future = self._future
        if future is not None:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        try:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
        finally:
            self._future = None
        if future is not None and not future.cancelled():
            producer_error = future.exception()
            if producer_error is not None and not active_exception:
                raise producer_error

    def __enter__(self) -> SequenceBatchPrefetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
