from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch
from torch.utils.data import DataLoader

from wnn.resources import cpu_threads


class CudaPrefetcher:
    """Iterate a DataLoader of pinned CPU batches, copying batch b+1 to the GPU on a side stream while
    batch b is being consumed. Yields device tensors plus the CPU copy of ``scored``."""

    def __init__(self, loader: DataLoader, device: torch.device, timer: Any = None) -> None:
        """Init."""
        self.loader, self.device, self.timer = (loader, device, timer)
        self.stream = torch.cuda.Stream()

    def _load(
        self, it: Iterator[Sequence[torch.Tensor]]
    ) -> tuple[list[torch.Tensor], torch.Tensor] | None:
        """Load."""
        t = time.perf_counter()
        try:
            x, y, scored, *rest = next(it)
        except StopIteration:
            return None
        if self.timer is not None and self.timer.enabled:
            self.timer.loader_wait.append(time.perf_counter() - t)
        with torch.cuda.stream(self.stream):
            dev = [v.to(self.device, non_blocking=True) for v in (x, y, scored, *rest)]
        return (dev, scored)

    def __iter__(self) -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        """Iter."""
        it = iter(self.loader)
        nxt = self._load(it)
        torch.set_num_threads(cpu_threads())
        while nxt is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            dev, scored_cpu = nxt
            for v in dev:
                v.record_stream(torch.cuda.current_stream())
            nxt = self._load(it)
            yield (dev, scored_cpu)

    def __len__(self) -> int:
        """Len."""
        return len(self.loader)
