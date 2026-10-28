from __future__ import annotations

import gc
import math
import statistics
import time
from typing import Literal

import torch

from wnn.galam.config import GALAMConfig
from wnn.galam.state import GALAMState
from wnn.galam.variants import GALAMAdaptation, build_galam_variant


def _update(
    model: GALAMAdaptation,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    state: GALAMState,
) -> tuple[GALAMState, torch.Tensor]:
    """Exercise forward, synthetic WMSE backward, clipping and one complete AdamW update."""
    optimizer.zero_grad(set_to_none=True)
    output, state = model.forward_chunk(x, state)
    weights = target.abs().clamp_min(0.1)
    loss = ((output - target).square() * weights).sum() / weights.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return state.detach(), loss.detach()


def benchmark(
    config: GALAMConfig,
    device: torch.device,
    backend: Literal["step", "block"],
    batch_size: int,
    chunk_rows: int,
    warmup: int,
    updates: int,
) -> dict[str, object]:
    """Measure bounded full updates with carried detached state and identical seeded inputs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(1001)
    model = build_galam_variant(config, "schematic_gru_full", execution_backend=backend).to(device)
    if not isinstance(model, GALAMAdaptation):
        raise TypeError("Unexpected GALAM model")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    state = model.reset_state(batch_size, device, torch.float32)
    x = torch.randn(batch_size, chunk_rows, config.input_features, device=device)
    target = torch.randn(batch_size, chunk_rows, config.output_features, device=device)
    for _ in range(warmup):
        state, _ = _update(model, optimizer, x, target, state)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall_ms, cuda_ms, losses = [], [], []
    for _ in range(updates):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        state, loss = _update(model, optimizer, x, target, state)
        stop.record()
        torch.cuda.synchronize(device)
        wall_ms.append(1000 * (time.perf_counter() - begin))
        cuda_ms.append(start.elapsed_time(stop))
        losses.append(float(loss.item()))
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite parameter after engineering profile")
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Nonfinite loss during engineering profile")
    if not all(
        torch.isfinite(value).all().item()
        for item in optimizer.state.values()
        for value in item.values()
        if isinstance(value, torch.Tensor)
    ):
        raise RuntimeError("Nonfinite optimizer state after engineering profile")
    return {
        "backend": backend,
        "warmup_updates": warmup,
        "measured_updates": updates,
        "rows_per_update": batch_size * chunk_rows,
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
        "wall_median_ms": statistics.median(wall_ms),
        "cuda_median_ms": statistics.median(cuda_ms),
        "rows_per_second": batch_size * chunk_rows * 1000 / statistics.mean(wall_ms),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "synthetic_losses": losses,
        "steps_seen": state.steps_seen,
    }
