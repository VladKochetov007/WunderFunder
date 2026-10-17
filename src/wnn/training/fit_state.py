from __future__ import annotations

from collections.abc import Callable, Iterator

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch
from attrs import define

from wnn.nn import PooledCorrMemory
from wnn.optim import GradStats
from wnn.run_identity import EpochUpdateCounter
from wnn.training.averaging import FastEMA
from wnn.training.profiling import StageTimer
from wnn.training_dynamics import Dynamics


@define
class OptimizationState:
    """Optimization state owned by one training run; tensors are never copied here."""

    amp_dtype: torch.dtype | None = None
    dynamics: Dynamics | None = None
    ema_model: FastEMA | torch.optim.swa_utils.AveragedModel | None = None
    grad_stats: GradStats | None = None
    objective: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    opt: torch.optim.Optimizer | None = None
    pool_memory: PooledCorrMemory | None = None
    scaler: torch.amp.GradScaler | None = None
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
    swa_model: torch.optim.swa_utils.AveragedModel | None = None
    swa_start: int | None = None
    use_autocast: bool | None = None


@define
class ProgressState:
    """Progress state owned by one training run; tensors are never copied here."""

    activation_profile_first_batch_seconds: float | None = None
    activation_profile_updates: list[dict[str, Any]] | None = None
    best: float | None = None
    cumulative_updates: dict[str, Any] | None = None
    epoch: int | None = None
    epoch_counter: EpochUpdateCounter | None = None
    epoch_grad_stats: dict[str, Any] | None = None
    epoch_order_identity: dict[str, Any] | None = None
    eval_model: torch.nn.Module | None = None
    evaluation_score: float | None = None
    fitting_started: float | None = None
    history: list[dict[str, Any]] | None = None
    losses_trace: list[torch.Tensor] | None = None
    norm_ema: torch.Tensor | None = None
    prior_fit_seconds: float | None = None
    profile_batches: Iterator[Any] | None = None
    profile_losses: list[torch.Tensor] | None = None
    start_epoch: int | None = None
    step_count: int | None = None
    t0: float | None = None
    timer: StageTimer | None = None
    training_contract: dict[str, Any] | None = None


@define
class BatchState:
    """Batch state owned by one training run; tensors are never copied here."""

    aux: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    b: int | None = None
    batch: Any | None = None
    br_lab: torch.Tensor | None = None
    ce_w: torch.Tensor | None = None
    cl_lab: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    donor_masses: torch.Tensor | None = None
    family_targets: torch.Tensor | None = None
    h: Any | None = None
    h_ema: Any | None = None
    h_initializer: Any | None = None
    inputs: torch.Tensor | None = None
    kth: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    loss: torch.Tensor | None = None
    main_valid: torch.Tensor | None = None
    mask_rows: torch.Tensor | None = None
    nw: torch.Tensor | None = None
    nx_rows: torch.Tensor | None = None
    nx_up: torch.Tensor | None = None
    nxt_all: torch.Tensor | None = None
    nxt_label: torch.Tensor | None = None
    out: torch.Tensor | None = None
    out_ema: torch.Tensor | None = None
    profile_batch_started: float | None = None
    qat_cache_identity: dict[str, Any] | None = None
    quote_label: torch.Tensor | None = None
    s: int | None = None
    scored_dev: torch.Tensor | None = None
    sl: slice | None = None
    sp_lab: torch.Tensor | None = None
    spread_encoded: torch.Tensor | None = None
    sw: torch.Tensor | None = None
    td_a: float | None = None
    teach: torch.Tensor | None = None
    tier_label: torch.Tensor | None = None
    tier_weight: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    window_index: int | None = None
    windows: list[slice] | None = None
    y: torch.Tensor | None = None
