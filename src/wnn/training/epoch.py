from __future__ import annotations

import json
import time

import torch

from wnn import student_frontier
from wnn.run_identity import EpochUpdateCounter, sample_order_identity
from wnn.training.prefetch import CudaPrefetcher
from wnn.training.reporting import emit
from wnn.training.state import TrainingRun
from wnn.training_sampling import EpochOrderSampler


def prepare_epoch(run: TrainingRun) -> None:
    """Set epoch sampling identity and construct the original prefetched batch iterator."""
    args = run.options.args
    campaign_split = run.data.campaign_split
    device = run.options.device
    epoch = run.progress.epoch
    epoch_counter = run.progress.epoch_counter
    epoch_order_identity = run.progress.epoch_order_identity
    extra_valid = run.data.extra_valid
    frontier_contract = run.data.frontier_contract
    frontier_stage = run.options.frontier_stage
    model = run.network.model
    profile_batches = run.progress.profile_batches
    provenance = run.data.provenance
    timer = run.progress.timer
    train_idx = run.data.train_idx
    train_loader = run.data.train_loader
    model.train()
    epoch_counter = EpochUpdateCounter(device)
    if isinstance(train_loader.sampler, EpochOrderSampler):
        train_loader.sampler.set_epoch(epoch)
    epoch_order_identity = sample_order_identity(train_loader)
    if isinstance(train_loader.sampler, EpochOrderSampler):
        epoch_order_identity["seeded_order"] = {
            "seed": args.seed,
            "epoch": epoch,
            "antithetic": args.antithetic,
            "shared_dropped_tail": True,
        }
    if frontier_contract is not None:
        epoch_order_identity["student_frontier_order"] = student_frontier.validate_training_order(
            args,
            frontier_contract,
            frontier_stage,
            campaign_split,
            train_idx,
            extra_valid,
            train_loader,
        )
    if args.activation_profile_first_chunk:
        emit(
            "PROFILE_BATCH_PROVENANCE",
            json.dumps(
                {
                    "batch_size": args.batch,
                    "first_batch_index": 0,
                    "sample_order_identity": epoch_order_identity,
                    "same_batch_reused_for_warmup_and_measured_update": True,
                    "forward_rows": [0, args.chunk],
                    "initial_state": "zero for each update; first rows from sequence origin",
                    "fit_subset_members_sha256": provenance.get("fit_subset_members_sha256"),
                }
            ),
            flush=True,
        )
    run.progress.fit_loss_sum = None
    run.progress.fit_loss_windows = 0
    run.progress.fit_lr_first = None
    run.progress.fit_lr_last = None
    run.progress.fit_lr_sum = None
    batches = CudaPrefetcher(train_loader, device, timer) if args.pin_memory else train_loader
    if args.activation_profile_first_chunk:
        first_batch = next(iter(batches))
        profile_batches = iter((first_batch, first_batch))
    else:
        profile_batches = iter(batches)
    run.progress.epoch_counter = epoch_counter
    run.progress.epoch_order_identity = epoch_order_identity
    run.progress.profile_batches = profile_batches


@torch.no_grad()
def record_fit_window(run: TrainingRun) -> None:
    """Accumulate detached objective and pre-update learning rates for fit reporting."""
    loss = run.batch.loss
    sched = run.optimization.sched
    if loss is None or sched is None:
        raise RuntimeError("fit window lacks its objective or scheduler")
    fit_loss_sum = run.progress.fit_loss_sum
    fit_lr_sum = run.progress.fit_lr_sum
    rates = [float(value) for value in sched.get_last_lr()]
    if fit_loss_sum is None:
        run.progress.fit_loss_sum = loss.detach().clone()
        run.progress.fit_lr_first = rates.copy()
        run.progress.fit_lr_sum = rates.copy()
    else:
        if fit_lr_sum is None or len(fit_lr_sum) != len(rates):
            raise RuntimeError("scheduler learning-rate groups changed within an epoch")
        fit_loss_sum.add_(loss.detach())
        for index, rate in enumerate(rates):
            fit_lr_sum[index] += rate
    run.progress.fit_lr_last = rates
    run.progress.fit_loss_windows += 1


def fit_epoch_summary(run: TrainingRun, epoch_update_counts: dict[str, int]) -> dict[str, object]:
    """Return fit-only loss, learning-rate and optimizer-count telemetry."""
    count = run.progress.fit_loss_windows
    loss_sum = run.progress.fit_loss_sum
    lr_first = run.progress.fit_lr_first
    lr_last = run.progress.fit_lr_last
    lr_sum = run.progress.fit_lr_sum
    mean_loss = None if loss_sum is None or count == 0 else float(loss_sum.item()) / count
    mean_lr = None if lr_sum is None or count == 0 else [rate / count for rate in lr_sum]
    groups = []
    if lr_first is not None and lr_last is not None and mean_lr is not None:
        groups = [
            {"group": index, "first": first, "last": last, "mean": mean}
            for index, (first, last, mean) in enumerate(
                zip(lr_first, lr_last, mean_lr, strict=True)
            )
        ]
    return {
        "objective_mean": mean_loss,
        "objective_reduction": "arithmetic mean of per-window scalar objectives before optimizer update",
        "optimizer_windows": count,
        "learning_rate_by_group": groups,
        "optimizer_updates": epoch_update_counts,
    }


def finish_batch(run: TrainingRun) -> bool:
    """Close timing and report progress; return false at the declared profile boundary."""
    activation_profile_first_batch_seconds = run.progress.activation_profile_first_batch_seconds
    args = run.options.args
    b = run.batch.b
    device = run.options.device
    epoch = run.progress.epoch
    epoch_counter = run.progress.epoch_counter
    loss = run.batch.loss
    profile_batch_started = run.batch.profile_batch_started
    sched = run.optimization.sched
    t0 = run.progress.t0
    timer = run.progress.timer
    train_loader = run.data.train_loader
    timer.stop()
    if profile_batch_started is not None and activation_profile_first_batch_seconds is None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        activation_profile_first_batch_seconds = time.perf_counter() - profile_batch_started
    epoch_counter.completed_batches += 1
    if args.profile_steps and b + 1 == args.max_batches:
        run.progress.activation_profile_first_batch_seconds = activation_profile_first_batch_seconds
        return False
    if b % 20 == 0 and (not args.profile_steps):
        rate = (b + 1) * args.batch / (time.time() - t0)
        emit(
            f"ep {epoch} batch {b}/{len(train_loader)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} seq/s {rate:.1f}",
            flush=True,
        )
    run.progress.activation_profile_first_batch_seconds = activation_profile_first_batch_seconds
    return True


def epoch_diagnostics(run: TrainingRun) -> None:
    """Finish per-epoch gradient diagnostics without changing their reporting clock."""
    epoch = run.progress.epoch
    epoch_grad_stats = run.progress.epoch_grad_stats
    grad_stats = run.optimization.grad_stats
    epoch_grad_stats = None if grad_stats is None else {"epoch": epoch, **grad_stats.finish()}
    if epoch_grad_stats is not None:
        emit("GRAD_STATS", json.dumps(epoch_grad_stats), flush=True)
    run.progress.epoch_grad_stats = epoch_grad_stats
