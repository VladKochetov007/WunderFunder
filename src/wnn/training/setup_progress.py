from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from wnn.resources import cpu_threads
from wnn.splits import file_sha256
from wnn.training.checkpoints import resume_contract, serialized_args
from wnn.training.profiling import StageTimer
from wnn.training.reporting import emit
from wnn.training.state import TrainingRun
from wnn.training_dynamics import switch_optimizer
from wnn.training_state import restore_rng, restore_training_state
from wnn.training_views import origin_windows


def initialize_progress(run: TrainingRun) -> None:
    """Initialize run clocks, profiling and the strict source-bound continuation contract."""
    environment = run.environment
    activation_profile_first_batch_seconds = run.progress.activation_profile_first_batch_seconds
    activation_profile_updates = run.progress.activation_profile_updates
    args = run.options.args
    best = run.progress.best
    continuation = run.options.continuation
    cumulative_updates = run.progress.cumulative_updates
    detach_rows = run.options.detach_rows
    device = run.options.device
    eligible_valid_count = run.data.eligible_valid_count
    ema_update_decay = run.options.ema_update_decay
    extra_valid = run.data.extra_valid
    fitting_started = run.progress.fitting_started
    high = run.data.high
    history = run.progress.history
    losses_trace = run.progress.losses_trace
    low = run.data.low
    model = run.network.model
    n_valid = run.data.n_valid
    norm_ema = run.progress.norm_ema
    prefetch = run.data.prefetch
    prior_fit_seconds = run.progress.prior_fit_seconds
    profile_losses = run.progress.profile_losses
    provenance = run.data.provenance
    start_epoch = run.progress.start_epoch
    t0 = run.progress.t0
    thin = run.data.thin
    timer = run.progress.timer
    total = run.network.total
    train_idx = run.data.train_idx
    train_loader = run.data.train_loader
    train_split = run.data.train_split
    training_contract = run.progress.training_contract
    update_rows = run.options.update_rows
    workers = run.data.workers
    emit(json.dumps(serialized_args(args)), flush=True)
    emit(
        json.dumps(
            {
                "source_train_sequences": len(train_idx),
                "eligible_valid_train_sequences": eligible_valid_count,
                "used_valid_train_sequences": len(extra_valid)
                if extra_valid is not None
                else len(train_idx)
                if train_split == "valid"
                else 0,
                "eval_sequences": n_valid,
            }
        ),
        flush=True,
    )
    emit(
        f"loader workers {workers} prefetch {prefetch}; params {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M; steps {total}",
        flush=True,
    )
    timer = StageTimer(args.profile_steps > 0, args.profile_warmup)
    if args.profile_steps and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    norm_ema = None
    losses_trace = []
    profile_losses: list[torch.Tensor] | None = [] if args.profile_steps else None
    best, t0, history = (-1.0, time.time(), [])
    fitting_started = t0
    cumulative_updates = {}
    provenance["update_diagnostics"] = []
    provenance["training_clocks"] = {
        "micro_rows": args.chunk,
        "detach_rows": detach_rows,
        "update_rows": update_rows,
        "ema_decay_per_update": ema_update_decay,
        "ema_clock": "attempted updates including AMP skips",
    }
    if thin is not None:
        provenance["row_thinning"] = {
            "mode": args.thin_mode,
            "probability": args.thin_prob,
            "drop_range": [low, high],
            "observed_time_axis": "original"
            if args.thin_mode == "retained-mass"
            else "compressed then padded",
            "rows_per_sequence": environment.sequence_length,
            "update_schedule": "unchanged padded row count",
            "normalization": "frozen parent" if args.init else "fit on training views",
        }
    if args.credit_mode != "deterministic":
        provenance["training_clocks"]["credit_estimator"] = {
            "mode": args.credit_mode,
            "keep_probability": args.credit_keep_probability,
            "target": "fixed update-window gradient conditional on detached incoming state",
            "optimizer_claim": "affine SGD/momentum step conditional on fixed incoming optimizer state",
            "maximum_graph_rows": update_rows,
            "random_edge_spacing_rows": args.chunk,
        }
    if args.origin_view != "off":
        initial_windows = origin_windows(
            environment.sequence_length,
            update_rows,
            args.chunk,
            args.origin_view,
            args.origin_seed,
            0,
            0,
        )
        provenance["training_clocks"].update(
            {
                "origin_view": args.origin_view,
                "prefix_rows_per_sequence": (
                    sum(min(window.start, args.burnin_rows) for window in initial_windows)
                    if args.origin_view == "burnin-prefix"
                    else 0
                    if args.origin_view == "burnin-reset"
                    else sum(window.start for window in initial_windows)
                ),
                "supervised_rows_per_sequence": environment.sequence_length,
                "variable_update_width": args.origin_view == "prefix-random",
                "uses_replayed_state": args.origin_view not in ("prefix-discard", "burnin-reset"),
                "burnin_rows": args.burnin_rows,
                "ema_clock": "actual update rows"
                if args.ema_reference_rows
                else "attempted updates including AMP skips",
            }
        )
    activation_profile_first_batch_seconds = None
    activation_profile_updates = []
    torch.set_num_threads(cpu_threads())
    training_contract = (
        resume_contract(args, provenance, train_loader, total, environment=environment)
        if continuation
        else None
    )
    start_epoch = 0
    prior_fit_seconds = 0.0
    run.progress.activation_profile_first_batch_seconds = activation_profile_first_batch_seconds
    run.progress.activation_profile_updates = activation_profile_updates
    run.progress.best = best
    run.progress.cumulative_updates = cumulative_updates
    run.progress.fitting_started = fitting_started
    run.progress.history = history
    run.progress.losses_trace = losses_trace
    run.progress.norm_ema = norm_ema
    run.progress.prior_fit_seconds = prior_fit_seconds
    run.progress.profile_losses = profile_losses
    run.progress.start_epoch = start_epoch
    run.progress.t0 = t0
    run.progress.timer = timer
    run.progress.training_contract = training_contract


def restore_progress(run: TrainingRun) -> None:
    """Restore optimizer and stream clocks, validating ancestry before restoring RNG last."""
    ap = run.options.ap
    args = run.options.args
    best = run.progress.best
    cumulative_updates = run.progress.cumulative_updates
    device = run.options.device
    dynamics = run.optimization.dynamics
    ema_model = run.optimization.ema_model
    history = run.progress.history
    model = run.network.model
    moments = run.network.moments
    norm_ema = run.progress.norm_ema
    opt = run.optimization.opt
    pool_memory = run.optimization.pool_memory
    prior_fit_seconds = run.progress.prior_fit_seconds
    provenance = run.data.provenance
    saved_training = run.data.saved_training
    scaler = run.optimization.scaler
    sched = run.optimization.sched
    start_epoch = run.progress.start_epoch
    step_count = run.progress.step_count
    swa_model = run.optimization.swa_model
    train_loader = run.data.train_loader
    training_contract = run.progress.training_contract
    if saved_training is not None:
        if saved_training["contract"] != training_contract:
            ap.error(
                "resume contract differs: recipe, schedule, data, code, lineage or runtime changed"
            )
        saved_dynamics = saved_training["auxiliary"]["dynamics"]
        if saved_dynamics["is_switched"]:
            opt, _ = switch_optimizer(opt, sched, saved_dynamics["switch"][1])
        restore_training_state(
            saved_training,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            loader=train_loader,
            ema=ema_model,
            swa=swa_model,
        )
        dynamics.load_state_dict(saved_dynamics)
        provenance["optimizer_switch_events"] = dynamics.events
        progress = saved_training["progress"]
        start_epoch, step_count = (saved_training["next_epoch"], progress["step_count"])
        best, history = (progress["best"], progress["history"])
        cumulative_updates = progress["cumulative_updates"]
        provenance["update_diagnostics"] = progress["update_diagnostics"]
        provenance["resumed_from"] = {
            "path": str(args.resume_state),
            "sha256": file_sha256(Path(args.resume_state)),
            "next_epoch": start_epoch,
        }
        prior_fit_seconds = progress["fitting_seconds"]
        auxiliary_state = saved_training["auxiliary"]
        norm_ema = auxiliary_state["norm_ema"]
        if norm_ema is not None:
            norm_ema = norm_ema.to(device)
        if pool_memory is not None:
            value = auxiliary_state["pool_memory"]
            pool_memory.memory = None if value is None else value.to(device)
        if moments is not None:
            for field, value in auxiliary_state["readout_moments"].items():
                getattr(moments, field).copy_(value.to(device))
        restore_rng(saved_training["rng"])
    run.progress.best = best
    run.progress.cumulative_updates = cumulative_updates
    run.progress.history = history
    run.progress.norm_ema = norm_ema
    run.optimization.opt = opt
    run.progress.prior_fit_seconds = prior_fit_seconds
    run.progress.start_epoch = start_epoch
    run.progress.step_count = step_count
