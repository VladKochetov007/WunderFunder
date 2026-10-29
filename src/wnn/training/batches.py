from __future__ import annotations

import math
import time

import numpy as np
import torch

from wnn.data import N_FEATURES, TARGET_QUANTA
from wnn.outcomes import N_SP_AUX, outcome_codes, sequence_class, structure_labels
from wnn.training.dynamics_auxiliary import dynamics_targets_torch
from wnn.training.inputs import build_inputs
from wnn.training.state import TrainingRun
from wnn.training.targets import (
    kth_change_labels,
    main_row_weights,
    next_change_labels,
    next_label_change,
    shifted_targets,
)
from wnn.training_controls import shuffled_masses
from wnn.training_views import origin_windows


def prepare_batch(run: TrainingRun) -> None:
    """Move one sequence batch, construct causal inputs and preserve writable target-copy isolation."""
    environment = run.environment
    args = run.options.args
    b = run.batch.b
    batch = run.batch.batch
    device = run.options.device
    family_targets = run.batch.family_targets
    dynamics_targets = run.batch.dynamics_targets
    dynamics_valid = run.batch.dynamics_valid
    initializer_teacher = run.network.initializer_teacher
    inputs = run.batch.inputs
    main_valid = run.batch.main_valid
    profile_batch_started = run.batch.profile_batch_started
    prop_model = run.data.prop_model
    quote_label = run.batch.quote_label
    scored_dev = run.batch.scored_dev
    spread_encoded = run.batch.spread_encoded
    teach = run.batch.teach
    thin = run.data.thin
    timer = run.progress.timer
    valid = run.batch.valid
    y = run.batch.y
    profile_batch_started = (
        time.perf_counter()
        if args.profile_steps and args.activation_backend != "none" and (b == 0)
        else None
    )
    timer.start()
    prop_cached = spread_encoded = kept_len = None
    if args.pin_memory:
        (x, y, scored_dev, *rest), scored = batch
    else:
        x, y, scored, *rest = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        scored_dev = scored.to(device, non_blocking=True)
    if thin is not None:
        kept_len = rest.pop()
        if not args.pin_memory:
            kept_len = kept_len.to(device, non_blocking=True)
    if args.spread_aux_mode != "original":
        spread_encoded = rest.pop()
        if not args.pin_memory:
            spread_encoded = spread_encoded.to(device, non_blocking=True)
    if rest:
        prop_cached = rest[0] if args.pin_memory else rest[0].to(device, non_blocking=True)
    timer.mark("h2d")
    members = getattr(run.network.model, "ensemble_k", 1)
    if members > 1:
        # TabM: each member trains on every sequence, so row-weighted losses average members.
        x, y, scored_dev, kept_len, spread_encoded, prop_cached = (
            None if value is None else value.repeat_interleave(members, dim=0)
            for value in (x, y, scored_dev, kept_len, spread_encoded, prop_cached)
        )
    inputs = build_inputs(x, environment=environment)
    timer.mark("inputs")
    teach = None
    family_targets = None
    quote_label = None
    if y.shape[-1] > 2:
        if args.aux_quote > 0:
            quote_label, y = (y[..., 2:4], y[..., :2])
        else:
            if args.distill_family_weight > 0:
                if y.shape[-1] != 12:
                    raise ValueError("four-family cache must append exactly eight family targets")
                family_targets = y[..., 4:12]
            teach, y = (y[..., 2:4], y[..., :2])
            if args.distill_cache_scale != 1.0:
                teach = teach * args.distill_cache_scale
            if initializer_teacher is not None:
                teach = teach.clone()
    if initializer_teacher is not None and teach is None:
        raise ValueError("final initializer distillation requires cached main-target columns")
    steps = torch.arange(inputs.shape[1], device=device)[None].expand(inputs.shape[0], -1)
    valid = steps >= environment.warmup
    if kept_len is not None:
        valid = valid & (
            kept_len if args.thin_mode == "retained-mass" else steps < kept_len[:, None]
        )
    dynamics_targets = dynamics_valid = None
    if args.dynamics_aux_head:
        raw_targets, dynamics_valid = dynamics_targets_torch(
            x[..., :N_FEATURES],
            run.network.model.mean[:N_FEATURES],
            run.network.model.std[:N_FEATURES],
            valid,
            lengths=kept_len if kept_len is not None and kept_len.ndim == 1 else None,
        )
        dynamics_targets = (
            raw_targets - run.network.model.dynamics_target_mean
        ) / run.network.model.dynamics_target_std
    main_valid = main_row_weights(
        inputs, y, valid, scored_dev, prop_model, prop_cached, args, x, environment=environment
    )
    timer.mark("weights")
    run.batch.family_targets = family_targets
    run.batch.dynamics_targets = dynamics_targets
    run.batch.dynamics_valid = dynamics_valid
    run.batch.inputs = inputs
    run.batch.main_valid = main_valid
    run.batch.profile_batch_started = profile_batch_started
    run.batch.quote_label = quote_label
    run.batch.scored_dev = scored_dev
    run.batch.spread_encoded = spread_encoded
    run.batch.teach = teach
    run.batch.valid = valid
    run.batch.y = y


def prepare_labels(run: TrainingRun) -> None:
    """Construct auxiliary labels and masks once per full sequence, before TBPTT slicing."""
    args = run.options.args
    aux = run.batch.aux
    br_lab = run.batch.br_lab
    ce_w = run.batch.ce_w
    cl_lab = run.batch.cl_lab
    codes = run.batch.codes
    epoch = run.progress.epoch
    ks = run.network.ks
    kth = run.batch.kth
    main_valid = run.batch.main_valid
    mask_rows = run.batch.mask_rows
    moments = run.network.moments
    nw = run.batch.nw
    nx_rows = run.batch.nx_rows
    nx_up = run.batch.nx_up
    nxt_all = run.batch.nxt_all
    nxt_label = run.batch.nxt_label
    scored_dev = run.batch.scored_dev
    shifts = run.data.shifts
    sp_lab = run.batch.sp_lab
    spread_encoded = run.batch.spread_encoded
    step_count = run.progress.step_count
    sw = run.batch.sw
    td_a = run.batch.td_a
    tier_label = run.batch.tier_label
    tier_weight = run.batch.tier_weight
    timer = run.progress.timer
    train_loader = run.data.train_loader
    valid = run.batch.valid
    y = run.batch.y
    aux = shifted_targets(y, valid, shifts)
    if args.aux_struct > 0:
        sp_lab, br_lab, cl_lab = structure_labels(y, sequence_class(y))
        if args.spread_aux_mode == "current_known":
            sp_lab = spread_encoded.clamp_max(N_SP_AUX - 1).long()
        sw = main_valid.float()
    if args.aux_next > 0:
        nx_rows, nx_up, nx_ok = next_change_labels(y, valid)
        nw = (
            main_valid.float() * nx_ok.float()
            if main_valid.dtype != torch.bool
            else (main_valid & nx_ok).float()
        )
        kth = [kth_change_labels(y, valid, kk) for kk in ks]
    if args.aux_tier > 0:
        tier_label = (
            (y.float().max(dim=-1).values / float(TARGET_QUANTA[3])).round().long().clamp(0, 3)
        )
        tier_weight = (
            main_valid.float()
            * (sequence_class(y) == 3)[:, None]
            * y.float().clamp(-2, 2).abs().mean(dim=-1)
        )
    if args.aux_mask > 0:
        mask_rows = (valid & scored_dev.any(dim=1)[:, None]).float()
    if moments is not None:
        codes = outcome_codes(y, sequence_class(y))
        row_w = main_valid.float()
        ce_w = row_w * y.float().clamp(-2, 2).abs().mean(-1) if args.ce_tilt else row_w
        if epoch == 0:
            moments.update(codes, y, row_w)
    nxt_label = (
        next_label_change(y) if args.distill_alpha_far is not None or args.distill_boot_h else None
    )
    if args.td_alpha > 0:
        nxt_all = next_label_change(y)
        ramp_steps = max(1, int(args.td_ramp * len(train_loader) * math.ceil(20000 / args.chunk)))
        td_a = args.td_alpha * min(1.0, step_count / ramp_steps)
    timer.mark("labels")
    run.batch.aux = aux
    run.batch.br_lab = br_lab
    run.batch.ce_w = ce_w
    run.batch.cl_lab = cl_lab
    run.batch.codes = codes
    run.batch.kth = kth
    run.batch.mask_rows = mask_rows
    run.batch.nw = nw
    run.batch.nx_rows = nx_rows
    run.batch.nx_up = nx_up
    run.batch.nxt_all = nxt_all
    run.batch.nxt_label = nxt_label
    run.batch.sp_lab = sp_lab
    run.batch.sw = sw
    run.batch.td_a = td_a
    run.batch.tier_label = tier_label
    run.batch.tier_weight = tier_weight


def prepare_windows(run: TrainingRun) -> None:
    """Reset sequence states and freeze the original update windows and donor masses."""
    args = run.options.args
    b = run.batch.b
    device = run.options.device
    donor_masses = run.batch.donor_masses
    epoch = run.progress.epoch
    h = run.batch.h
    h_ema = run.batch.h_ema
    h_initializer = run.batch.h_initializer
    inputs = run.batch.inputs
    main_valid = run.batch.main_valid
    update_rows = run.options.update_rows
    windows = run.batch.windows
    y = run.batch.y
    h = h_ema = None
    h_initializer = None
    donor_masses = None
    windows = origin_windows(
        inputs.shape[1], update_rows, args.chunk, args.origin_view, args.origin_seed, epoch, b
    )
    if args.activation_profile_first_chunk:
        windows = windows[:1]
    if args.distill_denominator == "shuffled":
        if args.origin_view == "off":
            donor_masses = shuffled_masses(
                y.float().clamp(-2, 2).abs() * main_valid[..., None],
                update_rows,
                args.denominator_shuffle_seed + 104729 * epoch + b,
            )
        else:
            masses = torch.stack(
                [
                    (y[:, window].float().clamp(-2, 2).abs() * main_valid[:, window, None])
                    .sum()
                    .detach()
                    for window in windows
                ]
            )
            order = np.random.default_rng(
                args.denominator_shuffle_seed + 104729 * epoch + b
            ).permutation(len(windows))
            donor_masses = masses[torch.as_tensor(order, device=device)]
    run.batch.donor_masses = donor_masses
    run.batch.h = h
    run.batch.h_ema = h_ema
    run.batch.h_initializer = h_initializer
    run.batch.windows = windows
