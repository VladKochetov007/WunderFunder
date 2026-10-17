from __future__ import annotations

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch

from wnn.outcomes import N_SP_AUX
from wnn.training.initializer import replace_final_initializer_chunk
from wnn.training.state import TrainingRun
from wnn.training_controls import DistilledLoss, detach_boundary_offset, distilled_macro_losses
from wnn.training_credit import credit_edge_scales


def prepare_window(run: TrainingRun) -> None:
    """Advance the optimizer-switch clock and the independent frozen-initializer state."""
    dynamics = run.optimization.dynamics
    h_initializer = run.batch.h_initializer
    initializer_teacher = run.network.initializer_teacher
    inputs = run.batch.inputs
    opt = run.optimization.opt
    qat_cache_identity = run.batch.qat_cache_identity
    s = run.batch.s
    sched = run.optimization.sched
    sl = run.batch.sl
    step_count = run.progress.step_count
    teach = run.batch.teach
    s = sl.start
    qat_cache_identity = None
    opt = dynamics.maybe_switch(step_count, opt, sched)
    if initializer_teacher is not None:
        h_initializer = replace_final_initializer_chunk(
            initializer_teacher, inputs, sl, h_initializer, teach, environment=run.environment
        )
    run.batch.h_initializer = h_initializer
    run.optimization.opt = opt
    run.batch.qat_cache_identity = qat_cache_identity
    run.batch.s = s


def backward_controlled(run: TrainingRun) -> None:
    """Accumulate the original macro-normalized gradients with unchanged recurrent boundaries."""
    environment = run.environment
    amp_dtype = run.optimization.amp_dtype
    args = run.options.args
    br_lab = run.batch.br_lab
    cl_lab = run.batch.cl_lab
    detach_rows = run.options.detach_rows
    device = run.options.device
    donor_masses = run.batch.donor_masses
    ema_model = run.optimization.ema_model
    h = run.batch.h
    h_ema = run.batch.h_ema
    inputs = run.batch.inputs
    loss = run.batch.loss
    main_valid = run.batch.main_valid
    model = run.network.model
    next_at = run.network.next_at
    nw = run.batch.nw
    nx_rows = run.batch.nx_rows
    nx_up = run.batch.nx_up
    nxt_label = run.batch.nxt_label
    opt = run.optimization.opt
    profile_losses = run.progress.profile_losses
    scaler = run.optimization.scaler
    sl = run.batch.sl
    sp_lab = run.batch.sp_lab
    step_count = run.progress.step_count
    struct_at = run.network.struct_at
    teach = run.batch.teach
    timer = run.progress.timer
    use_autocast = run.optimization.use_autocast
    window_index = run.batch.window_index
    y = run.batch.y

    def raw_forward(window: slice, state: Any) -> tuple[torch.Tensor, Any]:
        """Raw forward."""
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
            return environment.forward_chunk(model, inputs, window, state)

    def teacher_forward(window: slice, state: Any) -> tuple[torch.Tensor, Any]:
        """Teacher forward."""
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
            return environment.forward_chunk(ema_model.module, inputs, window, state)

    if args.origin_view != "off":
        carried_h, carried_h_ema = (h, h_ema)
        h = h_ema = None
        with torch.no_grad():
            for prefix_start in range(0, sl.start, args.chunk):
                prefix = slice(prefix_start, prefix_start + args.chunk)
                _, h = raw_forward(prefix, h)
                if args.ema_teacher:
                    _, h_ema = teacher_forward(prefix, h_ema)
        if args.origin_view == "prefix-discard":
            h, h_ema = (carried_h, carried_h_ema)
    opt.zero_grad(set_to_none=True)
    loss = torch.zeros((), device=device)
    configuration = DistilledLoss(
        args.distill_alpha,
        args.distill_boot_h,
        args.ema_teacher,
        args.ema_metric_weight,
        args.aux_struct,
        struct_at,
        N_SP_AUX,
        args.aux_next,
        next_at,
        args.distill_reanchor,
    )
    for contribution, next_h, next_h_ema in distilled_macro_losses(
        forward=raw_forward,
        ema_forward=teacher_forward,
        state=h,
        ema_state=h_ema,
        labels=y,
        teacher=teach,
        row_weights=main_valid,
        next_change=nxt_label,
        structural_labels=(sp_lab, br_lab, cl_lab) if args.aux_struct else None,
        next_labels=(nx_rows, nx_up, nw) if args.aux_next else None,
        rows=sl,
        micro_rows=args.chunk,
        detach_rows=detach_rows,
        config=configuration,
        denominator=args.distill_denominator,
        reference_mass=args.distill_reference_mass,
        detach_offset_rows=detach_boundary_offset(
            args.chunk, detach_rows, args.detach_jitter_seed, step_count
        )
        if args.detach_boundary_jitter
        else 0,
        credit_scales=credit_edge_scales(
            (sl.stop - sl.start) // args.chunk - 1,
            args.credit_keep_probability,
            args.credit_seed,
            step_count,
            args.credit_mode == "artbp",
        )
        if args.credit_mode != "deterministic"
        else None,
        donor_mass=None if donor_masses is None else donor_masses[window_index],
    ):
        h, h_ema = (next_h, next_h_ema)
        scaler.scale(contribution).backward()
        loss = loss + contribution.detach()
    if profile_losses is not None:
        profile_losses.append(loss.detach())
    timer.mark("loss")
    run.batch.h = h
    run.batch.h_ema = h_ema
    run.batch.loss = loss
