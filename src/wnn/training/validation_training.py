from __future__ import annotations

import math

import torch

from wnn.data import N_FEATURES
from wnn.research_features import research_feature_control_width
from wnn.resources import configure_torch
from wnn.training.checkpoints import configure_checkpoint_policy
from wnn.training.initializer import final_initializer_distillation_error
from wnn.training.model_validation import independent_width_configuration_error
from wnn.training.state import TrainingRun
from wnn.training_controls import ema_decay_per_update
from wnn.training_credit import credit_edge_scales
from wnn.training_views import origin_windows


def validate_models(run: TrainingRun) -> None:
    """Resolve inherited model arguments and the declared checkpoint/continuation policy."""
    ap = run.options.ap
    args = run.options.args
    continuation = run.options.continuation
    final_initializer_error = final_initializer_distillation_error(args)
    if final_initializer_error is not None:
        ap.error(final_initializer_error)
    try:
        research_feature_control_width(args.research_features, args.research_feature_control)
    except ValueError as exc:
        ap.error(str(exc))
    if args.readout_activation != "gelu" and (
        args.arch != "gru"
        or args.cell != "gru"
        or args.enc != "linear"
        or (args.qat != "none")
        or args.recurrent_rank
    ):
        ap.error("fixed readout mixture requires a plain dense GRU, linear encoder and no QAT")
    width_override = (
        any(
            value is not None
            for value in (args.encoder_width, args.recurrent_width, args.readout_width)
        )
        or args.head_layers != 1
    )
    width_error = independent_width_configuration_error(vars(args))
    if width_error is not None:
        ap.error(width_error)
    if width_override and args.student_frontier_preflight_only:
        ap.error("student-frontier preflight uses its frozen equal-width shape contract")
    if any(
        value is not None and value <= 0
        for value in (args.encoder_width, args.recurrent_width, args.readout_width)
    ):
        ap.error("encoder, recurrent and readout widths must be positive")
    if args.recurrent_rank and (
        args.precision != "fp32"
        or args.optim != "adamw"
        or args.optim_switch
        or (args.qat != "none")
        or (not args.init)
        or (args.arch != "gru")
        or (args.cell != "gru")
        or (args.enc != "linear")
        or (args.recurrent_rank > (args.recurrent_width or args.hidden))
    ):
        ap.error(
            "factorized recurrence needs dense --init, plain GRU, linear encoder, rank <= hidden, AdamW, explicit FP32, no QAT or optimizer switch"
        )
    if (
        args.sentinel_base_width
        or args.sentinel_values != "unchanged"
        or args.sentinel_indicators != "none"
    ):
        if (
            not args.init
            or not args.inherit_init_preprocessing
            or args.sentinel_base_width < N_FEATURES
            or (args.arch != "gru")
            or (args.cell != "gru")
            or (args.enc != "linear")
            or (args.qat != "none")
            or args.recurrent_rank
        ):
            ap.error(
                "sentinel arms require a prepared plain dense GRU --init with --inherit-init-preprocessing"
            )
        if args.sentinel_indicators != "none" and any(
            "inp".startswith(prefix) for prefix in filter(None, args.freeze.split(","))
        ):
            ap.error("sentinel indicator arms require trainable input-projection columns")
    configure_checkpoint_policy(args, ap)
    continuation = bool(args.save_training_state or args.resume_state)
    if continuation and (
        args.checkpoint_policy != "fixed" or args.checkpoint_epoch != args.epochs - 1
    ):
        ap.error("training-state continuation requires a fixed last-epoch checkpoint")
    run.options.continuation = continuation


def validate_clocks(run: TrainingRun) -> None:
    """Validate update/detach/view clocks before seeding and initializing the device."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    detach_rows = run.options.detach_rows
    device = run.options.device
    distillation_requested = run.options.distillation_requested
    duty = run.options.duty
    ema_update_decay = run.options.ema_update_decay
    out_dir = run.options.out_dir
    update_rows = run.options.update_rows
    if args.thin_mode == "retained-mass" and (
        not args.thin_prob or not args.init or (not args.inherit_init_preprocessing)
    ):
        ap.error(
            "retained-mass control requires positive thinning and inherited parent preprocessing"
        )
    if args.credit_mode != "deterministic":
        if (
            args.optim != "sgd"
            or args.precision != "fp32"
            or (not args.no_gradient_clip)
            or args.grad_noise
            or args.optim_switch
            or args.detach_boundary_jitter
            or (args.origin_view != "off")
            or (args.chunk >= update_rows)
            or (args.detach_rows not in (0, update_rows))
        ):
            ap.error(
                "random credit requires unclipped fp32 SGD, chunk < update rows, full-window detach bound, and no other credit/noise/switch arms"
            )
        try:
            credit_edge_scales(0, args.credit_keep_probability, args.credit_seed, 0, True)
        except ValueError as error:
            ap.error(str(error))
        detach_rows = update_rows
    controlled_loss = bool(
        args.update_rows
        or args.detach_rows
        or args.distill_denominator != "chunk"
        or args.detach_boundary_jitter
        or (args.origin_view != "off")
        or (args.credit_mode != "deterministic")
    )
    if args.loss == "huber_curvature" and (
        distillation_requested
        or args.td_alpha
        or args.ema_teacher
        or controlled_loss
        or args.loss_norm != "chunk"
        or not math.isfinite(args.huber_delta)
        or args.huber_delta <= 0
    ):
        ap.error("huber_curvature needs label-only standard training and positive finite delta")
    if args.origin_view != "off":
        try:
            origin_windows(
                environment.sequence_length,
                update_rows,
                args.chunk,
                args.origin_view,
                args.origin_seed,
                0,
                0,
            )
        except ValueError as error:
            ap.error(str(error))
    if args.detach_boundary_jitter and (detach_rows <= args.chunk or args.detach_jitter_seed < 0):
        ap.error(
            "detach-boundary jitter needs --detach-rows larger than --chunk and a nonnegative seed"
        )
    if controlled_loss:
        if (
            args.chunk <= 0
            or update_rows <= 0
            or detach_rows <= 0
            or detach_rows % args.chunk
            or update_rows % detach_rows
            or environment.sequence_length % update_rows
        ):
            ap.error(
                "controlled micro/detach/update rows must divide each other and the sequence length"
            )
        if (
            not distillation_requested
            or args.loss != "wmse"
            or args.arch not in ("gru", "specialist", "prefix-adapter")
            or (args.enc != "linear")
            or args.dropout
            or (args.distill_alpha_far is not None)
            or args.td_alpha
            or args.aux
            or args.aux_next_k
            or args.aux_tier
            or args.aux_mask
            or args.aux_sign
            or args.aux_quote
            or args.thin_prob
            or (args.spread_aux_mode != "original")
            or (args.loss_norm != "chunk")
        ):
            ap.error(
                "controlled macro losses require cached-distilled GRU-family wmse, linear encoding, and no auxiliary arms"
            )
        if args.distill_family_weight > 0:
            ap.error("the family-head loss is not available with controlled macro losses")
        if args.distill_denominator != "chunk" and (
            args.distill_reference_mass <= 0 or not math.isfinite(args.distill_reference_mass)
        ):
            ap.error(
                "fixed/shuffled distilled normalization needs a positive finite frozen reference mass"
            )
    if args.ema_reference_rows and environment.sequence_length % update_rows:
        ap.error("EMA row clock requires update rows dividing sequence length")
    ema_update_decay = ema_decay_per_update(args.ema_decay, update_rows, args.ema_reference_rows)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if (
        device.type == "cpu"
        and (args.pin_memory or args.fused_adam)
        and (not args.student_frontier_preflight_only)
    ):
        ap.error("CPU implementation checks require unfused, synchronous loaders (omit --fast)")
    duty = configure_torch()
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    out_dir = environment.root / "artifacts" / "runs" / args.name
    if not args.student_frontier_preflight_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    run.options.detach_rows = detach_rows
    run.options.device = device
    run.options.duty = duty
    run.options.ema_update_decay = ema_update_decay
    run.options.out_dir = out_dir
    run.options.controlled_loss = controlled_loss
