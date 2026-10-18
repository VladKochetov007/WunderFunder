from __future__ import annotations

import math

import torch

from wnn.i1_history import I1_INPUT_WIDTH
from wnn.training.i1_inputs import validate_i1_input_configuration
from wnn.training.inheritance import inherit_init_arguments
from wnn.training.model_validation import apply_block_memory_defaults, block_memory_rejection
from wnn.training.state import TrainingRun
from wnn.training_qat import QAT_LEARN_SCALE_MODES


def validate_inputs(run: TrainingRun) -> None:
    """Reject incompatible input, subset and distillation options before data loading."""
    ap = run.options.ap
    args = run.options.args
    distillation_requested = run.options.distillation_requested
    try:
        validate_i1_input_configuration(vars(args), I1_INPUT_WIDTH)
    except ValueError as exc:
        ap.error(str(exc))
    if args.profile_steps:
        args.max_batches = args.profile_warmup + args.profile_steps
    if args.final_model_adaptation and (
        not args.init
        or not args.inherit_init_preprocessing
        or args.split_manifest is None
        or (args.fit_subset_manifest is None)
        or (not args.no_eval_during_fit)
        or (args.checkpoint_policy != "fixed")
        or args.resume_state
        or args.save_training_state
    ):
        ap.error(
            "final model adaptation requires inherited preprocessing, fixed component subset and no fit-time evaluation or continuation"
        )
    apply_block_memory_defaults(args, run.environment.argv)
    block_memory_error = block_memory_rejection(args)
    if block_memory_error is not None:
        ap.error(block_memory_error)
    if args.numeric_map != "none":
        if (
            args.arch != "gru"
            or args.cell != "gru"
            or args.enc != "linear"
            or (args.hidden != 128)
            or (args.layers != 1)
            or (args.dropout != 0.0)
            or (args.train_source != "train+valid")
            or (args.split_manifest is None)
            or (args.fit_subset_manifest is None)
            or args.train_classes
            or args.class_balance
            or (args.valid_train_fraction != 1.0)
            or (args.valid_repeat != 1)
            or args.epoch_samples
            or args.init
            or args.resume_state
            or args.fit_groups_manifest
            or args.distill
            or args.distill_cache_root
            or args.distill_family_weight
            or (args.privileged_mode != "none")
            or args.propensity
            or args.prop_input
            or args.aux
            or args.aux_struct
            or args.aux_next
            or args.aux_tier
            or args.aux_mask
            or args.aux_sign
            or args.aux_quote
            or args.aux_next_k
            or (args.spread_aux_mode != "original")
            or args.spread_cache_root
            or args.raw_only
            or args.online_norm
            or args.heuristics
            or (args.research_features != "none")
            or args.quote_cache_root
            or args.sentinel_base_width
            or (args.sentinel_values != "unchanged")
            or (args.sentinel_indicators != "none")
            or (args.loss != "wmse")
            or args.mask_weight
            or args.class_weight
            or (args.valid_true_mask != -1.0)
            or args.ema_teacher
            or args.swa_frac
            or args.eval_ema
            or (not args.no_eval_during_fit)
            or (args.checkpoint_policy != "fixed")
            or (args.checkpoint_epoch is None)
        ):
            ap.error("E540 numeric maps require the matched fixed-subset plain-GRU WMSE screen")
        if args.ple_intervals != 8:
            ap.error("E540 has a fixed n_intervals=8 comparison")
    elif args.fit_subset_manifest is not None:
        if (
            args.split_manifest is None
            or args.train_source != "train+valid"
            or args.include_heldout_valid
            or (args.train_seqs != 10607)
            or args.train_classes
            or args.class_balance
            or (args.valid_train_fraction != 1.0)
            or (args.valid_repeat != 1)
            or args.epoch_samples
            or args.fit_groups_manifest
            or (not args.no_eval_during_fit)
            or (args.checkpoint_policy != "fixed")
            or (args.checkpoint_epoch is None)
        ):
            ap.error(
                "component fit subsets require full train+valid source membership, no sampler filters or fit-time evaluation, and a fixed checkpoint"
            )
        if not args.fit_subset_experiment_id:
            ap.error("non-E540 component subsets require --fit-subset-experiment-id")
    elif args.fit_subset_experiment_id is not None:
        ap.error("--fit-subset-experiment-id requires --fit-subset-manifest")
    distillation_requested = bool(args.distill or args.distill_cache_root)
    if args.distill and args.distill_cache_root:
        ap.error("--distill and --distill-cache-root are mutually exclusive")
    if not math.isfinite(args.distill_family_weight) or args.distill_family_weight < 0:
        ap.error("--distill-family-weight must be finite and nonnegative")
    if not math.isfinite(args.distill_cache_scale) or args.distill_cache_scale <= 0:
        ap.error("--distill-cache-scale must be finite and positive")
    if args.distill_cache_scale != 1.0 and (not args.distill_cache_root):
        ap.error("--distill-cache-scale other than one requires --distill-cache-root")
    run.options.distillation_requested = distillation_requested


def validate_distillation(run: TrainingRun) -> None:
    """Validate target calibration and architecture/precision combinations."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    distillation_requested = run.options.distillation_requested
    init_checkpoint = run.options.init_checkpoint
    if args.distill_cache_scale != 1.0 and args.distill_family_weight > 0:
        ap.error("scaled main cache targets cannot be combined with family-target loss")
    if args.distill_family_weight > 0 and (not args.distill_cache_root):
        ap.error("--distill-family-weight requires --distill-cache-root")
    if not math.isfinite(args.distill_reanchor) or not 0 <= args.distill_reanchor <= 1:
        ap.error("--distill-reanchor must be finite and between zero and one")
    if args.distill_reanchor and (
        not distillation_requested or args.distill_boot_h <= 0 or args.distill_alpha_far is not None
    ):
        ap.error(
            "--distill-reanchor requires constant-alpha distillation and positive bootstrap horizon"
        )
    if args.recurrent_rank < 0:
        ap.error("--recurrent-rank cannot be negative")
    if args.arch == "specialist":
        if args.rnn_init != "default":
            ap.error("--arch specialist currently supports only --rnn-init default")
        if args.muon_scope == "hidden":
            ap.error(
                "--arch specialist supports Muon recurrent matrices only; --muon-scope hidden is unsupported"
            )
        if args.head_lr_mult != 1:
            ap.error("--arch specialist has multiple heads; --head-lr-mult is unsupported")
    if args.qat != "none" and (
        args.precision != "fp32"
        or args.arch != "gru"
        or args.cell != "gru"
        or (args.enc != "linear")
    ):
        ap.error("QAT requires explicit --precision fp32 with plain GRU and linear input encoder")
    if args.activation_backend != "none":
        if (
            args.precision != "fp32"
            or args.arch != "gru"
            or args.cell != "gru"
            or (args.enc != "linear")
            or args.recurrent_rank
        ):
            ap.error("activation scan backend supports only fp32 dense GRU with linear encoder")
        if args.activation_backend == "scan_a8" and args.dropout != 0:
            ap.error("the activation scan backend currently requires zero inter-layer dropout")
    if args.activation_profile_first_chunk and (
        args.activation_backend == "none" or args.profile_steps <= 0
    ):
        ap.error("--activation-profile-first-chunk requires activation backend profile mode")
    if args.qat_learn_scales and args.qat not in QAT_LEARN_SCALE_MODES:
        ap.error("--qat-learn-scales requires --qat recurrent_w4 or input_w4_recurrent_w8")
    if args.qat_learn_scales and (not math.isfinite(args.wd) or args.wd < 0):
        ap.error("learned QAT scales require finite nonnegative --wd")
    if args.calibrate_distill_reference and (
        args.device != "cpu"
        or args.resume_state
        or args.thin_prob
        or (args.spread_aux_mode != "original")
        or (args.calibration_epochs < 1)
        or args.profile_steps
        or args.max_batches
        or (args.distill_denominator != "chunk")
        or (args.loss != "wmse")
    ):
        ap.error(
            "reference calibration requires CPU wmse, chunk normalization, complete unthinned fit batches"
        )
    if args.zoneout != 0:
        ap.error("exact timestep zoneout is unsupported for fused nn.GRU")
    if args.batch_order == "qmc":
        ap.error(
            "QMC sampling requires a bound input-only strata cache; trainer integration is not implemented"
        )
    if args.antithetic and args.batch_order != "seeded":
        ap.error("--antithetic requires --batch-order seeded and a same-seed control")
    if args.init_qmc and (
        args.init or args.rnn_init != "default" or args.enc != "linear" or (args.arch != "gru")
    ):
        ap.error("--init-qmc supports a fresh linear-encoder GRU/LSTM only")
    init_checkpoint = None
    if args.inherit_init_preprocessing:
        if not args.init:
            ap.error("--inherit-init-preprocessing requires --init")
        parent_run, _, parent_epoch = args.init.partition("@ep")
        parent_file = f"model_ep{int(parent_epoch)}.pt" if parent_epoch else "model.pt"
        init_checkpoint = torch.load(
            environment.root / "artifacts" / "runs" / parent_run / parent_file,
            map_location="cpu",
            weights_only=False,
        )
        inherit_init_arguments(args, init_checkpoint, run.environment.argv)
    run.options.init_checkpoint = init_checkpoint
