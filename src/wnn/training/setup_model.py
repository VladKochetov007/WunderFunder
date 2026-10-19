from __future__ import annotations

import torch

from wnn.data import N_FEATURES
from wnn.distillation import append_family_output_head, extend_family_output_head
from wnn.features import N_EVENT_FEATURES
from wnn.optim import rnn_init_
from wnn.outcomes import N_CODES, N_SP_AUX, ReadoutMoments
from wnn.propensity import load_propensity
from wnn.research_features import research_feature_names
from wnn.sentinel import SentinelPreprocessor
from wnn.sentinel_calibration import validate_fit as validate_sentinel_fit
from wnn.splits import file_sha256
from wnn.spread_aux import validate_spread_parent
from wnn.training.inheritance import (
    can_extend_research_features,
    expand_research_feature_state,
    validate_campaign_checkpoint,
)
from wnn.training.input_representation import (
    extend_input_projection_state,
    representation_input_width,
)
from wnn.training.mean_variance import (
    append_log_variance_output_head,
    extend_mean_variance_state,
)
from wnn.training.mean_variance import (
    configuration_error as mean_variance_configuration_error,
)
from wnn.training.representation_validation import representation_configuration_error
from wnn.training.state import TrainingRun
from wnn.training_sampling import init_qmc_


def build_model(run: TrainingRun) -> None:
    """Allocate the requested architecture and exact auxiliary output offsets."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    device = run.options.device
    family_at = run.network.family_at
    init = run.network.init
    ks = run.network.ks
    mask_at = run.network.mask_at
    model = run.network.model
    moments = run.network.moments
    n_in = run.network.n_in
    n_out = run.network.n_out
    next_at = run.network.next_at
    quote_at = run.network.quote_at
    saved_training = run.data.saved_training
    shifts = run.data.shifts
    sign_at = run.network.sign_at
    struct_at = run.network.struct_at
    tier_at = run.network.tier_at
    mean_variance_error = mean_variance_configuration_error(args, run.options.controlled_loss)
    if mean_variance_error is not None:
        ap.error(mean_variance_error)
    representation_error = representation_configuration_error(args)
    if representation_error is not None:
        ap.error(representation_error)
    environment.inputs.research_features = args.research_features
    environment.inputs.research_feature_prior = None
    environment.inputs.research_feature_control = args.research_feature_control
    representation_mode = args.representation_mode
    environment.inputs.representation_mode = representation_mode or "none"
    environment.inputs.privileged_mode, environment.inputs.privileged_horizon = (
        args.privileged_mode,
        args.privileged_horizon,
    )
    environment.inputs.sentinel_preprocessor = None
    n_in = (
        N_FEATURES
        + N_EVENT_FEATURES
        + len(research_feature_names(environment.inputs.research_features))
        + (N_FEATURES if environment.inputs.privileged_mode != "none" else 0)
    )
    if args.raw_only:
        environment.inputs.raw_only = True
    if args.heuristics:
        from wnn.heuristics import K as N_HEUR

        environment.inputs.heuristics = True
        n_in += N_HEUR
    if args.online_norm:
        environment.inputs.online_norm = True
        n_in += N_FEATURES
    if args.prop_input:
        environment.inputs.propensity_model = load_propensity(
            environment.root / "artifacts" / "runs" / args.prop_input / "model.pt", device
        )
        n_in += 1
    if args.sentinel_base_width:
        if n_in != args.sentinel_base_width:
            ap.error("sentinel parent base width differs from inherited preprocessing")
        if args.sentinel_indicators != "none":
            n_in += N_FEATURES
    if representation_mode is not None:
        n_in += representation_input_width(representation_mode)
    if args.loss == "ce_joint" and shifts:
        ap.error("--aux is not supported with the outcome-code head")
    n_out = N_CODES if args.loss == "ce_joint" else 2 + 2 * len(shifts)
    struct_at = n_out
    if args.aux_struct > 0:
        n_out += N_SP_AUX + 3 + 4
    next_at = n_out
    ks = [int(v) for v in args.aux_next_k.split(",") if v]
    if args.aux_next > 0:
        n_out += 2 + 2 * len(ks)
    tier_at = n_out
    if args.aux_tier > 0:
        n_out += 4
    mask_at = n_out
    if args.aux_mask > 0:
        n_out += 1
    sign_at = n_out
    if args.aux_sign > 0:
        n_out += 1
    quote_at = n_out
    if args.aux_quote > 0:
        n_out += 2
    family_at = n_out
    base_n_out = n_out
    if args.distill_family_weight > 0:
        n_out += 8
    model = environment.make_model(vars(args), n_in, base_n_out)
    if args.distill_family_weight > 0:
        model = append_family_output_head(model)
    model = model.to(device)
    if args.init_qmc and saved_training is None:
        init_qmc_(model, args.seed)
    if args.rnn_init != "default" and (not args.init):
        rnn_init_(model, args.rnn_init, args.chrono_tmax, args.seed)
    if args.mean_variance_head:
        model = append_log_variance_output_head(model)
        n_out += 2
    moments = ReadoutMoments(device) if args.loss == "ce_joint" else None
    init = None
    run.network.family_at = family_at
    run.network.init = init
    run.network.ks = ks
    run.network.mask_at = mask_at
    run.network.model = model
    run.network.moments = moments
    run.network.n_in = n_in
    run.network.n_out = n_out
    run.network.next_at = next_at
    run.network.quote_at = quote_at
    run.network.sign_at = sign_at
    run.network.struct_at = struct_at
    run.network.tier_at = tier_at


def load_initializer(run: TrainingRun) -> None:
    """Load the selected parent and verify its preprocessing and evaluation exclusions."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    representation_mode = args.representation_mode
    campaign_split = run.data.campaign_split
    init = run.network.init
    init_checkpoint = run.options.init_checkpoint
    model = run.network.model
    provenance = run.data.provenance
    if args.init:
        init_run, _, init_epoch = args.init.partition("@ep")
        init_file = f"model_ep{int(init_epoch)}.pt" if init_epoch else "model.pt"
        init_path = environment.root / "artifacts" / "runs" / init_run / init_file
        init = (
            init_checkpoint
            if init_checkpoint is not None
            else torch.load(init_path, map_location="cpu", weights_only=False)
        )
        if init["args"].get("readout_activation", "gelu") != args.readout_activation:
            ap.error(
                "readout activation differs from initialization; inherit the parent preprocessing"
            )
        if args.recurrent_rank and (init.get("quantization") or init.get("factorization")):
            ap.error(
                "factorized initialization must be a dense float checkpoint; use raw state for exact resume"
            )
        if args.qat != "none" and init.get("factorization"):
            ap.error("combined QAT and factorized initialization is not implemented")
        validate_campaign_checkpoint(init, args, final_initializer=args.final_model_adaptation)
        if args.spread_aux_mode != "original":
            validate_spread_parent(init, campaign_split.sha256, provenance["fit_row_groups"])
        provenance["initial_checkpoint_sha256"] = file_sha256(init_path)
        parent_mode = init.get("args", {}).get("research_features", "none")
        parent_state = init["model"]
        if parent_mode != environment.inputs.research_features:
            if not args.inherit_init_preprocessing or not can_extend_research_features(
                parent_mode, environment.inputs.research_features
            ):
                ap.error("initializer research features require a supported causal extension")
            if args.numeric_map != "none" or args.sentinel_base_width:
                ap.error("causal feature expansion does not support mapped or sentinel projections")
            target_state = model.state_dict()
            expansion_target = dict(target_state)
            for name in ("head.2.weight", "head.2.bias"):
                if name in parent_state:
                    expansion_target[name] = parent_state[name]
            parent_state = expand_research_feature_state(
                parent_state,
                expansion_target,
                parent_mode,
                environment.inputs.research_features,
                N_FEATURES + N_EVENT_FEATURES + (N_FEATURES if args.online_norm else 0),
            )
            provenance["research_feature_extension"] = {
                "parent_mode": parent_mode,
                "child_mode": environment.inputs.research_features,
                "added_columns": len(research_feature_names(environment.inputs.research_features))
                - len(research_feature_names(parent_mode)),
                "input_weights": "zero_initialized",
                "normalization": "mean_zero_std_one",
            }
        parent_representation_mode = init.get("args", {}).get("representation_mode")
        if parent_representation_mode != representation_mode:
            if parent_representation_mode not in (None, "none") or representation_mode is None:
                ap.error("representation initializer must be the canonical 211-column parent")
            if parent_state.get("inp.weight") is None or parent_state["inp.weight"].shape[1] != (
                N_FEATURES + N_EVENT_FEATURES
            ):
                ap.error("representation initializer has an unlabelled noncanonical input width")
            parent_state = extend_input_projection_state(
                parent_state, model.state_dict(), parent_width=N_FEATURES + N_EVENT_FEATURES
            )
            provenance["representation_input_extension"] = {
                "parent_mode": parent_representation_mode or "none",
                "child_mode": representation_mode,
                "added_columns": representation_input_width(representation_mode),
                "input_projection": "new columns zero; original columns copied",
                "normalization": "new means zero and scales one",
            }
        if args.mean_variance_head:
            if init.get("args", {}).get("mean_variance_head", 0):
                ap.error("mean/logvariance head extension requires a parent without that head")
            parent_has_family = init.get("args", {}).get("distill_family_weight", 0.0) > 0
            child_has_family = args.distill_family_weight > 0
            if parent_has_family != child_has_family:
                ap.error("mean/logvariance warm start requires matching family-head layouts")
            parent_state = extend_mean_variance_state(
                parent_state,
                model.state_dict(),
                family_outputs=8 if child_has_family and not parent_has_family else 0,
            )
        elif args.distill_family_weight > 0:
            parent_state = extend_family_output_head(parent_state, model.state_dict())
        model.load_state_dict(parent_state)
        sentinel_metadata = init.get("sentinel_preprocessing")
        if bool(sentinel_metadata) != bool(args.sentinel_base_width):
            ap.error("sentinel metadata requires matching inherited preprocessing options")
        if sentinel_metadata:
            if (
                sentinel_metadata["value_mode"] != args.sentinel_values
                or sentinel_metadata["indicators"] != args.sentinel_indicators
                or sentinel_metadata["base_width"] != args.sentinel_base_width
            ):
                ap.error("sentinel initializer modes differ from training options")
            environment.inputs.sentinel_preprocessor = SentinelPreprocessor(sentinel_metadata)
            validate_sentinel_fit(
                sentinel_metadata["calibration"],
                environment.root,
                provenance["fit_row_groups"],
                provenance["split_manifest_sha256"],
            )
            model._sentinel_preprocessing = sentinel_metadata
            provenance["sentinel_preprocessing"] = sentinel_metadata
    run.network.init = init
