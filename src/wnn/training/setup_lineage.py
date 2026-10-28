from __future__ import annotations

import argparse
import math

from wnn import student_frontier
from wnn.privileged import load_inner_manifest
from wnn.resources import configure_strict_fp32
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
    load_campaign_split,
)
from wnn.training.membership import load_component_fit_subset
from wnn.training.state import TrainingRun
from wnn.training_state import load_training_state


def require_chronological_primary_training(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    campaign_split: CampaignSplit | None,
) -> None:
    """Reject future recurrent fits without the official train/full-valid boundary."""
    if campaign_split is None:
        parser.error(
            "future recurrent fits require --split-manifest with chronological primary membership"
        )
    if campaign_split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        parser.error("random-fold manifests are historical; use a chronological primary manifest")
    if campaign_split.valid_groups:
        parser.error("chronological primary validation forbids fitting any valid sequences")
    if (
        args.train_source != "train"
        or args.include_heldout_valid
        or args.valid_train_fraction != 1
        or args.valid_repeat != 1
        or args.valid_fold >= 0
        or args.no_eval_during_fit
        or args.checkpoint_policy != "fixed"
        or args.eval_source not in ("dev", "vfold")
        or args.fit_subset_manifest is not None
        or args.fit_groups_manifest is not None
        or args.class_balance
    ):
        parser.error(
            "chronological primary fitting accepts official train only and full-valid evaluation; fit-valid, random-fold and no-eval options are forbidden"
        )


def load_lineage(run: TrainingRun) -> None:
    """Load source-qualified split, subset and continuation identities."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    campaign_split = run.data.campaign_split
    continuation = run.options.continuation
    distillation_requested = run.options.distillation_requested
    fit_subset = run.data.fit_subset
    inner_crossfit = run.data.inner_crossfit
    saved_training = run.data.saved_training
    if continuation and (args.profile_steps or args.parity_dump):
        ap.error("profiling/parity early exits cannot save epoch continuation state")
    saved_training = load_training_state(args.resume_state) if args.resume_state else None
    if args.resume_state:
        args.resume_state = str(args.resume_state)
    campaign_split = (
        load_campaign_split(args.split_manifest, environment.root) if args.split_manifest else None
    )
    require_chronological_primary_training(ap, args, campaign_split)
    fit_subset = None
    if args.fit_subset_manifest is not None:
        if campaign_split is None:
            ap.error("E540 fit subsets require --split-manifest")
        subset_path = args.fit_subset_manifest
        if not subset_path.is_absolute():
            subset_path = environment.root / subset_path
        subset_path = subset_path.resolve()
        expected_experiment_id = (
            "E540" if args.numeric_map != "none" else args.fit_subset_experiment_id
        )
        if args.numeric_map != "none" and args.fit_subset_experiment_id not in (None, "E540"):
            ap.error("E540 numeric-map subsets require experiment identity E540")
        fit_subset = load_component_fit_subset(
            subset_path, campaign_split, expected_experiment_id=expected_experiment_id
        )
        args.fit_subset_manifest = str(subset_path)
        args.fit_subset_manifest_sha256 = file_sha256(subset_path)
    inner_crossfit = None
    if args.fit_groups_manifest is not None:
        if (
            campaign_split is None
            or args.train_source != "train+valid"
            or args.init
            or distillation_requested
            or args.propensity
            or args.prop_input
            or (args.train_seqs != 10607)
            or (args.valid_train_fraction != 1)
            or args.train_classes
        ):
            ap.error(
                "inner crossfit requires a fresh train+valid teacher, no learned external ancestors, and full inner membership"
            )
        inner_crossfit = load_inner_manifest(args.fit_groups_manifest, campaign_split)
        args.fit_groups_manifest_sha256 = file_sha256(args.fit_groups_manifest)
        args.fit_groups_manifest = str(args.fit_groups_manifest)
    elif args.privileged_mode != "none":
        ap.error("teacher-only current/future inputs require --fit-groups-manifest")
    if args.privileged_horizon < 1:
        ap.error("--privileged-horizon must be positive")
    run.data.campaign_split = campaign_split
    run.data.fit_subset = fit_subset
    run.data.inner_crossfit = inner_crossfit
    run.data.saved_training = saved_training


def validate_membership(run: TrainingRun) -> None:
    """Reject cross-fold ancestry and incompatible data/target combinations."""
    ap = run.options.ap
    args = run.options.args
    campaign_split = run.data.campaign_split
    distillation_requested = run.options.distillation_requested
    frontier_contract = run.data.frontier_contract
    if campaign_split is not None:
        if args.valid_fold not in (-1, campaign_split.fold):
            ap.error("--valid-fold disagrees with --split-manifest")
        if args.include_heldout_valid or args.train_source == "valid_devA":
            ap.error(
                "campaign splits require train, valid, or train+valid sources without heldout inclusion"
            )
        if args.prop_input:
            ap.error("campaign splits require explicit provenance for propensity input")
        args.valid_fold, args.n_valid_folds = (campaign_split.fold, campaign_split.n_folds)
        args.eval_source = "vfold"
        args.split_manifest = str(campaign_split.path)
        args.split_manifest_sha256 = campaign_split.sha256
    if args.loss == "gcorr_pool" and args.aux:
        ap.error(
            "--loss gcorr_pool keeps one pooled memory for the main head; --aux shift heads would share it"
        )
    if args.distill_cache_root and campaign_split is None:
        ap.error("--distill-cache-root requires a clean --split-manifest")
    if distillation_requested and args.loss != "wmse":
        ap.error("cached distillation trains wMSE toward the blended target; pass --loss wmse")
    if distillation_requested and args.loss_norm != "chunk":
        ap.error(
            "--loss-norm running does not apply to cached distillation; use --distill-denominator"
        )
    if args.distill_boot_h and (not distillation_requested):
        ap.error("--distill-boot-h bootstraps from the distillation teacher; enable distillation")
    if args.aux_quote > 0 and distillation_requested:
        ap.error("--aux-quote and distillation both use target columns 2:4")
    if args.distill_family_weight > 0 and (
        args.arch != "gru"
        or args.cell != "gru"
        or args.enc != "linear"
        or args.aux_quote
        or (args.spread_aux_mode != "original")
    ):
        ap.error(
            "the four-family head requires a plain GRU with no quote-path or spread-cache head"
        )
    if args.quote_cache_root and args.aux_quote <= 0:
        ap.error("--quote-cache-root requires --aux-quote")
    if args.fast:
        args.sync_free = args.fused_adam = args.pin_memory = args.prop_cache = True
    frontier_contract = None
    run.data.frontier_contract = frontier_contract


def configure_runtime(run: TrainingRun) -> None:
    """Validate the frontier/device envelope before constructing data and models."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    campaign_split = run.data.campaign_split
    detach_rows = run.options.detach_rows
    frontier_contract = run.data.frontier_contract
    frontier_contract_sha256 = run.options.frontier_contract_sha256
    frontier_stage = run.options.frontier_stage
    thin_high = run.options.thin_high
    thin_low = run.options.thin_low
    update_rows = run.options.update_rows
    frontier_contract_sha256 = None
    frontier_stage = None
    if args.student_frontier_contract is not None:
        if campaign_split is None or not args.student_frontier_stage:
            ap.error("student-frontier recipes require --split-manifest and a named stage")
        frontier_contract, frontier_contract_sha256 = student_frontier.load_contract(
            args.student_frontier_contract
        )
        frontier_stage = student_frontier.validate_stage_args(
            args, frontier_contract, args.student_frontier_stage, environment.root
        )
        expected_device = "cpu" if args.student_frontier_preflight_only else "cuda"
        if args.device != expected_device:
            ap.error(f"student-frontier stage requires --device {expected_device}")
    elif args.student_frontier_stage or args.student_frontier_preflight_only:
        ap.error("student-frontier stage/preflight flags require --student-frontier-contract")
    if args.precision == "fp32" or args.student_frontier_contract is not None:
        configure_strict_fp32()
    if args.activation_profile_first_chunk and args.pin_memory:
        ap.error("first-chunk profile batch reuse requires a non-pinned DataLoader")
    if args.mask_weight and args.train_source == "train":
        ap.error("--mask-weight needs is_scored, which train.parquet does not have")
    if args.include_heldout_valid and (
        args.valid_fold >= 0 or args.train_source not in ("train+valid", "valid")
    ):
        ap.error("--include-heldout-valid is only for a final train+valid or valid fit")
    if not 0 < args.valid_train_fraction <= 1:
        ap.error("--valid-train-fraction must be in (0, 1]")
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        ap.error("--amp-init-scale must be finite and positive")
    if args.valid_train_fraction < 1 and args.train_source not in ("train+valid", "valid"):
        ap.error("--valid-train-fraction requires a validation training source")
    if args.epoch_samples < 0 or (args.epoch_samples and args.valid_repeat != 1):
        ap.error("--epoch-samples must be nonnegative and requires --valid-repeat 1")
    if args.epoch_samples and args.class_balance:
        ap.error("--epoch-samples cannot be combined with --class-balance")
    update_rows = args.update_rows or args.chunk
    detach_rows = args.detach_rows or args.chunk
    if not 0 <= args.thin_prob <= 1:
        ap.error("--thin-prob must lie in [0, 1]")
    if args.thin_prob:
        try:
            thin_low, thin_high = map(float, args.thin_range.split(","))
        except ValueError:
            ap.error("--thin-range requires two drop shares low,high")
        if not 0 <= thin_low <= thin_high <= 1:
            ap.error("--thin-range drop shares must satisfy 0 <= low <= high <= 1")
    run.options.detach_rows = detach_rows
    run.data.frontier_contract = frontier_contract
    run.options.frontier_contract_sha256 = frontier_contract_sha256
    run.options.frontier_stage = frontier_stage
    run.options.thin_high = thin_high
    run.options.thin_low = thin_low
    run.options.update_rows = update_rows
