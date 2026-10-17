from __future__ import annotations

import json
from pathlib import Path

import torch

from wnn.propensity import load_propensity
from wnn.splits import file_sha256
from wnn.training.checkpoints import serialized_args
from wnn.training.inheritance import validate_campaign_checkpoint
from wnn.training.source_identity import training_source_identity
from wnn.training.state import TrainingRun


def record_provenance(run: TrainingRun) -> None:
    """Record source-qualified fitting, cache and parent identities before training."""
    environment = run.environment
    args = run.options.args
    campaign_split = run.data.campaign_split
    device = run.options.device
    distill_cache_validation = run.data.distill_cache_validation
    extra_valid = run.data.extra_valid
    fit_subset = run.data.fit_subset
    frontier_cache_binding = run.data.frontier_cache_binding
    frontier_contract = run.data.frontier_contract
    frontier_contract_sha256 = run.options.frontier_contract_sha256
    frontier_order = run.data.frontier_order
    frontier_stage = run.options.frontier_stage
    inner_crossfit = run.data.inner_crossfit
    prop_cache = run.data.prop_cache
    prop_model = run.data.prop_model
    provenance = run.data.provenance
    quote_cache_provenance = run.data.quote_cache_provenance
    shifts = run.data.shifts
    spread_provenance = run.data.spread_provenance
    train_idx = run.data.train_idx
    train_split = run.data.train_split
    valid_idx = run.data.valid_idx
    shifts = [int(v) for v in args.aux.split(",") if v]
    provenance = {
        "seed": args.seed,
        "args": serialized_args(args),
        "split_manifest_sha256": campaign_split.sha256 if campaign_split is not None else None,
        "fit_row_groups": {
            "train": train_idx.tolist() if train_split == "train" else [],
            "valid": train_idx.tolist()
            if train_split == "valid"
            else extra_valid.tolist()
            if extra_valid is not None
            else [],
        },
        "evaluation_row_groups": valid_idx.tolist(),
        "checkpoint_sha256": {},
    }
    provenance["trainer_implementation"] = training_source_identity(environment)
    if fit_subset is not None:
        provenance["fit_subset_manifest_sha256"] = args.fit_subset_manifest_sha256
        provenance["fit_subset_members_sha256"] = fit_subset["manifest"]["selected_members_sha256"]
    if distill_cache_validation is not None:
        cache_metadata = distill_cache_validation["metadata"]
        provenance["distillation_cache"] = {
            "root": args.distill_cache_root,
            "variant": args.distill_cache_variant,
            "main_target_scale": args.distill_cache_scale,
            "main_target_scale_applied_before_training_clip": True,
            "metadata_sha256": distill_cache_validation["metadata_sha256"],
            "verified_files": distill_cache_validation["verified_files"],
            "source_coefficients": cache_metadata["source_coefficients"],
            "coefficient_sum": cache_metadata["coefficient_sum"],
            "homogeneous_family": cache_metadata["homogeneous_family"],
            "family_order": cache_metadata["family_order"],
            "epoch_convention": cache_metadata["epoch_convention"],
        }
    if frontier_contract is not None:
        frontier_sources = (
            Path(environment.entrypoint).resolve(),
            environment.source_root / "src/wnn/student_frontier.py",
            environment.source_root / "src/wnn/student_frontier_order.py",
            environment.source_root / "src/wnn/training_sampling.py",
            environment.source_root / "src/wnn/nn.py",
            environment.source_root / "src/wnn/features.py",
            environment.source_root / "src/wnn/distillation_cache.py",
            environment.source_root / "src/wnn/splits.py",
            environment.source_root / "src/wnn/propensity.py",
            environment.source_root / "src/wnn/training_dynamics.py",
            environment.source_root / "src/wnn/optim.py",
        )
        provenance["student_frontier"] = {
            "contract_id": frontier_contract["id"],
            "contract_sha256": frontier_contract_sha256,
            "stage": args.student_frontier_stage,
            "stage_settings": frontier_stage,
            "cache_binding": frontier_cache_binding,
            "sample_order": frontier_order,
            "source_sha256": {
                str(path.relative_to(environment.source_root)): file_sha256(path)
                for path in frontier_sources
            },
        }
    if inner_crossfit is not None:
        provenance["inner_crossfit"] = inner_crossfit
        provenance["deployment_safe"] = args.privileged_mode == "none"
    if args.readout_activation != "gelu":
        provenance["native_export_supported"] = False
        provenance["readout_activation"] = args.readout_activation
    if args.arch == "structured":
        provenance["model_source_sha256"] = file_sha256(
            environment.source_root / "src/wnn/structured_state.py"
        )
    if spread_provenance is not None:
        provenance["spread_aux"] = spread_provenance
    if args.arch == "event-content":
        provenance["model_source_sha256"] = file_sha256(
            environment.source_root / "src/wnn/event_content.py"
        )
    if args.arch in ("tcn", "twoclock"):
        provenance["model_source_sha256"] = file_sha256(environment.source_root / "src/wnn/nn.py")
    if args.arch in ("specialist", "prefix-adapter"):
        provenance["model_source_sha256"] = file_sha256(
            environment.source_root / "src/wnn/specialist.py"
        )
    if quote_cache_provenance is not None:
        provenance["quote_cache"] = quote_cache_provenance
    if args.propensity:
        propensity_path = environment.root / "artifacts" / "runs" / args.propensity / "model.pt"
        if campaign_split is not None:
            validate_campaign_checkpoint(
                torch.load(propensity_path, map_location="cpu", weights_only=False), args
            )
        provenance["propensity_checkpoint_sha256"] = file_sha256(propensity_path)
        if args.prop_cache and args.profile_steps and (args.activation_backend != "none"):
            provenance["propensity_cache"] = {
                split: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "metadata_sha256": file_sha256(path.with_suffix(".json")),
                    "metadata": json.loads(path.with_suffix(".json").read_text()),
                }
                for split, path in prop_cache.items()
            }
    prop_model = (
        load_propensity(
            environment.root / "artifacts" / "runs" / args.propensity / "model.pt", device
        )
        if args.propensity
        else None
    )
    run.data.prop_model = prop_model
    run.data.provenance = provenance
    run.data.shifts = shifts
