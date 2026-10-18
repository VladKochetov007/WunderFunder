from __future__ import annotations

import json
from pathlib import Path

from wnn import student_frontier
from wnn.distillation_cache import validate_distillation_cache
from wnn.outcomes import N_SP_AUX
from wnn.splits import file_sha256
from wnn.spread_aux import validate_spread_cache
from wnn.training.cache_identity import bind_fold_teachers, prop_cache_path
from wnn.training.state import TrainingRun


def bind_caches(run: TrainingRun) -> None:
    """Validate teacher, propensity and auxiliary caches against the selected fitting membership."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    campaign_split = run.data.campaign_split
    distill_cache_validation = run.data.distill_cache_validation
    extra_valid = run.data.extra_valid
    frontier_cache_binding = run.data.frontier_cache_binding
    frontier_contract = run.data.frontier_contract
    frontier_contract_sha256 = run.options.frontier_contract_sha256
    prop_cache = run.data.prop_cache
    quote_cache_provenance = run.data.quote_cache_provenance
    spread_provenance = run.data.spread_provenance
    teachers = run.data.teachers
    thin = run.data.thin
    train_idx = run.data.train_idx
    train_split = run.data.train_split
    teachers = None
    if args.distill:
        teachers = [
            (t.partition("*")[0], float(t.partition("*")[2] or 1.0))
            for t in args.distill.split(",")
        ]
        if campaign_split is not None:
            used = {"train": ("train",), "valid": ("valid",)}.get(
                args.train_source, ("train", "valid")
            )
            args.distill_provenance = bind_fold_teachers(
                teachers, campaign_split, used, environment=environment
            )
        else:
            for name, _ in teachers:
                for split in ("train", "valid"):
                    sidecar = environment.root / "cache" / f"pred_{name}_{split}.json"
                    if sidecar.exists() and "crossfit_format" in json.loads(sidecar.read_text()):
                        ap.error(
                            "component-crossfit teachers require their matching --split-manifest"
                        )
    prop_cache = {}
    if args.prop_cache and args.propensity:
        prop_cache = {
            sp: prop_cache_path(
                args.propensity, sp, environment=environment, campaign_split=campaign_split
            )
            for sp in ("train", "valid")
        }
    quote_cache_provenance = None
    if args.quote_cache_root:
        if campaign_split is None:
            ap.error("--quote-cache-root requires a clean --split-manifest")
        cache_root = Path(args.quote_cache_root)
        if not cache_root.is_absolute():
            cache_root = environment.root / cache_root
        cache_root = cache_root.resolve()
        metadata_path = cache_root.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("split_manifest_sha256") != campaign_split.sha256
            or set(metadata.get("fit_train_groups", [])) != set(campaign_split.train_groups)
            or set(metadata.get("fit_valid_groups", [])) != set(campaign_split.valid_groups)
            or (
                set(metadata.get("evaluation_valid_groups", []))
                != set(campaign_split.evaluation_groups)
            )
        ):
            raise ValueError("quote-cache provenance disagrees with clean fold")
        if not set(train_idx).issubset(
            metadata["fit_train_groups"] if train_split == "train" else metadata["fit_valid_groups"]
        ) or (
            extra_valid is not None
            and (not set(extra_valid).issubset(metadata["fit_valid_groups"]))
        ):
            raise ValueError("training row groups are outside the quote cache fit pool")
        cache_files = {}
        for split in ("train", "valid"):
            path = cache_root / f"quote_path_{split}.npy"
            entry = metadata["files"][f"{cache_root.name}/{split}"]
            if not path.is_file() or len(entry["sha256"]) != 64:
                raise ValueError(f"missing or unproven quote cache: {path}")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"quote cache content hash disagrees with metadata: {path}")
            cache_files[split] = entry["sha256"]
        args.quote_cache_root = str(cache_root)
        quote_cache_provenance = {
            "metadata_sha256": file_sha256(metadata_path),
            "array_sha256_expected": cache_files,
            "variant": cache_root.name,
        }
    spread_provenance = None
    if args.spread_aux_mode != "original":
        if (
            campaign_split is None
            or args.aux_struct <= 0
            or (not args.spread_cache_root)
            or (args.train_source != "train+valid")
            or (not args.init)
            or (not args.inherit_init_preprocessing)
            or (args.checkpoint_policy != "fixed")
            or (args.checkpoint_epoch != args.epochs - 1)
            or (not args.no_eval_during_fit)
        ):
            raise ValueError(
                "known-code spread screen requires a fixed final-epoch clean continuation with inherited parent preprocessing"
            )
        spread_root = Path(args.spread_cache_root)
        if not spread_root.is_absolute():
            spread_root = environment.root / spread_root
        spread_root = spread_root.resolve()
        spread_provenance = validate_spread_cache(
            spread_root,
            campaign_split.sha256,
            train_idx,
            extra_valid,
            campaign_split.evaluation_groups,
            sequence_rows=environment.sequence_length,
        )
        if json.loads((spread_root / "metadata.json").read_text())["N_SP_AUX"] != N_SP_AUX:
            raise ValueError("spread cache head dimension differs")
        spread_provenance["mode"] = args.spread_aux_mode
        spread_provenance["source_sha256"] = file_sha256(
            environment.source_root / "src/wnn/spread_aux.py"
        )
        args.spread_cache_root = str(spread_root)
    elif args.spread_cache_root:
        raise ValueError("spread cache supplied without a matched known-code mode")
    distill_cache_validation = None
    frontier_cache_binding = None
    if args.distill_cache_root:
        cache_root = Path(args.distill_cache_root)
        if not cache_root.is_absolute():
            cache_root = environment.root / cache_root
        cache_root = cache_root.resolve()
        distill_cache_validation = validate_distillation_cache(
            cache_root,
            environment.root,
            campaign_split,
            args.distill_cache_variant,
            require_families=args.distill_family_weight > 0,
        )
        args.distill_cache_root = str(cache_root)
        if frontier_contract is not None:
            frontier_cache_binding = student_frontier.validate_cache_binding(
                environment.root,
                frontier_contract,
                frontier_contract_sha256,
                campaign_split,
                distill_cache_validation,
            )
            frontier_cache_binding["stage"] = args.student_frontier_stage
    thin = None
    run.data.distill_cache_validation = distill_cache_validation
    run.data.frontier_cache_binding = frontier_cache_binding
    run.data.prop_cache = prop_cache
    run.data.quote_cache_provenance = quote_cache_provenance
    run.data.spread_provenance = spread_provenance
    run.data.teachers = teachers
    run.data.thin = thin
