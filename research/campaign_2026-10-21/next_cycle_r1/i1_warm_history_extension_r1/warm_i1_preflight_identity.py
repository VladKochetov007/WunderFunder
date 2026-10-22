from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from wnn.training.state import TrainingRun

FEATURE_PRIOR_SOURCES = {
    "src/wnn/research_features.py",
    "src/wnn/i1_history.py",
    "src/wnn/i1_history_scan.py",
    "src/wnn/i1_history_streaming.py",
    "src/wnn/features.py",
    "src/wnn/training/inputs.py",
}


def sha256(path: Path) -> str:
    """Hash a file with a fixed-size read buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bounded_json(path: Path, max_bytes: int = 2_000_000) -> dict[str, object]:
    """Read a metadata JSON file only after bounding and checking its size."""
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"metadata JSON has unsupported size: {path} ({size} bytes)")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"metadata JSON changed while it was read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"metadata JSON must be an object: {path}")
    return cast(dict[str, object], value)


def verify_recipe_and_files(
    args: argparse.Namespace, cell: dict[str, object], project_root: Path
) -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    """Bind selected command arguments and small parent/propensity metadata to a cell."""
    parent_spec = cast(dict[str, object], cell["parent"])
    cache_spec = cast(dict[str, object], cell["target_cache"])
    propensity_spec = cast(dict[str, object], cell["propensity"])
    split_spec = cast(dict[str, object], cell["split"])

    parent_epoch = Path(str(parent_spec["checkpoint"])).stem.removeprefix("model_ep")
    if not parent_epoch.isdigit():
        raise ValueError("frozen parent checkpoint name has no model epoch")
    expected_init = f"{parent_spec['run']}@ep{parent_epoch}"
    if args.init != expected_init:
        raise ValueError(f"initializer command differs from frozen parent: {args.init}")
    if args.propensity != propensity_spec["run"]:
        raise ValueError("propensity command differs from the frozen cell")
    expected_cache_root = (project_root / str(cache_spec["root"])).resolve()
    if Path(args.distill_cache_root).resolve() != expected_cache_root:
        raise ValueError("distillation cache root differs from the frozen cell")
    if args.distill_cache_variant != cache_spec["variant"]:
        raise ValueError("distillation cache variant differs from the frozen cell")
    if Path(args.split_manifest).resolve() != (project_root / str(split_spec["path"])).resolve():
        raise ValueError("split path differs from the frozen cell")
    if args.valid_fold != cell["fold"] or args.train_source != "train+valid":
        raise ValueError("preflight command does not use the frozen train+valid fold")
    if (
        args.train_seqs != 10607
        or args.valid_train_fraction != 1
        or args.valid_repeat != 1
        or args.seed != cell["fit_seed"]
        or args.epochs != 2
        or args.batch != 16
        or args.chunk != 2000
        or args.hidden != 256
        or args.layers != 2
        or args.head_layers != 1
        or args.dropout != 0
        or args.arch != "gru"
        or args.cell != "gru"
        or args.enc != "linear"
        or args.batch_order != "random"
        or args.checkpoint_policy != "fixed"
        or args.checkpoint_epoch != 1
        or args.loss != "wmse"
        or args.loss_norm != "chunk"
        or args.mean_variance_head != 1
        or args.optim != "adamw"
        or args.lr != 3e-5
        or args.wd != 0.01
        or args.beta2 != 0.999
        or args.sched != "onecycle"
        or args.warmup_frac != 0.05
        or args.clip != 1.0
        or args.distill_alpha != 0.9
        or args.distill_cache_scale != 1.0
        or args.prop_power != 2
        or args.prop_eps != 0.03
        or args.valid_true_mask != 0.03
        or args.aux_next != 0.3
        or args.aux_struct != 0.3
    ):
        raise ValueError("preflight command changed complete fitting membership")
    if (
        args.precision != "fp32"
        or args.device != "cpu"
        or args.research_features != "i1_history"
        or args.i1_warm_normalization != "fit-appended"
        or args.drop_i1
        or args.research_feature_control not in ("all", "none")
        or args.resume_state
        or args.save_training_state
        or not args.no_eval_during_fit
    ):
        raise ValueError("preflight command no longer matches the supported warm setup")

    parent_checkpoint = project_root / str(parent_spec["checkpoint"])
    if sha256(parent_checkpoint) != parent_spec["checkpoint_sha256"]:
        raise ValueError("actual parent checkpoint bytes differ from the frozen plan")
    parent_provenance = (
        project_root / "artifacts" / "runs" / str(parent_spec["run"]) / "provenance.json"
    )
    if sha256(parent_provenance) != parent_spec["provenance_sha256"]:
        raise ValueError("actual parent provenance bytes differ from the frozen plan")
    parent_record = read_bounded_json(parent_provenance)
    if (
        parent_record.get("initial_checkpoint_sha256")
        != parent_spec["initial_ancestor_checkpoint_sha256"]
    ):
        raise ValueError("actual parent initializer ancestry differs from the frozen plan")

    propensity_checkpoint = (
        project_root / "artifacts" / "runs" / str(propensity_spec["run"]) / "model.pt"
    )
    propensity_provenance = propensity_checkpoint.parent / "provenance.json"
    if sha256(propensity_checkpoint) != propensity_spec["checkpoint_sha256"]:
        raise ValueError("actual propensity checkpoint bytes differ from the frozen plan")
    if sha256(propensity_provenance) != propensity_spec["provenance_sha256"]:
        raise ValueError("actual propensity provenance bytes differ from the frozen plan")
    return parent_spec, cache_spec, propensity_spec, split_spec, parent_record


def verify_setup_provenance(
    provenance: dict[str, object],
    plan: dict[str, object],
    parent_spec: dict[str, object],
    cache_spec: dict[str, object],
    propensity_spec: dict[str, object],
    split_spec: dict[str, object],
    parent_record: dict[str, object],
    run: TrainingRun,
) -> tuple[dict[str, object], dict[str, str]]:
    """Recheck actual fold, target cache, feature code and fitting membership."""
    split = run.data.campaign_split
    if split is None or split.sha256 != split_spec["sha256"]:
        raise ValueError("canonical setup did not load the exact frozen split")
    parent_groups = cast(dict[str, list[int]], parent_record["fit_row_groups"])
    if set(parent_groups["train"]) != set(split.train_groups) or set(parent_groups["valid"]) != set(
        split.valid_groups
    ):
        raise ValueError("actual parent fitting membership differs from the source split")
    if provenance.get("initial_checkpoint_sha256") != parent_spec["checkpoint_sha256"]:
        raise ValueError("loaded parent checkpoint SHA differs from the frozen plan")
    if provenance.get("propensity_checkpoint_sha256") != propensity_spec["checkpoint_sha256"]:
        raise ValueError("loaded propensity checkpoint SHA differs from the frozen plan")
    if provenance.get("split_manifest_sha256") != split_spec["sha256"]:
        raise ValueError("loaded split SHA differs from the frozen plan")

    cache = cast(dict[str, object], provenance["distillation_cache"])
    verified = {
        f"main_{cache_spec['variant']}_train": cache_spec["main_train_sha256"],
        f"main_{cache_spec['variant']}_valid": cache_spec["main_valid_sha256"],
    }
    expected_root = str((run.environment.root / str(cache_spec["root"])).resolve())
    if (
        cache.get("root") != expected_root
        or cache.get("metadata_sha256") != cache_spec["metadata_sha256"]
        or cache.get("verified_files") != verified
    ):
        raise ValueError("loaded target-cache bytes or metadata differ from the frozen plan")

    source = cast(dict[str, object], plan["source"])
    source_pins = cast(dict[str, str], source["current_source_sha256"])
    prior_sources = cast(dict[str, str], provenance["causal_feature_prior_source_sha256"])
    if set(prior_sources) != FEATURE_PRIOR_SOURCES:
        raise ValueError("feature-prior source set differs from the canonical i1 implementation")
    if prior_sources != {name: source_pins[name] for name in FEATURE_PRIOR_SOURCES}:
        raise ValueError("feature-prior implementation hashes differ from pinned source")
    if len(run.data.train_idx) != split_spec["train_fit_groups"]:
        raise ValueError("selected train fitting membership count differs from the plan")
    if run.data.extra_valid is None or len(run.data.extra_valid) != split_spec["valid_fit_groups"]:
        raise ValueError("selected valid fitting membership count differs from the plan")
    if len(split.evaluation_groups) != split_spec["evaluation_groups"]:
        raise ValueError("evaluation fold size differs from the frozen split")
    membership = cast(dict[str, object], provenance["fit_membership_identity"])
    train_identity = cast(dict[str, object], membership["train"])
    valid_identity = cast(dict[str, object], membership["valid"])
    if (
        train_identity.get("ordered_int64_le_sha256") != split_spec["fit_train_order_sha256"]
        or valid_identity.get("ordered_int64_le_sha256") != split_spec["fit_valid_order_sha256"]
        or membership.get("valid_repeat") != 1
        or membership.get("dataset_length")
        != split_spec["train_fit_groups"] + split_spec["valid_fit_groups"]
    ):
        raise ValueError("selected fitting order or dataset length differs from the frozen plan")
    return cache, prior_sources
