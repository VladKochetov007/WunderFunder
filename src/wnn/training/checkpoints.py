from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wnn.research_features import MOMENT_FEATURE_MODES, validate_causal_moment_prior
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.averaging import FastEMA
from wnn.training.cache_identity import prop_cache_path
from wnn.training.environment import TrainingEnvironment
from wnn.training_factorized import factorized_inference_state
from wnn.training_qat import inference_state
from wnn.training_state import loader_state


def configure_checkpoint_policy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Require predeclared checkpoint selection when evaluation is withheld."""
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.checkpoint_policy == "fixed":
        if args.checkpoint_epoch is None:
            args.checkpoint_epoch = args.epochs - 1
        if not 0 <= args.checkpoint_epoch < args.epochs:
            parser.error("--checkpoint-epoch must be within the fitted epochs")
    elif args.checkpoint_epoch is not None:
        parser.error("--checkpoint-epoch requires --checkpoint-policy fixed")
    if args.no_eval_during_fit and args.checkpoint_policy != "fixed":
        parser.error("--no-eval-during-fit requires --checkpoint-policy fixed")
    if args.split_manifest and (
        not (args.no_eval_during_fit and args.checkpoint_policy == "fixed")
    ):
        parser.error("campaign manifests require fixed checkpoints and --no-eval-during-fit")


def checkpoint_selected(
    epoch: int,
    policy: str,
    fixed_epoch: int | None,
    evaluation_score: float | None,
    best_score: float,
) -> bool:
    """Select a fixed epoch without consulting evaluation, or retain legacy best selection."""
    if policy == "fixed":
        return epoch == fixed_epoch
    if evaluation_score is None:
        raise ValueError("best-checkpoint selection requires evaluation")
    return evaluation_score > best_score


def checkpoint_model(
    model: torch.nn.Module,
    ema_model: FastEMA | torch.optim.swa_utils.AveragedModel | None,
    swa_model: torch.optim.swa_utils.AveragedModel | None,
    use_ema: bool,
) -> torch.nn.Module:
    """Choose fitted snapshot weights independently of whether evaluation runs."""
    if use_ema:
        if ema_model is None:
            raise ValueError("EMA checkpoint requested without an EMA model")
        return ema_model.module
    return swa_model.module if swa_model is not None else model


def serialized_args(args: argparse.Namespace) -> dict[str, object]:
    """Return CLI arguments with path values safe for JSON and checkpoints."""
    return {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
    }


def inference_checkpoint_fields(
    model: torch.nn.Module, source: str, *, environment: TrainingEnvironment | None = None
) -> dict:
    """Export effective weights and representation metadata without changing raw resume state."""
    environment = environment if environment is not None else TrainingEnvironment()
    if getattr(model, "_recurrent_rank", 0):
        exported, metadata = factorized_inference_state(model, source)
        result = {"model": exported, "factorization": metadata}
    else:
        exported, metadata = inference_state(model, source)
        result = {"model": exported, **({"quantization": metadata} if metadata is not None else {})}
    if hasattr(model, "_sentinel_preprocessing"):
        result["sentinel_preprocessing"] = model._sentinel_preprocessing
    if environment.inputs.research_features in MOMENT_FEATURE_MODES:
        if environment.inputs.research_feature_prior is None:
            raise ValueError("causal moment checkpoint is missing its fitting-only prior")
        result["research_feature_prior"] = validate_causal_moment_prior(
            environment.inputs.research_feature_prior
        )
    return result


def resume_contract(
    args: argparse.Namespace,
    provenance: dict,
    loader: DataLoader,
    total_steps: int,
    *,
    environment: TrainingEnvironment | None = None,
) -> dict:
    """Bind an epoch continuation to its original data, code and fixed schedule."""
    environment = environment if environment is not None else TrainingEnvironment()
    excluded = {
        "name",
        "resume_state",
        "save_training_state",
        "stop_after_epoch",
        "split_manifest",
        "fit_groups_manifest",
    }
    options = {key: value for key, value in vars(args).items() if key not in excluded}
    package = Path(environment.entrypoint).resolve().parents[1] / "src/wnn"
    sources = {
        str(path.relative_to(package)): file_sha256(path) for path in sorted(package.rglob("*.py"))
    }
    sources["trainer"] = file_sha256(Path(environment.entrypoint))
    data = {}
    used_splits = {split for split, groups in provenance["fit_row_groups"].items() if groups}
    for split in used_splits | {"valid"}:
        path = environment.root / "datasets" / f"{split}.parquet"
        registry_path = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
        if registry_path:
            from wnn.portability import verify_portable_source

            registry = Path(registry_path)
            logical_name = f"datasets/{split}.parquet"
            entry = json.loads(registry.read_text())["sources"][logical_name]
            verify_portable_source(path, logical_name, entry["original_identity"], registry)
            data[split] = {"bytes": path.stat().st_size, "sha256": entry["sha256"]}
        else:
            data[split] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    teachers = {}
    for teacher in filter(None, args.distill.split(",")):
        name = teacher.partition("*")[0]
        for split, groups in provenance["fit_row_groups"].items():
            if groups:
                filename = f"pred_{name}_{split}.npy"
                teachers[filename] = file_sha256(environment.root / "cache" / filename)
    caches = {}
    campaign_split = (
        load_campaign_split(Path(args.split_manifest), environment.root)
        if args.prop_cache and args.propensity and args.split_manifest
        else None
    )
    for split in used_splits:
        if args.distill:
            filename = f"{split}_targets.npz"
            caches[filename] = file_sha256(environment.root / "cache" / filename)
        if args.propensity and args.prop_cache:
            path = prop_cache_path(
                args.propensity,
                split,
                environment=environment,
                campaign_split=campaign_split,
            )
            caches[path.name] = file_sha256(path)
            caches[path.with_suffix(".json").name] = file_sha256(path.with_suffix(".json"))
    cache_provenance = provenance.get("distillation_cache")
    if cache_provenance:
        caches["distillation_cache_metadata"] = cache_provenance["metadata_sha256"]
        for name, digest in cache_provenance["verified_files"].items():
            caches[f"distillation_{name}"] = digest
    if args.prop_input:
        caches["input_propensity"] = file_sha256(
            environment.root / "artifacts/runs" / args.prop_input / "model.pt"
        )
    sampling = loader_state(loader)
    for key in ("loader_generator", "sampler_generator", "sampler_state", "sampler_epoch"):
        sampling.pop(key)
    return {
        "epochs": args.epochs,
        "total_steps": total_steps,
        "options": options,
        "sources": sources,
        "datasets": data,
        "teacher_predictions": teachers,
        "consumed_caches": caches,
        "fit_membership": provenance["fit_membership_identity"],
        "normalization": provenance["normalization_identity"],
        "ancestors": {
            key: provenance.get(key)
            for key in (
                "initial_checkpoint_sha256",
                "propensity_checkpoint_sha256",
                "quote_cache",
                "spread_aux",
            )
        },
        "loader": sampling,
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "devices": []
            if args.device == "cpu"
            else [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
    }
