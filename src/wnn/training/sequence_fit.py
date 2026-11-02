from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from wnn.training.sequence_config import (
    load_config,
    validate_expected_fold,
    validate_runtime_requirements,
)
from wnn.training.sequence_data import SequenceData, build_data
from wnn.training.sequence_distributed import (
    DistributedContext,
    collective_output_setup,
    coordinated_call,
    initialize_distributed,
    shutdown_distributed,
    wrap_distributed,
)
from wnn.training.sequence_fit_modes import run_full_fit
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    make_optimizer,
    seed_worker,
    total_updates,
)
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_optimizer_telemetry import optimizer_contract
from wnn.training.sequence_profile import run_profile
from wnn.training.sequence_provenance import (
    assert_same_identity,
    recorder_for_run,
    run_identity,
    runtime_identity,
    source_identity,
    validate_profile_receipt,
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Validate the heterogeneous mapping boundary returned by the config loader."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _prepare_model(
    config: dict[str, Any], data: SequenceData, context: DistributedContext
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """Build strict-FP32 weights and install fitting-only normalization buffers."""
    model = build_sequence_model(config["model"])
    if not isinstance(model, torch.nn.Module):
        raise TypeError("sequence model builder must return a torch.nn.Module")
    model = model.to(device=context.device, dtype=torch.float32)
    mean, std = data.normalization()
    mean = mean.to(device=context.device, dtype=torch.float32).reshape(-1)
    std = std.to(device=context.device, dtype=torch.float32).reshape(-1)
    valid_normalization = (
        mean.shape == (112,)
        and std.shape == (112,)
        and bool(torch.isfinite(mean).all())
        and bool(torch.isfinite(std).all())
        and bool((std > 0).all())
    )
    if not valid_normalization:
        raise ValueError("normalization must contain 112 finite means and positive FP32 scales")
    model_mean = getattr(model, "mean", None)
    model_std = getattr(model, "std", None)
    if not isinstance(model_mean, torch.Tensor) or not isinstance(model_std, torch.Tensor):
        raise TypeError("sequence model must expose mean and std normalization buffers")
    with torch.no_grad():
        model_mean.copy_(mean)
        model_std.copy_(std)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("all trainable and frozen model parameters must use strict FP32")
    if any(
        buffer.is_floating_point() and buffer.dtype != torch.float32 for buffer in model.buffers()
    ):
        raise TypeError("floating sequence-model buffers must use strict FP32")
    normalization = {"mean": mean.detach().cpu().tolist(), "std": std.detach().cpu().tolist()}
    return model, normalization


def run_sequence_fit(
    config_path: Path,
    data_root: Path,
    split_manifest: Path,
    output: Path,
    entrypoint: Path,
    teacher_cache: Path | None = None,
    resume: Path | None = None,
    profile_updates: int = 0,
    profile_receipt: Path | None = None,
) -> None:
    """Execute one torchrun fit or a non-scientific bounded profile on supplied data."""
    config = _mapping(load_config(config_path), "config")
    output = output.resolve()
    resume_path = resume.resolve() if resume is not None else None
    receipt_path = profile_receipt.resolve() if profile_receipt is not None else None
    if resume_path is not None and resume_path != output / "training_state.pt":
        raise ValueError("--resume must name <output>/training_state.pt")
    if profile_updates < 0 or (profile_updates and (resume_path or receipt_path)):
        raise ValueError("profile runs cannot resume or consume a profile admission receipt")
    if not profile_updates and receipt_path is None:
        raise ValueError("a full fit requires --profile-receipt with explicit approval")
    context = initialize_distributed()
    try:
        coordinated_call(
            lambda: validate_runtime_requirements(
                config,
                context.world_size,
                torch.cuda.get_device_name(context.local_rank),
            ),
            "runtime requirement validation",
            context,
        )
        collective_output_setup(output, resume_path is not None, context)
        seed_worker(int(config["seed"]), 0)
        data = coordinated_call(
            lambda: build_data(config, data_root, split_manifest, teacher_cache),
            "sequence data construction",
            context,
        )
        coordinated_call(
            lambda: validate_expected_fold(config, data.split.fold),
            "split fold validation",
            context,
        )
        seed_worker(int(config["seed"]), 0)
        base_model, normalization = coordinated_call(
            lambda: _prepare_model(config, data, context), "model construction", context
        )
        model = wrap_distributed(base_model, context)
        seed_worker(int(config["seed"]), context.rank)
        fit_dataset = cast(SequenceDataset, data.fit)
        schedule_updates = total_updates(config, len(fit_dataset), context.world_size)
        if schedule_updates < 1:
            raise ValueError("configured fit has no optimizer updates")
        optimizer, scheduler = make_optimizer(base_model, config, schedule_updates)
        root = Path(__file__).resolve().parents[3]
        source = source_identity(root, entrypoint)
        runtime = runtime_identity(context)
        identity = run_identity(config, data.provenance, source, runtime, context.world_size)
        identity["model_parameters"] = {
            "total": sum(parameter.numel() for parameter in base_model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in base_model.parameters()
                if parameter.requires_grad
            ),
        }
        identity["optimizer"] = optimizer_contract(optimizer, config, schedule_updates)
        assert_same_identity(identity, context)
        if receipt_path is None:
            profile_hash = None
        else:
            profile_hash = coordinated_call(
                lambda: validate_profile_receipt(receipt_path, identity),
                "profile admission validation",
                context,
            )
        recorder_result = coordinated_call(
            lambda: recorder_for_run(output, config, identity) if context.is_primary else False,
            "research run metadata setup",
            context,
        )
        recorder = recorder_result if context.is_primary else None
        rank_log = coordinated_call(
            lambda: RankJsonlLog(
                output / f"rank{context.rank}.jsonl", resume=resume_path is not None
            ),
            "rank-local log setup",
            context,
        )
        try:
            if profile_updates:
                run_profile(
                    fit_dataset,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    context,
                    rank_log,
                    recorder,
                    identity,
                    output,
                    profile_updates,
                    schedule_updates,
                )
                if config.get("disagreement_mode") is not None:
                    coordinated_call(
                        data.fit.cache.verify_unchanged,
                        "disagreement source recheck after profile",
                        context,
                    )
                return
            if receipt_path is None or profile_hash is None:
                raise ValueError("full fit profile admission was not validated")
            run_full_fit(
                fit_dataset,
                model,
                base_model,
                optimizer,
                scheduler,
                config,
                context,
                rank_log,
                recorder,
                identity,
                normalization,
                data.provenance,
                output,
                resume_path,
                receipt_path,
                profile_hash,
            )
        finally:
            rank_log.close()
    finally:
        shutdown_distributed(context)
