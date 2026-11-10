from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_config import (
    load_config,
    validate_runtime_requirements,
)
from wnn.training.sequence_distributed import (
    collective_output_setup,
    coordinated_call,
    initialize_distributed,
    shutdown_distributed,
    wrap_distributed,
)
from wnn.training.sequence_fit_admission import validate_fit_reference_inputs
from wnn.training.sequence_fit_data import prepare_sequence_fit_data
from wnn.training.sequence_fit_model import prepare_fit_model
from wnn.training.sequence_fit_modes import run_full_fit
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    make_optimizer,
    seed_worker,
    total_updates,
)
from wnn.training.sequence_initialization import (
    initialize_from_parent_receipt,
)
from wnn.training.sequence_monitor_progress import MonitorProgress, close_monitor_progress
from wnn.training.sequence_optimizer_telemetry import optimizer_contract
from wnn.training.sequence_pretraining_receipt import (
    write_pretraining_parent_receipt,
)
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
    reference_cache: Path | None = None,
    monitor_reference_root: Path | None = None,
    initializer_receipt: Path | None = None,
) -> None:
    """Execute one torchrun fit or a non-scientific bounded profile on supplied data."""
    config = _mapping(load_config(config_path), "config")
    validate_fit_reference_inputs(config, teacher_cache, reference_cache, monitor_reference_root)
    output = output.resolve()
    resume_path = resume.resolve() if resume is not None else None
    receipt_path = profile_receipt.resolve() if profile_receipt is not None else None
    if initializer_receipt is not None and initializer_receipt.is_symlink():
        raise ValueError("--initializer-receipt must name a regular, non-symlink file")
    initializer_path = initializer_receipt.resolve() if initializer_receipt is not None else None
    if (config.get("parent_initializer") is not None) != (initializer_path is not None):
        raise ValueError("--initializer-receipt must match the declared parent_initializer config")
    if config.get("pretraining") is not None and initializer_path is not None:
        raise ValueError("future-input pretraining cannot consume a parent initializer")
    if config.get("pretraining") is not None and any(
        value is not None for value in (teacher_cache, reference_cache, monitor_reference_root)
    ):
        raise ValueError("future-input pretraining rejects teacher and validation-reference inputs")
    if resume_path is not None and resume_path != output / "training_state.pt":
        raise ValueError("--resume must name <output>/training_state.pt")
    if profile_updates < 0 or (profile_updates and (resume_path or receipt_path)):
        raise ValueError("profile runs cannot resume or consume a profile admission receipt")
    if not profile_updates and receipt_path is None:
        raise ValueError("a full fit requires --profile-receipt with explicit approval")
    context = initialize_distributed()
    monitor: ComplementMonitor | None = None
    pretraining_progress: MonitorProgress | None = None
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
        if config.get("pretraining") is not None and context.is_primary:
            pretraining_progress = MonitorProgress(output)
            pretraining_progress.phase("pretraining_setup", 0, 0)
        seed_worker(int(config["seed"]), 0)
        data, monitor = prepare_sequence_fit_data(
            config,
            data_root,
            split_manifest,
            output,
            context,
            teacher_cache=teacher_cache,
            reference_cache=reference_cache,
            monitor_reference_root=monitor_reference_root,
            resume=resume_path is not None,
            profile_updates=profile_updates,
        )
        seed_worker(int(config["seed"]), 0)
        base_model, normalization = coordinated_call(
            lambda: prepare_fit_model(config, data, context), "model construction", context
        )
        root = Path(__file__).resolve().parents[3]
        source = source_identity(root, entrypoint)
        if initializer_path is not None:
            parent_identity = coordinated_call(
                lambda: initialize_from_parent_receipt(
                    base_model,
                    initializer_path,
                    config,
                    data.provenance,
                    normalization,
                    cast(dict[str, str], source["files"]),
                ),
                "future-innovation parent validation and backbone initialization",
                context,
            )
            data.provenance["initializer"] = parent_identity
        model = wrap_distributed(base_model, context)
        seed_worker(int(config["seed"]), context.rank)
        fit_dataset = cast(SequenceDataset, data.fit)
        schedule_updates = total_updates(config, len(fit_dataset), context.world_size)
        if schedule_updates < 1:
            raise ValueError("configured fit has no optimizer updates")
        optimizer, scheduler = make_optimizer(base_model, config, schedule_updates)
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
                    progress=(monitor.progress if monitor is not None else pretraining_progress),
                )
                if monitor is not None and monitor.progress is not None:
                    monitor.progress.phase("profile_completed", profile_updates, 0)
                if pretraining_progress is not None:
                    pretraining_progress.phase("profile_completed", profile_updates, 0)
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
                monitor=monitor,
                progress=pretraining_progress,
            )
            if config.get("pretraining") is not None:
                coordinated_call(
                    lambda: (
                        write_pretraining_parent_receipt(
                            output, identity, config, data.provenance, normalization
                        )
                        if context.is_primary
                        else True
                    ),
                    "future-innovation parent receipt publication",
                    context,
                )
        finally:
            rank_log.close()
    finally:
        close_monitor_progress(monitor)
        if pretraining_progress is not None:
            pretraining_progress.close()
        shutdown_distributed(context)
