from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from wnn.training.sequence_config import validate_runtime_requirements
from wnn.training.sequence_data import SequenceData
from wnn.training.sequence_distributed import (
    DistributedContext,
    collective_output_setup,
    coordinated_call,
    initialize_distributed,
    shutdown_distributed,
    wrap_distributed,
)
from wnn.training.sequence_fit_admission import prepare_fit_admission, validate_fit_data_contract
from wnn.training.sequence_fit_data_setup import build_fit_data
from wnn.training.sequence_fit_modes import run_full_fit
from wnn.training.sequence_fit_monitor_setup import (
    initialize_monitor,
    prepare_monitor_event_budgets,
    record_monitoring_plan,
    select_train_eval_members,
)
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    make_optimizer,
    seed_worker,
    total_updates,
)
from wnn.training.sequence_initializer import load_fitting_initializer
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor_progress import close_monitor_progress
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
    model_mean = getattr(model, "mean", None)
    model_std = getattr(model, "std", None)
    if not isinstance(model_mean, torch.Tensor) or not isinstance(model_std, torch.Tensor):
        raise TypeError("sequence model must expose mean and std normalization buffers")
    valid_normalization = (
        mean.ndim == 1
        and mean.numel() > 0
        and mean.shape == std.shape == model_mean.shape == model_std.shape
        and bool(torch.isfinite(mean).all())
        and bool(torch.isfinite(std).all())
        and bool((std > 0).all())
    )
    if not valid_normalization:
        raise ValueError(
            "normalization must match model buffers and contain finite positive scales"
        )
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
    split_manifest: Path | None,
    output: Path,
    entrypoint: Path,
    teacher_cache: Path | None = None,
    resume: Path | None = None,
    profile_updates: int = 0,
    profile_receipt: Path | None = None,
    initializer: Path | None = None,
    final_fit_manifest: Path | None = None,
    final_fit_source_validation_receipt: Path | None = None,
    final_fit_source_validation_sha256: str | None = None,
) -> None:
    """Execute one torchrun fit or a non-scientific bounded profile on supplied data."""
    admission = prepare_fit_admission(
        config_path,
        output,
        split_manifest,
        teacher_cache,
        resume,
        profile_updates,
        profile_receipt,
        initializer,
        final_fit_manifest,
        final_fit_source_validation_receipt,
        final_fit_source_validation_sha256,
        data_root,
    )
    config = admission.config
    output = admission.output
    resume_path = admission.resume_path
    receipt_path = admission.receipt_path
    initializer_path = admission.initializer_path
    context = initialize_distributed()
    monitor = None
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
            lambda: build_fit_data(
                config,
                data_root,
                split_manifest,
                teacher_cache,
                admission.final_membership,
                admission.final_source_validation_receipt,
            ),
            "sequence data construction",
            context,
        )
        coordinated_call(
            lambda: validate_fit_data_contract(config, data, context.world_size),
            "batch and split validation",
            context,
        )
        monitor_config = config.get("monitoring")
        train_eval_members = select_train_eval_members(monitor_config, data)
        seed_worker(int(config["seed"]), 0)
        base_model, normalization = coordinated_call(
            lambda: _prepare_model(config, data, context), "model construction", context
        )
        if initializer_path is not None:
            initializer_identity = coordinated_call(
                lambda: load_fitting_initializer(
                    initializer_path, base_model, config, data.provenance, normalization
                ),
                "fitting-only initializer validation and load",
                context,
            )
            data.provenance["initializer"] = initializer_identity
        model = wrap_distributed(base_model, context)
        seed_worker(int(config["seed"]), context.rank)
        fit_dataset = cast(SequenceDataset, data.fit)
        schedule_updates = total_updates(config, len(fit_dataset), context.world_size)
        if schedule_updates < 1:
            raise ValueError("configured fit has no optimizer updates")
        monitor_event_budget, profile_event_budget = prepare_monitor_event_budgets(
            monitor_config, schedule_updates, int(config["epochs"]), profile_updates
        )
        optimizer, scheduler = make_optimizer(base_model, config, schedule_updates)
        root = Path(__file__).resolve().parents[3]
        if monitor_config is not None:
            model_config = config["model"]
            input_feature_mode = str(model_config.get("input_feature_mode", "raw112"))
            record_monitoring_plan(
                data,
                monitor_config,
                train_eval_members,
                input_feature_mode,
                monitor_event_budget,
            )
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
        train_eval_dataset, monitor = initialize_monitor(
            data_root,
            output,
            monitor_config,
            train_eval_members,
            data,
            identity,
            resume_path is not None,
            context,
            str(config["model"].get("input_feature_mode", "raw112")),
        )
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
                    monitor=monitor,
                    monitor_enabled=monitor_config is not None,
                    monitoring_plan=data.provenance.get("monitoring_plan"),
                    monitor_event_budget=(
                        None if profile_event_budget is None else profile_event_budget.as_dict()
                    ),
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
                evaluation_dataset=cast(SequenceDataset, data.evaluation)
                if monitor is not None
                else None,
                train_eval_dataset=train_eval_dataset,
                monitor=monitor,
            )
        finally:
            rank_log.close()
    finally:
        close_monitor_progress(monitor)
        shutdown_distributed(context)
