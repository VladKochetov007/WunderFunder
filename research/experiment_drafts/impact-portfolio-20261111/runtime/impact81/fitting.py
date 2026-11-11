from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, cast

import torch
from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_fit_admission import validate_fit_reference_inputs
from wnn.training.sequence_monitor_progress import close_monitor_progress

from impact81.bindings import digest, load_cell, read_object, source_identity, write_new
from impact81.modeling import build_model
from impact81.preprocessing import load_preprocessing
from impact81.variants import EXTENSION_CARDS
from wnn.training.sequence_config import (
    load_config,
    validate_expected_fold,
    validate_runtime_requirements,
)
from wnn.training.sequence_data import build_data
from wnn.training.sequence_distributed import (
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
from wnn.training.sequence_optimizer_telemetry import optimizer_contract
from wnn.training.sequence_profile import run_profile
from wnn.training.sequence_provenance import (
    assert_same_identity,
    recorder_for_run,
    run_identity,
    runtime_identity,
    validate_profile_receipt,
)

# The canonical trainer accepts heterogeneous validated recipe/provenance mappings.


def run(args: argparse.Namespace, runtime_root: Path) -> None:
    """Compose an isolated experiment runner using pinned canonical training components."""
    cell, native_path = load_cell(args.cell)
    config: dict[str, Any] = load_config(native_path)
    validate_fit_reference_inputs(config, None, None, None)
    output = args.output.resolve()
    resume = args.resume.resolve() if args.resume else None
    receipt = args.profile_receipt.resolve() if args.profile_receipt else None
    if resume is not None and resume != output / "training_state.pt":
        raise ValueError("resume must name this output's training_state.pt")
    if args.profile_updates and (resume or receipt):
        raise ValueError("profiles cannot resume or consume a fit admission receipt")
    if not args.profile_updates and receipt is None:
        raise ValueError("full fitting requires the canonical approved profile receipt")
    checks = read_object(args.checks_receipt)
    if (
        checks.get("status") != "passed"
        or checks.get("source_identity") != source_identity(runtime_root)
        or checks.get("preprocessing_sha256") != digest(args.preprocessing)
        or checks.get("precision_mode") != config["precision_mode"]
    ):
        raise ValueError(
            "server acceptance must bind these exact sources, preprocessing and precision"
        )
    if cell["card"] in EXTENSION_CARDS:
        from impact81.auxiliary_loop import get_engine
        from impact81.lifecycle import run_full_fit as fit_runner
        from impact81.profile import run_profile as profile_runner

        if cell["card"] not in checks.get("extension_cards", []):
            raise ValueError("this card requires its family's current receiver receipt")
        engine_options = {"engine": get_engine()}
    else:
        fit_runner, profile_runner = run_full_fit, run_profile
        engine_options = {}
    context = initialize_distributed()
    monitor = None
    rank_log = None
    try:
        coordinated_call(
            lambda: validate_runtime_requirements(
                config, context.world_size, torch.cuda.get_device_name(context.local_rank)
            ),
            "runtime validation",
            context,
        )
        collective_output_setup(output, resume is not None, context)
        seed_worker(config["seed"], 0)
        data = coordinated_call(
            lambda: build_data(config, args.data_root, args.split_manifest, None, None),
            "canonical data construction",
            context,
        )
        coordinated_call(
            lambda: validate_expected_fold(config, data.split.fold), "fold validation", context
        )
        artifact = coordinated_call(
            lambda: load_preprocessing(args.preprocessing, data),
            "train preprocessing identity",
            context,
        )
        if cell["card"] == "10":
            from impact81.preparation_later import validate_feasibility

            coordinated_call(
                lambda: _admit_clip_events(artifact, validate_feasibility),
                "train-only tail feasibility",
                context,
            )
        extension = {
            "schema": "impact81.model.v1",
            "cell": cell,
            "preprocessing": artifact,
            "preprocessing_sha256": digest(args.preprocessing),
        }
        config["impact81"] = extension
        data.provenance["impact81_preprocessing_sha256"] = extension["preprocessing_sha256"]
        monitor = coordinated_call(
            lambda: ComplementMonitor(
                data,
                None,
                config,
                output,
                resume=resume is not None,
                primary=context.is_primary,
                report_epochs=not args.profile_updates,
            ),
            "canonical report-only monitor",
            context,
        )
        seed_worker(config["seed"], 0)
        base_model = coordinated_call(
            lambda: build_model(config, context.device), "experiment model composition", context
        )
        normalization = {"mean": artifact["mean"], "std": artifact["std"]}
        model = wrap_distributed(base_model, context)
        seed_worker(config["seed"], context.rank)
        dataset = cast(SequenceDataset, data.fit)
        updates = total_updates(config, len(dataset), context.world_size)
        optimizer, scheduler = make_optimizer(base_model, config, updates)
        source = source_identity(runtime_root)
        identity = run_identity(
            config, data.provenance, source, runtime_identity(context), context.world_size
        )
        identity["model_parameters"] = {
            "total": sum(parameter.numel() for parameter in base_model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in base_model.parameters()
                if parameter.requires_grad
            ),
        }
        if cell["card"] in ("04", "05", "06"):
            from impact81.depth_metadata import parameter_layout

            identity["impact81_depth_layout"] = parameter_layout(base_model)
        identity["optimizer"] = optimizer_contract(optimizer, config, updates)
        identity["impact81_checks_sha256"] = digest(args.checks_receipt)
        assert_same_identity(identity, context)
        profile_hash = (
            coordinated_call(
                lambda: _profile_admission(receipt, identity, config), "profile admission", context
            )
            if receipt is not None
            else None
        )
        recorder = coordinated_call(
            lambda: recorder_for_run(output, config, identity) if context.is_primary else False,
            "immutable canonical metadata",
            context,
        )
        if not context.is_primary:
            recorder = None
        coordinated_call(
            lambda: (
                _owner_receipt(output, args, cell, data.split.sha256)
                if context.is_primary
                else True
            ),
            "owner/attempt receipt",
            context,
        )
        rank_log = coordinated_call(
            lambda: RankJsonlLog(output / f"rank{context.rank}.jsonl", resume=resume is not None),
            "rank log",
            context,
        )
        if args.profile_updates:
            profile_runner(
                dataset,
                model,
                optimizer,
                scheduler,
                config,
                context,
                rank_log,
                recorder,
                identity,
                output,
                args.profile_updates,
                updates,
                progress=monitor.progress,
                **engine_options,
            )
        else:
            if receipt is None or profile_hash is None:
                raise ValueError("missing fit profile identity")
            fit_runner(
                dataset,
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
                resume,
                receipt,
                profile_hash,
                monitor=monitor,
                **engine_options,
            )
    finally:
        if rank_log is not None:
            rank_log.close()
        close_monitor_progress(monitor)
        shutdown_distributed(context)


def _profile_admission(path: Path, identity: dict[str, object], config: dict[str, Any]) -> str:
    """Validate signed auxiliary semantics before the unchanged canonical resource gate."""
    from impact81.profile_objective_admission import validate_profile_objective

    validate_profile_objective(read_object(path), config)
    return validate_profile_receipt(path, identity)


def _admit_clip_events(artifact: dict[str, Any], validator: Any) -> bool:
    """Make the card10 fixed-threshold feasibility decision collective and explicit."""
    validator(artifact["clip_event_feasibility"], require_feasible=True)
    return True


def _owner_receipt(
    output: Path, args: argparse.Namespace, cell: dict[str, Any], split_sha256: str
) -> bool:
    """Keep launch ownership separate from immutable model/profile identity."""
    split_copy = output / "split.json"
    if digest(args.split_manifest) != split_sha256:
        raise ValueError("split manifest changed after data admission")
    if split_copy.exists():
        if not args.resume or digest(split_copy) != split_sha256:
            raise ValueError("existing split snapshot is incompatible with this attempt")
    else:
        with split_copy.open("xb") as stream:
            stream.write(args.split_manifest.read_bytes())
        if digest(split_copy) != split_sha256:
            raise ValueError("split snapshot identity mismatch")
    suffix = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = output / (f"resume_owner_{suffix}.json" if args.resume else "owner.json")
    write_new(
        path,
        {
            "schema": "impact81.owner.v1",
            "owner": args.owner,
            "cell_id": cell["id"],
            "timeout_seconds": args.timeout_seconds,
            "output": str(output),
            "mode": "profile" if args.profile_updates else "fit",
            "optimization_changes": False,
        },
    )
    return True
