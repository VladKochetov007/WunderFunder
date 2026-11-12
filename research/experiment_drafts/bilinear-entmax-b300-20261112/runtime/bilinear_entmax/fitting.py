"""Compose Bilinear Entmax variants with the unchanged canonical fit lifecycle."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, cast

import torch
from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_fit_admission import validate_fit_reference_inputs
from wnn.training.sequence_monitor_progress import close_monitor_progress

from bilinear_entmax.bindings import (
    SPLIT_SHA256,
    canonical,
    digest,
    load_cell,
    source_identity,
    write_new,
)
from bilinear_entmax.fitting_admission import admit_profile, validate_checks, validate_profile_gate
from bilinear_entmax.modeling import build_model
from bilinear_entmax.preprocessing import load_preprocessing
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
)


def _owner_receipt(
    output: Path, args: argparse.Namespace, cell: dict[str, Any], split_sha256: str
) -> bool:
    """Record process ownership and preserve exact split bytes for downstream evidence."""
    if digest(args.split_manifest) != split_sha256:
        raise ValueError("split manifest changed after data admission")
    split_copy = output / "split.json"
    if split_copy.exists():
        if not args.resume or digest(split_copy) != split_sha256:
            raise ValueError("existing split snapshot is incompatible with this attempt")
    else:
        with split_copy.open("xb") as stream:
            stream.write(args.split_manifest.read_bytes())
        if digest(split_copy) != split_sha256:
            raise ValueError("split snapshot digest differs from the admitted manifest")
    suffix = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"resume_owner_{suffix}.json" if args.resume else "owner.json"
    write_new(
        output / name,
        {
            "schema": "bilinear-entmax.owner.v1",
            "owner": args.owner,
            "cell_id": cell["id"],
            "timeout_seconds": args.timeout_seconds,
            "output": str(output),
            "mode": "fit" if not args.profile_updates else "profile",
            "optimization_changes": False,
        },
    )
    return True


def run(args: argparse.Namespace, runtime_root: Path) -> None:
    """Fit or profile one frozen cell through shared PR75 data, update, and checkpoint code."""
    cell, native_config = load_cell(args.cell)
    config: dict[str, Any] = load_config(native_config)
    validate_fit_reference_inputs(config, None, None, None)
    output = args.output.resolve()
    resume = args.resume.resolve() if args.resume else None
    if args.profile_receipt is not None and (
        args.profile_receipt.is_symlink() or not args.profile_receipt.is_file()
    ):
        raise ValueError("profile receipt must be a regular non-symlink file")
    if args.preprocessing.is_symlink() or not args.preprocessing.is_file():
        raise ValueError("preprocessing receipt must be a regular non-symlink file")
    profile_receipt = args.profile_receipt.resolve(strict=True) if args.profile_receipt else None
    if resume is not None and resume != output / "training_state.pt":
        raise ValueError("resume must name this output's canonical training_state.pt")
    if args.profile_updates and (resume is not None or profile_receipt is not None):
        raise ValueError("profiles cannot resume or consume a fit admission receipt")
    if args.profile_updates not in (0, 50):
        raise ValueError("Bilinear Entmax uses exactly 50 updates for each engineering profile")
    if not args.profile_updates and profile_receipt is None:
        raise ValueError("a full fit requires the approved profile receipt")
    artifact_path = args.preprocessing.resolve(strict=True)
    artifact_sha256 = digest(artifact_path)
    source = source_identity(runtime_root)
    validate_checks(
        args.checks_receipt,
        source,
        artifact_sha256,
        str(config["precision_mode"]),
    )
    if profile_receipt is not None and not args.profile_updates:
        validate_profile_gate(profile_receipt, source, cell)
    context = initialize_distributed()
    monitor: ComplementMonitor | None = None
    rank_log: RankJsonlLog | None = None
    try:
        coordinated_call(
            lambda: validate_runtime_requirements(
                config, context.world_size, torch.cuda.get_device_name(context.local_rank)
            ),
            "canonical runtime validation",
            context,
        )
        collective_output_setup(output, resume is not None, context)
        seed_worker(config["seed"], 0)
        data = coordinated_call(
            lambda: build_data(config, args.data_root, args.split_manifest, None, None),
            "canonical sequence data construction",
            context,
        )
        if data.split.sha256 != SPLIT_SHA256:
            raise ValueError("loaded sequence split differs from the frozen official split")
        coordinated_call(
            lambda: validate_expected_fold(config, data.split.fold), "fold validation", context
        )
        artifact = coordinated_call(
            lambda: load_preprocessing(artifact_path, data),
            "train-only preprocessing validation",
            context,
        )
        extension = {
            "schema": "bilinear-entmax.model.v1",
            "cell": cell,
            "preprocessing": artifact,
            "preprocessing_sha256": artifact_sha256,
        }
        config["bilinear_entmax"] = extension
        data.provenance["bilinear_entmax_preprocessing_sha256"] = artifact_sha256
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
            lambda: build_model(config, context.device),
            "Bilinear Entmax model composition",
            context,
        )
        model = wrap_distributed(base_model, context)
        seed_worker(config["seed"], context.rank)
        dataset = cast(SequenceDataset, data.fit)
        updates = total_updates(config, len(dataset), context.world_size)
        optimizer, scheduler = make_optimizer(base_model, config, updates)
        identity = run_identity(
            config, data.provenance, source, runtime_identity(context), context.world_size
        )
        identity["bilinear_entmax_cell"] = cell
        identity["bilinear_entmax_checks_sha256"] = digest(args.checks_receipt)
        identity["model_parameters"] = {
            "total": sum(parameter.numel() for parameter in base_model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in base_model.parameters()
                if parameter.requires_grad
            ),
        }
        identity["optimizer"] = optimizer_contract(optimizer, config, updates)
        assert_same_identity(identity, context)
        profile_admission = (
            coordinated_call(
                lambda: admit_profile(profile_receipt, identity, cell, config),
                "approved profile admission",
                context,
            )
            if profile_receipt is not None
            else None
        )
        profile_hash = profile_admission[0] if profile_admission is not None else None
        if profile_admission is not None:
            acceptance = profile_admission[1]
            identity["bilinear_entmax_profile_receipt_sha256"] = profile_hash
            identity["bilinear_entmax_full_shape_acceptance"] = acceptance
            identity["bilinear_entmax_full_shape_acceptance_sha256"] = canonical(acceptance)
        assert_same_identity(identity, context)
        recorder = coordinated_call(
            lambda: recorder_for_run(output, config, identity) if context.is_primary else False,
            "canonical run metadata",
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
            "owner and split receipts",
            context,
        )
        rank_log = coordinated_call(
            lambda: RankJsonlLog(output / f"rank{context.rank}.jsonl", resume=resume is not None),
            "canonical rank log",
            context,
        )
        if args.profile_updates:
            run_profile(
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
            )
        else:
            if profile_receipt is None or profile_hash is None:
                raise ValueError("missing approved canonical profile identity")
            run_full_fit(
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
                {"mean": artifact["mean"], "std": artifact["std"]},
                data.provenance,
                output,
                resume,
                profile_receipt,
                profile_hash,
                monitor=monitor,
            )
    finally:
        if rank_log is not None:
            rank_log.close()
        close_monitor_progress(monitor)
        shutdown_distributed(context)
