from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from multistage_native_fit_receipts import read_jsonl, regular_file, verify_completion_receipt

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.multistage_primary_report import (
    AUXILIARY_SPLIT_SHA256,
    PRIMARY_SPLIT_SHA256,
    _validate_endpoint,
    validate_auxiliary_report_membership,
)
from wnn.multistage_search import atomic_write_json, canonical_json_sha256, file_sha256, read_json
from wnn.splits import load_campaign_split
from wnn.training.sequence_config import load_config
from wnn.training.sequence_epoch_index import load_epoch_index
from wnn.training.sequence_provenance import source_identity, validate_profile_receipt


def verify_native_fit(
    args: argparse.Namespace, output: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify the fixed-final endpoint, exact profile/source/config, and train-only ancestry."""
    fit_dir = args.fit_dir.resolve(strict=True)
    checkpoint = regular_file(fit_dir / "model.pt", "native fixed-final checkpoint")
    config_path = regular_file(args.config, "raw config", 4_000_000)
    profile_path = regular_file(args.profile_receipt, "native profile receipt", 8_000_000)
    approved_path = regular_file(args.approved_profile, "approved profile receipt", 8_000_000)
    config = load_config(config_path)
    config_raw_sha = file_sha256(config_path)
    if args.expected_config_sha256 and config_raw_sha != args.expected_config_sha256:
        raise ValueError("raw config SHA differs from the explicit source pin")
    source_request = (
        read_json(regular_file(args.source_request, "source request", 8_000_000))
        if args.source_request
        else {}
    )
    if source_request and (
        source_request.get("config_sha256") != config_raw_sha
        or source_request.get("stage") != args.stage
        or source_request.get("trial_number") != args.trial_number
        or source_request.get("family") != args.family
    ):
        raise ValueError("source request does not bind the raw config and requested fit identity")
    if not args.expected_config_sha256 and not source_request:
        raise ValueError("raw config needs an explicit SHA pin or a source request binding")

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("native checkpoint payload is malformed")
    metadata_path = regular_file(fit_dir / "run_metadata.json", "run metadata", 16_000_000)
    metadata = read_json(metadata_path)
    provenance = payload.get("provenance")
    identity = provenance.get("identity") if isinstance(provenance, dict) else None
    if not isinstance(identity, dict):
        raise ValueError("fixed-final checkpoint omits its run identity")
    native_profile, approved_profile = read_json(profile_path), read_json(approved_path)
    if (
        native_profile.get("status") != "profile_complete"
        or native_profile.get("approved") is not False
        or approved_profile.get("approved") is not True
        or approved_profile.get("identity") != identity
        or native_profile.get("identity") != identity
        or metadata.get("config") != config
        or metadata.get("source")
        != {"sha256": identity.get("source_sha256"), "files": identity.get("source_files")}
        or identity.get("config_sha256") != canonical_json_sha256(config)
        or metadata.get("provenance", {}).get("data") != identity.get("data")
        or metadata.get("provenance", {}).get("data_sha256") != identity.get("data_sha256")
        or metadata.get("provenance", {}).get("runtime") != identity.get("runtime")
    ):
        raise ValueError("checkpoint, run metadata, config or exact profile identity disagrees")
    validate_profile_receipt(approved_path, identity)
    source_root = args.training_source_root.resolve(strict=True)
    training_source = source_identity(source_root, source_root / "scripts/train_sequence.py")
    if training_source.get("sha256") != identity.get("source_sha256") or training_source.get(
        "files"
    ) != identity.get("source_files"):
        raise ValueError("native fit source differs from the exact Source4/5B snapshot")
    profile_binding = provenance.get("profile_receipt") if isinstance(provenance, dict) else None
    if not isinstance(profile_binding, dict) or profile_binding.get("sha256") != file_sha256(
        approved_path
    ):
        raise ValueError("fixed-final checkpoint does not bind its approved profile receipt")

    primary = load_campaign_split(args.primary_split_manifest, args.data_root)
    auxiliary = load_campaign_split(args.auxiliary_split_manifest, args.data_root)
    if primary.sha256 != PRIMARY_SPLIT_SHA256 or auxiliary.sha256 != AUXILIARY_SPLIT_SHA256:
        raise ValueError("postprocessing split manifests differ from the frozen B300 protocol")
    primary_raw = json.loads(args.primary_split_manifest.read_bytes())
    primary_sources = primary_raw.get("sources")
    if not isinstance(primary_sources, dict):
        raise ValueError("primary split does not contain raw source fingerprints")
    validate_auxiliary_report_membership(
        payload["data"], config, primary, auxiliary, primary_sources
    )
    producer, model_sha256 = _validate_endpoint(
        checkpoint, payload, primary, auxiliary, primary_sources
    )
    if config["model"].get("architecture") != args.family:
        raise ValueError("requested family differs from the native model architecture")
    if source_request and source_request.get("source_sha256") not in (
        None,
        identity.get("source_sha256"),
    ):
        raise ValueError("source request training source SHA differs from the native fit")

    terminal_path = regular_file(fit_dir / "terminal_status.json", "native terminal status")
    index_path = regular_file(fit_dir / "epoch_index.json", "native epoch index", 16_000_000)
    terminal = read_json(terminal_path)
    index = load_epoch_index(index_path, identity)
    epochs = config["epochs"]
    terminal_file_sha = file_sha256(terminal_path)
    index_file_sha = file_sha256(index_path)
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or payload.get("config") != config
        or payload.get("completed_epochs") != epochs
        or payload.get("epoch") != epochs - 1
        or provenance.get("fixed_checkpoint_policy") != "configured-final-epoch"
        or provenance.get("completed_epochs") != epochs
        or type(provenance.get("global_updates")) is not int
        or provenance["global_updates"] < 1
        or terminal.get("status") != "completed"
        or terminal.get("completion_kind") != "max_epochs"
        or terminal.get("completed_epochs") != epochs
        or terminal.get("requested_epochs") != epochs
        or terminal.get("model_sha256") != model_sha256
        or terminal.get("identity") != identity
        or terminal.get("epoch_index_sha256") != index.get("index_sha256")
        or index.get("identity") != identity
        or index.get("completed_epochs") != epochs
        or index.get("global_step") != provenance["global_updates"]
        or terminal.get("global_updates") != provenance["global_updates"]
        or provenance.get("epoch_index_sha256") != index.get("index_sha256")
    ):
        raise ValueError("native terminal status, fixed-final model and epoch index disagree")
    for entry in index["entries"]:
        path = fit_dir / str(entry["checkpoint_path"])
        if (
            file_sha256(regular_file(path, "indexed epoch checkpoint"))
            != entry["checkpoint_sha256"]
        ):
            raise ValueError("native epoch index checkpoint bytes do not match their hashes")
    completion = verify_completion_receipt(
        fit_dir,
        child_receipt=args.child_receipt,
        terminal_attestation=args.legacy_terminal_attestation,
        terminal=terminal,
        index=index,
        model_sha256=model_sha256,
        index_file_sha256=index_file_sha,
        terminal_file_sha256=terminal_file_sha,
        stage=args.stage,
        trial_number=args.trial_number,
    )
    (output / "config.json").write_bytes(config_path.read_bytes())
    identity_sha = canonical_json_sha256(identity)
    request = {
        "schema": "wnn.multistage-trial-plan.v1",
        "stage": args.stage,
        "trial_number": args.trial_number,
        "family": args.family,
        "params": {"batch_size": config["batch_size"], "epochs": epochs},
        "config_path": str((output / "config.json").resolve()),
        "config_sha256": config_raw_sha,
        "config_identity_sha256": canonical_json_sha256(config),
        "profile_identity_sha256": identity_sha,
        "profile_receipt_path": str(profile_path),
        "profile_receipt_sha256": file_sha256(profile_path),
        "approved_profile_path": str(approved_path),
        "approved_profile_sha256": file_sha256(approved_path),
        "primary_split_sha256": primary.sha256,
        "auxiliary_split_manifest": str(args.auxiliary_split_manifest.resolve()),
        "auxiliary_split_sha256": auxiliary.sha256,
        "source_revision": args.training_source_revision
        or f"{source_root.name}:{training_source['sha256']}",
        "source_sha256": training_source["sha256"],
        "source_files_sha256": canonical_json_sha256(training_source["files"]),
        "objective_scope": "inner_train_proxy",
        "primary_valid_used": False,
    }
    if args.parent_config is not None:
        request["parent_config_sha256"] = file_sha256(
            regular_file(args.parent_config, "parent config")
        )
    atomic_write_json(output / "trial_request.json", request, replace=False)
    receipt = {
        "identity": identity,
        "producer": producer,
        "config": config,
        "config_raw_sha256": config_raw_sha,
        "profile_identity_sha256": identity_sha,
        "native_profile_sha256": file_sha256(profile_path),
        "approved_profile_sha256": file_sha256(approved_path),
        "run_metadata_sha256": file_sha256(metadata_path),
        "terminal_status_sha256": terminal_file_sha,
        "epoch_index_internal_sha256": index["index_sha256"],
        "epoch_index_file_sha256": index_file_sha,
        "model_sha256": model_sha256,
        "completion": completion,
        "primary_split_sha256": primary.sha256,
        "auxiliary_split_sha256": auxiliary.sha256,
        "primary_valid_fit_sequences": 0,
    }
    return receipt, request, {"identity": identity, "payload": payload, "index": index}


def verify_control_events(
    fit_dir: Path, identity: dict[str, Any], epochs: int
) -> list[dict[str, Any]]:
    """Require all native control epochs and a completed terminal monitor event."""
    evaluations = read_jsonl(
        fit_dir / "monitor_evaluations.jsonl", "evaluation journal", 256 * 1024 * 1024
    )
    events = read_jsonl(
        fit_dir / "monitor_events.jsonl", "monitor event journal", 1024 * 1024 * 1024
    )
    evaluations = [row for row in evaluations if row.get("kind") == "evaluation"]
    statuses = [row for row in events if row.get("kind") == "status"]
    if len(evaluations) != epochs or not statuses:
        raise ValueError("native fit lacks each requested control evaluation and terminal status")
    data = identity.get("data")
    if not isinstance(data, dict):
        raise ValueError("native control journal has no run data identity")
    members = data.get("evaluation_members")
    from wnn.training.sequence_monitor_identity import membership_sha256

    if not isinstance(members, list) or not members:
        raise ValueError("native run identity has no auxiliary control membership")
    member_sha = membership_sha256([tuple(member) for member in members])
    selected_rows = len(members) * (SEQUENCE_LENGTH - WARMUP)
    for number, event in enumerate(evaluations, start=1):
        context, metrics = event.get("context"), event.get("metrics")
        control = context.get("control") if isinstance(context, dict) else None
        if (
            event.get("epoch") != number
            or not isinstance(control, dict)
            or control.get("scope") != "inner_train_proxy"
            or control.get("role") != "control"
            or control.get("source") != "train"
            or control.get("row_mask") != "all_after_warmup"
            or control.get("split_sha256") != data.get("split_sha256")
            or control.get("primary_split_sha256") != PRIMARY_SPLIT_SHA256
            or control.get("member_count") != len(members)
            or control.get("member_sha256") != member_sha
            or control.get("selected_rows") != selected_rows
            or ("is_scored_used" in control and control.get("is_scored_used") is not False)
            or not isinstance(metrics, dict)
            or not all(
                isinstance(metrics.get(key), (int, float))
                and not isinstance(metrics.get(key), bool)
                and math.isfinite(float(metrics[key]))
                for key in ("control/wp", "control/wp/t0", "control/wp/t1")
            )
        ):
            raise ValueError("native control evaluation differs from the auxiliary proxy cohort")
    final = statuses[-1]
    if (
        not isinstance(final.get("context"), dict)
        or final["context"].get("status") != "completed"
        or final.get("epoch") != epochs
    ):
        raise ValueError("native monitor journal lacks its final completed status event")
    return evaluations
