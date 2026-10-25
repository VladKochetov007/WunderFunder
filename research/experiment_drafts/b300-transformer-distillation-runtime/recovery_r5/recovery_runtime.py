"""Prepare and explicitly execute the four R5 recovery fits through the R4 runner."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
R4 = PACKET.parent / "high_compute_r4"
sys.path.insert(0, str(R4))

_campaign_io = importlib.import_module("campaign_io")
_cells = importlib.import_module("cell_materialize")
_checks = importlib.import_module("runner_checks")
load_campaign = _campaign_io.load_campaign
read_object = _campaign_io.read_object
require = _campaign_io.require
sha256_file = _campaign_io.sha256_file
write_json_new = _campaign_io.write_json_new
_new_cell = _cells._new_cell
_validate_record = _cells._validate_record
execute_cell = importlib.import_module("dispatch").execute_cell
accepted_checks = _checks.accepted_checks
execute = _checks.execute
receiver_check_environment = _checks.receiver_check_environment
run_checks = _checks.run_checks

# Campaign manifests are heterogeneous JSON validated at the I/O boundary.


def source_files() -> dict[str, str]:
    """Bind executable recovery code and frozen configs, excluding runtime outputs."""
    return {
        path.relative_to(PACKET).as_posix(): sha256_file(path)
        for path in sorted(PACKET.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".json", ".sh")
    }


def prepare(root: Path) -> dict[str, Any]:
    """Materialize a new four-fit panel; never modify an existing campaign recipe."""
    campaign = load_campaign(root)
    require(not (root / "recovery_r5.json").exists(), "R5 is already prepared")
    from split_audit import audit_splits

    audit_hash = write_json_new(root / "split_audit.json", audit_splits(campaign))
    records = []
    for lr in ("1e-4", "3e-4"):
        for fold in (0, 2):
            cell_id = f"R5_lr_{lr}_f{fold}"
            config = read_object(PACKET / "configs" / f"{cell_id}.json")
            records.append(
                _new_cell(
                    root,
                    campaign,
                    cell_id,
                    f"lr_{lr}",
                    fold,
                    int(config["seed"]),
                    "teacher",
                    config,
                    inner_split=True,
                )
            )
    manifest = {
        "schema": "wnn.transformer-recovery-r5.v1",
        "campaign_sha256": campaign["campaign_sha256"],
        "runtime_source_sha256": campaign["runtime_source_sha256"],
        "recovery_source_files": source_files(),
        "cells": records,
        "split_audit_sha256": audit_hash,
        "new_teacher_fits": 4,
        "new_student_fits": 0,
        "checkpoint_selection": "shared-epoch-max-mean-inner-wp-earlier-tie",
        "selection_requires_completed_epochs": 30,
    }
    write_json_new(root / "recovery_r5.json", manifest)
    return manifest


def load_recovery(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject changed runtime, recipe, cell, or split before any receiver action."""
    campaign = load_campaign(root)
    manifest = read_object(root / "recovery_r5.json")
    require(manifest.get("schema") == "wnn.transformer-recovery-r5.v1", "unsupported R5 plan")
    require(manifest["campaign_sha256"] == campaign["campaign_sha256"], "campaign changed")
    require(manifest["recovery_source_files"] == source_files(), "R5 source/config changed")
    require(
        manifest["split_audit_sha256"] == sha256_file(root / "split_audit.json"),
        "split audit changed",
    )
    for record in manifest["cells"]:
        _validate_record(root, campaign, record)
    return campaign, manifest


def get_recovery_cell(root: Path, cell_id: str) -> dict[str, Any]:
    """Resolve one declared R5 cell without accepting arbitrary commands or paths."""
    _, manifest = load_recovery(root)
    records = [record for record in manifest["cells"] if record["id"] == cell_id]
    require(len(records) == 1, "unknown or duplicate R5 cell")
    return records[0]


def check_receiver(root: Path, check_id: str) -> None:
    """Run existing hardware acceptance plus recovery checks on the B300 receiver."""
    campaign, manifest = load_recovery(root)
    pointer = root / "recovery_acceptance.json"
    require(not pointer.exists(), "recovery acceptance already exists")
    if not (root / "acceptance.json").exists():
        run_checks(root, check_id)
    accepted_checks(root, campaign["runtime_source_sha256"])
    log = root / "recovery_checks.log"
    execute(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_sequence_epoch_index.py",
            "tests/test_sequence_epoch_prediction.py",
            "tests/test_recovery_epoch_sweep.py",
            "tests/test_recovery_epoch_sweep_resume.py",
            "tests/test_midas_training_export.py",
        ],
        log,
        receiver_check_environment(),
    )
    write_json_new(
        pointer,
        {
            "schema": "wnn.recovery-checks.v1",
            "status": "passed",
            "runtime_source_sha256": campaign["runtime_source_sha256"],
            "recovery_source_files": manifest["recovery_source_files"],
            "log_sha256": sha256_file(log),
        },
    )


def receiver_acceptance(root: Path, campaign: dict[str, Any]) -> Path:
    """Require actual receiver test evidence; preparation never manufactures it."""
    receipt = read_object(root / "recovery_acceptance.json")
    require(receipt.get("schema") == "wnn.recovery-checks.v1", "invalid recovery checks")
    require(receipt.get("status") == "passed", "recovery checks have not passed")
    require(receipt["runtime_source_sha256"] == campaign["runtime_source_sha256"], "checks stale")
    require(receipt["recovery_source_files"] == source_files(), "recovery checks source stale")
    require(
        receipt["log_sha256"] == sha256_file(root / "recovery_checks.log"), "checks log changed"
    )
    return accepted_checks(root, campaign["runtime_source_sha256"])


def storage_admission(root: Path, cell: dict[str, Any]) -> None:
    """Reserve headroom for retained FP32 weights plus atomic resumable state writes."""
    profile = read_object(root / cell["run_dir"] / "approved_profile.json")
    config = read_object(root / cell["config"])
    parameters = profile["identity"]["model_parameters"]["total"]
    require(type(parameters) is int and parameters > 0, "profile model size is missing")
    # Conservative allocation before the first fit; resume checks remaining snapshots.
    needed = int(parameters * 4 * (int(config["epochs"]) + 10) * 1.15)
    require(
        shutil.disk_usage(root).free >= needed,
        f"epoch retention needs at least {needed} free disk bytes (conservative bound)",
    )


def main() -> None:
    """Run one explicit receiver operation; no loop schedules subsequent fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "list",
            "check",
            "profile-one",
            "profile-eight",
            "profile-peak",
            "admit",
            "fit",
            "resume",
            "evaluate",
            "bundle",
        ),
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--cell")
    parser.add_argument("--check-id", default="r5-receiver")
    args = parser.parse_args()
    root = args.campaign_root.expanduser().resolve()
    if args.action == "prepare":
        result = prepare(root)
    elif args.action == "check":
        check_receiver(root, args.check_id)
        result = {"receiver_checks": "passed"}
    else:
        campaign, manifest = load_recovery(root)
        if args.action == "list":
            result = manifest
        else:
            if not args.cell:
                parser.error("--cell is required for a receiver action")
            cell = get_recovery_cell(root, args.cell)
            checks = receiver_acceptance(root, campaign)
            if args.action == "fit":
                storage_admission(root, cell)
            if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
                parser.error("set CUBLAS_EMULATE_SINGLE_PRECISION=1 before BF16/BF16x9 operations")
            execute_cell(args.action, root, campaign, cell, checks)
            result = {"cell": args.cell, "action": args.action, "finished": True}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
