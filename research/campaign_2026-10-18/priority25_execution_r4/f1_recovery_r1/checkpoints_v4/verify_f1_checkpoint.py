#!/usr/bin/env python3
"""CPU-only structural checks for one F1/seed1001 checkpoint boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import f1_checkpoint_common as common_module
import f1_checkpoint_evidence as evidence_module
import f1_checkpoint_lineage as lineage_module
import f1_checkpoint_tensors as tensor_module
from f1_checkpoint_common import (
    ARTIFACTS,
    FOLD1_SHA256,
    PLAN_SHA256,
    SCHEDULE_BATCHES,
    SCHEDULE_UPDATES,
    SOURCE_REVISION,
    arm_spec,
    expected_counts,
    mapping,
    read_json,
    require,
    sha256_file,
    validate_attempt,
    validate_history,
    validate_plan,
    validate_recovery_binding,
)
from f1_checkpoint_evidence import (
    _verify_archive_manifest,
    _verify_collection,
    _verify_contract,
    _verify_registry,
    _verify_source_map,
    validate_manifest_membership,
)
from f1_checkpoint_lineage import verify_lineage
from f1_checkpoint_tensors import verify_checkpoint_tensors


def _helper_manifest() -> dict[str, str]:
    """Hash the exact imported sibling helper bytes used by this verifier."""
    modules = {
        "f1_checkpoint_common.py": common_module,
        "f1_checkpoint_evidence.py": evidence_module,
        "f1_checkpoint_lineage.py": lineage_module,
        "f1_checkpoint_tensors.py": tensor_module,
    }
    expected_parent = Path(__file__).resolve().parent
    result = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        require(path.parent == expected_parent and path.name == name, f"import path {name}")
        result[name] = sha256_file(path)
    collector_path = expected_parent / "collect_f1_checkpoint.py"
    result[collector_path.name] = sha256_file(collector_path)
    return result


def verify(args: argparse.Namespace) -> dict[str, object]:
    """Run all structural checks on CPU and return a non-quality receipt."""
    require((args.arm, args.phase) == ("wide", "terminal"), "recovery verifier scope")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    helper_sha256 = sha256_file(Path(__file__))
    plan = validate_plan(args.plan)
    spec = arm_spec(plan, args.arm, args.phase)
    attempt = validate_attempt(args.snapshot_dir / "attempt.json", plan, spec)
    artifacts, collection = _verify_collection(
        args.snapshot_dir, args.collection_receipt, spec, args.plan
    )
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    from wnn.training_state import load_training_state

    loader_path = Path(load_training_state.__code__.co_filename).resolve()
    expected_loader = (args.source_root.resolve() / "src/wnn/training_state.py").resolve()
    require(loader_path == expected_loader, "canonical training-state loader import path")
    helper_dependencies = _helper_manifest()
    helper_dependencies["wnn/training_state.py"] = sha256_file(loader_path)

    state = load_training_state(args.snapshot_dir / "training_state.pt")
    contract = mapping(state["contract"], "training contract")
    _verify_contract(contract, plan, spec)
    source_count = _verify_source_map(args.source_root.resolve(), contract)
    registry_sha = _verify_registry(contract, plan, args.data_registry)
    archive_sha = _verify_archive_manifest(args.archive_manifest)
    provenance = read_json(args.snapshot_dir / "provenance.json")
    fold = validate_manifest_membership(args.fold_manifest, contract, provenance)
    optional_epoch0_verified = False
    if args.epoch0_receipt is not None:
        require(args.phase == "stage1", "epoch0 receipt only applies to stage1")
        from f1_checkpoint_common import validate_optional_epoch0_binding

        optional_epoch0_verified = validate_optional_epoch0_binding(
            read_json(args.epoch0_receipt),
            args.arm,
            str(spec["stage1_run"]),
            artifacts["model_ep0.pt"]["sha256"],
        )
    _need_expected = {"epoch0": 1, "stage1": 2, "terminal": 6}[args.phase]
    require(state["next_epoch"] == _need_expected, "checkpoint next_epoch")
    require(
        (state["scaler"], state["ema"], state["swa"]) == ({}, None, None), "FP32 auxiliary state"
    )
    expected_total = expected_counts(_need_expected)
    progress = mapping(state["progress"], "training progress")
    require(progress["cumulative_updates"] == expected_total, "progress counters")
    require(
        progress["step_count"] == expected_total["optimizer_completed_updates"],
        "progress step count",
    )
    require(provenance.get("cumulative_updates") == expected_total, "provenance counters")
    require(
        provenance.get("update_diagnostics") == progress.get("update_diagnostics"),
        "provenance diagnostics",
    )
    if args.phase != "epoch0":
        history_file = json.loads((args.snapshot_dir / "history.json").read_text())
        history = validate_history(progress["history"], _need_expected)
        require(history_file == history, "history file/state equality")
    else:
        history = validate_history(progress["history"], 1)
    scheduler = mapping(state["scheduler"], "scheduler")
    require(
        (scheduler["last_epoch"], scheduler["_step_count"])
        == (
            expected_total["optimizer_completed_updates"],
            expected_total["optimizer_completed_updates"] + 1,
        ),
        "scheduler endpoint",
    )
    require(
        sha256_file(args.snapshot_dir / "training_state.pt")
        == artifacts["training_state.pt"]["sha256"],
        "state artifact hash",
    )

    parent_sha = verify_lineage(
        args,
        plan,
        spec,
        contract,
        history,
        provenance,
        collection,
        load_training_state,
    )

    tensor_result = verify_checkpoint_tensors(
        args, state, contract, spec, artifacts, provenance, torch
    )
    return {
        "schema": "priority25.f1_checkpoint_verification.recovery_v4",
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "phase": args.phase,
        "run": spec["run"],
        "next_epoch": state["next_epoch"],
        "attempt": attempt,
        "plan_sha256": PLAN_SHA256,
        "recovery_binding": validate_recovery_binding(),
        "collection_receipt_sha256": sha256_file(args.collection_receipt),
        "collection_helper_sha256": collection["collector_sha256"],
        "verification_helper_sha256": helper_sha256,
        "verification_dependencies": helper_dependencies,
        "artifacts": artifacts,
        "source_revision": SOURCE_REVISION,
        "source_files_verified": source_count,
        "source_archive_manifest_sha256": archive_sha,
        "portable_registry_sha256": registry_sha,
        "fold1_manifest_sha256": FOLD1_SHA256,
        "fold1_membership": fold,
        "fit_membership_identity": contract["fit_membership"],
        "normalization_identity": contract["normalization"],
        "stage1_parent_training_state_sha256": parent_sha,
        "completed_update_counts": expected_total,
        "planned_schedule_updates": SCHEDULE_UPDATES,
        "planned_schedule_batches": SCHEDULE_BATCHES,
        "scheduler_last_epoch": expected_total["optimizer_completed_updates"],
        "scheduler_step_count": expected_total["optimizer_completed_updates"] + 1,
        "model_tensors_verified": tensor_result["model_tensors_verified"],
        "trainable_parameters": tensor_result["trainable_parameters"],
        "adam_states_verified": tensor_result["adam_states_verified"],
        "adam_floating_tensors_verified": tensor_result["adam_floating_tensors_verified"],
        "history_epochs_verified": len(history),
        "peer_pair_contract_checked": args.peer_stage1_snapshot_dir is not None,
        "optional_epoch0_receipt_checked": optional_epoch0_verified,
        "cuda_initialized": False,
        "limitations": [
            "Structural checkpoint/source/data identity checks only; not measured GPU resume parity.",
            "Raw parquet bytes are matched to the sealed registry; no second full-file rehash is performed at checkpoint verification.",
            "No prediction score, serving latency, convergence, CP30 or quality claim follows.",
        ],
    }


def main() -> None:
    """Verify one already-collected checkpoint and write a create-only receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--phase", choices=tuple(ARTIFACTS), required=True)
    for name in (
        "snapshot-dir",
        "source-root",
        "collection-receipt",
        "plan",
        "fold-manifest",
        "data-registry",
        "archive-manifest",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage1-snapshot-dir", type=Path)
    parser.add_argument("--stage1-verification", type=Path)
    parser.add_argument("--peer-stage1-snapshot-dir", type=Path)
    parser.add_argument("--peer-stage1-verification", type=Path)
    parser.add_argument("--epoch0-receipt", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "verification receipt already exists")
    result = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
