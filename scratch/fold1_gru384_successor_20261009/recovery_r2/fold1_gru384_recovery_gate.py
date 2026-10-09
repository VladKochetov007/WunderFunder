#!/usr/bin/env python3
"""Prepare and gate one exact fold1 seed3 epoch-boundary resume.

This tool never edits the original run. Preparation is read-only apart from its
exclusive request receipt. Dispatch requires a separate root admission bound to
that receipt, checkpoint SHA, command hash, and finite resource limits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Json = dict[str, Any]
BASE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_exclusive_json(path: Path, value: object) -> None:
    """Publish one receipt without replacing a prior attempt's evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to replace existing recovery evidence: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def packet_for(root: Path, packet_path: Path, expected_sha: str) -> Json:
    if sha256_file(packet_path) != expected_sha:
        raise ValueError("static recovery packet hash changed")
    packet = read_json(packet_path)
    if packet.get("schema") != "wnn.fold1_gru384_recovery_gate_packet.r2":
        raise ValueError("unexpected static recovery packet schema")
    if Path(packet["root"]).resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("recovery packet names a different repository root")
    return packet


def load_original_inputs(root: Path, packet: Json) -> tuple[Json, Json, Json, Json]:
    queue_path = root / packet["queue_path"]
    allocation_path = root / packet["original_allocation_path"]
    queue_bytes = queue_path.read_bytes()
    allocation_bytes = allocation_path.read_bytes()
    if hashlib.sha256(queue_bytes).hexdigest() != packet["queue_sha256"]:
        raise ValueError("original fit queue hash changed")
    if hashlib.sha256(allocation_bytes).hexdigest() != packet["original_allocation_sha256"]:
        raise ValueError("original fit allocation hash changed")
    queue = json.loads(queue_bytes)
    allocation = json.loads(allocation_bytes)
    if queue.get("status") != "finite_two_fit_queue_prepared":
        raise ValueError("original queue status differs")
    if queue.get("root_authority_sha256") != packet["original_allocation_sha256"]:
        raise ValueError("original queue does not bind its allocation")
    if queue.get("source_pins_sha256") != packet["source_pins_sha256"]:
        raise ValueError("source pins differ from the static recovery packet")
    if allocation.get("status") != "conditional_exact_successor_authorized":
        raise ValueError("original allocation status differs")
    steps = queue.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise ValueError("original queue is not the exact seed3/seed4 pair")
    if steps[0].get("attempt_id") != packet["original_attempt_id"]:
        raise ValueError("original queue attempt differs")
    if steps[0].get("run_name") != packet["original_run_name"]:
        raise ValueError("original queue run name differs")
    if steps[0].get("command_sha256_canonical_json") != packet["original_command_sha256"]:
        raise ValueError("original seed3 command hash differs")
    if steps[0].get("command_sha256_canonical_json") != canonical_sha256(steps[0]["command"]):
        raise ValueError("original seed3 command is not bound to its receipt")
    for relative, expected in packet["source_pins_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"source pin changed: {relative}")
    return queue, allocation, steps[0], steps[1]


def terminal_attempt(root: Path, packet: Json, step: Json) -> tuple[Json, Json]:
    runtime = root / packet["runtime_relative"]
    attempt_path = runtime / "attempts" / f"{step['attempt_id']}.json"
    record = read_json(attempt_path)
    if record.get("id") != step["attempt_id"] or record.get("command") != step["command"]:
        raise ValueError("original attempt record identity/command differs")
    if record.get("contract_sha256") != packet["original_allocation_sha256"]:
        raise ValueError("original attempt record is not bound to the admitted allocation")
    if record.get("status") in {"reserved", "waiting", "running", "preflight", "launch_uncertain"}:
        raise RuntimeError("original fit is still active; leave it and the R1 waiter untouched")
    unit = record.get("unit")
    if unit != f"wnn-bandit-{step['attempt_id']}.service":
        raise ValueError("original attempt unit name differs")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    unit_state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if unit_state.get("ActiveState") not in {"inactive", "failed"}:
        raise RuntimeError(f"original fit unit is not terminal: {unit_state}")
    return record, unit_state


def verify_no_oom(record: Json, packet: Json) -> None:
    peak_ram = record.get("peak_ram_bytes")
    peak_vram = record.get("peak_observed_vram_bytes")
    if type(peak_ram) is not int or peak_ram <= 0:
        raise ValueError("terminal fit has no positive peak_ram_bytes measurement")
    if type(peak_vram) is not int or peak_vram <= 0:
        raise ValueError("terminal fit has no positive peak_observed_vram_bytes measurement")
    if peak_ram > int(packet["resource_request"]["memory_max_bytes"]):
        raise ValueError("original fit exceeded its memory allocation")
    if peak_vram >= int(packet["resource_request"]["observed_total_vram_max_bytes"]):
        raise ValueError("original fit reached or exceeded its observed VRAM allocation")
    event_maps = [
        record.get("memory_event_deltas"),
        record.get("final_memory_event_deltas"),
        record.get("parent_memory_event_deltas"),
        record.get("final_parent_memory_event_deltas"),
        record.get("parent_hierarchical_memory_event_deltas"),
        record.get("final_parent_hierarchical_memory_event_deltas"),
    ]
    if any(not isinstance(events, dict) for events in event_maps):
        raise ValueError("terminal fit is missing one or more cgroup OOM delta receipts")
    observer_status = record.get("observer_status")
    required_observers = {
        "attempt_memory_events": "ok",
        "attempt_memory_events_final": "ok",
        "parent_memory_events": "ok",
        "parent_memory_events_final": "ok",
        "parent_hierarchical_memory_events": "ok",
        "parent_hierarchical_memory_events_final": "ok",
        "attempt_memory_peak_final": "ok",
        "nvidia_smi_final": "ok",
    }
    if not isinstance(observer_status, dict) or any(
        observer_status.get(name) != status for name, status in required_observers.items()
    ):
        raise ValueError("terminal fit resource observer receipts are incomplete or unknown")
    for events in event_maps:
        for key in ("oom", "oom_kill", "oom_group_kill"):
            if key not in events:
                raise ValueError(f"terminal fit OOM receipt is missing {key}")
            if type(events[key]) is not int or events[key] != 0:
                raise ValueError(f"original fit has nonzero {key} event delta")


def load_old_gate(root: Path, packet: Json) -> Any:
    source = root / packet["original_postfit_gate_path"]
    if sha256_file(source) != packet["original_postfit_gate_sha256"]:
        raise ValueError("frozen R1 output verifier source hash changed")
    if (
        sha256_file(root / packet["original_postfit_gate_packet_path"])
        != packet["original_postfit_gate_packet_sha256"]
    ):
        raise ValueError("frozen R1 output verifier packet hash changed")
    spec = importlib.util.spec_from_file_location("fold1_gru384_postfit_gate_r1", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the already frozen R1 membership/output verifier")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "src"))
    spec.loader.exec_module(module)
    module.verify_policy(root, read_json(root / packet["original_postfit_gate_packet_path"]))
    return module


def verify_finite_tensors(value: object, torch_module: Any, path: str = "state") -> None:
    """Reject a corrupted continuation tensor without duplicating state in RAM."""
    if isinstance(value, torch_module.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch_module.isfinite(value).all()
        ):
            raise ValueError(f"non-finite continuation tensor at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            verify_finite_tensors(child, torch_module, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            verify_finite_tensors(child, torch_module, f"{path}[{index}]")


def file_stat_identity(path: Path) -> Json:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected a regular, non-symlink dataset file: {path}")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def verify_dataset_source_identity(
    root: Path, packet: Json, contract_datasets: Json, split_manifest: Json
) -> Json:
    """Check immutable-file identity and cached Parquet footer without reading full datasets.

    The trainer's resume-contract check still hashes the complete train/valid files
    before restoring RNG or entering the optimizer loop. This gate avoids duplicating
    that multi-gigabyte read during prepare and dispatch.
    """
    if os.environ.get("WNN_PORTABLE_DATA_MANIFEST"):
        raise ValueError(
            "portable source manifests trigger full-file verification in the split loader; "
            "recovery gate requires the native immutable dataset path"
        )
    helper_path = root / packet["source_identity_helper_path"]
    if sha256_file(helper_path) != packet["source_identity_helper_sha256"]:
        raise ValueError("cached dataset identity helper source hash changed")
    spec = importlib.util.spec_from_file_location("fold1_dataset_identity_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned cached dataset identity helper")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "src"))
    spec.loader.exec_module(module)

    expected_splits = {"train", "valid"}
    if set(contract_datasets) != expected_splits:
        raise ValueError("saved training contract does not bind exactly train and valid datasets")
    source_manifest = split_manifest.get("sources", {})
    identities: Json = {}
    for split_name in sorted(expected_splits):
        saved = contract_datasets[split_name]
        if not isinstance(saved, dict):
            raise ValueError(f"saved training dataset identity is malformed: {split_name}")
        saved_bytes = saved.get("bytes")
        saved_sha = saved.get("sha256")
        if (
            not isinstance(saved_bytes, int)
            or not isinstance(saved_sha, str)
            or len(saved_sha) != 64
            or any(character not in "0123456789abcdef" for character in saved_sha)
        ):
            raise ValueError(f"saved training dataset identity is incomplete: {split_name}")
        path = root / "datasets" / f"{split_name}.parquet"
        before = file_stat_identity(path)
        parquet_identity = module.source_identity(path, is_parquet=True)
        after = file_stat_identity(path)
        if before != after:
            raise ValueError(f"dataset changed while its bounded identity was read: {split_name}")
        expected = source_manifest.get(f"datasets/{split_name}.parquet")
        if parquet_identity != expected:
            raise ValueError(
                f"dataset identity differs from the frozen fold manifest: {split_name}"
            )
        if before["bytes"] != saved_bytes:
            raise ValueError(f"dataset size differs from the saved training contract: {split_name}")
        identities[split_name] = {
            "file_stat": before,
            "fold_manifest_identity": parquet_identity,
            "saved_training_contract": {"bytes": saved_bytes, "sha256": saved_sha},
        }
    return identities


def verify_input_state(
    root: Path, packet: Json, source_step: Json
) -> tuple[Json, str, Json, Json, Any, Any]:
    import torch

    output = root / source_step["expected_output_path"]
    provenance_path = output / "provenance.json"
    state_path = output / "training_state.pt"
    for path in (output, provenance_path, state_path):
        if path.is_symlink() or not path.exists():
            raise ValueError(f"original output is missing or symlinked: {path}")
    provenance = read_json(provenance_path)
    gate = load_old_gate(root, packet)
    if os.environ.get("WNN_PORTABLE_DATA_MANIFEST"):
        raise ValueError(
            "portable source manifests trigger full-file verification in the split loader; "
            "recovery gate requires the native immutable dataset path"
        )
    from wnn.splits import load_campaign_split

    split = load_campaign_split(root / packet["fold_manifest_path"], root)
    split_manifest = read_json(root / packet["fold_manifest_path"])
    gate.verify_provenance_membership(
        source_step, provenance, split, packet["source_pins_sha256"], False
    )
    state_sha = sha256_file(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or state.get("format") != "wnn.epoch-training-state"
        or state.get("version") != 1
        or state.get("boundary") != "epoch"
    ):
        raise ValueError("original file is not a full epoch-boundary training state")
    required_state = {
        "contract",
        "next_epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "swa",
        "loader",
        "auxiliary",
        "progress",
        "rng",
    }
    if required_state - state.keys():
        raise ValueError(f"training state is incomplete: {sorted(required_state - state.keys())}")
    verify_finite_tensors(state, torch)
    contract = state.get("contract")
    next_epoch = state.get("next_epoch")
    if not isinstance(contract, dict) or contract.get("epochs") != 4:
        raise ValueError("training-state contract does not preserve the four-epoch horizon")
    if not isinstance(next_epoch, int) or not 1 <= next_epoch <= 4:
        raise ValueError("training state is not a valid completed-epoch boundary")
    if contract.get("fit_membership") != provenance.get("fit_membership_identity"):
        raise ValueError("training-state fit membership differs from the producer provenance")
    provenance_args = provenance.get("args")
    if not isinstance(provenance_args, dict):
        raise ValueError("producer provenance has no effective trainer arguments")
    excluded_options = {
        "name",
        "resume_state",
        "save_training_state",
        "stop_after_epoch",
        "split_manifest",
        "fit_groups_manifest",
    }
    expected_contract_options = {
        key: value for key, value in provenance_args.items() if key not in excluded_options
    }
    if contract.get("options") != expected_contract_options:
        raise ValueError("training-state complete option contract differs from producer provenance")
    if (
        contract.get("sources", {}).get("trainer")
        != packet["source_pins_sha256"]["scripts/train_rnn.py"]
    ):
        raise ValueError("training-state trainer source differs from the frozen command")
    source_files = {
        str(path.relative_to(root / "src/wnn")): sha256_file(path)
        for path in sorted((root / "src/wnn").rglob("*.py"))
    }
    source_files["trainer"] = sha256_file(root / "scripts/train_rnn.py")
    if contract.get("sources") != source_files:
        raise ValueError("training-state full source contract differs from the live Python sources")
    datasets = contract.get("datasets", {})
    if not datasets:
        raise ValueError("training-state contract has no dataset identities")
    data_identity = verify_dataset_source_identity(root, packet, datasets, split_manifest)
    options = contract.get("options", {})
    expected_options = {
        "device": "cuda",
        "arch": "gru",
        "cell": "gru",
        "hidden": 384,
        "layers": 2,
        "dropout": 0.0,
        "seed": 3,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "train_seqs": 10607,
        "valid_fold": 1,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "batch": 16,
        "chunk": 2000,
        "lr": 0.001,
        "wd": 0.01,
        "optim": "adamw",
        "sched": "onecycle",
        "precision": "fp16",
    }
    if any(options.get(key) != value for key, value in expected_options.items()):
        raise ValueError("training-state recipe options differ from the admitted seed3 recipe")
    for name, tensor in state.get("model", {}).items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"training-state model entry is not a tensor: {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"training-state model tensor is non-finite: {name}")
    progress = state.get("progress", {})
    gate.finite_tree(progress)
    if len(progress.get("history", [])) != next_epoch:
        raise ValueError("training-state history does not end at its saved epoch boundary")
    diagnostics = progress.get("update_diagnostics", [])
    if len(diagnostics) != next_epoch:
        raise ValueError("training-state update diagnostics do not cover completed epochs")
    for index, entry in enumerate(diagnostics):
        epoch = entry.get("epoch", {})
        if entry.get("epoch_index") != index:
            raise ValueError("training-state update diagnostics are out of epoch order")
        attempts = int(epoch.get("optimizer_attempts", -1))
        completed = int(epoch.get("optimizer_completed_updates", -1))
        skipped = int(epoch.get("amp_skipped_updates", -1))
        if attempts <= 0 or completed <= 0 or skipped < 0 or attempts != completed + skipped:
            raise ValueError(f"training-state update counters do not reconcile at epoch {index}")
    contract_sha = canonical_sha256(contract)
    return (
        state,
        state_sha,
        {
            "path": str(state_path.resolve()),
            "sha256": state_sha,
            "next_epoch": next_epoch,
            "contract_sha256": contract_sha,
        },
        data_identity,
        split,
        gate,
    )


def command_for(source_command: list[str], packet: Json, timeout_seconds: int) -> list[str]:
    if not 1 <= timeout_seconds <= int(packet["maximum_recovery_timeout_seconds"]):
        raise ValueError("recovery timeout exceeds the explicit recovery ceiling")
    command = list(source_command)
    if command[:4] != ["timeout", "--signal=TERM", "--kill-after=20s", "7200"]:
        raise ValueError("original command timeout prefix differs")
    command[3] = str(timeout_seconds)
    name_index = command.index("--name") + 1
    if command[name_index] != packet["original_run_name"]:
        raise ValueError("original command run name differs")
    command[name_index] = packet["resume_run_name"]
    command.extend(
        [
            "--resume-state",
            str((Path(packet["root"]) / packet["original_training_state_path"]).resolve()),
        ]
    )
    if "--save-training-state" not in command:
        raise ValueError("original command does not preserve resume state")
    return command


def prepare(args: argparse.Namespace) -> Json:
    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    packet = packet_for(root, packet_path, args.packet_sha256)
    _queue, _allocation, source_step, seed4_step = load_original_inputs(root, packet)
    record, unit_state = terminal_attempt(root, packet, source_step)
    status = record.get("status")
    exit_status = record.get("exit_status")
    if status == "finished" and exit_status == 0:
        return {"status": "recovery_not_needed", "reason": "original seed3 finished successfully"}
    if status != "failed" or exit_status != 124:
        raise ValueError(
            f"original terminal state is not the admitted timeout case: {status}/{exit_status}"
        )
    if not 7100 <= float(record.get("elapsed_seconds", 0)) <= 7300:
        raise ValueError("exit 124 did not occur at the original 7200-second timeout boundary")
    verify_no_oom(record, packet)
    if (root / packet["runtime_relative"] / "STOP_ADMISSION.json").exists():
        raise RuntimeError("campaign stop marker is present; do not admit a new fit")
    if shutil.disk_usage(root).free < int(packet["minimum_free_bytes"]):
        raise RuntimeError("free disk is below the recovery admission floor")
    state, state_sha, state_identity, data_identity, split, gate = verify_input_state(
        root, packet, source_step
    )
    original_attempt_path = (
        root / packet["runtime_relative"] / "attempts" / f"{packet['original_attempt_id']}.json"
    )
    # If the timed command completed all artifacts before its wrapper deadline,
    # retain them and let the original seed4 gate continue; do not make a no-op run.
    try:
        complete_output = gate.verify_output(root, source_step, split, packet["source_pins_sha256"])
    except (OSError, ValueError, KeyError):
        complete_output = None
    if complete_output is not None:
        return {
            "status": "recovery_not_needed_output_complete",
            "original_attempt_id": source_step["attempt_id"],
            "original_attempt_exit_status": exit_status,
            "original_training_state_sha256": state_sha,
            "original_output_audit": complete_output,
            "reason": "all fixed epoch-3 artifacts passed the frozen R1 producer gate",
        }
    recovery_command = command_for(source_step["command"], packet, args.timeout_seconds)
    recovery_output = root / packet["resume_output_path"]
    recovery_attempt = (
        root / packet["runtime_relative"] / "attempts" / f"{packet['resume_attempt_id']}.json"
    )
    seed4_output = root / seed4_step["expected_output_path"]
    seed4_attempt = (
        root / packet["runtime_relative"] / "attempts" / f"{seed4_step['attempt_id']}.json"
    )
    for path in (recovery_output, recovery_attempt, seed4_output, seed4_attempt):
        if path.is_symlink() or path.exists():
            raise FileExistsError(
                f"recovery/seed4 output already exists; refusing overwrite: {path}"
            )
    request = {
        "schema": "wnn.fold1_gru384_resume_request.r2",
        "status": "prepared_for_root_admission",
        "prepared_utc": datetime.now(UTC).isoformat(),
        "packet_path": str(packet_path),
        "packet_sha256": args.packet_sha256,
        "recovery_gate_source_sha256": sha256_file(Path(__file__).resolve()),
        "original_postfit_gate_sha256": packet["original_postfit_gate_sha256"],
        "original_postfit_gate_packet_sha256": packet["original_postfit_gate_packet_sha256"],
        "original_queue_sha256": packet["queue_sha256"],
        "original_allocation_sha256": packet["original_allocation_sha256"],
        "original_attempt_id": source_step["attempt_id"],
        "original_attempt_record_sha256": sha256_file(original_attempt_path),
        "original_attempt_status": status,
        "original_attempt_exit_status": exit_status,
        "original_attempt_elapsed_seconds": record.get("elapsed_seconds"),
        "original_unit_terminal_state": unit_state,
        "original_training_state": state_identity,
        "dataset_identity_receipt": data_identity,
        "dataset_identity_receipt_sha256": canonical_sha256(data_identity),
        "source_identity_helper_sha256": packet["source_identity_helper_sha256"],
        "resume_attempt_id": packet["resume_attempt_id"],
        "resume_run_name": packet["resume_run_name"],
        "resume_output_path": packet["resume_output_path"],
        "resume_command": recovery_command,
        "resume_command_sha256_canonical_json": canonical_sha256(recovery_command),
        "resume_timeout_seconds": args.timeout_seconds,
        "same_four_epoch_horizon": state["contract"]["epochs"] == 4,
        "same_scientific_recipe": True,
        "seed4_attempt_remains_unlaunched": not seed4_attempt.exists(),
        "max_new_output_bytes": packet["resource_request"]["maximum_new_bytes"],
        "minimum_free_bytes": packet["minimum_free_bytes"],
        "resource_request": packet["resource_request"],
        "source_pins_sha256": packet["source_pins_sha256"],
    }
    write_exclusive_json(args.output, request)
    return request


def verify_authority(authority: Json, authority_sha: str, request: Json, request_sha: str) -> None:
    if request.get("schema") != "wnn.fold1_gru384_resume_request.r2":
        raise ValueError("unexpected recovery request schema")
    if request.get("status") != "prepared_for_root_admission":
        raise ValueError("recovery request is not in the prepared state")
    dataset_identity = request.get("dataset_identity_receipt")
    if not isinstance(dataset_identity, dict) or canonical_sha256(dataset_identity) != request.get(
        "dataset_identity_receipt_sha256"
    ):
        raise ValueError("recovery request dataset identity receipt hash is inconsistent")
    expected = {
        "schema": "wnn.fold1_gru384_resume_authority.r2",
        "status": "conditional_exact_fold1_seed3_resume_authorized",
        "request_sha256": request_sha,
        "recovery_gate_source_sha256": request["recovery_gate_source_sha256"],
        "packet_sha256": request["packet_sha256"],
        "original_postfit_gate_sha256": request["original_postfit_gate_sha256"],
        "original_postfit_gate_packet_sha256": request["original_postfit_gate_packet_sha256"],
        "original_queue_sha256": request["original_queue_sha256"],
        "original_allocation_sha256": request["original_allocation_sha256"],
        "original_attempt_id": request["original_attempt_id"],
        "original_attempt_record_sha256": request["original_attempt_record_sha256"],
        "original_training_state_sha256": request["original_training_state"]["sha256"],
        "original_training_state_contract_sha256": request["original_training_state"][
            "contract_sha256"
        ],
        "original_next_epoch": request["original_training_state"]["next_epoch"],
        "dataset_identity_receipt_sha256": request["dataset_identity_receipt_sha256"],
        "source_identity_helper_sha256": request["source_identity_helper_sha256"],
        "resume_attempt_id": request["resume_attempt_id"],
        "resume_run_name": request["resume_run_name"],
        "resume_output_path": request["resume_output_path"],
        "resume_timeout_seconds": request["resume_timeout_seconds"],
        "command_sha256_canonical_json": request["resume_command_sha256_canonical_json"],
        "source_pins_sha256": request["source_pins_sha256"],
        "resource_request": request["resource_request"],
        "maximum_new_bytes": request["max_new_output_bytes"],
    }
    mismatches = {
        key: (authority.get(key), value)
        for key, value in expected.items()
        if authority.get(key) != value
    }
    if mismatches:
        raise ValueError(f"root authority does not bind the exact recovery request: {mismatches}")
    if len(authority_sha) != 64:
        raise ValueError("invalid root authority SHA-256")


def dispatch(args: argparse.Namespace) -> Json:
    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    packet = packet_for(root, packet_path, args.packet_sha256)
    request_path = args.request.resolve(strict=True)
    authority_path = args.authority.resolve(strict=True)
    request_sha = sha256_file(request_path)
    authority_sha = sha256_file(authority_path)
    if args.authority_sha256 != authority_sha:
        raise ValueError("root authority hash differs from the dispatch command")
    request = read_json(request_path)
    authority = read_json(authority_path)
    verify_authority(authority, authority_sha, request, request_sha)
    if request.get("packet_sha256") != args.packet_sha256:
        raise ValueError("request was created from a different static recovery packet")
    if (
        request.get("packet_path") != str(packet_path)
        or request.get("original_attempt_id") != packet["original_attempt_id"]
        or request.get("resume_attempt_id") != packet["resume_attempt_id"]
        or request.get("resume_run_name") != packet["resume_run_name"]
        or request.get("resume_output_path") != packet["resume_output_path"]
        or request.get("source_pins_sha256") != packet["source_pins_sha256"]
        or request.get("resource_request") != packet["resource_request"]
        or request.get("max_new_output_bytes") != packet["resource_request"]["maximum_new_bytes"]
        or request.get("minimum_free_bytes") != packet["minimum_free_bytes"]
    ):
        raise ValueError("recovery request differs from the frozen packet fields")
    if (
        request.get("original_queue_sha256") != packet["queue_sha256"]
        or request.get("original_allocation_sha256") != packet["original_allocation_sha256"]
        or request.get("original_postfit_gate_sha256") != packet["original_postfit_gate_sha256"]
        or request.get("original_postfit_gate_packet_sha256")
        != packet["original_postfit_gate_packet_sha256"]
    ):
        raise ValueError("request is not bound to the frozen R1 producer/queue lineage")
    if request.get("source_identity_helper_sha256") != packet["source_identity_helper_sha256"]:
        raise ValueError("request does not bind the pinned dataset identity helper")
    if sha256_file(Path(__file__).resolve()) != request.get("recovery_gate_source_sha256"):
        raise ValueError("recovery gate source changed after request creation")
    _queue, _old_allocation, source_step, _seed4_step = load_original_inputs(root, packet)
    record, unit_state = terminal_attempt(root, packet, source_step)
    original_attempt_path = (
        root / packet["runtime_relative"] / "attempts" / f"{packet['original_attempt_id']}.json"
    )
    if record.get("status") != "failed" or record.get("exit_status") != 124:
        raise ValueError("original timeout precondition no longer holds")
    if (
        record.get("elapsed_seconds") != request.get("original_attempt_elapsed_seconds")
        or record.get("status") != request.get("original_attempt_status")
        or record.get("exit_status") != request.get("original_attempt_exit_status")
        or unit_state != request.get("original_unit_terminal_state")
    ):
        raise ValueError("original terminal timeout evidence changed after root admission")
    if sha256_file(original_attempt_path) != request.get("original_attempt_record_sha256"):
        raise ValueError("original terminal attempt receipt changed after root admission")
    verify_no_oom(record, packet)
    _state, state_sha, state_identity, data_identity, _split, _gate = verify_input_state(
        root, packet, source_step
    )
    if (
        state_sha != request["original_training_state"]["sha256"]
        or state_identity != request["original_training_state"]
    ):
        raise ValueError("original training state changed after root admission")
    if (
        data_identity != request.get("dataset_identity_receipt")
        or canonical_sha256(data_identity) != request.get("dataset_identity_receipt_sha256")
        or request.get("source_identity_helper_sha256") != packet["source_identity_helper_sha256"]
    ):
        raise ValueError("dataset identity changed after root admission")
    expected_command = command_for(
        source_step["command"], packet, int(request["resume_timeout_seconds"])
    )
    if expected_command != request.get("resume_command"):
        raise ValueError("resume command differs from the frozen original scientific recipe")
    if (root / packet["runtime_relative"] / "STOP_ADMISSION.json").exists():
        raise RuntimeError("campaign stop marker is present; recovery remains unlaunched")
    if shutil.disk_usage(root).free < int(packet["minimum_free_bytes"]):
        raise RuntimeError("free disk fell below the admitted recovery floor")
    output = root / packet["resume_output_path"]
    attempt_path = (
        root / packet["runtime_relative"] / "attempts" / f"{packet['resume_attempt_id']}.json"
    )
    if output.is_symlink() or output.exists() or attempt_path.exists():
        raise FileExistsError("recovery output/attempt already exists; refusing relaunch")
    command = request["resume_command"]
    if canonical_sha256(command) != request["resume_command_sha256_canonical_json"]:
        raise ValueError("recovery command hash differs from the frozen request")
    launch = [
        sys.executable,
        str(root / "scripts/bandit_resources.py"),
        "--root",
        str(root),
        "--state",
        str(root / packet["runtime_relative"]),
        "launch",
        "--attempt",
        packet["resume_attempt_id"],
        "--kind",
        "gpu",
        "--cpus",
        "2",
        "--reserve-bytes",
        "9000000000",
        "--profile",
        "FOLD1-GRU384-SEED3-EXACT-EPOCH-RESUME",
        "--contract",
        str(authority_path),
        "--admission-timeout",
        "45",
        "--",
        *command,
    ]
    result = subprocess.run(launch, cwd=root, capture_output=True, text=True, timeout=50)
    if result.returncode:
        raise RuntimeError(f"guarded recovery launch failed: {result.stderr[-3000:]}")
    launched = read_json(attempt_path)
    if launched.get("command") != command or launched.get("contract_sha256") != authority_sha:
        raise ValueError(
            "launched recovery attempt is not bound to the admitted command/allocation"
        )
    if launched.get("status") not in {"reserved", "waiting", "running"}:
        raise RuntimeError(f"guardian did not reserve a live recovery: {launched.get('status')}")
    return {
        "status": launched["status"],
        "attempt_id": packet["resume_attempt_id"],
        "unit": launched["unit"],
        "authority_sha256": authority_sha,
        "command_sha256_canonical_json": canonical_sha256(command),
    }


def audit(args: argparse.Namespace) -> Json:
    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    packet = packet_for(root, packet_path, args.packet_sha256)
    request = read_json(args.request.resolve(strict=True))
    authority_path = args.authority.resolve(strict=True)
    authority_sha = sha256_file(authority_path)
    if args.authority_sha256 != authority_sha:
        raise ValueError("root authority hash differs from the audit command")
    verify_authority(
        read_json(authority_path),
        authority_sha,
        request,
        sha256_file(args.request.resolve(strict=True)),
    )
    if request.get("packet_sha256") != args.packet_sha256:
        raise ValueError("request was created from a different static recovery packet")
    if (
        request.get("packet_path") != str(packet_path)
        or request.get("original_attempt_id") != packet["original_attempt_id"]
        or request.get("resume_attempt_id") != packet["resume_attempt_id"]
        or request.get("resume_run_name") != packet["resume_run_name"]
        or request.get("resume_output_path") != packet["resume_output_path"]
        or request.get("source_pins_sha256") != packet["source_pins_sha256"]
        or request.get("resource_request") != packet["resource_request"]
        or request.get("max_new_output_bytes") != packet["resource_request"]["maximum_new_bytes"]
        or request.get("minimum_free_bytes") != packet["minimum_free_bytes"]
    ):
        raise ValueError("recovery request differs from the frozen packet fields")
    if request.get("source_identity_helper_sha256") != packet["source_identity_helper_sha256"]:
        raise ValueError("request does not bind the pinned dataset identity helper")
    if (
        request.get("original_queue_sha256") != packet["queue_sha256"]
        or request.get("original_allocation_sha256") != packet["original_allocation_sha256"]
        or request.get("original_postfit_gate_sha256") != packet["original_postfit_gate_sha256"]
        or request.get("original_postfit_gate_packet_sha256")
        != packet["original_postfit_gate_packet_sha256"]
    ):
        raise ValueError("request is not bound to the frozen R1 producer/queue lineage")
    if sha256_file(Path(__file__).resolve()) != request.get("recovery_gate_source_sha256"):
        raise ValueError("recovery gate source changed after request creation")
    _queue, _allocation, source_step, _seed4_step = load_original_inputs(root, packet)
    original_attempt_path = (
        root / packet["runtime_relative"] / "attempts" / f"{packet['original_attempt_id']}.json"
    )
    if sha256_file(original_attempt_path) != request.get("original_attempt_record_sha256"):
        raise ValueError("original terminal attempt receipt changed after root admission")
    expected_command = command_for(
        source_step["command"], packet, int(request["resume_timeout_seconds"])
    )
    if request.get("resume_command") != expected_command or canonical_sha256(
        expected_command
    ) != request.get("resume_command_sha256_canonical_json"):
        raise ValueError("audited recovery command differs from the frozen scientific recipe")
    recovery_step = {
        "attempt_id": packet["resume_attempt_id"],
        "command": request["resume_command"],
        "command_sha256_canonical_json": request["resume_command_sha256_canonical_json"],
        "run_name": packet["resume_run_name"],
        "expected_output_path": packet["resume_output_path"],
        "maximum_new_bytes": int(request["max_new_output_bytes"]),
    }
    runtime = root / packet["runtime_relative"]
    attempt = read_json(runtime / "attempts" / f"{packet['resume_attempt_id']}.json")
    gate = load_old_gate(root, packet)
    gate.verify_attempt_record(attempt, recovery_step, authority_sha)
    if os.environ.get("WNN_PORTABLE_DATA_MANIFEST"):
        raise ValueError(
            "portable source manifests trigger full-file verification in the split loader; "
            "recovery audit requires the native immutable dataset path"
        )
    from wnn.splits import load_campaign_split

    split = load_campaign_split(root / packet["fold_manifest_path"], root)
    source_pins = packet["source_pins_sha256"]
    result = gate.verify_output(root, recovery_step, split, source_pins)
    provenance = read_json(root / packet["resume_output_path"] / "provenance.json")
    resumed_from = provenance.get("resumed_from", {})
    if (
        resumed_from.get("sha256") != request["original_training_state"]["sha256"]
        or resumed_from.get("next_epoch") != request["original_training_state"]["next_epoch"]
        or Path(resumed_from.get("path", "")).resolve()
        != Path(request["original_training_state"]["path"]).resolve()
    ):
        raise ValueError("recovered output does not cite the exact original terminal state")
    import torch

    selected_checkpoint = torch.load(
        root / packet["resume_output_path"] / "model_ep3.pt",
        map_location="cpu",
        weights_only=False,
    )
    embedded_resume = selected_checkpoint.get("provenance", {}).get("resumed_from", {})
    if (
        embedded_resume.get("sha256") != request["original_training_state"]["sha256"]
        or embedded_resume.get("next_epoch") != request["original_training_state"]["next_epoch"]
        or Path(embedded_resume.get("path", "")).resolve()
        != Path(request["original_training_state"]["path"]).resolve()
    ):
        raise ValueError("epoch-3 checkpoint does not embed the exact original resume ancestry")
    del selected_checkpoint

    recovered = torch.load(
        root / packet["resume_output_path"] / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    recovered_contract = recovered.get("contract")
    recovered_next_epoch = recovered.get("next_epoch")
    del recovered
    original_path = Path(request["original_training_state"]["path"])
    if sha256_file(original_path) != request["original_training_state"]["sha256"]:
        raise ValueError("original terminal state changed after recovery")
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    if (
        original.get("next_epoch") != request["original_training_state"]["next_epoch"]
        or recovered_contract != original.get("contract")
        or recovered_next_epoch != 4
    ):
        raise ValueError(
            "recovery changed the training contract or failed to complete the four-epoch horizon"
        )
    del original
    receipt = {
        "schema": "wnn.fold1_gru384_resume_terminal_audit.r2",
        "status": "recovery_fit_and_output_passed",
        "attempt_id": packet["resume_attempt_id"],
        "attempt_sha256": sha256_file(runtime / "attempts" / f"{packet['resume_attempt_id']}.json"),
        "original_attempt_record_sha256": request["original_attempt_record_sha256"],
        "authority_sha256": authority_sha,
        "request_sha256": sha256_file(args.request.resolve(strict=True)),
        "original_training_state_sha256": request["original_training_state"]["sha256"],
        "original_training_state_contract_sha256": request["original_training_state"][
            "contract_sha256"
        ],
        "dataset_identity_receipt_sha256": request["dataset_identity_receipt_sha256"],
        "source_identity_helper_sha256": request["source_identity_helper_sha256"],
        "recovered_training_state_sha256": result["training_state_sha256"],
        "recovered_checkpoint_sha256": result["checkpoint_sha256"],
        "recovered_model_sha256": result["model_sha256"],
        "fit_membership_counts": result["fit_membership_counts"],
        "optimizer_totals": result["optimizer_totals"],
        "seed4_still_requires_its_separate_root_admission": True,
    }
    write_exclusive_json(args.output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--packet", type=Path, default=BASE / "RECOVERY_GATE_PACKET_R2.json")
    parser.add_argument("--packet-sha256", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare", help="inspect the terminal timeout and freeze its state SHA"
    )
    prepare_parser.add_argument("--timeout-seconds", type=int, default=14400)
    prepare_parser.add_argument("--output", type=Path, default=BASE / "RESUME_REQUEST_R2.json")
    dispatch_parser = subparsers.add_parser(
        "dispatch", help="launch only under a matching root admission"
    )
    dispatch_parser.add_argument("--request", type=Path, default=BASE / "RESUME_REQUEST_R2.json")
    dispatch_parser.add_argument("--authority", type=Path, required=True)
    dispatch_parser.add_argument("--authority-sha256", required=True)
    audit_parser = subparsers.add_parser(
        "audit", help="verify the completed recovery and preserve a terminal receipt"
    )
    audit_parser.add_argument("--request", type=Path, default=BASE / "RESUME_REQUEST_R2.json")
    audit_parser.add_argument("--authority", type=Path, required=True)
    audit_parser.add_argument("--authority-sha256", required=True)
    audit_parser.add_argument("--output", type=Path, default=BASE / "RESUME_TERMINAL_AUDIT_R2.json")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args)
    elif args.action == "dispatch":
        result = dispatch(args)
    else:
        result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
