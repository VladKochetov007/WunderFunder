#!/usr/bin/env python3
"""Freeze and audit the exact final-epoch fold1 seed3 continuation R3.

Preparation is read-only except for an exclusive request receipt. There is no
fit launch command in this module; root admission and the campaign controller
own dispatch. Audit requires the exact root authority created from that request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Json = dict[str, Any]
BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
DEFAULT_PACKET = BASE / "RECOVERY_GATE_PACKET_R3.json"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible values in canonical controller format."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Json:
    """Read one JSON object and reject other top-level shapes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_exclusive_json(path: Path, value: object) -> None:
    """Persist a new receipt without replacing evidence from a previous attempt."""
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
        raise FileExistsError(f"refusing to replace recovery evidence: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def source_pin_partition(packet: Json) -> tuple[list[str], list[str]]:
    """Split controller-captured paths from the packet-loader-only inventory pin."""
    pins = packet.get("source_pins_sha256")
    attempt_paths = packet.get("attempt_source_receipt_paths")
    loader_paths = packet.get("packet_loader_only_source_paths")
    if not isinstance(pins, dict) or not all(
        isinstance(path, str) and isinstance(digest, str) and len(digest) == 64
        for path, digest in pins.items()
    ):
        raise ValueError("source pins must map repository paths to SHA-256 values")
    if (
        not isinstance(attempt_paths, list)
        or not isinstance(loader_paths, list)
        or any(not isinstance(path, str) for path in (*attempt_paths, *loader_paths))
        or len(attempt_paths) != len(set(attempt_paths))
        or len(loader_paths) != len(set(loader_paths))
        or set(attempt_paths) & set(loader_paths)
        or set(attempt_paths) | set(loader_paths) != set(pins)
    ):
        raise ValueError("source pin receipt paths do not partition every pinned input")
    inventory_path = (
        "research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/"
        "sol_continuation/FOLD1_DEPENDENCY_INVENTORY_R1.json"
    )
    if loader_paths != [inventory_path]:
        raise ValueError("only the exact dependency inventory may be loader-only pinned")
    return attempt_paths, loader_paths


def verify_packet_source_pins(root: Path, packet: Json) -> None:
    """Hash all source pins from disk, including files outside controller capture."""
    source_pin_partition(packet)
    for relative, expected in packet["source_pins_sha256"].items():
        path = root / relative
        if sha256_file(path) != expected:
            raise ValueError(f"source pin changed: {relative}")


def verify_attempt_source_receipts(attempt: Json, packet: Json) -> None:
    """Match only command/source paths emitted by the resource-controller receipt."""
    attempt_paths, _loader_paths = source_pin_partition(packet)
    observed = attempt.get("source_sha256")
    if not isinstance(observed, dict):
        raise ValueError("R3 attempt source receipt is missing")
    for relative in attempt_paths:
        if observed.get(relative) != packet["source_pins_sha256"][relative]:
            raise ValueError(f"R3 attempt source receipt differs: {relative}")


def load_packet(root: Path, packet_path: Path, expected_sha: str) -> Json:
    """Validate the static R3 packet and every frozen predecessor file."""
    if sha256_file(packet_path) != expected_sha:
        raise ValueError("R3 static recovery packet hash changed")
    packet = read_json(packet_path)
    if packet.get("schema") != "wnn.fold1_gru384_recovery_gate_packet.r3":
        raise ValueError("unexpected R3 packet schema")
    if Path(packet["root"]).resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("R3 packet names a different project root")
    for section in ("r2", "postfit_gate", "allocation", "root_cache_handoff"):
        values = packet[section]
        for key, expected in values.items():
            if key.endswith("_path"):
                hash_key = key.removesuffix("_path") + "_sha256"
                if hash_key not in values:
                    continue
                path = root / expected
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"required frozen input is missing or symlinked: {path}")
                if sha256_file(path) != values[hash_key]:
                    raise ValueError(f"frozen R3 input changed: {path}")
    handoff = packet["root_cache_handoff"]
    if sha256_file(root / handoff["path"]) != handoff["sha256"]:
        raise ValueError("root cache-handoff receipt changed")
    for dropin_path_key, dropin_sha_key in (
        ("unit_memory_high_dropin_path", "unit_memory_high_dropin_sha256"),
        ("seed4_unit_memory_high_dropin_path", "seed4_unit_memory_high_dropin_sha256"),
    ):
        section = packet["allocation"] if dropin_path_key.startswith("seed4_") else packet["resume"]
        dropin = Path(section[dropin_path_key])
        if not dropin.is_file() or sha256_file(dropin) != section[dropin_sha_key]:
            raise ValueError(f"required preinstalled MemoryHigh drop-in changed: {dropin}")
    verify_packet_source_pins(root, packet)
    return packet


def load_r2_gate(root: Path, packet: Json) -> Any:
    """Load only the exact previously reviewed R2 gate source."""
    source = root / packet["r2"]["gate_path"]
    if sha256_file(source) != packet["r2"]["gate_sha256"]:
        raise ValueError("frozen R2 recovery gate source changed")
    spec = importlib.util.spec_from_file_location("fold1_gru384_recovery_gate_r2", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen R2 recovery gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_r2_inputs(root: Path, packet: Json) -> tuple[Any, Json, Json, Json, Json, Json]:
    """Validate R2 authority/request and recover the original queue lineage."""
    gate = load_r2_gate(root, packet)
    r2 = packet["r2"]
    r2_packet_path = root / r2["packet_path"]
    r2_packet = gate.packet_for(root, r2_packet_path, r2["packet_sha256"])
    request_path = root / r2["request_path"]
    authority_path = root / r2["authority_path"]
    request = gate.read_json(request_path)
    authority = gate.read_json(authority_path)
    gate.verify_authority(
        authority,
        sha256_file(authority_path),
        request,
        sha256_file(request_path),
    )
    if request.get("recovery_gate_source_sha256") != r2["gate_sha256"]:
        raise ValueError("R2 request does not bind its frozen gate source")
    queue, allocation, original_seed3, seed4 = gate.load_original_inputs(root, r2_packet)
    if allocation.get("status") != "conditional_exact_successor_authorized":
        raise ValueError("original two-fit authority is not the frozen conditional allocation")
    original_attempt_path = (
        root
        / r2_packet["runtime_relative"]
        / "attempts"
        / f"{r2_packet['original_attempt_id']}.json"
    )
    if sha256_file(original_attempt_path) != request.get("original_attempt_record_sha256"):
        raise ValueError("original seed3 terminal attempt changed after R2 admission")
    gate.terminal_attempt(root, r2_packet, original_seed3)
    return (
        gate,
        r2_packet,
        request,
        authority,
        queue,
        {
            "original_seed3": original_seed3,
            "seed4": seed4,
        },
    )


def require_zero_oom_deltas(record: Json) -> None:
    """Fail closed if any attempt, parent or hierarchical OOM counter is absent/nonzero."""
    for name in (
        "memory_event_deltas",
        "final_memory_event_deltas",
        "parent_memory_event_deltas",
        "final_parent_memory_event_deltas",
        "parent_hierarchical_memory_event_deltas",
        "final_parent_hierarchical_memory_event_deltas",
    ):
        events = record.get(name)
        if not isinstance(events, dict) or any(
            type(events.get(key)) is not int or events.get(key) != 0
            for key in ("oom", "oom_kill", "oom_group_kill")
        ):
            raise ValueError(f"R2 attempt has missing/nonzero OOM evidence in {name}")


def verify_terminal_r2(
    root: Path, packet: Json, r2_gate: Any, r2_request: Json
) -> tuple[Json, Json]:
    """Prove R2 failed at the sampled reserve, not from CUDA/cgroup OOM."""
    path = root / packet["r2"]["attempt_path"]
    record = r2_gate.read_json(path)
    if sha256_file(path) != packet["r2"]["attempt_sha256"]:
        raise ValueError("terminal R2 attempt receipt changed")
    if record.get("id") != packet["r2"]["attempt_id"]:
        raise ValueError("R2 attempt ID differs from the packet")
    if (
        record.get("status") != "failed"
        or record.get("phase") != "terminal"
        or record.get("exit_status") != -15
        or record.get("failure")
        != "attempt memory use exceeded its 9000000000 byte profile reserve"
    ):
        raise ValueError("R2 terminal reason is not the exact bounded reserve kill")
    if record.get("command") != r2_request.get("resume_command"):
        raise ValueError("R2 executed argv differs from its frozen request")
    if canonical_sha256(record["command"]) != r2_request.get(
        "resume_command_sha256_canonical_json"
    ):
        raise ValueError("R2 command hash differs from its frozen request")
    if record.get("contract_sha256") != packet["r2"]["authority_sha256"]:
        raise ValueError("R2 attempt is not bound to the frozen R2 authority")
    peaks = (record.get("peak_ram_bytes"), record.get("peak_observed_vram_bytes"))
    if any(type(value) is not int or value <= 0 for value in peaks):
        raise ValueError("R2 terminal attempt lacks positive RAM/VRAM measurements")
    if record["peak_ram_bytes"] <= 9_000_000_000:
        raise ValueError("R2 record does not establish the supervisor reserve trigger")
    if record["peak_observed_vram_bytes"] >= 5_000_000_000:
        raise ValueError("R2 reached its observed VRAM cap")
    require_zero_oom_deltas(record)
    observers = record.get("observer_status", {})
    for key in (
        "attempt_memory_events_final",
        "attempt_memory_peak_final",
        "attempt_systemd_runtime",
        "parent_memory_events_final",
        "parent_hierarchical_memory_events_final",
        "nvidia_smi_final",
    ):
        if observers.get(key) != "ok":
            raise ValueError(f"R2 resource observer is incomplete: {key}")
    unit = record.get("unit")
    if unit != f"wnn-bandit-{packet['r2']['attempt_id']}.service":
        raise ValueError("R2 systemd unit identity differs")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus,MemoryPeak,MemoryMax",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    unit_state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if unit_state.get("ActiveState") != "failed" or unit_state.get("Result") != "exit-code":
        raise ValueError(f"R2 service is not in its recorded failed terminal state: {unit_state}")
    if unit_state.get("ExecMainStatus") != "1":
        raise ValueError(f"R2 supervisor service exit differs from receipt: {unit_state}")
    if int(unit_state.get("MemoryPeak", "0")) > 9_000_000_000:
        raise ValueError("R2 systemd peak exceeds its 9 GB MemoryMax")
    if unit_state.get("MemoryMax") != "9000000000":
        raise ValueError(f"R2 MemoryMax differs from its admitted cap: {unit_state}")
    log_path = root / packet["r2"]["log_path"]
    if sha256_file(log_path) != packet["r2"]["log_sha256"]:
        raise ValueError("terminal R2 training log changed")
    log_text = log_path.read_text(encoding="utf-8")
    if "ep 3 batch 460/746" not in log_text or "Traceback (most recent call last)" in log_text:
        raise ValueError(
            "R2 log does not establish the recorded partial final epoch without a trainer traceback"
        )
    if '"epoch": 2' not in log_text or '"epoch": 3' in log_text:
        raise ValueError("R2 log does not end after exactly three saved epoch boundaries")
    return record, unit_state


def verify_cache_handoff(root: Path, packet: Json) -> Json:
    """Bind the reviewed file-cache reclaim and require its recorded headroom."""
    handoff = read_json(root / packet["root_cache_handoff"]["path"])
    if handoff.get("predecessor_sha256") != packet["r2"]["attempt_sha256"]:
        raise ValueError("cache handoff does not name the exact failed R2 attempt")
    if handoff.get("admission_headroom_pass") is not True:
        raise ValueError("root cache handoff did not pass admission headroom")
    after_current = handoff.get("after", {}).get("current")
    high = packet["resume"]["resource_request"]["unit_memory_high_bytes"]
    if type(after_current) is not int or after_current >= high:
        raise ValueError("post-reclaim parent memory is not below the requested 8 GB MemoryHigh")
    return handoff


def validate_r2_checkpoint(
    root: Path, packet: Json, r2_gate: Any, r2_packet: Json, r2_request: Json
) -> Json:
    """Revalidate the chain and epoch-2 state that R3 will consume."""
    _queue, _allocation, original_step, _seed4 = r2_gate.load_original_inputs(root, r2_packet)
    source_state, source_sha, source_identity, data_identity, split, old_gate = (
        r2_gate.verify_input_state(root, r2_packet, original_step)
    )
    if source_sha != packet["r2"]["source_training_state_sha256"]:
        raise ValueError("R2 ancestry source checkpoint changed")
    state_path = root / packet["r2"]["training_state_path"]
    provenance_path = root / packet["r2"]["output_path"] / "provenance.json"
    checkpoint_path = root / packet["r2"]["checkpoint_path"]
    if sha256_file(provenance_path) != packet["r2"]["provenance_sha256"]:
        raise ValueError("terminal R2 provenance changed")
    provenance = r2_gate.read_json(provenance_path)
    r2_step = {
        "attempt_id": packet["r2"]["attempt_id"],
        "run_name": r2_request["resume_run_name"],
        "expected_output_path": packet["r2"]["output_path"],
        "command": r2_request["resume_command"],
        "command_sha256_canonical_json": r2_request["resume_command_sha256_canonical_json"],
    }
    old_gate.verify_provenance_membership(
        r2_step, provenance, split, r2_packet["source_pins_sha256"], False
    )
    resumed = provenance.get("resumed_from", {})
    if (
        resumed.get("sha256") != source_sha
        or resumed.get("next_epoch") != source_identity["next_epoch"]
        or Path(resumed.get("path", "")).resolve() != Path(source_identity["path"]).resolve()
    ):
        raise ValueError("R2 output provenance does not preserve exact source-state ancestry")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != packet["r2"]["checkpoint_sha256"]:
        raise ValueError("epoch-2 R2 checkpoint changed")
    if provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != checkpoint_hash:
        raise ValueError("R2 provenance does not bind the epoch-2 checkpoint")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != 2
        or checkpoint.get("args") != provenance.get("args")
        or checkpoint.get("args", {}).get("name") != r2_request["resume_run_name"]
    ):
        raise ValueError("R2 checkpoint is not the exact fixed-recipe epoch-2 snapshot")
    old_gate.verify_provenance_membership(
        r2_step,
        checkpoint.get("provenance", {}),
        split,
        r2_packet["source_pins_sha256"],
        False,
    )
    r2_gate.verify_finite_tensors(checkpoint, torch)
    del checkpoint

    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("R2 continuation state is missing or symlinked")
    state_hash = sha256_file(state_path)
    if state_hash != packet["r2"]["training_state_sha256"]:
        raise ValueError("R2 epoch-boundary training state changed")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or state.get("format") != "wnn.epoch-training-state"
        or state.get("version") != 1
        or state.get("boundary") != "epoch"
        or state.get("next_epoch") != 3
    ):
        raise ValueError("R2 state is not the epoch-2 completion boundary")
    required = {
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
    if required - state.keys():
        raise ValueError(f"R2 checkpoint is incomplete: {sorted(required - state.keys())}")
    if state["contract"] != source_state["contract"]:
        raise ValueError("R2 continuation changed the original fit/source/data contract")
    if (
        canonical_sha256(state["contract"])
        != r2_request["original_training_state"]["contract_sha256"]
    ):
        raise ValueError("R2 continuation contract hash differs from the admitted source contract")
    if state["contract"].get("total_steps") != 29840 or state["contract"].get("epochs") != 4:
        raise ValueError("R2 state no longer spans the frozen four-epoch scheduler horizon")
    if state["scheduler"].get("total_steps") != 29840:
        raise ValueError("R2 scheduler horizon differs from the admitted four-epoch schedule")
    if not {"torch", "numpy", "python", "cuda"}.issubset(state["rng"]):
        raise ValueError("R2 checkpoint lacks one or more saved RNG streams")
    if state["contract"].get("datasets") != source_state["contract"].get("datasets"):
        raise ValueError("R2 checkpoint dataset hashes differ from the original continuation")
    r2_gate.verify_finite_tensors(state, torch)
    progress = state["progress"]
    history = progress.get("history", [])
    if [entry.get("epoch") for entry in history] != [0, 1, 2] or any(
        entry.get("evaluation") != "withheld" for entry in history
    ):
        raise ValueError("R2 state does not contain exactly three withheld-score epochs")
    diagnostics = progress.get("update_diagnostics", [])
    if len(diagnostics) != 3:
        raise ValueError("R2 state does not contain diagnostics for its three completed epochs")
    for index, entry in enumerate(diagnostics):
        epoch = entry.get("epoch", {})
        attempts = epoch.get("optimizer_attempts")
        completed = epoch.get("optimizer_completed_updates")
        skipped = epoch.get("amp_skipped_updates")
        batches = epoch.get("completed_batches")
        if (
            entry.get("epoch_index") != index
            or type(attempts) is not int
            or type(completed) is not int
            or type(skipped) is not int
            or type(batches) is not int
            or attempts <= 0
            or completed <= 0
            or skipped < 0
            or batches <= 0
            or attempts != completed + skipped
        ):
            raise ValueError(f"R2 epoch update diagnostics do not reconcile at {index}")
    verified = {
        "path": str(state_path.resolve()),
        "sha256": state_hash,
        "bytes": state_path.stat().st_size,
        "next_epoch": state["next_epoch"],
        "contract_sha256": canonical_sha256(state["contract"]),
        "completed_epoch_count": len(history),
        "completed_optimizer_attempts": sum(
            entry["epoch"]["optimizer_attempts"] for entry in diagnostics
        ),
        "completed_optimizer_updates": sum(
            entry["epoch"]["optimizer_completed_updates"] for entry in diagnostics
        ),
        "completed_amp_skips": sum(entry["epoch"]["amp_skipped_updates"] for entry in diagnostics),
        "scheduler_total_steps": state["scheduler"]["total_steps"],
        "rng_streams": sorted(state["rng"]),
        "dataset_identity_receipt": data_identity,
        "dataset_identity_receipt_sha256": r2_gate.canonical_sha256(data_identity),
    }
    del state, source_state
    return verified


def resume_command(r2_request: Json, packet: Json) -> list[str]:
    """Clone R2 argv, changing only timeout, unique output name and input state."""
    command = list(r2_request["resume_command"])
    if command[:4] != ["timeout", "--signal=TERM", "--kill-after=20s", "14400"]:
        raise ValueError("R2 timeout prefix differs from the frozen recovery command")
    command[3] = str(packet["resume"]["timeout_seconds"])
    name_index = command.index("--name") + 1
    if command[name_index] != packet["r2"]["output_path"].split("/")[-1]:
        raise ValueError("R2 name does not match its frozen output name")
    command[name_index] = packet["resume"]["run_name"]
    state_index = command.index("--resume-state") + 1
    if command[state_index] != str(
        (Path(packet["root"]) / packet["r2"]["source_training_state_path"]).resolve()
    ):
        raise ValueError("R2 command does not resume from the original epoch-0 checkpoint")
    command[state_index] = str(
        (Path(packet["root"]) / packet["resume"]["resume_from_path"]).resolve()
    )
    if command.count("--resume-state") != 1 or "--save-training-state" not in command:
        raise ValueError("R3 command does not preserve full continuation-state saving")
    if "--epochs" not in command or command[command.index("--epochs") + 1] != "4":
        raise ValueError("R3 command must retain the original four-epoch schedule horizon")
    return command


def prepare(args: argparse.Namespace) -> Json:
    """Verify terminal R2, freeze its exact epoch-boundary state and R3 argv."""
    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    packet = load_packet(root, packet_path, args.packet_sha256)
    r2_gate, r2_packet, r2_request, _r2_authority, _queue, steps = load_r2_inputs(root, packet)
    record, unit_state = verify_terminal_r2(root, packet, r2_gate, r2_request)
    cache_handoff = verify_cache_handoff(root, packet)
    _verified_r2_state = validate_r2_checkpoint(root, packet, r2_gate, r2_packet, r2_request)
    runtime = root / packet["runtime_relative"]
    stop_path = runtime / "STOP_ADMISSION.json"
    admission_barrier = (
        {"present": True, "sha256": sha256_file(stop_path), "receipt": read_json(stop_path)}
        if stop_path.is_file()
        else {"present": False}
    )
    if shutil.disk_usage(root).free < packet["minimum_free_bytes"]:
        raise RuntimeError("free disk is below the R3 checkpoint-output floor")
    output = root / packet["resume"]["output_path"]
    attempt_path = runtime / "attempts" / f"{packet['resume']['attempt_id']}.json"
    seed4_output = root / packet["allocation"]["seed4_output_path"]
    seed4_attempt = runtime / "attempts" / f"{packet['allocation']['seed4_attempt_id']}.json"
    if any(path.is_symlink() or path.exists() for path in (output, attempt_path)):
        raise FileExistsError("R3 output/attempt already exists; refusing to overwrite or repeat")
    if seed4_output.exists() or seed4_attempt.exists():
        raise FileExistsError("seed4 is no longer pending; reconcile its existing artifacts")
    command = resume_command(r2_request, packet)
    seed4_step = steps["seed4"]
    request = {
        "schema": "wnn.fold1_gru384_resume_request.r3",
        "status": "prepared_for_root_review_not_admitted",
        "prepared_utc": datetime.now(UTC).isoformat(),
        "packet_path": packet["packet_relative_path"],
        "packet_sha256": args.packet_sha256,
        "recovery_gate_source_sha256": sha256_file(Path(__file__).resolve()),
        "r2_gate_sha256": packet["r2"]["gate_sha256"],
        "r2_packet_sha256": packet["r2"]["packet_sha256"],
        "r2_request_sha256": packet["r2"]["request_sha256"],
        "r2_authority_sha256": packet["r2"]["authority_sha256"],
        "r2_attempt_id": packet["r2"]["attempt_id"],
        "r2_attempt_record_sha256": packet["r2"]["attempt_sha256"],
        "r2_terminal_attempt_status": record["status"],
        "r2_terminal_exit_status": record["exit_status"],
        "r2_terminal_failure": record["failure"],
        "r2_systemd_terminal_state": unit_state,
        "r2_training_state": _verified_r2_state,
        "r2_epoch2_checkpoint_sha256": packet["r2"]["checkpoint_sha256"],
        "root_cache_handoff_sha256": packet["root_cache_handoff"]["sha256"],
        "root_cache_handoff_current_bytes": cache_handoff["after"]["current"],
        "source_training_state_sha256": packet["r2"]["source_training_state_sha256"],
        "resume_attempt_id": packet["resume"]["attempt_id"],
        "resume_run_name": packet["resume"]["run_name"],
        "resume_output_path": packet["resume"]["output_path"],
        "resume_from_path": packet["resume"]["resume_from_path"],
        "resume_training_state_sha256": packet["r2"]["training_state_sha256"],
        "resume_command": command,
        "resume_command_sha256_canonical_json": canonical_sha256(command),
        "schedule": {
            "horizon_epochs": 4,
            "saved_next_epoch_zero_based": 3,
            "remaining_epochs": 1,
            "scheduler_total_steps": 29840,
            "replay_partial_uncheckpointed_epoch": True,
            "checkpoint_boundary_source": "R2 training_state.pt after completed epochs 0,1,2",
        },
        "dataset_identity_receipt": _verified_r2_state["dataset_identity_receipt"],
        "dataset_identity_receipt_sha256": _verified_r2_state["dataset_identity_receipt_sha256"],
        "source_identity_helper_sha256": packet["source_identity_helper_sha256"],
        "source_pins_sha256": packet["source_pins_sha256"],
        "attempt_source_receipt_paths": packet["attempt_source_receipt_paths"],
        "packet_loader_only_source_paths": packet["packet_loader_only_source_paths"],
        "resource_request": packet["resume"]["resource_request"],
        "unit_memory_high_dropin_path": packet["resume"]["unit_memory_high_dropin_path"],
        "unit_memory_high_dropin_sha256": packet["resume"]["unit_memory_high_dropin_sha256"],
        "seed4_conditional_successor": {
            "attempt_id": seed4_step["attempt_id"],
            "run_name": seed4_step["run_name"],
            "output_path": seed4_step["expected_output_path"],
            "command_sha256_canonical_json": seed4_step["command_sha256_canonical_json"],
            "original_authority_sha256": packet["allocation"]["authority_sha256"],
            "still_unlaunched": True,
            "requires_successful_R3_audit_before_handoff": True,
        },
        "admission_barrier_at_preparation": admission_barrier,
        "minimum_free_bytes": packet["minimum_free_bytes"],
        "maximum_new_bytes": packet["maximum_new_bytes"],
    }
    if args.check_only:
        return {
            "status": "static_and_actual_checkpoint_checks_passed_no_receipt",
            "request": request,
        }
    write_exclusive_json(args.output, request)
    return request


def verify_authority(authority: Json, authority_sha: str, request: Json, request_sha: str) -> None:
    """Require root admission to bind this exact checkpoint, command and 8 GB high limit."""
    expected = {
        "schema": "wnn.fold1_gru384_resume_authority.r3",
        "status": "conditional_exact_fold1_seed3_resume_r3_authorized",
        "request_sha256": request_sha,
        "packet_sha256": request["packet_sha256"],
        "recovery_gate_source_sha256": request["recovery_gate_source_sha256"],
        "r2_attempt_record_sha256": request["r2_attempt_record_sha256"],
        "r2_training_state_sha256": request["r2_training_state"]["sha256"],
        "r2_training_state_contract_sha256": request["r2_training_state"]["contract_sha256"],
        "dataset_identity_receipt_sha256": request["dataset_identity_receipt_sha256"],
        "resume_attempt_id": request["resume_attempt_id"],
        "resume_run_name": request["resume_run_name"],
        "resume_output_path": request["resume_output_path"],
        "resume_command_sha256_canonical_json": request["resume_command_sha256_canonical_json"],
        "source_pins_sha256": request["source_pins_sha256"],
        "attempt_source_receipt_paths": request["attempt_source_receipt_paths"],
        "packet_loader_only_source_paths": request["packet_loader_only_source_paths"],
        "resource_request": request["resource_request"],
        "unit_memory_high_dropin_path": request["unit_memory_high_dropin_path"],
        "unit_memory_high_dropin_sha256": request["unit_memory_high_dropin_sha256"],
        "maximum_new_bytes": request["maximum_new_bytes"],
    }
    mismatch = {
        key: (authority.get(key), value)
        for key, value in expected.items()
        if authority.get(key) != value
    }
    if mismatch:
        raise ValueError(f"root authority does not bind the exact R3 request: {mismatch}")
    if len(authority_sha) != 64:
        raise ValueError("invalid root-authority SHA-256")


def recorded_resource_properties(attempt: Json, packet: Json, effective: Json) -> dict[str, str]:
    """Check run-time receipts; unloaded transient units lose their live settings."""
    expected = {
        "MemoryHigh": str(packet["resume"]["resource_request"]["unit_memory_high_bytes"]),
        "MemoryMax": str(packet["resume"]["resource_request"]["unit_memory_max_bytes"]),
        "MemorySwapMax": "0",
        "CPUQuotaPerSecUSec": "2s",
    }
    start = attempt.get("attempt_properties_at_start", {})
    if start.get("ActiveState") != "active":
        raise ValueError("missing active R3 resource receipt")
    if any(start.get(key) != value for key, value in expected.items() if key != "MemoryHigh"):
        raise ValueError("recorded R3 resource properties differ from admission")
    group = start.get("ControlGroup", "")
    if not group.endswith("/wnn-campaign.slice/wnn-bandit-fold1-gru384-s3-resume-r3.service"):
        raise ValueError("recorded R3 cgroup identity differs")
    if any(effective.get(key) != value for key, value in expected.items()):
        raise ValueError("root observed effective R3 resource properties differ")
    if effective.get("ActiveState") not in {"activating", "active"}:
        raise ValueError("root resource observation was not during startup/run")
    if packet["resume"]["unit_memory_high_dropin_path"] not in effective.get("DropInPaths", "").split():
        raise ValueError("root resource observation lacks the exact reviewed drop-in")
    return expected


def audit(args: argparse.Namespace) -> Json:
    """Audit a successful R3 attempt and its final fixed-epoch artifacts."""
    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    packet = load_packet(root, packet_path, args.packet_sha256)
    request_path = args.request.resolve(strict=True)
    authority_path = args.authority.resolve(strict=True)
    request = read_json(request_path)
    authority = read_json(authority_path)
    request_sha = sha256_file(request_path)
    authority_sha = sha256_file(authority_path)
    if args.authority_sha256 != authority_sha:
        raise ValueError("root authority hash differs from the explicit audit pin")
    if sha256_file(BASE / "fold1_gru384_recovery_gate_r3.py") != request.get("recovery_gate_source_sha256"):
        raise ValueError("R3 gate source changed after root review")
    verify_authority(authority, authority_sha, request, request_sha)
    if request.get("packet_sha256") != args.packet_sha256:
        raise ValueError("R3 request names a different static packet")
    r2_gate, r2_packet, r2_request, _r2_authority, _queue, _steps = load_r2_inputs(root, packet)
    if (
        request.get("status") != "prepared_for_root_review_not_admitted"
        or request.get("packet_path") != packet["packet_relative_path"]
        or request.get("resume_command") != resume_command(r2_request, packet)
        or request.get("resume_command_sha256_canonical_json")
        != canonical_sha256(resume_command(r2_request, packet))
        or request.get("source_pins_sha256") != packet["source_pins_sha256"]
        or request.get("attempt_source_receipt_paths") != packet["attempt_source_receipt_paths"]
        or request.get("packet_loader_only_source_paths")
        != packet["packet_loader_only_source_paths"]
        or request.get("resource_request") != packet["resume"]["resource_request"]
        or request.get("resume_training_state_sha256") != packet["r2"]["training_state_sha256"]
        or request.get("unit_memory_high_dropin_sha256")
        != packet["resume"]["unit_memory_high_dropin_sha256"]
    ):
        raise ValueError(
            "R3 request changed from the exact prepared command/source/resource contract"
        )
    old_source_state = validate_r2_checkpoint(root, packet, r2_gate, r2_packet, r2_request)
    if old_source_state != request.get("r2_training_state"):
        raise ValueError("R2 checkpoint/data identity changed after R3 root admission")
    r2_record, _r2_unit = verify_terminal_r2(root, packet, r2_gate, r2_request)
    if sha256_file(root / packet["r2"]["attempt_path"]) != request.get("r2_attempt_record_sha256"):
        raise ValueError("R2 terminal attempt changed after R3 root admission")
    if sha256_file(root / packet["r2"]["training_state_path"]) != request.get(
        "resume_training_state_sha256"
    ):
        raise ValueError("R3 resume checkpoint changed after root admission")
    if r2_record.get("status") != "failed":
        raise ValueError("R2 predecessor status changed")

    runtime = root / packet["runtime_relative"]
    attempt_path = runtime / "attempts" / f"{packet['resume']['attempt_id']}.json"
    attempt = read_json(attempt_path)
    if (
        attempt.get("id") != packet["resume"]["attempt_id"]
        or attempt.get("command") != request.get("resume_command")
        or attempt.get("contract_sha256") != authority_sha
        or attempt.get("status") != "finished"
        or attempt.get("phase") != "terminal"
        or attempt.get("exit_status") != 0
    ):
        raise ValueError("R3 attempt did not finish under the exact root-admitted request")
    verify_attempt_source_receipts(attempt, packet)
    for key in (
        "memory_event_deltas",
        "final_memory_event_deltas",
        "parent_memory_event_deltas",
        "final_parent_memory_event_deltas",
        "parent_hierarchical_memory_event_deltas",
        "final_parent_hierarchical_memory_event_deltas",
    ):
        events = attempt.get(key)
        if not isinstance(events, dict) or any(
            type(events.get(event)) is not int or events.get(event) != 0
            for event in ("oom", "oom_kill", "oom_group_kill")
        ):
            raise ValueError(f"R3 has missing/nonzero OOM evidence in {key}")
    observers = attempt.get("observer_status", {})
    for key in (
        "attempt_memory_events_final",
        "attempt_memory_peak_final",
        "attempt_systemd_runtime",
        "parent_memory_events_final",
        "parent_hierarchical_memory_events_final",
        "nvidia_smi_final",
    ):
        if observers.get(key) != "ok":
            raise ValueError(f"R3 resource observer is incomplete: {key}")
    if (
        type(attempt.get("peak_ram_bytes")) is not int
        or attempt["peak_ram_bytes"] > 9_000_000_000
        or type(attempt.get("peak_observed_vram_bytes")) is not int
        or attempt["peak_observed_vram_bytes"] >= 5_000_000_000
    ):
        raise ValueError("R3 exceeded the admitted RAM/VRAM limits")
    unit = attempt.get("unit")
    if unit != f"wnn-bandit-{packet['resume']['attempt_id']}.service":
        raise ValueError("R3 systemd unit name differs from the identified attempt")
    old_gate = r2_gate.load_old_gate(root, r2_packet)
    unit_receipt = old_gate.verify_unit(unit)
    effective_path = BASE / "ROOT_R3_EFFECTIVE_UNIT_PROPERTIES_R1.json"
    if sha256_file(effective_path) != "57597bea25ff2518fcb0054fd43398b92d934393b89655ad543f6d20286c151d":
        raise ValueError("root startup resource observation changed")
    properties = recorded_resource_properties(attempt, packet, read_json(effective_path))

    r2_queue, _allocation, _original3, seed4 = r2_gate.load_original_inputs(root, r2_packet)
    step = {
        "attempt_id": packet["resume"]["attempt_id"],
        "run_name": packet["resume"]["run_name"],
        "expected_output_path": packet["resume"]["output_path"],
        "maximum_new_bytes": packet["maximum_new_bytes"],
    }
    sys.path.insert(0, str(root / "src"))
    from wnn.splits import load_campaign_split

    split = load_campaign_split(root / packet["fold_manifest_path"], root)
    output_result = old_gate.verify_output(root, step, split, packet["source_pins_sha256"])
    output_dir = root / packet["resume"]["output_path"]
    provenance = read_json(output_dir / "provenance.json")
    resumed = provenance.get("resumed_from", {})
    if (
        resumed.get("sha256") != packet["r2"]["training_state_sha256"]
        or resumed.get("next_epoch") != 3
        or Path(resumed.get("path", "")).resolve()
        != (root / packet["r2"]["training_state_path"]).resolve()
    ):
        raise ValueError("R3 output does not cite the exact R2 epoch-2 boundary state")
    receipt = {
        "schema": "wnn.fold1_gru384_resume_terminal_audit.r3",
        "status": "recovery_fit_and_output_passed",
        "attempt_id": packet["resume"]["attempt_id"],
        "attempt_sha256": sha256_file(attempt_path),
        "authority_sha256": authority_sha,
        "request_sha256": request_sha,
        "r2_attempt_record_sha256": packet["r2"]["attempt_sha256"],
        "r2_training_state_sha256": packet["r2"]["training_state_sha256"],
        "r2_training_state_contract_sha256": request["r2_training_state"]["contract_sha256"],
        "dataset_identity_receipt_sha256": request["dataset_identity_receipt_sha256"],
        "source_pins_sha256": packet["source_pins_sha256"],
        "unit_terminal_state": unit_receipt,
        "unit_resource_properties": properties,
        "unit_resource_evidence": "controller attempt_properties_at_start plus hash-bound root effective startup observation; terminal transient unit may be unloaded",
        "terminal_audit_revision": "r4_transient_unit_receipt_consumer",
        "audit_source_sha256": sha256_file(Path(__file__).resolve()),
        "output_audit": output_result,
        "seed4_original_attempt_id": seed4["attempt_id"],
        "seed4_still_requires_explicit_handoff": True,
    }
    if args.output.exists():
        if read_json(args.output) != receipt:
            raise ValueError(
                "existing terminal audit receipt differs from current verified outputs"
            )
    else:
        write_exclusive_json(args.output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--packet-sha256", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare", help="verify R2 and freeze the exact R3 request"
    )
    prepare_parser.add_argument("--check-only", action="store_true")
    prepare_parser.add_argument("--output", type=Path, default=BASE / "RESUME_REQUEST_R3.json")
    audit_parser = subparsers.add_parser(
        "audit", help="audit successful R3 outputs under root admission"
    )
    audit_parser.add_argument("--request", type=Path, default=BASE / "RESUME_REQUEST_R3.json")
    audit_parser.add_argument("--authority", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, default=BASE / "RESUME_TERMINAL_AUDIT_R3.json")
    audit_parser.add_argument("--authority-sha256", required=True)
    args = parser.parse_args()
    result = prepare(args) if args.action == "prepare" else audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
