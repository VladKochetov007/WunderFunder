"""Profile one selected final-refit config and preserve independent review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from primary_refit_actor_common import (
    POLICY_NAME,
    REVIEWER_NAME,
    _admission,
    _campaign_inputs,
    _nvml,
    _packet,
    _request,
    _runtime_paths,
    _source_and_config,
)
from primary_refit_actor_support import (
    finish_profile_attempt,
    load_device_inventory,
    profile_environment,
    read_json,
    reserve_profile_attempt,
    resolve_device,
    sha256,
    write_new,
)


def run_profile(args: argparse.Namespace) -> dict[str, object]:
    operator_root = args.operator_root.resolve(strict=True)
    packet = _packet(operator_root)
    prepared_root, runtime_root, run_dir = (
        path.resolve(strict=True) for path in (args.prepared_root, args.runtime_root, args.run_dir)
    )
    selection_path, _, request, request_sha = _request(prepared_root, args.fit_id)
    _, deadline = _campaign_inputs(run_dir, runtime_root)
    admission = _admission(runtime_root, run_dir)
    inventory_path, inventory = load_device_inventory(
        runtime_root, admission, args.device_inventory
    )
    source_root, _, command = _source_and_config(request)
    device = resolve_device(admission, inventory, args.gpu_key)
    nvml = _nvml(device)
    profile_plan = request["profile"]
    output = Path(str(profile_plan["output"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError("prepared profile output already exists; preserve prior evidence")
    profile_id = f"stage4-primary-refit-{args.fit_id}"
    actor_paths = _runtime_paths(runtime_root, args.fit_id)
    profile_record = reserve_profile_attempt(
        run_dir,
        args.fit_id,
        int(request["selected_by"]["trial_number"]),
        args.gpu_key,
        device,
        deadline,
    )
    env = profile_environment(request, device)
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import current_process_owner

    child_receipt = actor_paths["receipts"] / f"{profile_id}.native-child.json"
    worker_path = actor_paths["receipts"] / f"{profile_id}.worker.json"
    log_path = actor_paths["logs"] / f"{profile_id}.log"
    owner = current_process_owner()
    try:
        code, timed_out, raw = run_owned_command(
            command,
            cwd=source_root,
            env=env,
            log_path=log_path,
            receipt_path=child_receipt,
            owner=owner,
            stage=4,
            trial_number=int(request["selected_by"]["trial_number"]),
            mode="primary_refit_profile",
            gpu_key=args.gpu_key,
            output_dir=output,
            timeout_seconds=180,
        )
    except Exception:
        finish_profile_attempt(run_dir, profile_id, "launch_failed", 127)
        raise
    status = "censored_timeout" if timed_out else "completed" if code == 0 else "failed"
    finish_profile_attempt(run_dir, profile_id, status, int(code or 0))
    native_path = output / "profile_receipt.json"
    if status != "completed" or not native_path.is_file():
        return {
            "status": status,
            "fit_id": args.fit_id,
            "exit_code": code,
            "profile_id": profile_id,
        }

    env_receipt = {
        key: env[key]
        for key in set(profile_plan["environment_template"]) | {"CUBLAS_EMULATE_SINGLE_PRECISION"}
        if key in env
    }
    worker = {
        "schema": "wnn.primary-refit-profile-worker.v1",
        "fit_id": args.fit_id,
        "request_sha256": request_sha,
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_root": str(source_root),
        "owner_process": owner,
        "child_process": raw.get("child"),
        "child_environment": env_receipt,
        "device": device,
        "nvml_pre_profile": nvml,
        "started_at_epoch": raw.get("started_at_epoch"),
        "finished_at_epoch": raw.get("finished_at_epoch"),
        "status": "completed",
        "exit_code": 0,
    }
    write_new(worker_path, worker)
    native = read_json(native_path, "native profile receipt")
    if native.get("approved") is not False or native.get("status") != "profile_complete":
        raise ValueError("trainer did not produce its unapproved complete native profile receipt")

    def snapshot(source: Path, destination: Path) -> str:
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        return sha256(destination)

    budget_snapshot = output / "budget_snapshot.json"
    deadline_snapshot = output / "deadline_receipt.json"
    inventory_snapshot = output / "device_inventory.json"
    budget_sha = snapshot(run_dir / "budget.json", budget_snapshot)
    deadline_sha = snapshot(run_dir / "campaign_deadline.json", deadline_snapshot)
    inventory_sha = snapshot(inventory_path, inventory_snapshot)
    budget = read_json(budget_snapshot, "profile budget snapshot")
    attempt = {
        "schema": "wnn.primary-refit-profile-attempt.v1",
        "fit_id": args.fit_id,
        "request_sha256": request_sha,
        "profile_path": str(native_path.resolve()),
        "profile_sha256": sha256(native_path),
        "worker_receipt_path": str(worker_path.resolve()),
        "worker_receipt_sha256": sha256(worker_path),
        "budget_snapshot_path": str(budget_snapshot.resolve()),
        "budget_snapshot_sha256": budget_sha,
        "deadline_receipt_path": str(deadline_snapshot.resolve()),
        "deadline_receipt_sha256": deadline_sha,
        "device_inventory_path": str(inventory_snapshot.resolve()),
        "device_inventory_sha256": inventory_sha,
        "rank0_log_path": str((output / "rank0.jsonl").resolve()),
        "rank0_log_sha256": sha256(output / "rank0.jsonl"),
        "owner_process": owner,
        "child_process": raw.get("child"),
        "device": device,
        "nvml_pre_profile": nvml,
        "profile_attempt_count_at_completion": len(budget["profile_attempts"]),
        "fit_attempt_count_at_completion": len(budget["fit_attempts"]),
        "actual_wall_seconds": float(raw["finished_at_epoch"]) - float(raw["started_at_epoch"]),
        "started_at_epoch": raw["started_at_epoch"],
        "finished_at_epoch": raw["finished_at_epoch"],
        "status": "completed",
        "exit_code": 0,
    }
    attempt_path = output / "profile_attempt_receipt.json"
    write_new(attempt_path, attempt)
    review_path = output / "engineering_review.json"
    reviewer = packet / REVIEWER_NAME
    review_command = [
        command[0],
        str(reviewer),
        "--request",
        str((prepared_root / "requests" / f"{args.fit_id}.json").resolve()),
        "--selection",
        str(selection_path.resolve()),
        "--profile",
        str(native_path.resolve()),
        "--worker",
        str(worker_path.resolve()),
        "--attempt",
        str(attempt_path.resolve()),
        "--budget",
        str(budget_snapshot.resolve()),
        "--deadline",
        str(deadline_snapshot.resolve()),
        "--device-inventory",
        str(inventory_snapshot.resolve()),
        "--receipt",
        str(review_path.resolve()),
        "--policy",
        str((packet / POLICY_NAME).resolve()),
    ]
    reviewed = subprocess.run(
        review_command, cwd=source_root, env=env, capture_output=True, text=True, timeout=180
    )
    if reviewed.returncode != 0:
        raise RuntimeError(f"engineering review failed; preserved receipt at {review_path}")
    review = read_json(review_path, "independent engineering review")
    if review.get("status") != "engineering_checks_passed" or review.get("approved") is not False:
        raise ValueError("reviewer did not pass engineering checks without approval")
    return {
        "status": "engineering_checks_passed_pending_root_approval",
        "fit_id": args.fit_id,
        "review_path": str(review_path),
        "review_sha256": sha256(review_path),
        "profile_sha256": sha256(native_path),
        "device": device,
        "reservation": profile_record,
    }
