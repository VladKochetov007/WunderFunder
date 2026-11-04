from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import append_event, atomic_write_json, file_lock, file_sha256, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_runtime import process_identity

_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def adopt_external_worker(run_dir: Path, receipt_path: Path) -> dict[str, object]:
    """Register one root-launched fit using its independently captured process identity."""
    receipt_path = receipt_path.resolve(strict=True)
    receipt = read_json(receipt_path)
    required = {
        "schema": "wnn.multistage-external-worker.v1",
        "mode": "fit",
        "stage": 1,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("external worker receipt must describe one stage-one fit")
    worker_id = receipt.get("worker_id")
    if not isinstance(worker_id, str) or not _WORKER_ID.fullmatch(worker_id):
        raise ValueError("external worker id is malformed")
    pid, birth, pgid = receipt.get("pid"), receipt.get("birth"), receipt.get("pgid")
    if type(pid) is not int or not isinstance(birth, str) or type(pgid) is not int:
        raise ValueError("external worker receipt lacks PID/birth/PGID ownership")
    command = receipt.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
    ):
        raise ValueError("external worker receipt lacks the exact argv list")
    command_sha = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if receipt.get("command_sha256") != command_sha:
        raise ValueError("external worker command hash does not match its argv")
    output_dir = receipt.get("output_dir")
    attempt_id = receipt.get("attempt_id")
    config_identity = receipt.get("config_identity_sha256")
    profile_receipt_path = receipt.get("profile_receipt_path")
    profile_receipt_sha = receipt.get("profile_receipt_sha256")
    started = receipt.get("started_at_epoch")
    deadline_epoch = receipt.get("deadline_epoch")
    campaign_deadline_epoch = receipt.get("campaign_deadline_epoch")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("external worker receipt needs its output directory")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("external worker receipt needs a pre-registered fit attempt id")
    if (
        not isinstance(config_identity, str)
        or len(config_identity) != 64
        or not isinstance(profile_receipt_path, str)
        or not isinstance(profile_receipt_sha, str)
        or len(profile_receipt_sha) != 64
    ):
        raise ValueError("external worker receipt must bind its exact profile and config identity")
    if (
        not isinstance(started, (int, float))
        or not isinstance(deadline_epoch, (int, float))
        or not isinstance(campaign_deadline_epoch, (int, float))
    ):
        raise ValueError("external worker receipt needs start, fit deadline, and campaign deadline")
    if started > time.time() or deadline_epoch <= started:
        raise ValueError("external worker receipt has an invalid wall-time interval")
    process_state = receipt.get("status", "running")
    identity = process_identity(pid)
    if process_state == "running":
        if identity != {"pid": pid, "birth": birth, "pgid": pgid}:
            raise ValueError("root-launched process is no longer live with the recorded identity")
    elif process_state in {"completed", "failed", "timed_out"}:
        finished = receipt.get("finished_at_epoch")
        exit_code = receipt.get("exit_code")
        if (
            identity is not None
            or not isinstance(finished, (int, float))
            or finished < started
            or finished > deadline_epoch
            or type(exit_code) is not int
        ):
            raise ValueError("terminal external worker receipt has inconsistent finish evidence")
    else:
        raise ValueError("external worker status must be running or terminal")

    run_manifest = read_json(run_dir / "run.json")
    deadline = read_json(run_dir / "campaign_deadline.json")
    reserve_start = deadline.get("reserve_start_epoch")
    max_fit_wall = run_manifest.get("limits", {}).get("max_fit_wall_seconds")
    if (
        campaign_deadline_epoch != deadline.get("deadline_epoch")
        or not isinstance(reserve_start, (int, float))
        or type(max_fit_wall) is not int
        or started >= reserve_start
        or deadline_epoch > reserve_start
        or deadline_epoch - started > max_fit_wall
    ):
        raise ValueError("external fit deadline exceeds the frozen campaign or fit bounds")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    devices = admission["profile_policy"]["devices"]
    gpu_key = str(receipt.get("gpu_key"))
    if gpu_key not in {str(key) for key in devices}:
        raise ValueError("external worker uses a GPU outside the admitted device set")
    if gpu_key in {str(value) for value in admission.get("reserved_gpu_keys", [])}:
        raise ValueError("external worker uses a slot reserved by the campaign policy")
    attempts = read_json(run_dir / "budget.json").get("fit_attempts")
    if not isinstance(attempts, list):
        raise ValueError("fit attempt ledger is malformed")
    matching = [
        row for row in attempts if isinstance(row, Mapping) and row.get("attempt_id") == attempt_id
    ]
    if len(matching) != 1 or matching[0].get("stage") != 1:
        raise ValueError("external fit must match one pre-registered stage-one attempt")
    expected_profile = matching[0].get("profile_receipt")
    expected_output = matching[0].get("output_dir")
    if (
        matching[0].get("config_identity_sha256") != config_identity
        or not isinstance(expected_profile, Mapping)
        or expected_profile.get("path") != profile_receipt_path
        or expected_profile.get("sha256") != profile_receipt_sha
        or expected_output != output_dir
    ):
        raise ValueError("external worker does not match the registered config/profile/output")
    profile_path = Path(profile_receipt_path).resolve(strict=True)
    profile = read_json(profile_path)
    profile_identity = profile.get("identity")
    profile_data = profile_identity.get("data") if isinstance(profile_identity, Mapping) else None
    training_source_sha = matching[0].get("training_source_sha256")
    if (
        file_sha256(profile_path) != profile_receipt_sha
        or profile.get("approved") is not True
        or not isinstance(profile_identity, Mapping)
        or profile_identity.get("config_sha256") != config_identity
        or profile_identity.get("source_sha256") != training_source_sha
        or not isinstance(profile_data, Mapping)
        or profile_data.get("split_sha256")
        != run_manifest.get("auxiliary_manifest", {}).get("sha256")
    ):
        raise ValueError("registered approved profile bytes do not match the exact fit identity")
    worker_dir = run_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    owner_path = worker_dir / f"{worker_id}.json"
    preserved_path = worker_dir / f"{worker_id}.external.json"
    with file_lock(run_dir / "launch.lock"):
        if owner_path.exists() or preserved_path.exists():
            raise FileExistsError(f"worker id already exists: {worker_id}")
        for path in worker_dir.glob("*.json"):
            if path.name.endswith(".launch_receipt.json") or path.name.endswith(".external.json"):
                continue
            prior = read_json(path)
            if str(prior.get("gpu_key")) != gpu_key or type(prior.get("pid")) is not int:
                continue
            prior_identity = process_identity(int(prior["pid"]))
            if prior_identity is not None and prior_identity.get("birth") == prior.get("birth"):
                raise RuntimeError(f"GPU {gpu_key} already has an owned live worker")
        preserved_path.write_bytes(receipt_path.read_bytes())
        owner = {
            "schema": "wnn.multistage-worker-owner.v1",
            "worker_id": worker_id,
            "mode": "fit",
            "stage": 1,
            "gpu_key": gpu_key,
            "pid": pid,
            "birth": birth,
            "pgid": pgid,
            "command": command,
            "command_sha256": command_sha,
            "started_at_epoch": started,
            "deadline_epoch": deadline_epoch,
            "output_dir": str(Path(output_dir).resolve()),
            "attempt_id": attempt_id,
            "config_identity_sha256": config_identity,
            "profile_receipt_path": profile_receipt_path,
            "profile_receipt_sha256": profile_receipt_sha,
            "trial_number": receipt.get("trial_number"),
            "status": "running" if process_state == "running" else "worker_exited",
            "external_exit_code": receipt.get("exit_code"),
            "external_finished_at_epoch": receipt.get("finished_at_epoch"),
            "external": True,
            "external_receipt_path": str(preserved_path.resolve()),
            "external_receipt_sha256": file_sha256(preserved_path),
            "adopted_at_utc": utc_now(),
        }
        atomic_write_json(owner_path, owner, replace=False)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "external_worker_adopted",
            "worker_id": worker_id,
            "attempt_id": attempt_id,
            "gpu_key": gpu_key,
            "pid": pid,
            "birth": birth,
            "pgid": pgid,
            "external_receipt_sha256": owner["external_receipt_sha256"],
            "at_utc": utc_now(),
        },
    )
    return owner
