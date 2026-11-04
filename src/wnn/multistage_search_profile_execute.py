from __future__ import annotations

import os
import sys
from pathlib import Path

from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import finish_profile_attempt, reserve_profile, utc_now
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import current_process_owner
from wnn.multistage_search_worker_state import (
    _check_cache,
    _resolve_duplicate,
    _set_cache,
    _set_worker_status,
    _tell_fail,
)
from wnn.multistage_search_worker_support import (
    _failed_launch_receipt,
    _nvml_memory_snapshot,
    _process_tuple,
    _rank_log_refs,
    _validate_profile_output,
)
from wnn.multistage_search_worker_types import ProfileJob


def _execute_profile(job: ProfileJob) -> dict[str, object]:
    """Reserve, launch, validate, and durably journal one short exact profile."""
    stage = job.stage
    initializer = job.request.get("initializer_path")
    teacher_cache = job.request.get("teacher_cache_path")
    if isinstance(initializer, str) and file_sha256(Path(initializer)) != job.request.get(
        "initializer_sha256"
    ):
        raise ValueError("profile initializer changed after its successor definition")
    if isinstance(teacher_cache, str) and file_sha256(
        Path(teacher_cache) / "manifest.json"
    ) != job.request.get("teacher_cache_manifest_sha256"):
        raise ValueError("profile teacher cache changed after its successor definition")
    cache_result = _check_cache(job.run_dir, job.proposal_key)
    if cache_result is not None:
        return _resolve_duplicate(
            job.study,
            job.trial,
            job.run_dir,
            job.trial_number,
            cache_result,
            job.proposal_key,
            stage=stage,
        )
    _set_cache(
        job.run_dir,
        job.proposal_key,
        {"status": "profile_reserved", "trial_number": job.trial_number},
    )
    _set_worker_status(
        job.run_dir,
        job.worker_id,
        "running",
        job.trial_number,
        mode="profile",
        gpu_key=job.gpu_key,
    )
    profile_wall = int(job.run_manifest["profile_wall_seconds"])
    reservation = reserve_profile(job.run_dir, job.trial_number, profile_wall, stage=stage)
    append_event(
        job.run_dir / "events.jsonl",
        {
            "event": "profile_started",
            "at_utc": utc_now(),
            "stage": stage,
            "trial_number": job.trial_number,
            "family": job.family,
            "config_sha256": job.config_sha256,
            "profile_attempt_id": reservation["profile_id"],
            "worker_id": job.worker_id,
            "gpu_key": job.gpu_key,
            "status": "running",
        },
    )
    nvml_snapshot = _nvml_memory_snapshot(str(job.device["cuda_visible_devices"]))
    command = [
        os.fspath(sys.executable),
        str(job.repo_root / "scripts" / "train_sequence.py"),
        "--config",
        str(job.config_path),
        "--data-root",
        str(job.request["data_root"]),
        "--split-manifest",
        str(job.auxiliary_path),
        "--output",
        str(job.output_dir),
        "--profile-updates",
        str(job.profile_updates),
    ]
    if isinstance(initializer, str):
        command.extend(["--initializer", initializer])
    if isinstance(teacher_cache, str):
        command.extend(["--teacher-cache", teacher_cache])
    owner = current_process_owner() | {"worker_id": job.worker_id, "mode": "profile"}
    try:
        exit_code, timed_out, worker_receipt = run_owned_command(
            command,
            cwd=job.repo_root,
            env=job.worker_env,
            log_path=job.profile_dir / "profile_worker.log",
            receipt_path=job.worker_receipt_path,
            owner=owner,
            stage=stage,
            trial_number=job.trial_number,
            mode="profile",
            gpu_key=job.gpu_key,
            output_dir=job.output_dir,
            timeout_seconds=profile_wall,
        )
    except (OSError, RuntimeError) as error:
        exit_code, timed_out = None, False
        worker_receipt = _failed_launch_receipt(
            job.worker_receipt_path, owner, command, job.output_dir, error
        )
    profile_status = "profile_complete" if exit_code == 0 and not timed_out else "profile_failed"
    if job.worker_receipt_path.exists():
        worker_receipt = read_json(job.worker_receipt_path)
        worker_receipt["nvml_pre_profile"] = nvml_snapshot
        atomic_write_json(job.worker_receipt_path, worker_receipt)
    attempt_receipt = finish_profile_attempt(
        job.run_dir,
        job.trial_number,
        profile_status,
        exit_code,
        plan_path=job.plan_path,
        worker_receipt_path=job.worker_receipt_path,
        native_profile_path=job.native_profile_path,
        nvml_snapshot=nvml_snapshot,
        stage=stage,
    )
    profile_ok = (
        _validate_profile_output(
            job.native_profile_path,
            job.output_dir,
            job.profile_updates,
            job.request,
            job.source_sha256,
        )
        if profile_status == "profile_complete"
        else False
    )
    if not profile_ok:
        _tell_fail(job.study, job.trial)
        _set_cache(
            job.run_dir,
            job.proposal_key,
            {"status": "profile_failed", "trial_number": job.trial_number},
        )
        append_event(
            job.run_dir / "events.jsonl",
            {
                "event": "trial_failed",
                "at_utc": utc_now(),
                "stage": stage,
                "trial_number": job.trial_number,
                "family": job.family,
                "status": "profile_failed",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "profile_attempt_receipt_sha256": file_sha256(
                    job.profile_dir / "profile_attempt_receipt.json"
                ),
            },
        )
        return {"status": "profile_failed", "trial_number": job.trial_number}
    rank_logs = _rank_log_refs(job.output_dir)
    job.request.update(
        {
            "trial_plan_path": str(job.plan_path.resolve()),
            "trial_plan_sha256": file_sha256(job.plan_path),
            "profile_identity_sha256": canonical_json_sha256(
                read_json(job.native_profile_path)["identity"]
            ),
            "profile_receipt_sha256": file_sha256(job.native_profile_path),
            "profile_attempt_receipt_path": str(
                (job.profile_dir / "profile_attempt_receipt.json").resolve()
            ),
            "profile_attempt_receipt_sha256": file_sha256(
                job.profile_dir / "profile_attempt_receipt.json"
            ),
            "rank_log_refs": rank_logs,
            "worker_receipt_sha256": file_sha256(job.worker_receipt_path),
            "cpu_thread_env": worker_receipt.get("cpu_thread_env"),
            "worker_owned_process": _process_tuple(worker_receipt.get("owner")),
            "worker_owned_process_command_sha256": job.request[
                "worker_owned_process_command_sha256"
            ],
            "worker_owner_receipt_path": job.request["worker_owner_receipt_path"],
            "worker_owner_receipt_sha256": job.request["worker_owner_receipt_sha256"],
            "profile_child_process": _process_tuple(worker_receipt.get("child")),
            "profile_child_process_command_sha256": worker_receipt.get("command_sha256"),
            "deadline_receipt_path": str((job.run_dir / "campaign_deadline.json").resolve()),
            "deadline_receipt_sha256": file_sha256(job.run_dir / "campaign_deadline.json"),
        }
    )
    request_path = job.plan_path.parent / "trial_request.json"
    atomic_write_json(request_path, job.request, replace=False)
    job.trial.set_user_attr(
        f"stage{stage}",
        {"trial_dir": str(job.plan_path.parent), "status": "waiting_profile_review"},
    )
    _set_cache(
        job.run_dir,
        job.proposal_key,
        {
            "status": "profile_pending_review",
            "trial_number": job.trial_number,
            "trial_request_path": str(request_path.resolve()),
            "profile_identity_sha256": job.request["profile_identity_sha256"],
        },
    )
    append_event(
        job.run_dir / "events.jsonl",
        {
            "event": "profile_waiting_review",
            "at_utc": utc_now(),
            "stage": stage,
            "trial_number": job.trial_number,
            "family": job.family,
            "config_sha256": job.config_sha256,
            "profile_attempt_receipt_sha256": job.request["profile_attempt_receipt_sha256"],
            "trial_request_sha256": file_sha256(request_path),
            "status": "waiting_independent_profile_review",
        },
    )
    return {
        "status": "profile_complete_waiting_review",
        "trial_number": job.trial_number,
        "trial_request_path": str(request_path),
        "profile_attempt_receipt": attempt_receipt,
    }
