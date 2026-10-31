from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_receiver import dynamic_reserved_gpu_keys
from wnn.multistage_search import append_event, atomic_write_json, file_lock, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_controller import launch_worker, status_summary
from wnn.multistage_search_study import open_study


def supervise_stage1(
    run_dir: Path,
    *,
    worker_prefix: str = "stage1",
    max_workers: int = 6,
    poll_seconds: float = 5.0,
    max_trials: int | None = None,
) -> dict[str, object]:
    """Keep admitted stage-one profile→fit workers moving until a frozen bound closes."""
    if not worker_prefix or max_workers < 1 or poll_seconds < 0.2:
        raise ValueError("supervisor prefix, concurrency, or poll interval is invalid")
    run_manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    policy = admission.get("profile_policy")
    devices = policy.get("devices") if isinstance(policy, Mapping) else None
    if not isinstance(devices, Mapping) or not devices:
        raise ValueError("supervisor requires an admitted measured device set")
    admitted_gpu_keys = sorted((str(key) for key in devices), key=int)
    concurrency_limit = min(max_workers, len(admitted_gpu_keys))
    stage_caps = run_manifest.get("stage_fit_caps")
    if not isinstance(stage_caps, Mapping) or type(stage_caps.get("1")) is not int:
        raise ValueError("run manifest lacks the frozen stage-one fit cap")
    trial_cap = max_trials
    if trial_cap is None:
        configured = admission.get("stage1", {}).get("max_trials")
        trial_cap = configured if type(configured) is int else int(stage_caps["1"])
    if trial_cap < 1:
        raise ValueError("max_trials must be positive")

    supervisor_path = run_dir / "supervisor.json"
    with file_lock(run_dir / "supervisor.lock", blocking=False):
        if supervisor_path.is_file():
            supervisor = read_json(supervisor_path)
            if supervisor.get("schema") != "wnn.multistage-supervisor.v1":
                raise ValueError("existing supervisor state has an unsupported schema")
            sequence = supervisor.get("next_worker_sequence")
            if type(sequence) is not int or sequence < 1:
                raise ValueError("existing supervisor sequence is malformed")
        else:
            sequence = 1
            supervisor = {
                "schema": "wnn.multistage-supervisor.v1",
                "worker_prefix": worker_prefix,
                "started_at_utc": utc_now(),
                "next_worker_sequence": sequence,
                "max_workers": concurrency_limit,
                "max_trials": trial_cap,
                "poll_seconds": poll_seconds,
                "status": "running",
            }
            atomic_write_json(supervisor_path, supervisor, replace=False)

        deadline = read_json(run_dir / "campaign_deadline.json")
        reserve_epoch = deadline.get("reserve_start_epoch")
        if not isinstance(reserve_epoch, (int, float)):
            raise ValueError("supervisor requires a started campaign clock")
        terminal_reason = ""
        while True:
            reserved = set(dynamic_reserved_gpu_keys(admission))
            gpu_keys = [key for key in admitted_gpu_keys if key not in reserved]
            concurrency = min(max_workers, len(gpu_keys))
            summary = status_summary(run_dir)
            alive = [
                item
                for item in summary["owned_workers"]
                if isinstance(item, Mapping) and item.get("process_alive") is True
            ]
            occupied = {str(item.get("gpu_key")) for item in alive}
            active_stage1 = [
                item
                for item in alive
                if item.get("mode") in {"profile", "fit"}
                and item.get("stage", 1) == 1
                and (item.get("external") is not True or item.get("attempt_id"))
            ]
            budget = read_json(run_dir / "budget.json")
            fit_attempts = budget.get("fit_attempts")
            profile_attempts = budget.get("profile_attempts")
            if not isinstance(fit_attempts, list) or not isinstance(profile_attempts, list):
                raise ValueError("campaign attempt ledgers are malformed")
            stage1_count = sum(
                isinstance(row, Mapping) and row.get("stage") == 1 for row in fit_attempts
            )
            stage1_by_trial = {
                row.get("trial_number")
                for row in fit_attempts
                if isinstance(row, Mapping)
                and row.get("stage") == 1
                and type(row.get("trial_number")) is int
            }
            stage1_ids = {
                row.get("attempt_id")
                for row in fit_attempts
                if isinstance(row, Mapping)
                and row.get("stage") == 1
                and isinstance(row.get("attempt_id"), str)
            }
            active_unreserved = sum(
                item.get("trial_number") not in stage1_by_trial
                and item.get("attempt_id") not in stage1_ids
                for item in active_stage1
            )
            stage1_cap = int(stage_caps["1"])
            study = open_study(run_dir, f"{worker_prefix}-supervisor")
            trials = study.get_trials(deepcopy=False)
            trial_count = len(trials)
            running_trials = sum(
                getattr(getattr(item, "state", None), "name", "") == "RUNNING" for item in trials
            )
            now = time.time()
            if now >= reserve_epoch:
                terminal_reason = "collection_reserve_started"
            elif stage1_count >= stage1_cap:
                terminal_reason = "stage1_fit_cap_reached"
            elif len(profile_attempts) >= int(run_manifest["limits"]["max_profile_attempts"]):
                terminal_reason = "profile_attempt_cap_reached"
            elif trial_count >= trial_cap:
                terminal_reason = "max_trials_reached"

            launched: list[dict[str, object]] = []
            if not terminal_reason:
                fit_slots = max(0, stage1_cap - stage1_count - active_unreserved)
                profile_slots = max(
                    0,
                    int(run_manifest["limits"]["max_profile_attempts"])
                    - len(profile_attempts)
                    - len(active_stage1),
                )
                trial_slots = max(0, trial_cap - trial_count)
                available = min(
                    concurrency - len(active_stage1), fit_slots, profile_slots, trial_slots
                )
                for gpu_key in gpu_keys:
                    if available <= 0:
                        break
                    if gpu_key in occupied:
                        continue
                    worker_id = f"{worker_prefix}-{sequence:06d}"
                    sequence += 1
                    record = launch_worker(run_dir, "profile", worker_id, gpu_key)
                    launched.append(
                        {"worker_id": worker_id, "gpu_key": gpu_key, "pid": record["pid"]}
                    )
                    occupied.add(gpu_key)
                    available -= 1
                    supervisor["next_worker_sequence"] = sequence
                    supervisor["last_launch_at_utc"] = utc_now()
                    atomic_write_json(supervisor_path, supervisor)
                    append_event(
                        run_dir / "events.jsonl",
                        {
                            "event": "supervisor_worker_launched",
                            "worker_id": worker_id,
                            "gpu_key": gpu_key,
                            "mode": "profile",
                            "at_utc": utc_now(),
                        },
                    )
            if launched:
                continue
            if not gpu_keys and not terminal_reason:
                time.sleep(min(poll_seconds, max(0.2, reserve_epoch - now)))
                continue
            if not active_stage1:
                if not terminal_reason and running_trials:
                    terminal_reason = "waiting_for_external_profile_review_or_admission"
                elif not terminal_reason:
                    terminal_reason = "no_admissible_stage1_work"
                break
            time.sleep(min(poll_seconds, max(0.2, reserve_epoch - now)))

        supervisor.update(
            {
                "next_worker_sequence": sequence,
                "status": "stopped_at_bound"
                if terminal_reason != "waiting_for_external_profile_review_or_admission"
                else "waiting_review",
                "terminal_reason": terminal_reason,
                "finished_at_utc": utc_now(),
            }
        )
        atomic_write_json(supervisor_path, supervisor)
        append_event(
            run_dir / "events.jsonl",
            {
                "event": "supervisor_stopped",
                "status": supervisor["status"],
                "reason": terminal_reason,
                "at_utc": utc_now(),
            },
        )
        return {"supervisor": supervisor, "status": status_summary(run_dir)}
