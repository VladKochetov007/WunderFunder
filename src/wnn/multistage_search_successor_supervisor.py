from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import append_event, atomic_write_json, file_lock, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_controller import launch_worker, status_summary
from wnn.multistage_search_study import open_study
from wnn.multistage_search_successor_approval import approve_successor_profile


def supervise_successors(
    run_dir: Path,
    *,
    stage: int,
    worker_prefix: str,
    max_workers: int = 6,
    poll_seconds: float = 5.0,
    max_runtime_seconds: int = 21600,
) -> dict[str, object]:
    """Roll prepared Stage2/3 profiles into externally reviewed fits under the campaign budget."""
    if stage not in {2, 3}:
        raise ValueError("successor supervisor supports stage2 and stage3")
    if not worker_prefix or max_workers < 1 or poll_seconds < 0.2 or max_runtime_seconds < 1:
        raise ValueError("successor supervisor limits are invalid")
    run_manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    policy = admission.get("profile_policy")
    devices = policy.get("devices") if isinstance(policy, Mapping) else None
    if not isinstance(devices, Mapping) or not devices:
        raise ValueError("successor supervisor requires measured admitted devices")
    reserved = admission.get("reserved_gpu_keys", [])
    if not isinstance(reserved, list):
        raise ValueError("reserved GPU keys are malformed")
    gpu_keys = [str(key) for key in devices if str(key) not in reserved]
    concurrency = min(max_workers, len(gpu_keys))
    if concurrency < 1:
        raise ValueError("no admitted GPU slots are available to successor workers")
    caps = run_manifest.get("stage_fit_caps")
    if not isinstance(caps, Mapping) or type(caps.get(str(stage))) is not int:
        raise ValueError("run manifest lacks this stage's frozen fit cap")
    supervisor_path = run_dir / f"stage{stage}" / "supervisor.json"
    start = time.time()
    with file_lock(run_dir / f"stage{stage}" / "supervisor.lock", blocking=False):
        supervisor = {
            "schema": "wnn.multistage-successor-supervisor.v1",
            "stage": stage,
            "worker_prefix": worker_prefix,
            "max_workers": concurrency,
            "poll_seconds": poll_seconds,
            "started_at_utc": utc_now(),
            "status": "running",
            "next_worker_sequence": 1,
        }
        if supervisor_path.exists():
            previous = read_json(supervisor_path)
            if previous.get("schema") != supervisor["schema"]:
                raise ValueError("successor supervisor state has an unsupported schema")
            supervisor["next_worker_sequence"] = int(previous.get("next_worker_sequence", 1))
        atomic_write_json(supervisor_path, supervisor)
        deadline = read_json(run_dir / "campaign_deadline.json")
        reserve_epoch = deadline.get("reserve_start_epoch")
        if not isinstance(reserve_epoch, (int, float)):
            raise ValueError("successor supervisor requires the campaign deadline")
        terminal_reason = ""
        while not terminal_reason:
            now = time.time()
            if now >= reserve_epoch:
                terminal_reason = "collection_reserve_started"
                break
            if now - start >= max_runtime_seconds:
                terminal_reason = "supervisor_runtime_cap_reached"
                break
            summary = status_summary(run_dir)
            alive = [
                item
                for item in summary["owned_workers"]
                if isinstance(item, Mapping) and item.get("process_alive") is True
            ]
            occupied = {str(item.get("gpu_key")) for item in alive}
            stage_workers = [
                item
                for item in alive
                if item.get("stage") == stage and item.get("mode") in {"profile", "fit"}
            ]
            budget = read_json(run_dir / "budget.json")
            fits = budget.get("fit_attempts")
            profiles = budget.get("profile_attempts")
            if not isinstance(fits, list) or not isinstance(profiles, list):
                raise ValueError("successor supervisor attempt ledger is malformed")
            stage_fits = [
                row for row in fits if isinstance(row, Mapping) and row.get("stage") == stage
            ]
            fit_admission_closed = len(stage_fits) >= int(caps[str(stage)])
            profile_admission_closed = len(profiles) >= int(
                run_manifest["limits"]["max_profile_attempts"]
            )
            study = open_study(run_dir, worker_prefix, stage=stage)
            trial_rows = study.get_trials(deepcopy=False)
            running = [
                row
                for row in trial_rows
                if getattr(getattr(row, "state", None), "name", "") == "RUNNING"
                and isinstance(getattr(row, "user_attrs", {}), Mapping)
                and isinstance(row.user_attrs.get(f"stage{stage}"), Mapping)
            ]
            if not running and not stage_workers:
                terminal_reason = "all_prepared_trials_terminal"
                break
            active_numbers = {
                item.get("trial_number")
                for item in stage_workers
                if type(item.get("trial_number")) is int
            }
            fit_attempted = {
                row.get("trial_number")
                for row in stage_fits
                if type(row.get("trial_number")) is int
            }
            profile_attempted = {
                row.get("trial_number")
                for row in profiles
                if isinstance(row, Mapping)
                and row.get("stage") == stage
                and type(row.get("trial_number")) is int
            }
            fit_ready: list[int] = []
            profile_ready: list[int] = []
            waiting_reviews: list[int] = []
            for trial in running:
                number = int(trial.number)
                if number in active_numbers or number in fit_attempted:
                    continue
                trial_dir = run_dir / f"stage{stage}" / "trials" / f"{number:06d}"
                approval = trial_dir / "profile" / "approval_record.json"
                review = trial_dir / "profile" / "successor_profile_review_receipt.json"
                request = trial_dir / "trial_request.json"
                if approval.is_file():
                    fit_ready.append(number)
                elif request.is_file():
                    if review.is_file():
                        try:
                            approve_successor_profile(run_dir, stage, number, review)
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            append_event(
                                run_dir / "events.jsonl",
                                {
                                    "event": "successor_profile_review_rejected",
                                    "at_utc": utc_now(),
                                    "stage": stage,
                                    "trial_number": number,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                },
                            )
                            waiting_reviews.append(number)
                        else:
                            fit_ready.append(number)
                    else:
                        waiting_reviews.append(number)
                elif number not in profile_attempted:
                    profile_ready.append(number)
            launched: list[dict[str, object]] = []
            next_sequence = int(supervisor["next_worker_sequence"])
            remaining_fit = int(caps[str(stage)]) - len(stage_fits)
            remaining_profile = int(run_manifest["limits"]["max_profile_attempts"]) - len(profiles)
            for task_mode, candidates, capacity, closed in (
                ("fit", sorted(fit_ready), remaining_fit, fit_admission_closed),
                ("profile", sorted(profile_ready), remaining_profile, profile_admission_closed),
            ):
                if closed or capacity <= 0:
                    continue
                for trial_number in candidates:
                    if (
                        len(stage_workers) + len(launched) >= concurrency
                        or len(launched) >= capacity
                    ):
                        break
                    gpu_key = next((key for key in gpu_keys if key not in occupied), None)
                    if gpu_key is None:
                        break
                    worker_id = f"{worker_prefix}-{next_sequence:06d}"
                    next_sequence += 1
                    record = launch_worker(
                        run_dir,
                        task_mode,
                        worker_id,
                        gpu_key,
                        trial_number,
                        stage=stage,
                    )
                    launched.append(
                        {
                            "worker_id": worker_id,
                            "mode": task_mode,
                            "trial_number": trial_number,
                            "gpu_key": gpu_key,
                            "pid": record["pid"],
                        }
                    )
                    occupied.add(gpu_key)
                    supervisor["next_worker_sequence"] = next_sequence
                    supervisor["last_launch_at_utc"] = utc_now()
                    atomic_write_json(supervisor_path, supervisor)
                    append_event(
                        run_dir / "events.jsonl",
                        {
                            "event": "successor_supervisor_worker_launched",
                            "stage": stage,
                            "trial_number": trial_number,
                            "worker_id": worker_id,
                            "gpu_key": gpu_key,
                            "mode": task_mode,
                            "at_utc": utc_now(),
                        },
                    )
            if launched:
                continue
            if fit_admission_closed and profile_admission_closed and not stage_workers:
                terminal_reason = "stage_and_profile_caps_reached"
                break
            if fit_admission_closed and not stage_workers:
                terminal_reason = "stage_fit_cap_reached"
                break
            if profile_admission_closed and not stage_workers and not waiting_reviews:
                terminal_reason = "profile_attempt_cap_reached"
                break
            if not stage_workers and not profile_ready and not fit_ready and not waiting_reviews:
                terminal_reason = "no_admissible_successor_work"
                break
            time.sleep(min(poll_seconds, max(0.2, reserve_epoch - now)))
        supervisor.update(
            {
                "status": "waiting_review"
                if terminal_reason == "no_admissible_successor_work"
                else "stopped_at_bound",
                "terminal_reason": terminal_reason,
                "finished_at_utc": utc_now(),
            }
        )
        atomic_write_json(supervisor_path, supervisor)
        append_event(
            run_dir / "events.jsonl",
            {
                "event": "successor_supervisor_stopped",
                "at_utc": utc_now(),
                "stage": stage,
                "status": supervisor["status"],
                "reason": terminal_reason,
            },
        )
        return {"supervisor": supervisor, "status": status_summary(run_dir)}
