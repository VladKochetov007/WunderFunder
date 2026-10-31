from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import append_event, atomic_write_json, file_lock, read_json
from wnn.multistage_search_budget import fail_reserved_profile, utc_now
from wnn.multistage_search_study import StudyHandle, open_study
from wnn.multistage_search_worker_types import AskTrial

_LOGGER = logging.getLogger(__name__)


def _tell_fail(study: StudyHandle, trial: object) -> None:
    import optuna

    trial_number = getattr(trial, "number", None)
    if type(trial_number) is not int or trial_number < 0:
        raise ValueError("failed Optuna trial has an invalid number")
    study.tell(trial_number, state=optuna.trial.TrialState.FAIL)


def _fail_owned_profile_trial(
    run_dir: Path, worker_id: str, error: Exception, *, stage: int = 1
) -> None:
    """Fail only a running profile proposal whose durable worker id matches this process."""
    try:
        study = open_study(run_dir, worker_id, stage=stage)
        matches = []
        for trial in study.get_trials(deepcopy=False):
            state = getattr(getattr(trial, "state", None), "name", None)
            attrs = getattr(trial, "user_attrs", {})
            if (
                state == "RUNNING"
                and isinstance(attrs, Mapping)
                and attrs.get("worker_id") == worker_id
                and isinstance(attrs.get(f"stage{stage}"), Mapping)
            ):
                matches.append(trial)
        if len(matches) != 1:
            return
        trial = matches[0]
        number = int(trial.number)
        _tell_fail(study, trial)
        fail_reserved_profile(run_dir, number, f"{type(error).__name__}: {error}", stage=stage)
        append_event(
            run_dir / "events.jsonl",
            {
                "event": "trial_failed",
                "at_utc": utc_now(),
                "stage": stage,
                "trial_number": number,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        _LOGGER.exception("could not record the failed profile proposal for worker %s", worker_id)
        return


def _tell_pruned(study: StudyHandle, trial: object) -> None:
    import optuna

    study.tell(trial, state=optuna.trial.TrialState.PRUNED)


def _check_cache(run_dir: Path, key: str) -> dict[str, object] | None:
    with file_lock(run_dir / "cache.lock"):
        cache = read_json(run_dir / "cache.json")
        entries = cache.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("trial result cache is malformed")
        value = entries.get(key)
        return dict(value) if isinstance(value, Mapping) else None


def _set_cache(run_dir: Path, key: str, value: Mapping[str, object]) -> None:
    with file_lock(run_dir / "cache.lock"):
        cache = read_json(run_dir / "cache.json")
        entries = cache.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("trial result cache is malformed")
        entries[key] = dict(value)
        atomic_write_json(run_dir / "cache.json", cache)


def _resolve_duplicate(
    study: StudyHandle,
    trial: AskTrial,
    run_dir: Path,
    trial_number: int,
    previous: Mapping[str, object],
    proposal_key: str,
    *,
    stage: int = 1,
) -> dict[str, object]:
    if previous.get("status") == "complete" and isinstance(
        previous.get("objective_value"), (int, float)
    ):
        value = float(previous["objective_value"])
        trial.set_user_attr("cache_hit", previous.get("source_trial_number"))
        study.tell(trial, value)
        event = "trial_cached"
        state = "COMPLETE"
    else:
        _tell_pruned(study, trial)
        event = "trial_duplicate_skipped"
        state = "PRUNED"
    append_event(
        run_dir / "events.jsonl",
        {
            "event": event,
            "at_utc": utc_now(),
            "stage": stage,
            "trial_number": trial_number,
            "proposal_key": proposal_key,
            "source_trial_number": previous.get("source_trial_number"),
            "status": state,
            "reason": previous.get("status"),
        },
    )
    return {"status": state.casefold(), "trial_number": trial_number}


def _set_worker_status(
    run_dir: Path, worker_id: str, status: str, trial_number: int, *, mode: str, gpu_key: str
) -> None:
    path = run_dir / "workers" / f"{worker_id}.json"
    if path.is_file():
        record = read_json(path)
        record.update(
            {
                "status": status,
                "mode": mode,
                "gpu_key": gpu_key,
                "trial_number": trial_number,
                "last_update_utc": utc_now(),
            }
        )
        atomic_write_json(path, record)


def _finish_worker_owner(run_dir: Path, worker_id: str) -> None:
    path = run_dir / "workers" / f"{worker_id}.json"
    if path.is_file():
        record = read_json(path)
        record.update({"status": "worker_exited", "finished_at_utc": utc_now()})
        atomic_write_json(path, record)
