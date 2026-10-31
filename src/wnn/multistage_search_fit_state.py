from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    file_lock,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_study import StudyHandle


def _approved_trial(
    study: StudyHandle, run_dir: Path, requested_number: int | None, *, stage: int = 1
) -> tuple[int, Path, dict[str, object], dict[str, object]]:
    """Find one unique running Optuna trial with an exact profile approval record."""
    trials = study.get_trials(deepcopy=False)
    for trial in trials:
        number = getattr(trial, "number", None)
        if type(number) is not int or (requested_number is not None and number != requested_number):
            continue
        state = getattr(getattr(trial, "state", None), "name", None)
        attrs = getattr(trial, "user_attrs", {})
        stage_attrs = attrs.get(f"stage{stage}") if isinstance(attrs, Mapping) else None
        if (
            state != "RUNNING"
            or not isinstance(stage_attrs, Mapping)
            or not isinstance(stage_attrs.get("trial_dir"), str)
        ):
            continue
        trial_dir = Path(stage_attrs["trial_dir"])
        approval_path = trial_dir / "profile" / "approval_record.json"
        request_path = trial_dir / "trial_request.json"
        if approval_path.is_file() and request_path.is_file():
            return number, trial_dir, read_json(request_path), read_json(approval_path)
    raise RuntimeError("no exact-profile trial is approved and waiting for a fit")


def _complete_trial(
    study: StudyHandle,
    run_dir: Path,
    trial_number: int,
    value: float,
    result: Mapping[str, object],
    *,
    stage: int = 1,
) -> None:
    """Tell Optuna only after all source-qualified outputs and the inner score are durable."""
    study.tell(trial_number, value)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "trial_finished",
            "at_utc": utc_now(),
            "stage": stage,
            "trial_number": trial_number,
            "family": result["family"],
            "objective_value": value,
            "duration_seconds": result["duration_seconds"],
            "status": "COMPLETE",
            "result_path": result["fit_output"],
            "config_sha256": result["config_sha256"],
            "selected_checkpoint_sha256": result["selected_endpoint"]["checkpoint_sha256"],
        },
    )


def _fail_trial(
    study: StudyHandle,
    run_dir: Path,
    trial_number: int,
    error: Exception,
    *,
    stage: int = 1,
) -> None:
    import optuna

    study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "trial_failed",
            "at_utc": utc_now(),
            "stage": stage,
            "trial_number": trial_number,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def _cache_complete(
    run_dir: Path,
    key: str,
    result: Mapping[str, object],
    result_path: Path,
    source_trial_number: int,
) -> None:
    """Keep one actual-identity-bound completed fit per exact stage1 recipe key."""

    with file_lock(run_dir / "cache.lock"):
        cache = read_json(run_dir / "cache.json")
        entries = cache.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("trial cache is malformed")
        entries[key] = {
            "status": "complete",
            "source_trial_number": source_trial_number,
            "objective_value": result["objective"]["objective"]["value"],
            "profile_identity_sha256": result["profile_identity_sha256"],
            "result_path": str(result_path.resolve()),
            "result_sha256": file_sha256(result_path),
        }
        atomic_write_json(run_dir / "cache.json", cache)
