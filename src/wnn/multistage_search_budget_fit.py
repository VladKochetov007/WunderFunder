from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import atomic_write_json, file_lock, read_json
from wnn.multistage_search_budget import utc_now


def reserve_fit(
    run_dir: Path, stage: int, attempt_id: str, trial_number: int, max_wall_seconds: int
) -> tuple[dict[str, object], int]:
    """Reserve one scientific fit while retaining the global and stage budgets."""
    with file_lock(run_dir / "budget.lock"):
        deadline = read_json(run_dir / "campaign_deadline.json")
        reserve_start = deadline.get("reserve_start_epoch")
        if not isinstance(reserve_start, (int, float)):
            raise ValueError("campaign clock has not started")
        now = time.time()
        wall_limit = min(max_wall_seconds, int(reserve_start - now))
        if wall_limit <= 0:
            raise RuntimeError("scientific-fit admission is closed for the collection reserve")
        budget = read_json(run_dir / "budget.json")
        attempts, caps = budget.get("fit_attempts"), budget.get("stage_fit_caps")
        global_cap = budget.get("max_fit_attempts")
        if not isinstance(attempts, list) or not isinstance(caps, Mapping):
            raise ValueError("fit budget ledger is malformed")
        if type(global_cap) is not int or len(attempts) >= global_cap:
            raise RuntimeError("global scientific-fit attempt cap is exhausted")
        if str(stage) not in caps or type(caps[str(stage)]) is not int:
            raise ValueError(f"stage {stage} has no admitted fit cap")
        stage_count = sum(item.get("stage") == stage for item in attempts if isinstance(item, dict))
        if stage_count >= int(caps[str(stage)]):
            raise RuntimeError(f"stage {stage} scientific-fit cap is exhausted")
        if any(item.get("attempt_id") == attempt_id for item in attempts if isinstance(item, dict)):
            raise ValueError("scientific-fit attempt id already exists")
        record = {
            "attempt_id": attempt_id,
            "stage": stage,
            "trial_number": trial_number,
            "status": "reserved",
            "reserved_at_utc": utc_now(),
            "reserved_at_epoch": now,
            "wall_limit_seconds": wall_limit,
        }
        attempts.append(record)
        atomic_write_json(run_dir / "budget.json", budget)
    return record, wall_limit


def finish_budget_attempt(
    run_dir: Path, attempt_id: str, status: str, exit_code: int | None
) -> None:
    """Mark a reserved scientific fit terminal without deleting its attempt."""
    with file_lock(run_dir / "budget.lock"):
        budget = read_json(run_dir / "budget.json")
        attempts = budget.get("fit_attempts")
        if not isinstance(attempts, list):
            raise ValueError("fit budget ledger is malformed")
        matches = [
            item
            for item in attempts
            if isinstance(item, dict) and item.get("attempt_id") == attempt_id
        ]
        if len(matches) != 1:
            raise ValueError("fit budget attempt id is missing or duplicated")
        matches[0].update({"status": status, "exit_code": exit_code, "finished_at_utc": utc_now()})
        atomic_write_json(run_dir / "budget.json", budget)
