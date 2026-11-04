from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from wnn.multistage_search import (
    atomic_write_json,
    file_lock,
    file_sha256,
    read_json,
)


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp for durable controller events."""
    return datetime.now(UTC).isoformat()


def initialize_budget(
    run_dir: Path,
    *,
    max_fit_attempts: int,
    max_profile_attempts: int,
    stage_fit_caps: Mapping[str, object],
    prior_fit_attempts: Sequence[Mapping[str, object]],
    prior_profile_attempts: int,
    first_profile_epoch: float | None,
    campaign_wall_seconds: int,
    collection_reserve_seconds: int,
) -> None:
    """Create the shared attempt ledger and campaign deadline once."""
    if prior_profile_attempts < 0 or prior_profile_attempts > max_profile_attempts:
        raise ValueError("prior profile attempts exceed the admitted profile cap")
    seen_ids: set[str] = set()
    fit_rows: list[dict[str, object]] = []
    for item in prior_fit_attempts:
        attempt_id, stage = item.get("attempt_id"), item.get("stage")
        if not isinstance(attempt_id, str) or not attempt_id or type(stage) is not int:
            raise ValueError("preexisting fit attempts must bind an id and stage")
        if attempt_id in seen_ids or str(stage) not in stage_fit_caps:
            raise ValueError("preexisting fit attempt id or stage is invalid")
        seen_ids.add(attempt_id)
        fit_rows.append(dict(item) | {"status": item.get("status", "preexisting")})
    if len(fit_rows) > max_fit_attempts:
        raise ValueError("preexisting attempts exceed the global scientific-fit cap")
    budget = {
        "schema": "wnn.multistage-budget.v1",
        "max_fit_attempts": max_fit_attempts,
        "max_profile_attempts": max_profile_attempts,
        "stage_fit_caps": dict(stage_fit_caps),
        "fit_attempts": fit_rows,
        "profile_attempts": [
            {"profile_id": f"preexisting-{index + 1:03d}", "status": "preexisting"}
            for index in range(prior_profile_attempts)
        ],
    }
    atomic_write_json(run_dir / "budget.json", budget, replace=False)
    deadline = {
        "schema": "wnn.multistage-deadline.v1",
        "first_profile_epoch": first_profile_epoch,
        "campaign_wall_seconds": campaign_wall_seconds,
        "collection_reserve_seconds": collection_reserve_seconds,
        "deadline_epoch": (
            None if first_profile_epoch is None else first_profile_epoch + campaign_wall_seconds
        ),
        "reserve_start_epoch": (
            None
            if first_profile_epoch is None
            else first_profile_epoch + campaign_wall_seconds - collection_reserve_seconds
        ),
    }
    atomic_write_json(run_dir / "campaign_deadline.json", deadline, replace=False)


def ensure_campaign_clock(run_dir: Path) -> dict[str, object]:
    """Start the six-hour clock on the first real profile if root has not pinned it."""
    path = run_dir / "campaign_deadline.json"
    with file_lock(run_dir / "budget.lock"):
        deadline = read_json(path)
        if deadline.get("first_profile_epoch") is None:
            start = time.time()
            wall, reserve = (
                deadline.get("campaign_wall_seconds"),
                deadline.get("collection_reserve_seconds"),
            )
            if type(wall) is not int or type(reserve) is not int:
                raise ValueError("campaign deadline is missing fixed wall limits")
            deadline.update(
                {
                    "first_profile_epoch": start,
                    "first_profile_utc": utc_now(),
                    "deadline_epoch": start + wall,
                    "reserve_start_epoch": start + wall - reserve,
                }
            )
            atomic_write_json(path, deadline)
        return deadline


def reserve_profile(
    run_dir: Path, trial_number: int, max_wall_seconds: int, *, stage: int = 1
) -> dict[str, object]:
    """Reserve one fresh profile inside the total profile and campaign limits."""
    if stage not in {1, 2, 3}:
        raise ValueError("profile attempts are supported for fitted stages one through three")
    ensure_campaign_clock(run_dir)
    with file_lock(run_dir / "budget.lock"):
        deadline = read_json(run_dir / "campaign_deadline.json")
        now = time.time()
        reserve_start = deadline.get("reserve_start_epoch")
        if not isinstance(reserve_start, (int, float)) or now + max_wall_seconds > reserve_start:
            raise RuntimeError("profile would consume the final collection reserve")
        budget = read_json(run_dir / "budget.json")
        attempts = budget.get("profile_attempts")
        cap = budget.get("max_profile_attempts")
        if not isinstance(attempts, list) or type(cap) is not int:
            raise ValueError("profile budget ledger is malformed")
        if len(attempts) >= cap:
            raise RuntimeError("total exact-profile attempt cap is exhausted")
        profile_id = f"stage{stage}-trial-{trial_number:06d}"
        if any(
            isinstance(item, dict) and item.get("profile_id") == profile_id for item in attempts
        ):
            raise ValueError("profile attempt id already exists")
        record = {
            "profile_id": profile_id,
            "stage": stage,
            "trial_number": trial_number,
            "status": "reserved",
            "reserved_at_utc": utc_now(),
            "reserved_at_epoch": now,
            "wall_limit_seconds": max_wall_seconds,
            "profile_attempt_count_at_reservation": len(attempts) + 1,
            "fit_attempt_count_at_reservation": len(budget.get("fit_attempts", [])),
            "max_profile_attempts": cap,
            "max_fit_attempts": budget.get("max_fit_attempts"),
        }
        attempts.append(record)
        atomic_write_json(run_dir / "budget.json", budget)
    return record


def finish_profile_attempt(
    run_dir: Path,
    trial_number: int,
    status: str,
    exit_code: int | None,
    *,
    plan_path: Path,
    worker_receipt_path: Path,
    native_profile_path: Path,
    nvml_snapshot: Mapping[str, object],
    stage: int = 1,
) -> dict[str, object]:
    """Mark a reserved exact profile terminal while retaining its consumed cap."""
    with file_lock(run_dir / "budget.lock"):
        budget = read_json(run_dir / "budget.json")
        attempts = budget.get("profile_attempts")
        if not isinstance(attempts, list):
            raise ValueError("profile budget ledger is malformed")
        matches = [
            item
            for item in attempts
            if isinstance(item, dict)
            and item.get("profile_id") == f"stage{stage}-trial-{trial_number:06d}"
        ]
        if len(matches) != 1:
            raise ValueError("profile budget attempt id is missing or duplicated")
        matches[0].update({"status": status, "exit_code": exit_code, "finished_at_utc": utc_now()})
        atomic_write_json(run_dir / "budget.json", budget)
        profile_dir = plan_path.parent / "profile"
        budget_snapshot_path = profile_dir / "budget_snapshot.json"
        _copy_file_once(run_dir / "budget.json", budget_snapshot_path)
        worker_receipt = read_json(worker_receipt_path)
        native_sha = file_sha256(native_profile_path) if native_profile_path.is_file() else None
        deadline_path = run_dir / "campaign_deadline.json"
        attempt_receipt = {
            "schema": "wnn.multistage-profile-attempt.v1",
            "trial_number": trial_number,
            "stage": stage,
            "profile_id": matches[0]["profile_id"],
            "terminal_attempt": dict(matches[0]),
            "status": status,
            "exit_code": exit_code,
            "actual_wall_seconds": (
                float(worker_receipt["finished_at_epoch"])
                - float(worker_receipt["started_at_epoch"])
                if isinstance(worker_receipt.get("finished_at_epoch"), (int, float))
                and isinstance(worker_receipt.get("started_at_epoch"), (int, float))
                else None
            ),
            "profile_attempt_count_at_completion": len(budget["profile_attempts"]),
            "fit_attempt_count_at_completion": len(budget["fit_attempts"]),
            "max_profile_attempts": budget["max_profile_attempts"],
            "max_fit_attempts": budget["max_fit_attempts"],
            "stage_fit_caps": budget["stage_fit_caps"],
            "campaign_deadline_path": str(deadline_path),
            "campaign_deadline_sha256": file_sha256(deadline_path),
            "budget_snapshot_path": str(budget_snapshot_path),
            "budget_snapshot_sha256": file_sha256(budget_snapshot_path),
            "trial_plan_path": str(plan_path),
            "trial_plan_sha256": file_sha256(plan_path),
            "worker_receipt_path": str(worker_receipt_path),
            "worker_receipt_sha256": file_sha256(worker_receipt_path),
            "native_profile_path": str(native_profile_path),
            "native_profile_sha256": native_sha,
            "nvml_pre_profile": dict(nvml_snapshot),
            "completed_at_utc": utc_now(),
        }
        receipt_path = profile_dir / "profile_attempt_receipt.json"
        atomic_write_json(receipt_path, attempt_receipt, replace=False)
        return attempt_receipt


def fail_reserved_profile(
    run_dir: Path, trial_number: int, error: str, *, stage: int = 1
) -> None:
    """Mark a consumed profile reservation failed if setup aborts before a native receipt."""
    with file_lock(run_dir / "budget.lock"):
        budget = read_json(run_dir / "budget.json")
        attempts = budget.get("profile_attempts")
        if not isinstance(attempts, list):
            raise ValueError("profile budget ledger is malformed")
        matches = [
            item
            for item in attempts
            if isinstance(item, dict)
            and item.get("profile_id") == f"stage{stage}-trial-{trial_number:06d}"
        ]
        if not matches:
            return
        if len(matches) != 1:
            raise ValueError("profile attempt id is duplicated")
        if matches[0].get("status") == "reserved":
            matches[0].update(
                {"status": "profile_failed", "error": error, "finished_at_utc": utc_now()}
            )
            atomic_write_json(run_dir / "budget.json", budget)


def _copy_file_once(source: Path, destination: Path) -> None:
    """Persist an exact immutable byte snapshot with a same-directory atomic link."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"budget snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temp_name)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            while block := incoming.read(1 << 20):
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
