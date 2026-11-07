from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from wnn.multistage_search import file_sha256, read_json

STAGE_CAPS = {"1": 16, "2": 8, "3": 12, "4": 12}


def validate_frozen_stage2_limits(
    run_dir: Path,
    run_manifest: Mapping[str, object],
    admission: Mapping[str, object],
) -> None:
    """Confirm the proposal-only amendment leaves every campaign budget intact."""
    budget = read_json(run_dir / "budget.json")
    deadline = read_json(run_dir / "campaign_deadline.json")
    data = admission.get("data")
    auxiliary = data.get("auxiliary_manifest") if isinstance(data, Mapping) else None
    limits = run_manifest.get("limits")
    if (
        admission.get("stage2", {}).get("max_trials") != 6
        or admission.get("stage3", {}).get("max_trials") != 12
        or admission.get("stage_fit_caps") != STAGE_CAPS
        or not isinstance(data, Mapping)
        or data.get("primary_split_sha256")
        != "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
        or not isinstance(auxiliary, Mapping)
        or auxiliary.get("sha256")
        != "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
        or budget.get("max_fit_attempts") != 48
        or budget.get("max_profile_attempts") != 56
        or budget.get("stage_fit_caps") != STAGE_CAPS
        or deadline.get("campaign_wall_seconds") != 21600
        or deadline.get("collection_reserve_seconds") != 1200
        or not isinstance(limits, Mapping)
        or limits.get("max_fit_attempts") != 48
        or limits.get("max_profile_attempts") != 56
        or limits.get("campaign_wall_seconds") != 21600
        or limits.get("collection_reserve_seconds") != 1200
        or limits.get("max_profile_wall_seconds") != 180
    ):
        raise ValueError("Stage2 proposal override differs from frozen campaign limits or split")
    fits, profiles = budget.get("fit_attempts"), budget.get("profile_attempts")
    if not isinstance(fits, list) or not isinstance(profiles, list):
        raise ValueError("Stage2 campaign budget ledgers are malformed")
    if len(fits) > 48 or len(profiles) > 56:
        raise ValueError("Stage2 proposal override would exceed a global attempt cap")
    if sum(isinstance(row, Mapping) and row.get("stage") == 2 for row in fits) > 8:
        raise ValueError("Stage2 fit-attempt cap has already been exceeded")


def validate_stage2_proposal_history(
    run_dir: Path,
    admission: Mapping[str, object],
    trials: Sequence[object],
) -> None:
    """Check proposal 0 was a zero-update failure and preserve approved/queued rows."""
    by_number = {int(row.number): row for row in trials}
    if sorted(by_number) not in (list(range(6)), list(range(7))):
        raise ValueError("Stage2 proposal history is not exactly trials 0 through 5 or 6")
    budget = read_json(run_dir / "budget.json")
    _validate_trial0_failure(run_dir, by_number[0], budget)
    _validate_approved_profiles(run_dir, budget)
    for number in (4, 5):
        trial_dir = run_dir / "stage2" / "trials" / f"{number:06d}"
        if not (trial_dir / "trial_definition.json").is_file():
            raise ValueError("the already queued Stage2 trials 4 and 5 must be preserved")
    if (
        6 in by_number
        and not (run_dir / "stage2" / "trials" / "000006" / "trial_definition.json").is_file()
    ):
        raise ValueError("Stage2 trial 6 exists without its durable definition")
    if admission.get("stage2", {}).get("max_trials") != 6:
        raise ValueError("the frozen Stage2 admission was modified")


def _validate_trial0_failure(run_dir: Path, trial: object, budget: Mapping[str, object]) -> None:
    if getattr(getattr(trial, "state", None), "name", None) != "FAIL":
        raise ValueError("Stage2 trial 0 is not the recorded terminal failure")
    trial_dir = run_dir / "stage2" / "trials" / "000000"
    profile_dir = trial_dir / "profile"
    attempt_path = profile_dir / "profile_attempt_receipt.json"
    if attempt_path.is_symlink() or not attempt_path.is_file():
        raise ValueError("Stage2 trial 0 lacks its failure attempt receipt")
    attempt = read_json(attempt_path)
    worker_path = Path(str(attempt.get("worker_receipt_path"))).resolve(strict=True)
    worker = read_json(worker_path)
    native_path = Path(str(attempt.get("native_profile_path"))).resolve()
    output_dir = Path(str(worker.get("output_dir"))).resolve()
    profile_rows = _budget_rows(budget, "profile_attempts", 0)
    fit_rows = _budget_rows(budget, "fit_attempts", 0)
    if (
        attempt.get("schema") != "wnn.multistage-profile-attempt.v1"
        or attempt.get("stage") != 2
        or attempt.get("trial_number") != 0
        or attempt.get("status") != "profile_failed"
        or attempt.get("exit_code") != 1
        or attempt.get("native_profile_sha256") is not None
        or worker_path != (profile_dir / "worker_receipt.json").resolve()
        or attempt.get("worker_receipt_sha256") != file_sha256(worker_path)
        or worker.get("schema") != "wnn.multistage-owned-process.v1"
        or worker.get("stage") != 2
        or worker.get("trial_number") != 0
        or worker.get("mode") != "profile"
        or worker.get("status") != "failed"
        or worker.get("exit_code") != 1
        or worker.get("timed_out") is not False
        or native_path.parent != output_dir
        or output_dir != (profile_dir / "training_output").resolve()
        or native_path.exists()
        or (output_dir / "rank0.jsonl").exists()
        or output_dir.is_symlink()
        or (output_dir.exists() and not output_dir.is_dir())
        or len(profile_rows) != 1
        or profile_rows[0].get("status") != "profile_failed"
        or profile_rows[0].get("exit_code") != 1
        or fit_rows
        or (run_dir / "stage2" / "runs" / "000000" / "model.pt").exists()
    ):
        raise ValueError("Stage2 trial 0 is not the independently reviewed zero-update failure")


def _validate_approved_profiles(run_dir: Path, budget: Mapping[str, object]) -> None:
    for number in (1, 2, 3):
        profile_dir = run_dir / "stage2" / "trials" / f"{number:06d}" / "profile"
        attempt = read_json(profile_dir / "profile_attempt_receipt.json")
        approval = read_json(profile_dir / "approval_record.json")
        review_path = Path(str(approval["review_receipt_path"])).resolve(strict=True)
        review = read_json(review_path)
        native_path = Path(str(approval["native_profile_path"])).resolve(strict=True)
        approved_path = Path(str(approval["approved_profile_path"])).resolve(strict=True)
        profile_rows = _budget_rows(budget, "profile_attempts", number)
        native = read_json(native_path)
        approved = read_json(approved_path)
        external = approved.get("external_review")
        if (
            attempt.get("status") != "profile_complete"
            or attempt.get("exit_code") != 0
            or attempt.get("native_profile_sha256") != file_sha256(native_path)
            or approval.get("status") != "approved_for_exact_fit_identity"
            or approval.get("review_revision") != "R6"
            or approval.get("native_profile_sha256") != file_sha256(native_path)
            or approval.get("approved_profile_sha256") != file_sha256(approved_path)
            or approval.get("review_receipt_sha256") != file_sha256(review_path)
            or approved.get("approved") is not True
            or review.get("status") != "approved"
            or review.get("stage") != 2
            or review.get("trial_number") != number
            or not isinstance(external, Mapping)
            or external.get("review_receipt_sha256") != file_sha256(review_path)
            or external.get("native_profile_sha256") != file_sha256(native_path)
            or native_path.is_symlink()
            or native.get("approved") is not False
            or len(profile_rows) != 1
            or profile_rows[0].get("status") != "profile_complete"
            or profile_rows[0].get("exit_code") != 0
        ):
            raise ValueError("existing R6-approved Stage2 profiles 1-3 must remain intact")


def _budget_rows(
    budget: Mapping[str, object], field: str, trial_number: int
) -> list[Mapping[str, object]]:
    rows = budget.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"Stage2 {field} ledger is malformed")
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("stage") == 2
        and row.get("trial_number") == trial_number
    ]
