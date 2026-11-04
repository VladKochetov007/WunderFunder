from __future__ import annotations

from pathlib import Path

from wnn.multistage_search import read_json, validate_admission
from wnn.multistage_search_fit import run_stage1_fit, run_stage_fit
from wnn.multistage_search_profile import run_profile_trial
from wnn.multistage_search_review import run_profile_reviewer
from wnn.multistage_search_runtime import hold_gpu_slot
from wnn.multistage_search_successor_profile import run_successor_profile
from wnn.multistage_search_teachers import export_teacher
from wnn.multistage_search_worker_state import _fail_owned_profile_trial, _finish_worker_owner
from wnn.multistage_search_worker_support import _device_spec


def run_worker(
    run_dir: Path,
    *,
    mode: str,
    worker_id: str,
    gpu_key: str,
    trial_number: int | None = None,
    stage: int = 1,
    result_path: Path | None = None,
) -> dict[str, object]:
    """Run one bounded profile, approved fit, or fixed-final teacher export."""
    if stage not in {1, 2, 3} or mode not in {"profile", "fit", "teacher-export"}:
        raise ValueError("worker mode or stage is unsupported")
    if mode == "teacher-export" and stage not in {1, 2}:
        raise ValueError("teacher export uses its source stage1/2")
    if mode == "teacher-export" and result_path is None and trial_number is None:
        raise ValueError("teacher export requires an exact trial number or result path")
    if result_path is not None and mode != "teacher-export":
        raise ValueError("a result path may be supplied only for a teacher export")
    if stage in {2, 3} and mode in {"profile", "fit"} and trial_number is None:
        raise ValueError("successor profile/fit workers require an exact trial number")
    run_manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    contract = read_json(Path(str(run_manifest["contract"]["path"])))
    validate_admission(contract, admission, str(run_manifest["contract"]["sha256"]))
    device = _device_spec(admission, gpu_key)
    try:
        with hold_gpu_slot(run_dir, gpu_key):
            if mode == "teacher-export":
                exact_result = (
                    result_path.resolve(strict=True)
                    if result_path is not None
                    else run_dir
                    / f"stage{stage}"
                    / "trials"
                    / f"{int(trial_number):06d}"
                    / "trial_result.json"
                )
                if not exact_result.is_relative_to(run_dir.resolve(strict=True)):
                    raise ValueError("teacher result path must stay inside the frozen campaign")
                result = read_json(exact_result)
                if (
                    result.get("status") != "COMPLETE"
                    or result.get("stage") != stage
                    or type(result.get("trial_number")) is not int
                    or (trial_number is not None and result.get("trial_number") != trial_number)
                ):
                    raise ValueError("teacher result does not match the exact worker request")
                return export_teacher(
                    run_dir,
                    exact_result,
                    worker_id=worker_id,
                    gpu_key=gpu_key,
                    device=device,
                )
            if mode == "profile" and stage == 1:
                profile_result = run_profile_trial(
                    run_dir,
                    admission,
                    worker_id,
                    gpu_key,
                    device,
                    prepared_trial_number=trial_number,
                )
                if profile_result.get("status") != "profile_complete_waiting_review":
                    return profile_result
                number = int(profile_result["trial_number"])
                review = run_profile_reviewer(run_dir, number)
                if review.get("status") == "pending_anchor_review":
                    return {
                        "status": "waiting_root_anchor_review",
                        "trial_number": number,
                        "review_receipt_sha256": review.get("review_receipt_sha256"),
                    }
                if review.get("status") != "approved":
                    raise ValueError("independent profile checker did not approve this trial")
                from wnn.multistage_search_approval import approve_profile

                review_path = (
                    run_dir
                    / "stage1"
                    / "trials"
                    / f"{number:06d}"
                    / "profile"
                    / "profile_review_receipt.json"
                )
                approve_profile(run_dir, number, review_path)
                _finish_worker_owner(run_dir, worker_id)
                return run_stage1_fit(run_dir, admission, worker_id, gpu_key, number, device)
            if mode == "profile":
                return run_successor_profile(
                    run_dir,
                    stage=stage,
                    trial_number=int(trial_number),
                    worker_id=worker_id,
                    gpu_key=gpu_key,
                    device=device,
                )
            return run_stage_fit(
                run_dir, admission, worker_id, gpu_key, trial_number, device, stage=stage
            )
    except Exception as error:
        if mode == "profile":
            _fail_owned_profile_trial(run_dir, worker_id, error, stage=stage)
        raise
    finally:
        _finish_worker_owner(run_dir, worker_id)
