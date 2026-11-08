from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_budget_fit import finish_budget_attempt, reserve_fit
from wnn.multistage_search_reviewed_successor_validation import (
    approve_profile,
    ensure_gpu_slot_released,
    validate_initializer,
    validate_profile_members,
    validate_stage3_teacher,
)
from wnn.multistage_search_runtime import launch_detached_worker
from wnn.multistage_search_sources import training_source_for_stage
from wnn.multistage_search_successor_review import validate_successor_review
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import source_identity

SOURCE5B_SHA = "2661a89285bd528e1bca690a340e71a21816a7b54e3ba1635d14925f3a9c1e4e"
REQUEST_POLICY_R2_SHA = "153655303a89dfd9dd2372a20238327aba36337a36141373c51c5cc1e30c2f0e"


def launch_reviewed_successor_fit(
    run_dir: Path,
    stage: int,
    trial_number: int,
    review_path: Path,
    review_sha256: str,
    operator_root: Path,
    guardian_path: Path | None = None,
) -> dict[str, object]:
    """Validate one reviewed Stage2/3 request, reserve, then launch its R2 guardian."""
    if stage not in {2, 3} or type(trial_number) is not int or trial_number < 0:
        raise ValueError("successor fit identity must be a stage2/3 trial number")
    run_dir, operator_root = run_dir.resolve(strict=True), operator_root.resolve(strict=True)
    if not (operator_root / "src" / "wnn").is_dir():
        raise FileNotFoundError("operator source root must contain src/wnn")
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{trial_number:06d}"
    request_path = trial_dir / "trial_request.json"
    if trial_dir.is_symlink() or request_path.is_symlink():
        raise ValueError("successor trial request cannot be a symlink")
    request = read_json(request_path)
    review_path = review_path.resolve(strict=True)
    if file_sha256(review_path) != review_sha256:
        raise ValueError("independent review bytes differ from the pinned SHA256")
    run_manifest = read_json(run_dir / "run.json")
    review_proof = validate_successor_review(
        run_dir, trial_dir, stage, trial_number, request, review_path
    )
    if (
        request.get("schema") != "wnn.multistage-trial-plan.v1"
        or request.get("stage") != stage
        or request.get("trial_number") != trial_number
        or request.get("primary_valid_used") is not False
        or request.get("objective_scope") != "inner_train_proxy"
        or request.get("profile_checker_policy_sha256") != REQUEST_POLICY_R2_SHA
        or request.get("profile_policy_sha256") != REQUEST_POLICY_R2_SHA
    ):
        raise ValueError("successor request is not the exact train-only trial")

    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    source_root, _revision, source_sha, source_files_sha = training_source_for_stage(
        run_manifest, stage
    )
    source = source_identity(source_root, source_root / "scripts" / "train_sequence.py")
    if (
        source_sha != SOURCE5B_SHA
        or source.get("sha256") != SOURCE5B_SHA
        or canonical_json_sha256(source.get("files")) != source_files_sha
        or request.get("source_sha256") != SOURCE5B_SHA
        or request.get("source_files_sha256") != source_files_sha
        or (
            request.get("training_source_root") is not None
            and Path(str(request["training_source_root"])).resolve(strict=True) != source_root
        )
    ):
        raise ValueError("successor trainer is not the frozen Source5B identity")
    config_path = Path(str(request["config_path"])).resolve(strict=True)
    if file_sha256(config_path) != request.get("config_sha256"):
        raise ValueError("successor raw config differs from its exact request")
    config = load_config(config_path)
    if canonical_json_sha256(config) != request.get("config_identity_sha256"):
        raise ValueError("successor normalized config differs from the profile identity")
    data = admission["data"]
    auxiliary = data["auxiliary_manifest"]
    auxiliary_path = Path(str(auxiliary["path"])).resolve(strict=True)
    data_root = Path(str(data["data_root"])).resolve(strict=True)
    if (
        file_sha256(auxiliary_path) != auxiliary.get("sha256")
        or request.get("auxiliary_split_sha256") != auxiliary.get("sha256")
        or request.get("primary_split_sha256") != data.get("primary_split_sha256")
        or Path(str(request["auxiliary_split_manifest"])).resolve(strict=True) != auxiliary_path
        or Path(str(request["data_root"])).resolve(strict=True) != data_root
    ):
        raise ValueError("successor split or data root differs from frozen admission")
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members

    split = load_campaign_split(auxiliary_path, data_root)
    fit_members, control_members = auxiliary_members(split)
    native_path = Path(str(request["profile_receipt_path"])).resolve(strict=True)
    native = read_json(native_path)
    identity = native.get("identity")
    identity_data = identity.get("data") if isinstance(identity, Mapping) else None
    validate_profile_members(identity_data, split, fit_members, control_members)
    if stage == 2:
        validate_initializer(request, config, identity, identity_data, fit_members, control_members)
    else:
        validate_stage3_teacher(request, config, identity, identity_data, split, fit_members)

    output = run_dir / f"stage{stage}" / "runs" / f"{trial_number:06d}"
    attempt_id = f"stage{stage}-trial-{trial_number:06d}"
    runtime_root = run_dir.parent
    fit_request = runtime_root / "requests" / f"{attempt_id}.json"
    fit_log_path = runtime_root / "logs" / f"{attempt_id}.log"
    owner_path = runtime_root / "receipts" / f"{attempt_id}.guardian-owner.json"
    child_receipt_path = runtime_root / "receipts" / f"{attempt_id}.child.json"
    approved_path = trial_dir / "profile" / "approved_profile_receipt.json"
    approval_path = trial_dir / "profile" / "approval_record.json"
    guardian = guardian_path or runtime_root / "requests" / "fit_guardian_r2.py"
    if not guardian.is_file():
        guardian = (
            Path(__file__).parents[2]
            / "research"
            / "experiment_drafts"
            / ("multistage-optuna-b300-20261031")
            / "fit_guardian_r2.py"
        )
    if any(
        path.exists() or path.is_symlink()
        for path in (
            output,
            fit_request,
            fit_log_path,
            owner_path,
            owner_path.with_suffix(".log"),
            child_receipt_path,
            approved_path,
            approval_path,
        )
    ):
        raise FileExistsError("successor fit attempt already has output or launch receipts")
    if not guardian.is_file():
        raise FileNotFoundError("fit_guardian_r2.py is unavailable")
    deps = runtime_root / "python-deps"
    portable_manifest = runtime_root / "inputs" / "portable-data-r1.json"
    if not deps.is_dir() or not portable_manifest.is_file():
        raise FileNotFoundError("Source5B dependencies or portable data manifest are missing")
    gpu_key = request.get("gpu_key")
    selectors = request.get("requested_device_set")
    if not isinstance(gpu_key, str) or not isinstance(selectors, list) or len(selectors) != 1:
        raise ValueError("successor request must bind one admitted GPU key and selector")
    ensure_gpu_slot_released(run_dir, gpu_key)

    limits = run_manifest["limits"]
    stage_policy = admission.get(f"stage{stage}", {})
    score_reserve = int(stage_policy.get("postfit_eval_reserve_seconds", 900))
    reserve_start = read_json(run_dir / "campaign_deadline.json").get("reserve_start_epoch")
    if not isinstance(reserve_start, (int, float)):
        raise ValueError("campaign deadline has no collection-reserve epoch")
    maximum_wall = min(
        7200, int(limits["max_fit_wall_seconds"]), int(reserve_start - time.time()) - score_reserve
    )
    if maximum_wall <= 0:
        raise RuntimeError("fit admission is closed to preserve scoring and collection reserves")
    budget_record, wall_limit = reserve_fit(run_dir, stage, attempt_id, trial_number, maximum_wall)
    remaining_with_score_reserve = int(reserve_start - time.time()) - score_reserve
    if wall_limit > remaining_with_score_reserve:
        finish_budget_attempt(run_dir, attempt_id, "failed", None)
        raise RuntimeError("fit reservation no longer preserves the post-fit collection reserve")
    try:
        approved = approve_profile(run_dir, stage, trial_number, review_path, identity)
        if approved != approved_path.resolve(strict=True):
            raise ValueError("approved profile path differs from the exact successor trial")
        command = [
            sys.executable,
            str(source_root / "scripts" / "train_sequence.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(auxiliary_path),
            "--output",
            str(output),
            "--profile-receipt",
            str(approved),
        ]
        if stage == 2:
            command.extend(["--initializer", str(request["initializer_path"])])
        elif float(config["kd_weight"]) > 0:
            command.extend(["--teacher-cache", str(request["teacher_cache_path"])])
        child_environment = {
            "PYTHONPATH": f"{source_root / 'src'}:{deps}",
            "CUDA_VISIBLE_DEVICES": str(selectors[0]),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "WNN_PORTABLE_DATA_MANIFEST": str(portable_manifest),
            "WNN_PROJECT_ROOT": str(data_root),
        }
        fit_payload = {
            "command": command,
            "source_root": str(source_root),
            "stage": stage,
            "trial_number": trial_number,
            "gpu_key": gpu_key,
            "output_dir": str(output),
            "timeout_seconds": wall_limit,
            "log_path": str(fit_log_path),
            "child_receipt_path": str(child_receipt_path),
            "budget_run_dir": str(run_dir),
            "fit_attempt_id": attempt_id,
            "child_environment": child_environment,
            "trial_request_sha256": file_sha256(request_path),
            "review_sha256": file_sha256(review_path),
            "approval_sha256": file_sha256(approval_path),
            "approved_profile_sha256": file_sha256(approved),
            "config_file_sha256": file_sha256(config_path),
            "config_identity_sha256": canonical_json_sha256(config),
            "budget_reservation": budget_record,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        fit_request.parent.mkdir(parents=True, exist_ok=True)
        (runtime_root / "logs").mkdir(parents=True, exist_ok=True)
        _write_new_json(fit_request, fit_payload)
        parent_environment = os.environ.copy()
        parent_environment["PYTHONPATH"] = f"{operator_root / 'src'}:{deps}"
        launch_detached_worker(
            [sys.executable, str(guardian.resolve()), "--request", str(fit_request.resolve())],
            cwd=operator_root,
            env=parent_environment,
            receipt_path=owner_path,
            metadata={
                "worker_id": f"reviewed-stage{stage}-{trial_number:06d}",
                "stage": stage,
                "trial_number": trial_number,
                "gpu_key": gpu_key,
                "fit_attempt_id": attempt_id,
            },
        )
    except Exception:
        finish_budget_attempt(run_dir, attempt_id, "failed", None)
        raise
    return {
        "status": "guardian_launched",
        "stage": stage,
        "trial_number": trial_number,
        "fit_attempt_id": attempt_id,
        "output_dir": str(output),
        "fit_request_path": str(fit_request.resolve()),
        "fit_request_sha256": file_sha256(fit_request),
        "guardian_owner_path": str(owner_path.resolve()),
        "guardian_owner_sha256": file_sha256(owner_path),
        "review_sha256": review_sha256,
        "review_revision": review_proof["revision"],
        "source_sha256": SOURCE5B_SHA,
        "wall_limit_seconds": wall_limit,
    }


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
