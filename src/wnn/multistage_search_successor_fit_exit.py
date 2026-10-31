from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_sources import training_source_for_stage
from wnn.training.sequence_provenance import source_identity


def validate_fit_exit(
    run_dir: Path,
    trial_dir: Path,
    fit_dir: Path,
    request: Mapping[str, object],
    approved_path: Path,
    review_sha256: str,
    attempt_id: str,
    stage: int,
    trial_number: int,
) -> dict[str, str]:
    """Require guardian-observed successful exit and its exact global budget row."""
    receipt_path = run_dir.parent / "receipts" / f"{attempt_id}.child.json"
    launch_path = run_dir.parent / "requests" / f"{attempt_id}.json"
    receipt = read_json(receipt_path)
    launch = read_json(launch_path)
    approval_path = trial_dir / "profile" / "approval_record.json"
    config_path = Path(str(request["config_path"])).resolve(strict=True)
    source_root, source_revision, expected_source_sha, expected_files_sha = (
        training_source_for_stage(read_json(run_dir / "run.json"), stage)
    )
    source = source_identity(source_root, source_root / "scripts" / "train_sequence.py")
    launch_source_root = Path(str(launch["source_root"])).resolve(strict=True)
    if (
        launch_source_root != source_root
        or request.get("source_revision") != source_revision
        or request.get("source_sha256") != expected_source_sha
        or source.get("sha256") != expected_source_sha
        or request.get("source_files_sha256") != expected_files_sha
        or canonical_json_sha256(source.get("files")) != expected_files_sha
    ):
        raise ValueError("fit launch source differs from the frozen stage source identity")
    params = request.get("params")
    kd_weight = params.get("kd_weight", 0.0) if isinstance(params, Mapping) else 0.0
    expected_command = [
        str(source_root / "scripts" / "train_sequence.py"),
        "--config",
        str(config_path),
        "--data-root",
        str(Path(str(request["data_root"])).resolve(strict=True)),
        "--split-manifest",
        str(Path(str(request["auxiliary_split_manifest"])).resolve(strict=True)),
        "--output",
        str(fit_dir.resolve()),
        "--profile-receipt",
        str(approved_path),
    ]
    if stage == 2:
        expected_command.extend(["--initializer", str(request["initializer_path"])])
    elif float(kd_weight) > 0:
        expected_command.extend(["--teacher-cache", str(request["teacher_cache_path"])])
    command = receipt.get("command")
    child_environment = launch.get("child_environment")
    selectors = request.get("requested_device_set")
    expected_environment = {
        "PYTHONPATH": f"{source_root / 'src'}{os.pathsep}{run_dir.parent / 'python-deps'}",
        "CUDA_VISIBLE_DEVICES": selectors[0] if isinstance(selectors, list) and selectors else None,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "WNN_PORTABLE_DATA_MANIFEST": str(run_dir.parent / "inputs" / "portable-data-r1.json"),
        "WNN_PROJECT_ROOT": str(Path(str(request["data_root"])).resolve(strict=True)),
    }
    if (
        launch.get("stage") != stage
        or launch.get("trial_number") != trial_number
        or launch.get("fit_attempt_id") != attempt_id
        or Path(str(launch.get("output_dir"))).resolve() != fit_dir.resolve()
        or launch.get("trial_request_sha256") != file_sha256(trial_dir / "trial_request.json")
        or launch.get("review_sha256") != review_sha256
        or launch.get("approval_sha256") != file_sha256(approval_path)
        or launch.get("approved_profile_sha256") != file_sha256(approved_path)
        or launch.get("config_file_sha256") != file_sha256(config_path)
        or launch.get("config_identity_sha256") != request.get("config_identity_sha256")
        or not isinstance(command, list)
        or not command
        or command[1:] != expected_command
        or launch.get("command") != command
        or receipt.get("command") != command
        or receipt.get("command_sha256") != canonical_json_sha256(command)
        or not isinstance(child_environment, Mapping)
        or any(child_environment.get(key) != value for key, value in expected_environment.items())
        or receipt.get("schema") != "wnn.multistage-owned-process.v1"
        or receipt.get("stage") != stage
        or receipt.get("trial_number") != trial_number
        or receipt.get("mode") not in {"fit", f"stage{stage}_fit"}
        or receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
        or Path(str(receipt.get("output_dir"))).resolve() != fit_dir.resolve()
        or receipt.get("gpu_key") != request.get("gpu_key")
    ):
        raise ValueError("successor fit lacks its successful exact child-process receipt")
    budget = read_json(run_dir / "budget.json")
    attempts = budget.get("fit_attempts")
    if not isinstance(attempts, list):
        raise ValueError("successor fit budget ledger is malformed")
    rows = [
        row for row in attempts if isinstance(row, Mapping) and row.get("attempt_id") == attempt_id
    ]
    if (
        len(rows) != 1
        or rows[0].get("status") not in {"complete", "completed"}
        or rows[0].get("exit_code") != 0
    ):
        raise ValueError("successor fit budget ledger does not attest one successful attempt")
    return {"path": str(receipt_path.resolve()), "sha256": file_sha256(receipt_path)}
