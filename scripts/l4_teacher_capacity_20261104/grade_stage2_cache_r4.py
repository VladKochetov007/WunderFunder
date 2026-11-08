"""Run the frozen cache grader with only its R3 recovery gate replaced by R4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.l4_teacher_capacity_20261104 import grade_stage2_cache as base  # noqa: E402
from wnn.distillation_cache import file_sha256  # noqa: E402

ADMISSION_SCHEMA = "wnn.l4-student-recovery-source-admission-r4.v1"
STATE_SCHEMA = "wnn.l4-teacher-capacity-continuation-recovery-r4.v1"
SUMMARY_SCHEMA = "wnn.l4-teacher-stage2-metadata-recovery-r4.v1"
RELEASE_SCHEMA = "wnn.root-gpu-release.v1"
CONTROLLER_ID = "student_continuation_recovery_r4_controller"
FINALIZER_ID = "causal_transformer_stage2_compact_finalize_r4"
FINALIZER_CPU = tuple(range(32, 40))
CONTROLLER_CPU = tuple(range(64))
_ADMISSION_PATH: Path | None = None


def _canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _same_file(path: Path, expected_path: Any, expected_sha: Any, label: str) -> None:
    base._require(path.is_file() and not path.is_symlink(), f"{label} is absent: {path}")
    base._require(
        expected_path == str(path.resolve()) and expected_sha == file_sha256(path),
        f"{label} path or SHA256 differs",
    )


def _check_r4_admission(
    panel: dict[str, Any], panel_path: Path, panel_hash: str
) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    if _ADMISSION_PATH is None:
        raise RuntimeError("R4 recovery admission path was not supplied")
    paths, identity = panel["paths"], panel["identity"]
    control = Path(paths["control_root"]).resolve(strict=True)
    admission_path = _ADMISSION_PATH.resolve(strict=True)
    base._require(admission_path.parent == control, "R4 admission must reside in the bound control root")
    admission = base._read_json(admission_path)
    base._require(
        admission.get("schema") == ADMISSION_SCHEMA
        and str(admission.get("status", "")).startswith("approved")
        and admission.get("continuation_panel_sha256") == panel_hash
        and all(
            admission.get(key) == identity.get(key)
            for key in (
                "data_binding_sha256",
                "split_sha256",
                "fit_members_sha256",
                "baseline_ancestry_sha256",
            )
        )
        and _valid_sha(admission.get("source_binding_sha256")),
        "approved R4 recovery admission does not bind this frozen panel/source",
    )
    runtime = Path(paths["runtime_root"])
    for key, filename in (
        ("r3_start_receipt", "STUDENT_CONTINUATION_RECOVERY_R3_START.json"),
        ("r3_failure_receipt", "STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json"),
    ):
        receipt_path = runtime / "receipts" / "continuation" / filename
        _same_file(
            receipt_path,
            admission.get(f"{key}_path"),
            admission.get(f"{key}_sha256"),
            f"pinned predecessor {key}",
        )
    start = base._read_json(runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_START.json")
    base._require(
        start.get("schema") == STATE_SCHEMA
        and start.get("status") == "running"
        and start.get("panel_sha256") == panel_hash
        and start.get("recovery_source_binding_sha256") == admission["source_binding_sha256"]
        and start.get("recovery_admission_receipt_path") == str(admission_path)
        and start.get("recovery_admission_receipt_sha256") == file_sha256(admission_path),
        "R4 controller start does not bind its approved admission and source",
    )
    controller_ref = start.get("controller_receipt", {})
    controller_path = Path(str(controller_ref.get("path", ""))).resolve(strict=True)
    controller = base._read_json(controller_path)
    script_path = Path(str(controller.get("owner", {}).get("recovery_entrypoint_path", "")))
    owner = controller.get("owner", {})
    source_sha = admission["source_binding_sha256"]
    base._require(
        controller.get("schema") == "wnn.multistage-owned-process.v1"
        and controller.get("status") == "running"
        and controller.get("child") == controller_ref.get("child")
        and controller.get("command_sha256") == start.get("controller_command_sha256")
        and owner.get("cell_id") == CONTROLLER_ID
        and owner.get("run_name") == CONTROLLER_ID
        and owner.get("family") == "continuation"
        and owner.get("continuation_panel_sha256") == panel_hash
        and owner.get("recovery_source_binding_sha256") == source_sha
        and owner.get("recovery_admission_receipt_path") == str(admission_path)
        and owner.get("recovery_admission_receipt_sha256") == file_sha256(admission_path)
        and script_path.name == "continue_students_recovery_r4.py"
        and file_sha256(script_path) == owner.get("recovery_entrypoint_sha256")
        and owner.get("recovery_entrypoint_path") == str(script_path.resolve())
        and owner.get("source_binding_sha256") == identity["source_binding_sha256"]
        and owner.get("data_binding_sha256") == identity["data_binding_sha256"]
        and owner.get("split_sha256") == identity["split_sha256"]
        and owner.get("fit_members_sha256") == identity["fit_members_sha256"]
        and owner.get("baseline_ancestry_sha256") == identity["baseline_ancestry_sha256"],
        "R4 controller process/source identity differs from its admitted receipt",
    )
    taskset = shutil.which("taskset")
    expected_controller = [
        taskset,
        "-c",
        ",".join(map(str, CONTROLLER_CPU)),
        paths["python"],
        str(script_path.resolve()),
        "--controller-mode",
        "continue",
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path),
        "--controller-receipt",
        str(controller_path),
        "--gpu2-release-receipt",
        str(Path(admission["gpu2_release_receipt_path"]).resolve()),
    ]
    base._require(
        taskset is not None
        and controller.get("command") == expected_controller
        and owner.get("command_sha256")
        == hashlib.sha256(json.dumps(expected_controller, separators=(",", ":")).encode()).hexdigest(),
        "R4 controller command or 64-core affinity differs",
    )
    return admission, admission_path, file_sha256(admission_path), {
        "start_path": runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_START.json",
        "controller_path": controller_path,
        "controller": controller,
        "controller_script": script_path.resolve(),
    }


def _check_gpu2_release(panel: dict[str, Any], admission: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    release_path = Path(admission.get("gpu2_release_receipt_path", "")).resolve(strict=True)
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    tf_binding = panel["families"]["causal_transformer"]
    base._require(
        admission.get("gpu2_release_schema") == RELEASE_SCHEMA
        and admission.get("gpu2_uuid") == tf_binding["export_gpu_uuid"],
        "R4 admission GPU2 release schema/UUID differs from the TF exporter",
    )
    base._require(
        release_path.parent == control and release_path.name == "GPU2_RELEASE_R1.json",
        "R4 GPU2 release is not the root-approved control receipt",
    )
    release = base._read_json(release_path)
    fit_path = Path(str(release.get("objective_fit_receipt_path", "")))
    fit = base._read_json(fit_path)
    fit_owner = fit.get("owner", {})
    base._require(
        release.get("schema") == RELEASE_SCHEMA
        and release.get("status") == "released"
        and release.get("gpu_uuid") == admission["gpu2_uuid"]
        and _valid_sha(release.get("objective_panel_sha256"))
        and release.get("objective_fit_receipt_sha256") == file_sha256(fit_path)
        and fit.get("status") == "completed"
        and fit.get("exit_code") == 0
        and fit.get("timed_out") is False
        and fit_owner.get("cell_id") == "clip5_s1001"
        and fit_owner.get("panel_sha256", fit_owner.get("continuation_panel_sha256"))
        == release["objective_panel_sha256"],
        "R4 GPU2 release does not bind a terminal successful objective fit",
    )
    return release_path, release


def _check_stage2_recovery_r4(
    panel: dict[str, Any],
    panel_path: Path,
    panel_hash: str,
    family: str,
    cache: Path,
    init_add_records: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Validate the R4 TF finalizer while leaving all base grade checks unchanged."""
    base._require(family == "causal_transformer", "R4 adapter is scoped to the pending TF family")
    admission, admission_path, admission_sha, lifecycle = _check_r4_admission(
        panel, panel_path, panel_hash
    )
    release_path, release = _check_gpu2_release(panel, admission)
    paths, identity = panel["paths"], panel["identity"]
    runtime = Path(paths["runtime_root"])
    receipts_root = runtime / "receipts" / "stage2"
    summary_path = receipts_root / "causal_transformer_metadata_recovery_r4.json"
    summary = base._read_json(summary_path)
    worker_path = receipts_root / f"{FINALIZER_ID}.json"
    worker = base._read_json(worker_path)
    worker_owner = worker.get("owner", {})
    worker_child = worker.get("child")
    worker_command = worker.get("command")
    entrypoint = Path(str(worker_owner.get("recovery_entrypoint_path", ""))).resolve(strict=True)
    entrypoint_sha = file_sha256(entrypoint)
    base._require(
        summary.get("schema") == SUMMARY_SCHEMA
        and summary.get("status") == "compact_finalized_and_promoted"
        and summary.get("family") == family
        and summary.get("continuation_panel_sha256") == panel_hash
        and summary.get("recovery_source_binding_sha256") == admission["source_binding_sha256"]
        and summary.get("worker_receipt_path") == str(worker_path.resolve())
        and summary.get("worker_receipt_sha256") == file_sha256(worker_path)
        and summary.get("worker_child") == worker_child
        and entrypoint.name == "continue_students_recovery_r4.py"
        and worker_owner.get("recovery_entrypoint_sha256") == entrypoint_sha,
        "R4 compact finalizer summary/source binding differs",
    )
    expected_command = [
        str(paths["python"]),
        str(entrypoint),
        "--worker-mode",
        "tf-finalize",
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path),
        "--controller-receipt",
        str(lifecycle["controller_path"]),
        "--worker-receipt",
        str(worker_path.resolve()),
        "--gpu2-release-receipt",
        str(release_path),
    ]
    expected_owned_command = [
        shutil.which("taskset"),
        "-c",
        ",".join(map(str, FINALIZER_CPU)),
        *expected_command,
    ]
    base._require(
        worker.get("schema") == "wnn.multistage-owned-process.v1"
        and worker.get("status") == "completed"
        and worker.get("exit_code") == 0
        and worker.get("timed_out") is False
        and worker.get("child") == summary.get("worker_child")
        and worker_command == expected_owned_command
        and worker.get("output_dir") == str(cache.resolve())
        and worker.get("gpu_key") == "cpu-stage2"
        and worker_owner.get("cell_id") == FINALIZER_ID
        and worker_owner.get("run_name") == FINALIZER_ID
        and worker_owner.get("family") == family
        and worker_owner.get("continuation_panel_sha256") == panel_hash
        and worker_owner.get("source_binding_sha256") == identity["source_binding_sha256"]
        and worker_owner.get("data_binding_sha256") == identity["data_binding_sha256"]
        and worker_owner.get("split_sha256") == identity["split_sha256"]
        and worker_owner.get("fit_members_sha256") == identity["fit_members_sha256"]
        and worker_owner.get("baseline_ancestry_sha256") == identity["baseline_ancestry_sha256"]
        and worker_owner.get("recovery_source_binding_sha256") == admission["source_binding_sha256"]
        and worker_owner.get("recovery_admission_receipt_path") == str(admission_path)
        and worker_owner.get("recovery_admission_receipt_sha256") == admission_sha
        and worker_owner.get("recovery_entrypoint_path") == str(entrypoint)
        and worker_owner.get("command_sha256")
        == hashlib.sha256(json.dumps(worker_command, separators=(",", ":")).encode()).hexdigest(),
        "R4 finalizer is not the exact terminal owned CPU command",
    )
    worker_started, worker_finished = worker.get("started_at_epoch"), worker.get("finished_at_epoch")
    base._require(
        isinstance(worker_started, (int, float))
        and isinstance(worker_finished, (int, float))
        and worker_started <= worker_finished,
        "R4 finalizer timing is incomplete",
    )

    metadata_path = cache / "metadata.json"
    metadata = base._read_json(metadata_path)
    build_path = cache / "stage2_build_receipt.json"
    build = base._read_json(build_path)
    contract = metadata.get("target_contract", {})
    main = metadata.get("files", {}).get("main_source_train", {})
    main_path = cache / str(main.get("path", "main_source_train.npy"))
    split = base.load_campaign_split(Path(paths["split_manifest"]), Path(paths["project_root"]))
    expected_runs = list(base.EXPECTED_RUNS[family])
    fit_order = list(map(int, split.train_groups))
    base._require(
        split.sha256 == base.SPLIT_SHA256
        and build.get("schema") == "wnn.l4-teacher-stage2-stream.v2"
        and summary.get("build_state_schema") == build.get("schema")
        and build.get("family") == family
        and build.get("expected_runs") == expected_runs
        and build.get("processed_runs") == expected_runs
        and build.get("pending_run") is None
        and build.get("split_manifest_sha256") == split.sha256
        and build.get("fit_row_groups") == fit_order
        and build.get("teacher_records") == contract.get("teachers")
        and metadata_path.stat().st_size <= base.MAX_CACHE_METADATA_BYTES
        and summary.get("target_cache_path") == str(cache.resolve())
        and summary.get("target_metadata_sha256") == file_sha256(metadata_path)
        and summary.get("target_contract_sha256") == _canonical_sha(contract)
        and main.get("sha256") == file_sha256(main_path)
        and summary.get("stage2_build_receipt_path") == str(build_path.resolve())
        and summary.get("stage2_build_receipt_sha256") == file_sha256(build_path)
        and summary.get("stage2_init_add_receipts") == init_add_records,
        "R4 summary differs from actual compact metadata/main/build/split/fit receipts",
    )
    original_finalize = receipts_root / "causal_transformer_stage2_04.json"
    original_finalize_log = runtime / "logs/stage2/causal_transformer_stage2_04.log"
    base._require(
        not original_finalize.exists() and not original_finalize_log.exists(),
        "Transformer R1 finalization was attempted; preserve and stop",
    )
    add_times = [base._read_json(Path(item["path"])) for item in init_add_records]
    latest_add = max(float(item["finished_at_epoch"]) for item in add_times)
    base._require(latest_add <= worker_started, "R4 finalizer preceded an init/add receipt")
    timeline = {
        "action": "finalize_recovery",
        "cell_id": FINALIZER_ID,
        "path": str(worker_path.resolve()),
        "sha256": file_sha256(worker_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "recovery_source_binding_sha256": admission["source_binding_sha256"],
        "recovery_admission_receipt_path": str(admission_path),
        "recovery_admission_receipt_sha256": admission_sha,
        "recovery_entrypoint_path": str(entrypoint),
        "recovery_entrypoint_sha256": entrypoint_sha,
        "r4_controller_receipt_path": str(lifecycle["controller_path"]),
        "r4_controller_receipt_sha256": file_sha256(lifecycle["controller_path"]),
        "r4_start_receipt_path": str(lifecycle["start_path"].resolve()),
        "r4_start_receipt_sha256": file_sha256(lifecycle["start_path"]),
        "gpu2_release_receipt_path": str(release_path),
        "gpu2_release_receipt_sha256": file_sha256(release_path),
        "started_at_epoch": worker_started,
        "finished_at_epoch": worker_finished,
    }
    return [timeline]


def main() -> None:
    global _ADMISSION_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-panel", required=True, type=Path)
    parser.add_argument("--family", choices=("causal_transformer",), required=True)
    parser.add_argument("--recovery-admission", required=True, type=Path)
    args = parser.parse_args()
    _ADMISSION_PATH = args.recovery_admission.resolve(strict=True)
    base._check_stage2_recovery = _check_stage2_recovery_r4
    # The review receipt records this adapter as the executable reviewer; all
    # other grading functions and scientific checks remain from the frozen base.
    old_module_path = base.__file__
    base.__file__ = str(Path(__file__).resolve())
    try:
        receipt = base.grade_cache(args.continuation_panel.resolve(strict=True), args.family)
    finally:
        base.__file__ = old_module_path
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "family",
                    "target_cache_path",
                    "target_metadata_sha256",
                    "source_binding_sha256",
                    "data_binding_sha256",
                    "split_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
