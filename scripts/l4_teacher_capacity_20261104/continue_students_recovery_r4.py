"""Adopt the R3 wide students and continue the pending TF queue once."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.l4_teacher_capacity_20261104 import continue_students as cs  # noqa: E402
from scripts.l4_teacher_capacity_20261104.recover_students import (  # noqa: E402
    _expected_stage2_receipt,
    _validate_teacher_admission,
    _wait_cache_review,
    finalize_target_compact,
)
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_runtime import current_process_owner  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

ADMISSION_SCHEMA = "wnn.l4-student-recovery-source-admission-r4.v1"
STATE_SCHEMA = "wnn.l4-teacher-capacity-continuation-recovery-r4.v1"
RELEASE_SCHEMA = "wnn.root-gpu-release.v1"
CONTROLLER_ID = "student_continuation_recovery_r4_controller"
FINALIZER_ID = "causal_transformer_stage2_compact_finalize_r4"
FINALIZER_CPU = list(range(32, 40))
DISK_RESERVE = 6_500_000_000


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 << 20:
        raise ValueError(f"expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"preserve existing recovery evidence: {path}")
    atomic_write_json(path, value, replace=False)


def _admit(path: Path, panel: dict[str, Any], panel_hash: str, release_path: Path) -> tuple[dict, str]:
    admission = _read(path)
    identities = panel["identity"]
    expected = {
        "schema": ADMISSION_SCHEMA,
        "continuation_panel_sha256": panel_hash,
        "data_binding_sha256": identities["data_binding_sha256"],
        "split_sha256": identities["split_sha256"],
        "fit_members_sha256": identities["fit_members_sha256"],
        "baseline_ancestry_sha256": identities["baseline_ancestry_sha256"],
        "gpu2_release_receipt_path": str(release_path.resolve()),
        "gpu2_release_schema": RELEASE_SCHEMA,
        "gpu2_uuid": panel["families"]["causal_transformer"]["export_gpu_uuid"],
    }
    if not str(admission.get("status", "")).startswith("approved") or any(
        admission.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("R4 admission differs from the frozen panel or GPU2 gate")
    if len(str(admission.get("source_binding_sha256", ""))) != 64:
        raise ValueError("R4 admission lacks its pinned source binding")
    runtime = cs._runtime(panel)
    for key, filename in (
        ("r3_start_receipt", "STUDENT_CONTINUATION_RECOVERY_R3_START.json"),
        ("r3_failure_receipt", "STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json"),
    ):
        receipt = runtime / "receipts" / "continuation" / filename
        if not receipt.is_file() or receipt.is_symlink():
            raise FileNotFoundError(f"R4 predecessor receipt is absent: {receipt}")
        expected_record = {"path": str(receipt.resolve()), "sha256": file_sha256(receipt)}
        if admission.get(f"{key}_path") != expected_record["path"] or admission.get(
            f"{key}_sha256"
        ) != expected_record["sha256"]:
            raise ValueError(f"R4 admission does not pin the original R3 {key}")
    return admission, file_sha256(path)


def _r3_wide_failure(panel: dict, panel_hash: str) -> tuple[dict, dict]:
    runtime = cs._runtime(panel)
    start = _read(runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_START.json")
    failed = _read(runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json")
    if (
        start.get("schema") != "wnn.l4-teacher-capacity-continuation-recovery.v1"
        or start.get("status") != "running"
        or failed.get("schema") != start["schema"]
        or failed.get("status") != "failed"
        or start.get("panel_sha256") != panel_hash
        or failed.get("panel_sha256") != panel_hash
        or failed.get("error_type") != "ValueError"
        or "prediction sidecar identity or row order differs" not in failed.get("error", "")
        or failed.get("completed_families") != {}
    ):
        raise ValueError("R3 did not stop at the known wide reference-sidecar grading defect")
    for key, record in (("r3_start_receipt", start), ("r3_failure_receipt", failed)):
        if record.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]:
            raise ValueError(f"R3 {key} changed the original source identity")
    if (runtime / "receipts/continuation/wide_gru_RECOVERY_COMPLETE_R3.json").exists():
        raise ValueError("R3 wide completion already exists; preserve it and stop")
    return start, failed


def _existing_wide_cells(panel: dict, panel_hash: str, admission: dict) -> list[dict]:
    project = Path(panel["paths"]["project_root"])
    cells = sorted(
        [cell for cell in panel["students"]["cells"] if cell["family"] == "wide_gru"],
        key=lambda cell: cell["seed"],
    )
    pins = {item["cell_id"]: item for item in admission["wide_existing_student_artifacts"]}
    if (
        len(cells) != 2
        or len(admission["wide_existing_student_artifacts"]) != 2
        or {cell["seed"] for cell in cells} != {1001, 1002}
        or set(pins) != {cell["cell_id"] for cell in cells}
    ):
        raise ValueError("R4 admission must bind exactly the two completed wide student cells")
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("taskset is required to verify the original student receipts")
    for cell in cells:
        run, cell_id = cell["run_name"], cell["cell_id"]
        fit_log, fit_path = cs._receipt(panel, cell, "fit")
        export_cell = {**cell, "cell_id": f"{cell_id}_valid_export"}
        export_log, export_path = cs._receipt(panel, export_cell, "student_export")
        checkpoint = project / "artifacts" / "runs" / run / "model_ep1.pt"
        prediction = project / "cache" / f"pred_{run}_valid_scored.npy"
        required = {
            "fit_receipt_path": fit_path,
            "export_receipt_path": export_path,
            "checkpoint_path": checkpoint,
            "prediction_path": prediction,
        }
        pin = pins[cell_id]
        if any(pin[key] != str(path.resolve()) for key, path in required.items()):
            raise ValueError(f"{cell_id}: R4 artifact paths differ from the frozen panel")
        if any(not path.is_file() or path.is_symlink() for path in (*required.values(), fit_log, export_log)):
            raise FileNotFoundError(f"{cell_id}: preserve and inspect incomplete R3 outputs")
        if any(file_sha256(path) != pin[key.replace("_path", "_sha256")] for key, path in required.items()):
            raise ValueError(f"{cell_id}: a root-pinned R3 student artifact changed")
        for receipt_path, expected_cell in ((fit_path, cell_id), (export_path, f"{cell_id}_valid_export")):
            receipt = _read(receipt_path)
            owner = receipt.get("owner", {})
            if (
                receipt.get("status") != "completed"
                or receipt.get("exit_code") != 0
                or receipt.get("timed_out") is not False
                or owner.get("continuation_panel_sha256") != panel_hash
                or owner.get("cell_id") != expected_cell
                or owner.get("run_name") != run
                or owner.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
                or owner.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
            ):
                raise ValueError(f"{expected_cell}: original owned process is not a successful fit/export")
    return cells


def _gpu2_release(path: Path, admission: dict, deadline: int) -> dict:
    while not path.is_file():
        if cs._remaining(deadline) < 30:
            raise TimeoutError("GPU2 release did not arrive before the collection reserve")
        time.sleep(5)
    release = _read(path)
    fit_path = Path(release.get("objective_fit_receipt_path", ""))
    fit = _read(fit_path)
    if (
        release.get("schema") != RELEASE_SCHEMA
        or release.get("status") != "released"
        or release.get("gpu_uuid") != admission["gpu2_uuid"]
        or release.get("objective_fit_receipt_sha256") != file_sha256(fit_path)
        or fit.get("status") != "completed"
        or fit.get("exit_code") != 0
        or fit.get("timed_out") is not False
    ):
        raise ValueError("root GPU2 release does not bind its actual successful objective fit")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), **release}


def _wait_disk(panel: dict, deadline: int) -> None:
    mount = Path(panel["paths"]["disk_mount"])
    while shutil.disk_usage(mount).free < DISK_RESERVE:
        if cs._remaining(deadline) < 30:
            raise TimeoutError("data2 free space remained below the 6.5 GB TF target reserve")
        time.sleep(10)


def _finalize_worker(args: argparse.Namespace) -> int:
    panel_path = Path(args.panel)
    panel, panel_hash = cs.validate_panel(panel_path)
    admission_path, controller_path, worker_path = map(Path, (args.recovery_admission, args.controller_receipt, args.worker_receipt))
    admission, admission_sha = _admit(
        admission_path, panel, panel_hash, Path(args.gpu2_release_receipt)
    )
    source_sha = admission["source_binding_sha256"]
    current = current_process_owner()
    outer = _read(controller_path)
    if (outer.get("status") != "running" or outer.get("owner", {}).get("recovery_source_binding_sha256") != source_sha
            or outer.get("owner", {}).get("continuation_panel_sha256") != panel_hash):
        raise ValueError("R4 finalizer is not under its admitted controller")
    _, expected_worker = cs._receipt(panel, {"cell_id": FINALIZER_ID, "run_name": FINALIZER_ID}, "stage2")
    if worker_path.resolve() != expected_worker.resolve():
        raise ValueError("R4 finalizer receipt path differs from its frozen cell")
    process = _read(worker_path)
    command = [panel["paths"]["python"], str(Path(__file__).resolve()), "--worker-mode", "tf-finalize",
        "--panel", str(panel_path.resolve()), "--recovery-admission", str(admission_path.resolve()),
        "--controller-receipt", str(controller_path.resolve()), "--worker-receipt", str(worker_path.resolve()),
        "--gpu2-release-receipt", str(Path(args.gpu2_release_receipt).resolve())]
    expected_command = [shutil.which("taskset"), "-c", ",".join(map(str, FINALIZER_CPU)), *command]
    owner = process.get("owner", {})
    if (process.get("status") != "running" or process.get("command") != expected_command
            or process.get("child") != current or owner.get("cell_id") != FINALIZER_ID
            or owner.get("continuation_panel_sha256") != panel_hash
            or owner.get("recovery_source_binding_sha256") != source_sha
            or owner.get("recovery_admission_receipt_sha256") != admission_sha):
        raise ValueError("R4 finalizer is not the exact owned child")
    result = finalize_target_compact(Path(panel["families"]["causal_transformer"]["target_cache"]), "causal_transformer")
    print(json.dumps(result, allow_nan=False))
    return 0


def _run_finalizer(
    panel: dict, panel_path: Path, panel_hash: str, source_sha: str,
    admission_path: Path, admission_sha: str, controller_path: Path,
    release_path: Path, deadline: int
) -> tuple[dict, Path]:
    cell = {"cell_id": FINALIZER_ID, "run_name": FINALIZER_ID}
    log, receipt_path = cs._receipt(panel, cell, "stage2")
    if log.exists() or receipt_path.exists():
        raise FileExistsError("preserve any prior R4 finalizer attempt")
    command = [panel["paths"]["python"], str(Path(__file__).resolve()), "--worker-mode", "tf-finalize",
        "--panel", str(panel_path.resolve()), "--recovery-admission", str(admission_path.resolve()),
        "--controller-receipt", str(controller_path.resolve()), "--worker-receipt", str(receipt_path.resolve()),
        "--gpu2-release-receipt", str(release_path.resolve())]
    cell.update({"family": "causal_transformer", "cpu_affinity": FINALIZER_CPU, "cpu_threads": 8})
    _wait_disk(panel, deadline)
    receipt = cs._run_owned(
        panel, panel_hash, cell, "stage2", command, deadline,
        output_dir=Path(panel["families"]["causal_transformer"]["target_cache"]),
        gpu_uuid=None, stage=3, source_root=ROOT,
        owner_extra={"recovery_source_binding_sha256": source_sha,
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "recovery_entrypoint_path": str(Path(__file__).resolve()),
            "recovery_entrypoint_sha256": file_sha256(Path(__file__).resolve())})
    if receipt.get("status") != "completed" or receipt.get("exit_code") != 0 or receipt.get("timed_out") is not False:
        raise RuntimeError("R4 TF compact finalizer failed; preserve the owned receipt")
    output = Path(panel["families"]["causal_transformer"]["target_cache"])
    metadata, build = _read(output / "metadata.json"), _read(output / "stage2_build_receipt.json")
    adds = []
    for index, action, run in [(1, "init", None), *[(i, "add", run) for i, run in enumerate(cs.EXPECTED_RUNS["causal_transformer"], 2)]]:
        _, path, _ = _expected_stage2_receipt(panel, panel_hash, "causal_transformer", index,
            cs._stage2_args(panel, "causal_transformer", action, run), status="completed", exit_code=0)
        adds.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    summary = {"schema": "wnn.l4-teacher-stage2-metadata-recovery-r4.v1", "status": "compact_finalized_and_promoted",
        "family": "causal_transformer", "continuation_panel_sha256": panel_hash,
        "recovery_source_binding_sha256": source_sha, "target_cache_path": str(output.resolve()),
        "target_metadata_sha256": file_sha256(output / "metadata.json"),
        "target_contract_sha256": hashlib.sha256(json.dumps(metadata["target_contract"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "stage2_build_receipt_path": str((output / "stage2_build_receipt.json").resolve()),
        "stage2_build_receipt_sha256": file_sha256(output / "stage2_build_receipt.json"),
        "stage2_init_add_receipts": adds, "worker_receipt_path": str(receipt_path.resolve()),
        "worker_receipt_sha256": file_sha256(receipt_path), "worker_child": receipt.get("child"),
        "build_state_schema": build.get("schema")}
    _write_once(cs._runtime(panel) / "receipts/stage2/causal_transformer_metadata_recovery_r4.json", summary)
    return summary, receipt_path


def _controller_context(args: argparse.Namespace, panel: dict, panel_hash: str,
                         admission: dict, admission_sha: str) -> tuple[dict, str]:
    receipt = _read(Path(args.controller_receipt))
    command = [shutil.which("taskset"), "-c", ",".join(map(str, range(64))), panel["paths"]["python"],
        str(Path(__file__).resolve()), "--controller-mode", "continue", "--panel", str(Path(args.panel).resolve()),
        "--recovery-admission", str(Path(args.recovery_admission).resolve()), "--controller-receipt", str(Path(args.controller_receipt).resolve()),
        "--gpu2-release-receipt", str(Path(args.gpu2_release_receipt).resolve())]
    owner, current = receipt.get("owner", {}), current_process_owner()
    command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    if (receipt.get("status") != "running" or receipt.get("child") != current or receipt.get("command") != command
            or receipt.get("command_sha256") != command_sha
            or owner.get("continuation_panel_sha256") != panel_hash
            or owner.get("recovery_source_binding_sha256") != admission["source_binding_sha256"]
            or owner.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
            or owner.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
            or owner.get("split_sha256") != panel["identity"]["split_sha256"]
            or owner.get("fit_members_sha256") != panel["identity"]["fit_members_sha256"]
            or owner.get("baseline_ancestry_sha256") != panel["identity"]["baseline_ancestry_sha256"]
            or owner.get("recovery_admission_receipt_path") != str(Path(args.recovery_admission).resolve())
            or owner.get("recovery_admission_receipt_sha256") != admission_sha
            or owner.get("recovery_entrypoint_path") != str(Path(__file__).resolve())
            or owner.get("recovery_entrypoint_sha256") != file_sha256(Path(__file__).resolve())
            or owner.get("cell_id") != CONTROLLER_ID or owner.get("run_name") != CONTROLLER_ID
            or owner.get("family") != "continuation"):
        raise ValueError("R4 continuation is not running under its exact owned controller receipt")
    return {"path": str(Path(args.controller_receipt).resolve()), "child": current}, command_sha


def _run(args: argparse.Namespace) -> dict[str, Any]:
    panel, panel_hash = cs.validate_panel(Path(args.panel))
    admission, admission_sha = _admit(Path(args.recovery_admission), panel, panel_hash, Path(args.gpu2_release_receipt))
    controller, command_sha = _controller_context(args, panel, panel_hash, admission, admission_sha)
    start_path = cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_START.json"
    failed_path = cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_FAILED.json"
    complete_path = cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_COMPLETE.json"
    if any(path.exists() for path in (start_path, failed_path, complete_path)):
        raise FileExistsError("R4 lifecycle already has evidence; do not restart")
    _r3_wide_failure(panel, panel_hash)
    deadline = cs._deadline(panel)
    base = {"schema": STATE_SCHEMA, "panel_sha256": panel_hash,
        "source_binding_sha256": panel["identity"]["source_binding_sha256"],
        "recovery_source_binding_sha256": admission["source_binding_sha256"],
        "recovery_admission_receipt_path": str(Path(args.recovery_admission).resolve()),
        "recovery_admission_receipt_sha256": admission_sha, "controller_receipt": controller,
        "controller_command_sha256": command_sha,
        "r3_start_receipt_path": str((cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_START.json").resolve()),
        "r3_start_receipt_sha256": file_sha256(cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_START.json"),
        "r3_failure_receipt_path": str((cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json").resolve()),
        "r3_failure_receipt_sha256": file_sha256(cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json"),
        "hard_deadline_epoch": deadline}
    _write_once(start_path, {**base, "status": "running", "started_epoch": time.time()})
    families: dict[str, Any] = {}
    phase = "wide_existing_grade"
    try:
        _, wide_review_path = _wait_cache_review(panel, panel_hash, "wide_gru", deadline)
        wide_cells = _existing_wide_cells(panel, panel_hash, admission)
        wide_grade = cs._grade_family(panel, panel_hash, "wide_gru", wide_cells, deadline)
        wide_receipt_path = wide_grade / "grade_receipt.json"
        wide_receipt = _read(wide_receipt_path)
        if wide_receipt.get("status") != "complete":
            raise ValueError("wide canonical grade did not complete")
        families["wide_gru"] = {"family": "wide_gru", "grade_receipt_path": str(wide_receipt_path.resolve()),
            "grade_receipt_sha256": file_sha256(wide_receipt_path), "grade_status": "complete",
            "candidate_rows": wide_receipt["candidate_rows"], "existing_student_artifacts": admission["wide_existing_student_artifacts"],
            "cache_review_path": str(wide_review_path.resolve()), "cache_review_sha256": file_sha256(wide_review_path)}
        _write_once(cs._runtime(panel) / "receipts/continuation/wide_gru_GRADE_RECOVERY_R4.json",
            {"schema": STATE_SCHEMA, "status": "complete", **base, **families["wide_gru"]})

        phase = "tf_teacher_wait_and_target"
        family = "causal_transformer"
        rows = cs._wait_teachers(panel, family, deadline)
        _validate_teacher_admission(panel, family, rows)
        _wait_disk(panel, deadline)
        _gpu2_release(Path(args.gpu2_release_receipt), admission, deadline)
        output = Path(panel["families"][family]["target_cache"])
        if output.exists() or output.is_symlink() or cs.partial_root(output).exists():
            raise FileExistsError("TF target already has output/partial; preserve and stop")
        cs._run_stage2(panel, panel_hash, family, cs._stage2_args(panel, family, "init"), deadline, 1)
        for index, row in enumerate(rows, start=2):
            _wait_disk(panel, deadline)
            _gpu2_release(Path(args.gpu2_release_receipt), admission, deadline)
            export = cs._teacher_export(panel, panel_hash, family, row, deadline, index)
            stage_args = cs._stage2_args(panel, family, "add", row["run"])
            stage_args.extend(["--sequence-export", str(export), "--teacher-panel", str(row["teacher_panel"])])
            cs._run_stage2(panel, panel_hash, family, stage_args, deadline, index)
        phase = "tf_compact_finalize_and_review"
        summary, finalizer_receipt = _run_finalizer(
            panel, Path(args.panel), panel_hash, admission["source_binding_sha256"],
            Path(args.recovery_admission), admission_sha, Path(args.controller_receipt),
            Path(args.gpu2_release_receipt), deadline
        )
        review, review_path = _wait_cache_review(panel, panel_hash, family, deadline)
        phase = "tf_students_and_grade"
        student = cs._student_family(panel, panel_hash, family, deadline)
        grade_path = Path(student["grade_receipt_path"])
        families[family] = {"family": family, "target_cache_path": summary["target_cache_path"],
            "target_metadata_sha256": summary["target_metadata_sha256"], "target_contract_sha256": summary["target_contract_sha256"],
            "finalizer_receipt_path": str(finalizer_receipt.resolve()), "finalizer_receipt_sha256": file_sha256(finalizer_receipt),
            "cache_review_path": str(review_path.resolve()), "cache_review_sha256": file_sha256(review_path),
            "profiles": student["profiles"], "fits": student["fit_artifacts"], "valid_exports": student["valid_exports"],
            "grade_receipt_path": str(grade_path.resolve()), "grade_receipt_sha256": file_sha256(grade_path),
            "grade_status": student["grade_status"], "candidate_rows": student["grade_candidate_rows"]}
        _write_once(cs._runtime(panel) / "receipts/continuation/causal_transformer_RECOVERY_COMPLETE_R4.json",
            {"schema": STATE_SCHEMA, "status": "complete", **base, **families[family]})
    except Exception as error:
        _write_once(failed_path, {**base, "status": "failed", "phase": phase,
            "error_type": type(error).__name__, "error": str(error), "completed_families": families})
        raise
    complete = {**base, "status": "complete", "finished_epoch": time.time(), "families": families,
        "all_family_grades_complete": all(value["grade_status"] == "complete" for value in families.values()),
        "checkpoint_endpoint_epoch": 1, "fixed_weights": {"TL3": 0.9, "candidate": 0.1}}
    _write_once(complete_path, complete)
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--controller-mode", choices=("continue",))
    modes.add_argument("--worker-mode", choices=("tf-finalize",))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--recovery-admission", type=Path, required=True)
    parser.add_argument("--controller-receipt", type=Path, required=True)
    parser.add_argument("--gpu2-release-receipt", type=Path)
    parser.add_argument("--worker-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.worker_mode:
        if args.worker_receipt is None or args.gpu2_release_receipt is None:
            parser.error("worker mode requires --worker-receipt and --gpu2-release-receipt")
        return _finalize_worker(args)
    if args.gpu2_release_receipt is None or args.worker_receipt is not None:
        parser.error("controller mode requires the GPU2 release receipt and no worker receipt")
    print(json.dumps(_run(args), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        sys.stderr.write(f"R4 student continuation stopped: {type(error).__name__}: {error}\n")
        raise SystemExit(2) from error
