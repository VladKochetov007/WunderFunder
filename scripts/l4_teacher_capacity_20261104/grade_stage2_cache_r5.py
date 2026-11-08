"""Grade the R5 Transformer cache with frozen checks and the admitted amendment."""

from __future__ import annotations

import argparse
import copy
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

ADMISSION_SCHEMA = "wnn.l4-student-recovery-source-admission-r5.v1"
STATE_SCHEMA = "wnn.l4-teacher-capacity-continuation-recovery-r5.v1"
SUMMARY_SCHEMA = "wnn.l4-teacher-stage2-metadata-recovery-r5.v1"
AMENDMENT_SCHEMA = "wnn.root-storage-amendment.v1"
RELEASE_SCHEMA = "wnn.root-gpu-release.v1"
CONTROLLER_ID = "student_continuation_recovery_r5_controller"
FINALIZER_ID = "causal_transformer_stage2_compact_finalize_r5"
FINALIZER_CPU = tuple(range(32, 40))
CONTROLLER_CPU = tuple(range(64))
_ADMISSION_PATH: Path | None = None
_AMENDMENT_PATH: Path | None = None
_AMENDMENT: dict[str, Any] | None = None
_R5_CONTROLLER_CHILD: dict[str, Any] | None = None


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


def _check_amendment(
    panel: dict[str, Any], panel_hash: str, admission: dict[str, Any]
) -> tuple[dict[str, Any], Path, str]:
    if _AMENDMENT_PATH is None:
        raise RuntimeError("R5 storage amendment path was not supplied")
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    path = _AMENDMENT_PATH.resolve(strict=True)
    base._require(
        path.parent == control and path.name == "ROOT_STORAGE_AMENDMENT_R1.json",
        "R5 storage amendment must be the bound root control receipt",
    )
    amendment = base._read_json(path)
    state = base._read_json(Path(panel["paths"]["shared_state"]))
    expected = {
        "schema": AMENDMENT_SCHEMA,
        "status": "approved",
        "continuation_panel_sha256": panel_hash,
        "source_binding_sha256": admission["source_binding_sha256"],
        "original_output_cap_bytes": 6_500_000_000,
        "amended_output_cap_bytes": 7_100_000_000,
        "minimum_free_data2_bytes": 3_000_000_000,
        "publication_reserve_bytes": 500_000_000,
        "publication_reserve_scope": "collection only; no external new-wave writes during streaming",
        "hard_deadline_epoch": state["hard_deadline_epoch"],
        "max_scientific_fits": 12,
        "max_profiles": 24,
        "stream_init_required_free_bytes": 6_400_000_000,
        "stream_export_required_free_bytes": 3_100_000_000,
        "stream_finalize_required_free_bytes": 3_100_000_000,
        "transient_ram_export_cap_bytes": 3_600_000_000,
        "data2_cap_scope": "data2 outputs; temporary export banks charged to RAM cap",
    }
    base._require(
        all(amendment.get(key) == value for key, value in expected.items()),
        "R5 resource amendment differs from the frozen scope",
    )
    new_wave = Path(str(amendment.get("new_wave_complete_receipt_path", "")))
    _same_file(
        new_wave,
        amendment.get("new_wave_complete_receipt_path"),
        amendment.get("new_wave_complete_receipt_sha256"),
        "closed new-wave receipt",
    )
    allocations = amendment.get("export_allocations")
    expected_gpu2 = panel["families"]["causal_transformer"]["export_gpu_uuid"]
    base._require(
        isinstance(allocations, list)
        and len(allocations) == 2
        and all(isinstance(item, dict) for item in allocations)
        and [item.get("run") for item in allocations]
        == list(base.EXPECTED_RUNS["causal_transformer"])
        and [item.get("cpu_affinity") for item in allocations]
        == [list(range(16, 24)), list(range(24, 32))]
        and allocations[0].get("gpu_uuid") == expected_gpu2
        and allocations[1].get("gpu_uuid") != expected_gpu2
        and all(str(item.get("gpu_uuid", "")).startswith("GPU-") for item in allocations),
        "R5 parallel export GPU/CPU allocation differs",
    )
    export_root = Path(str(amendment.get("ram_export_root", ""))).resolve(strict=True)
    alias = Path(panel["paths"]["project_root"]) / "cache/l4_teacher_sequence_exports"
    base._require(
        alias.is_symlink() and alias.resolve(strict=True) == export_root,
        "R5 export path is not the admitted temporary export alias",
    )
    _same_file(
        path,
        admission.get("storage_amendment_receipt_path"),
        admission.get("storage_amendment_receipt_sha256"),
        "R5 storage amendment",
    )
    return amendment, path, file_sha256(path)


def _check_r5_stop_and_wide(
    panel: dict[str, Any], panel_hash: str, admission: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    stop_path = control / "STUDENT_RECOVERY_R4_STOP_R1.json"
    _same_file(
        stop_path,
        admission.get("r4_stop_receipt_path"),
        admission.get("r4_stop_receipt_sha256"),
        "R4 pre-target stop proof",
    )
    stop = base._read_json(stop_path)
    base._require(
        stop.get("schema") == "wnn.owned-controller-stop.v1"
        and stop.get("status") == "stopped_before_tf_target"
        and stop.get("tf_stage2_init_absent") is True
        and stop.get("tf_export_receipts_absent") is True
        and stop.get("no_scientific_fits_signaled") is True,
        "R4 stop proof does not establish the unchanged pre-target boundary",
    )
    runtime = Path(panel["paths"]["runtime_root"])
    r4_start_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_START.json"
    r4_start = base._read_json(r4_start_path)
    r4_ref = r4_start.get("controller_receipt", {})
    r4_controller_path = Path(str(r4_ref.get("path", ""))).resolve(strict=True)
    _same_file(
        r4_controller_path,
        stop.get("controller_receipt_path"),
        stop.get("controller_receipt_sha256"),
        "stopped R4 controller",
    )
    r4_controller = base._read_json(r4_controller_path)
    identity = panel["identity"]
    r4_owner = r4_controller.get("owner", {})
    r4_finished = r4_controller.get("finished_at_epoch")
    base._require(
        r4_start.get("schema") == "wnn.l4-teacher-capacity-continuation-recovery-r4.v1"
        and r4_start.get("status") == "running"
        and r4_start.get("panel_sha256") == panel_hash
        and r4_controller.get("status") != "running"
        and r4_controller.get("child") == r4_ref.get("child") == stop.get("controller_child")
        and isinstance(r4_finished, (int, float))
        and stop.get("created_epoch", 0) >= r4_finished
        and r4_owner.get("cell_id") == "student_continuation_recovery_r4_controller"
        and r4_owner.get("continuation_panel_sha256") == panel_hash
        and r4_owner.get("source_binding_sha256") == identity["source_binding_sha256"]
        and r4_owner.get("data_binding_sha256") == identity["data_binding_sha256"]
        and r4_owner.get("split_sha256") == identity["split_sha256"]
        and r4_owner.get("fit_members_sha256") == identity["fit_members_sha256"]
        and r4_owner.get("baseline_ancestry_sha256") == identity["baseline_ancestry_sha256"],
        "R4 stop proof does not match the actual terminal controller receipt",
    )
    tf_binding = panel["families"]["causal_transformer"]
    teacher_panel = base._read_json(Path(tf_binding["teacher_panel"]))
    teacher_cells = {cell["run_name"]: cell for cell in teacher_panel["cells"]}
    unchanged = stop.get("tf_fit_children_unchanged")
    base._require(
        isinstance(unchanged, list) and len(unchanged) == 2,
        "R4 stop proof omits a Transformer fit identity",
    )
    unchanged_by_cell = {record.get("cell_id"): record.get("identity") for record in unchanged}
    for run in base.EXPECTED_RUNS["causal_transformer"]:
        cell = teacher_cells[run]
        receipt = base._read_json(Path(tf_binding["fit_receipts"][run]))
        base._require(
            unchanged_by_cell.get(cell.get("cell_id")) == receipt.get("child"),
            f"{run}: teacher fit child identity differs from the R4 stop proof",
        )
    grade_path = Path(str(stop.get("wide_grade_receipt_path", ""))).resolve(strict=True)
    _same_file(
        grade_path,
        stop.get("wide_grade_receipt_path"),
        stop.get("wide_grade_receipt_sha256"),
        "adopted wide grade",
    )
    grade = base._read_json(grade_path)
    r5_adoption_path = runtime / "receipts/continuation/wide_gru_GRADE_ADOPTION_R5.json"
    adoption = base._read_json(r5_adoption_path)
    base._require(
        grade.get("status") == "complete"
        and grade.get("family") == "wide_gru"
        and grade.get("panel_sha256") == panel_hash
        and adoption.get("schema") == STATE_SCHEMA
        and adoption.get("status") == "complete"
        and adoption.get("grade_receipt_path") == str(grade_path)
        and adoption.get("grade_receipt_sha256") == file_sha256(grade_path)
        and adoption.get("manifest_path") == grade.get("manifest_path")
        and adoption.get("manifest_sha256") == grade.get("manifest_sha256")
        and adoption.get("report_path") == grade.get("report_path")
        and adoption.get("report_sha256") == grade.get("report_sha256"),
        "R5 wide grade adoption differs from the root-pinned completed grade",
    )
    for field in ("manifest", "report"):
        path_value = grade.get(f"{field}_path")
        sha_value = grade.get(f"{field}_sha256")
        _same_file(Path(str(path_value)), path_value, sha_value, f"wide canonical {field}")
    manifest = base._read_json(Path(grade["manifest_path"]))
    report = base._read_json(Path(grade["report_path"]))
    cache_review_path = Path(grade["cache_admission_receipt_path"]).resolve(strict=True)
    _same_file(
        cache_review_path,
        grade.get("cache_admission_receipt_path"),
        grade.get("cache_admission_receipt_sha256"),
        "wide independent cache review",
    )
    cache_review = base._read_json(cache_review_path)
    base._require(
        manifest.get("schema") == "wnn.l4tc-fixed-student-blend-inputs.v1"
        and manifest.get("split", {}).get("sha256") == base.SPLIT_SHA256
        and manifest.get("assembly", {}).get("family") == "wide_gru"
        and manifest.get("assembly", {}).get("continuation_panel_sha256") == panel_hash
        and report.get("schema") == "wnn.heterogeneous-ensemble-score.v1"
        and report.get("status") == "complete"
        and report.get("selected_rows") == 3_528_273
        and report.get("input_manifest_sha256") == file_sha256(Path(grade["manifest_path"]))
        and abs(report.get("member_scores", {}).get("TL3", {}).get("wp", 0) - 0.6923774245652725)
        <= 1e-7
        and cache_review.get("status")
        == "approved_stage2_cache_lineage_formula_not_numerically_recomputed"
        and cache_review.get("family") == "wide_gru"
        and adoption.get("cache_review_path") == str(cache_review_path)
        and adoption.get("cache_review_sha256") == file_sha256(cache_review_path),
        "adopted wide report, manifest, or cache-review evidence differs",
    )
    return stop, {
        "path": str(r5_adoption_path.resolve()),
        "sha256": file_sha256(r5_adoption_path),
        "grade": grade,
        "r4_controller_path": str(r4_controller_path),
        "r4_controller_sha256": file_sha256(r4_controller_path),
    }


def _check_r5_admission(
    panel: dict[str, Any], panel_path: Path, panel_hash: str
) -> tuple[dict[str, Any], Path, str, dict[str, Any], dict[str, Any]]:
    global _AMENDMENT_PATH, _AMENDMENT, _R5_CONTROLLER_CHILD
    if _ADMISSION_PATH is None:
        raise RuntimeError("R5 recovery admission path was not supplied")
    paths, identity = panel["paths"], panel["identity"]
    control = Path(paths["control_root"]).resolve(strict=True)
    admission_path = _ADMISSION_PATH.resolve(strict=True)
    base._require(
        admission_path.parent == control, "R5 admission must reside in the bound control root"
    )
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
        "approved R5 recovery admission does not bind this frozen panel/source",
    )
    amendment, amendment_path, amendment_sha = _check_amendment(panel, panel_hash, admission)
    stop, adoption = _check_r5_stop_and_wide(panel, panel_hash, admission)
    _AMENDMENT_PATH = amendment_path
    _AMENDMENT = amendment
    runtime = Path(paths["runtime_root"])
    start_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_START.json"
    start = base._read_json(start_path)
    base._require(
        start.get("schema") == STATE_SCHEMA
        and start.get("status") == "running"
        and start.get("panel_sha256") == panel_hash
        and start.get("recovery_source_binding_sha256") == admission["source_binding_sha256"]
        and start.get("recovery_admission_receipt_path") == str(admission_path)
        and start.get("recovery_admission_receipt_sha256") == file_sha256(admission_path),
        "R5 controller start does not bind its approved admission and source",
    )
    base._require(
        start.get("storage_amendment_receipt_path") == str(amendment_path)
        and start.get("storage_amendment_receipt_sha256") == amendment_sha
        and start.get("r4_stop_receipt_path") == str(control / "STUDENT_RECOVERY_R4_STOP_R1.json")
        and start.get("r4_stop_receipt_sha256")
        == file_sha256(control / "STUDENT_RECOVERY_R4_STOP_R1.json")
        and start.get("wide_grade_adoption", {}).get("grade_receipt_sha256")
        == stop.get("wide_grade_receipt_sha256"),
        "R5 controller start does not bind the amendment and R4 adoption boundary",
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
        and owner.get("storage_amendment_receipt_path") == str(amendment_path)
        and owner.get("storage_amendment_receipt_sha256") == amendment_sha
        and owner.get("r4_stop_receipt_path") == str(control / "STUDENT_RECOVERY_R4_STOP_R1.json")
        and owner.get("r4_stop_receipt_sha256")
        == file_sha256(control / "STUDENT_RECOVERY_R4_STOP_R1.json")
        and owner.get("baseline_ancestry_sha256") == identity["baseline_ancestry_sha256"]
        and script_path.name == "continue_students_recovery_r5.py"
        and file_sha256(script_path) == owner.get("recovery_entrypoint_sha256")
        and owner.get("recovery_entrypoint_path") == str(script_path.resolve())
        and owner.get("source_binding_sha256") == identity["source_binding_sha256"]
        and owner.get("data_binding_sha256") == identity["data_binding_sha256"]
        and owner.get("split_sha256") == identity["split_sha256"]
        and owner.get("fit_members_sha256") == identity["fit_members_sha256"]
        and owner.get("baseline_ancestry_sha256") == identity["baseline_ancestry_sha256"],
        "R5 controller process/source identity differs from its admitted receipt",
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
        "--storage-amendment",
        str(amendment_path),
        "--controller-receipt",
        str(controller_path),
        "--gpu2-release-receipt",
        str(Path(admission["gpu2_release_receipt_path"]).resolve()),
    ]
    base._require(
        taskset is not None
        and controller.get("command") == expected_controller
        and owner.get("command_sha256")
        == hashlib.sha256(
            json.dumps(expected_controller, separators=(",", ":")).encode()
        ).hexdigest(),
        "R5 controller command or 64-core affinity differs",
    )
    _R5_CONTROLLER_CHILD = controller.get("child")
    return (
        admission,
        admission_path,
        file_sha256(admission_path),
        amendment,
        {
            "start_path": start_path,
            "controller_path": controller_path,
            "controller": controller,
            "controller_script": script_path.resolve(),
            "amendment_path": amendment_path,
            "amendment_sha256": amendment_sha,
            "stop_path": control / "STUDENT_RECOVERY_R4_STOP_R1.json",
            "stop_sha256": file_sha256(control / "STUDENT_RECOVERY_R4_STOP_R1.json"),
            "adoption": adoption,
        },
    )


def _check_gpu2_release(
    panel: dict[str, Any], admission: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    release_path = Path(admission.get("gpu2_release_receipt_path", "")).resolve(strict=True)
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    tf_binding = panel["families"]["causal_transformer"]
    base._require(
        admission.get("gpu2_release_schema") == RELEASE_SCHEMA
        and admission.get("gpu2_uuid") == tf_binding["export_gpu_uuid"],
        "R5 admission GPU2 release schema/UUID differs from the TF exporter",
    )
    base._require(
        release_path.parent == control and release_path.name == "GPU2_RELEASE_R1.json",
        "R5 GPU2 release is not the root-approved control receipt",
    )
    release = base._read_json(release_path)
    fit_path = Path(str(release.get("objective_fit_receipt_path", "")))
    fit = base._read_json(fit_path)
    fit_owner = fit.get("owner", {})
    base._require(
        release.get("schema") == RELEASE_SCHEMA
        and release.get("status") == "released"
        and release.get("gpu_uuid") == admission["gpu2_uuid"]
        and _AMENDMENT is not None
        and _AMENDMENT["export_allocations"][0]["gpu_uuid"] == release.get("gpu_uuid")
        and _valid_sha(release.get("objective_panel_sha256"))
        and release.get("objective_fit_receipt_sha256") == file_sha256(fit_path)
        and fit.get("status") == "completed"
        and fit.get("exit_code") == 0
        and fit.get("timed_out") is False
        and fit_owner.get("cell_id") == "clip5_s1001"
        and fit_owner.get("panel_sha256", fit_owner.get("continuation_panel_sha256"))
        == release["objective_panel_sha256"],
        "R5 GPU2 release does not bind a terminal successful objective fit",
    )
    return release_path, release


_BASE_TRANSFORMER_EXPORT_CHECK = base._check_transformer_export_receipt


def _check_r5_transformer_export_receipt(
    receipt_path: Path,
    teacher: dict[str, Any],
    run: str,
    export: Path,
    project: Path,
    split_path: Path,
) -> dict[str, Any]:
    """Apply only R5's admitted per-seed CPU affinity to the frozen export check."""
    base._require(_AMENDMENT is not None, "R5 export allocation was not admitted")
    panel = copy.deepcopy(teacher["continuation_panel"])
    cells = teacher["teacher_panel"].get("cells", [])
    cell = next((item for item in cells if item.get("run_name") == run), None)
    base._require(cell is not None, f"{run}: fit panel cell is missing")
    allocation = next(
        (item for item in _AMENDMENT["export_allocations"] if item["run"] == run), None
    )
    base._require(allocation is not None, f"{run}: no R5 export allocation is admitted")
    panel["families"]["causal_transformer"]["export_cpu_affinity"] = allocation["cpu_affinity"]
    panel["families"]["causal_transformer"]["export_gpu_uuid"] = allocation["gpu_uuid"]
    teacher_copy = {**teacher, "continuation_panel": panel}
    result = _BASE_TRANSFORMER_EXPORT_CHECK(
        receipt_path, teacher_copy, run, export, project, split_path
    )
    receipt = base._read_json(receipt_path)
    owner = receipt.get("owner", {})
    base._require(
        receipt.get("gpu_key") == allocation["gpu_uuid"]
        and _R5_CONTROLLER_CHILD is not None
        and owner.get("pid") == _R5_CONTROLLER_CHILD.get("pid")
        and owner.get("birth") == _R5_CONTROLLER_CHILD.get("birth"),
        f"{run}: export GPU or R5 controller ownership differs",
    )
    return result


def _check_stage2_recovery_r5(
    panel: dict[str, Any],
    panel_path: Path,
    panel_hash: str,
    family: str,
    cache: Path,
    init_add_records: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Validate R5 recovery while leaving the base grade checks unchanged."""
    base._require(family == "causal_transformer", "R5 adapter is scoped to the pending TF family")
    admission, admission_path, admission_sha, amendment, lifecycle = _check_r5_admission(
        panel, panel_path, panel_hash
    )
    release_path, release = _check_gpu2_release(panel, admission)
    paths, identity = panel["paths"], panel["identity"]
    runtime = Path(paths["runtime_root"])
    receipts_root = runtime / "receipts" / "stage2"
    summary_path = receipts_root / "causal_transformer_metadata_recovery_r5.json"
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
        and summary.get("storage_amendment_receipt_path") == str(lifecycle["amendment_path"])
        and summary.get("storage_amendment_receipt_sha256") == lifecycle["amendment_sha256"]
        and summary.get("r4_stop_receipt_path") == str(lifecycle["stop_path"])
        and summary.get("r4_stop_receipt_sha256") == lifecycle["stop_sha256"]
        and summary.get("worker_receipt_path") == str(worker_path.resolve())
        and summary.get("worker_receipt_sha256") == file_sha256(worker_path)
        and summary.get("worker_child") == worker_child
        and entrypoint.name == "continue_students_recovery_r5.py"
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
        "--storage-amendment",
        str(lifecycle["amendment_path"]),
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
        and worker_owner.get("storage_amendment_receipt_path") == str(lifecycle["amendment_path"])
        and worker_owner.get("storage_amendment_receipt_sha256") == lifecycle["amendment_sha256"]
        and worker_owner.get("r4_stop_receipt_path") == str(lifecycle["stop_path"])
        and worker_owner.get("r4_stop_receipt_sha256") == lifecycle["stop_sha256"]
        and worker_owner.get("recovery_entrypoint_path") == str(entrypoint)
        and worker_owner.get("command_sha256")
        == hashlib.sha256(json.dumps(worker_command, separators=(",", ":")).encode()).hexdigest(),
        "R5 finalizer is not the exact terminal owned CPU command",
    )
    worker_started, worker_finished = (
        worker.get("started_at_epoch"),
        worker.get("finished_at_epoch"),
    )
    base._require(
        isinstance(worker_started, (int, float))
        and isinstance(worker_finished, (int, float))
        and worker_started <= worker_finished,
        "R5 finalizer timing is incomplete",
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
        "R5 summary differs from actual compact metadata/main/build/split/fit receipts",
    )
    retained = summary.get("retained_export_manifests")
    base._require(
        isinstance(retained, list) and len(retained) == 2,
        "R5 summary must retain both temporary export manifests",
    )
    retained_by_run = {item.get("run"): item for item in retained if isinstance(item, dict)}
    expected_runs = list(base.EXPECTED_RUNS[family])
    export_parent = Path(paths["project_root"]) / "cache/l4_teacher_sequence_exports"
    retained_root = Path(paths["control_root"]) / "retained_tf_export_manifests_r5"
    base._require(
        set(retained_by_run) == set(expected_runs), "R5 retained manifest run set differs"
    )
    for run in expected_runs:
        record = retained_by_run[run]
        source = export_parent / run / "manifest.json"
        destination = retained_root / run / "manifest.json"
        _same_file(
            source, record.get("source_path"), record.get("source_sha256"), f"{run} RAM manifest"
        )
        _same_file(
            destination,
            record.get("durable_path"),
            record.get("durable_sha256"),
            f"{run} durable manifest",
        )
        base._require(
            file_sha256(source) == file_sha256(destination)
            and record.get("bytes") == destination.stat().st_size,
            f"{run}: durable and temporary export manifests differ",
        )
    base._require(
        isinstance(summary.get("transient_ram_export_bytes"), int)
        and 0
        <= summary["transient_ram_export_bytes"]
        <= amendment["transient_ram_export_cap_bytes"],
        "R5 temporary teacher banks exceed their admitted RAM cap",
    )
    original_finalize = receipts_root / "causal_transformer_stage2_04.json"
    original_finalize_log = runtime / "logs/stage2/causal_transformer_stage2_04.log"
    base._require(
        not original_finalize.exists() and not original_finalize_log.exists(),
        "Transformer R1 finalization was attempted; preserve and stop",
    )
    add_times = [base._read_json(Path(item["path"])) for item in init_add_records]
    latest_add = max(float(item["finished_at_epoch"]) for item in add_times)
    base._require(latest_add <= worker_started, "R5 finalizer preceded an init/add receipt")
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
        "r5_controller_receipt_path": str(lifecycle["controller_path"]),
        "r5_controller_receipt_sha256": file_sha256(lifecycle["controller_path"]),
        "r5_start_receipt_path": str(lifecycle["start_path"].resolve()),
        "r5_start_receipt_sha256": file_sha256(lifecycle["start_path"]),
        "r4_stop_receipt_path": str(lifecycle["stop_path"].resolve()),
        "r4_stop_receipt_sha256": lifecycle["stop_sha256"],
        "storage_amendment_receipt_path": str(lifecycle["amendment_path"]),
        "storage_amendment_receipt_sha256": lifecycle["amendment_sha256"],
        "wide_grade_adoption_receipt_path": lifecycle["adoption"]["path"],
        "wide_grade_adoption_receipt_sha256": lifecycle["adoption"]["sha256"],
        "gpu2_release_receipt_path": str(release_path),
        "gpu2_release_receipt_sha256": file_sha256(release_path),
        "started_at_epoch": worker_started,
        "finished_at_epoch": worker_finished,
    }
    return [timeline]


def main() -> None:
    global _ADMISSION_PATH, _AMENDMENT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-panel", required=True, type=Path)
    parser.add_argument("--family", choices=("causal_transformer",), required=True)
    parser.add_argument("--recovery-admission", required=True, type=Path)
    parser.add_argument("--storage-amendment", required=True, type=Path)
    args = parser.parse_args()
    _ADMISSION_PATH = args.recovery_admission.resolve(strict=True)
    _AMENDMENT_PATH = args.storage_amendment.resolve(strict=True)
    panel_path = args.continuation_panel.resolve(strict=True)
    panel, panel_hash, _ = base._check_panel(panel_path, args.family)
    admission, admission_path, admission_sha, amendment, lifecycle = _check_r5_admission(
        panel, panel_path, panel_hash
    )
    old_recovery_check = base._check_stage2_recovery
    old_export_check = base._check_transformer_export_receipt
    old_write_json = base._write_json
    old_output_cap = base.OUTPUT_CAP_BYTES
    old_module_path = base.__file__
    review_path = Path(panel["families"][args.family]["cache_admission_receipt"]).resolve()

    def write_review(path: Path, value: dict[str, Any]) -> None:
        if Path(path).resolve() == review_path:
            value = dict(value)
            value["storage_amendment"] = {
                key: amendment.get(key)
                for key in (
                    "schema",
                    "status",
                    "continuation_panel_sha256",
                    "source_binding_sha256",
                    "original_output_cap_bytes",
                    "amended_output_cap_bytes",
                    "minimum_free_data2_bytes",
                    "publication_reserve_bytes",
                    "publication_reserve_scope",
                    "hard_deadline_epoch",
                    "max_scientific_fits",
                    "max_profiles",
                    "stream_init_required_free_bytes",
                    "stream_export_required_free_bytes",
                    "stream_finalize_required_free_bytes",
                    "transient_ram_export_cap_bytes",
                    "ram_export_root",
                    "export_allocations",
                    "data2_cap_scope",
                    "new_wave_complete_receipt_path",
                    "new_wave_complete_receipt_sha256",
                )
            }
            value["storage_amendment"]["path"] = str(_AMENDMENT_PATH)
            value["storage_amendment"]["sha256"] = file_sha256(_AMENDMENT_PATH)
            value["r4_stop_proof"] = {
                "path": str(lifecycle["stop_path"].resolve()),
                "sha256": lifecycle["stop_sha256"],
                "controller_receipt_path": str(lifecycle["r4_controller_path"]),
                "controller_receipt_sha256": lifecycle["r4_controller_sha256"],
            }
            wide_grade = lifecycle["adoption"]["grade"]
            value["wide_grade_adoption"] = {
                key: wide_grade[key]
                for key in (
                    "grade_receipt_path",
                    "grade_receipt_sha256",
                    "manifest_path",
                    "manifest_sha256",
                    "report_path",
                    "report_sha256",
                )
            }
        old_write_json(path, value)

    base._check_stage2_recovery = _check_stage2_recovery_r5
    base._check_transformer_export_receipt = _check_r5_transformer_export_receipt
    base._write_json = write_review
    base.OUTPUT_CAP_BYTES = int(amendment["amended_output_cap_bytes"])
    base.__file__ = str(Path(__file__).resolve())
    try:
        receipt = base.grade_cache(panel_path, args.family)
    finally:
        base._check_stage2_recovery = old_recovery_check
        base._check_transformer_export_receipt = old_export_check
        base._write_json = old_write_json
        base.OUTPUT_CAP_BYTES = old_output_cap
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
