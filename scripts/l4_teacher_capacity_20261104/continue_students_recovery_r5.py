"""Adopt the existing wide grade and continue the pending TF queue once."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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
    _stage2_args,
    _validate_teacher_admission,
    _wait_cache_review,
    finalize_target_compact,
)
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_runtime import current_process_owner  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

ADMISSION_SCHEMA = "wnn.l4-student-recovery-source-admission-r5.v1"
STATE_SCHEMA = "wnn.l4-teacher-capacity-continuation-recovery-r5.v1"
AMENDMENT_SCHEMA = "wnn.root-storage-amendment.v1"
RELEASE_SCHEMA = "wnn.root-gpu-release.v1"
CONTROLLER_ID = "student_continuation_recovery_r5_controller"
FINALIZER_ID = "causal_transformer_stage2_compact_finalize_r5"
FINALIZER_CPU = list(range(32, 40))
INIT_RESERVE = 6_400_000_000  # Above the frozen producer's 6.349GB pre-init guard.
EXPORT_RESERVE = 3_100_000_000  # Prediction banks are charged to the separate RAM cap.
FINALIZE_RESERVE = 3_100_000_000
OUTPUT_CAP = 7_100_000_000
MIN_FREE = 3_000_000_000
PUBLICATION_RESERVE = 500_000_000
TRANSIENT_RAM_EXPORT_CAP = 3_600_000_000
R4_STOP_NAME = "STUDENT_RECOVERY_R4_STOP_R1.json"
STORAGE_AMENDMENT_NAME = "ROOT_STORAGE_AMENDMENT_R1.json"


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


def _same_file(path: Path, expected_path: Any, expected_sha: Any, label: str) -> None:
    if path.is_symlink() or not path.is_file() or expected_path != str(path.resolve()):
        raise ValueError(f"{label} path differs or is not a regular file: {path}")
    if file_sha256(path) != expected_sha:
        raise ValueError(f"{label} SHA256 differs: {path}")


def _admit(
    path: Path,
    panel: dict[str, Any],
    panel_hash: str,
    amendment_path: Path,
    release_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    admission = _read(path)
    identities = panel["identity"]
    expected = {
        "schema": ADMISSION_SCHEMA,
        "continuation_panel_sha256": panel_hash,
        "source_binding_sha256": admission.get("source_binding_sha256"),
        "data_binding_sha256": identities["data_binding_sha256"],
        "split_sha256": identities["split_sha256"],
        "fit_members_sha256": identities["fit_members_sha256"],
        "baseline_ancestry_sha256": identities["baseline_ancestry_sha256"],
        "gpu2_release_receipt_path": str(release_path.resolve()),
        "gpu2_release_schema": RELEASE_SCHEMA,
        "gpu2_uuid": panel["families"]["causal_transformer"]["export_gpu_uuid"],
        "r4_stop_receipt_path": str(
            (Path(panel["paths"]["control_root"]) / R4_STOP_NAME).resolve()
        ),
        "storage_amendment_receipt_path": str(amendment_path.resolve()),
    }
    if not str(admission.get("status", "")).startswith("approved") or any(
        admission.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("R5 admission differs from the frozen panel, amendment, or GPU2 gate")
    source_sha = admission.get("source_binding_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("R5 admission lacks its pinned source binding")
    stop_path = Path(admission["r4_stop_receipt_path"])
    stop_sha = admission.get("r4_stop_receipt_sha256")
    _same_file(stop_path, admission["r4_stop_receipt_path"], stop_sha, "R4 stop proof")
    amendment, amendment_sha = _check_storage_amendment(
        amendment_path, panel, panel_hash, source_sha
    )
    _same_file(
        amendment_path,
        admission.get("storage_amendment_receipt_path"),
        admission.get("storage_amendment_receipt_sha256"),
        "approved storage amendment",
    )
    return admission, file_sha256(path), amendment, amendment_sha, _read(stop_path), stop_sha


def _check_storage_amendment(
    path: Path, panel: dict[str, Any], panel_hash: str, source_sha: str
) -> tuple[dict[str, Any], str]:
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    if path.resolve(strict=True) != control / STORAGE_AMENDMENT_NAME:
        raise ValueError("storage amendment path is not the root-issued control receipt")
    amendment = _read(path)
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "research/experiment_drafts/l4-teacher-capacity-20261104/CONTRACT_R1.json"
    )
    contract = _read(contract_path)
    state = _read(Path(panel["paths"]["shared_state"]))
    expected = {
        "schema": AMENDMENT_SCHEMA,
        "status": "approved",
        "continuation_panel_sha256": panel_hash,
        "source_binding_sha256": source_sha,
        "original_output_cap_bytes": 6_500_000_000,
        "amended_output_cap_bytes": OUTPUT_CAP,
        "minimum_free_data2_bytes": MIN_FREE,
        "publication_reserve_bytes": PUBLICATION_RESERVE,
        "publication_reserve_scope": "collection only; no external new-wave writes during streaming",
        "hard_deadline_epoch": state.get("hard_deadline_epoch"),
        "max_scientific_fits": contract["budget"]["new_scientific_fits_total"],
        "max_profiles": contract["budget"]["max_profiles"],
        "stream_init_required_free_bytes": INIT_RESERVE,
        "stream_export_required_free_bytes": EXPORT_RESERVE,
        "stream_finalize_required_free_bytes": FINALIZE_RESERVE,
        "transient_ram_export_cap_bytes": TRANSIENT_RAM_EXPORT_CAP,
        "data2_cap_scope": "data2 outputs; temporary export banks charged to RAM cap",
    }
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise ValueError("R5 storage amendment differs from its exact scope or resource limits")
    _same_file(
        Path(amendment.get("new_wave_complete_receipt_path", "")),
        amendment.get("new_wave_complete_receipt_path"),
        amendment.get("new_wave_complete_receipt_sha256"),
        "closed new-wave receipt",
    )
    allocations = amendment.get("export_allocations")
    expected_runs = list(cs.EXPECTED_RUNS["causal_transformer"])
    expected_affinities = [list(range(16, 24)), list(range(24, 32))]
    base_gpu = panel["families"]["causal_transformer"]["export_gpu_uuid"]
    if (
        not isinstance(allocations, list)
        or len(allocations) != 2
        or any(not isinstance(item, dict) for item in allocations)
        or [item.get("run") for item in allocations] != expected_runs
        or [item.get("cpu_affinity") for item in allocations] != expected_affinities
        or allocations[0].get("gpu_uuid") != base_gpu
        or allocations[1].get("gpu_uuid") == base_gpu
        or any(not str(item.get("gpu_uuid", "")).startswith("GPU-") for item in allocations)
    ):
        raise ValueError("R5 export allocation amendment differs from the two-adapter GPU/CPU plan")
    export_root = Path(str(amendment.get("ram_export_root", ""))).resolve(strict=True)
    export_alias = Path(panel["paths"]["project_root"]) / "cache/l4_teacher_sequence_exports"
    if not export_alias.is_symlink() or export_alias.resolve(strict=True) != export_root:
        raise ValueError("R5 RAM export alias does not resolve to its admitted tmpfs root")
    return amendment, file_sha256(path)


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


def _verify_r4_stop(
    panel: dict[str, Any],
    panel_hash: str,
    stop: dict[str, Any],
    stop_sha: str,
) -> dict[str, Any]:
    """Prove R4's waiting controller stopped without touching teacher fits."""
    runtime = cs._runtime(panel)
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    stop_path = control / R4_STOP_NAME
    if (
        stop.get("schema") != "wnn.owned-controller-stop.v1"
        or stop.get("status") != "stopped_before_tf_target"
        or stop.get("tf_stage2_init_absent") is not True
        or stop.get("tf_export_receipts_absent") is not True
        or stop.get("no_scientific_fits_signaled") is not True
        or stop.get("controller_receipt_path") is None
        or not isinstance(stop.get("created_epoch"), (float, int))
    ):
        raise ValueError("R4 stop proof does not certify a safe pre-target stop")
    r4_start_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R4_START.json"
    r4_start = _read(r4_start_path)
    controller_ref = r4_start.get("controller_receipt", {})
    controller_path = Path(str(controller_ref.get("path", ""))).resolve(strict=True)
    _same_file(
        controller_path,
        stop.get("controller_receipt_path"),
        stop.get("controller_receipt_sha256"),
        "terminal R4 controller receipt",
    )
    controller = _read(controller_path)
    identity = panel["identity"]
    finished = controller.get("finished_at_epoch")
    if (
        r4_start.get("schema") != "wnn.l4-teacher-capacity-continuation-recovery-r4.v1"
        or r4_start.get("status") != "running"
        or r4_start.get("panel_sha256") != panel_hash
        or controller.get("schema") != "wnn.multistage-owned-process.v1"
        or controller.get("status") == "running"
        or controller.get("child") != controller_ref.get("child")
        or controller.get("child") != stop.get("controller_child")
        or not isinstance(finished, (int, float))
        or stop["created_epoch"] < finished
        or controller.get("owner", {}).get("cell_id")
        != "student_continuation_recovery_r4_controller"
        or controller.get("owner", {}).get("continuation_panel_sha256") != panel_hash
        or controller.get("owner", {}).get("source_binding_sha256")
        != identity["source_binding_sha256"]
        or controller.get("owner", {}).get("data_binding_sha256") != identity["data_binding_sha256"]
        or controller.get("owner", {}).get("split_sha256") != identity["split_sha256"]
        or controller.get("owner", {}).get("fit_members_sha256") != identity["fit_members_sha256"]
        or controller.get("owner", {}).get("baseline_ancestry_sha256")
        != identity["baseline_ancestry_sha256"]
    ):
        raise ValueError("R4 stop proof does not bind its exact terminal owned controller")
    _r3_wide_failure(panel, panel_hash)

    fits = cs._teacher_cells(panel, "causal_transformer")
    unchanged = stop.get("tf_fit_children_unchanged")
    if not isinstance(unchanged, list) or len(unchanged) != len(fits):
        raise ValueError("R4 stop proof omits one or more live Transformer fit identities")
    actual = {row["cell_id"]: row for row in fits}
    for record in unchanged:
        row = actual.pop(record.get("cell_id", ""), None)
        if row is None:
            raise ValueError("R4 stop proof names an unknown Transformer teacher")
        fit = _read(row["receipt"])
        if fit.get("child") != record.get("identity"):
            raise ValueError("Transformer teacher child identity changed across the R4 stop")
    if actual:
        raise ValueError("R4 stop proof omitted a Transformer teacher")

    for index, row in enumerate(fits, start=2):
        export_cell = {
            **row["teacher_cell"],
            "cell_id": f"causal_transformer_export_{index:02d}_{row['run']}",
        }
        export_log, export_receipt = cs._receipt(panel, export_cell, "teacher_export")
        if export_log.exists() or export_receipt.exists():
            raise FileExistsError(
                "R4 created a Transformer export; preserve and audit its artifacts"
            )
    target = Path(panel["families"]["causal_transformer"]["target_cache"])
    stage2_receipts = [
        cs._receipt(panel, {"cell_id": f"causal_transformer_stage2_{index:02d}"}, "stage2")
        for index in range(1, 6)
    ]
    if (
        target.exists()
        or target.is_symlink()
        or cs.partial_root(target).exists()
        or any(log.exists() or receipt.exists() for log, receipt in stage2_receipts)
    ):
        raise FileExistsError("Transformer target work exists despite the R4 pre-target stop proof")
    return {
        "path": str(stop_path.resolve()),
        "sha256": stop_sha,
        "controller_path": str(controller_path),
        "controller_sha256": file_sha256(controller_path),
        "controller_child": controller.get("child"),
        "tf_fit_children_unchanged": unchanged,
    }


def _adopt_wide_grade(
    panel: dict[str, Any], panel_hash: str, stop: dict[str, Any]
) -> dict[str, Any]:
    """Bind the existing completed wide grade without rerunning its scorer."""
    grade_path = Path(stop.get("wide_grade_receipt_path", "")).resolve(strict=True)
    grade_sha = stop.get("wide_grade_receipt_sha256")
    _same_file(grade_path, stop.get("wide_grade_receipt_path"), grade_sha, "completed wide grade")
    grade = _read(grade_path)
    if (
        grade.get("schema") != "wnn.l4tc-fixed-student-grade.v1"
        or grade.get("status") != "complete"
        or grade.get("family") != "wide_gru"
        or grade.get("panel_sha256") != panel_hash
        or Path(grade.get("target_cache_path", "")).resolve()
        != Path(panel["families"]["wide_gru"]["target_cache"]).resolve()
    ):
        raise ValueError("pinned wide grade receipt is not the completed grade for this panel")
    report_path = Path(grade.get("report_path", "")).resolve(strict=True)
    manifest_path = Path(grade.get("manifest_path", "")).resolve(strict=True)
    _same_file(
        report_path, grade.get("report_path"), grade.get("report_sha256"), "wide grade report"
    )
    _same_file(
        manifest_path,
        grade.get("manifest_path"),
        grade.get("manifest_sha256"),
        "wide grade manifest",
    )
    report, manifest = _read(report_path), _read(manifest_path)
    if (
        report.get("schema") != "wnn.heterogeneous-ensemble-score.v1"
        or report.get("status") != "complete"
        or report.get("selected_rows") != 3_528_273
        or report.get("input_manifest_sha256") != file_sha256(manifest_path)
        or manifest.get("schema") != "wnn.l4tc-fixed-student-blend-inputs.v1"
        or manifest.get("assembly", {}).get("family") != "wide_gru"
        or manifest.get("assembly", {}).get("continuation_panel_sha256") != panel_hash
    ):
        raise ValueError("wide grade report/manifest do not bind the frozen canonical evaluation")
    r4_summary_path = cs._runtime(panel) / "receipts/continuation/wide_gru_GRADE_RECOVERY_R4.json"
    r4_summary = _read(r4_summary_path)
    if (
        r4_summary.get("schema") != "wnn.l4-teacher-capacity-continuation-recovery-r4.v1"
        or r4_summary.get("status") != "complete"
        or r4_summary.get("grade_receipt_path") != str(grade_path)
        or r4_summary.get("grade_receipt_sha256") != grade_sha
    ):
        raise ValueError("R4 wide adoption summary differs from the pinned canonical grade")
    review_path = Path(grade.get("cache_admission_receipt_path", "")).resolve(strict=True)
    _same_file(
        review_path,
        grade.get("cache_admission_receipt_path"),
        grade.get("cache_admission_receipt_sha256"),
        "wide cache review",
    )
    review = _read(review_path)
    if (
        review.get("status") != "approved_stage2_cache_lineage_formula_not_numerically_recomputed"
        or review.get("family") != "wide_gru"
    ):
        raise ValueError("wide target cache has no approved independent review")
    return {
        "grade_receipt_path": str(grade_path),
        "grade_receipt_sha256": grade_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "r4_grade_summary_path": str(r4_summary_path.resolve()),
        "r4_grade_summary_sha256": file_sha256(r4_summary_path),
        "cache_review_path": str(review_path),
        "cache_review_sha256": file_sha256(review_path),
        "candidate_rows": grade.get("candidate_rows"),
    }


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


def _wait_disk(panel: dict, deadline: int, required_free: int) -> None:
    mount = Path(panel["paths"]["disk_mount"])
    while shutil.disk_usage(mount).free < required_free:
        if cs._remaining(deadline) < 30:
            raise TimeoutError(
                f"data2 free space remained below the {required_free}-byte phase reserve"
            )
        time.sleep(10)


def _parallel_teacher_exports(
    panel: dict[str, Any],
    panel_hash: str,
    rows: list[dict[str, Any]],
    amendment: dict[str, Any],
    deadline: int,
) -> tuple[dict[str, Path], int]:
    """Export the two fixed teachers concurrently on the admitted GPUs."""
    allocations = {item["run"]: item for item in amendment["export_allocations"]}
    if [int(row["teacher_cell"]["seed"]) for row in rows] != [1001, 1002]:
        raise ValueError("Transformer export order differs from the two frozen seeds")

    def export(row: dict[str, Any], index: int) -> Path:
        allocation = allocations[row["run"]]
        worker_panel = copy.deepcopy(panel)
        binding = worker_panel["families"]["causal_transformer"]
        binding["export_cpu_affinity"] = list(allocation["cpu_affinity"])
        binding["export_cpu_threads"] = len(allocation["cpu_affinity"])
        binding["export_gpu_uuid"] = allocation["gpu_uuid"]
        return cs._teacher_export(
            worker_panel, panel_hash, "causal_transformer", row, deadline, index
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(export, row, index) for index, row in enumerate(rows, start=2)]
        exports = {row["run"]: future.result() for row, future in zip(rows, futures, strict=True)}
    total_bytes = 0
    for export_root in exports.values():
        for path in export_root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"temporary teacher export contains a symlink: {path}")
            if path.is_file():
                total_bytes += path.stat().st_size
    if total_bytes > TRANSIENT_RAM_EXPORT_CAP:
        raise ValueError(f"temporary teacher exports exceeded the admitted RAM cap: {total_bytes}")
    return exports, total_bytes


def _retain_export_manifests(
    panel: dict[str, Any], rows: list[dict[str, Any]], exports: dict[str, Path]
) -> list[dict[str, Any]]:
    """Keep durable copies of the two small fit-only export manifests."""
    root = Path(panel["paths"]["control_root"]) / "retained_tf_export_manifests_r5"
    if root.is_symlink():
        raise ValueError("durable manifest root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    retained = []
    for row in rows:
        source = exports[row["run"]] / "manifest.json"
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"fit-only export manifest is missing: {source}")
        destination_root = root / row["run"]
        destination = destination_root / "manifest.json"
        if destination_root.exists() or destination_root.is_symlink():
            raise FileExistsError(f"preserve any prior R5 manifest copy: {destination_root}")
        destination_root.mkdir()
        temporary = destination.with_suffix(".json.partial")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        if file_sha256(source) != file_sha256(destination):
            raise OSError(f"durable export manifest copy hash differs: {destination}")
        retained.append(
            {
                "run": row["run"],
                "source_path": str(source.resolve()),
                "source_sha256": file_sha256(source),
                "durable_path": str(destination.resolve()),
                "durable_sha256": file_sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return retained


def _finalize_worker(args: argparse.Namespace) -> int:
    panel_path = Path(args.panel)
    panel, panel_hash = cs.validate_panel(panel_path)
    admission_path, controller_path, worker_path = map(
        Path, (args.recovery_admission, args.controller_receipt, args.worker_receipt)
    )
    admission, admission_sha, amendment, amendment_sha, stop, stop_sha = _admit(
        admission_path,
        panel,
        panel_hash,
        Path(args.storage_amendment),
        Path(args.gpu2_release_receipt),
    )
    source_sha = admission["source_binding_sha256"]
    current = current_process_owner()
    outer = _read(controller_path)
    if (
        outer.get("status") != "running"
        or outer.get("owner", {}).get("recovery_source_binding_sha256") != source_sha
        or outer.get("owner", {}).get("continuation_panel_sha256") != panel_hash
    ):
        raise ValueError("R4 finalizer is not under its admitted controller")
    _, expected_worker = cs._receipt(
        panel, {"cell_id": FINALIZER_ID, "run_name": FINALIZER_ID}, "stage2"
    )
    if worker_path.resolve() != expected_worker.resolve():
        raise ValueError("R4 finalizer receipt path differs from its frozen cell")
    process = _read(worker_path)
    command = [
        panel["paths"]["python"],
        str(Path(__file__).resolve()),
        "--worker-mode",
        "tf-finalize",
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path.resolve()),
        "--storage-amendment",
        str(Path(args.storage_amendment).resolve()),
        "--controller-receipt",
        str(controller_path.resolve()),
        "--worker-receipt",
        str(worker_path.resolve()),
        "--gpu2-release-receipt",
        str(Path(args.gpu2_release_receipt).resolve()),
    ]
    expected_command = [shutil.which("taskset"), "-c", ",".join(map(str, FINALIZER_CPU)), *command]
    owner = process.get("owner", {})
    if (
        process.get("status") != "running"
        or process.get("command") != expected_command
        or process.get("child") != current
        or owner.get("cell_id") != FINALIZER_ID
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("recovery_source_binding_sha256") != source_sha
        or owner.get("recovery_admission_receipt_sha256") != admission_sha
        or owner.get("storage_amendment_receipt_sha256") != amendment_sha
        or owner.get("r4_stop_receipt_sha256") != stop_sha
    ):
        raise ValueError("R5 finalizer is not the exact owned child")
    result = finalize_target_compact(
        Path(panel["families"]["causal_transformer"]["target_cache"]), "causal_transformer"
    )
    print(json.dumps(result, allow_nan=False))
    return 0


def _run_finalizer(
    panel: dict,
    panel_path: Path,
    panel_hash: str,
    source_sha: str,
    admission_path: Path,
    admission_sha: str,
    amendment_path: Path,
    amendment_sha: str,
    stop_path: Path,
    stop_sha: str,
    controller_path: Path,
    release_path: Path,
    retained_manifests: list[dict[str, Any]],
    transient_ram_export_bytes: int,
    deadline: int,
) -> tuple[dict, Path]:
    cell = {"cell_id": FINALIZER_ID, "run_name": FINALIZER_ID}
    log, receipt_path = cs._receipt(panel, cell, "stage2")
    if log.exists() or receipt_path.exists():
        raise FileExistsError("preserve any prior R4 finalizer attempt")
    command = [
        panel["paths"]["python"],
        str(Path(__file__).resolve()),
        "--worker-mode",
        "tf-finalize",
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path.resolve()),
        "--storage-amendment",
        str(amendment_path.resolve()),
        "--controller-receipt",
        str(controller_path.resolve()),
        "--worker-receipt",
        str(receipt_path.resolve()),
        "--gpu2-release-receipt",
        str(release_path.resolve()),
    ]
    cell.update({"family": "causal_transformer", "cpu_affinity": FINALIZER_CPU, "cpu_threads": 8})
    _wait_disk(panel, deadline, FINALIZE_RESERVE)
    receipt = cs._run_owned(
        panel,
        panel_hash,
        cell,
        "stage2",
        command,
        deadline,
        output_dir=Path(panel["families"]["causal_transformer"]["target_cache"]),
        gpu_uuid=None,
        stage=3,
        source_root=ROOT,
        owner_extra={
            "recovery_source_binding_sha256": source_sha,
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "storage_amendment_receipt_path": str(amendment_path.resolve()),
            "storage_amendment_receipt_sha256": amendment_sha,
            "r4_stop_receipt_path": str(stop_path.resolve()),
            "r4_stop_receipt_sha256": stop_sha,
            "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
            "recovery_entrypoint_path": str(Path(__file__).resolve()),
            "recovery_entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        },
    )
    if (
        receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
    ):
        raise RuntimeError("R4 TF compact finalizer failed; preserve the owned receipt")
    output = Path(panel["families"]["causal_transformer"]["target_cache"])
    metadata, build = _read(output / "metadata.json"), _read(output / "stage2_build_receipt.json")
    adds = []
    for index, action, run in [
        (1, "init", None),
        *[(i, "add", run) for i, run in enumerate(cs.EXPECTED_RUNS["causal_transformer"], 2)],
    ]:
        _, path, _ = _expected_stage2_receipt(
            panel,
            panel_hash,
            "causal_transformer",
            index,
            _stage2_args(panel, "causal_transformer", action, run),
            status="completed",
            exit_code=0,
        )
        adds.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    summary = {
        "schema": "wnn.l4-teacher-stage2-metadata-recovery-r5.v1",
        "status": "compact_finalized_and_promoted",
        "family": "causal_transformer",
        "continuation_panel_sha256": panel_hash,
        "recovery_source_binding_sha256": source_sha,
        "target_cache_path": str(output.resolve()),
        "storage_amendment_receipt_path": str(amendment_path.resolve()),
        "storage_amendment_receipt_sha256": amendment_sha,
        "r4_stop_receipt_path": str(stop_path.resolve()),
        "r4_stop_receipt_sha256": stop_sha,
        "retained_export_manifests": retained_manifests,
        "transient_ram_export_bytes": transient_ram_export_bytes,
        "target_metadata_sha256": file_sha256(output / "metadata.json"),
        "target_contract_sha256": hashlib.sha256(
            json.dumps(
                metadata["target_contract"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "stage2_build_receipt_path": str((output / "stage2_build_receipt.json").resolve()),
        "stage2_build_receipt_sha256": file_sha256(output / "stage2_build_receipt.json"),
        "stage2_init_add_receipts": adds,
        "worker_receipt_path": str(receipt_path.resolve()),
        "worker_receipt_sha256": file_sha256(receipt_path),
        "worker_child": receipt.get("child"),
        "build_state_schema": build.get("schema"),
    }
    _write_once(
        cs._runtime(panel) / "receipts/stage2/causal_transformer_metadata_recovery_r5.json", summary
    )
    return summary, receipt_path


def _controller_context(
    args: argparse.Namespace,
    panel: dict[str, Any],
    panel_hash: str,
    admission: dict[str, Any],
    admission_sha: str,
    amendment_sha: str,
    stop_sha: str,
) -> tuple[dict[str, Any], str]:
    receipt = _read(Path(args.controller_receipt))
    command = [
        shutil.which("taskset"),
        "-c",
        ",".join(map(str, range(64))),
        panel["paths"]["python"],
        str(Path(__file__).resolve()),
        "--controller-mode",
        "continue",
        "--panel",
        str(Path(args.panel).resolve()),
        "--recovery-admission",
        str(Path(args.recovery_admission).resolve()),
        "--storage-amendment",
        str(Path(args.storage_amendment).resolve()),
        "--controller-receipt",
        str(Path(args.controller_receipt).resolve()),
        "--gpu2-release-receipt",
        str(Path(args.gpu2_release_receipt).resolve()),
    ]
    owner, current = receipt.get("owner", {}), current_process_owner()
    command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    if (
        receipt.get("status") != "running"
        or receipt.get("child") != current
        or receipt.get("command") != command
        or receipt.get("command_sha256") != command_sha
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("recovery_source_binding_sha256") != admission["source_binding_sha256"]
        or owner.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
        or owner.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or owner.get("split_sha256") != panel["identity"]["split_sha256"]
        or owner.get("fit_members_sha256") != panel["identity"]["fit_members_sha256"]
        or owner.get("baseline_ancestry_sha256") != panel["identity"]["baseline_ancestry_sha256"]
        or owner.get("recovery_admission_receipt_path")
        != str(Path(args.recovery_admission).resolve())
        or owner.get("recovery_admission_receipt_sha256") != admission_sha
        or owner.get("storage_amendment_receipt_path")
        != str(Path(args.storage_amendment).resolve())
        or owner.get("storage_amendment_receipt_sha256") != amendment_sha
        or owner.get("r4_stop_receipt_path") != admission.get("r4_stop_receipt_path")
        or owner.get("r4_stop_receipt_sha256") != stop_sha
        or owner.get("recovery_entrypoint_path") != str(Path(__file__).resolve())
        or owner.get("recovery_entrypoint_sha256") != file_sha256(Path(__file__).resolve())
        or owner.get("baseline_ancestry_sha256") != panel["identity"]["baseline_ancestry_sha256"]
        or owner.get("cell_id") != CONTROLLER_ID
        or owner.get("run_name") != CONTROLLER_ID
        or owner.get("family") != "continuation"
    ):
        raise ValueError("R5 continuation is not running under its exact owned controller receipt")
    return {"path": str(Path(args.controller_receipt).resolve()), "child": current}, command_sha


def _run(args: argparse.Namespace) -> dict[str, Any]:
    panel, panel_hash = cs.validate_panel(Path(args.panel))
    admission, admission_sha, amendment, amendment_sha, stop, stop_sha = _admit(
        Path(args.recovery_admission),
        panel,
        panel_hash,
        Path(args.storage_amendment),
        Path(args.gpu2_release_receipt),
    )
    stop_adoption = _verify_r4_stop(panel, panel_hash, stop, stop_sha)
    wide_adoption = _adopt_wide_grade(panel, panel_hash, stop)
    controller, command_sha = _controller_context(
        args, panel, panel_hash, admission, admission_sha, amendment_sha, stop_sha
    )
    start_path = (
        cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_START.json"
    )
    failed_path = (
        cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_FAILED.json"
    )
    complete_path = (
        cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_COMPLETE.json"
    )
    if any(path.exists() for path in (start_path, failed_path, complete_path)):
        raise FileExistsError("R5 lifecycle already has evidence; do not restart")
    deadline = cs._deadline(panel)
    base = {
        "schema": STATE_SCHEMA,
        "panel_sha256": panel_hash,
        "source_binding_sha256": panel["identity"]["source_binding_sha256"],
        "recovery_source_binding_sha256": admission["source_binding_sha256"],
        "recovery_admission_receipt_path": str(Path(args.recovery_admission).resolve()),
        "recovery_admission_receipt_sha256": admission_sha,
        "storage_amendment_receipt_path": str(Path(args.storage_amendment).resolve()),
        "storage_amendment_receipt_sha256": amendment_sha,
        "storage_amendment": {
            key: amendment[key]
            for key in (
                "amended_output_cap_bytes",
                "minimum_free_data2_bytes",
                "publication_reserve_bytes",
                "hard_deadline_epoch",
                "max_scientific_fits",
                "max_profiles",
            )
        },
        "r4_stop_receipt_path": stop_adoption["path"],
        "r4_stop_receipt_sha256": stop_sha,
        "r4_controller_receipt_path": stop_adoption["controller_path"],
        "r4_controller_receipt_sha256": stop_adoption["controller_sha256"],
        "wide_grade_adoption": wide_adoption,
        "controller_receipt": controller,
        "controller_command_sha256": command_sha,
        "r3_start_receipt_path": str(
            (
                cs._runtime(panel)
                / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_START.json"
            ).resolve()
        ),
        "r3_start_receipt_sha256": file_sha256(
            cs._runtime(panel) / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_START.json"
        ),
        "r3_failure_receipt_path": str(
            (
                cs._runtime(panel)
                / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json"
            ).resolve()
        ),
        "r3_failure_receipt_sha256": file_sha256(
            cs._runtime(panel)
            / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R3_FAILED.json"
        ),
        "hard_deadline_epoch": deadline,
    }
    _write_once(start_path, {**base, "status": "running", "started_epoch": time.time()})
    families: dict[str, Any] = {}
    phase = "wide_grade_adoption"
    try:
        families["wide_gru"] = {"family": "wide_gru", "grade_status": "complete", **wide_adoption}
        _write_once(
            cs._runtime(panel) / "receipts/continuation/wide_gru_GRADE_ADOPTION_R5.json",
            {"schema": STATE_SCHEMA, "status": "complete", **base, **families["wide_gru"]},
        )

        phase = "tf_teacher_wait_and_target"
        family = "causal_transformer"
        rows = cs._wait_teachers(panel, family, deadline)
        _validate_teacher_admission(panel, family, rows)
        output = Path(panel["families"][family]["target_cache"])
        if output.exists() or output.is_symlink() or cs.partial_root(output).exists():
            raise FileExistsError("TF target already has output/partial; preserve and stop")
        _wait_disk(panel, deadline, INIT_RESERVE)
        cs._run_stage2(panel, panel_hash, family, _stage2_args(panel, family, "init"), deadline, 1)
        _wait_disk(panel, deadline, EXPORT_RESERVE)
        _gpu2_release(Path(args.gpu2_release_receipt), admission, deadline)
        exports, transient_ram_export_bytes = _parallel_teacher_exports(
            panel, panel_hash, rows, amendment, deadline
        )
        for index, row in enumerate(rows, start=2):
            export = exports[row["run"]]
            stage_args = _stage2_args(panel, family, "add", row["run"])
            stage_args.extend(
                ["--sequence-export", str(export), "--teacher-panel", str(row["teacher_panel"])]
            )
            cs._run_stage2(panel, panel_hash, family, stage_args, deadline, index)
        retained_manifests = _retain_export_manifests(panel, rows, exports)
        phase = "tf_compact_finalize_and_review"
        summary, finalizer_receipt = _run_finalizer(
            panel,
            Path(args.panel),
            panel_hash,
            admission["source_binding_sha256"],
            Path(args.recovery_admission),
            admission_sha,
            Path(args.storage_amendment),
            amendment_sha,
            Path(admission["r4_stop_receipt_path"]),
            stop_sha,
            Path(args.controller_receipt),
            Path(args.gpu2_release_receipt),
            retained_manifests,
            transient_ram_export_bytes,
            deadline,
        )
        review, review_path = _wait_cache_review(panel, panel_hash, family, deadline)
        phase = "tf_students_and_grade"
        student = cs._student_family(panel, panel_hash, family, deadline)
        grade_path = Path(student["grade_receipt_path"])
        families[family] = {
            "family": family,
            "target_cache_path": summary["target_cache_path"],
            "target_metadata_sha256": summary["target_metadata_sha256"],
            "target_contract_sha256": summary["target_contract_sha256"],
            "finalizer_receipt_path": str(finalizer_receipt.resolve()),
            "finalizer_receipt_sha256": file_sha256(finalizer_receipt),
            "cache_review_path": str(review_path.resolve()),
            "cache_review_sha256": file_sha256(review_path),
            "profiles": student["profiles"],
            "fits": student["fit_artifacts"],
            "valid_exports": student["valid_exports"],
            "grade_receipt_path": str(grade_path.resolve()),
            "grade_receipt_sha256": file_sha256(grade_path),
            "grade_status": student["grade_status"],
            "candidate_rows": student["grade_candidate_rows"],
        }
        _write_once(
            cs._runtime(panel)
            / "receipts/continuation/causal_transformer_RECOVERY_COMPLETE_R5.json",
            {"schema": STATE_SCHEMA, "status": "complete", **base, **families[family]},
        )
    except Exception as error:
        _write_once(
            failed_path,
            {
                **base,
                "status": "failed",
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_families": families,
            },
        )
        raise
    complete = {
        **base,
        "status": "complete",
        "finished_epoch": time.time(),
        "families": families,
        "all_family_grades_complete": all(
            value["grade_status"] == "complete" for value in families.values()
        ),
        "checkpoint_endpoint_epoch": 1,
        "fixed_weights": {"TL3": 0.9, "candidate": 0.1},
    }
    _write_once(complete_path, complete)
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--controller-mode", choices=("continue",))
    modes.add_argument("--worker-mode", choices=("tf-finalize",))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--recovery-admission", type=Path, required=True)
    parser.add_argument("--storage-amendment", type=Path, required=True)
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
        sys.stderr.write(f"R5 student continuation stopped: {type(error).__name__}: {error}\n")
        raise SystemExit(2) from error
