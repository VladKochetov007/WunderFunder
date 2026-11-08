"""R6 wrapper for the frozen R5 cache grader and its two-export recovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.l4_teacher_capacity_20261104 import grade_stage2_cache_r5 as r5  # noqa: E402

base = r5.base
R6_ADMISSION_SCHEMA = "wnn.l4-stage2-cache-review-source-admission-r6-r2.v1"
R6_ADMISSION_NAME = "STAGE2_CACHE_REVIEW_SOURCE_ADMISSION_R6_R2.json"
MINIMUM_STARTED_FREE_BYTES = 6_488_796_672

_ADMISSION: dict[str, Any] | None = None
_ADMISSION_PATH: Path | None = None
_ADMISSION_SHA256: str | None = None
_PANEL: dict[str, Any] | None = None
_PANEL_PATH: Path | None = None
_PANEL_SHA256: str | None = None
_EXPORTS: dict[str, dict[str, Any]] = {}
_TIMELINE_PROOF: dict[str, Any] | None = None


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_path_sha(path: Path, expected_path: Any, expected_sha: Any, label: str) -> None:
    _need(path.is_file() and not path.is_symlink(), f"{label} is absent or not regular: {path}")
    _need(
        expected_path == str(path.resolve()) and expected_sha == base.file_sha256(path),
        f"{label} path/hash differs from the R6 admission",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = base._read_json(path)
    _need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _validate_admission(
    panel_path: Path,
    family: str,
    recovery_path: Path,
    amendment_path: Path,
    admission_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    global _ADMISSION, _ADMISSION_PATH, _ADMISSION_SHA256, _PANEL, _PANEL_PATH, _PANEL_SHA256
    panel, panel_sha, teacher = base._check_panel(panel_path, family)
    control = Path(panel["paths"]["control_root"]).resolve(strict=True)
    _need(
        admission_path.resolve(strict=True).parent == control
        and admission_path.name == R6_ADMISSION_NAME,
        "R6 source admission must be the named root control receipt",
    )
    admission = _read_json(admission_path)
    recovery_path = recovery_path.resolve(strict=True)
    amendment_path = amendment_path.resolve(strict=True)
    recovery = _read_json(recovery_path)
    amendment = _read_json(amendment_path)
    r5_start_path = (
        Path(panel["paths"]["runtime_root"])
        / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_START.json"
    )
    r5_start = _read_json(r5_start_path)
    controller_ref = r5_start.get("controller_receipt", {})
    controller_path = Path(str(controller_ref.get("path", ""))).resolve(strict=True)
    controller = _read_json(controller_path)
    controller_owner = controller.get("owner", {})
    producer_path = Path(str(controller_owner.get("recovery_entrypoint_path", ""))).resolve(
        strict=True
    )
    r5_grader_path = Path(r5.__file__).resolve(strict=True)
    r6_grader_path = Path(__file__).resolve(strict=True)
    _need(
        producer_path.name == "continue_students_recovery_r5.py"
        and controller_owner.get("recovery_source_binding_sha256")
        == recovery.get("source_binding_sha256")
        and controller_owner.get("recovery_entrypoint_sha256") == base.file_sha256(producer_path),
        "R5 controller owner does not bind the admitted S5 producer path, digest and source",
    )
    expected = {
        "schema": R6_ADMISSION_SCHEMA,
        "status": "approved",
        "continuation_panel_sha256": panel_sha,
        "r5_source_binding_sha256": recovery.get("source_binding_sha256"),
        "r5_entrypoint_path": str(producer_path.resolve()),
        "r5_entrypoint_sha256": base.file_sha256(producer_path),
        "r5_grader_path": str(r5_grader_path),
        "r5_grader_sha256": base.file_sha256(r5_grader_path),
        "r6_entrypoint_path": str(r6_grader_path),
        "r6_entrypoint_sha256": base.file_sha256(r6_grader_path),
        "r5_admission_path": str(recovery_path),
        "r5_admission_sha256": base.file_sha256(recovery_path),
        "storage_amendment_path": str(amendment_path),
        "storage_amendment_sha256": base.file_sha256(amendment_path),
        "gpu2_release_receipt_path": recovery.get("gpu2_release_receipt_path"),
        "gpu2_release_receipt_sha256": admission.get("gpu2_release_receipt_sha256"),
        "gpu2_uuid": panel["families"][family]["export_gpu_uuid"],
        "objective_panel_sha256": admission.get("objective_panel_sha256"),
        "objective_fit_receipt_path": admission.get("objective_fit_receipt_path"),
        "objective_fit_receipt_sha256": admission.get("objective_fit_receipt_sha256"),
        "new_wave_complete_receipt_path": amendment.get("new_wave_complete_receipt_path"),
        "new_wave_complete_receipt_sha256": amendment.get("new_wave_complete_receipt_sha256"),
        "minimum_started_free_bytes": MINIMUM_STARTED_FREE_BYTES,
    }
    _need(
        all(admission.get(key) == value for key, value in expected.items()),
        "R6 source admission differs from the frozen R5 panel, source or recovery authority",
    )
    _need(r5._valid_sha(admission.get("source_binding_sha256")), "R6 source binding is invalid")
    _need(
        all(
            admission.get(key) == panel["identity"].get(key)
            for key in (
                "data_binding_sha256",
                "split_sha256",
                "fit_members_sha256",
                "baseline_ancestry_sha256",
            )
        ),
        "R6 source admission data/split/fit/baseline identity differs",
    )
    _same_path_sha(
        producer_path,
        admission.get("r5_entrypoint_path"),
        admission.get("r5_entrypoint_sha256"),
        "R5 continuation producer",
    )
    _same_path_sha(
        r5_grader_path,
        admission.get("r5_grader_path"),
        admission.get("r5_grader_sha256"),
        "R5 cache grader",
    )
    _same_path_sha(
        r6_grader_path,
        admission.get("r6_entrypoint_path"),
        admission.get("r6_entrypoint_sha256"),
        "R6 cache grader",
    )
    new_wave = Path(str(amendment["new_wave_complete_receipt_path"]))
    _same_path_sha(
        new_wave,
        admission.get("new_wave_complete_receipt_path"),
        admission.get("new_wave_complete_receipt_sha256"),
        "closed new-wave receipt",
    )
    _need(
        recovery_path.parent == control, "R5 recovery admission is outside the bound control root"
    )
    _need(
        amendment_path.parent == control, "R5 storage amendment is outside the bound control root"
    )
    _ADMISSION = admission
    _ADMISSION_PATH = admission_path.resolve()
    _ADMISSION_SHA256 = base.file_sha256(admission_path)
    _PANEL = panel
    _PANEL_PATH = panel_path.resolve()
    _PANEL_SHA256 = panel_sha
    return panel, panel_sha, teacher


def _check_r5_stop_and_wide_r6(
    panel: dict[str, Any], panel_hash: str, admission: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain R5 adoption checks while reading TL3's canonical ensemble row."""
    stop_path = Path(panel["paths"]["control_root"]) / "STUDENT_RECOVERY_R4_STOP_R1.json"
    r5._same_file(
        stop_path,
        admission.get("r4_stop_receipt_path"),
        admission.get("r4_stop_receipt_sha256"),
        "R4 pre-target stop proof",
    )
    stop = _read_json(stop_path)
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
    r4_start = _read_json(r4_start_path)
    r4_ref = r4_start.get("controller_receipt", {})
    r4_controller_path = Path(str(r4_ref.get("path", ""))).resolve(strict=True)
    r5._same_file(
        r4_controller_path,
        stop.get("controller_receipt_path"),
        stop.get("controller_receipt_sha256"),
        "stopped R4 controller",
    )
    r4_controller = _read_json(r4_controller_path)
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
    teacher_panel = _read_json(Path(tf_binding["teacher_panel"]))
    teacher_cells = {cell["run_name"]: cell for cell in teacher_panel["cells"]}
    unchanged = stop.get("tf_fit_children_unchanged")
    base._require(
        isinstance(unchanged, list) and len(unchanged) == 2,
        "R4 stop proof omits a Transformer fit identity",
    )
    unchanged_by_cell = {record.get("cell_id"): record.get("identity") for record in unchanged}
    for run in base.EXPECTED_RUNS["causal_transformer"]:
        cell = teacher_cells[run]
        receipt = _read_json(Path(tf_binding["fit_receipts"][run]))
        base._require(
            unchanged_by_cell.get(cell.get("cell_id")) == receipt.get("child"),
            f"{run}: teacher fit child identity differs from the R4 stop proof",
        )
    grade_path = Path(str(stop.get("wide_grade_receipt_path", ""))).resolve(strict=True)
    r5._same_file(
        grade_path,
        stop.get("wide_grade_receipt_path"),
        stop.get("wide_grade_receipt_sha256"),
        "adopted wide grade",
    )
    grade = _read_json(grade_path)
    adoption_path = runtime / "receipts/continuation/wide_gru_GRADE_ADOPTION_R5.json"
    adoption = _read_json(adoption_path)
    base._require(
        grade.get("status") == "complete"
        and grade.get("family") == "wide_gru"
        and grade.get("panel_sha256") == panel_hash
        and adoption.get("schema") == r5.STATE_SCHEMA
        and adoption.get("status") == "complete"
        and adoption.get("grade_receipt_path") == str(grade_path)
        and adoption.get("grade_receipt_sha256") == base.file_sha256(grade_path)
        and adoption.get("manifest_path") == grade.get("manifest_path")
        and adoption.get("manifest_sha256") == grade.get("manifest_sha256")
        and adoption.get("report_path") == grade.get("report_path")
        and adoption.get("report_sha256") == grade.get("report_sha256"),
        "R5 wide grade adoption differs from the root-pinned completed grade",
    )
    for field in ("manifest", "report"):
        path_value = grade.get(f"{field}_path")
        sha_value = grade.get(f"{field}_sha256")
        r5._same_file(Path(str(path_value)), path_value, sha_value, f"wide canonical {field}")
    manifest = _read_json(Path(grade["manifest_path"]))
    report = _read_json(Path(grade["report_path"]))
    cache_review_path = Path(grade["cache_admission_receipt_path"]).resolve(strict=True)
    r5._same_file(
        cache_review_path,
        grade.get("cache_admission_receipt_path"),
        grade.get("cache_admission_receipt_sha256"),
        "wide independent cache review",
    )
    cache_review = _read_json(cache_review_path)
    baseline_name = report.get("baseline")
    baseline_rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, dict) and row.get("name") == baseline_name
    ]
    baseline_row = baseline_rows[0] if len(baseline_rows) == 1 else {}
    baseline_score = baseline_row.get("score", {})
    base._require(
        manifest.get("schema") == "wnn.l4tc-fixed-student-blend-inputs.v1"
        and manifest.get("split", {}).get("sha256") == base.SPLIT_SHA256
        and manifest.get("assembly", {}).get("family") == "wide_gru"
        and manifest.get("assembly", {}).get("continuation_panel_sha256") == panel_hash
        and report.get("schema") == "wnn.heterogeneous-ensemble-score.v1"
        and report.get("status") == "complete"
        and report.get("selected_rows") == 3_528_273
        and report.get("input_manifest_sha256") == base.file_sha256(Path(grade["manifest_path"]))
        and baseline_name == "TL3"
        and baseline_row.get("status") == "complete"
        and baseline_row.get("missing_members") == []
        and baseline_score.get("selected_rows") == 3_528_273
        and baseline_score.get("metric_id") == "global_wp_clip2_v1"
        and abs(baseline_score.get("wp", 0) - 0.6923774245652725) <= 1e-7
        and cache_review.get("status")
        == "approved_stage2_cache_lineage_formula_not_numerically_recomputed"
        and cache_review.get("family") == "wide_gru"
        and adoption.get("cache_review_path") == str(cache_review_path)
        and adoption.get("cache_review_sha256") == base.file_sha256(cache_review_path),
        "adopted wide report, manifest, or cache-review evidence differs",
    )
    return stop, {
        "path": str(adoption_path.resolve()),
        "sha256": base.file_sha256(adoption_path),
        "grade": grade,
        "r4_controller_path": str(r4_controller_path),
        "r4_controller_sha256": base.file_sha256(r4_controller_path),
    }


def _check_gpu2_release_r6(
    panel: dict[str, Any], recovery_admission: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Accept only the R6-pinned release in its objective runtime's control root."""
    if _ADMISSION is None or r5._AMENDMENT is None:
        raise RuntimeError("R6 root release or R5 amendment was not admitted")
    release_path = Path(str(recovery_admission.get("gpu2_release_receipt_path", ""))).resolve(
        strict=True
    )
    fit_path = Path(str(_ADMISSION["objective_fit_receipt_path"])).resolve(strict=True)
    objective_runtime = fit_path.parents[2]
    expected_release = objective_runtime / "control/GPU2_RELEASE_R1.json"
    _need(release_path == expected_release, "GPU2 release is not under the bound objective runtime")
    _same_path_sha(
        release_path,
        _ADMISSION.get("gpu2_release_receipt_path"),
        _ADMISSION.get("gpu2_release_receipt_sha256"),
        "root GPU2 release",
    )
    _need(
        recovery_admission.get("gpu2_release_receipt_path") == str(release_path)
        and recovery_admission.get("gpu2_release_schema") == r5.RELEASE_SCHEMA
        and recovery_admission.get("gpu2_uuid") == _ADMISSION["gpu2_uuid"],
        "R5 recovery admission differs from the R6-pinned GPU2 release",
    )
    release = _read_json(release_path)
    fit = _read_json(fit_path)
    fit_owner = fit.get("owner", {})
    wave_path = Path(str(r5._AMENDMENT["new_wave_complete_receipt_path"]))
    wave = _read_json(wave_path)
    wave_fits = wave.get("fit_receipts", [])
    _need(
        release.get("schema") == r5.RELEASE_SCHEMA
        and release.get("status") == "released"
        and release.get("gpu_uuid") == _ADMISSION["gpu2_uuid"]
        and release.get("objective_panel_sha256") == _ADMISSION["objective_panel_sha256"]
        and release.get("objective_fit_receipt_path") == str(fit_path)
        and release.get("objective_fit_receipt_sha256") == base.file_sha256(fit_path)
        and release.get("terminal_owned_child_absent") is True,
        "GPU2 release does not bind a completed, reaped objective fit",
    )
    _need(
        fit.get("status") == "completed"
        and fit.get("exit_code") == 0
        and fit.get("timed_out") is False
        and fit.get("gpu_key") == _ADMISSION["gpu2_uuid"]
        and fit_owner.get("cell_id") == "clip5_s1001"
        and fit_owner.get("panel_sha256", fit_owner.get("continuation_panel_sha256"))
        == release["objective_panel_sha256"],
        "objective fit receipt is not the exact successful GPU2 owner",
    )
    _need(
        wave.get("status") == "completed_no_students"
        and wave.get("objective_panel_sha256") == release["objective_panel_sha256"]
        and isinstance(wave_fits, list)
        and any(
            isinstance(item, dict)
            and item.get("cell_id") == "clip5_s1001"
            and Path(str(item.get("receipt_path", ""))).resolve() == fit_path
            and item.get("receipt_sha256") == base.file_sha256(fit_path)
            for item in wave_fits
        ),
        "closed new-wave receipt does not bind the released objective fit",
    )
    return release_path, release


def _parallel_timeline_proof(panel: dict[str, Any], family: str) -> dict[str, Any]:
    runs = list(base.EXPECTED_RUNS[family])
    _need(family == "causal_transformer" and len(runs) == 2, "R6 is scoped to the two TF exports")
    _need(set(_EXPORTS) == set(runs), "R6 did not observe exactly the two checked exports")
    runtime = Path(panel["paths"]["runtime_root"])
    receipts_root = runtime / "receipts/stage2"
    init_path = receipts_root / f"{family}_stage2_01.json"
    init = _read_json(init_path)
    init_finished = init.get("finished_at_epoch")
    exports = []
    for run in runs:
        item = _EXPORTS[run]
        started, finished = item.get("started_at_epoch"), item.get("finished_at_epoch")
        _need(
            isinstance(started, (int, float))
            and isinstance(finished, (int, float))
            and started <= finished,
            f"{run}: export timing is incomplete or reversed",
        )
        exports.append({"run": run, "started_at_epoch": started, "finished_at_epoch": finished})
    _need(
        isinstance(init_finished, (int, float))
        and all(init_finished <= row["started_at_epoch"] for row in exports),
        "R6 target init did not finish before both teacher exports",
    )
    add_receipts = [
        _read_json(receipts_root / f"{family}_stage2_{index:02d}.json") for index in (2, 3)
    ]
    add_timing = []
    for run, receipt in zip(runs, add_receipts, strict=True):
        started, finished = receipt.get("started_at_epoch"), receipt.get("finished_at_epoch")
        _need(
            isinstance(started, (int, float))
            and isinstance(finished, (int, float))
            and started <= finished,
            f"{run}: stage2 add timing is incomplete or reversed",
        )
        add_timing.append({"run": run, "started_at_epoch": started, "finished_at_epoch": finished})
    _need(
        max(row["finished_at_epoch"] for row in exports) <= add_timing[0]["started_at_epoch"]
        and add_timing[0]["finished_at_epoch"] <= add_timing[1]["started_at_epoch"],
        "both exports must finish before the first add; adds must retain frozen teacher order",
    )
    finalizer_path = receipts_root / f"{r5.FINALIZER_ID}.json"
    finalizer = _read_json(finalizer_path)
    finalizer_started = finalizer.get("started_at_epoch")
    _need(
        isinstance(finalizer_started, (int, float))
        and add_timing[-1]["finished_at_epoch"] <= finalizer_started,
        "compact finalization preceded a completed ordered add",
    )
    return {
        "policy": "init before both exports; both exports before ordered adds; adds before finalizer",
        "raw_timestamps_preserved": True,
        "init": {
            "path": str(init_path.resolve()),
            "sha256": base.file_sha256(init_path),
            "finished_at_epoch": init_finished,
        },
        "exports": exports,
        "adds": [
            {
                **row,
                "path": str((receipts_root / f"{family}_stage2_{index:02d}.json").resolve()),
                "sha256": base.file_sha256(receipts_root / f"{family}_stage2_{index:02d}.json"),
            }
            for row, index in zip(add_timing, (2, 3), strict=True)
        ],
        "finalizer": {
            "path": str(finalizer_path.resolve()),
            "sha256": base.file_sha256(finalizer_path),
            "started_at_epoch": finalizer_started,
        },
    }


def main() -> None:
    global _TIMELINE_PROOF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-panel", required=True, type=Path)
    parser.add_argument("--family", choices=("causal_transformer",), required=True)
    parser.add_argument("--recovery-admission", required=True, type=Path)
    parser.add_argument("--storage-amendment", required=True, type=Path)
    parser.add_argument("--grader-source-admission", required=True, type=Path)
    args = parser.parse_args()
    panel_path = args.continuation_panel.resolve(strict=True)
    panel, _, _ = _validate_admission(
        panel_path,
        args.family,
        args.recovery_admission,
        args.storage_amendment,
        args.grader_source_admission,
    )
    cache = Path(panel["families"][args.family]["target_cache"]).resolve()
    build = _read_json(cache / "stage2_build_receipt.json")
    started_free = int(build.get("started_free_bytes", 0))
    _need(
        started_free >= MINIMUM_STARTED_FREE_BYTES,
        "R5 stream start did not retain the frozen conservative base-grader peak headroom",
    )

    old_require = base._require
    old_gpu_release = r5._check_gpu2_release
    old_wide_check = r5._check_r5_stop_and_wide
    old_export_wrapper = r5._check_r5_transformer_export_receipt
    old_write_json = base._write_json
    review_path = Path(panel["families"][args.family]["cache_admission_receipt"]).resolve()

    def require_r6(condition: bool, message: str) -> None:
        global _TIMELINE_PROOF
        if message.endswith(
            "teacher exports and target aggregation were not streamed one at a time"
        ):
            if _TIMELINE_PROOF is None:
                _TIMELINE_PROOF = _parallel_timeline_proof(panel, args.family)
            return
        old_require(condition, message)

    def export_wrapper(*positional: Any, **keyword: Any) -> dict[str, Any]:
        result = old_export_wrapper(*positional, **keyword)
        run = positional[2] if len(positional) > 2 else keyword["run"]
        _EXPORTS[run] = result
        return result

    def write_review(path: Path, value: dict[str, Any]) -> None:
        if Path(path).resolve() == review_path:
            _need(_TIMELINE_PROOF is not None, "R6 parallel timeline proof was not completed")
            value = dict(value)
            value["r6_source_admission"] = {
                "path": str(_ADMISSION_PATH),
                "sha256": _ADMISSION_SHA256,
                "schema": R6_ADMISSION_SCHEMA,
                "source_binding_sha256": _ADMISSION["source_binding_sha256"],
                "entrypoint_path": _ADMISSION["r6_entrypoint_path"],
                "entrypoint_sha256": _ADMISSION["r6_entrypoint_sha256"],
            }
            value["parallel_export_timeline"] = _TIMELINE_PROOF
            value["minimum_started_free_bytes"] = MINIMUM_STARTED_FREE_BYTES
            value["observed_started_free_bytes"] = started_free
        old_write_json(path, value)

    base._require = require_r6
    r5._check_gpu2_release = _check_gpu2_release_r6
    r5._check_r5_stop_and_wide = _check_r5_stop_and_wide_r6
    r5._check_r5_transformer_export_receipt = export_wrapper
    base._write_json = write_review
    original_argv = sys.argv
    sys.argv = [
        str(Path(r5.__file__).resolve()),
        "--continuation-panel",
        str(panel_path),
        "--family",
        args.family,
        "--recovery-admission",
        str(args.recovery_admission.resolve()),
        "--storage-amendment",
        str(args.storage_amendment.resolve()),
    ]
    try:
        r5.main()
        _need(_TIMELINE_PROOF is not None, "R6 grader did not exercise its timeline proof")
    finally:
        sys.argv = original_argv
        base._require = old_require
        r5._check_gpu2_release = old_gpu_release
        r5._check_r5_stop_and_wide = old_wide_check
        r5._check_r5_transformer_export_receipt = old_export_wrapper
        base._write_json = old_write_json


if __name__ == "__main__":
    main()
