"""Poll PR61 R5 and publish its verified fixed-epoch artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import w18_l4_quality_finalize_20261104 as publisher  # noqa: E402
from scripts.l4_teacher_capacity_20261104 import continue_students as cs  # noqa: E402
from scripts.l4_teacher_capacity_20261104 import pr61_observer as observer  # noqa: E402

REPOSITORY = "VladKochetov007/WunderFunder"
LOGIN = "SemyonEpanov"
PR_NUMBER = 61
HEAD_REF = "experiment/l4-teacher-capacity-20261104"
TAG = "tabdata-pr61-results-20261104-r5"
PREVIOUS_TAG = "tabdata-pr61-completed-eight-20261104-r4"
COMMENT_MARKER = f"<!-- l4-teacher-capacity-pr61-results:{TAG} -->"
VALID_ROWS = 3_528_273
VALID_SEQUENCES = 1_873
EXPECTED_FITS = 12
FAMILY_RUNS = {"wide_gru": 4, "causal_transformer": 2}
TARGET_FORMULA = "float32(0.8*T_old + 0.2*0.8125*mean(float32_clip(each_new_teacher,-2,2)))"
COMMON_IDENTITY = ("data_binding_sha256", "split_sha256", "fit_members_sha256")


def _read(path: Path, limit: int = 64 << 20) -> dict[str, Any] | None:
    if path.is_symlink() or not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"expected a regular bounded JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_file(path: Path, expected_path: object, expected_sha: object, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != Path(str(expected_path)).resolve()
    ):
        raise ValueError(f"{label} is absent or path-bound differently: {path}")
    if _sha(path) != expected_sha:
        raise ValueError(f"{label} SHA256 differs: {path}")


def _identity(panel: dict[str, Any], other: dict[str, Any], label: str) -> None:
    for key in COMMON_IDENTITY:
        if panel.get("identity", {}).get(key) != other.get("identity", {}).get(key):
            raise ValueError(f"{label} differs on shared identity {key}")


def _validate_native_tf_owner(
    panel: dict[str, Any],
    cell: dict[str, Any],
    receipt: dict[str, Any],
    expected_panel_sha: str,
    command: object,
) -> None:
    """Bind a native TF receipt even when its owner omits the redundant run name."""
    owner = receipt.get("owner", {})
    identity = panel.get("identity", {})
    output = Path(str(cell.get("fit_output", ""))).resolve()
    config_hashes = {
        value
        for value in (cell.get("config_sha256"), cell.get("normalized_config_sha256"))
        if isinstance(value, str)
    }
    source_git = cell.get("source_git") or identity.get("source_git") or panel.get("source_git")
    if (
        panel.get("schema") != "wnn.l4-teacher-capacity-transformer-panel.v1"
        or owner.get("cell_id") != cell.get("cell_id")
        or owner.get("run_name") not in (None, cell.get("run_name"))
        or owner.get("panel_sha256") != expected_panel_sha
        or owner.get("source_binding_sha256") != identity.get("source_binding_sha256")
        or owner.get("data_binding_sha256") != identity.get("data_binding_sha256")
        or owner.get("split_sha256") != identity.get("split_sha256")
        or owner.get("fit_members_sha256") != identity.get("fit_members_sha256")
        or not config_hashes
        or owner.get("config_sha256") not in config_hashes
        or not output.is_absolute()
        or receipt.get("output_dir") != str(output)
        or receipt.get("command") != command
    ):
        raise ValueError(f"{cell['cell_id']}: native Transformer fit identity differs")
    if source_git is not None and owner.get("source_git") != source_git:
        raise ValueError(f"{cell['cell_id']}: native Transformer source git differs")
    observed_git = owner.get("source_git")
    if observed_git is not None and (
        not isinstance(observed_git, str)
        or len(observed_git) != 40
        or any(character not in "0123456789abcdef" for character in observed_git)
    ):
        raise ValueError(f"{cell['cell_id']}: native Transformer source git is malformed")


def _controller_outcome(receipt: dict[str, Any] | None) -> str:
    if receipt is None or receipt.get("status") == "running":
        return "waiting"
    if (
        receipt.get("status") == "completed"
        and receipt.get("exit_code") == 0
        and receipt.get("timed_out") is False
    ):
        return "completed"
    return "failed"


def _r5_controller_outcome(complete: dict[str, Any]) -> str:
    reference = complete.get("controller_receipt")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return "failed"
    receipt = _read(Path(reference["path"]))
    if receipt is None:
        return "waiting"
    if receipt.get("child") != reference.get("child"):
        return "failed"
    return _controller_outcome(receipt)


def _complete_deadline_matches(value: object, deadline: float) -> bool:
    return type(value) is int and value == int(deadline)


def _api_head(client: publisher.GitHub) -> str:
    user = client.request("GET", "/user")
    pull = client.request("GET", f"/repos/{REPOSITORY}/pulls/{PR_NUMBER}")
    head = pull.get("head") if isinstance(pull, dict) else None
    repo = head.get("repo") if isinstance(head, dict) else None
    sha = head.get("sha") if isinstance(head, dict) else None
    if (
        not isinstance(user, dict)
        or user.get("login") != LOGIN
        or not isinstance(pull, dict)
        or pull.get("number") != PR_NUMBER
        or pull.get("state") != "open"
        or not isinstance(head, dict)
        or head.get("ref") != HEAD_REF
        or not isinstance(repo, dict)
        or repo.get("full_name", "").lower() != REPOSITORY.lower()
        or not isinstance(sha, str)
        or len(sha) != 40
    ):
        raise publisher.FinalizeError("GitHub identity is outside the frozen PR61 scope")
    return sha.lower()


def _snapshot(
    panel: dict[str, Any], panel_sha: str, gru: dict[str, Any], tf: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = observer._scientific_cells(gru, tf, panel)
    result = []
    for item in rows:
        family, source_panel, cell = item["family"], item["panel"], item["cell"]
        receipt_path = observer._fit_path(item)
        receipt = _read(receipt_path)
        output = observer._output_dir(source_panel, cell)
        terminal = observer._terminal(receipt)
        profile = _read(observer._phase_receipt(source_panel, cell, "profile"))
        profile_status = observer._terminal(profile)
        metrics = observer._metrics(output) if output.is_dir() and not output.is_symlink() else []
        owner = receipt.get("owner", {}) if receipt else {}
        source_sha = source_panel.get("identity", {}).get("source_binding_sha256")
        expected_panel_sha = (
            panel_sha if "students" in source_panel else source_panel.get("_r5_panel_sha256")
        )
        command = cell.get("fit_argv") if "students" not in source_panel else None
        if "students" in source_panel:
            affinity = ",".join(map(str, cell["cpu_affinity"]))
            args = list(cell["train_args"])
            command = [
                shutil.which("taskset") or "/usr/bin/taskset",
                "-c",
                affinity,
                str(source_panel["paths"]["python"]),
                str(Path(source_panel["paths"]["source_root"]) / "scripts/train_rnn.py"),
                *args,
            ]
        if receipt is not None:
            owner_panel = owner.get("panel_sha256", owner.get("cohort_panel_sha256"))
            native_tf = source_panel.get("schema") == "wnn.l4-teacher-capacity-transformer-panel.v1"
            if native_tf:
                if owner.get("run_name") not in (None, cell["run_name"]):
                    raise ValueError(f"{cell['cell_id']}: native Transformer run name differs")
                _validate_native_tf_owner(
                    source_panel, cell, receipt, str(expected_panel_sha), command
                )
            elif (
                owner_panel != expected_panel_sha
                or owner.get("cell_id") != cell["cell_id"]
                or owner.get("run_name") != cell["run_name"]
                or (isinstance(command, list) and receipt.get("command") != command)
            ):
                raise ValueError(f"{cell['cell_id']}: owned fit receipt identity/command differs")
            if receipt.get("gpu_key") not in (None, cell.get("gpu_uuid")):
                raise ValueError(f"{cell['cell_id']}: owned fit GPU differs from its frozen panel")
            expected_source = source_sha
            if owner.get("source_binding_sha256") not in (None, expected_source):
                raise ValueError(f"{cell['cell_id']}: owner source binding differs from its panel")
        latest = metrics[-1] if metrics else {}
        result.append(
            {
                "family": family,
                "run": str(cell["run_name"]),
                "cell_id": str(cell["cell_id"]),
                "status": terminal,
                "terminal": terminal in {"completed", "failed"},
                "phase": (
                    "fit complete"
                    if terminal == "completed"
                    else "fit failed"
                    if terminal == "failed"
                    else f"training, epoch {latest.get('epoch')}"
                    if terminal == "running" and latest.get("epoch") is not None
                    else "training"
                    if terminal == "running"
                    else "profile complete; awaiting fit"
                    if profile_status == "completed"
                    else "profile failed"
                    if profile_status == "failed"
                    else "profiling"
                    if profile_status == "running"
                    else "queued"
                ),
                "receipt_path": str(receipt_path),
                "receipt_sha256": _sha(receipt_path) if receipt_path.is_file() else None,
                "source_binding_sha256": source_sha,
                "epoch": latest.get("epoch"),
                "valid_wp": latest.get("valid_wp"),
                "valid_wmse": latest.get("valid_wmse"),
                "grad_norm": latest.get("grad_norm"),
                "clip": latest.get("clip_threshold"),
            }
        )
    if len(result) != EXPECTED_FITS or len({row["run"] for row in result}) != EXPECTED_FITS:
        raise ValueError("frozen inventory is not exactly twelve unique scientific fits")
    return result


def _profile_counts(
    panel: dict[str, Any], gru: dict[str, Any], tf: dict[str, Any]
) -> dict[str, int]:
    runtime = Path(panel["paths"]["runtime_root"])
    counts: dict[str, int] = {}
    for path in observer._profile_receipts(runtime, gru, tf, panel):
        receipt = _read(path)
        if receipt is not None:
            status = observer._terminal(receipt)
            counts[status] = counts.get(status, 0) + 1
    return counts


def preflight_existing(panel_path: Path) -> dict[str, Any]:
    """Read-only preflight for completed GRU runs and the adopted wide grade."""
    panel, panel_sha = cs.validate_panel(panel_path)
    runtime = Path(panel["paths"]["runtime_root"])
    teacher_panels: dict[str, dict[str, Any]] = {}
    for family in FAMILY_RUNS:
        path = Path(panel["families"][family]["teacher_panel"])
        value = _read(path)
        if value is None or _sha(path) != panel["families"][family]["teacher_panel_sha256"]:
            raise ValueError(f"{family}: frozen teacher panel is absent or changed")
        _identity(panel, value, f"{family} panel")
        if len(cs._teacher_cells(panel, family)) != FAMILY_RUNS[family]:
            raise ValueError(f"{family}: frozen stage-two teacher set differs")
        value["_r5_panel_sha256"] = _sha(path)
        teacher_panels[family] = value
    gru, tf = teacher_panels["wide_gru"], teacher_panels["causal_transformer"]
    snapshot = _snapshot(panel, panel_sha, gru, tf)
    gru_teachers = [row for row in snapshot if row["family"] == "wide_gru"]
    tf_teachers = [row for row in snapshot if row["family"] == "causal_transformer"]
    wide_students = [row for row in snapshot if row["family"] == "student_wide_gru"]
    if (
        len(gru_teachers) != 6
        or any(row["status"] != "completed" for row in gru_teachers)
        or len(wide_students) != 2
        or any(row["status"] != "completed" for row in wide_students)
        or len(tf_teachers) != 2
        or any(row["status"] not in {"running", "completed"} for row in tf_teachers)
    ):
        raise ValueError(
            "preflight requires six completed GRUs, two completed wide students, and two active/successful TF teachers"
        )
    adoption_path = runtime / "receipts/continuation/wide_gru_GRADE_ADOPTION_R5.json"
    adoption = _read(adoption_path)
    if (
        adoption is None
        or adoption.get("schema") != "wnn.l4-teacher-capacity-continuation-recovery-r5.v1"
        or adoption.get("status") != "complete"
        or adoption.get("panel_sha256") != panel_sha
        or adoption.get("grade_status") != "complete"
    ):
        raise ValueError("preflight requires the actual R5 wide-grade adoption receipt")
    grade_path = Path(str(adoption.get("grade_receipt_path", "")))
    _grade("wide_gru", grade_path, panel, panel_sha, adoption)
    cache = Path(panel["families"]["wide_gru"]["target_cache"])
    metadata = _read(cache / "metadata.json")
    main_record = metadata["files"]["main_source_train"]
    main_path = cache.joinpath(*PurePosixPath(main_record["path"]).parts)
    return {
        "schema": "wnn.l4tc-pr61-r5-readonly-preflight.v1",
        "status": "ready_for_r5_poll",
        "panel_sha256": panel_sha,
        "existing_completed_fit_receipts": [
            {
                key: row[key]
                for key in (
                    "family",
                    "run",
                    "cell_id",
                    "receipt_path",
                    "receipt_sha256",
                    "source_binding_sha256",
                )
            }
            for row in [*gru_teachers, *wide_students]
        ],
        "pending_or_completed_tf_teacher_receipts": [
            {
                key: row[key]
                for key in (
                    "family",
                    "run",
                    "cell_id",
                    "status",
                    "receipt_path",
                    "receipt_sha256",
                    "source_binding_sha256",
                )
            }
            for row in tf_teachers
        ],
        "wide_grade_receipt_path": str(grade_path.resolve()),
        "wide_grade_receipt_sha256": _sha(grade_path),
        "wide_cache_metadata_sha256": _sha(cache / "metadata.json"),
        "wide_cache_main_sha256": _sha(main_path),
    }


def _fit_and_profile_inventory(
    panel: dict[str, Any], panel_sha: str, gru: dict[str, Any], tf: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Bind every owned fit/profile to its frozen panel command and source identity."""
    rows = observer._scientific_cells(gru, tf, panel)
    profiles: list[Path] = []
    for item in rows:
        source_panel, cell = item["panel"], item["cell"]
        is_student = "students" in source_panel
        expected_panel_sha = panel_sha if is_student else source_panel["_r5_panel_sha256"]
        profile_path = observer._phase_receipt(source_panel, cell, "profile")
        profile = _read(profile_path)
        if (
            profile is None
            or profile.get("status") != "completed"
            or profile.get("exit_code") != 0
            or profile.get("timed_out") is not False
        ):
            raise ValueError(f"{cell['cell_id']}: successful actual-shape profile is required")
        profile_owner = profile.get("owner", {})
        if (
            profile_owner.get("cell_id") != cell["cell_id"]
            or profile_owner.get("panel_sha256", profile_owner.get("cohort_panel_sha256"))
            != expected_panel_sha
        ):
            raise ValueError(f"{cell['cell_id']}: profile owner receipt identity differs")
        profiles.append(profile_path)
        fit = _read(observer._fit_path(item))
        if (
            fit is None
            or fit.get("status") != "completed"
            or fit.get("exit_code") != 0
            or fit.get("timed_out") is not False
        ):
            raise ValueError(f"{cell['cell_id']}: twelve-fit terminal gate is not complete")
        owner = fit.get("owner", {})
        if (
            owner.get("cell_id") != cell["cell_id"]
            or owner.get("run_name") != cell["run_name"]
            or owner.get("panel_sha256", owner.get("cohort_panel_sha256")) != expected_panel_sha
        ):
            raise ValueError(f"{cell['cell_id']}: fit owner receipt identity differs")
        if is_student:
            binding = panel["identity"]
            if any(
                owner.get(key) != binding[key]
                for key in ("source_binding_sha256", "data_binding_sha256", "split_sha256")
            ):
                raise ValueError(
                    f"{cell['cell_id']}: student fit source/data/split identity differs"
                )
    snapshot = _snapshot(panel, panel_sha, gru, tf)
    counts = {
        key: sum(row["family"] == key for row in snapshot)
        for key in (
            "wide_gru",
            "causal_transformer",
            "student_wide_gru",
            "student_causal_transformer",
        )
    }
    if counts != {
        "wide_gru": 6,
        "causal_transformer": 2,
        "student_wide_gru": 2,
        "student_causal_transformer": 2,
    }:
        raise ValueError(f"frozen fit inventory has unexpected family counts: {counts}")
    if any(row["status"] != "completed" for row in snapshot):
        raise ValueError("all twelve actual fit receipts must be terminal zero-exit before publish")
    return snapshot, profiles


def _file_entries(paths: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: dict[str, Path] = {}
    entries = []
    for source, name in paths:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"publication evidence is missing or unsafe: {source}")
        safe = PurePosixPath(name)
        if safe.is_absolute() or ".." in safe.parts or "\\" in name:
            raise ValueError(f"unsafe archive path: {name}")
        existing = seen.get(name)
        if existing is not None:
            if existing.resolve() != source.resolve():
                raise ValueError(f"archive member name collides: {name}")
            continue
        seen[name] = source
        entries.append((source, name))
    return entries


def _add(entries: list[tuple[Path, str]], source: Path, name: str) -> None:
    entries.append((source, name))


def _verify_or_write_archive(
    path: Path, entries: list[tuple[Path, str]], metadata: dict[str, Any]
) -> dict[str, Any]:
    expected = [
        {"path": name, "bytes": source.stat().st_size, "sha256": _sha(source)}
        for source, name in entries
    ]
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe existing archive: {path}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"existing archive CRC failed: {path}")
            manifest = json.loads(archive.read("MANIFEST.json"))
            if (
                any(manifest.get(key) != value for key, value in metadata.items())
                or manifest.get("files") != expected
            ):
                raise ValueError(f"existing archive differs from current verified inputs: {path}")
            for item in expected:
                if hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]:
                    raise ValueError(f"existing archive member changed: {item['path']}")
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
    result = publisher._write_archive(path, entries, metadata)
    return {key: result[key] for key in ("path", "bytes", "sha256")}


def _grade(
    family: str, grade_path: Path, panel: dict[str, Any], panel_sha: str, r5_family: dict[str, Any]
) -> tuple[dict[str, Any], list[Path]]:
    grade = _read(grade_path)
    if (
        grade is None
        or grade.get("schema") != "wnn.l4tc-fixed-student-grade.v1"
        or grade.get("status") != "complete"
        or grade.get("family") != family
        or grade.get("panel_sha256") != panel_sha
        or grade.get("selected_rows") != VALID_ROWS
        or grade.get("valid_sequences") != VALID_SEQUENCES
        or grade.get("endpoint_epoch") != 1
        or grade.get("fixed_weights") != {"TL3": 0.9, "candidate": 0.1}
        or len(grade.get("candidate_rows", [])) != 2
        or set(grade.get("student_member_rows", {})) != {"1001", "1002"}
        or set(grade.get("matched_student_deltas", {})) != {"1001", "1002"}
        or grade.get("admission_receipt_sha256") != panel.get("admission_receipt_sha256")
        or grade.get("teacher_admission_receipt_sha256")
        != panel["families"][family].get("teacher_admission_receipt_sha256")
    ):
        raise ValueError(
            f"{family}: canonical full-valid grade receipt is incomplete or mismatched"
        )
    if (
        r5_family.get("grade_status") != "complete"
        or Path(str(r5_family.get("grade_receipt_path", ""))).resolve() != grade_path.resolve()
        or r5_family.get("grade_receipt_sha256") != _sha(grade_path)
    ):
        raise ValueError(f"{family}: R5 completion does not pin the actual grade receipt")
    binding = panel["families"][family]
    cache = Path(str(binding["target_cache"]))
    metadata_path = cache / "metadata.json"
    metadata = _read(metadata_path)
    contract = grade.get("target_contract")
    if (
        metadata is None
        or grade.get("target_cache_path") != str(cache.resolve())
        or grade.get("target_cache_metadata_sha256") != _sha(metadata_path)
        or r5_family.get("target_metadata_sha256", r5_family.get("target_cache_metadata_sha256"))
        not in (None, _sha(metadata_path))
        or r5_family.get("target_cache_path") not in (None, str(cache.resolve()))
        or not isinstance(contract, dict)
        or contract.get("family") != family
        or contract.get("formula") != TARGET_FORMULA
        or contract.get("teacher_runs_equal_weight") != list(binding["fit_receipts"])
        or contract.get("teachers") != grade.get("target_teacher_records")
        or contract != metadata.get("target_contract")
        or metadata.get("base_cache", {}).get("metadata_sha256") != cs.PR60_CACHE_METADATA_SHA256
        or metadata.get("base_cache", {}).get("main_source_train_sha256")
        != cs.PR60_MAIN_TRAIN_SHA256
    ):
        raise ValueError(f"{family}: target-cache metadata/teacher contract differs")
    cache_review = Path(str(grade.get("cache_admission_receipt_path", "")))
    review = _read(cache_review)
    if (
        review is None
        or not str(review.get("status", "")).startswith("approved")
        or review.get("family") != family
        or review.get("target_cache_path") != str(cache.resolve())
        or review.get("target_metadata_sha256") != _sha(metadata_path)
        or grade.get("cache_admission_receipt_sha256") != _sha(cache_review)
    ):
        raise ValueError(f"{family}: independent target-cache admission is not valid")
    main_record = metadata.get("files", {}).get("main_source_train", {})
    main_relative = PurePosixPath(str(main_record.get("path", "")))
    if main_relative.is_absolute() or ".." in main_relative.parts or "\\" in str(main_relative):
        raise ValueError(f"{family}: target main-array path is unsafe")
    main_path = cache.joinpath(*main_relative.parts)
    if main_path.is_symlink() or not main_path.resolve().is_relative_to(cache.resolve()):
        raise ValueError(f"{family}: target main array resolves outside its cache")
    main_sha = _sha(main_path)
    main_bytes = main_path.stat().st_size
    if (
        main_record.get("sha256") != main_sha
        or main_record.get("bytes") != main_bytes
        or main_bytes != cs.TARGET_BYTES
        or review.get("target_arrays", {}).get("main_source_train_sha256") != main_sha
        or review.get("target_arrays", {}).get("main_source_train_bytes") != main_bytes
    ):
        raise ValueError(f"{family}: target main array differs from its cache review")
    expected_candidate_names = {
        f"TL3_plus_{family}_s1001",
        f"TL3_plus_{family}_s1002",
    }
    if {
        row.get("name") for row in grade["candidate_rows"] if isinstance(row, dict)
    } != expected_candidate_names:
        raise ValueError(f"{family}: grade candidates are not the two frozen student seeds")
    for row in grade["candidate_rows"]:
        member = str(row["name"])[len("TL3_plus_") :]
        if row.get("weights") != {"TL3": 0.9, member: 0.1}:
            raise ValueError(f"{family}: candidate blend weights are not the fixed 90/10 recipe")
    report_path = Path(str(grade.get("report_path", "")))
    manifest_path = Path(str(grade.get("manifest_path", "")))
    _same_file(
        report_path, grade.get("report_path"), grade.get("report_sha256"), f"{family} report"
    )
    _same_file(
        manifest_path,
        grade.get("manifest_path"),
        grade.get("manifest_sha256"),
        f"{family} manifest",
    )
    report = _read(report_path)
    if (
        report is None
        or report.get("schema") != "wnn.heterogeneous-ensemble-score.v1"
        or report.get("status") != "complete"
        or report.get("selected_rows") != VALID_ROWS
        or report.get("valid_sequences") != VALID_SEQUENCES
        or report.get("split_sha256") != panel["identity"]["split_sha256"]
        or report.get("input_manifest_sha256") != _sha(manifest_path)
        or report.get("rows") is None
        or report.get("matched_student_deltas") != grade.get("matched_student_deltas")
        or report.get("baseline_available") is not True
        or grade.get("baseline_row", {}).get("name") != "TL3"
        or [row for row in report.get("rows", []) if row.get("name", "").startswith("TL3_plus_")]
        != grade.get("candidate_rows")
    ):
        raise ValueError(f"{family}: canonical report lacks exact full-valid coverage")
    evidence = [grade_path, metadata_path, cache_review, report_path, manifest_path]
    teacher_panel_path = Path(str(binding["teacher_panel"]))
    teacher_panel = _read(teacher_panel_path)
    teacher_identity = teacher_panel.get("identity", {}) if teacher_panel else {}
    teacher_admission = Path(str(binding["teacher_admission_receipt"]))
    teacher_review = _read(teacher_admission)
    if (
        teacher_review is None
        or not str(teacher_review.get("status", "")).startswith("approved")
        or teacher_review.get("family") != family
        or teacher_review.get("teacher_panel_sha256") != _sha(teacher_panel_path)
        or teacher_review.get("source_binding_sha256")
        != teacher_identity.get("source_binding_sha256")
        or teacher_review.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or teacher_review.get("split_sha256") != panel["identity"]["split_sha256"]
    ):
        raise ValueError(f"{family}: teacher admission is unavailable")
    if _sha(teacher_admission) != binding.get("teacher_admission_receipt_sha256"):
        raise ValueError(f"{family}: teacher admission SHA changed")
    evidence.append(teacher_admission)
    teacher_artifacts = grade.get("teacher_fit_artifacts", [])
    student_artifacts = grade.get("student_fit_artifacts", [])
    expected_teachers = set(binding["fit_receipts"])
    student_cells = [cell for cell in panel["students"]["cells"] if cell["family"] == family]
    expected_students = {cell["run_name"] for cell in student_cells}
    if (
        len(teacher_artifacts) != FAMILY_RUNS[family]
        or {item.get("run_name") for item in teacher_artifacts} != expected_teachers
        or len(student_artifacts) != 2
        or {item.get("run_name") for item in student_artifacts} != expected_students
    ):
        raise ValueError(
            f"{family}: grade fit artifacts do not bind its exact two teachers and students"
        )
    expected_fit_paths = dict(binding["fit_receipts"])
    project = Path(panel["paths"]["project_root"])
    for cell in student_cells:
        expected_fit_paths[cell["run_name"]] = str(
            Path(panel["paths"]["runtime_root"]) / "receipts/fits" / f"{cell['cell_id']}.json"
        )
    for artifact in teacher_artifacts + student_artifacts:
        run = str(artifact.get("run_name", ""))
        is_teacher = artifact in teacher_artifacts
        for key in ("fit_receipt", "checkpoint"):
            path_key, sha_key = f"{key}_path", f"{key}_sha256"
            artifact_path = Path(str(artifact[path_key]))
            _same_file(artifact_path, artifact[path_key], artifact[sha_key], f"{family} {key}")
            if key == "fit_receipt":
                if artifact_path != Path(expected_fit_paths[run]).resolve():
                    raise ValueError(f"{family}: graded fit receipt path differs for {run}")
                evidence.append(artifact_path)
        if artifact.get("endpoint_epoch") != (3 if is_teacher else 1):
            raise ValueError(f"{family}: graded endpoint epoch differs")
        if is_teacher:
            checkpoint_name = "model_ep3.pt" if family == "wide_gru" else "model.pt"
        else:
            checkpoint_name = "model_ep1.pt"
        if (
            Path(str(artifact["checkpoint_path"])).resolve()
            != (project / "artifacts/runs" / run / checkpoint_name).resolve()
        ):
            raise ValueError(f"{family}: graded checkpoint path differs for {run}")
    for pred in grade.get("compact_prediction_artifacts", []):
        path = Path(str(pred["path"]))
        _same_file(path, pred["path"], pred["sha256"], f"{family} compact valid prediction")
        evidence.append(path)
    if len(grade.get("compact_prediction_artifacts", [])) != 2:
        raise ValueError(f"{family}: both compact full-valid student predictions must be pinned")
    expected_predictions = {f"pred_{cell['run_name']}_valid_scored.npy" for cell in student_cells}
    if {
        Path(item["path"]).name for item in grade["compact_prediction_artifacts"]
    } != expected_predictions:
        raise ValueError(
            f"{family}: compact prediction files differ from the frozen student members"
        )
    grade["_verified_target_main_sha256"] = main_sha
    return grade, evidence


def _completion(
    panel_path: Path, r5_path: Path, deadline: float
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[Path],
    list[tuple[Path, str]],
    list[tuple[Path, str]],
    dict[str, Any],
]:
    panel, panel_sha = cs.validate_panel(panel_path)
    runtime = Path(panel["paths"]["runtime_root"])
    state = _read(Path(panel["paths"]["shared_state"]))
    if state is None or state.get("hard_deadline_epoch") != deadline:
        raise ValueError("publisher deadline differs from the original shared teacher deadline")
    expected = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_COMPLETE.json"
    if r5_path.resolve() != expected.resolve():
        raise ValueError("R5 completion path is outside the frozen continuation runtime")
    complete = _read(r5_path)
    if (
        complete is None
        or complete.get("schema") != "wnn.l4-teacher-capacity-continuation-recovery-r5.v1"
        or complete.get("status") != "complete"
        or complete.get("panel_sha256") != panel_sha
        or complete.get("all_family_grades_complete") is not True
        or complete.get("checkpoint_endpoint_epoch") != 1
        or complete.get("fixed_weights") != {"TL3": 0.9, "candidate": 0.1}
        or not _complete_deadline_matches(complete.get("hard_deadline_epoch"), deadline)
        or not isinstance(complete.get("finished_epoch"), (int, float))
        or complete["finished_epoch"] > deadline
    ):
        raise ValueError("R5 final receipt is absent, incomplete, or outside its original deadline")
    identity = panel["identity"]
    if complete.get("source_binding_sha256") != identity["source_binding_sha256"]:
        raise ValueError("R5 completion changed the original panel source identity")
    admission_path = Path(str(complete.get("recovery_admission_receipt_path", "")))
    admission = _read(admission_path)
    if (
        admission is None
        or not str(admission.get("status", "")).startswith("approved")
        or _sha(admission_path) != complete.get("recovery_admission_receipt_sha256")
        or admission.get("continuation_panel_sha256") != panel_sha
        or admission.get("source_binding_sha256") != complete.get("recovery_source_binding_sha256")
        or any(
            admission.get(key) != identity.get(key)
            for key in (*COMMON_IDENTITY, "baseline_ancestry_sha256")
        )
    ):
        raise ValueError(
            "R5 recovery admission does not bind both source identities and shared data"
        )
    controller_ref = complete.get("controller_receipt")
    if not isinstance(controller_ref, dict):
        raise ValueError("R5 controller process receipt reference is missing")
    controller_path = Path(str(controller_ref.get("path", "")))
    controller = _read(controller_path)
    if (
        controller is None
        or controller.get("schema") != "wnn.multistage-owned-process.v1"
        or controller.get("status") != "completed"
        or controller.get("exit_code") != 0
        or controller.get("timed_out") is not False
        or controller.get("child") != controller_ref.get("child")
        or controller.get("command_sha256") != complete.get("controller_command_sha256")
        or controller.get("owner", {}).get("continuation_panel_sha256") != panel_sha
        or controller.get("owner", {}).get("source_binding_sha256")
        != identity["source_binding_sha256"]
        or controller.get("owner", {}).get("recovery_source_binding_sha256")
        != complete.get("recovery_source_binding_sha256")
        or controller.get("owner", {}).get("recovery_admission_receipt_sha256")
        != complete.get("recovery_admission_receipt_sha256")
        or any(
            controller.get("owner", {}).get(key) != identity[key]
            for key in (
                "data_binding_sha256",
                "split_sha256",
                "fit_members_sha256",
                "baseline_ancestry_sha256",
            )
        )
    ):
        raise ValueError("R5 controller has not reached its owned zero-exit terminal receipt")
    families = complete.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILY_RUNS):
        raise ValueError("R5 must bind both canonical student-family grades")
    gru_path = Path(panel["families"]["wide_gru"]["teacher_panel"])
    tf_path = Path(panel["families"]["causal_transformer"]["teacher_panel"])
    gru = _read(gru_path)
    tf = _read(tf_path)
    if (
        gru is None
        or tf is None
        or _sha(gru_path) != panel["families"]["wide_gru"]["teacher_panel_sha256"]
        or _sha(tf_path) != panel["families"]["causal_transformer"]["teacher_panel_sha256"]
    ):
        raise ValueError("actual teacher panel bytes differ from the continuation identity")
    _identity(panel, gru, "GRU teacher panel")
    _identity(panel, tf, "Transformer teacher panel")
    if (
        len(cs._teacher_cells(panel, "wide_gru")) != 4
        or len(cs._teacher_cells(panel, "causal_transformer")) != 2
    ):
        raise ValueError("R5 targets do not bind the frozen four-GRU/two-TF teacher subsets")
    gru["_r5_panel_sha256"], tf["_r5_panel_sha256"] = _sha(gru_path), _sha(tf_path)
    student_rows, profiles = _fit_and_profile_inventory(panel, panel_sha, gru, tf)
    if any(row["status"] == "failed" for row in student_rows):
        raise ValueError("a scientific fit failed; preserve partial artifacts and do not retry")
    grades, evidence = {}, []
    for family in FAMILY_RUNS:
        grade_path = runtime / "grades" / family / "grade_receipt.json"
        grades[family], family_files = _grade(
            family, grade_path, panel, panel_sha, families[family]
        )
        evidence.extend(family_files)
    evidence.extend([panel_path, r5_path, controller_path, admission_path, gru_path, tf_path])
    for key in (
        "storage_amendment_receipt_path",
        "r4_stop_receipt_path",
        "r3_start_receipt_path",
        "r3_failure_receipt_path",
    ):
        value = complete.get(key)
        if isinstance(value, str):
            evidence.append(Path(value))
    evidence.extend(
        [
            Path(panel["admission_receipt"]),
            Path(panel["paths"]["split_manifest"]),
            Path(panel["paths"]["shared_state"]),
        ]
    )
    for item in student_rows:
        evidence.append(Path(item["receipt_path"]))
        output = observer._output_dir(
            panel
            if item["family"].startswith("student_")
            else (gru if item["family"] == "wide_gru" else tf),
            next(
                row["cell"]
                for row in observer._scientific_cells(gru, tf, panel)
                if row["cell"]["run_name"] == item["run"]
            ),
        )
        for name in ("history.json", "provenance.json"):
            candidate = output / name
            if candidate.is_file() and not candidate.is_symlink():
                evidence.append(candidate)
    for profile in profiles:
        evidence.append(profile)
    evidence.extend(
        [
            runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_START.json",
            runtime / "receipts/continuation/wide_gru_GRADE_ADOPTION_R5.json",
            runtime / "receipts/continuation/causal_transformer_RECOVERY_COMPLETE_R5.json",
        ]
    )
    evidence_entries: list[tuple[Path, str]] = []
    for path in dict.fromkeys(evidence):
        try:
            name = path.relative_to(runtime).as_posix()
        except ValueError:
            name = f"external/{_sha(path)[:12]}_{path.name}"
        _add(evidence_entries, path, name)
    evidence_entries = _file_entries(evidence_entries)
    weights_entries = _weights(panel, grades["causal_transformer"])
    return (
        panel,
        complete,
        grades,
        student_rows,
        profiles,
        evidence_entries,
        weights_entries,
        {
            "gru": gru,
            "tf": tf,
            "panel_sha256": panel_sha,
            "controller_sha256": _sha(controller_path),
            "r5_complete_sha256": _sha(r5_path),
        },
    )


def _weights(panel: dict[str, Any], grade: dict[str, Any]) -> list[tuple[Path, str]]:
    from wnn.training.sequence_epoch_index import load_epoch_index

    entries: list[tuple[Path, str]] = []
    teacher_panel_path = Path(panel["families"]["causal_transformer"]["teacher_panel"])
    teacher_panel = _read(teacher_panel_path)
    for cell in teacher_panel["cells"]:
        output = Path(cell["fit_output"])
        run = str(cell["run_name"])
        index_path = output / "epoch_index.json"
        index = load_epoch_index(index_path)
        index_identity = index.get("identity", {})
        entries_index = index.get("entries")
        normalized_config = cell.get("normalized_config_sha256")
        if (
            index.get("completed_epochs") != 4
            or not isinstance(entries_index, list)
            or len(entries_index) != 4
            or (
                isinstance(normalized_config, str)
                and index_identity.get("config_sha256") != normalized_config
            )
        ):
            raise ValueError(f"{run}: expected four immutable native epoch checkpoints")
        for epoch, entry in enumerate(entries_index):
            if entry.get("epoch") != epoch or entry.get("completed_epochs") != epoch + 1:
                raise ValueError(f"{run}: native epoch index is not the exact 0..3 prefix")
            relative = PurePosixPath(str(entry["checkpoint_path"]))
            checkpoint = output.joinpath(*relative.parts).resolve()
            if not checkpoint.is_relative_to(output.resolve()) or _sha(checkpoint) != entry.get(
                "checkpoint_sha256"
            ):
                raise ValueError(f"{run}: native epoch checkpoint index/hash differs")
            _add(
                entries,
                checkpoint,
                f"weights/causal_transformer_teacher/{run}/{relative.as_posix()}",
            )
        _add(entries, index_path, f"evidence/teacher_epoch_index/{run}.json")
        model = output / "model.pt"
        expected = next(row for row in grade["teacher_fit_artifacts"] if row["run_name"] == run)
        _same_file(
            model, expected["checkpoint_path"], expected["checkpoint_sha256"], f"{run} fixed model"
        )
        _add(entries, model, f"weights/causal_transformer_teacher/{run}/model.pt")
    for artifact in grade["student_fit_artifacts"]:
        run = str(artifact["run_name"])
        output = Path(artifact["checkpoint_path"]).parent
        for name in ("model_ep0.pt", "model_ep1.pt"):
            path = output / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{run}: expected student epoch weight is missing: {name}")
            _add(entries, path, f"weights/causal_transformer_student/{run}/{name}")
        alias = output / "model.pt"
        if alias.is_file() and not alias.is_symlink():
            _add(entries, alias, f"weights/causal_transformer_student/{run}/model.pt")
    return _file_entries(entries)


def _grade_body(grades: dict[str, dict[str, Any]], prior: dict[str, Any]) -> str:
    lines = [
        "## PR61 R5 fixed-epoch continuation results",
        "",
        f"The earlier eight completed fits remain in [{PREVIOUS_TAG}]({prior['html_url']}); this release adds the two native Transformer teachers and two Transformer-family students.",
        "All twelve fit receipts are terminal zero-exit and bound to their frozen panels. The canonical official-valid partition is development-exposed; this is not an independent holdout, public score, or served-successor claim.",
        "",
        "### Canonical fixed 90/10 TL3/student blends",
        "",
        "| Family | Blend | WP | WMSE | Delta vs TL3 |",
        "|---|---|---:|---:|---:|",
    ]
    for family, grade in grades.items():
        for row in grade["candidate_rows"]:
            lines.append(
                f"| `{family}` | `{row['name']}` | {row['score']['wp']} | {row['wmse']['wmse']} | {row['delta_to_tl3']} |"
            )
    lines.extend(
        [
            "",
            "### Matched student/control members",
            "",
            "| Family | Seed | Student WP | Control WP | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, grade in grades.items():
        for seed, row in grade["matched_student_deltas"].items():
            lines.append(
                f"| `{family}` | {seed} | {row['student_wp']} | {row['control_wp']} | {row['delta_student_minus_control_wp']} |"
            )
    lines.extend(
        [
            "",
            "Weights are fixed at TL3 0.9 / candidate 0.1 and the student endpoint is epoch index 1. Full official valid remains exposed development data; no public/private performance is claimed.",
        ]
    )
    return "\n".join(lines)


def _previous_release(client: publisher.GitHub) -> dict[str, Any]:
    value = client.request("GET", f"/repos/{REPOSITORY}/releases/tags/{PREVIOUS_TAG}")
    if (
        not isinstance(value, dict)
        or value.get("tag_name") != PREVIOUS_TAG
        or value.get("draft") is not False
        or value.get("prerelease") is not True
        or value.get("author", {}).get("login") != LOGIN
        or not isinstance(value.get("html_url"), str)
    ):
        raise publisher.FinalizeError("qualified earlier eight-fit PR61 release is unavailable")
    assets = value.get("assets")
    if not isinstance(assets, list) or not any(
        isinstance(asset, dict)
        and asset.get("state") == "uploaded"
        and asset.get("name", "").endswith("weights_R1.zip")
        and str(asset.get("digest", "")).startswith("sha256:")
        for asset in assets
    ):
        raise publisher.FinalizeError("earlier eight-fit release lacks a verified weights archive")
    return value


def _publish(
    panel: dict[str, Any],
    complete: dict[str, Any],
    grades: dict[str, dict[str, Any]],
    evidence_entries: list[tuple[Path, str]],
    weights_entries: list[tuple[Path, str]],
    lineage: dict[str, Any],
    client: publisher.GitHub,
    head: str,
) -> dict[str, Any]:
    runtime = Path(panel["paths"]["runtime_root"])
    output = runtime / "publication"
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "PR61_L4_teacher_capacity_evidence_R5.zip"
    weights = output / "PR61_L4_teacher_capacity_weights_R5.zip"
    prior = _previous_release(client)
    source_bindings = {
        key: panel_value["identity"].get("source_binding_sha256")
        for key, panel_value in (
            ("gru", lineage["gru"]),
            ("transformer", lineage["tf"]),
            ("students", panel),
        )
    }
    metadata = {
        "schema": "wnn.l4tc-pr61-r5-results.v1",
        "continuation_panel_sha256": lineage["panel_sha256"],
        "r5_complete_sha256": lineage["r5_complete_sha256"],
        "source_bindings": source_bindings,
        "data_binding_sha256": panel["identity"]["data_binding_sha256"],
        "split_sha256": panel["identity"]["split_sha256"],
        "fit_members_sha256": panel["identity"]["fit_members_sha256"],
        "family_grade_targets": {
            family: {
                "grade_receipt_sha256": _sha(
                    Path(str(panel["paths"]["runtime_root"]))
                    / "grades"
                    / family
                    / "grade_receipt.json"
                ),
                "target_metadata_sha256": grade["target_cache_metadata_sha256"],
                "target_main_sha256": grade["_verified_target_main_sha256"],
                "target_formula": grade["target_contract"]["formula"],
            }
            for family, grade in grades.items()
        },
        "prior_eight_fit_release": {"tag": PREVIOUS_TAG, "url": prior["html_url"]},
        "grade_gate": "both canonical fixed-epoch grades and target-cache admissions complete",
    }
    weight_metadata = {**metadata, "kind": "new_transformer_teacher_and_student_weights"}
    body = _grade_body(grades, prior)
    _verify_archive_budget(panel, complete, evidence_entries, weights_entries, evidence, weights)
    evidence_result = _verify_or_write_archive(
        evidence, evidence_entries, {**metadata, "kind": "evidence"}
    )
    weights_result = _verify_or_write_archive(weights, weights_entries, weight_metadata)
    release = publisher._publish_release(
        client,
        [evidence, weights],
        head,
        body,
        repository_name=REPOSITORY,
        publisher_login=LOGIN,
        release_tag=TAG,
        release_name="PR61 L4 teacher-capacity R5 results — 2026-11-04",
    )
    return {"release": release, "archives": [evidence_result, weights_result], "body": body}


def _verify_archive_budget(
    panel: dict[str, Any],
    complete: dict[str, Any],
    evidence: list[tuple[Path, str]],
    weights: list[tuple[Path, str]],
    evidence_path: Path,
    weights_path: Path,
) -> None:
    mount = Path(str(panel["paths"].get("disk_mount", "/data2")))
    free = shutil.disk_usage(mount).free
    state = _read(Path(panel["paths"]["shared_state"])) or {}
    initial = int(state.get("initial_data2_free_bytes", 10_172_567_552))
    amendment = _read(Path(complete["storage_amendment_receipt_path"]))
    if amendment is None:
        raise FileNotFoundError("R5 storage amendment is missing")
    floor = int(amendment["minimum_free_data2_bytes"])
    cap = int(amendment["amended_output_cap_bytes"])
    new_bytes = sum(
        sum(source.stat().st_size for source, _ in entries)
        for path, entries in ((evidence_path, evidence), (weights_path, weights))
        if not path.exists()
    )
    upper = new_bytes + max(16 << 20, new_bytes // 100)
    used = max(0, initial - free)
    if free - upper < floor or used + upper > cap:
        raise OSError(
            f"R5 archive upper bound {upper} bytes violates free-space floor or amended output cap"
        )


def _poll(args: argparse.Namespace) -> int:
    panel, panel_sha = cs.validate_panel(args.panel)
    runtime = Path(panel["paths"]["runtime_root"])
    if args.token_file.resolve().is_relative_to(runtime.resolve()):
        raise ValueError("publication credential must remain outside the runtime tree")
    token_identity = publisher._token_identity(args.token_file)
    token = publisher._read_token(args.token_file, token_identity)
    client = publisher.GitHub(token, user_agent="WunderFunder-PR61-R5-results-publisher")
    state_path = runtime / "receipts/publication/PR61_R5_PUBLISHER_STATE_R1.json"
    state = _read(state_path) or {}
    if state and (state.get("panel_sha256") != panel_sha or state.get("deadline") != args.deadline):
        raise ValueError("existing publisher state belongs to another panel/deadline")
    gru_path, tf_path = (Path(panel["families"][family]["teacher_panel"]) for family in FAMILY_RUNS)
    gru, tf = _read(gru_path), _read(tf_path)
    if gru is None or tf is None:
        raise FileNotFoundError("actual frozen GRU and Transformer teacher panels are required")
    _identity(panel, gru, "GRU teacher panel")
    _identity(panel, tf, "Transformer teacher panel")
    gru["_r5_panel_sha256"], tf["_r5_panel_sha256"] = _sha(gru_path), _sha(tf_path)
    r5_failed = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R5_FAILED.json"
    r5_complete = args.r5_complete.resolve()
    last_post = float(state.get("last_post_epoch", 0))
    terminal_seen = set(state.get("terminal_fits", []))
    while True:
        snapshot = _snapshot(panel, panel_sha, gru, tf)
        failed = [row for row in snapshot if row["status"] == "failed"]
        newly_terminal = {
            f"{row['family']}:{row['run']}:{row['status']}" for row in snapshot if row["terminal"]
        } - terminal_seen
        recovery_failure = _read(r5_failed)
        if failed or recovery_failure is not None:
            detail = (
                f"Recovery failed in phase `{recovery_failure.get('phase')}`: "
                f"{recovery_failure.get('error_type')}: {recovery_failure.get('error')}"
                if recovery_failure
                else "One or more fit receipts are terminal failures."
            )
            body = _progress_body(
                snapshot, "R5 failed; no refit or release", detail, _profile_counts(panel, gru, tf)
            )
            _api_head(client)
            comment = publisher._post_comment(
                client,
                body,
                repository_name=REPOSITORY,
                pr_number=PR_NUMBER,
                publisher_login=LOGIN,
                idempotency_marker=COMMENT_MARKER,
            )
            _save_state(
                state_path, panel_sha, args.deadline, time.time(), snapshot, comment, "failed"
            )
            return 1
        complete = _read(r5_complete)
        controller_outcome = _r5_controller_outcome(complete) if complete is not None else "waiting"
        if complete is not None and controller_outcome == "failed":
            body = _progress_body(
                snapshot,
                "R5 controller failed after completion record; no release",
                "The owned controller receipt is not terminal zero-exit; preserve artifacts and do not refit.",
                _profile_counts(panel, gru, tf),
            )
            _api_head(client)
            comment = publisher._post_comment(
                client,
                body,
                repository_name=REPOSITORY,
                pr_number=PR_NUMBER,
                publisher_login=LOGIN,
                idempotency_marker=COMMENT_MARKER,
            )
            _save_state(
                state_path, panel_sha, args.deadline, time.time(), snapshot, comment, "failed"
            )
            return 1
        if complete is not None and controller_outcome == "completed":
            try:
                payload = _completion(args.panel, r5_complete, args.deadline)
                validated_panel, final, grades, fit_rows, profiles, evidence, weights, lineage = (
                    payload
                )
                if any(row["status"] != "completed" for row in fit_rows):
                    raise ValueError("R5 completion exists before all twelve fits became zero-exit")
                release_head = _api_head(client)
                result = _publish(
                    validated_panel, final, grades, evidence, weights, lineage, client, release_head
                )
                _api_head(client)
                comment = publisher._post_comment(
                    client,
                    result["body"] + f"\n\nRelease: [{TAG}]({result['release']['url']}).",
                    repository_name=REPOSITORY,
                    pr_number=PR_NUMBER,
                    publisher_login=LOGIN,
                    idempotency_marker=COMMENT_MARKER,
                )
                receipt_path = runtime / "receipts/publication/PR61_R5_PUBLICATION_R1.json"
                publisher._write_json(
                    receipt_path,
                    {
                        "schema": "wnn.l4tc-pr61-r5-publication.v1",
                        "status": "published",
                        "panel_sha256": panel_sha,
                        "r5_complete_sha256": _sha(r5_complete),
                        "hard_deadline_epoch": args.deadline,
                        "release": result["release"],
                        "archives": result["archives"],
                        "comment": comment,
                        "fit_count": EXPECTED_FITS,
                        "published_epoch": time.time(),
                    },
                )
                _save_state(
                    state_path,
                    panel_sha,
                    args.deadline,
                    time.time(),
                    snapshot,
                    comment,
                    "published",
                )
                return 0
            except OSError as error:
                if "archive upper bound" not in str(error):
                    raise
                if time.time() >= args.deadline:
                    body = _progress_body(
                        snapshot,
                        "R5 grades complete; publication blocked by storage budget",
                        str(error),
                    )
                    _api_head(client)
                    comment = publisher._post_comment(
                        client,
                        body,
                        repository_name=REPOSITORY,
                        pr_number=PR_NUMBER,
                        publisher_login=LOGIN,
                        idempotency_marker=COMMENT_MARKER,
                    )
                    _save_state(
                        state_path,
                        panel_sha,
                        args.deadline,
                        time.time(),
                        snapshot,
                        comment,
                        "pending_storage",
                    )
                    return 1
        hourly = time.time() - last_post >= 3_600
        if newly_terminal or hourly or not state:
            event = (
                "fit completion: " + ", ".join(sorted(newly_terminal))
                if newly_terminal
                else "hourly R5 status"
            )
            body = _progress_body(snapshot, event, profile_counts=_profile_counts(panel, gru, tf))
            _api_head(client)
            comment = publisher._post_comment(
                client,
                body,
                repository_name=REPOSITORY,
                pr_number=PR_NUMBER,
                publisher_login=LOGIN,
                idempotency_marker=COMMENT_MARKER,
            )
            terminal_seen.update(newly_terminal)
            last_post = time.time()
            state = _save_state(
                state_path,
                panel_sha,
                args.deadline,
                last_post,
                snapshot,
                comment,
                "running",
                terminal_seen,
            )
        if time.time() >= args.deadline:
            body = _progress_body(
                snapshot,
                "Original deadline reached; R5 incomplete",
                "No grade-complete release was created.",
                _profile_counts(panel, gru, tf),
            )
            _api_head(client)
            comment = publisher._post_comment(
                client,
                body,
                repository_name=REPOSITORY,
                pr_number=PR_NUMBER,
                publisher_login=LOGIN,
                idempotency_marker=COMMENT_MARKER,
            )
            _save_state(
                state_path, panel_sha, args.deadline, time.time(), snapshot, comment, "deadline"
            )
            return 1
        time.sleep(min(args.poll_seconds, max(1, int(args.deadline - time.time()))))


def _save_state(
    path: Path,
    panel_sha: str,
    deadline: float,
    epoch: float,
    snapshot: list[dict[str, Any]],
    comment: dict[str, Any],
    status: str,
    terminal_seen: set[str] | None = None,
) -> dict[str, Any]:
    previous = _read(path) or {}
    terminals = terminal_seen or set(previous.get("terminal_fits", []))
    value = {
        "schema": "wnn.l4tc-pr61-r5-publisher-state.v1",
        "status": status,
        "panel_sha256": panel_sha,
        "deadline": deadline,
        "last_post_epoch": epoch,
        "terminal_fits": sorted(terminals),
        "fit_statuses": snapshot,
        "last_comment": comment,
    }
    publisher._write_json(path, value)
    return value


def _progress_body(
    snapshot: list[dict[str, Any]],
    event: str,
    detail: str | None = None,
    profile_counts: dict[str, int] | None = None,
) -> str:
    completed = sum(row["status"] == "completed" for row in snapshot)
    lines = [
        "## PR61 R5 execution status",
        "",
        f"Event: {event}. Scientific fits completed: {completed}/{EXPECTED_FITS}.",
        "",
        "| Family | Run | Status | Phase | Epoch | Full-valid WP | Full-valid WMSE | Grad norm | Clip |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in snapshot:
        values = [row.get(key) for key in ("epoch", "valid_wp", "valid_wmse", "grad_norm", "clip")]
        rendered = " | ".join("?" if value is None else str(value) for value in values)
        lines.append(
            f"| `{row['family']}` | `{row['run']}` | `{row['status']}` | {row['phase']} | {rendered} |"
        )
    if profile_counts:
        lines.extend(
            [
                "",
                f"Profile receipts found: {sum(profile_counts.values())}; statuses: {profile_counts}.",
            ]
        )
    if detail:
        lines.extend(["", detail])
    lines.extend(
        [
            "",
            "Official valid is development-exposed. Fit failure is preserved; no automatic retry or new fit is authorized.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--r5-complete", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight_existing(args.panel), sort_keys=True, allow_nan=False))
        return 0
    if args.r5_complete is None or args.token_file is None or args.deadline is None:
        parser.error("polling requires --r5-complete, --token-file and --deadline")
    if args.poll_seconds < 10 or args.poll_seconds > 300 or not math.isfinite(args.deadline):
        parser.error("poll interval must be 10..300 seconds and deadline finite")
    return _poll(args)


if __name__ == "__main__":
    raise SystemExit(main())
