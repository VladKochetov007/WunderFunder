"""Publish bounded PR61 status updates and release only graded L4 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any  # JSON receipts have heterogeneous producer-defined fields.

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

from scripts import w18_l4_quality_finalize_20261104 as publisher  # noqa: E402

REPOSITORY = "VladKochetov007/WunderFunder"
PUBLISHER = "SemyonEpanov"
PR_NUMBER = 61
HEAD_REF = "experiment/l4-teacher-capacity-20261104"
RELEASE_TAG = "tabdata-pr61-l4-teacher-capacity-20261104-r1"
COMMENT_MARKER = "<!-- l4-teacher-capacity-pr61-status-r1 -->"
EXPECTED_FITS = 12
EXPECTED_FIT_GROUPS = 10_325
EXPECTED_VALID_SEQUENCES = 1_873
EXPECTED_VALID_ROWS = 3_528_273


def _read(path: Path) -> dict[str, Any] | None:
    """Read a regular JSON object, treating an absent in-progress receipt as pending."""
    if path.is_symlink() or not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"receipt path is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _credential_identity(credential: Path, runtime: Path) -> tuple[int, int]:
    """Require a real credential file outside the experiment output tree."""
    if credential.resolve().is_relative_to(runtime.resolve()):
        raise ValueError("credential must remain outside the experiment runtime and archives")
    identity = publisher._token_identity(credential)
    if identity is None:
        raise FileNotFoundError("external PR61 publication credential is missing or unsafe")
    return identity


def _verify_api_scope(client: publisher.GitHub) -> str:
    """Check credential and live PR identity before any comment or release call."""
    user = client.request("GET", "/user")
    if not isinstance(user, dict) or user.get("login") != PUBLISHER:
        raise publisher.FinalizeError("credential is not the authorized PR61 publisher")
    pull = client.request("GET", f"/repos/{REPOSITORY}/pulls/{PR_NUMBER}")
    if not isinstance(pull, dict):
        raise publisher.FinalizeError("GitHub PR61 response is malformed")
    head = pull.get("head")
    head_repo = head.get("repo") if isinstance(head, dict) else None
    sha = head.get("sha") if isinstance(head, dict) else None
    if (
        pull.get("number") != PR_NUMBER
        or pull.get("state") != "open"
        or not isinstance(head, dict)
        or head.get("ref") != HEAD_REF
        or not isinstance(head_repo, dict)
        or str(head_repo.get("full_name", "")).lower() != REPOSITORY.lower()
        or not isinstance(sha, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None
    ):
        raise publisher.FinalizeError("live pull request is outside the frozen PR61 scope")
    return sha.lower()


def _phase_receipt(panel: dict[str, Any], cell: dict[str, Any], phase: str) -> Path:
    explicit = cell.get(f"{phase}_process_receipt")
    if isinstance(explicit, str):
        return Path(explicit)
    runtime = Path(str(panel["paths"]["runtime_root"]))
    folder = "profiles" if phase == "profile" else "fits"
    if panel.get("schema") == "wnn.l4-teacher-capacity-transformer-panel.v1":
        return runtime / "receipts" / "transformer" / folder / f"{cell['cell_id']}.json"
    return runtime / "receipts" / folder / f"{cell['cell_id']}.json"


def _output_dir(panel: dict[str, Any], cell: dict[str, Any]) -> Path:
    explicit = cell.get("fit_output")
    if isinstance(explicit, str):
        return Path(explicit)
    return Path(str(panel["paths"]["project_root"])) / "artifacts" / "runs" / str(cell["run_name"])


def _terminal(receipt: dict[str, Any] | None) -> str:
    if receipt is None:
        return "not_started"
    status = str(receipt.get("status", "unknown"))
    if (
        status == "completed"
        and receipt.get("exit_code") == 0
        and receipt.get("timed_out") is False
    ):
        return "completed"
    if status in {"failed", "timed_out", "progress_callback_failed", "cancelled"}:
        return "failed"
    if type(receipt.get("exit_code")) is int and receipt["exit_code"] != 0:
        return "failed"
    return "running" if status == "running" else status


def _history_records(output: Path) -> list[dict[str, Any]]:
    path = output / "history.json"
    if path.is_symlink() or not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    history = value.get("history", value.get("epochs", value)) if isinstance(value, dict) else value
    if isinstance(history, list):
        return [row for row in history if isinstance(row, dict)]
    return []


def _jsonl(path: Path, limit: int = 50_000) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _metric(row: dict[str, Any], names: tuple[str, ...]) -> object:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    nested = row.get("metrics")
    if isinstance(nested, dict):
        for name in names:
            if name in nested and nested[name] is not None:
                return nested[name]
    return None


def _metrics(output: Path) -> list[dict[str, object]]:
    """Return completed epoch WP/WMSE and real gradient/clip diagnostics."""
    rows = _history_records(output)
    step_events = [
        row
        for row in _jsonl(output / "monitor_events.jsonl", limit=2_000)
        if row.get("kind") in {"train_step", "diagnostics"}
    ]
    latest_step_metrics = step_events[-1].get("metrics", {}) if step_events else {}
    latest_step_metrics = latest_step_metrics if isinstance(latest_step_metrics, dict) else {}
    evaluation_events = [
        row
        for row in _jsonl(output / "monitor_evaluations.jsonl")
        if row.get("kind") == "evaluation"
    ]
    if evaluation_events:
        # Native sequence evaluation journal is the epoch-complete source of record.
        by_epoch: dict[object, dict[str, Any]] = {}
        for event in evaluation_events:
            if event.get("role") in (None, "valid"):
                by_epoch[event.get("epoch")] = event
        rows = [by_epoch[key] for key in sorted(by_epoch, key=lambda value: str(value))]
    output_rows: list[dict[str, object]] = []
    provenance = _read(output / "provenance.json") or {}
    diagnostics = provenance.get("update_diagnostics", [])
    diagnostics_by_epoch = (
        {
            row.get("epoch_index", row.get("epoch")): row
            for row in diagnostics
            if isinstance(row, dict)
        }
        if isinstance(diagnostics, list)
        else {}
    )
    for index, row in enumerate(rows):
        wp = _metric(row, ("valid_wp", "weighted_pearson", "valid/score", "valid/wp"))
        wmse = _metric(row, ("valid_wmse", "wmse", "valid/wmse"))
        epoch_value = row.get("epoch", index)
        epoch = int(epoch_value) + (0 if row.get("kind") == "evaluation" else 1)
        grad: object = row.get("update_diagnostics") or row.get("grad_stats")
        if not grad:
            grad = diagnostics_by_epoch.get(epoch - 1, diagnostics_by_epoch.get(epoch))
        grad = grad if isinstance(grad, dict) else {}
        stats = grad.get("grad_stats", grad)
        stats = stats if isinstance(stats, dict) else {}
        norm = stats.get("total_q05_q50_q95")
        if isinstance(norm, list) and len(norm) == 3:
            norm = {"q05": norm[0], "q50": norm[1], "q95": norm[2]}
        clip = stats.get("clip", stats.get("clip_threshold"))
        clip_rate = stats.get("clip_rate")
        if not stats:
            norm = latest_step_metrics.get("optimizer/gradient_norm_preclip")
            clip = latest_step_metrics.get("optimizer/gradient_clip_threshold")
            clip_rate = latest_step_metrics.get("optimizer/was_clipped")
        if wp is not None or wmse is not None:
            output_rows.append(
                {
                    "epoch": epoch,
                    "valid_wp": wp,
                    "valid_wmse": wmse,
                    "grad_norm": norm,
                    "clip_threshold": clip,
                    "clip_rate_or_latest_was_clipped": clip_rate,
                }
            )
    if latest_step_metrics:
        norm = latest_step_metrics.get("optimizer/gradient_norm_preclip")
        clip = latest_step_metrics.get("optimizer/gradient_clip_threshold")
        was_clipped = latest_step_metrics.get("optimizer/was_clipped")
        if output_rows:
            if output_rows[-1].get("grad_norm") is None:
                output_rows[-1]["grad_norm"] = norm
                output_rows[-1]["clip_threshold"] = clip
                output_rows[-1]["clip_rate_or_latest_was_clipped"] = was_clipped
        elif norm is not None or clip is not None or was_clipped is not None:
            output_rows.append(
                {
                    "epoch": step_events[-1].get("epoch"),
                    "valid_wp": None,
                    "valid_wmse": None,
                    "grad_norm": norm,
                    "clip_threshold": clip,
                    "clip_rate_or_latest_was_clipped": was_clipped,
                }
            )
    return output_rows


def _scientific_cells(
    gru: dict[str, Any], tf: dict[str, Any], continuation: dict[str, Any] | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, panel in (("wide_gru", gru), ("causal_transformer", tf)):
        for cell in panel.get("cells", []):
            rows.append({"family": family, "panel": panel, "cell": cell})
    if continuation is not None:
        for cell in continuation.get("students", {}).get("cells", []):
            family = str(cell.get("family", ""))
            if family not in {"wide_gru", "causal_transformer"}:
                continue
            rows.append({"family": f"student_{family}", "panel": continuation, "cell": cell})
    return rows


def _fit_path(row: dict[str, Any]) -> Path:
    panel, cell = row["panel"], row["cell"]
    if "students" in panel:
        runtime = Path(str(panel["paths"]["runtime_root"]))
        return runtime / "receipts" / "fits" / f"{cell['cell_id']}.json"
    return _phase_receipt(panel, cell, "fit")


def _profile_receipts(
    runtime: Path,
    gru: dict[str, Any],
    tf: dict[str, Any],
    continuation: dict[str, Any] | None,
) -> list[Path]:
    candidates: set[Path] = set()
    for panel in (gru, tf):
        for cell in panel.get("cells", []):
            candidates.add(_phase_receipt(panel, cell, "profile"))
    # The Transformer engineering wave used several approved profile configs;
    # inventory its dedicated receipt directory so the displayed count is actual.
    tf_profile_dir = runtime / "receipts" / "transformer" / "profiles"
    if tf_profile_dir.is_dir() and not tf_profile_dir.is_symlink():
        candidates.update(tf_profile_dir.glob("*.json"))
    gru_profile_dir = runtime / "receipts" / "profiles"
    if gru_profile_dir.is_dir() and not gru_profile_dir.is_symlink():
        candidates.update(gru_profile_dir.glob("*.json"))
    if continuation is not None:
        for cell in continuation.get("students", {}).get("cells", []):
            candidates.add(_phase_receipt(continuation, cell, "profile"))
    return sorted(candidates)


def _snapshot(gru: dict[str, Any], tf: dict[str, Any], continuation_path: Path) -> dict[str, Any]:
    continuation = _read(continuation_path)
    runtime = Path(str(gru["paths"]["runtime_root"]))
    if continuation is not None:
        if (
            continuation.get("schema") != "wnn.l4-teacher-capacity-continuation.v1"
            or continuation.get("status") != "frozen"
            or Path(str(continuation.get("paths", {}).get("runtime_root", ""))) != runtime
        ):
            raise ValueError("continuation panel is outside the frozen PR61 runtime")
        for identity_key in ("source_binding_sha256", "data_binding_sha256", "split_sha256"):
            if continuation.get("identity", {}).get(identity_key) != gru.get("identity", {}).get(
                identity_key
            ):
                raise ValueError(f"continuation panel identity differs on {identity_key}")
    state = _read(runtime / "GRU_STATE_R1.json")
    if state is None or not isinstance(state.get("hard_deadline_epoch"), (int, float)):
        raise ValueError("shared GRU_STATE_R1 hard deadline is absent or malformed")
    fits = []
    for row in _scientific_cells(gru, tf, continuation):
        cell, panel = row["cell"], row["panel"]
        receipt_path = _fit_path(row)
        receipt = _read(receipt_path)
        output = _output_dir(panel, cell)
        phase = _terminal(receipt)
        epoch_metrics = _metrics(output) if output.is_dir() and not output.is_symlink() else []
        fits.append(
            {
                "family": row["family"],
                "run": str(cell["run_name"]),
                "cell_id": str(cell["cell_id"]),
                "seed": cell.get("seed"),
                "status": phase,
                "phase": (
                    "fixed endpoint available"
                    if phase == "completed"
                    else "fit failed"
                    if phase == "failed"
                    else f"training, epoch {epoch_metrics[-1]['epoch'] + 1}"
                    if phase == "running" and epoch_metrics
                    else "training"
                    if phase == "running"
                    else "queued"
                ),
                "receipt": str(receipt_path),
                "epochs": epoch_metrics,
                "terminal": phase in {"completed", "failed"},
            }
        )
    profile_paths = _profile_receipts(runtime, gru, tf, continuation)
    profiles = []
    for path in profile_paths:
        receipt = _read(path)
        if receipt is None or "status" not in receipt:
            continue
        # Profile inventory is restricted to these two frozen panels and the
        # continuation's four cell IDs; this avoids unrelated profiles nearby.
        name = path.stem
        if not name.startswith(("l4tc_", "tf", "gru", "student_")):
            continue
        profiles.append({"path": str(path), "status": _terminal(receipt)})
    grades = []
    for family in ("wide_gru", "causal_transformer"):
        grade = _read(runtime / "grades" / family / "grade_receipt.json")
        if grade is not None and grade.get("schema") == "wnn.l4tc-fixed-student-grade.v1":
            grades.append(
                {
                    "family": family,
                    "status": grade.get("status"),
                    "baseline": grade.get("baseline_row"),
                    "candidates": grade.get("candidate_rows", []),
                    "matched_student_deltas": grade.get("matched_student_deltas", {}),
                }
            )
    return {
        "observed_at_epoch": time.time(),
        "hard_deadline_epoch": float(state["hard_deadline_epoch"]),
        "profile_receipts": profiles,
        "profile_count": len(profiles),
        "profile_completed": sum(row["status"] == "completed" for row in profiles),
        "fits": fits,
        "fit_count": len(fits),
        "fit_terminal": sum(row["terminal"] for row in fits),
        "fit_completed": sum(row["status"] == "completed" for row in fits),
        "grades": grades,
        "continuation_status": continuation.get("status") if continuation else "pending_panel",
        "continuation_panel_path": str(continuation_path),
    }


def _status_text(snapshot: dict[str, Any], event: str, release: dict[str, Any] | None) -> str:
    remaining = max(0, int(snapshot["hard_deadline_epoch"] - time.time()))
    lines = [
        "## L4 teacher-capacity continuation status",
        f"Event: `{event}`. Remaining campaign time: `{remaining // 3600}h {(remaining % 3600) // 60}m`.",
        f"Engineering profiles: **{snapshot['profile_completed']}/{snapshot['profile_count']} completed** (receipt inventory).",
        f"Scientific fits: **{snapshot['fit_completed']}/{EXPECTED_FITS} completed**, {snapshot['fit_terminal']}/{snapshot['fit_count']} terminal.",
        f"Student continuation: `{snapshot['continuation_status']}`.",
        "",
        "| Family / run | Status | Phase | Epoch validation WP / WMSE | Grad norm | Clip |",
        "|---|---|---|---|---|---|",
    ]
    for fit in snapshot["fits"]:
        epochs = fit["epochs"]
        metric_text = (
            "; ".join(
                f"ep{item['epoch']} WP={item['valid_wp']} WMSE={item['valid_wmse']}"
                for item in epochs
            )
            or "evaluation pending"
        )
        norm_text = (
            "; ".join(
                f"ep{item['epoch']} {json.dumps(item['grad_norm'], allow_nan=False)}"
                for item in epochs
                if item.get("grad_norm") is not None
            )
            or "pending"
        )
        clip_text = (
            "; ".join(
                f"ep{item['epoch']} threshold={item['clip_threshold']} rate/latest={item['clip_rate_or_latest_was_clipped']}"
                for item in epochs
                if item.get("clip_threshold") is not None
                or item.get("clip_rate_or_latest_was_clipped") is not None
            )
            or "pending"
        )
        lines.append(
            f"| `{fit['family']} / {fit['run']}` | `{fit['status']}` | `{fit['phase']}` | {metric_text} | {norm_text} | {clip_text} |"
        )
    lines.extend(
        [
            "",
            "Validation is the full official chronological valid partition and remains development-exposed; these updates report observed metrics without selecting a best epoch.",
        ]
    )
    if snapshot["grades"]:
        lines.extend(
            [
                "",
                "### Canonical fixed-epoch student grades",
                "Weights are fixed at TL3 0.9 / candidate 0.1; official valid remains development-exposed.",
                "| Family | Status | Blend | WP | WMSE | Delta WP to TL3 |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for grade in snapshot["grades"]:
            for candidate in grade["candidates"]:
                score = candidate.get("score", {})
                wmse = candidate.get("wmse", {})
                lines.append(
                    f"| `{grade['family']}` | `{grade['status']}` | `{candidate.get('name')}` | {score.get('wp')} | {wmse.get('wmse')} | {candidate.get('delta_to_tl3')} |"
                )
    if release is None:
        lines.append(
            "Weights/evidence release: **pending** until both canonical family grades and target-cache lineage receipts are complete."
        )
    else:
        lines.append(f"Verified release: [{release['tag']}]({release['url']}).")
    return "\n".join(lines)


def _validate_grade_gate(
    continuation_path: Path, continuation: dict[str, Any], panel: dict[str, Any], panel_sha: str
) -> list[Path] | None:
    """Require complete canonical grades and their independently approved cache receipts."""
    runtime = Path(str(panel["paths"]["runtime_root"]))
    complete_path = runtime / "receipts" / "continuation" / "STUDENT_CONTINUATION_COMPLETE_R1.json"
    complete = _read(complete_path)
    if (
        complete is None
        or complete.get("schema") != "wnn.l4-teacher-capacity-continuation-state.v1"
        or complete.get("status") != "complete"
        or complete.get("panel_sha256") != panel_sha
        or complete.get("all_family_grades_complete") is not True
        or complete.get("selected_rows") not in (None, EXPECTED_VALID_ROWS)
    ):
        return None
    if _sha(continuation_path) != panel_sha:
        raise ValueError("continuation panel changed after its hash was recorded")
    identity = panel.get("identity", {})
    admission_path = Path(str(panel.get("admission_receipt", "")))
    admission = _read(admission_path)
    if (
        admission is None
        or not str(admission.get("status", "")).startswith("approved")
        or _sha(admission_path) != panel.get("admission_receipt_sha256")
        or admission.get("continuation_panel_sha256") not in (None, panel_sha)
        or any(
            admission.get(key) != identity.get(key)
            for key in (
                "source_binding_sha256",
                "data_binding_sha256",
                "split_sha256",
                "fit_members_sha256",
                "baseline_ancestry_sha256",
            )
        )
    ):
        return None
    grade_paths: list[Path] = []
    for family in ("wide_gru", "causal_transformer"):
        binding = panel.get("families", {}).get(family)
        if not isinstance(binding, dict):
            return None
        grade = runtime / "grades" / family / "grade_receipt.json"
        report = runtime / "grades" / family / "report.json"
        grade_value, report_value = _read(grade), _read(report)
        if (
            grade_value is None
            or report_value is None
            or grade_value.get("schema") != "wnn.l4tc-fixed-student-grade.v1"
            or grade_value.get("status") != "complete"
            or grade_value.get("family") != family
            or grade_value.get("panel_sha256") != panel_sha
            or grade_value.get("selected_rows") != EXPECTED_VALID_ROWS
            or grade_value.get("valid_sequences") != EXPECTED_VALID_SEQUENCES
            or report_value.get("schema") != "wnn.heterogeneous-ensemble-score.v1"
            or report_value.get("status") != "complete"
            or report_value.get("selected_rows") != EXPECTED_VALID_ROWS
            or report_value.get("valid_sequences") != EXPECTED_VALID_SEQUENCES
        ):
            return None
        cache_metadata = Path(str(binding["target_cache"])) / "metadata.json"
        cache_review_path = Path(str(binding["cache_admission_receipt"]))
        cache_review = _read(cache_review_path)
        teacher_review_path = Path(str(binding["teacher_admission_receipt"]))
        teacher_review = _read(teacher_review_path)
        teacher_panel_path = Path(str(binding["teacher_panel"]))
        if (
            cache_review is None
            or not str(cache_review.get("status", "")).startswith("approved")
            or cache_review.get("family") != family
            or cache_review.get("continuation_panel_sha256") != panel_sha
            or cache_review.get("source_binding_sha256") != identity.get("source_binding_sha256")
            or cache_review.get("data_binding_sha256") != identity.get("data_binding_sha256")
            or cache_review.get("split_sha256") != identity.get("split_sha256")
            or cache_review.get("target_metadata_sha256") != _sha(cache_metadata)
            or grade_value.get("target_cache_metadata_sha256") != _sha(cache_metadata)
            or Path(str(grade_value.get("target_cache_path", ""))).resolve()
            != Path(str(binding["target_cache"])).resolve()
            or grade_value.get("cache_admission_receipt_sha256") != _sha(cache_review_path)
            or teacher_review is None
            or not str(teacher_review.get("status", "")).startswith("approved")
            or grade_value.get("teacher_admission_receipt_sha256") != _sha(teacher_review_path)
            or grade_value.get("teacher_admission_receipt_sha256")
            != binding.get("teacher_admission_receipt_sha256")
            or _sha(teacher_panel_path) != binding.get("teacher_panel_sha256")
        ):
            return None
        target_contract = grade_value.get("target_contract")
        target_metadata_value = _read(cache_metadata) or {}
        positions = (
            target_contract.get("fit_source_positions", [])
            if isinstance(target_contract, dict)
            else []
        )
        positions_digest = (
            hashlib.sha256(struct.pack(f"<{len(positions)}q", *positions)).hexdigest()
            if isinstance(positions, list) and all(type(value) is int for value in positions)
            else None
        )
        if (
            not isinstance(target_contract, dict)
            or target_metadata_value.get("cache_profile") != "l4_teacher_capacity_stage2_stream_v2"
            or target_contract.get("formula")
            != "float32(0.8*T_old + 0.2*0.8125*mean(float32_clip(each_new_teacher,-2,2)))"
            or target_contract.get("source_train_index_rows") != 10_607
            or len(positions) != EXPECTED_FIT_GROUPS
            or len(set(positions)) != EXPECTED_FIT_GROUPS
            or any(value < 0 or value >= 10_607 for value in positions)
            or target_contract.get("fit_source_positions_sha256") != positions_digest
            or len(target_contract.get("fit_groups", [])) != EXPECTED_FIT_GROUPS
            or not isinstance(target_contract.get("source_train_index_sha256"), str)
            or not isinstance(target_contract.get("teachers"), list)
            or len(target_contract["teachers"]) != 4
            or target_contract.get("teacher_runs_equal_weight") != list(binding["fit_receipts"])
        ):
            return None
        family_results = complete.get("families", [])
        family_result = (
            next(
                (
                    item
                    for item in family_results
                    if isinstance(item, dict) and item.get("family") == family
                ),
                None,
            )
            if isinstance(family_results, list)
            else None
        )
        manifest_path = runtime / "grades" / family / "manifest.json"
        if (
            family_result is None
            or family_result.get("grade_receipt_sha256") != _sha(grade)
            or grade_value.get("manifest_sha256") != _sha(manifest_path)
            or grade_value.get("report_sha256") != _sha(report)
        ):
            return None
        grade_paths.extend((grade, report, runtime / "grades" / family / "manifest.json"))
        for row in grade_value.get("teacher_fit_artifacts", []) + grade_value.get(
            "student_fit_artifacts", []
        ):
            for key in ("fit_receipt_path", "checkpoint_path"):
                source = Path(str(row[key]))
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or _sha(source) != row[f"{key.removesuffix('_path')}_sha256"]
                ):
                    raise ValueError(
                        f"graded weights/evidence changed after canonical grading: {source.name}"
                    )
                grade_paths.append(source)
        grade_paths.extend((cache_metadata, cache_review_path))
    return grade_paths


def _archives(
    runtime: Path,
    gru_panel_path: Path,
    tf_panel_path: Path,
    continuation_path: Path,
    state_path: Path,
    evidence_paths: list[Path],
) -> list[Path]:
    output = runtime / "publication"
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "PR61_L4_teacher_capacity_evidence_R1.zip"
    weights = output / "PR61_L4_teacher_capacity_weights_R1.zip"
    evidence_entries: list[tuple[Path, str]] = []
    for path in dict.fromkeys(
        [gru_panel_path, tf_panel_path, continuation_path, state_path, *evidence_paths]
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"graded evidence input is absent or unsafe: {path}")
        if path.suffix in {".pt", ".safetensors"}:
            continue
        try:
            name = path.relative_to(runtime).as_posix()
        except ValueError:
            name = f"external/{path.name}"
        publisher._add_file(evidence_entries, path, name)
    weights_entries: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path in evidence_paths:
        if path.name.endswith((".pt", ".safetensors")) and path not in seen:
            seen.add(path)
            weights_entries.append((path, f"weights/{path.parent.name}/{path.name}"))
    metadata = {
        "schema": "wnn.l4tc-pr61-archives.v1",
        "grade_gate": "canonical_complete_and_cache_lineage_approved",
    }
    if evidence.exists() or weights.exists():
        if not all(path.is_file() and not path.is_symlink() for path in (evidence, weights)):
            raise ValueError("only one final archive exists; preserve and inspect before retry")
        for archive_path, kind, entries in (
            (evidence, "evidence", evidence_entries),
            (weights, "fixed_endpoint_weights", weights_entries),
        ):
            expected = [
                {
                    "path": name,
                    "bytes": source.stat().st_size,
                    "sha256": publisher._hash_file(source),
                }
                for source, name in entries
            ]
            with zipfile.ZipFile(archive_path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"existing archive CRC check failed: {archive_path.name}")
                manifest = json.loads(archive.read("MANIFEST.json"))
                if (
                    manifest.get("schema") != metadata["schema"]
                    or manifest.get("kind") != kind
                    or manifest.get("grade_gate") != metadata["grade_gate"]
                    or manifest.get("files") != expected
                ):
                    raise ValueError(
                        f"existing archive manifest differs from current grades: {archive_path.name}"
                    )
                for member in expected:
                    digest = hashlib.sha256()
                    with archive.open(member["path"]) as stream:
                        for block in iter(lambda: stream.read(1 << 20), b""):
                            digest.update(block)
                    if digest.hexdigest() != member["sha256"]:
                        raise ValueError(f"existing archive member SHA differs: {member['path']}")
        return [evidence, weights]
    publisher._write_archive(evidence, evidence_entries, {**metadata, "kind": "evidence"})
    publisher._write_archive(
        weights, weights_entries, {**metadata, "kind": "fixed_endpoint_weights"}
    )
    return [evidence, weights]


def _archive_budget_allows(
    panel: dict[str, Any], runtime: Path, evidence_paths: list[Path]
) -> bool:
    """Reserve the frozen output cap and free-space floor before writing archives."""
    paths = panel["paths"]
    budget = panel.get("budget", {})
    mount = Path(str(paths.get("disk_mount", "/data2")))
    free = shutil.disk_usage(mount).free
    minimum_free = int(budget.get("minimum_free_bytes", 3_000_000_000))
    output_cap = int(budget.get("max_output_bytes", 6_500_000_000))
    state = _read(runtime / "GRU_STATE_R1.json") or {}
    initial_free = int(state.get("initial_data2_free_bytes", 10_172_567_552))
    current_output = max(0, initial_free - free)
    archive_paths = [
        runtime / "publication" / "PR61_L4_teacher_capacity_evidence_R1.zip",
        runtime / "publication" / "PR61_L4_teacher_capacity_weights_R1.zip",
    ]
    if all(path.is_file() and not path.is_symlink() for path in archive_paths):
        estimated_new_bytes = 0
    else:
        estimated_new_bytes = (
            sum(
                path.stat().st_size
                for path in dict.fromkeys(evidence_paths)
                if path.is_file() and not path.is_symlink()
            )
            + 1_000_000
            + 1024 * len(evidence_paths)
        )
    return (
        free - estimated_new_bytes >= minimum_free
        and current_output + estimated_new_bytes <= output_cap
    )


def observe(
    gru_path: Path,
    tf_path: Path,
    continuation_path: Path,
    credential: Path,
    poll_seconds: int = 30,
) -> None:
    """Poll receipts and publish hourly/terminal PR61 updates until deadline/release."""
    if poll_seconds < 10 or poll_seconds > 300:
        raise ValueError("poll interval must be 10–300 seconds")
    gru = _read(gru_path)
    tf = _read(tf_path)
    if gru is None or tf is None:
        raise FileNotFoundError("actual GRU/Transformer teacher panels are required")
    if (
        gru.get("schema") != "wnn.l4-teacher-capacity-gru.v1"
        or tf.get("schema") != "wnn.l4-teacher-capacity-transformer-panel.v1"
    ):
        raise ValueError("observer requires the frozen actual GRU and Transformer teacher panels")
    if len(gru.get("cells", [])) != 6 or len(tf.get("cells", [])) != 2:
        raise ValueError(
            "observer panel cell inventory differs from the frozen 6-GRU/2-TF teacher set"
        )
    for identity_key in ("source_binding_sha256", "data_binding_sha256", "split_sha256"):
        if gru.get("identity", {}).get(identity_key) != tf.get("identity", {}).get(identity_key):
            raise ValueError(f"teacher panels disagree on {identity_key}")
    runtime = Path(str(gru["paths"]["runtime_root"]))
    if Path(str(tf["paths"]["runtime_root"])) != runtime:
        raise ValueError("teacher panels disagree on runtime root")
    state = _read(runtime / "GRU_STATE_R1.json")
    if state is None or state.get("panel_sha256") != _sha(gru_path):
        raise ValueError("shared hard-deadline state is not bound to the actual GRU panel")
    state_path = runtime / "receipts" / "publication" / "PR61_OBSERVER_STATE_R1.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    token_identity = _credential_identity(credential, runtime)
    try:
        token = publisher._read_token(credential, token_identity)
        client = publisher.GitHub(token, user_agent="WunderFunder-L4-PR61-observer")
        pull_head = _verify_api_scope(client)
        state = _read(state_path) or {}
        last_post = float(state.get("last_post_epoch", 0.0))
        terminal_seen = set(state.get("terminal_fits", []))
        release: dict[str, Any] | None = (
            state.get("release") if isinstance(state.get("release"), dict) else None
        )
        panel_value = _read(continuation_path)
        continuation_sha = _sha(continuation_path) if panel_value is not None else ""
        while True:
            snapshot = _snapshot(gru, tf, continuation_path)
            newly_terminal = {
                f"{fit['family']}:{fit['run']}:{fit['status']}"
                for fit in snapshot["fits"]
                if fit["terminal"]
            } - terminal_seen
            hourly = time.time() - last_post >= 3_600
            if newly_terminal or hourly or not state:
                event = (
                    "fit completion: " + ", ".join(sorted(newly_terminal))
                    if newly_terminal
                    else "hourly status"
                )
                body = _status_text(snapshot, event, release)
                _verify_api_scope(client)
                result = publisher._post_comment(
                    client,
                    body,
                    repository_name=REPOSITORY,
                    pr_number=PR_NUMBER,
                    publisher_login=PUBLISHER,
                    idempotency_marker=COMMENT_MARKER,
                )
                terminal_seen.update(newly_terminal)
                last_post = time.time()
                state.update(
                    {
                        "last_comment": result,
                        "last_post_epoch": last_post,
                        "terminal_fits": sorted(terminal_seen),
                    }
                )
                publisher._write_json(
                    state_path, {"schema": "wnn.l4tc-pr61-observer-state.v1", **state}
                )
            retry_grade_gate = (
                state.get("publication_status") != "pending_storage_budget"
                or time.time() - float(state.get("last_budget_check_epoch", 0.0)) >= 3_600
            )
            if release is None and panel_value is not None and retry_grade_gate:
                grade_paths = _validate_grade_gate(
                    continuation_path, panel_value, panel_value, continuation_sha
                )
                if grade_paths is not None:
                    complete_state = (
                        runtime
                        / "receipts"
                        / "continuation"
                        / "STUDENT_CONTINUATION_COMPLETE_R1.json"
                    )
                    grade_paths.extend((gru_path, tf_path, continuation_path, complete_state))
                    grade_paths.extend(
                        Path(str(panel_value[key]))
                        for key in ("admission_receipt",)
                        if panel_value.get(key)
                    )
                    for family_binding in panel_value.get("families", {}).values():
                        for key in ("teacher_panel", "teacher_admission_receipt"):
                            if family_binding.get(key):
                                grade_paths.append(Path(str(family_binding[key])))
                    for receipt_path in _profile_receipts(runtime, gru, tf, panel_value):
                        if receipt_path.is_file() and not receipt_path.is_symlink():
                            grade_paths.append(receipt_path)
                    for evidence_path in tuple(grade_paths):
                        if evidence_path.suffix in {".pt", ".safetensors"}:
                            for sidecar in (
                                "history.json",
                                "provenance.json",
                                "monitor_evaluations.jsonl",
                            ):
                                candidate = evidence_path.parent / sidecar
                                if candidate.is_file() and not candidate.is_symlink():
                                    grade_paths.append(candidate)
                    for folder in ("stage2", "teacher_export", "continuation"):
                        receipt_dir = runtime / "receipts" / folder
                        if receipt_dir.is_dir() and not receipt_dir.is_symlink():
                            grade_paths.extend(
                                path
                                for path in sorted(receipt_dir.glob("*.json"))
                                if path.is_file() and not path.is_symlink()
                            )
                    if not _archive_budget_allows(panel_value, runtime, grade_paths):
                        state["publication_status"] = "pending_storage_budget"
                        state["last_budget_check_epoch"] = time.time()
                        publisher._write_json(
                            state_path, {"schema": "wnn.l4tc-pr61-observer-state.v1", **state}
                        )
                        if time.time() >= snapshot["hard_deadline_epoch"]:
                            return
                        time.sleep(poll_seconds)
                        continue
                    pull_head = _verify_api_scope(client)
                    archives = _archives(
                        runtime, gru_path, tf_path, continuation_path, complete_state, grade_paths
                    )
                    release = publisher._publish_release(
                        client,
                        archives,
                        pull_head,
                        "L4 teacher-capacity continuation: fixed-epoch canonical full-valid grades, matched controls, and approved target-cache lineage. Full official validation is development-exposed.",
                        repository_name=REPOSITORY,
                        publisher_login=PUBLISHER,
                        release_tag=RELEASE_TAG,
                        release_name="L4 teacher-capacity results — 2026-11-04",
                    )
                    state["publication_status"] = "published"
                    state["release"] = release
                    state["release_head_sha"] = pull_head
                    publisher._write_json(
                        state_path, {"schema": "wnn.l4tc-pr61-observer-state.v1", **state}
                    )
                    body = _status_text(snapshot, "canonical grades complete", release)
                    _verify_api_scope(client)
                    result = publisher._post_comment(
                        client,
                        body,
                        repository_name=REPOSITORY,
                        pr_number=PR_NUMBER,
                        publisher_login=PUBLISHER,
                        idempotency_marker=COMMENT_MARKER,
                    )
                    state.update({"last_comment": result, "last_post_epoch": time.time()})
                    publisher._write_json(
                        state_path, {"schema": "wnn.l4tc-pr61-observer-state.v1", **state}
                    )
                    return
            if time.time() >= snapshot["hard_deadline_epoch"]:
                return
            time.sleep(
                min(poll_seconds, max(1, int(snapshot["hard_deadline_epoch"] - time.time())))
            )
    finally:
        if not publisher._unlink_token(credential, token_identity):
            raise publisher.FinalizeError(
                "credential inode changed; refusing to remove a replacement file"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru-panel", required=True, type=Path)
    parser.add_argument("--tf-panel", required=True, type=Path)
    parser.add_argument("--continuation-panel", required=True, type=Path)
    parser.add_argument("--credential", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    observe(
        args.gru_panel, args.tf_panel, args.continuation_panel, args.credential, args.poll_seconds
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        sys.stderr.write(f"PR61 observer stopped: {type(error).__name__}: {error}\n")
        raise SystemExit(2) from error
