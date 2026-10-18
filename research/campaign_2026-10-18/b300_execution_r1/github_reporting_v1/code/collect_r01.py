"""Collect receipt-verified R01 endpoint scores without running training or grading."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

AUTHORITY_SHA256 = "364c1296b8f49712e2d8c5f70b9bdc18a9e806d097cb6ee8da47e3547d69f8d5"
CONTROL_BUNDLE_SHA256 = "89ccb760cdb05609bbce5239274c407922169dfbcfd1bccd44935fc1241ab238"
PANEL_SHA256 = "471f465cc1d13fc6e5153a4c998390bb14e06823fed4c6721f929ce95fee47fb"
DECISION_SHA256 = "32a9f0aaee4a0a839468c618853ec441d65e5bce91860a5b202454e931b235d7"
EXECUTION_MANIFEST_SHA256 = "95f82d442eefe425dde6574222b037ba1deba2d3ae36534aeec41e31e450874f"
SOURCE_TREE_SHA256 = "f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130"
ACTOR_SHA256 = "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220"
OFFICIAL_METRIC = "global_wp_clip2_v1"
CONTROL_MODULES = (
    "policy",
    "journal",
    "process_start",
    "profiles",
    "runtime",
    "runtime_gates",
    "frozen_panel",
    "control_manifest",
    "fit_stage",
    "postprocess_common",
    "postprocess_fit",
    "postprocess_export",
    "postprocess_grade",
)
ERROR_EVENTS = {
    "fit_failed",
    "fit_deadline",
    "fit_validation_failed",
    "fit_admission_failed",
    "delivery_failed",
    "export_failed",
    "export_not_admitted",
    "export_admission_stopped",
    "grade_failed",
    "execution_failed",
    "controller_exception",
    "old_main_guard_failed",
}


def _sha256(path: Path) -> str:
    """Hash one regular, non-symlink file without loading it all into memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = importlib.import_module("hashlib").sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one regular JSON object; receipt payload fields are schema-flexible."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"receipt is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_modules(
    control_dir: Path, packet_control_dir: Path, packet_code_dir: Path
) -> dict[str, Any]:
    """Load successor guards and frozen-v4 validators without module-cache collisions."""
    module_dirs = (control_dir, packet_control_dir, packet_code_dir)
    module_names = {path.stem for directory in module_dirs for path in directory.glob("*.py")}
    module_names.update(CONTROL_MODULES)
    saved_modules = {name: sys.modules.pop(name, None) for name in module_names}
    original_path = list(sys.path)
    sys.path[:0] = [str(path) for path in module_dirs]
    try:
        return {name: importlib.import_module(name) for name in CONTROL_MODULES}
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        sys.path[:] = original_path


def _verify_deployed_code(root: Path) -> tuple[Path, Path]:
    """Hash both source bundles before importing any module from either bundle."""
    packet = root / "runtime/b300-r01-v1"
    manifest_path = packet / "EXECUTION_MANIFEST.json"
    if _sha256(manifest_path) != EXECUTION_MANIFEST_SHA256:
        raise ValueError("frozen R01-v4 execution manifest differs from its reviewed pin")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("frozen R01-v4 execution manifest has no file map")
    for relative, digest in files.items():
        path = packet / relative
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("frozen R01-v4 manifest entries must be string paths and hashes")
        if _sha256(path) != digest:
            raise ValueError(f"frozen R01-v4 code differs from its manifest: {relative}")

    control = root / "runtime/r01-density4-v1"
    control_manifest = _read_json(control / "CONTROL_MANIFEST.json")
    control_files = control_manifest.get("files")
    if not isinstance(control_files, dict) or not control_files:
        raise ValueError("successor control manifest has no file map")
    actual = {}
    for relative in sorted(control_files):
        if not isinstance(relative, str):
            raise ValueError("successor control manifest contains a non-string path")
        actual[relative] = _sha256(control / relative)
    digest = (
        importlib.import_module("hashlib")
        .sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        .hexdigest()
    )
    if (
        actual != control_files
        or digest != CONTROL_BUNDLE_SHA256
        or control_manifest.get("sha256") != digest
    ):
        raise ValueError("successor control bundle differs from its reviewed pin")
    return packet, control


def _authority(root: Path, modules: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    """Verify literal takeover authority and both frozen R01 packet identities."""
    authority_path = root / "receipts/r01-density4-admission-v1/AUTHORITY.json"
    if _sha256(authority_path) != AUTHORITY_SHA256:
        raise ValueError("R01 takeover authority differs from its reviewed receipt")
    authority = _read_json(authority_path)
    packet = root / "runtime/b300-r01-v1"
    control = root / "runtime/r01-density4-v1"
    expected_paths = {
        "root": str(root),
        "packet_root": str(packet),
        "old_state": str(root / "state/r01-execution-v4"),
        "successor_state_path": str(root / "state/r01-density4-v1"),
        "control_manifest_path": str(control / "CONTROL_MANIFEST.json"),
        "control_bundle_sha256": CONTROL_BUNDLE_SHA256,
    }
    if any(authority.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("R01 authority paths or control bundle do not match the frozen host")
    modules["control_manifest"].verify_control_manifest(
        control, control / "CONTROL_MANIFEST.json", CONTROL_BUNDLE_SHA256
    )
    modules["frozen_panel"].verify_frozen_packet(packet)
    if modules["frozen_panel"].sha256_file(packet / "PANEL.json") != PANEL_SHA256:
        raise ValueError("R01 CP30 panel differs from the frozen panel pin")
    if modules["frozen_panel"].sha256_file(packet / "DECISION_CONTRACT.json") != DECISION_SHA256:
        raise ValueError("R01 decision contract differs from the frozen pin")
    return authority, packet, control


def _score_value(score: object, label: str) -> float:
    """Require the official weighted-Pearson output to be finite and numeric."""
    if not isinstance(score, dict):
        raise ValueError(f"{label} score is not an object")
    value = score.get("wp")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} official WP score is missing or nonfinite")
    return float(value)


def _flag(argv: list[str], name: str) -> str:
    """Read one unique CLI flag value from a previously validated command."""
    positions = [index for index, item in enumerate(argv[:-1]) if item == name]
    if len(positions) != 1:
        raise ValueError(f"grade command must contain exactly one {name} flag")
    return argv[positions[0] + 1]


def _grade_links(
    root: Path, view: Path, pair: dict[str, Any], exports: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Return the exact immutable paths the official paired grader was given."""
    links = {"datasets": str(root / "data/datasets")}
    for name in ("chains.npz", "chain_folds.npz", "valid_targets.npz", "train_targets.npz"):
        links[f"cache/{name}"] = str(root / "data/cache" / name)
    for arm in ("wide", "deep"):
        job = pair[arm]["job"]
        run_name = str(job["id"])
        artifact = root / "runs" / run_name / "artifacts/runs" / run_name
        links[f"artifacts/runs/{run_name}"] = str(artifact)
        output = exports[arm]["output_root"]
        for suffix in ("npy", "json"):
            filename = f"pred_{run_name}_valid.{suffix}"
            links[f"cache/{filename}"] = str(output / "cache" / filename)
    for relative, target in links.items():
        link = view / relative
        if not link.is_symlink() or link.resolve(strict=True) != Path(target).resolve(strict=True):
            raise ValueError(f"grade view link does not resolve to the frozen input: {relative}")
    return links


def _evidence_paths(
    root: Path,
    old_state: Path,
    new_state: Path,
    job: dict[str, Any],
    grade_root: Path,
) -> tuple[list[str], list[str]]:
    """List existing evidence and weight files for one endpoint without copying them."""
    run_name = str(job["id"])
    run_dir = root / "runs" / run_name
    export = root / "exports" / run_name
    output = export / "export-r1"
    candidate_logs = (
        old_state / "logs" / f"{run_name}.log",
        new_state / "logs" / f"{run_name}.log",
    )
    evidence = [
        root / "splits" / f"fold{job['fold']}.json",
        run_dir / "producer_receipt.json",
        run_dir / "command.json",
        run_dir / "adapter_preflight.json",
        run_dir / "train_cell_resource_receipt.json",
        old_state / "receipts" / f"fit-{run_name}.json",
        new_state / "receipts" / f"fit-{run_name}.json",
        run_dir / "artifacts/runs" / run_name / "history.json",
        run_dir / "artifacts/runs" / run_name / "provenance.json",
        export / "postprocess_receipt.json",
        export / "postprocess_command.json",
        export / "postprocess.stdout.txt",
        export / "postprocess.stderr.txt",
        output / "export_receipt.json",
        output / "export_command.json",
        output / "export_cell_resource_receipt.json",
        output / "cache" / f"pred_{run_name}_valid.json",
        grade_root / "grade_receipt.json",
        grade_root / "grade_command.json",
        grade_root / "grade.json",
        grade_root / "grade.stdout.txt",
        grade_root / "grade.stderr.txt",
    ]
    log = next((path for path in candidate_logs if path.is_file() and not path.is_symlink()), None)
    if log is not None:
        evidence.append(log)
    root_resolved = root.resolve(strict=True)

    def safe_paths(items: list[Path]) -> list[str]:
        found = set()
        for path in items:
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if resolved.is_relative_to(root_resolved):
                found.add(str(resolved))
        return sorted(found)

    evidence_paths = safe_paths(evidence)
    artifact_dir = run_dir / "artifacts/runs" / run_name
    weights = [artifact_dir / name for name in ("model_ep5.pt", "model.pt", "training_state.pt")]
    weight_paths = safe_paths(weights)
    return evidence_paths, weight_paths


def _validate_pair(
    root: Path,
    pair_jobs: tuple[dict[str, Any], dict[str, Any]],
    paths: dict[str, Path],
    modules: dict[str, Any],
    old_state: Path,
    new_state: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Validate one exact wide/deep grade receipt through fit, export, and source."""
    wide_job = next(job for job in pair_jobs if job["arm"] == "wide")
    deep_job = next(job for job in pair_jobs if job["arm"] == "deep")
    fold, seed = int(deep_job["fold"]), int(deep_job["seed"])
    grade_root = root / "grades" / f"seed{seed}" / f"fold{fold}"
    receipt_path = grade_root / "grade_receipt.json"
    candidate, reference = str(deep_job["id"]), str(wide_job["id"])
    receipt = modules["postprocess_grade"]._existing_grade(grade_root, fold, candidate, reference)
    if receipt.get("status") != "graded":
        raise ValueError(f"pair seed{seed}/fold{fold} has no valid terminal grade receipt")
    grade = _read_json(grade_root / "grade.json")
    command_path = grade_root / "grade_command.json"
    command = _read_json(command_path)
    fit_bindings: dict[str, dict[str, Any]] = {}
    exports: dict[str, dict[str, Any]] = {}
    for arm, job in (("wide", wide_job), ("deep", deep_job)):
        modules["fit_stage"]._validate_fit_receipt(job)
        binding = modules["postprocess_fit"]._fit_binding(root, job)
        fit_bindings[arm] = binding
        if binding["split_sha256"] != fit_bindings["wide"]["split_sha256"]:
            raise ValueError("paired fit receipts do not share one frozen split manifest")
        output = root / "exports" / str(job["id"]) / "export-r1"
        attempt = root / "exports" / str(job["id"])
        export = modules["postprocess_export"]._validate_export(
            paths, binding, output, attempt, require_controller_receipt=True
        )
        export["output_root"] = output
        exports[arm] = export
    if fit_bindings["wide"]["groups"] != fit_bindings["deep"]["groups"]:
        raise ValueError("paired fit receipts disagree on evaluation membership")
    command_argv = command.get("argv")
    if not isinstance(command_argv, list) or any(
        not isinstance(item, str) for item in command_argv
    ):
        raise ValueError("grade command argv is malformed")
    view = root / "grades" / f"seed{seed}" / f"fold{fold}" / "view"
    scratch = root / "scratch" / "grading" / f"seed{seed}" / f"fold{fold}"
    if (
        Path(_flag(command_argv, "--root")) != view
        or Path(_flag(command_argv, "--output")) != grade_root / "grade.json"
        or Path(_flag(command_argv, "--scratch-dir")) != scratch
    ):
        raise ValueError("grade command targets a different view, output, or scratch directory")
    expected_argv = modules["postprocess_grade"]._grade_argv(
        paths,
        view,
        candidate,
        reference,
        fit_bindings["deep"]["split_path"],
        grade_root / "grade.json",
        scratch,
    )
    expected_receipts = {arm: fit_bindings[arm]["fit_receipt_sha256"] for arm in ("wide", "deep")}
    expected_exports = {arm: exports[arm]["export_receipt_sha256"] for arm in ("wide", "deep")}
    expected_shapes = {
        arm: [len(fit_bindings[arm]["groups"]), 20_000, 2] for arm in ("wide", "deep")
    }
    if (
        command.get("schema") != "tabdata-b300-grade-command.v1"
        or command_argv != expected_argv
        or command.get("grader_sha256") != modules["postprocess_grade"].GRADER_SHA256
        or command.get("actor_bundle_sha256") != modules["postprocess_grade"].ACTOR_SHA256
        or command.get("source_tree_identity_sha256")
        != modules["postprocess_grade"].SOURCE_TREE_SHA256
        or command.get("fit_receipts") != expected_receipts
        or command.get("export_receipts") != expected_exports
        or command.get("prediction_shapes") != expected_shapes
        or command.get("prediction_dtypes") != {"wide": "float32", "deep": "float32"}
        or command.get("fold") != fold
        or command.get("candidate") != candidate
        or command.get("reference") != reference
        or command.get("epoch") != 5
        or command.get("split_manifest_sha256") != fit_bindings["deep"]["split_sha256"]
        or command.get("evaluation_groups") != fit_bindings["deep"]["groups"]
        or command.get("bootstrap") != 2000
    ):
        raise ValueError("grade command does not bind both canonical endpoint receipts")
    if command.get("grade_view_links") != _grade_links(root, view, fit_bindings, exports):
        raise ValueError("grade command input links differ from the immutable fit/export inputs")
    receipt_sha = _sha256(receipt_path)
    grade_sha = _sha256(grade_root / "grade.json")
    if (
        receipt.get("schema") != "tabdata-b300-grade-receipt.v1"
        or receipt.get("status") != "graded"
        or receipt.get("grade_output", {}).get("sha256") != grade_sha
        or receipt.get("command_sha256") != _sha256(command_path)
        or receipt.get("execution", {}).get("status") != "exited"
        or receipt.get("execution", {}).get("exit_code") != 0
        or receipt.get("grader_sha256") != modules["postprocess_grade"].GRADER_SHA256
        or receipt.get("actor_bundle_sha256") != modules["postprocess_grade"].ACTOR_SHA256
        or receipt.get("source_tree_identity_sha256")
        != modules["postprocess_grade"].SOURCE_TREE_SHA256
        or receipt.get("official_metric_id") != OFFICIAL_METRIC
        or grade.get("schema") != "priority25.endpoint_grade.v1"
        or grade.get("metric_id") != OFFICIAL_METRIC
        or grade.get("fold") != fold
        or grade.get("split_manifest_sha256") != fit_bindings["deep"]["split_sha256"]
        or grade.get("evaluation_row_groups") != fit_bindings["deep"]["groups"]
        or grade.get("endpoint", {}).get("run") != candidate
        or grade.get("endpoint", {}).get("epoch") != 5
        or grade.get("reference", {}).get("run") != reference
        or grade.get("reference", {}).get("epoch") != 5
        or grade.get("paired_component_bootstrap", {}).get("resamples") != 2000
    ):
        raise ValueError(
            "official grade JSON or receipt does not match the registered R01 endpoint"
        )
    endpoints = {"deep": grade["endpoint"], "wide": grade["reference"]}
    rows = []
    source_builder = modules["postprocess_common"]._read_json(paths["builder"])
    source = {
        "commit": source_builder["expected_base_source_revision"],
        "canonical_source_identity_sha256": source_builder["expected_source_identity_sha256"],
        "canonical_source_manifest_sha256": source_builder["source_manifest_sha256"],
        "source_tree_identity_sha256": source_builder["expected_source_tree_identity_sha256"],
        "source_tree_archive_sha256": source_builder["source_tree_archive_sha256"],
        "source_tree_manifest_sha256": source_builder["source_tree_manifest_sha256"],
        "actor_bundle_sha256": ACTOR_SHA256,
        "grader_sha256": modules["postprocess_grade"].GRADER_SHA256,
    }
    pair_id = f"r01-fold{fold}-seed{seed}"
    for arm, job, other in (("wide", wide_job, deep_job), ("deep", deep_job, wide_job)):
        endpoint = endpoints[arm]
        score = endpoint.get("score")
        wp = _score_value(score, f"{arm} fold{fold} seed{seed}")
        binding, export = fit_bindings[arm], exports[arm]
        if (
            endpoint.get("checkpoint_sha256") != binding["checkpoint_sha256"]
            or endpoint.get("prediction_sha256") != export["prediction"]["sha256"]
        ):
            raise ValueError(f"{arm} grade endpoint hashes differ from its fit/export receipts")
        ancestry = binding["fit_receipt"].get("input_ancestry", {})
        data = {
            "split_manifest_sha256": binding["split_sha256"],
            "evaluation_groups": binding["groups"],
            "source_identities": ancestry.get("source_identities"),
            "cache_inputs": ancestry.get("cache_inputs"),
        }
        evidence_paths, weights_paths = _evidence_paths(root, old_state, new_state, job, grade_root)
        rows.append(
            {
                "id": str(job["id"]),
                "arm": arm,
                "fold": fold,
                "seed": seed,
                "wp": wp,
                "metric_id": OFFICIAL_METRIC,
                "score": score,
                "pair_id": pair_id,
                "paired_with": str(other["id"]),
                "checkpoint": binding["checkpoint_sha256"],
                "prediction": export["prediction"]["sha256"],
                "source": source,
                "data": data,
                "receipt_sha256": receipt_sha,
                "fit_receipt_sha256": binding["fit_receipt_sha256"],
                "export_receipt_sha256": export["export_receipt_sha256"],
                "grade_output_sha256": grade_sha,
                "grade_command_sha256": _sha256(command_path),
                "evidence_paths": evidence_paths,
                "weights_paths": weights_paths,
            }
        )
    return rows, receipt_sha


def _state_counts(
    jobs: tuple[dict[str, Any], ...],
    old_events: dict[str, Any],
    new_events: dict[str, Any],
    old_progress: dict[str, Any],
    completed_ids: set[str],
) -> tuple[int, int, list[str]]:
    """Count each panel cell once across original, adopted, and successor events."""
    known = {str(job["id"]) for job in jobs}
    touched = set(old_events["referenced"]) | set(new_events["referenced"])
    touched |= set(old_progress.get("claimed", set()))
    errors: list[str] = []
    failed: set[str] = set()
    all_events = old_events["events"] + new_events["events"]
    for event in all_events:
        kind = event.get("event")
        job_id = event.get("job_id")
        if kind in {"heartbeat"}:
            for field in ("active_fit_ids", "active_delivery_ids"):
                ids = event.get(field, [])
                if isinstance(ids, list):
                    touched.update(item for item in ids if isinstance(item, str) and item in known)
        failed_event = kind in ERROR_EVENTS or (
            kind == "grade_finished" and event.get("status") != "graded"
        )
        if failed_event:
            phase = event.get("phase", kind)
            message = str(event.get("error", event.get("reason", "")))[:400]
            label = f"{job_id}: {phase}: {message}" if job_id else f"{phase}: {message}"
            errors.append(label.rstrip(": "))
            if isinstance(job_id, str) and job_id in known:
                failed.add(job_id)
            grade_id = event.get("grade_id")
            if isinstance(grade_id, str) and grade_id.startswith("seed") and "_fold" in grade_id:
                try:
                    seed_text, fold_text = grade_id.removeprefix("seed").split("_fold", 1)
                    pair_key = (int(seed_text), int(fold_text))
                    failed.update(
                        str(job["id"]) for job in jobs if (job["seed"], job["fold"]) == pair_key
                    )
                except ValueError:
                    errors.append(f"unrecognized R01 grade identifier: {grade_id}")
            if kind == "grade_finished":
                try:
                    pair_key = (int(event["seed"]), int(event["fold"]))
                    failed.update(
                        str(job["id"]) for job in jobs if (job["seed"], job["fold"]) == pair_key
                    )
                except (KeyError, TypeError, ValueError):
                    errors.append("grade_finished failure lacks a valid fold/seed identity")
        if kind == "fit_spawn_intent" and isinstance(job_id, str):
            later_start = any(
                next_event.get("job_id") == job_id
                and next_event.get("event") == "fit_started"
                and next_event.get("time_unix", 0) >= event.get("time_unix", 0)
                for next_event in all_events
            )
            failed_start = any(
                next_event.get("job_id") == job_id
                and next_event.get("event") == "fit_admission_failed"
                for next_event in all_events
            )
            if not later_start and not failed_start and job_id not in completed_ids:
                errors.append(f"{job_id}: incomplete fit spawn transition")
                failed.add(job_id)
    touched.update(completed_ids)
    failed -= completed_ids
    active = len(touched - completed_ids - failed)
    pending = len(known - touched - completed_ids - failed)
    return active, pending, sorted(set(item for item in errors if item))[:200]


def collect(root: Path) -> dict[str, Any]:
    """Return only receipt-verified R01 scores and unique live/pending/error counts."""
    try:
        return _collect(root)
    except Exception as error:
        # Keep a broken R01 receipt tree from suppressing an independent PR15 report.
        return {
            "schema": "tabdata-github-r01-report.v1",
            "experiment_id": "E-B300-R01",
            "expected_cells": 60,
            "expected_per_arm": 30,
            "completed": [],
            "active": 0,
            "pending": 60,
            "errors": [f"R01 frozen receipt-tree validation failed ({type(error).__name__})"],
            "terminal": False,
            "controller_status": None,
            "source": {
                "canonical_source_tree_identity_sha256": SOURCE_TREE_SHA256,
                "actor_bundle_sha256": ACTOR_SHA256,
                "expected_authority_sha256": AUTHORITY_SHA256,
                "expected_control_bundle_sha256": CONTROL_BUNDLE_SHA256,
                "expected_panel_sha256": PANEL_SHA256,
            },
            "metric_id": OFFICIAL_METRIC,
            "panel_sha256": PANEL_SHA256,
            "decision_contract_sha256": DECISION_SHA256,
            "journal_sha256": {},
            "grade_pair_receipts": {},
            "contract_note": (
                "R01 results are unavailable because local frozen receipt validation failed; "
                "no endpoint scores are inferred from this status."
            ),
        }


def _collect(root: Path) -> dict[str, Any]:
    """Validate the R01 receipt tree and return only source-bound official scores."""
    root = root.resolve(strict=True)
    packet, control_dir = _verify_deployed_code(root)
    modules = _load_modules(control_dir, packet / "control", packet / "code")
    authority, packet, _ = _authority(root, modules)
    panel = modules["frozen_panel"].read_object(packet / "PANEL.json")
    jobs = modules["frozen_panel"].validate_panel(panel, root)
    if (
        len(jobs) != 60
        or panel.get("experiment_id") != "E-B300-R01"
        or panel.get("cells_per_arm") != 30
    ):
        raise ValueError("R01 panel does not describe the 60-cell CP30 comparison")
    if {arm: sum(job["arm"] == arm for job in jobs) for arm in ("wide", "deep")} != {
        "wide": 30,
        "deep": 30,
    }:
        raise ValueError("R01 panel does not contain 30 registered cells per arm")
    identity = panel.get("actor_bundle_identity", {})
    if (
        identity.get("sha256") != ACTOR_SHA256
        or identity.get("source_tree_identity_sha256") != SOURCE_TREE_SHA256
    ):
        raise ValueError("R01 panel actor or source tree differs from the pinned recipe")
    old_state = Path(authority["old_state"])
    new_state = Path(authority["successor_state_path"])
    old_journal = old_state / "events.jsonl"
    new_journal = new_state / "events.jsonl"
    for journal in (old_journal, new_journal):
        if journal.is_symlink() or not journal.is_file():
            raise FileNotFoundError(f"R01 event journal is missing or unsafe: {journal}")
    old_events = modules["frozen_panel"].parse_journal(old_journal, jobs)
    new_events = modules["frozen_panel"].parse_journal(new_journal, jobs)
    duplicate_starts = set(old_events["starts"]) & set(new_events["starts"])
    if duplicate_starts:
        raise ValueError(
            f"R01 cell has fit_started events in both controllers: {sorted(duplicate_starts)}"
        )
    old_starts = old_events["starts"]
    adopted_ids: set[str] = set()
    for event in new_events["events"]:
        if event.get("event") != "fit_adopted":
            continue
        job_id = event.get("job_id")
        prior = old_starts.get(job_id)
        required = ("pid", "gpu_uuid", "arm")
        supplied_identity = ("fold", "seed")
        if (
            job_id in adopted_ids
            or prior is None
            or any(event.get(key) != prior.get(key) for key in required)
            or any(key in event and event[key] != prior.get(key) for key in supplied_identity)
            or (
                "pid_start_ticks" in prior
                and event.get("pid_start_ticks") != prior["pid_start_ticks"]
            )
        ):
            raise ValueError(f"R01 adoption event is not bound to the original fit start: {job_id}")
        adopted_ids.add(job_id)
    old_progress = modules["frozen_panel"].progress_state(old_state / "PROGRESS.json", jobs)
    paths = modules["postprocess_common"]._runtime_paths(root, list(jobs))
    pairs: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for job in jobs:
        pairs.setdefault((int(job["seed"]), int(job["fold"])), []).append(job)
    completed: list[dict[str, Any]] = []
    pair_receipts: dict[str, str] = {}
    errors: list[str] = []
    for (seed, fold), pair in sorted(pairs.items()):
        grade_root = root / "grades" / f"seed{seed}" / f"fold{fold}"
        grade_receipt = grade_root / "grade_receipt.json"
        if not grade_receipt.exists() and not grade_receipt.is_symlink():
            continue
        try:
            rows, receipt_sha = _validate_pair(
                root,
                tuple(pair),
                paths,
                modules,
                old_state,
                new_state,
            )
            completed.extend(rows)
            pair_receipts[f"fold{fold}_seed{seed}"] = receipt_sha
        except Exception as error:
            errors.append(
                f"fold{fold}/seed{seed} grade chain rejected: {type(error).__name__}: {error}"
            )
    completed_ids = {str(row["id"]) for row in completed}
    active, pending, event_errors = _state_counts(
        jobs, old_events, new_events, old_progress, completed_ids
    )
    errors.extend(event_errors)
    summary_path = new_state / "SUMMARY.json"
    summary: dict[str, Any] | None = None
    terminal = False
    if summary_path.exists() or summary_path.is_symlink():
        summary = _read_json(summary_path)
        finish_events = [
            event for event in new_events["events"] if event.get("event") == "controller_finished"
        ]
        if (
            summary.get("schema") != "tabdata-r01-density4-successor-summary.v1"
            or summary.get("panel_sha256") != PANEL_SHA256
            or summary.get("frozen_execution_manifest_sha256")
            != modules["frozen_panel"].R01_MANIFEST_SHA256
            or not finish_events
            or finish_events[-1].get("summary_sha256") != _sha256(summary_path)
            or finish_events[-1].get("status") != summary.get("status")
            or summary.get("grade_pair_count") != len(pair_receipts)
            or summary.get("pending_fit_count") != pending
        ):
            errors.append(
                "R01 terminal summary does not reconcile to the frozen panel and grade receipts"
            )
        else:
            terminal = summary.get("status") in {"complete", "partial"} and active == 0
            if summary.get("status") == "complete" and (
                len(completed) != 60 or pending != 0 or summary.get("grade_pair_count") != 30
            ):
                terminal = False
                errors.append("R01 complete summary does not contain all 30 scored paired cells")
    source_builder = modules["postprocess_common"]._read_json(paths["builder"])
    source = {
        "commit": source_builder["expected_base_source_revision"],
        "canonical_source_identity_sha256": source_builder["expected_source_identity_sha256"],
        "canonical_source_manifest_sha256": source_builder["source_manifest_sha256"],
        "source_tree_identity_sha256": source_builder["expected_source_tree_identity_sha256"],
        "source_tree_archive_sha256": source_builder["source_tree_archive_sha256"],
        "source_tree_manifest_sha256": source_builder["source_tree_manifest_sha256"],
        "actor_bundle_sha256": ACTOR_SHA256,
        "grader_sha256": modules["postprocess_grade"].GRADER_SHA256,
    }
    return {
        "schema": "tabdata-github-r01-report.v1",
        "experiment_id": "E-B300-R01",
        "expected_cells": 60,
        "expected_per_arm": 30,
        "completed": sorted(completed, key=lambda row: (row["fold"], row["seed"], row["arm"])),
        "active": active,
        "pending": pending,
        "errors": sorted(set(errors))[:200],
        "terminal": terminal,
        "controller_status": summary.get("status") if summary else None,
        "source": source,
        "metric_id": OFFICIAL_METRIC,
        "panel_sha256": PANEL_SHA256,
        "decision_contract_sha256": DECISION_SHA256,
        "journal_sha256": {"original": old_events["sha256"], "successor": new_events["sha256"]},
        "grade_pair_receipts": pair_receipts,
        "contract_note": (
            "H512x1 wide vs H384x2 deep; six-epoch strict-FP32 fixed-epoch-5 labels-only recipe; "
            "near-parameter comparison, not matched FLOPs; all four folds are exposed."
        ),
    }
