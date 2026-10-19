"""Receipt-only collection and rendering for native-feature successor pairs."""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

FAMILIES = {
    "age": {
        "source_family": "field_age",
        "control_arm": "group",
        "feature_arm": "field",
        "experiment_id": "E-B300-PR4-AGE-SUCCESSOR-V1",
        "expected_cells": 30,
    },
    "innovation": {
        "source_family": "causal_innovation",
        "control_arm": "moments",
        "feature_arm": "innovations",
        "experiment_id": "E-B300-PR14-INNOVATION-SUCCESSOR-V1",
        "expected_cells": 30,
    },
}
FAMILY_NAMES = {"field_age": "age", "causal_innovation": "innovation"}
ARM_NAMES = {
    "group_age": "group",
    "field_age": "field",
    "matched_moments": "moments",
    "moments_plus_innovation": "innovations",
}
_SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)")


def normalized_job(raw: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the controller's exact internal job representation."""
    return dict(
        raw,
        family=FAMILY_NAMES[raw["family"]],
        arm=ARM_NAMES[raw["arm"]],
        source_family=raw["family"],
        source_arm=raw["arm"],
        cpus=raw.get("cpu_ids"),
    )


def expected_pairs(panel: dict[str, Any], family: str) -> dict[tuple[int, int], dict[str, Any]]:
    """Bind each planned pair to exactly one control and one feature job."""
    rule = FAMILIES[family]
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in panel["jobs"]:
        if raw.get("family") != rule["source_family"]:
            continue
        cell = (raw["fold"], raw["seed"])
        pair = pairs.setdefault(cell, {})
        arm = ARM_NAMES[raw["arm"]]
        if arm in pair:
            raise ValueError(f"duplicate {family} arm in fold/seed cell {cell}")
        pair[arm] = normalized_job(raw)
    if len(pairs) != rule["expected_cells"] or any(
        set(pair) != {rule["control_arm"], rule["feature_arm"]} for pair in pairs.values()
    ):
        raise ValueError(f"{family} panel does not contain 30 complete matched cells")
    return pairs


def equal_fold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired differences with equal weight for each observed fold."""
    by_fold: dict[int, list[float]] = defaultdict(list)
    control: dict[int, list[float]] = defaultdict(list)
    feature: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        fold = row["fold"]
        by_fold[fold].append(float(row["delta_wp"]))
        control[fold].append(float(row["control_wp"]))
        feature[fold].append(float(row["feature_wp"]))

    def mean_folds(values: dict[int, list[float]]) -> float | None:
        if not values:
            return None
        return sum(sum(fold_values) / len(fold_values) for fold_values in values.values()) / len(
            values
        )

    return {
        "scored_pairs": len(rows),
        "delta_wp_paired_cell_mean": sum(float(row["delta_wp"]) for row in rows) / len(rows)
        if rows
        else None,
        "folds_observed": sorted(by_fold),
        "control_wp_equal_fold_mean": mean_folds(control),
        "feature_wp_equal_fold_mean": mean_folds(feature),
        "delta_wp_equal_fold_mean": mean_folds(by_fold),
    }


def report_body(snapshot: dict[str, Any], evidence_url: str, *, rows: list[dict] | None) -> str:
    """Render a marked status or one bounded batch of verified paired results."""
    from render_report import marker

    experiment = snapshot["experiment_id"]
    kind = "status" if rows is None else "results"
    digest = "" if rows is None else canonical_sha(rows)
    completed = len(snapshot["completed"])
    summary = equal_fold_summary(snapshot["completed"])
    label = "complete" if snapshot["terminal"] and completed == 30 else "partial / in progress"
    lines = [
        marker(experiment, kind, digest),
        f"### {experiment}: {kind} update",
        f"**Status:** {label}. Scored matched pairs {completed}/30; active paired cells {snapshot['active']}; "
        f"pending paired cells {snapshot['pending']}; terminal {str(snapshot['terminal']).lower()}.",
        snapshot["contract_note"],
        "The compared endpoints are the fixed epoch-5 strict-FP32 successor arms on the same "
        "fold and seed. These are exposed development folds; partial means are descriptive and "
        "do not establish significance, pristine holdout performance, or serving readiness.",
        f"Primary paired-cell mean delta WP: {_fmt(summary['delta_wp_paired_cell_mean'], signed=True)}. Equal-fold means below are secondary descriptive summaries.",
        "",
        "| Matched pairs | Control WP (equal-fold mean) | Feature WP (equal-fold mean) | "
        "Paired delta (equal-fold mean) |",
        "|---:|---:|---:|---:|",
        f"| {summary['scored_pairs']} | {_fmt(summary['control_wp_equal_fold_mean'])} | "
        f"{_fmt(summary['feature_wp_equal_fold_mean'])} | "
        f"{_fmt(summary['delta_wp_equal_fold_mean'], signed=True)} |",
        "",
        "Source identity: `" + str(snapshot["source"].get("commit", "unknown")) + "`.",
        f"Evidence and frozen contract: {evidence_url}",
    ]
    if rows is not None:
        lines.extend(
            marker(
                experiment,
                "pair",
                canonical_sha({"pair_id": row["pair_id"], "receipt_sha256": row["receipt_sha256"]}),
            )
            for row in rows
        )
        lines.extend(
            [
                f"**New scored pairs in this comment:** {len(rows)}.",
                "",
                "| Fold | Seed | Control WP | Feature WP | Delta | Control checkpoint | "
                "Feature checkpoint | Grade SHA-256 | Control archive manifest | "
                "Feature archive manifest |",
                "|---:|---:|---:|---:|---:|---|---|---|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['fold']} | {row['seed']} | {row['control_wp']:.8f} | "
                f"{row['feature_wp']:.8f} | {row['delta_wp']:+.8f} | "
                f"`{row['control']['checkpoint_sha256']}` | "
                f"`{row['feature']['checkpoint_sha256']}` | `{row['grade_sha256']}` | "
                f"`{row['control'].get('archive_manifest_sha256', '—')}` | "
                f"`{row['feature'].get('archive_manifest_sha256', '—')}` |"
            )
        lines.append(
            "Each pair binds both FIT/EXPORTED receipts, prediction sidecars, the official "
            "paired-grade JSON and the unchanged grader source. Full evidence and weights are "
            "retained separately on approved private storage."
        )
    if snapshot.get("errors"):
        lines.extend(
            [
                "**Collection errors retained:**",
                *[f"- {str(e)[:500]}" for e in snapshot["errors"][:12]],
            ]
        )
    lines.append("Automated reporting only; this process does not fit, export, or grade models.")
    return "\n".join(lines) + "\n"


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.8f}" if signed else f"{value:.8f}"


def _read_events(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise ValueError("Controller event log cannot be a symlink")
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Controller event {line_number} is not an object")
        result.append(value)
    return result


def _file_digest(common: Any, path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Evidence file is missing or is a symlink: " + str(path))
    return common.sha(path)


def _claimed_job(
    state: Path, expected: dict[str, Any], common: Any, panel_module: Any
) -> dict[str, Any]:
    """Bind immutable panel fields to the claim and verify its scheduled CPU slot."""
    claim_path = state / "claims" / (expected["id"] + ".json")
    claim = common.read(claim_path)
    expected_keys = set(expected) | {"cpu_ids", "slot_id"}
    if set(claim) != expected_keys or any(
        claim.get(key) != value for key, value in expected.items() if key != "cpus"
    ):
        raise ValueError("controller claim differs from the immutable panel job")
    slot_id = claim["slot_id"]
    if type(slot_id) is not int:
        raise ValueError("controller claim slot must be an integer")
    expected_cpus = list(panel_module.cpu_pair(claim["gpu_index"], slot_id))
    cpu_ids, cpus = claim["cpu_ids"], claim["cpus"]
    if (
        not isinstance(cpu_ids, list)
        or not isinstance(cpus, list)
        or any(type(cpu) is not int for cpu in cpu_ids + cpus)
        or cpu_ids != expected_cpus
        or cpus != expected_cpus
    ):
        raise ValueError("controller claim CPU affinity differs from its frozen GPU slot")
    return claim


def _endpoint_paths(
    root: Path, state: Path, work: Path, grade_dir: Path
) -> tuple[list[str], list[str]]:
    """Return only explicit result/receipt paths, excluding prediction and supplied data arrays."""
    run = work / "artifacts/runs" / work.name
    evidence = [
        state / "PANEL.json",
        state / "SOURCE.json",
        state / "events.jsonl",
        state / "claims" / (work.name + ".json"),
        state / "endpoints" / (work.name + ".json"),
        state / "logs" / (work.name + ".log"),
        work / "INPUTS.json",
        work / "worker_request.json",
        work / "fit_phase.json",
        work / "FIT.json",
        work / "EXPORTED.json",
        work / "export_phase.json",
        run / "history.json",
        run / "provenance.json",
        work / "cache" / f"pred_{work.name}_valid.json",
        state / "scored" / (grade_dir.name + ".json"),
        grade_dir / "RECEIPT.json",
        grade_dir / "grade.json",
        grade_dir / "command.json",
        grade_dir / "stdout.txt",
        grade_dir / "stderr.txt",
    ]
    evidence.extend(grade_dir / name for name in ("view", "scratch"))
    for phase_name in ("fit_phase.json", "export_phase.json"):
        phase_path = work / phase_name
        if phase_path.is_file() and not phase_path.is_symlink():
            phase = json.loads(phase_path.read_text(encoding="utf-8"))
            for key in ("log", "resource_receipt"):
                value = phase.get(key)
                if isinstance(value, str):
                    candidate = Path(value)
                    evidence.append(candidate if candidate.is_absolute() else work / candidate)
    safe = set()
    for path in evidence:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and resolved.is_relative_to(root) and resolved.suffix != ".npy":
            safe.add(str(resolved))
    return sorted(safe), [
        str(run / name)
        for name in ("model_ep5.pt", "model.pt", "training_state.pt")
        if (run / name).is_file() and not (run / name).is_symlink()
    ]


def _verify_endpoint(
    root: Path, state: Path, job: dict[str, Any], modules: tuple[Any, Any, Any, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common, validation, outputs, _panel_module = modules
    work = state / "work" / job["id"]
    fit_path, export_path = work / "FIT.json", work / "EXPORTED.json"
    fit_record, exported = common.read(fit_path), common.read(export_path)
    checked_fit = validation.validate_fit(root, work, job)
    if (
        fit_record.get("job") != job
        or fit_record.get("validation") != checked_fit
        or exported.get("job") != job
        or exported.get("status") != "fit_export_completed"
        or exported.get("fit") != checked_fit
        or exported.get("fit_receipt_sha256") != common.sha(fit_path)
    ):
        raise ValueError("endpoint fit/export receipt identity differs from the frozen job")
    export_phase = common.read(work / "export_phase.json")
    _verify_export_phase(root, work, job, export_phase, common)
    if exported.get("export") != export_phase:
        raise ValueError("endpoint export phase differs from the retained actor receipt")
    endpoint_path = state / "endpoints" / (job["id"] + ".json")
    endpoint = common.read(endpoint_path)
    if endpoint != exported:
        raise ValueError("controller endpoint copy differs from the worker export receipt")
    prediction = outputs.validate_export(root, work, job, checked_fit["checkpoint_sha256"])
    if exported.get("prediction") != prediction:
        raise ValueError("exported prediction receipt differs from the independent sidecar check")
    return checked_fit, prediction, exported


def _verify_export_phase(
    root: Path, work: Path, job: dict[str, Any], phase: dict, common: Any
) -> None:
    """Verify the successful B300 export actor, log and in-work resource envelope."""
    package = root / "runtime/pr15-host-v2"
    command = phase.get("command")
    suffix = [
        "-u",
        str(package / "gpu_entrypoint.py"),
        str(package / "export_cell.py"),
        "--request",
        str(work / "worker_request.json"),
    ]
    elapsed = phase.get("elapsed_seconds")
    if (
        type(phase.get("exit_code")) is not int
        or phase.get("exit_code") != 0
        or not isinstance(command, list)
        or len(command) != len(suffix) + 1
        or not isinstance(command[0], str)
        or not Path(command[0]).is_absolute()
        or command[1:] != suffix
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("export actor phase is not the bounded canonical B300 invocation")
    log = Path(str(phase.get("log", "")))
    if not log.is_absolute():
        log = work / log
    if (
        log.is_symlink()
        or not log.is_file()
        or not log.resolve(strict=True).is_relative_to(root)
        or common.sha(log) != phase.get("log_sha256")
    ):
        raise ValueError("export actor log is missing, outside the project, or changed")
    resource_path = Path(str(phase.get("resource_receipt", "")))
    if not resource_path.is_absolute():
        resource_path = work / resource_path
    if (
        resource_path.is_symlink()
        or not resource_path.is_file()
        or not resource_path.resolve(strict=True).is_relative_to(work)
        or common.sha(resource_path) != phase.get("resource_receipt_sha256")
    ):
        raise ValueError("export actor resource receipt is missing or changed")
    resource = common.read(resource_path)
    if (
        resource.get("schema") != "tabdata-b300-process-envelope.v1"
        or resource.get("status") != "actor_returned"
        or resource.get("script") != "export_cell.py"
        or resource.get("visible_devices") != job["gpu_uuid"]
        or resource.get("cpu_affinity") != job["cpus"]
        or resource.get("allocator_budget_bytes") != 6 * 1024**3
        or resource.get("torch_threads") != 2
        or "B300" not in str(resource.get("gpu", ""))
    ):
        raise ValueError("export actor resource envelope differs from the fixed B300 cell")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = resource.get(key)
        if type(value) is not int or not 0 <= value <= 6 * 1024**3:
            raise ValueError("export actor exceeded its allocator envelope: " + key)


def _verify_pair(
    root: Path,
    state: Path,
    family: str,
    fold: int,
    seed: int,
    jobs: dict[str, Any],
    modules: tuple[Any, Any, Any, Any],
) -> dict[str, Any]:
    common, _, outputs, panel_module = modules
    rule = FAMILIES[family]
    control = _claimed_job(state, jobs[rule["control_arm"]], common, panel_module)
    feature = _claimed_job(state, jobs[rule["feature_arm"]], common, panel_module)
    scored_path = state / "scored" / f"{family}_f{fold}_s{seed}.json"
    score_receipt = common.read(scored_path)
    grade_dir = state / "grades" / f"{family}_f{fold}_s{seed}"
    receipt_path, grade_path, command_path = (
        grade_dir / "RECEIPT.json",
        grade_dir / "grade.json",
        grade_dir / "command.json",
    )
    receipt = common.read(receipt_path)
    grade, command = common.read(grade_path), common.read(command_path)
    if (
        score_receipt != receipt
        or receipt.get("schema") != "tabdata-native-feature-paired-grade.v1"
    ):
        raise ValueError("score index and immutable pair grade receipt differ")
    if any(
        receipt.get(key) != value
        for key, value in {
            "status": "graded",
            "family": family,
            "fold": fold,
            "seed": seed,
            "candidate": feature["id"],
            "control": control["id"],
            "grade_path": str(grade_path),
            "grade_sha256": common.sha(grade_path),
            "grader_sha256": common.GRADER_SHA256,
            "command_sha256": common.sha(command_path),
            "stdout_sha256": common.sha(grade_dir / "stdout.txt"),
            "stderr_sha256": common.sha(grade_dir / "stderr.txt"),
        }.items()
    ):
        raise ValueError("pair-grade receipt hashes or scientific cell differ")
    if (
        command.get("grader_sha256") != common.GRADER_SHA256
        or command.get("candidate_export_sha256")
        != common.sha(state / "work" / feature["id"] / "EXPORTED.json")
        or command.get("control_export_sha256")
        != common.sha(state / "work" / control["id"] / "EXPORTED.json")
    ):
        raise ValueError("official grade command is not bound to the two current exports")
    argv = command.get("argv", [])
    expected_flags = {
        "--root": str(grade_dir / "view"),
        "--run": feature["id"],
        "--reference": control["id"] + "@5",
        "--split": str(root / "splits" / f"fold{fold}.json"),
        "--epoch": "5",
        "--output": str(grade_path),
        "--bootstrap": "2000",
        "--scratch-dir": str(grade_dir / "scratch"),
    }
    if not isinstance(argv, list) or any(
        argv.count(flag) != 1
        or argv.index(flag) + 1 >= len(argv)
        or argv[argv.index(flag) + 1] != value
        for flag, value in expected_flags.items()
    ):
        raise ValueError("official grade command differs from the fixed epoch-5 paired recipe")
    grader_path = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    if len(argv) < 2 or argv[1] != str(grader_path) or not Path(argv[0]).is_absolute():
        raise ValueError("official grade command does not invoke the pinned grader")

    fit_control, pred_control, export_control = _verify_endpoint(root, state, control, modules)
    fit_feature, pred_feature, export_feature = _verify_endpoint(root, state, feature, modules)
    endpoint_grade = grade.get("endpoint")
    reference_grade = grade.get("reference")
    bootstrap = grade.get("paired_component_bootstrap")
    if not all(isinstance(value, dict) for value in (endpoint_grade, reference_grade, bootstrap)):
        raise ValueError("official paired grade omitted endpoint, reference or bootstrap evidence")
    if (
        grade.get("metric_id") != "global_wp_clip2_v1"
        or grade.get("fold") != fold
        or grade.get("split_manifest_sha256") != pred_feature["split_sha256"]
        or endpoint_grade.get("run") != feature["id"]
        or endpoint_grade.get("epoch") != 5
        or reference_grade.get("run") != control["id"]
        or reference_grade.get("epoch") != 5
        or endpoint_grade.get("checkpoint_sha256") != fit_feature["checkpoint_sha256"]
        or reference_grade.get("checkpoint_sha256") != fit_control["checkpoint_sha256"]
        or endpoint_grade.get("prediction_sha256") != pred_feature["sha256"]
        or reference_grade.get("prediction_sha256") != pred_control["sha256"]
        or endpoint_grade.get("sidecar_sha256") != pred_feature["metadata_sha256"]
        or reference_grade.get("sidecar_sha256") != pred_control["metadata_sha256"]
        or bootstrap.get("resamples") != 2000
        or bootstrap.get("seed") != 271828
        or grade.get("evaluation_row_groups")
        != common.read(root / "splits" / f"fold{fold}.json")["evaluation"]["valid"]
    ):
        raise ValueError("official grade outputs differ from current epoch-5 endpoint exports")
    control_raw = reference_grade.get("score", {}).get("wp")
    feature_raw = endpoint_grade.get("score", {}).get("wp")
    if isinstance(control_raw, bool) or isinstance(feature_raw, bool):
        raise ValueError("official pair grade WP cannot be boolean")
    control_wp, feature_wp = float(control_raw), float(feature_raw)
    if not math.isfinite(control_wp) or not math.isfinite(feature_wp):
        raise ValueError("official pair grade contains a nonfinite WP")
    if receipt.get("control_wp") != control_wp or receipt.get("candidate_wp") != feature_wp:
        raise ValueError("pair score receipt differs from the official score JSON")
    endpoints = {}
    for role, job, fit, prediction, exported in (
        ("control", control, fit_control, pred_control, export_control),
        ("feature", feature, fit_feature, pred_feature, export_feature),
    ):
        work = state / "work" / job["id"]
        endpoint_path = state / "endpoints" / (job["id"] + ".json")
        evidence_paths, weights_paths = _endpoint_paths(root, state, work, grade_dir)
        endpoints[role] = {
            "id": job["id"],
            "arm": job["arm"],
            "wp": control_wp if role == "control" else feature_wp,
            "source_commit": exported["source"]["commit"],
            "source_sha256": exported["source"]["tree"]["deployed_tree_identity_sha256"],
            "checkpoint_sha256": fit["checkpoint_sha256"],
            "checkpoint_payload_sha256": fit["payload_sha256"],
            "prediction_sha256": prediction["sha256"],
            "prediction_metadata_sha256": prediction["metadata_sha256"],
            "fit_receipt_sha256": common.sha(work / "FIT.json"),
            "receipt_sha256": common.sha(endpoint_path),
            "evidence_paths": evidence_paths,
            "weights_paths": weights_paths,
        }
    pair_id = f"{family}_f{fold}_s{seed}"
    return {
        "id": pair_id,
        "pair_id": pair_id,
        "family": family,
        "fold": fold,
        "seed": seed,
        "control_wp": control_wp,
        "feature_wp": feature_wp,
        "delta_wp": feature_wp - control_wp,
        "metric_id": "global_wp_clip2_v1",
        "control": endpoints["control"],
        "feature": endpoints["feature"],
        "receipt_sha256": common.sha(scored_path),
        "grade_receipt_sha256": common.sha(receipt_path),
        "grade_sha256": common.sha(grade_path),
        "grade_command_sha256": common.sha(command_path),
    }


def collect(root: Path, reporter_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and independently verify every published pair without running a grade."""
    code_dir = Path(reporter_config["feature_code_dir"]).resolve(strict=True)
    sys.path.insert(0, str(code_dir))
    common = importlib.import_module("feature_common")
    feature_config = common.verify_config(
        Path(reporter_config["feature_config_path"]), reporter_config["feature_config_sha256"]
    )
    if Path(feature_config["root"]).resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("feature config root differs from reporting root")
    panel_module = importlib.import_module("panel")
    panel_module.validate_panel(feature_config["panel"])
    source = common.verify_source(root)
    state = root / "state/parallel-feature-execution-v1"
    if state.exists():
        state_panel, state_source = (
            common.read(state / "PANEL.json"),
            common.read(state / "SOURCE.json"),
        )
        if state_panel != feature_config["panel"] or state_source != source:
            raise ValueError("controller state panel/source differs from the frozen feature config")
        events = _read_events(state / "events.jsonl")
        latest = next(
            (event for event in reversed(events) if event.get("event") == "heartbeat"), {}
        )
        summary = common.read(state / "SUMMARY.json") if (state / "SUMMARY.json").is_file() else {}
    else:
        events, latest, summary = [], {}, {}
    validation = importlib.import_module("feature_validation")
    outputs = importlib.import_module("feature_outputs")
    modules = (common, validation, outputs, panel_module)
    snapshots = []
    for family, rule in FAMILIES.items():
        pairs = expected_pairs(feature_config["panel"], family)
        completed: list[dict[str, Any]] = []
        errors: list[str] = []
        failed_pairs: set[tuple[int, int]] = set()
        active_jobs = set(latest.get("active", []))
        active_grade_keys = set(latest.get("active_grades", []))
        for event in events:
            kind = event.get("event")
            job = event.get("job")
            if (
                isinstance(job, dict)
                and job.get("family") == family
                and kind in {"fit_failed", "admission_failed"}
            ):
                failed_pairs.add((job["fold"], job["seed"]))
                errors.append(f"{job['id']}: {event.get('reason', event.get('error', kind))}")
            if kind == "grade_finished" and event.get("returncode") != 0:
                key = str(event.get("pair", ""))
                match = re.fullmatch(rf"{family}_f(\d+)_s(\d+)", key)
                if match:
                    failed_pairs.add((int(match.group(1)), int(match.group(2))))
                    errors.append(f"{key}: official pair grade exit {event.get('returncode')}")
        for (fold, seed), arms in sorted(pairs.items()):
            score_path = state / "scored" / f"{family}_f{fold}_s{seed}.json"
            if not score_path.is_file():
                continue
            try:
                completed.append(_verify_pair(root, state, family, fold, seed, arms, modules))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                errors.append(f"{family}_f{fold}_s{seed}: {type(error).__name__}: {error}")
                failed_pairs.add((fold, seed))
        completed_cells = {(row["fold"], row["seed"]) for row in completed}
        active_cells = set()
        for (fold, seed), arms in pairs.items():
            if any(job["id"] in active_jobs for job in arms.values()) or (
                f"{family}_f{fold}_s{seed}" in active_grade_keys
            ):
                active_cells.add((fold, seed))
        failed_pairs -= completed_cells
        pending = max(0, len(pairs) - len(completed_cells) - len(active_cells) - len(failed_pairs))
        terminal = bool(summary) and not active_jobs and not active_grade_keys
        note = (
            f"{rule['experiment_id']} is a new PR{4 if family == 'age' else 14} successor; "
            "it does not claim that the source PR was executed. Each reported value is one "
            "matched fold/seed comparison using the original official metric and epoch-5 endpoints."
        )
        snapshot = {
            "experiment_id": rule["experiment_id"],
            "expected_cells": 30,
            "expected_per_arm": 30,
            "completed": sorted(completed, key=lambda row: (row["fold"], row["seed"])),
            "active": len(active_cells),
            "pending": pending,
            "terminal": terminal,
            "source": {
                "commit": source["commit"],
                "trainer_sha256": source["trainer"]["source_identity_sha256"],
                "feature_config_sha256": common.sha(Path(reporter_config["feature_config_path"])),
                "feature_code_files": feature_config["code_files"],
            },
            "contract_note": note,
            "errors": errors[:40],
            "details": {
                "failed_pairs": len(failed_pairs),
                "active_job_ids": sorted(active_jobs),
                "active_grade_pairs": sorted(active_grade_keys),
                "summary_status": summary.get("status"),
                "summary_sha256": common.sha(state / "SUMMARY.json")
                if summary and (state / "SUMMARY.json").is_file()
                else None,
            },
        }
        if _SECRET.search(json.dumps(snapshot, sort_keys=True)):
            raise ValueError("Feature report rejected because a field resembles a credential")
        snapshots.append(snapshot)
    return snapshots


def canonical_sha(value: object) -> str:
    """Hash canonical JSON for durable outbox identity and comment markers."""
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
