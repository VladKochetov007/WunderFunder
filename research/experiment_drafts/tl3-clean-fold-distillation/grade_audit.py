"""Artifact and provenance gates for the TL3 clean-fold KD grader."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any  # Any covers schema-checked artifacts and canonical split adapters.

from grade_artifacts import _architecture_args, _little_endian_groups, _same_membership


def _valid_sha(value: Any) -> bool:
    """Return whether a value is a lowercase-or-uppercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def build_run_plan(
    recipe: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Validate the fixed F0/F2 panel and derive every expected run and mixture weight."""
    raw_cells = recipe.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 2:
        raise ValueError("recipe must expose every cell: exactly F0 and F2 are required")
    expected = [("F0", 0, 1001), ("F2", 2, 1002)]
    if [(item.get("id"), item.get("fold"), item.get("seed")) for item in raw_cells] != expected:
        raise ValueError("recipe cell identity differs from the frozen F0/F2 screen")
    epochs = recipe.get("fit", {})
    if epochs.get("teacher_epochs") != 4 or epochs.get("student_epochs") != 2:
        raise ValueError("recipe must retain four teacher and two student epochs")
    plan: list[dict[str, Any]] = []
    for cell in raw_cells:
        teachers = cell.get("teachers")
        if not isinstance(teachers, list) or len(teachers) != 2:
            raise ValueError(f"{cell['id']} must declare both cold teacher seeds")
        if teachers != ([2001, 2002] if cell["id"] == "F0" else [2003, 2004]):
            raise ValueError(f"{cell['id']} teacher seeds differ from the frozen recipe")
        if any(type(seed) is not int or seed <= 0 for seed in teachers):
            raise ValueError(f"{cell['id']} teacher seeds must be positive integers")
        for role, seed in zip(("teacher_a", "teacher_b"), teachers, strict=True):
            plan.append(_run_record(cell, role, seed, 4, "teacher"))
        for role in ("labels", "kd"):
            plan.append(_run_record(cell, role, int(cell["seed"]), 2, "student"))

    raw_weights = recipe.get("metrics", {}).get("full_mixture_weights")
    required = {"student": 0.34, "teacher_a": 0.33, "teacher_b": 0.33}
    if not isinstance(raw_weights, dict) or set(raw_weights) != set(required):
        raise ValueError("recipe must pin student/teacher_a/teacher_b mixture weights")
    weights = {key: float(value) for key, value in raw_weights.items()}
    if weights != required:
        raise ValueError("full-mixture weights must stay at 0.34/0.33/0.33")
    teacher_weights = recipe.get("metrics", {}).get("teacher_only_weights")
    if teacher_weights != {"teacher_a": 0.5, "teacher_b": 0.5}:
        raise ValueError("teacher-only reference must remain the equal teacher mean")
    return plan, weights


def _run_record(
    cell: dict[str, Any], role: str, seed: int, epochs: int, kind: str
) -> dict[str, Any]:
    """Return one immutable run identity from a recipe cell."""
    fold_id = str(cell["id"]).lower()
    return {
        "cell": str(cell["id"]),
        "fold": int(cell["fold"]),
        "cell_seed": int(cell["seed"]),
        "role": role,
        "seed": seed,
        "epochs": epochs,
        "kind": kind,
        "run": f"tl3kd_r1_{fold_id}_{role}",
    }


def verify_run(
    record: dict[str, Any],
    project_root: Path,
    evidence_root: Path,
    split: Any,
    split_sha: str,
    recipe_sha: str,
    source_sha: str,
    fit_audit: dict[str, Any],
    digest: Callable[[Path], str],
    read_json: Callable[[Path], dict[str, Any]],
    teacher_targets: list[dict[str, Any]] | None = None,
    teacher_cache_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify one passed run audit, fixed checkpoint, clean membership, and prediction sidecar."""
    run = str(record["run"])
    run_dir = project_root / "artifacts" / "runs" / run
    checkpoint_name = f"model_ep{int(record['epochs']) - 1}.pt"
    checkpoint = run_dir / checkpoint_name
    provenance_path = run_dir / "provenance.json"
    prediction_path = project_root / "cache" / f"pred_{run}_eval_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    audit_path = evidence_root / "audit" / f"{run}.json"
    terminal_path = evidence_root / "terminal" / f"{run}.json"
    for path in (
        checkpoint,
        provenance_path,
        prediction_path,
        sidecar_path,
        audit_path,
        terminal_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{run} is missing required evidence: {path}")
    checkpoint_sha, provenance_sha = digest(checkpoint), digest(provenance_path)
    prediction_sha, sidecar_sha = digest(prediction_path), digest(sidecar_path)
    audit_sha = digest(audit_path)
    terminal_sha = digest(terminal_path)
    audit = fit_audit
    if read_json(audit_path) != audit:
        raise ValueError(f"{run} audit receipt changed while being graded")
    audit_fields = {
        "checkpoint_sha256": checkpoint_sha,
        "provenance_sha256": provenance_sha,
        "source_sha256": source_sha,
        "recipe_sha256": recipe_sha,
        "split_sha256": split_sha,
    }
    if audit.get("status") != "passed" or audit.get("run") != run:
        raise ValueError(f"{run} has no successful matching run audit")
    if any(audit.get(key) != value for key, value in audit_fields.items()):
        raise ValueError(f"{run} audit hashes do not bind the current run inputs")
    terminal = read_json(terminal_path)
    if (
        terminal_sha != audit.get("terminal_sha256")
        or terminal.get("status") != "completed"
        or terminal.get("run") != run
        or terminal.get("profile")
        or audit.get("completed_epochs") != record["epochs"]
    ):
        raise ValueError(f"{run} has no byte-bound completed scientific run")
    if digest(checkpoint) != checkpoint_sha or digest(provenance_path) != provenance_sha:
        raise ValueError(f"{run} model/provenance changed while being graded")
    for key in ("runtime_sha256", "initial_state_sha256"):
        if not _valid_sha(audit.get(key)):
            raise ValueError(f"{run} audit lacks a valid {key}")
    provenance = read_json(provenance_path)
    for section in (audit.get("fit_row_groups"), provenance.get("fit_row_groups")):
        if not isinstance(section, dict):
            raise ValueError(f"{run} audit/provenance lacks fit memberships")
        _same_membership(section.get("train"), split.train_groups, f"{run} train")
        _same_membership(section.get("valid"), split.valid_groups, f"{run} valid-fit")
    args = provenance.get("args", {})
    fixed_args = {
        "seed": record["seed"],
        "train_source": "train+valid",
        "valid_fold": record["fold"],
        "epochs": record["epochs"],
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": int(record["epochs"]) - 1,
        "no_eval_during_fit": True,
        "include_heldout_valid": False,
        "init": "",
        "split_manifest_sha256": split_sha,
    }
    fixed_args.update(_architecture_args(record))
    if any(args.get(key) != value for key, value in fixed_args.items()):
        raise ValueError(f"{run} provenance differs from the fixed clean-fold fit")
    if provenance.get("split_manifest_sha256") != split_sha:
        raise ValueError(f"{run} provenance has a different campaign split")
    if (
        provenance.get("selected_epoch") != int(record["epochs"]) - 1
        or provenance.get("checkpoint_sha256", {}).get(checkpoint_name) != checkpoint_sha
    ):
        raise ValueError(f"{run} provenance does not bind its fixed final checkpoint")
    if record["kind"] == "student":
        if teacher_targets is None or teacher_cache_files is None:
            raise ValueError(f"{run} lacks verified teacher-cache inputs")
        expected_alpha = 0.0 if record["role"] == "labels" else 0.5
        if float(args.get("distill_alpha", -1.0)) != expected_alpha:
            raise ValueError(f"{run} does not use its prescribed label/KD alpha")
        if (
            provenance.get("distillation_cache")
            or args.get("distill_cache_root")
            or not _teacher_provenance_matches(args, teacher_cache_files)
        ):
            raise ValueError(f"{run} does not bind the exact two fold-teacher caches")
        actual_targets = audit.get("teacher_targets")
        if (
            not isinstance(actual_targets, list)
            or len(actual_targets) != 2
            or any(
                not isinstance(actual, dict)
                or any(actual.get(key) != value for key, value in expected.items())
                for actual, expected in zip(actual_targets, teacher_targets, strict=True)
            )
        ):
            raise ValueError(f"{run} audit teacher targets differ from verified cache bytes")
    elif (
        provenance.get("distillation_cache")
        or args.get("distill")
        or args.get("distill_cache_root")
    ):
        raise ValueError(f"{run} teacher unexpectedly depends on student targets")
    if provenance.get("evaluation_row_groups") != list(split.evaluation_groups) or audit.get(
        "evaluation_row_groups"
    ) != list(split.evaluation_groups):
        raise ValueError(f"{run} evaluation membership differs from the held-out split")
    metadata = read_json(sidecar_path)
    expected_membership = hashlib.sha256(_little_endian_groups(split.evaluation_groups)).hexdigest()
    sidecar_fields = {
        "run": run,
        "split": "valid",
        "epoch": int(record["epochs"]) - 1,
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": prediction_sha,
        "split_manifest_sha256": split_sha,
        "campaign_split_manifest_sha256": split_sha,
        "campaign_export_scope": "evaluation",
        "evaluation_membership_sha256": expected_membership,
        "row_groups": list(split.evaluation_groups),
        "feature_only": True,
        "inference_precision": "fp32",
    }
    if any(metadata.get(key) != value for key, value in sidecar_fields.items()):
        raise ValueError(f"{run} prediction sidecar does not bind the exact held-out export")
    return {
        **record,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "provenance_path": str(provenance_path),
        "provenance_sha256": provenance_sha,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "terminal_path": str(terminal_path),
        "terminal_sha256": terminal_sha,
        "source_sha256": source_sha,
        "runtime_sha256": audit["runtime_sha256"],
        "initial_state_sha256": audit["initial_state_sha256"],
        "normalization_identity": audit["normalization_identity"],
        "sample_order_identities": audit["sample_order_identities"],
        "teacher_targets": audit.get("teacher_targets"),
        "training_args": args,
    }


def _teacher_provenance_matches(args: dict[str, Any], cache_files: list[dict[str, Any]]) -> bool:
    """Match student fit provenance to both current teacher prediction files."""
    observed = args.get("distill_provenance")
    if (
        not isinstance(observed, list)
        or len(observed) != 4
        or any(not isinstance(item, dict) for item in observed)
    ):
        return False
    runs = list(dict.fromkeys(item["run"] for item in cache_files))
    distill = args.get("distill", "")
    if not isinstance(distill, str):
        return False
    try:
        declared = [
            (name, float(scale or 1.0))
            for item in distill.split(",")
            for name, _, scale in (item.partition("*"),)
        ]
    except ValueError:
        return False
    expected = {
        (
            item["run"],
            item["split"],
            1.0,
            item["epoch"],
            item["checkpoint_sha256"],
            item["prediction_sha256"],
        )
        for item in cache_files
    }
    actual = {
        (
            item["run"],
            item["split"],
            item["scale"],
            item["epoch"],
            item["checkpoint_sha256"],
            item["prediction_sha256"],
        )
        for item in observed
    }
    return declared == [(run, 1.0) for run in runs] and actual == expected
