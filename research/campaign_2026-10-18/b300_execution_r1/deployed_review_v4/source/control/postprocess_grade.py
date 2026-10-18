"""Read-only linked grade views and official paired-endpoint invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .postprocess_common import (
        ACTOR_SHA256,
        GRADER_SHA256,
        SOURCE_TREE_SHA256,
        _check_disk,
        _child_environment,
        _execute,
        _read_json,
        _remaining,
        _sha256,
        _write_new_json,
    )
except ImportError:
    from postprocess_common import (
        ACTOR_SHA256,
        GRADER_SHA256,
        SOURCE_TREE_SHA256,
        _check_disk,
        _child_environment,
        _execute,
        _read_json,
        _remaining,
        _sha256,
        _write_new_json,
    )

_CACHE_FILES = (
    "chains.npz",
    "chain_folds.npz",
    "valid_targets.npz",
    "train_targets.npz",
)


def _link(source: Path, destination: Path, *, directory: bool) -> None:
    """Create an immutable-view symlink without replacing any prior input."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace grade-view input {destination}")
    destination.symlink_to(source.resolve(strict=True), target_is_directory=directory)


def _grade_view(
    root: Path,
    fold: int,
    fits: dict[str, dict[str, Any]],
    exports: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, str]]:
    """Make the grader's expected tree from read-only links to frozen inputs."""
    fold_root = root / "grades" / f"seed{fits['wide']['job']['seed']}" / f"fold{fold}"
    view = fold_root / "view"
    if fold_root.exists() and any(fold_root.iterdir()):
        raise FileExistsError(f"grade attempt already exists; refusing to overwrite {fold_root}")
    _check_disk(root)
    fold_root.mkdir(parents=True, exist_ok=True)
    view.mkdir()
    _link(root / "data" / "datasets", view / "datasets", directory=True)
    cache = view / "cache"
    cache.mkdir()
    link_map: dict[str, str] = {"datasets": str(root / "data" / "datasets")}
    for name in _CACHE_FILES:
        source = root / "data" / "cache" / name
        _link(source, cache / name, directory=False)
        link_map[f"cache/{name}"] = str(source)
    artifact_runs = view / "artifacts" / "runs"
    artifact_runs.mkdir(parents=True)
    for arm in ("wide", "deep"):
        fit, export = fits[arm], exports[arm]
        run_name = str(fit["job"]["id"])
        artifact = fit["run_dir"] / "artifacts" / "runs" / run_name
        _link(artifact, artifact_runs / run_name, directory=True)
        for suffix in ("npy", "json"):
            source = export["output_root"] / "cache" / f"pred_{run_name}_valid.{suffix}"
            _link(source, cache / source.name, directory=False)
        link_map[f"artifacts/runs/{run_name}"] = str(artifact)
    for path in (view, cache, view / "artifacts", artifact_runs):
        path.chmod(0o555)
    return view, link_map


def _grade_argv(
    paths: dict[str, Path],
    view: Path,
    candidate: str,
    reference: str,
    split: Path,
    output: Path,
    scratch: Path,
) -> list[str]:
    """Build the unchanged official R4 grader command for one fixed pair."""
    return [
        str(paths["python"]),
        str(paths["grader"]),
        "--root",
        str(view),
        "--run",
        candidate,
        "--split",
        str(split),
        "--epoch",
        "5",
        "--reference",
        f"{reference}@5",
        "--output",
        str(output),
        "--bootstrap",
        "2000",
        "--scratch-dir",
        str(scratch),
    ]


def _existing_grade(fold_root: Path, fold: int, candidate: str, reference: str) -> dict[str, Any]:
    """Reconcile only a fully receipt-bound successful earlier grade attempt."""
    receipt_path = fold_root / "grade_receipt.json"
    grade_path = fold_root / "grade.json"
    command_path = fold_root / "grade_command.json"
    try:
        receipt, grade = _read_json(receipt_path), _read_json(grade_path)
        execution = receipt.get("execution")
        grade_output = receipt.get("grade_output")
        stdout, stderr = receipt.get("stdout"), receipt.get("stderr")
        if not all(isinstance(item, dict) for item in (execution, grade_output, stdout, stderr)):
            raise ValueError("grade receipt is missing execution, output, or log bindings")
        if (
            receipt.get("status") == "graded"
            and receipt.get("fold") == fold
            and receipt.get("candidate") == candidate
            and receipt.get("reference") == reference
            and receipt.get("grader_sha256") == GRADER_SHA256
            and grade_output.get("sha256") == _sha256(grade_path)
            and receipt.get("command_sha256") == _sha256(command_path)
            and execution.get("exit_code") == 0
            and stdout.get("sha256") == _sha256(fold_root / "grade.stdout.txt")
            and stderr.get("sha256") == _sha256(fold_root / "grade.stderr.txt")
            and grade.get("fold") == fold
            and grade.get("endpoint", {}).get("run") == candidate
            and grade.get("reference", {}).get("run") == reference
            and grade.get("endpoint", {}).get("epoch") == 5
            and grade.get("reference", {}).get("epoch") == 5
            and grade.get("metric_id") == receipt.get("official_metric_id")
            and grade.get("paired_component_bootstrap", {}).get("resamples") == 2000
        ):
            return receipt
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"fold": fold, "status": "existing_grade_attempt_preserved", "path": str(fold_root)}


def _grade_pair(
    root: Path,
    paths: dict[str, Path],
    fold: int,
    fits: dict[str, dict[str, Any]],
    exports: dict[str, dict[str, Any]],
    deadline: float,
) -> dict[str, Any]:
    """Run the unchanged official scorer once for one exact wide/deep fold pair."""
    wide, deep = fits["wide"], fits["deep"]
    if wide["job"]["seed"] != deep["job"]["seed"]:
        raise ValueError("paired endpoints must share the same seed")
    if wide["split_sha256"] != deep["split_sha256"] or wide["groups"] != deep["groups"]:
        raise ValueError(f"paired fold {fold} does not share one exact evaluation membership")
    fold_root = root / "grades" / f"seed{fits['wide']['job']['seed']}" / f"fold{fold}"
    candidate, reference = str(deep["job"]["id"]), str(wide["job"]["id"])
    if fold_root.exists() and any(fold_root.iterdir()):
        return _existing_grade(fold_root, fold, candidate, reference)
    _remaining(deadline)
    scratch = root / "scratch" / "grading" / f"seed{fits['wide']['job']['seed']}" / f"fold{fold}"
    if scratch.exists() and any(scratch.iterdir()):
        return {"fold": fold, "status": "existing_grade_scratch_preserved", "path": str(scratch)}
    scratch.mkdir(parents=True, exist_ok=True)
    temp = scratch / "tmp"
    temp.mkdir(exist_ok=True)
    _check_disk(root)
    view, links = _grade_view(root, fold, fits, exports)
    grade_path = fold_root / "grade.json"
    command_path = fold_root / "grade_command.json"
    argv = _grade_argv(paths, view, candidate, reference, deep["split_path"], grade_path, scratch)
    command = {
        "schema": "tabdata-b300-grade-command.v1",
        "argv": argv,
        "grader_sha256": GRADER_SHA256,
        "actor_bundle_sha256": ACTOR_SHA256,
        "source_tree_identity_sha256": SOURCE_TREE_SHA256,
        "fold": fold,
        "candidate": candidate,
        "reference": reference,
        "epoch": 5,
        "split_manifest_sha256": deep["split_sha256"],
        "evaluation_groups": deep["groups"],
        "fit_receipts": {arm: fits[arm]["fit_receipt_sha256"] for arm in ("wide", "deep")},
        "export_receipts": {arm: exports[arm]["export_receipt_sha256"] for arm in ("wide", "deep")},
        "grade_view_links": links,
        "prediction_shapes": {arm: exports[arm]["shape"] for arm in ("wide", "deep")},
        "prediction_dtypes": {arm: exports[arm]["dtype"] for arm in ("wide", "deep")},
        "bootstrap": 2000,
    }
    _write_new_json(command_path, command)
    env = _child_environment(root, "", scratch, grading=True)
    env["PYTHONPATH"] = str(paths["source"] / "src")
    stdout, stderr = fold_root / "grade.stdout.txt", fold_root / "grade.stderr.txt"
    _remaining(deadline)
    _check_disk(root)
    cpus = sorted(deep["job"]["cpu_ids_initial"])
    execution = _execute(argv, env, root, stdout, stderr, deadline, cpus)
    grade: dict[str, Any] | None = None
    status = "failed"
    if execution.get("exit_code") == 0 and grade_path.is_file():
        grade = _read_json(grade_path)
        endpoint, scored_reference = grade.get("endpoint", {}), grade.get("reference", {})
        if (
            not grade.get("metric_id")
            or grade.get("fold") != fold
            or grade.get("split_manifest_sha256") != deep["split_sha256"]
            or endpoint.get("run") != candidate
            or endpoint.get("epoch") != 5
            or scored_reference.get("run") != reference
            or scored_reference.get("epoch") != 5
            or grade.get("paired_component_bootstrap", {}).get("resamples") != 2000
        ):
            raise ValueError(
                f"official grade output does not bind the requested pair at fold {fold}"
            )
        status = "graded"
    receipt = {
        "schema": "tabdata-b300-grade-receipt.v1",
        "status": status,
        "fold": fold,
        "candidate": candidate,
        "reference": reference,
        "epoch": 5,
        "grader_sha256": GRADER_SHA256,
        "actor_bundle_sha256": ACTOR_SHA256,
        "source_tree_identity_sha256": SOURCE_TREE_SHA256,
        "command_sha256": _sha256(command_path),
        "execution": execution,
        "stdout": {"path": str(stdout), "bytes": stdout.stat().st_size, "sha256": _sha256(stdout)},
        "stderr": {"path": str(stderr), "bytes": stderr.stat().st_size, "sha256": _sha256(stderr)},
        "grade_output": (
            {
                "path": str(grade_path),
                "bytes": grade_path.stat().st_size,
                "sha256": _sha256(grade_path),
            }
            if grade_path.is_file()
            else None
        ),
        "official_metric_id": grade.get("metric_id") if grade else None,
        "candidate_score": grade.get("endpoint", {}).get("score") if grade else None,
        "reference_score": grade.get("reference", {}).get("score") if grade else None,
    }
    _write_new_json(fold_root / "grade_receipt.json", receipt)
    return receipt
