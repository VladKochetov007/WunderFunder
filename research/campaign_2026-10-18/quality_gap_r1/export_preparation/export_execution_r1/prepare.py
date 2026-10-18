"""Bind a fresh root admission and actual R2 receipt into one export contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from bindings import (
    B_GUARD,
    B_GUARD_STATE,
    B_LIMITED,
    B_PROJECT_ROOT,
    B_PYTHON,
    B_TMP,
    EPOCH,
    EXECUTION_DIR,
    MEMBERSHIP_REL,
    OLD_SPLIT_REL,
    OUTPUT_NAME,
    OVERLAY_REL,
    PINS,
    R2_RECEIPT_REL,
    R2_REVIEW_REL,
    REGISTRY_REL,
    RESOURCE_ENVELOPE,
    RUN_NAME,
    check_overlay,
    output_paths,
    parse_utc,
    read_json,
    reject_bytecode,
    require_outputs_absent,
    sha256_file,
    validate_preflight_receipt,
    validate_root_admission,
    write_json_exclusive,
)

ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def _check_regular(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"metadata identity mismatch: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _check_snapshot(project_root: Path) -> dict[str, Any]:
    snapshot = project_root / "frozen_source_snapshot"
    manifest_path = snapshot / "SOURCE_MANIFEST_H390.json"
    if sha256_file(manifest_path) != PINS["source_manifest_sha256"]:
        raise ValueError("frozen source manifest changed")
    manifest = read_json(manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("frozen snapshot source list is invalid")
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "SOURCE_MANIFEST_H390.json"
    }
    if actual != set(sources):
        raise ValueError("frozen snapshot file inventory changed")
    reject_bytecode(snapshot)
    for relative, expected in sources.items():
        source = snapshot / relative
        if source.is_symlink() or sha256_file(source) != expected:
            raise ValueError(f"frozen source file changed: {relative}")
    predictor = snapshot / "scripts/predict_run.py"
    if sha256_file(predictor) != PINS["predictor_sha256"]:
        raise ValueError("frozen predictor digest changed")
    if any(path.is_symlink() for path in snapshot.rglob("*")):
        raise ValueError("frozen snapshot contains a symlink")
    return {
        "path": str(snapshot),
        "source_manifest_sha256": PINS["source_manifest_sha256"],
        "predictor_sha256": PINS["predictor_sha256"],
        "file_count": len(sources),
    }


def _check_small_inputs(project_root: Path, checkpoint_bindings: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "old_split": _check_regular(project_root / OLD_SPLIT_REL, PINS["old_split_sha256"]),
        "portable_registry": _check_regular(project_root / REGISTRY_REL, PINS["registry_sha256"]),
        "membership_receipt": _check_regular(
            project_root / MEMBERSHIP_REL, PINS["membership_sha256"]
        ),
    }
    checked_artifacts: dict[str, dict[str, Any]] = {}
    for label, binding in checkpoint_bindings.items():
        path = project_root / binding["path"]
        expected_target = Path(binding["restore_path"]).resolve(strict=True)
        if not path.is_symlink() or path.resolve(strict=True) != expected_target:
            raise ValueError(
                f"artifact link does not resolve to its pinned restore source: {label}"
            )
        if path.stat().st_size != int(binding["bytes"]):
            raise ValueError(f"restored artifact size changed: {label}")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"restored artifact digest changed: {label}")
        checked_artifacts[label] = {
            "project_path": str(path),
            "resolved_path": str(path.resolve(strict=True)),
            "bytes": path.stat().st_size,
            "sha256": binding["sha256"],
        }
    return {"metadata": metadata, "artifacts": checked_artifacts}


def _check_portable_links(project_root: Path) -> dict[str, Any]:
    registry = read_json(project_root / REGISTRY_REL)
    result: dict[str, Any] = {}
    for logical, entry in registry["sources"].items():
        path = project_root / logical
        expected_source = (
            Path("/home/grinchenkoeg/computer-vision-homework-20261018/data") / logical
        )
        if not path.is_symlink() or os.readlink(path) != str(expected_source):
            raise ValueError(f"portable source link changed: {logical}")
        resolved = path.resolve(strict=True)
        expected_size = (
            462719
            if logical == "cache/chain_folds.npz"
            else int(entry["original_identity"]["bytes"])
        )
        if path.stat().st_size != expected_size:
            raise ValueError(f"portable source size changed: {logical}")
        if logical == "cache/chain_folds.npz":
            if (
                str(expected_source) != PINS["cache_logical_source"]
                or str(resolved) != PINS["cache_resolved_source"]
                or sha256_file(path) != PINS["cache_sha256"]
            ):
                raise ValueError("canonical chain-fold cache identity changed")
        elif expected_size <= 1_000_000 and sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"small portable source digest changed: {logical}")
        result[logical] = {
            "logical_path": logical,
            "literal_source": os.readlink(path),
            "resolved_source": str(resolved),
            "bytes": expected_size,
            "full_hash_deferred_to_frozen_loader": logical != "cache/chain_folds.npz",
        }
    return result


def _runtime_identity() -> dict[str, Any]:
    if Path(sys.executable).resolve() != B_PYTHON.resolve(strict=True):
        raise ValueError("run prepare.py with the B project virtual-environment interpreter")
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "pyarrow", "polars"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    guard_sha256 = sha256_file(B_GUARD)
    limited_sha256 = sha256_file(B_LIMITED)
    if guard_sha256 != PINS["bandit_resources_sha256"]:
        raise ValueError("frozen B bandit_resources.py hash differs from its reviewed pin")
    if limited_sha256 != PINS["limited_sha256"]:
        raise ValueError("frozen B scripts/limited hash differs from its reviewed pin")
    utility_names = (
        "bindings.py",
        "prepare.py",
        "worker.py",
        "launch.py",
        "capture.py",
    )
    utility_code = {
        name: {
            "bytes": (EXECUTION_DIR / name).stat().st_size,
            "sha256": sha256_file(EXECUTION_DIR / name),
        }
        for name in utility_names
    }
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "packages": packages,
        "python_binary_bytes": B_PYTHON.stat().st_size,
        "python_binary_sha256": sha256_file(B_PYTHON),
        "bandit_resources_sha256": guard_sha256,
        "limited_sha256": limited_sha256,
        "export_utility_code": utility_code,
    }


def build_contract(
    attempt_id: str,
    admission_path: Path,
    *,
    require_fresh_admission: bool = True,
    check_outputs: bool = True,
) -> dict[str, Any]:
    """Verify staged inputs and bind actual preflight/admission evidence."""
    if not ATTEMPT_RE.fullmatch(attempt_id):
        raise ValueError("attempt ID must contain 1-80 letters, digits, underscores or dashes")
    project_root = B_PROJECT_ROOT.resolve(strict=True)
    if project_root != B_PROJECT_ROOT:
        raise ValueError("unexpected staged project root")
    if not EXECUTION_DIR.resolve().is_relative_to(project_root):
        raise ValueError("export utility is outside the staged project")
    outputs = (
        require_outputs_absent(project_root)
        if check_outputs
        else {key: str(value) for key, value in output_paths(project_root).items()}
    )
    overlay_dir = project_root / OVERLAY_REL
    overlay = check_overlay(overlay_dir, project_root / R2_REVIEW_REL)
    preflight_path = project_root / R2_RECEIPT_REL
    preflight = validate_preflight_receipt(
        preflight_path, overlay_manifest=read_json(overlay_dir / "OVERLAY_MANIFEST_R2.json")
    )
    admission = validate_root_admission(
        admission_path,
        attempt_id=attempt_id,
        preflight_sha256=preflight["sha256"],
        require_fresh=require_fresh_admission,
    )
    split = _check_regular(project_root / OLD_SPLIT_REL, PINS["old_split_sha256"])
    registry = _check_regular(project_root / REGISTRY_REL, PINS["registry_sha256"])
    membership = _check_regular(project_root / MEMBERSHIP_REL, PINS["membership_sha256"])
    snapshot = _check_snapshot(project_root)
    inputs = _check_small_inputs(project_root, overlay["checkpoint_bindings"])
    sources = _check_portable_links(project_root)
    runtime = _runtime_identity()
    pythonpath = (
        f"{project_root / OVERLAY_REL / 'src'}:{project_root / 'frozen_source_snapshot/src'}"
    )
    return {
        "schema": "quality_gap.f0_strong_control_export_contract.r1",
        "status": "READY_FOR_ROOT_GUARDED_GPU_EXPORT",
        "created_utc": parse_utc(admission["created_utc"]).isoformat(),
        "attempt_id": attempt_id,
        "owner": "root",
        "project_root": str(project_root),
        "execution_dir": str(EXECUTION_DIR),
        "resource_envelope": RESOURCE_ENVELOPE,
        "gpu_uuid": admission["gpu_uuid"],
        "root_admission": admission,
        "preflight_receipt": preflight,
        "pins": {
            **PINS,
            "overlay_review_sha256": overlay["review_sha256"],
            "r2_receipt_sha256": preflight["sha256"],
            "old_split_bytes": split["bytes"],
            "registry_bytes": registry["bytes"],
            "membership_bytes": membership["bytes"],
        },
        "source_snapshot": snapshot,
        "overlay": overlay,
        "metadata": {"old_split": split, "registry": registry, "membership": membership},
        "artifacts": inputs["artifacts"],
        "portable_source_links": sources,
        "outputs_create_only": outputs,
        "runtime": runtime,
        "execution": {
            "python": str(B_PYTHON),
            "limited": str(B_LIMITED),
            "guard": str(B_GUARD),
            "guard_state": str(B_GUARD_STATE),
            "guard_unit_template": "wnn-bandit-{attempt}.service",
            "physical_tmp": str(B_TMP),
            "pythonpath": pythonpath,
            "predictor": str(project_root / "frozen_source_snapshot/scripts/predict_run.py"),
            "command": [
                RUN_NAME,
                "valid",
                "--epoch",
                str(EPOCH),
                "--output-name",
                OUTPUT_NAME,
                "--campaign-split-manifest",
                str(project_root / OLD_SPLIT_REL),
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
                "--campaign-full-valid-feature-only",
            ],
        },
        "expected_output": {
            "split": "valid",
            "epoch": EPOCH,
            "row_groups": 1873,
            "shape": [1873, 20000, 2],
            "dtype": "<f4",
            "bytes": 299680128,
            "grading_run": False,
            "training_run": False,
        },
        "unverified_gates": [
            "No measured runtime or throughput is available for this exact full-source hash pass plus strong-control export; the 7200-second timeout is a hard stop, not a forecast.",
            "The 1-CPU/8-GB guard reservation and 4-GB PyTorch allocator limit have not been measured for this exact control; profile is explicitly marked unmeasured.",
            "GPU UUID and active-claim state are taken from the five-minute root admission and are rechecked by bandit_resources at launch.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--root-admission", type=Path, required=True)
    args = parser.parse_args()
    record_dir = EXECUTION_DIR / "records" / args.attempt
    contract_path = EXECUTION_DIR / "contracts" / f"{args.attempt}.json"
    payload = build_contract(args.attempt, args.root_admission)
    payload["contract_path"] = str(contract_path)
    write_json_exclusive(contract_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "contract": str(contract_path),
                "contract_sha256": sha256_file(contract_path),
                "record_dir": str(record_dir),
                "attempt_id": args.attempt,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
