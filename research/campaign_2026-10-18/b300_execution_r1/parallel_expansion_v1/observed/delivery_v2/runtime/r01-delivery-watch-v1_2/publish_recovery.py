"""Publish source-qualified scores for already recovered R01 endpoints only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY = "VladKochetov007/WunderFunder"
PROJECT_ROOT = "/data/epanov_s/tabdata"
PUBLISHER_LOGIN = "SemyonEpanov"
PR_NUMBER = 25
HEAD_REF = "experiment/tabdata-b300-r01-cp30"
EXPERIMENT_ID = "E-B300-R01-DELIVERY-RECOVERY-V2"
PANEL_SHA256 = "471f465cc1d13fc6e5153a4c998390bb14e06823fed4c6721f929ce95fee47fb"
EXECUTION_MANIFEST_SHA256 = "95f82d442eefe425dde6574222b037ba1deba2d3ae36534aeec41e31e450874f"
HELPER_MANIFEST_SHA256 = "1af1ac94958f134ce87aa644b9fdb55b56c413b34f8649a67ab05038ccd3b6be"
ACTIVE_HELPER_MANIFEST_SHA256 = "d5ac193fc3241b7e54d985e71077167f1170fb0c2300967631226937ab22ae8f"
GITHUB_CODE_MANIFEST_SHA256 = "7ced43fa963239fd3b3b33dc049997ac127f57867cbe3b20b3eae2a392626328"
GRADER_SHA256 = "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31"
ACTOR_SHA256 = "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220"
SOURCE_TREE_SHA256 = "f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130"
METRIC_ID = "global_wp_clip2_v1"
_SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)")


def sha256_file(path: Path) -> str:
    """Hash a regular nonsymlink file in bounded memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _preserve_split_manifest(root: Path, fold: int, expected_sha256: str) -> Path:
    """Copy a bound split into publisher state for the archive allowlist, without replacement."""
    if (
        type(fold) is not int
        or fold not in range(4)
        or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
    ):
        raise ValueError("split evidence identity is malformed")
    source = root / "splits" / f"fold{fold}.json"
    if source.is_symlink() or sha256_file(source) != expected_sha256:
        raise ValueError(f"canonical fold {fold} split differs from the verified grade")
    payload = source.read_bytes()
    target_dir = root / "state/r01-delivery-recovery-v2/publisher-v1/split_manifests"
    if any(
        parent.is_symlink()
        for parent in (
            root / "state",
            root / "state/r01-delivery-recovery-v2",
            target_dir.parent,
            target_dir,
        )
    ):
        raise ValueError("split evidence destination contains a symbolic link")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"fold{fold}-{expected_sha256}.json"
    if target.is_symlink():
        raise ValueError("preserved split evidence cannot be a symbolic link")
    if not target.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".split-", dir=target_dir)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or sha256_file(target) != expected_sha256:
                    raise ValueError("competing preserved split bytes differ") from None
            else:
                if os.name != "nt":
                    directory_fd = os.open(target_dir, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if (
        target.is_symlink()
        or not target.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        or sha256_file(target) != expected_sha256
    ):
        raise ValueError("preserved split evidence failed exact-byte verification")
    return target


def _archive_evidence_paths(
    root: Path, fold: int, expected_sha256: str, evidence_paths: list[str]
) -> list[str]:
    """Replace the canonical split path with its exact state-local archive copy."""
    source = str((root / "splits" / f"fold{fold}.json").resolve(strict=True))
    if source not in evidence_paths:
        raise ValueError("verified endpoint evidence omits its canonical split manifest")
    preserved = _preserve_split_manifest(root, fold, expected_sha256)
    return sorted({path for path in evidence_paths if path != source} | {str(preserved)})


def read_json(path: Path) -> dict[str, Any]:
    """Read one object-valued receipt without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink JSON receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a regular object-valued JSON receipt: {path}")
    return value


def _verify_manifest(path: Path, expected_sha: str, expected_schema: str) -> dict[str, Any]:
    """Verify one pinned helper manifest and every safe member before use."""
    if sha256_file(path) != expected_sha:
        raise ValueError("runtime helper manifest differs from its literal pin")
    manifest = read_json(path)
    members = manifest.get("files")
    if manifest.get("schema") != expected_schema or not isinstance(members, dict) or not members:
        raise ValueError("runtime helper manifest has an unexpected schema or empty file map")
    base = path.parent.resolve(strict=True)
    for relative, digest in members.items():
        member = Path(relative)
        target = path.parent / member
        if (
            not isinstance(relative, str)
            or member.is_absolute()
            or ".." in member.parts
            or not isinstance(digest, str)
            or target.is_symlink()
            or not target.resolve(strict=True).is_relative_to(base)
            or sha256_file(target) != digest
        ):
            raise ValueError(f"runtime helper member failed its manifest: {relative}")
    return manifest


def _load_placement_verifier(root: Path, config: dict[str, Any]) -> Any:
    """Verify historical and active helper pins, then load only the pinned placement gate."""
    historical_path = root / "runtime/r01-delivery-recovery-v2_3/MANIFEST.json"
    active_path = root / "runtime/r01-delivery-recovery-v2_4/MANIFEST.json"
    if (
        Path(config["helper_manifest_path"]).resolve(strict=True)
        != active_path.resolve(strict=True)
        or config.get("helper_manifest_sha256") != ACTIVE_HELPER_MANIFEST_SHA256
    ):
        raise ValueError("active R01 helper path or literal v2.4 hash differs")
    _verify_manifest(
        historical_path, HELPER_MANIFEST_SHA256, "tabdata-r01-delivery-recovery-manifest.v2.3"
    )
    active = _verify_manifest(
        active_path,
        ACTIVE_HELPER_MANIFEST_SHA256,
        "tabdata-r01-delivery-recovery-manifest.v2.4",
    )
    if active.get("previous_helper_manifest_sha256") != HELPER_MANIFEST_SHA256:
        raise ValueError("active helper does not preserve the exact v2.3 historical pin")
    files = active.get("files", {})
    if "r01_recovery_placement.py" not in files or "r01_recovery_common.py" not in files:
        raise ValueError("active helper omits its pinned placement verifier dependencies")
    helper_dir = active_path.parent
    path_string = str(helper_dir)
    sys.path[:] = [entry for entry in sys.path if entry != path_string]
    sys.path.insert(0, path_string)
    for name in ("r01_recovery_common", "r01_recovery_placement"):
        sys.modules.pop(name, None)
    import importlib

    module = importlib.import_module("r01_recovery_placement")
    module_path = Path(module.__file__).resolve(strict=True)
    if not callable(getattr(module, "verify_placement", None)) or not module_path.is_relative_to(
        helper_dir.resolve(strict=True)
    ):
        raise ValueError("placement verifier resolved outside the pinned active helper")
    return module.verify_placement


def _verify_runtime(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], Path, Any]:
    """Check the active placement helper and original R01 packet/source pins."""
    placement_verifier = _load_placement_verifier(root, config)

    packet = root / "runtime/b300-r01-v1"
    execution_path = packet / "EXECUTION_MANIFEST.json"
    _verify_manifest(
        execution_path, EXECUTION_MANIFEST_SHA256, "tabdata-b300-execution-manifest.v4"
    )
    panel_path = packet / "PANEL.json"
    if sha256_file(panel_path) != PANEL_SHA256:
        raise ValueError("original R01 panel differs from the frozen source-qualified panel")
    panel = read_json(panel_path)
    identity = panel.get("actor_bundle_identity", {})
    if (
        panel.get("experiment_id") != "E-B300-R01"
        or panel.get("cells_per_arm") != 30
        or identity.get("sha256") != ACTOR_SHA256
        or identity.get("source_tree_identity_sha256") != SOURCE_TREE_SHA256
    ):
        raise ValueError("original R01 panel does not bind the fixed source and CP30 recipe")
    source_root = root / "runtime/source-a266-lf-subset-v1"
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("pinned R01 source root is missing or symlinked")
    actor_files = identity.get("files")
    if not isinstance(actor_files, dict) or not actor_files:
        raise ValueError("R01 panel has no actor-bundle file map")
    for relative, digest in actor_files.items():
        path = packet / "code" / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or sha256_file(path) != digest
        ):
            raise ValueError(f"R01 actor file differs from its panel pin: {relative}")
    actor_digest = hashlib.sha256(
        json.dumps(actor_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actor_digest != ACTOR_SHA256:
        raise ValueError("R01 actor bundle map differs from its frozen identity")
    deployed_manifest = read_json(packet / "code/source_package_manifest.json")
    deployed_files = deployed_manifest.get("files")
    if (
        deployed_manifest.get("schema") != "tabdata-deployed-subset-manifest.v1"
        or deployed_manifest.get("deployed_tree_identity_sha256") != SOURCE_TREE_SHA256
        or not isinstance(deployed_files, dict)
        or deployed_manifest.get("file_count") != len(deployed_files)
    ):
        raise ValueError("deployed R01 source package manifest differs from the frozen tree")
    for relative, record in deployed_files.items():
        path = source_root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, dict)
            or sha256_file(path) != record.get("deployed_sha256")
        ):
            raise ValueError(f"deployed R01 source member changed: {relative}")
    source_files = read_json(packet / "code/a266_source_manifest.json").get("files")
    if not isinstance(source_files, dict) or len(source_files) != 61:
        raise ValueError("R01 portable source manifest is not the pinned 61-file map")
    for relative, digest in source_files.items():
        path = source_root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or sha256_file(path) != digest
        ):
            raise ValueError(f"pinned R01 source file changed: {relative}")
    return panel, source_root, placement_verifier


def _load_grader(root: Path, source_root: Path) -> Any:
    """Load only the pinned official CPU-side checkpoint/prediction validators."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    path = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    if sha256_file(path) != GRADER_SHA256:
        raise ValueError("official original R01 grader differs from its frozen source pin")
    spec = importlib.util.spec_from_file_location("r01_recovery_original_grader", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the pinned original R01 grader helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("recovery reporter unexpectedly initialized CUDA")
    wnn = sys.modules.get("wnn")
    if wnn is not None and not Path(wnn.__file__).resolve().is_relative_to(source_root / "src"):
        raise ValueError("official grader imported a different trainer source tree")
    return module


def _job_map(panel: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    """Validate and index the original 30 same-fold/same-seed wide/deep cells."""
    jobs = panel.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise ValueError("R01 panel must contain exactly 60 original fit jobs")
    pairs: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for job in jobs:
        if not isinstance(job, dict) or job.get("arm") not in {"wide", "deep"}:
            raise ValueError("R01 panel contains a malformed endpoint")
        if type(job.get("fold")) is not int or type(job.get("seed")) is not int:
            raise ValueError("R01 panel fold/seed must be integer identities")
        cell = (job["fold"], job["seed"])
        arms = pairs.setdefault(cell, {})
        if job["arm"] in arms:
            raise ValueError(f"duplicate R01 arm in fold/seed {cell}")
        arms[job["arm"]] = job
    if len(pairs) != 30 or any(set(arms) != {"wide", "deep"} for arms in pairs.values()):
        raise ValueError("R01 panel does not contain 30 complete wide/deep pairs")
    return pairs


def _fit_export(
    root: Path,
    recovery: Path,
    job: dict[str, Any],
    placement_verifier: Callable[[Path, dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Verify one previously validated fit and export recovery chain, without side effects."""
    job_id = job["id"]
    fit_path = recovery / "receipts" / f"{job_id}.json"
    fit = read_json(fit_path)
    view = recovery / "fit_views" / job_id
    if (
        fit.get("schema") != "tabdata-r01-fit-recovery.v2"
        or fit.get("status") != "existing_fit_validated_no_refit"
        or fit.get("job") != job
        or fit.get("fit_view") != str(view)
        or fit.get("old_fit_outputs_modified") is not False
        or Path(fit.get("fit_view", "")).is_symlink()
    ):
        raise ValueError(f"fit recovery identity is invalid for {job_id}")
    producer_path = view / "producer_receipt.json"
    if sha256_file(producer_path) != fit.get("recovery_fit_receipt_sha256"):
        raise ValueError(f"isolated fit view receipt changed for {job_id}")
    identity = fit.get("checkpoint_identity", {})
    run_dir = Path(job["run_dir"])
    expected_run = root / "runs" / job_id
    if run_dir.resolve(strict=True) != expected_run.resolve(strict=True):
        raise ValueError(f"original R01 fit path differs from its panel binding: {job_id}")
    artifact = expected_run / "artifacts/runs" / job_id
    fixed_path, selected_path = artifact / "model_ep5.pt", artifact / "model.pt"
    if (
        identity.get("epoch") != 5
        or identity.get("input_width") != 211
        or identity.get("fixed_path") != str(fixed_path.resolve(strict=True))
        or identity.get("selected_path") != str(selected_path.resolve(strict=True))
        or sha256_file(fixed_path) != identity.get("fixed_file_sha256")
        or sha256_file(selected_path) != identity.get("selected_file_sha256")
    ):
        raise ValueError(f"fixed epoch-5 checkpoint identity changed for {job_id}")
    fit_budget = fit.get("optimizer_budget", {})
    if (
        fit_budget.get("epochs") != 6
        or type(fit_budget.get("total")) is not int
        or fit_budget["total"] <= 0
        or fit.get("original_actor_phase", {}).get("status")
        not in {"actor_returned", "actor_failed"}
        or not fit.get("original_actor_phase", {}).get("fit_completion_verified_from")
    ):
        raise ValueError(f"original full-budget fit recovery evidence is incomplete for {job_id}")

    actor_phase = fit["original_actor_phase"]
    train_resource_path = expected_run / "train_cell_resource_receipt.json"
    train_resource = read_json(train_resource_path)
    if sha256_file(train_resource_path) != actor_phase.get("resource_receipt_sha256"):
        raise ValueError(f"fit resource receipt differs from actor evidence for {job_id}")
    placement = placement_verifier(root, job, train_resource)
    if not isinstance(placement, dict) or placement.get("authority") not in {
        "frozen_panel_initial_placement",
        "sealed_density4_launch_event",
    }:
        raise ValueError(f"fit placement authority is invalid for {job_id}")
    recorded_placement = actor_phase.get("placement_authorization")
    if placement["authority"] == "frozen_panel_initial_placement":
        if recorded_placement not in (None, placement):
            raise ValueError(f"initial fit placement receipt differs for {job_id}")
    elif recorded_placement != placement:
        raise ValueError(f"moved fit placement lacks exact sealed launch evidence for {job_id}")

    export_path = recovery / "receipts" / f"{job_id}.export.json"
    exported = read_json(export_path)
    output = recovery / "exports" / job_id
    receipt_path = output / "export_receipt.json"
    command_path = output / "export_command.json"
    resource_path = output / "export_cell_resource_receipt.json"
    log_path = recovery / "logs" / f"{job_id}.export.log"
    receipt = read_json(receipt_path)
    if (
        exported.get("schema") != "tabdata-r01-export-recovery.v2"
        or exported.get("status") != "exported_fixed_epoch5_no_refit"
        or exported.get("job_id") != job_id
        or exported.get("old_fit_outputs_modified") is not False
        or exported.get("fit_recovery_receipt_sha256") != sha256_file(fit_path)
        or exported.get("export_receipt_path") != str(receipt_path)
        or exported.get("export_receipt_sha256") != sha256_file(receipt_path)
        or exported.get("export_command_sha256") != sha256_file(command_path)
        or exported.get("export_resource_sha256") != sha256_file(resource_path)
        or exported.get("actor_log_sha256") != sha256_file(log_path)
        or receipt.get("status") != "completed_fixed_epoch5_feature_only_fp32"
        or receipt.get("run_name") != job_id
        or receipt.get("fold") != job["fold"]
        or receipt.get("seed") != job["seed"]
        or receipt.get("epoch") != 5
        or receipt.get("feature_only") is not True
        or receipt.get("inference_precision") != "fp32"
        or receipt.get("target_reads_during_export") is not False
        or receipt.get("checkpoint_sha256") != identity.get("fixed_file_sha256")
        or exported.get("checkpoint_sha256") != identity.get("fixed_file_sha256")
        or receipt.get("prediction_sha256") != exported.get("prediction_sha256")
        or receipt.get("sidecar_sha256") != exported.get("sidecar_sha256")
    ):
        raise ValueError(f"fixed-endpoint export receipt chain is invalid for {job_id}")
    resource = read_json(resource_path)
    if (
        resource.get("schema") != "tabdata-b300-process-envelope.v1"
        or resource.get("status") != "actor_returned"
        or resource.get("script") != "export_cell.py"
        or resource.get("visible_devices") != exported.get("visible_gpu_uuid")
        or "B300" not in str(resource.get("gpu", ""))
        or resource.get("allocator_budget_bytes") != 6 * 1024**3
        or resource.get("torch_threads") != 2
        or type(resource.get("cuda_peak_allocated_bytes")) is not int
        or type(resource.get("cuda_peak_reserved_bytes")) is not int
        or not 0 <= resource["cuda_peak_allocated_bytes"] <= 6 * 1024**3
        or not 0 <= resource["cuda_peak_reserved_bytes"] <= 6 * 1024**3
    ):
        raise ValueError(f"export resource receipt is invalid for {job_id}")
    prediction_path = output / "cache" / f"pred_{job_id}_valid.npy"
    sidecar_path = output / "cache" / f"pred_{job_id}_valid.json"
    if sha256_file(prediction_path) != receipt.get("prediction_sha256") or sha256_file(
        sidecar_path
    ) != receipt.get("sidecar_sha256"):
        raise ValueError(f"exported prediction or sidecar bytes changed for {job_id}")
    return {
        "job": job,
        "fit": fit,
        "fit_path": fit_path,
        "fit_sha256": sha256_file(fit_path),
        "identity": identity,
        "placement_authorization": placement,
        "export": exported,
        "export_path": export_path,
        "export_sha256": sha256_file(export_path),
        "export_receipt": receipt,
        "output": output,
        "prediction_path": prediction_path,
        "sidecar_path": sidecar_path,
    }


def _flag(argv: list[str], flag: str) -> str:
    """Return one unique value from a stored grader argv."""
    positions = [index for index, value in enumerate(argv[:-1]) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"stored grade command must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _finite_wp(value: object, label: str) -> float:
    """Require one finite official weighted-Pearson score."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"official {label} WP is missing or nonfinite")
    return float(value)


def _verify_grade_bindings(
    grade: dict[str, Any],
    fold: int,
    split_sha: str,
    groups: list[int],
    wide_job: dict[str, Any],
    deep_job: dict[str, Any],
    wide_binding: dict[str, Any],
    deep_binding: dict[str, Any],
) -> tuple[float, float]:
    """Require official score endpoints to match both fixed checkpoints and predictions."""
    if (
        grade.get("schema") != "priority25.endpoint_grade.v1"
        or grade.get("metric_id") != METRIC_ID
        or grade.get("fold") != fold
        or grade.get("split_manifest_sha256") != split_sha
        or grade.get("evaluation_row_groups") != groups
        or grade.get("endpoint", {}).get("run") != deep_job["id"]
        or grade.get("endpoint", {}).get("epoch") != 5
        or grade.get("reference", {}).get("run") != wide_job["id"]
        or grade.get("reference", {}).get("epoch") != 5
        or grade.get("paired_component_bootstrap", {}).get("resamples") != 2000
        or grade.get("paired_component_bootstrap", {}).get("seed") != 271828
    ):
        raise ValueError("official grade JSON does not bind the original metric and endpoints")
    for label, endpoint, binding in (
        ("wide", grade["reference"], wide_binding),
        ("deep", grade["endpoint"], deep_binding),
    ):
        if any(
            endpoint.get(field) != binding.get(binding_field)
            for field, binding_field in (
                ("checkpoint_sha256", "checkpoint_sha256"),
                ("prediction_sha256", "prediction_sha256"),
                ("sidecar_sha256", "sidecar_sha256"),
            )
        ):
            raise ValueError(f"official grade hashes do not match the {label} fit/export chain")
    return (
        _finite_wp(grade["endpoint"].get("score", {}).get("wp"), "deep"),
        _finite_wp(grade["reference"].get("score", {}).get("wp"), "wide"),
    )


def _verify_pair(
    root: Path,
    recovery: Path,
    fold: int,
    seed: int,
    jobs: dict[str, dict[str, Any]],
    grader: Any,
    placement_verifier: Callable[[Path, dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Verify official grade JSON against both recovered checkpoints and exports."""
    wide = _fit_export(root, recovery, jobs["wide"], placement_verifier)
    deep = _fit_export(root, recovery, jobs["deep"], placement_verifier)
    grade_root = recovery / "grades" / f"seed{seed}" / f"fold{fold}"
    receipt_path, grade_path, command_path = (
        grade_root / "grade_receipt.json",
        grade_root / "grade.json",
        grade_root / "grade_command.json",
    )
    receipt, grade, command = (
        read_json(receipt_path),
        read_json(grade_path),
        read_json(command_path),
    )
    split_path = root / "splits" / f"fold{fold}.json"
    split_sha = sha256_file(split_path)
    if (
        receipt.get("schema") != "tabdata-r01-delivery-grade.v2"
        or receipt.get("status") != "graded"
        or receipt.get("candidate") != jobs["deep"]["id"]
        or receipt.get("reference") != jobs["wide"]["id"]
        or receipt.get("fold") != fold
        or receipt.get("seed") != seed
        or receipt.get("metric_id") != METRIC_ID
        or receipt.get("grade_path") != str(grade_path)
        or receipt.get("grade_sha256") != sha256_file(grade_path)
        or receipt.get("command_sha256") != sha256_file(command_path)
        or receipt.get("stdout_sha256") != sha256_file(grade_root / "grade.stdout.txt")
        or receipt.get("stderr_sha256") != sha256_file(grade_root / "grade.stderr.txt")
        or isinstance(receipt.get("elapsed_seconds"), bool)
        or not isinstance(receipt.get("elapsed_seconds"), (int, float))
        or not math.isfinite(receipt.get("elapsed_seconds"))
        or receipt.get("elapsed_seconds") <= 0
    ):
        raise ValueError("recovery grade receipt schema, cell, or file hashes are invalid")
    if (
        command.get("schema") != "tabdata-r01-delivery-grade-command.v2"
        or command.get("candidate") != jobs["deep"]["id"]
        or command.get("reference") != jobs["wide"]["id"]
        or command.get("fold") != fold
        or command.get("seed") != seed
        or command.get("epoch") != 5
        or command.get("grader_sha256") != GRADER_SHA256
        or command.get("wide_fit_recovery_sha256") != wide["fit_sha256"]
        or command.get("deep_fit_recovery_sha256") != deep["fit_sha256"]
        or command.get("wide_export_receipt_sha256")
        != sha256_file(wide["output"] / "export_receipt.json")
        or command.get("deep_export_receipt_sha256")
        != sha256_file(deep["output"] / "export_receipt.json")
        or command.get("split_sha256") != split_sha
        or command.get("cuda_disabled") is not True
    ):
        raise ValueError("stored grade command is not bound to the exact recovered pair")
    argv = command.get("argv")
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise ValueError("stored grade argv is malformed")
    expected_flags = {
        "--root": str(grade_root / "view"),
        "--run": jobs["deep"]["id"],
        "--split": str(split_path),
        "--epoch": "5",
        "--reference": jobs["wide"]["id"] + "@5",
        "--output": str(grade_path),
        "--bootstrap": "2000",
        "--scratch-dir": str(grade_root / "scratch"),
    }
    if argv[1] != str(root / "runtime/b300-r01-v1/grader/grade_endpoint.py") or any(
        _flag(argv, flag) != value for flag, value in expected_flags.items()
    ):
        raise ValueError("stored grade argv differs from the unchanged original grader call")
    view = grade_root / "view"
    split = grader.load_campaign_split(split_path, view)
    groups = list(split.evaluation_groups)
    if (
        split.fold != fold
        or split.sha256 != split_sha
        or grade.get("evaluation_row_groups") != groups
    ):
        raise ValueError("official grade evaluation membership differs from the frozen fold")
    predictions = {}
    for arm, item in (("wide", wide), ("deep", deep)):
        _prediction, binding = grader.validate_prediction(view, item["job"]["id"], 5, groups, split)
        if (
            binding.get("checkpoint_sha256") != item["identity"].get("fixed_file_sha256")
            or binding.get("prediction_sha256") != item["export_receipt"].get("prediction_sha256")
            or binding.get("sidecar_sha256") != item["export_receipt"].get("sidecar_sha256")
        ):
            raise ValueError(f"original grader prediction validator rejected {arm} endpoint")
        predictions[arm] = binding
    deep_wp, wide_wp = _verify_grade_bindings(
        grade,
        fold,
        split_sha,
        groups,
        jobs["wide"],
        jobs["deep"],
        predictions["wide"],
        predictions["deep"],
    )
    receipt_score = receipt.get("score", {})
    if (
        _finite_wp(receipt_score.get("deep_wp"), "recovery deep") != deep_wp
        or _finite_wp(receipt_score.get("wide_wp"), "recovery wide") != wide_wp
    ):
        raise ValueError("recovery grade receipt scores differ from the official grade JSON")
    pair_id = f"r01-fold{fold}-seed{seed}"
    endpoints_report = {}
    for arm, item, score in (("wide", wide, wide_wp), ("deep", deep, deep_wp)):
        job_id = item["job"]["id"]
        run = root / "runs" / job_id
        export_output = item["output"]
        evidence_paths = [
            root / "splits" / f"fold{fold}.json",
            item["fit_path"],
            recovery / "receipts" / f"{job_id}.export.json",
            item["output"] / "export_receipt.json",
            item["output"] / "export_command.json",
            item["output"] / "export_cell_resource_receipt.json",
            recovery / "logs" / f"{job_id}.export.log",
            run / "command.json",
            run / "adapter_preflight.json",
            run / "producer_receipt.json",
            run / "train_cell_resource_receipt.json",
            run / "artifacts/runs" / job_id / "history.json",
            run / "artifacts/runs" / job_id / "provenance.json",
            export_output / "cache" / f"pred_{job_id}_valid.json",
            grade_root / "grade_receipt.json",
            grade_path,
            command_path,
            grade_root / "grade.stdout.txt",
            grade_root / "grade.stderr.txt",
            recovery / "fit_views" / job_id / "producer_receipt.json",
        ]
        safe_evidence = [
            str(path.resolve(strict=True))
            for path in evidence_paths
            if not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root)
        ]
        artifact = run / "artifacts/runs" / job_id
        weights = [artifact / name for name in ("model_ep5.pt", "model.pt", "training_state.pt")]
        safe_weights = [
            str(path.resolve(strict=True))
            for path in weights
            if path.is_file() and not path.is_symlink()
        ]
        endpoints_report[arm] = {
            "id": job_id,
            "arm": arm,
            "wp": score,
            "checkpoint_sha256": item["identity"]["fixed_file_sha256"],
            "prediction_sha256": predictions[arm]["prediction_sha256"],
            "sidecar_sha256": predictions[arm]["sidecar_sha256"],
            "fit_recovery_sha256": item["fit_sha256"],
            "export_recovery_sha256": item["export_sha256"],
            "evidence_paths": sorted(set(safe_evidence)),
            "weights_paths": safe_weights,
        }
    return {
        "id": pair_id,
        "pair_id": pair_id,
        "fold": fold,
        "seed": seed,
        "metric_id": METRIC_ID,
        "split_manifest_sha256": split_sha,
        "wide_wp": wide_wp,
        "deep_wp": deep_wp,
        "delta_wp": deep_wp - wide_wp,
        "receipt_sha256": sha256_file(receipt_path),
        "grade_sha256": sha256_file(grade_path),
        "grade_command_sha256": sha256_file(command_path),
        "endpoints": endpoints_report,
    }


def collect(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read and independently validate completed recovery grades without running phases."""
    root = root.resolve(strict=True)
    panel, source_root, placement_verifier = _verify_runtime(root, config)
    grader = _load_grader(root, source_root)
    pairs = _job_map(panel)
    recovery = root / "state/r01-delivery-recovery-v2"
    completed, errors, active = [], [], set()
    for (fold, seed), jobs in sorted(pairs.items()):
        grade_root = recovery / "grades" / f"seed{seed}" / f"fold{fold}"
        receipt_path = grade_root / "grade_receipt.json"
        if not receipt_path.exists() and not receipt_path.is_symlink():
            if any(
                (recovery / "receipts" / f"{job['id']}.export.json").exists()
                for job in jobs.values()
            ):
                active.add((fold, seed))
            continue
        try:
            completed.append(
                _verify_pair(root, recovery, fold, seed, jobs, grader, placement_verifier)
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(f"fold{fold}/seed{seed}: {type(error).__name__}: {error}")
    complete_cells = {(row["fold"], row["seed"]) for row in completed}
    failed_cells = {
        (int(m[1]), int(m[2]))
        for error in errors
        if (m := re.search(r"fold(\d+)/seed(\d+)", error))
    }
    active -= complete_cells | failed_cells
    source = {
        "panel_sha256": PANEL_SHA256,
        "actor_bundle_sha256": ACTOR_SHA256,
        "source_tree_identity_sha256": SOURCE_TREE_SHA256,
        "grader_sha256": GRADER_SHA256,
        "helper_manifest_sha256": ACTIVE_HELPER_MANIFEST_SHA256,
        "historical_helper_manifest_sha256": HELPER_MANIFEST_SHA256,
    }
    return {
        "schema": "tabdata-r01-delivery-recovery-report.v2",
        "experiment_id": EXPERIMENT_ID,
        "expected_cells": 30,
        "expected_per_arm": 30,
        "completed": sorted(completed, key=lambda row: (row["fold"], row["seed"])),
        "active": len(active),
        "pending": max(0, 30 - len(complete_cells) - len(active) - len(failed_cells)),
        "terminal": len(complete_cells) == 30 and not errors,
        "source": source,
        "metric_id": METRIC_ID,
        "errors": errors[:100],
        "contract_note": (
            "Distinct delivery-recovery scope for already completed original E-B300-R01 fits. "
            "No refits were performed by this reporter. This source-qualified recovery status "
            "does not replace or edit the existing full-panel R01 status comment on PR25."
        ),
    }


def _canonical_sha(value: object) -> str:
    """Hash canonical JSON for durable comment markers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Durably publish one immutable JSON outbox item without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"outbox item already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".recovery-outbox-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _marker(kind: str, digest: str = "") -> str:
    """Create markers in a namespace separate from the full-panel R01 publisher."""
    return f"<!-- tabdata-b300-20261018:{EXPERIMENT_ID}:{kind}:{digest} -->"


def render(
    snapshot: dict[str, Any], *, rows: list[dict[str, Any]] | None, evidence_url: str
) -> str:
    """Render a compact, source-qualified status or paired-results comment."""
    kind = "status" if rows is None else "results"
    marker = _marker(kind, "" if rows is None else _canonical_sha(rows))
    by_fold: dict[int, list[float]] = defaultdict(list)
    for row in snapshot["completed"]:
        by_fold[row["fold"]].append(row["delta_wp"])
    delta = (
        None
        if not by_fold
        else sum(sum(values) / len(values) for values in by_fold.values()) / len(by_fold)
    )
    label = (
        "complete delivery evidence"
        if snapshot["terminal"]
        else "partial recovery delivery evidence"
    )
    lines = [
        marker,
        f"### {EXPERIMENT_ID}: {kind} update",
        f"**Status:** {label}; paired official grades {len(snapshot['completed'])}/30; delivery-pending {snapshot['pending']}; in-progress {snapshot['active']}.",
        snapshot["contract_note"],
        "This report uses a separate recovery marker; it does not replace or edit the existing full-panel R01 status comment on PR25. It records delivery-only recovery of already trained fixed-epoch-5 checkpoints and does not count additional fits.",
        f"Official metric `{METRIC_ID}`; equal-fold mean of observed paired deep-minus-wide deltas: `{('not available' if delta is None else f'{delta:+.8f}')}`. All four folds are exposed; this is descriptive evidence, not a pristine holdout or a significance claim.",
        f"Source pins: `{json.dumps(snapshot['source'], sort_keys=True)}`.",
        f"Contract and recovery evidence: {evidence_url}",
    ]
    if rows is not None:
        lines.append(f"**New scored original pairs in this comment:** {len(rows)}.")
        for row in rows:
            lines.append(
                _marker("pair", _canonical_sha({"id": row["id"], "receipt": row["receipt_sha256"]}))
            )
            lines.append(
                f"- Fold {row['fold']}, seed {row['seed']}: wide `{row['wide_wp']:.8f}`, "
                f"deep `{row['deep_wp']:.8f}`, delta `{row['delta_wp']:+.8f}`; "
                f"grade receipt `{row['receipt_sha256']}`, grade `{row['grade_sha256']}`."
            )
            for arm in ("wide", "deep"):
                endpoint = row["endpoints"][arm]
                lines.append(
                    f"  - {arm} `{endpoint['id']}` checkpoint `{endpoint['checkpoint_sha256']}`, "
                    f"prediction `{endpoint['prediction_sha256']}`, fit recovery `{endpoint['fit_recovery_sha256']}`, "
                    f"export recovery `{endpoint['export_recovery_sha256']}`."
                )
                archive = endpoint.get("archives", {})
                if archive:
                    evidence = archive.get("evidence_archive", {})
                    weights = archive.get("weights_archive", {})
                    lines.append(
                        f"    - Private archive manifest `{archive.get('archive_manifest_path')}` "
                        f"SHA-256 `{archive.get('archive_manifest_sha256')}`; evidence "
                        f"`{evidence.get('path')}` SHA-256 `{evidence.get('sha256')}`; weights "
                        f"`{weights.get('path')}` SHA-256 `{weights.get('sha256')}`."
                    )
    if snapshot["errors"]:
        lines.extend(
            [
                "**Rejected or incomplete receipt chains retained:**",
                *[f"- {error[:500]}" for error in snapshot["errors"][:12]],
            ]
        )
    lines.append(
        "The reporting process reads receipts and archives evidence/weights privately; it never trains, exports, or grades."
    )
    return "\n\n".join(lines)


def _load_github(code_dir: Path, expected_manifest_sha: str) -> dict[str, Any]:
    """Load pinned GitHub/archive APIs into this isolated one-shot reporter process."""
    manifest_path = code_dir / "CODE_MANIFEST.json"
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("github-reporting-v1 helper manifest differs from its pin")
    manifest = read_json(manifest_path)
    for name, digest in manifest.get("files", {}).items():
        if (
            Path(name).name != name
            or not name.endswith(".py")
            or sha256_file(code_dir / name) != digest
        ):
            raise ValueError(f"pinned GitHub reporting helper changed: {name}")
    names = ("github_api", "package_evidence", "render_report", "report_state")
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(code_dir))
    import importlib

    return {name: importlib.import_module(name) for name in names}


def _own_comments(comments: list[dict[str, Any]], login: str) -> list[dict[str, Any]]:
    """Only reconcile comments authored by the configured account."""
    return [comment for comment in comments if comment.get("user", {}).get("login") == login]


def _publish(
    client: Any,
    helpers: dict[str, Any],
    root: Path,
    snapshot: dict[str, Any],
    config: dict[str, Any],
    cursor: dict[str, Any],
) -> None:
    """Reconcile and publish up to eight new pairs plus this recovery scope's status."""
    metadata = client.request("GET", f"/repos/{REPOSITORY}/pulls/{PR_NUMBER}")
    if (
        metadata["state"] != "open"
        or metadata["base"]["ref"] != "research-team"
        or metadata["head"]["ref"] != HEAD_REF
        or metadata["base"]["repo"]["full_name"] != REPOSITORY
        or metadata["head"]["repo"]["full_name"] != REPOSITORY
    ):
        raise ValueError("PR25 identity is not the fixed R01 draft target")
    comments = _own_comments(client.comments(PR_NUMBER), config["publisher_login"])
    acknowledged = cursor.setdefault("pairs", {})
    for comment in comments:
        body = comment.get("body", "")
        for row in snapshot["completed"]:
            pair_marker = _marker(
                "pair", _canonical_sha({"id": row["id"], "receipt": row["receipt_sha256"]})
            )
            if pair_marker in body:
                acknowledged[row["id"]] = row["receipt_sha256"]
    pending = [
        row for row in snapshot["completed"] if acknowledged.get(row["id"]) != row["receipt_sha256"]
    ]
    evidence_url = config["evidence_url"]
    for row in pending[:8]:
        archived = {}
        for arm in ("wide", "deep"):
            endpoint = row["endpoints"][arm]
            archive_row = {
                "id": endpoint["id"],
                "receipt_sha256": row["receipt_sha256"],
                "evidence_paths": _archive_evidence_paths(
                    root,
                    row["fold"],
                    row["split_manifest_sha256"],
                    endpoint["evidence_paths"],
                ),
                "weights_paths": endpoint["weights_paths"],
            }
            archived[arm] = helpers["package_evidence"].ensure_archives(root, archive_row)
            endpoint["archives"] = archived[arm]
        row["archives"] = archived
    new_rows = pending[:8]
    if new_rows:
        batch_marker = _marker("results", _canonical_sha(new_rows))
        existing = next(
            (comment for comment in comments if batch_marker in comment.get("body", "")), None
        )
        if existing is None:
            existing = client.post(
                PR_NUMBER, render(snapshot, rows=new_rows, evidence_url=evidence_url)
            )
        for row in new_rows:
            acknowledged[row["id"]] = row["receipt_sha256"]
        cursor.setdefault("result_urls", [])
        if existing.get("html_url") not in cursor["result_urls"]:
            cursor["result_urls"].append(existing["html_url"])
    status_marker = _marker("status")
    status = next(
        (comment for comment in comments if status_marker in comment.get("body", "")), None
    )
    body = render(snapshot, rows=None, evidence_url=evidence_url)
    status = client.post(PR_NUMBER, body) if status is None else client.patch(status["id"], body)
    cursor["status_url"] = status["html_url"]
    cursor["snapshot_sha256"] = _canonical_sha(snapshot)
    cursor["last_success_unix"] = time.time()


def run(root: Path, config: dict[str, Any]) -> int:
    """Collect receipts, persist an outbox, and publish only to the fixed PR25 scope."""
    import fcntl

    root = root.resolve(strict=True)
    if config.get("root") != str(root) or config.get("repository") != REPOSITORY:
        raise ValueError("reporter root or repository differs from the fixed scope")
    github_code_dir = root / "runtime/github-reporting-v1/code"
    if (
        str(root) != PROJECT_ROOT
        or Path(config["github_code_dir"]) != github_code_dir
        or config.get("github_code_manifest_sha256") != GITHUB_CODE_MANIFEST_SHA256
        or Path(config["token_file"]) != root / "private/github-reporting-v1/github.token"
        or type(config.get("execute_enabled")) is not bool
    ):
        raise ValueError(
            "publisher helper, credential path, or host scope differs from its fixed pin"
        )
    recovery_root = root / "state/r01-delivery-recovery-v2"
    if (
        config.get("pr_number") != PR_NUMBER
        or config.get("head_ref") != HEAD_REF
        or config.get("publisher_login") != PUBLISHER_LOGIN
        or Path(config["helper_manifest_path"])
        != root / "runtime/r01-delivery-recovery-v2_4/MANIFEST.json"
        or config.get("helper_manifest_sha256") != ACTIVE_HELPER_MANIFEST_SHA256
    ):
        raise ValueError("reporter target or active v2.4 helper pin differs from the fixed scope")
    state_root = root / "state/r01-delivery-recovery-v2/publisher-v1"
    if recovery_root.is_symlink() or state_root.is_symlink():
        raise ValueError("recovery reporter state root cannot be a symlink")
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / "publisher.lock"
    if lock_path.is_symlink():
        raise ValueError("recovery reporter lock cannot be a symlink")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        snapshot = collect(root, config)
        outbox_dir = state_root / "outbox"
        if outbox_dir.is_symlink():
            raise ValueError("recovery reporter outbox cannot be a symlink")
        outbox_dir.mkdir(exist_ok=True)
        outbox_path = outbox_dir / f"{_canonical_sha(snapshot)}.json"
        if not outbox_path.exists() and not outbox_path.is_symlink():
            _write_new_json(outbox_path, snapshot)
        elif read_json(outbox_path) != snapshot:
            raise ValueError("existing recovery outbox differs from the current snapshot")
        if not config.get("execute_enabled"):
            return 0
        if time.time() > config.get("expires_unix", 0):
            raise ValueError("recovery reporter configuration has expired")
        helpers = _load_github(
            Path(config["github_code_dir"]), config["github_code_manifest_sha256"]
        )
        client = helpers["github_api"].GitHub(
            helpers["github_api"].read_token(Path(config["token_file"])), request_limit=24
        )
        user = client.request("GET", "/user")
        if user.get("login") != PUBLISHER_LOGIN:
            raise ValueError("GitHub account differs from the configured publisher")
        cursor_path = state_root / "CURSOR.json"
        cursor = helpers["report_state"].read_json(cursor_path, {"pairs": {}, "result_urls": []})
        _publish(client, helpers, root, snapshot, config, cursor)
        helpers["report_state"].write_json(cursor_path, cursor)
        helpers["report_state"].append_event(
            state_root / "publications.jsonl",
            "published",
            experiment_id=EXPERIMENT_ID,
            completed=len(snapshot["completed"]),
            outbox_sha256=_canonical_sha(snapshot),
        )
        return 0


def verify_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Require the service-pinned config for the fixed recovery reporter."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("R01 recovery reporter configuration differs from its service pin")
    config = read_json(path)
    if (
        config.get("schema") != "tabdata-r01-recovery-reporter-config.v1"
        or config.get("repository") != REPOSITORY
        or config.get("pr_number") != PR_NUMBER
        or config.get("head_ref") != HEAD_REF
        or config.get("publisher_login") != PUBLISHER_LOGIN
        or config.get("helper_manifest_sha256") != ACTIVE_HELPER_MANIFEST_SHA256
        or config.get("github_code_manifest_sha256") != GITHUB_CODE_MANIFEST_SHA256
        or config.get("root") != PROJECT_ROOT
    ):
        raise ValueError("R01 recovery reporter config is outside its fixed source-qualified scope")
    return config


def main() -> int:
    """Run one bounded receipt collection and publication pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    config = verify_config(args.config, args.config_sha256)
    return run(Path(config["root"]), config)


if __name__ == "__main__":
    raise SystemExit(main())
