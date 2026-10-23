"""Frozen artifact, manifest, and terminal-acceptance checks for the F1 grade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIT_NAME = "r01_h512_f1_s1004_kaggle_fit_r1"
SWA_NAME = "r01_h512_f1_s1004_kaggle_swa345_r1"
ORIGINAL_SHA = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
DERIVED_SHA = "548b42f06710a0f7c06377b8f2aaf7241469d75f0dad9de05bd4fbc29448d0c3"


def sha256(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other root values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def pinned(root: Path, item: dict[str, str]) -> Path:
    """Resolve a repository artifact and verify its frozen SHA-256."""
    path = (root / item["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or sha256(path) != item["sha256"]:
        raise ValueError(f"Frozen artifact mismatch: {path}")
    return path


def artifact_root(root: Path, value: str) -> Path:
    """Resolve accepted worker outputs under project or physical research scratch."""
    base = Path(value).expanduser()
    if not base.is_absolute():
        base = root / base
    base = base.resolve(strict=True)
    scratch = Path("/home/vlad/development/wunder/research_tmp").resolve(strict=True)
    if not (base.is_relative_to(root) or base.is_relative_to(scratch)):
        raise ValueError("Accepted artifact_root must be project or physical research scratch")
    return base


def accepted_file(base: Path, relative: str, expected: str) -> Path:
    """Resolve an accepted output below its declared artifact root and hash it."""
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Accepted artifact paths must be relative to artifact_root")
    resolved = (base / path).resolve(strict=True)
    if not resolved.is_relative_to(base) or sha256(resolved) != expected:
        raise ValueError(f"Accepted artifact bytes differ: {relative}")
    return resolved


def check_acceptance(root: Path, plan: dict[str, Any]) -> tuple[dict, dict, Path]:
    """Require the separate root terminal acceptance and matching worker evidence."""
    accept_spec = plan["acceptance"]
    acceptance_path = (root / accept_spec["path"]).resolve(strict=True)
    if not acceptance_path.is_relative_to(root):
        raise ValueError("Root acceptance path escaped the project")
    acceptance = read_json(acceptance_path)
    if (
        acceptance.get("schema") != accept_spec["schema"]
        or acceptance.get("status") != accept_spec["status"]
        or acceptance.get("fit_run") != FIT_NAME
        or acceptance.get("original_split_manifest_sha256") != ORIGINAL_SHA
        or acceptance.get("execution_split_manifest_sha256") != DERIVED_SHA
        or acceptance.get("worker_receipt_path") != accept_spec["worker_receipt_path"]
    ):
        raise ValueError("Root terminal acceptance differs from the frozen F1 contract")
    worker_path = pinned(
        root,
        {"path": acceptance["worker_receipt_path"], "sha256": acceptance["worker_receipt_sha256"]},
    )
    worker = read_json(worker_path)
    if (
        worker.get("schema") != "wnn.kaggle-h512-f1-conditional-worker-receipt.v1"
        or worker.get("status") != "CONDITIONAL_FIT_AND_EXPORT_COMPLETE"
        or worker.get("quality_fit_claim") is not None
        or worker.get("original_manifest_sha256") != ORIGINAL_SHA
        or worker.get("derived_manifest_sha256") != DERIVED_SHA
        or worker.get("run_names") != {"cp5": FIT_NAME, "swa345": SWA_NAME}
    ):
        raise ValueError("Worker receipt is incomplete or bound to another F1 run")
    if acceptance.get("artifact_root") is None:
        raise ValueError("Root acceptance must declare artifact_root for downloaded files")
    base = artifact_root(root, acceptance["artifact_root"])
    receipt = worker.get("input_identity", {})
    if (
        receipt.get("original_manifest_sha256") != ORIGINAL_SHA
        or receipt.get("derived_manifest_sha256") != DERIVED_SHA
        or receipt.get("proposal_sha256") != plan["identity"]["proposal"]["sha256"]
    ):
        raise ValueError("Worker source identity receipt differs from the frozen F1 plan")
    return acceptance, worker, base


def check_manifests(root: Path, plan: dict[str, Any], worker: dict) -> tuple[dict, dict]:
    """Preserve canonical identity while checking the source-only execution view."""
    canonical = read_json(pinned(root, plan["identity"]["canonical_manifest"]))
    derived = read_json(pinned(root, plan["identity"]["derived_execution_manifest"]))
    registry = read_json(pinned(root, plan["identity"]["portable_registry"]))
    proposal = read_json(pinned(root, plan["identity"]["proposal"]))
    if {**canonical, "sources": derived.get("sources")} != derived:
        raise ValueError("Execution manifest changed a canonical field other than sources")
    wanted_sources = {
        key: canonical["sources"][key]
        for key in ("datasets/train.parquet", "datasets/valid.parquet")
    }
    if derived.get("sources") != wanted_sources:
        raise ValueError("Execution view source metadata differs from canonical F1")
    if registry.get("split_manifest_sha256") != ORIGINAL_SHA:
        raise ValueError("Portable data registry is not bound to canonical F1")
    counts = plan["identity"]["membership"]
    if (
        derived.get("fold") != 1
        or derived.get("n_folds") != 4
        or derived.get("counts", {}).get("fit_train") != counts["fit_train_groups"]
        or derived.get("counts", {}).get("fit_valid") != counts["fit_valid_groups"]
        or derived.get("counts", {}).get("evaluation_valid") != counts["evaluation_groups"]
        or derived.get("counts", {}).get("evaluation_components") != counts["evaluation_components"]
    ):
        raise ValueError("Derived F1 membership counts differ from the frozen screen")
    receipt = worker["input_identity"]
    if receipt.get("registry_sha256") != plan["identity"]["portable_registry"]["sha256"]:
        raise ValueError("Worker portable registry hash differs from the frozen F1 registry")
    sources = registry.get("sources", {})
    observed_sources = receipt.get("raw_sources", {})
    expected_hashes = {
        "datasets/train.parquet": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "datasets/valid.parquet": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    }
    for name, expected in expected_hashes.items():
        registered = sources.get(name, {})
        observed = observed_sources.get(name, {})
        if registered.get("sha256") != expected or observed.get("sha256") != expected:
            raise ValueError(f"Raw source ancestry changed: {name}")
        if observed.get("bytes") != registered.get("original_identity", {}).get("bytes"):
            raise ValueError(f"Raw source size changed: {name}")
    if (
        proposal.get("fold_identity", {}).get("derived_execution_view", {}).get("sha256")
        != DERIVED_SHA
    ):
        raise ValueError("Conditional proposal does not name the frozen execution view")
    return canonical, derived
