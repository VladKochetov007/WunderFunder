from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PLAN_PATH = Path(__file__).with_name("PAIR_PREPARATION_R1.json")


class BindingError(ValueError):
    """Raised when a frozen input differs from its recorded identity."""


def sha256_file(path: Path) -> str:
    """Hash a small control/receipt file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_file_bindings(value: object) -> Iterator[dict[str, object]]:
    """Yield path/size/hash objects from the preparation manifest."""
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from iter_file_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_file_bindings(child)


def resolve_bound_path(project_root: Path, raw_path: str) -> Path:
    """Resolve a manifest path while refusing traversal outside project root."""
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    root = project_root.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise BindingError(f"bound path escapes project root: {raw_path}")
    return resolved


def verify_manifest_bindings(plan: dict[str, object], project_root: Path) -> list[str]:
    """Verify small frozen source, checkpoint, provenance and receipt files."""
    verified: list[str] = []
    seen: set[tuple[str, str]] = set()
    immutable = plan["immutable_bindings"]
    assert isinstance(immutable, dict)
    for binding in iter_file_bindings(immutable):
        path_text = str(binding["path"])
        expected_sha = str(binding["sha256"])
        key = (path_text, expected_sha)
        if key in seen:
            continue
        seen.add(key)
        path = resolve_bound_path(project_root, path_text)
        if path.stat().st_size != binding["bytes"]:
            raise BindingError(f"byte count differs: {path_text}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise BindingError(f"SHA256 differs: {path_text}")
        verified.append(path_text)

    parents = plan["parents"]
    assert isinstance(parents, list)
    for parent in parents:
        assert isinstance(parent, dict)
        if parent["checkpoint"]["sha256"] != parent["expected_checkpoint_sha256"]:
            raise BindingError(f"parent checkpoint binding differs: {parent['run']}")
        if parent["provenance"]["sha256"] != parent["expected_provenance_sha256"]:
            raise BindingError(f"parent provenance binding differs: {parent['run']}")
    return verified


def verify_command_derivation(plan: dict[str, object], recipe: dict[str, object]) -> None:
    """Ensure future fit/export vectors are exact copies except unique run IDs."""
    prefixes = {
        "control": "clean_t3_f0_combo_parent_s3_a9_t3r5_pair_r6",
        "treatment": "clean_t3_f0_muonall_parent_s3_a9_t3r5_pair_r6",
    }
    source_stages = recipe["stages"]
    target_stages = plan["fit_stages"]
    assert isinstance(source_stages, list) and isinstance(target_stages, list)
    if len(source_stages) != 6 or len(target_stages) != 6:
        raise BindingError("expected exactly six staged fit command vectors")
    names = {
        source["name"]: f"{prefixes[source['arm']]}_p{source['part']}"
        for source in source_stages
    }
    for source, target in zip(source_stages, target_stages, strict=True):
        assert isinstance(source, dict) and isinstance(target, dict)
        expected = []
        for item in source["command_argv"]:
            for old, new in names.items():
                item = item.replace(old, new)
            expected.append(item)
        if expected != target["argv_template"]:
            raise BindingError(f"fit argv drift in {source['arm']} stage {source['part']}")

    source_exports = recipe["final_student_exports"]
    target_exports = plan["prediction_exports"]
    assert isinstance(source_exports, list) and isinstance(target_exports, list)
    if len(source_exports) != 2 or len(target_exports) != 2:
        raise BindingError("expected exactly two full-valid prediction commands")
    for source, target in zip(source_exports, target_exports, strict=True):
        assert isinstance(source, dict) and isinstance(target, dict)
        run = prefixes[source["arm"]]
        expected = [item.replace(source["run"], run) for item in source["command_argv"]]
        if expected != target["argv_template"]:
            raise BindingError(f"export argv drift for {source['arm']}")

    profile = plan["profile"]
    assert isinstance(profile, dict)
    argv = profile["argv_template"]
    if not isinstance(argv, list):
        raise BindingError("profile command is not an argv list")
    def value_for(option: str) -> str | None:
        try:
            return str(argv[argv.index(option) + 1])
        except (ValueError, IndexError):
            return None

    if value_for("--profile-warmup") != "5" or value_for("--profile-steps") != "25":
        raise BindingError("profile window differs from its preparation contract")
    if value_for("--max-batches") != "30":
        raise BindingError("profile must bound itself to 30 complete train batches")
    if "--save-training-state" in argv or "--stop-after-epoch" in argv:
        raise BindingError("disposable profile must not create a resumable fit state")


def inspect_pending_readiness(plan: dict[str, object], project_root: Path) -> dict[str, object]:
    """Verify known producer bindings and return blockers without importing ML code."""
    if "torch" in sys.modules:
        raise BindingError("this static readiness check must run before torch import")
    verified = verify_manifest_bindings(plan, project_root)
    recipe_pin = plan["immutable_bindings"]["base_recipe"]
    recipe_path = resolve_bound_path(project_root, str(recipe_pin["path"]))
    recipe = json.loads(recipe_path.read_text())
    verify_command_derivation(plan, recipe)

    blockers: list[str] = []
    observations: list[dict[str, object]] = []
    target = plan["teacher_target"]
    assert isinstance(target, dict)
    for teacher in target["teachers"]:
        assert isinstance(teacher, dict)
        output = teacher["fit_train_prediction"]
        assert isinstance(output, dict)
        array_path = resolve_bound_path(project_root, str(output["path"]))
        sidecar = resolve_bound_path(project_root, str(output["sidecar_path"]))
        present = array_path.is_file()
        sidecar_present = sidecar.is_file()
        expected_bytes = int(output["bytes"])
        actual_bytes = array_path.stat().st_size if present else None
        sidecar_sha = sha256_file(sidecar) if sidecar_present else None
        expected_sidecar_sha = output.get("sidecar_sha256")
        observations.append({
            "seed": teacher["seed"],
            "run": teacher["run"],
            "array_path": str(array_path),
            "array_present": present,
            "array_bytes": actual_bytes,
            "expected_array_bytes": expected_bytes,
            "sidecar_present": sidecar_present,
            "sidecar_sha256": sidecar_sha,
            "sidecar_hash_matches_preparation": expected_sidecar_sha is not None and sidecar_sha == expected_sidecar_sha,
            "array_full_hash_recomputed": False,
            "producer_status_at_preparation": output.get("status", output.get("status_at_capture", "PENDING")),
        })
        if not present or actual_bytes != expected_bytes or not sidecar_present:
            blockers.append(f"teacher seed {teacher['seed']} FIT_TRAIN export is not complete at the expected path/size")
        if expected_sidecar_sha is not None and sidecar_sha != expected_sidecar_sha:
            raise BindingError(f"teacher seed {teacher['seed']} FIT_TRAIN sidecar changed")
        if not output.get("array_sha256") or not output.get("sidecar_sha256") or not output.get("ordered_row_groups_sha256"):
            blockers.append(f"teacher seed {teacher['seed']} lacks a terminal hash/order binding")

    cache_root = project_root / str(target["cache_root"])
    if cache_root.exists():
        blockers.append("student cache output path already exists; require a reviewed receipt or a new identified cache root")
    else:
        blockers.append("complete homogeneous student cache has not been materialized")
    if target.get("materialization_receipt") is None:
        blockers.append("cache materialization receipt is not bound")
    if target.get("root_cache_release") is None:
        blockers.append("root cache materialization release is not bound")
    delivery = plan["kaggle_delivery"]
    assert isinstance(delivery, dict)
    for key in ("teacher_cache_dataset_id_version", "source_dataset_id_version", "ready_receipt", "kernel_slug"):
        if delivery.get(key) is None:
            blockers.append(f"Kaggle delivery binding missing: {key}")
    if not plan["authorization"]["gpu_profile"]:
        blockers.append("no root authorization for the required actual-device profile")
    if not plan["authorization"]["student_fit"]:
        blockers.append("no root authorization for either paired six-epoch fit")

    return {
        "schema": "wnn.clean-t3-f0-matched-student-pair-pending-preflight.r1",
        "status": "BLOCKED_PENDING_CACHE_REVIEW_PROFILE_AND_ROOT_RELEASE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_id": plan["preparation_id"],
        "preparation_manifest_sha256": sha256_file(PLAN_PATH),
        "project_root": str(project_root.resolve(strict=True)),
        "static_file_bindings_verified": len(verified),
        "static_binding_paths": verified,
        "teacher_fit_train_observations": observations,
        "blockers": blockers,
        "launch_allowed": False,
        "cache_materialized": False,
        "student_fit_started": False,
        "teacher_predictions_fully_rehashed_by_this_preflight": False,
        "torch_imported": False,
        "provider_action": False,
        "device_action": False,
    }


def main() -> int:
    """Run a read-only pending-dependency check and optionally write a receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/home/vlad/development/wunder/wnn_connectome_starterpack"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    result = inspect_pending_readiness(plan, args.project_root.resolve(strict=True))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
