"""Source, profile, checkpoint, export and score validation for PR15 host runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from checkpoint_identity import (
    validate_provenance_file_hashes,
    verify_checkpoint_pair,
)

SOURCE_MANIFEST_SHA256 = "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0"
SOURCE_COMMIT = "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e"
SCORE_SUPPORT_MANIFEST_SHA256 = "698ae26944b96ffb1b2b2ccb2de11cab427d3c822c2ca74cdacff1b1ef7d6368"
SCORE_SUPPORT_BLOB_OID = "7c8450d40bdf3e355ced2cddeb2b846598ce1554"
SCORE_SUPPORT_SHA256 = "fb7a85a5837f61fa6c65cd10a799768006067e87d4c18e92d4b813ba83b9d82e"
PROFILE_PREFIXES = {
    "PROFILE ": "timing",
    "PROFILE_RESOURCE_EVIDENCE ": "resources",
    "PROFILE_UPDATE_EVIDENCE ": "updates",
}


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read an object-valued UTF-8 JSON file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def verify_source_tree(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify all 172 extracted source files against the exact PR15 manifest."""
    if sha256_file(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("PR15 source manifest differs from the reviewed immutable manifest")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "b300-source-package.v1"
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("file_count") != 172
        or not isinstance(manifest.get("files"), list)
    ):
        raise ValueError("unsupported or mismatched PR15 source manifest")
    expected: dict[str, dict[str, Any]] = {}
    for record in manifest["files"]:
        relative = PurePosixPath(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe path in PR15 source manifest")
        expected[relative.as_posix()] = record
    if len(expected) != 172:
        raise ValueError("PR15 source manifest has duplicate or missing file entries")
    actual: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"PR15 source tree contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(source_root).as_posix())
    if actual != set(expected):
        raise ValueError(
            f"PR15 extracted file set differs (missing={sorted(set(expected) - actual)[:8]}, "
            f"extra={sorted(actual - set(expected))[:8]})"
        )
    for relative, record in expected.items():
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"PR15 frozen source file changed: {relative}")
    return {
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": manifest["archive_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "file_count": len(expected),
        "total_uncompressed_bytes": manifest["total_uncompressed_bytes"],
    }


def verify_score_support(package_root: Path) -> dict[str, Any]:
    """Verify the exact e238 base-tree scorer helper, kept outside the 172-file bundle."""
    manifest_path = package_root / "score_support_manifest.json"
    helper_path = package_root / "score_support" / "grade_prediction_bank.py"
    if sha256_file(manifest_path) != SCORE_SUPPORT_MANIFEST_SHA256:
        raise ValueError("PR15 scorer support manifest differs from its Git-object pin")
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema") != "pr15-scorer-support.v1"
        or manifest.get("source_commit") != SOURCE_COMMIT
        or not isinstance(files, list)
        or len(files) != 1
    ):
        raise ValueError("unsupported PR15 scorer support manifest")
    record = files[0]
    if (
        record.get("git_blob_oid") != SCORE_SUPPORT_BLOB_OID
        or record.get("sha256") != SCORE_SUPPORT_SHA256
        or record.get("path") != "scripts/grade_prediction_bank.py"
        or record.get("size_bytes") != 11114
        or helper_path.stat().st_size != 11114
        or sha256_file(helper_path) != SCORE_SUPPORT_SHA256
    ):
        raise ValueError("PR15 scorer support helper is not the pinned e238 Git blob")
    return {
        "manifest_sha256": SCORE_SUPPORT_MANIFEST_SHA256,
        "git_blob_oid": SCORE_SUPPORT_BLOB_OID,
        "helper_sha256": SCORE_SUPPORT_SHA256,
        "helper_bytes": 11114,
        "recursive_dependencies": manifest["recursive_dependencies"],
    }


def parse_profile_events(log_text: str) -> dict[str, Any]:
    """Parse exactly one frozen profile, resource and update evidence event."""
    events: dict[str, Any] = {}
    for line in log_text.splitlines():
        for prefix, key in PROFILE_PREFIXES.items():
            if line.startswith(prefix):
                if key in events:
                    raise ValueError(f"profile log contains duplicate {key} evidence")
                event = json.loads(line[len(prefix) :])
                if not isinstance(event, dict):
                    raise ValueError(f"profile {key} evidence is not a JSON object")
                events[key] = event
    expected = set(PROFILE_PREFIXES.values())
    if set(events) != expected:
        raise ValueError(f"profile log is missing evidence: {sorted(expected - set(events))}")
    return events


def validate_profile_events(events: dict[str, Any]) -> None:
    """Enforce PR15's 2-warmup/8-measured-batch profile and strict-FP32 gates."""
    timing, resources, updates = (
        events["timing"],
        events["resources"],
        events["updates"],
    )
    for key, value in timing.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"profile timing {key} is not numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"profile timing {key} is negative or nonfinite")
    for key in ("step_wall", "gpu_total", "seq_per_s"):
        if timing.get(key, 0) <= 0:
            raise ValueError(f"profile timing {key} must be positive")
    if (
        resources.get("precision") != "fp32"
        or resources.get("matmul_tf32") is not False
        or resources.get("cudnn_tf32") is not False
        or resources.get("float32_matmul_precision") != "highest"
    ):
        raise ValueError("profile did not prove strict FP32 with TF32 disabled")
    exact = {
        "optimizer_attempts": 100,
        "optimizer_completed_updates": 100,
        "amp_skipped_updates": 0,
        "loss_checks": 100,
        "gradient_stats_steps": 100,
        "gradient_stats_nonfinite_steps": 0,
    }
    for key, expected in exact.items():
        value = updates.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"profile {key} expected {expected}, observed {value!r}")
    for key in (
        "loss_finite",
        "gradient_finite",
        "counts_consistent",
        "valid_for_throughput",
    ):
        if updates.get(key) is not True:
            raise ValueError(f"profile update evidence failed {key}")
    if updates.get("grad_scaler_enabled") is not False:
        raise ValueError("strict FP32 profile unexpectedly enabled GradScaler")


def validate_fit_artifacts(work: Path, name: str) -> dict[str, Any]:
    """Require the exact fixed epoch-2 checkpoint and complete source-native logs."""
    run_dir = work / "artifacts" / "runs" / name
    required = {
        "checkpoint_ep2": run_dir / "model_ep2.pt",
        "selected_checkpoint": run_dir / "model.pt",
        "history": run_dir / "history.json",
        "provenance": run_dir / "provenance.json",
        "training_state": run_dir / "training_state.pt",
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"fit did not produce required {label}: {path}")
    checkpoint_identity = verify_checkpoint_pair(
        required["checkpoint_ep2"], required["selected_checkpoint"]
    )
    checkpoint_hash = checkpoint_identity.fixed_file_sha256
    history = json.loads(required["history"].read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 3:
        raise ValueError("PR15 fit must record exactly three completed training epochs")
    if any(row.get("evaluation") != "withheld" for row in history):
        raise ValueError("PR15 fit unexpectedly evaluated during training")
    provenance = read_json(required["provenance"])
    validate_provenance_file_hashes(provenance, checkpoint_identity)
    if provenance.get("args", {}).get("checkpoint_epoch") != 2:
        raise ValueError("trainer provenance does not bind the fixed epoch-2 endpoint")
    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(required["checkpoint_ep2"]),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_identity": checkpoint_identity.receipt(),
        "provenance_path": str(required["provenance"]),
        "provenance_sha256": sha256_file(required["provenance"]),
        "history_path": str(required["history"]),
        "history_sha256": sha256_file(required["history"]),
        "training_state_path": str(required["training_state"]),
        "training_state_sha256": sha256_file(required["training_state"]),
        "epochs": len(history),
    }


def validate_prediction_and_score(
    work: Path, name: str, split: dict[str, Any], split_hash: str
) -> dict[str, Any]:
    """Check exported evaluation membership/shape and the official PR15 score."""
    import numpy as np

    prediction = work / "cache" / f"pred_{name}_valid.npy"
    metadata_path = prediction.with_suffix(".json")
    score_path = work / "artifacts" / "runs" / name / "score.json"
    for path in (prediction, metadata_path, score_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"prediction/scorer output is missing: {path}")
    metadata = read_json(metadata_path)
    groups = split.get("evaluation", {}).get("valid")
    if (
        metadata.get("run") != name
        or metadata.get("epoch") != 2
        or metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("campaign_split_manifest_sha256") != split_hash
        or metadata.get("row_groups") != groups
    ):
        raise ValueError("prediction sidecar does not bind the fixed fold evaluation set")
    array = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(groups), 20_000, 2) or array.dtype != np.float32:
        raise ValueError(f"unexpected PR15 prediction shape/dtype: {array.shape}/{array.dtype}")
    if metadata.get("prediction_sha256") != sha256_file(prediction):
        raise ValueError("prediction sidecar hash does not match exported bytes")
    score = read_json(score_path)
    if (
        score.get("schema") != "wnn.hpo-score.v1"
        or score.get("run") != name
        or score.get("epoch") != 2
    ):
        raise ValueError("source-native score output is not the PR15 fixed epoch-2 result")
    for key in ("wp", "t0", "t1", "finite_fraction"):
        value = score.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"official score field {key} is missing or nonfinite")
    if score["finite_fraction"] != 1.0:
        raise ValueError("official score reports nonfinite predictions on scored rows")
    return {
        "prediction_path": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        "prediction_metadata_path": str(metadata_path),
        "prediction_metadata_sha256": sha256_file(metadata_path),
        "prediction_shape": list(array.shape),
        "prediction_dtype": str(array.dtype),
        "score_path": str(score_path),
        "score_sha256": sha256_file(score_path),
        "score": score,
        "metric_id": "global_wp_clip2_v1",
        "scorer": "frozen PR15 scripts/score_run.py",
    }
