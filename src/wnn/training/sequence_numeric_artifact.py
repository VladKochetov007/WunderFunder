from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

# JSON artifact fields are heterogeneous until the schema checks at the file boundary.
from typing import Any

import numpy as np

from wnn.data import N_FEATURES
from wnn.splits import MAX_CAMPAIGN_SPLIT_BYTES
from wnn.training.sequence_numeric import NUMERIC_INPUT_WIDTH, NUMERIC_MISSING_OFFSET

ARTIFACT_SCHEMA = "wnn.numeric-inputs.v1"
FEATURE_SCHEMA = "raw112-event99-missing112-v1"
PREPARATION_SCHEMA = "wnn.numeric-input-preparation.v1"


def _canonical_bytes(value: object) -> bytes:
    """Encode JSON-compatible values in the format used by artifact hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_sha256(value: object) -> str:
    """Hash JSON-compatible values deterministically."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _module_hashes() -> dict[str, str]:
    """Pin preprocessing and canonical source implementations in the artifact."""
    repository = Path(__file__).resolve().parents[3]
    names = (
        "src/wnn/data.py",
        "src/wnn/features.py",
        "src/wnn/splits.py",
        "src/wnn/training/sequence_numeric.py",
        "src/wnn/training/sequence_numeric_artifact.py",
        "src/wnn/training/sequence_numeric_preprocessing.py",
    )
    return {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in names}


def _validate_fit_arrays(artifact: dict[str, Any]) -> None:
    """Reject malformed or nonfinite normalization and knot arrays."""
    mean = np.asarray(artifact.get("mean"), dtype=np.float64)
    std = np.asarray(artifact.get("std"), dtype=np.float64)
    if mean.shape != (NUMERIC_INPUT_WIDTH,) or std.shape != (NUMERIC_INPUT_WIDTH,):
        raise ValueError("numeric artifact mean and std must each contain 323 values")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("numeric artifact moments must be finite with positive std")
    mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
    if not np.isfinite(mean32).all() or not np.isfinite(std32).all() or np.any(std32 <= 0):
        raise ValueError("numeric artifact moments must be representable in float32")
    if np.any(mean[NUMERIC_MISSING_OFFSET:] != 0.0) or np.any(std[NUMERIC_MISSING_OFFSET:] != 1.0):
        raise ValueError("numeric missing-indicator normalization must remain identity")
    knots = artifact.get("knots")
    if not isinstance(knots, dict) or set(knots) != {"uniform16", "quantile16"}:
        raise ValueError("numeric artifact must contain both sixteen-bin knot sets")
    for name, values in knots.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (N_FEATURES, 17) or not np.isfinite(array).all():
            raise ValueError(f"numeric artifact {name} knots must be finite [112, 17]")
        if np.any(array[:, 1:] < array[:, :-1]):
            raise ValueError(f"numeric artifact {name} knots must be nondecreasing")
        if not np.isfinite(array.astype(np.float32)).all():
            raise ValueError(f"numeric artifact {name} knots must be representable in float32")
    uniform = np.asarray(knots["uniform16"], dtype=np.float32)
    quantile = np.asarray(knots["quantile16"], dtype=np.float32)
    if not np.array_equal(uniform[:, (0, -1)], quantile[:, (0, -1)]):
        raise ValueError("numeric artifact knot sets must share full-fit endpoints")


def _atomic_create(path: Path, payload: bytes) -> None:
    """Publish a complete file atomically without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_numeric_artifact(
    path: Path,
    split_sha256: str,
    members: list[tuple[str, int]],
    sources: dict[str, object],
) -> dict[str, Any]:
    """Validate a prepared artifact against the exact fitting and source pins."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_CAMPAIGN_SPLIT_BYTES:
        raise ValueError("numeric preprocessing artifact exceeds its bounded read size")
    artifact = json.loads(path.read_bytes())
    if not isinstance(artifact, dict) or artifact.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported numeric preprocessing artifact")
    if artifact.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("numeric preprocessing feature schema differs")
    if artifact.get("split_sha256") != split_sha256:
        raise ValueError("numeric preprocessing split hash differs")
    expected_members = [[source, int(group)] for source, group in members]
    if artifact.get("fit_members") != expected_members:
        raise ValueError("numeric preprocessing fit membership differs")
    if _canonical_bytes(artifact.get("sources")) != _canonical_bytes(sources):
        raise ValueError("numeric preprocessing source identities differ")
    artifact_hash = artifact.get("preparation_hash")
    content = {key: value for key, value in artifact.items() if key != "preparation_hash"}
    if not isinstance(artifact_hash, str) or artifact_hash != _canonical_sha256(content):
        raise ValueError("numeric preprocessing artifact hash is invalid")
    identity = artifact.get("preparation_identity")
    if not isinstance(identity, dict) or identity.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("numeric preprocessing identity is missing")
    if identity.get("algorithm") != "float64-moments-priority-reservoir-v1":
        raise ValueError("numeric preprocessing algorithm is unsupported")
    if identity.get("fit_members_sha256") != _canonical_sha256(expected_members):
        raise ValueError("numeric preprocessing member hash is invalid")
    code_hashes = identity.get("source_code_sha256")
    if not isinstance(code_hashes, dict) or code_hashes != _module_hashes():
        raise ValueError("numeric preprocessing source code hashes differ")
    if type(identity.get("calibration_seed")) is not int or identity["calibration_seed"] < 0:
        raise ValueError("numeric preprocessing calibration seed is invalid")
    if type(identity.get("full_scanned_rows")) is not int or identity["full_scanned_rows"] <= 0:
        raise ValueError("numeric preprocessing scanned row count is invalid")
    limit = identity.get("reservoir_rows_limit")
    sampled = identity.get("reservoir_rows_sampled")
    if type(limit) is not int or limit < 1 or type(sampled) is not int or not 1 <= sampled <= limit:
        raise ValueError("numeric preprocessing reservoir row count is invalid")
    sample_hash = identity.get("reservoir_sample_sha256")
    if not isinstance(sample_hash, str) or len(sample_hash) != 64:
        raise ValueError("numeric preprocessing reservoir hash is missing")
    try:
        int(sample_hash, 16)
    except ValueError as exc:
        raise ValueError("numeric preprocessing reservoir hash is malformed") from exc
    _validate_fit_arrays(artifact)
    return {
        "mean": artifact["mean"],
        "std": artifact["std"],
        "knots": artifact["knots"],
        "preparation_hash": artifact_hash,
        "preparation_identity": identity,
    }
