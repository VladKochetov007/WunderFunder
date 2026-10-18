"""Exact restored-file bindings for the isolated Main F0 compatibility export."""

from __future__ import annotations

import hashlib
from pathlib import Path

_F0_RESTORE_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1")
_F0_ARTIFACTS = {
    "control_checkpoint": {
        "path": "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt",
        "restore_path": _F0_RESTORE_BASE
        / "restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt",
        "bytes": 3983015,
        "sha256": "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda",
    },
    "control_provenance": {
        "path": "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json",
        "restore_path": _F0_RESTORE_BASE
        / "restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json",
        "bytes": 166152,
        "sha256": "bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961",
    },
    "propensity_checkpoint": {
        "path": "artifacts/runs/c20260929_prop_f0/model.pt",
        "restore_path": _F0_RESTORE_BASE
        / "restored-propensity-v1/artifacts/runs/c20260929_prop_f0/model.pt",
        "bytes": 165623,
        "sha256": "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7",
    },
    "propensity_provenance": {
        "path": "artifacts/runs/c20260929_prop_f0/provenance.json",
        "restore_path": _F0_RESTORE_BASE
        / "restored-propensity-v1/artifacts/runs/c20260929_prop_f0/provenance.json",
        "bytes": 21626,
        "sha256": "f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812",
    },
}


def _file_sha256(path: Path) -> str:
    """Hash a small restored artifact without loading tensors into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_f0_checkpoint_bindings() -> dict[str, dict[str, object]]:
    """Return immutable expected logical paths and exact B restore bindings."""
    return {
        name: {
            "path": str(binding["path"]),
            "restore_path": str(binding["restore_path"]),
            "bytes": int(binding["bytes"]),
            "sha256": str(binding["sha256"]),
        }
        for name, binding in _F0_ARTIFACTS.items()
    }


def verify_f0_checkpoint_bindings(project_root: Path) -> dict[str, dict[str, object]]:
    """Verify staged symlinks resolve to the pinned B restore files and hashes."""
    verified: dict[str, dict[str, object]] = {}
    for name, expected in expected_f0_checkpoint_bindings().items():
        restored_path = Path(str(expected["restore_path"]))
        staged_path = project_root / str(expected["path"])
        if staged_path.resolve(strict=True) != restored_path.resolve(strict=True):
            raise ValueError(f"staged F0 artifact does not point to its restored source: {name}")
        for path in (restored_path, staged_path):
            stat = path.stat()
            if stat.st_size != expected["bytes"] or _file_sha256(path) != expected["sha256"]:
                raise ValueError(f"restored F0 artifact has an unexpected identity: {name}")
        verified[name] = expected
    return verified
