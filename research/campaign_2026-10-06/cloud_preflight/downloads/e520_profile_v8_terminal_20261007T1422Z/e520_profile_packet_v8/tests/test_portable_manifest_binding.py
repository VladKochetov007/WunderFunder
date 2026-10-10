"""Focused checks for the E520 v8 portable-registry pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from prepare_profile_packet import KAGGLE_ID, KAGGLE_TITLE
from profile_worker import (
    PORTABLE_RELATIVE,
    ProfileAbort,
    bind_verified_portable_manifest,
)

from wnn.portability import verify_portable_source


def _portable_path(root: Path) -> Path:
    """Create the exact staged-manifest path used by the profile worker."""
    path = root / PORTABLE_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"version":1,"sources":{}}\n')
    return path


def test_manifest_environment_is_bound_only_after_hash_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _portable_path(tmp_path / "source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    monkeypatch.setattr("profile_worker.PORTABLE_SHA256", "0" * 64)
    receipt_path = tmp_path / "receipts" / "input_identity.json"
    identity: dict[str, object] = {"phase": "input_identity", "status": "in_progress"}

    with pytest.raises(ProfileAbort, match="portable_manifest_binding"):
        bind_verified_portable_manifest(tmp_path / "source", receipt_path, identity)

    assert os.environ.get("WNN_PORTABLE_DATA_MANIFEST") is None
    failed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    binding = "portable_manifest_binding"
    assert failed_receipt["expected"][binding]["sha256"] == "0" * 64
    assert failed_receipt["observed"][binding]["sha256"] == digest

    monkeypatch.setattr("profile_worker.PORTABLE_SHA256", digest)
    bound_path = bind_verified_portable_manifest(tmp_path / "source", receipt_path, identity)
    assert bound_path == path
    assert os.environ["WNN_PORTABLE_DATA_MANIFEST"] == str(path)
    matched_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert matched_receipt["portable_manifest_env"] == str(path)
    assert matched_receipt["observed"][binding]["sha256"] == digest


def test_portable_verifier_checks_content_after_relocation_changes_mtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "chain_folds.npz"
    source.write_bytes(b"feature-only fold assignment")
    stat = source.stat()
    original_identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    registry = tmp_path / "portable_inputs.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "cache/chain_folds.npz": {
                        "original_identity": original_identity,
                        "sha256": digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    verify_portable_source(source, "cache/chain_folds.npz", original_identity, registry)

    source.write_bytes(b"feature-only fold assignmenT")
    with pytest.raises(ValueError, match="content changed"):
        verify_portable_source(source, "cache/chain_folds.npz", original_identity, registry)


def test_v8_provider_slug_matches_the_title() -> None:
    assert KAGGLE_TITLE == "E520 Profile V8 GPU Observation Fallback"
    assert KAGGLE_ID.rsplit("/", maxsplit=1)[-1] == "e520-profile-v8-gpu-observation-fallback"
