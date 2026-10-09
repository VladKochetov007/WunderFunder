from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("e463_tpu_probe_v3", HERE / "tpu_probe_v3.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("TPU v3 probe cannot be imported")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a nested Kaggle mount with a manifest-bound direct source tree."""
    input_root = tmp_path / "input"
    source_mount = (
        input_root / "datasets" / "vladkochetov07" / "wnn-jax-realdata-e463-tpu-20261004-v1"
    )
    bundle_root = source_mount / "source_bundle"
    bundle_root.mkdir(parents=True)
    source_bytes = b"immutable JAX model source\n"
    source_name = "src/wnn_jax/rnn.py"
    source_path = bundle_root / source_name
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_bytes)
    archive_sha256 = hashlib.sha256(b"declared archive provenance").hexdigest()
    manifest = {
        "dataset": "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
        "version": 2,
        "format": "wnn.jax-realdata-preflight-inputs",
        "private": True,
        "internet": False,
        "gpu": False,
        "tpu": True,
        "archive_sha256": archive_sha256,
        "files": {
            source_name: {
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "size_bytes": len(source_bytes),
            }
        },
    }
    manifest_bytes = json.dumps(manifest).encode()
    manifest_path = source_mount / "source-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    asset_root = input_root / "datasets" / "vladkochetov07" / "wunder-funder"
    asset_root.mkdir()
    monkeypatch.setattr(probe, "INPUT_ROOT", input_root)
    monkeypatch.setattr(probe, "SOURCE_MOUNT_ROOT", source_mount)
    monkeypatch.setattr(probe, "ASSET_ROOT", asset_root)
    monkeypatch.setattr(probe, "SOURCE_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(probe, "SOURCE_ARCHIVE_SHA256", archive_sha256)
    return source_mount


def test_verifies_exact_nested_direct_tree_without_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept the provider's expanded tree after checking every manifest hash."""
    source_mount = make_fixture(tmp_path, monkeypatch)

    result = probe.verify_direct_source_tree()

    assert result["status"] == "verified_direct_read_only_tree"
    assert result["mount_path"] == str(source_mount)
    assert result["verified_file_count"] == 1
    assert result["source_archive_present"] is False
    assert result["archive_digest_used_as_gate"] is False
    assert result["asset_root_exists"] is True


def test_rejects_changed_manifest_listed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject changed bytes before TPU runtime startup."""
    source_mount = make_fixture(tmp_path, monkeypatch)
    (source_mount / "source_bundle/src/wnn_jax/rnn.py").write_bytes(b"changed")

    with pytest.raises(ValueError, match="direct source identity mismatch"):
        probe.verify_direct_source_tree()


def test_rejects_unlisted_direct_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject files that are absent from the frozen source manifest."""
    source_mount = make_fixture(tmp_path, monkeypatch)
    (source_mount / "source_bundle/unlisted.txt").write_text("extra")

    with pytest.raises(ValueError, match="member sets differ"):
        probe.verify_direct_source_tree()


def test_rejects_symbolic_link_in_direct_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject symlinks anywhere in the source bundle tree."""
    source_mount = make_fixture(tmp_path, monkeypatch)
    target = source_mount / "source_bundle/src/wnn_jax/rnn.py"
    target.rename(target.with_suffix(".original"))
    target.symlink_to(target.with_suffix(".original"))

    with pytest.raises(ValueError, match="is a symlink"):
        probe.verify_direct_source_tree()
