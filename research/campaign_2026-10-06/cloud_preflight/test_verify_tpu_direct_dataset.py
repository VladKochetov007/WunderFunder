from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("verify_tpu_direct_dataset.py")
SPEC = importlib.util.spec_from_file_location("verify_tpu_direct_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load dataset identity helper from {MODULE_PATH}")
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def make_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a small valid direct-mount fixture with a matching manifest."""
    bundle = tmp_path / "source_bundle" / "src"
    bundle.mkdir(parents=True)
    source_bytes = b"frozen source"
    source_path = bundle / "model.py"
    source_path.write_bytes(source_bytes)
    archive_sha256 = hashlib.sha256(b"declared archive").hexdigest()
    manifest = {
        "dataset": VERIFIER.EXPECTED_DATASET,
        "private": True,
        "internet": False,
        "gpu": False,
        "tpu": True,
        "archive_sha256": archive_sha256,
        "files": {
            "src/model.py": {
                "size_bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        },
    }
    manifest_bytes = json.dumps(manifest).encode()
    (tmp_path / "source-manifest.json").write_bytes(manifest_bytes)
    monkeypatch.setattr(VERIFIER, "EXPECTED_ARCHIVE_SHA256", archive_sha256)
    monkeypatch.setattr(
        VERIFIER, "EXPECTED_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest()
    )
    return tmp_path


def test_accepts_manifest_bound_direct_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept direct files that exactly match the frozen manifest."""
    dataset_root = make_dataset(tmp_path, monkeypatch)

    result = VERIFIER.verify(dataset_root, None)

    assert result["status"] == "passed"
    assert result["archive_file_present"] is False
    assert result["all_direct_bundle_files_match_manifest"] is True


def test_rejects_changed_manifest_listed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject byte changes even when the source path is present."""
    dataset_root = make_dataset(tmp_path, monkeypatch)
    (dataset_root / "source_bundle" / "src" / "model.py").write_bytes(b"changed source")

    with pytest.raises(ValueError, match="source file identity changed"):
        VERIFIER.verify(dataset_root, None)


def test_rejects_unlisted_direct_tree_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject extra files that are outside the frozen manifest."""
    dataset_root = make_dataset(tmp_path, monkeypatch)
    (dataset_root / "source_bundle" / "extra.txt").write_text("extra")

    with pytest.raises(ValueError, match="direct source tree differs from manifest"):
        VERIFIER.verify(dataset_root, None)


def test_rejects_provider_listing_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a local download that differs from the provider file listing."""
    dataset_root = make_dataset(tmp_path, monkeypatch)
    listing_path = tmp_path / "provider-files.json"
    listing_path.write_text(json.dumps([{"name": "source-manifest.json", "size": 0}]))

    with pytest.raises(ValueError, match="downloaded direct dataset differs from Kaggle listing"):
        VERIFIER.verify(dataset_root, listing_path)
