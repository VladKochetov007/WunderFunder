from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from scripts.kaggle_qat_kernel_r1 import (
    MATERIALIZED_CACHE_ROOTS,
    materialize_distillation_cache,
    release_mounts,
)


def producer_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    """Project the real producer inventory into bounded copy-test payloads."""
    repository = Path(__file__).resolve().parents[1]
    receipt_path = repository / (
        "research/campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/"
        "fixtures/actual_input_preflight_receipt_r1.json"
    )
    actual = json.loads(receipt_path.read_text())
    root = tmp_path / "experiment"
    mount = tmp_path / "datasets"
    mount.mkdir()
    receipt = {"status": "pass", "inputs": []}
    manifest = {"files": [], "external_mounts": {}}
    for item in actual["inputs"]:
        if str(Path(item["path"]).parent) not in MATERIALIZED_CACHE_ROOTS:
            continue
        # Actual logical paths/schema, deliberately reduced byte counts/hashes.
        content = item["sha256"].encode() + bytes(127)
        source = mount / Path(item["resolved_path"]).name
        source.write_bytes(content)
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        digest = hashlib.sha256(content).hexdigest()
        receipt["inputs"].append(
            dict(
                item,
                resolved_path=str(source),
                size=len(content),
                expected_size=len(content),
                sha256=digest,
                expected_sha256=digest,
            )
        )
        manifest["files"].append({"path": item["path"], "size": len(content), "sha256": digest})
        manifest["external_mounts"][item["path"]] = {"dataset": item["dataset"]}
    assert len(manifest["files"]) == 3
    return root, manifest, receipt


def test_materialization_preserves_source_bytes_and_restores_containment(tmp_path: Path) -> None:
    """Resolved cache paths become ordinary contained files without source writes."""
    root, manifest, receipt = producer_fixture(tmp_path)
    sources = {
        Path(item["resolved_path"]): Path(item["resolved_path"]).read_bytes()
        for item in receipt["inputs"]
    }
    copied = materialize_distillation_cache(
        root, manifest, receipt, tmp_path / "copy.json", time.monotonic() + 30
    )
    assert len(copied) == 3
    for target, item in zip(copied, manifest["files"], strict=True):
        assert not target.is_symlink()
        assert target.resolve().is_relative_to(root.resolve())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
    assert json.loads((tmp_path / "copy.json").read_text())["status"] == "pass"
    release_mounts(root, manifest["external_mounts"])
    assert all(not path.exists() for path in copied)
    assert all(path.read_bytes() == content for path, content in sources.items())


def test_changed_bytes_do_not_replace_mount_and_failure_is_recorded(tmp_path: Path) -> None:
    """A same-length mutation after preflight must fail before candidate admission."""
    root, manifest, receipt = producer_fixture(tmp_path)
    source = Path(receipt["inputs"][0]["resolved_path"])
    source.write_bytes(b"!" * source.stat().st_size)
    with pytest.raises(ValueError, match="differs from producer bytes"):
        materialize_distillation_cache(
            root, manifest, receipt, tmp_path / "copy.json", time.monotonic() + 30
        )
    assert (root / manifest["files"][0]["path"]).is_symlink()
    assert json.loads((tmp_path / "copy.json").read_text())["status"] == "failed"
    release_mounts(root, manifest["external_mounts"])
    assert not list(root.rglob("*.materializing"))


def test_rejects_manifest_receipt_disagreement_and_expired_deadline(tmp_path: Path) -> None:
    """Copied cache bytes remain bound to both trusted descriptors and the deadline."""
    root, manifest, receipt = producer_fixture(tmp_path)
    receipt["inputs"][0]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity differs"):
        materialize_distillation_cache(
            root, manifest, receipt, tmp_path / "copy.json", time.monotonic() + 30
        )
    with pytest.raises(TimeoutError, match="deadline"):
        materialize_distillation_cache(
            root, manifest, receipt, tmp_path / "timeout.json", time.monotonic() - 1
        )
