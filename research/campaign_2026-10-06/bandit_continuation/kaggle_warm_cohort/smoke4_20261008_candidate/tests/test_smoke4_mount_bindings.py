"""Offline checks for the reviewed Kaggle asset and registry handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

KERNEL_PATH = Path(__file__).parents[1] / "kernel" / "wnn-warm-cohort-smoke-20261008-v4.py"
SPEC = importlib.util.spec_from_file_location("smoke4_kernel", KERNEL_PATH)
assert SPEC is not None and SPEC.loader is not None
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_candidate_pins_reviewed_source_manifests_and_registry() -> None:
    references = KERNEL_PATH.parents[1] / "references"
    fold_manifest_path = references / "fold0-assets-manifest.json"
    source_manifest_path = references / "source-v2-assets-manifest.json"
    registry_path = references / "portable_inputs.json"
    fold_manifest = json.loads(fold_manifest_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())

    assert KERNEL.EXPECTED_ASSET_MANIFESTS == {
        "vladkochetov07/wnn-warm-cohort-src-20261008-v2": "96fa0dcbd2c0a56f8ef9d23620080337799022290ffee87196d52de58aca4f70",
        "vladkochetov07/wnn-c20261004-fold0-assets": "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7",
    }
    assert fold_manifest["dataset"] == "vladkochetov07/wnn-c20261004-fold0-assets"
    assert source_manifest["dataset"] == "vladkochetov07/wnn-warm-cohort-src-20261008-v2"
    assert _sha256(fold_manifest_path.read_bytes()) == KERNEL.EXPECTED_ASSET_MANIFESTS[
        fold_manifest["dataset"]
    ]
    assert _sha256(source_manifest_path.read_bytes()) == KERNEL.EXPECTED_ASSET_MANIFESTS[
        source_manifest["dataset"]
    ]
    assert _sha256(registry_path.read_bytes()) == KERNEL.PORTABLE_REGISTRY_SHA256
    registry_entries = [
        item
        for item in fold_manifest["files"]
        if item["path"] == str(KERNEL.PORTABLE_REGISTRY_RELATIVE_PATH)
    ]
    assert len(registry_entries) == 1
    assert registry_entries[0]["sha256"] == KERNEL.PORTABLE_REGISTRY_SHA256


def test_kernel_title_slug_matches_metadata_id() -> None:
    metadata = json.loads((KERNEL_PATH.parent / "kernel-metadata.json").read_text())
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")

    assert metadata["code_file"] == KERNEL_PATH.name
    assert metadata["id"].rsplit("/", 1)[1] == title_slug


def _asset_fixture(root: Path, *, dataset: str, payload: bytes) -> Path:
    directory = root / "datasets" / Path(dataset)
    directory.mkdir(parents=True)
    (directory / "payload.bin").write_bytes(payload)
    manifest = {
        "dataset": dataset,
        "version": 1,
        "files": [
            {
                "filename": "payload.bin",
                "path": "cache/payload.bin",
                "sha256": _sha256(payload),
                "size": len(payload),
            }
        ],
    }
    manifest_path = directory / "assets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path


def test_mount_manifests_checks_dataset_manifest_and_payload_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    payload = b"reviewed small asset"
    manifest_path = _asset_fixture(
        input_root, dataset="owner/reviewed-assets", payload=payload
    )
    monkeypatch.setattr(KERNEL, "INPUT", input_root)
    monkeypatch.setattr(KERNEL, "REPO", tmp_path / "repo")
    monkeypatch.setattr(KERNEL, "WORK", tmp_path / "work")
    monkeypatch.setattr(KERNEL, "phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        KERNEL,
        "EXPECTED_ASSET_MANIFESTS",
        {"owner/reviewed-assets": _sha256(manifest_path.read_bytes())},
    )

    mounted = KERNEL.mount_manifests()

    target = KERNEL.REPO / "cache/payload.bin"
    assert mounted == [
        {"path": "cache/payload.bin", "bytes": len(payload), "dataset_dir": str(manifest_path.parent)}
    ]
    assert target.is_symlink()
    assert target.read_bytes() == payload


def test_mount_manifests_rejects_an_unreviewed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    _asset_fixture(input_root, dataset="owner/reviewed-assets", payload=b"asset")
    monkeypatch.setattr(KERNEL, "INPUT", input_root)
    monkeypatch.setattr(KERNEL, "REPO", tmp_path / "repo")
    monkeypatch.setattr(KERNEL, "WORK", tmp_path / "work")
    monkeypatch.setattr(KERNEL, "phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(KERNEL, "EXPECTED_ASSET_MANIFESTS", {"owner/reviewed-assets": "0" * 64})

    with pytest.raises(ValueError, match="asset manifest changed"):
        KERNEL.mount_manifests()

    assert not (KERNEL.REPO / "cache/payload.bin").exists()


def test_bind_portable_registry_injects_exact_path_into_every_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "repo/research/campaign_2026-10-04/portable_inputs.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(b'{"version":1}\n')
    monkeypatch.setattr(KERNEL, "REPO", tmp_path / "repo")
    monkeypatch.setattr(KERNEL, "PORTABLE_REGISTRY_SHA256", _sha256(registry.read_bytes()))
    specs = [
        {"run_id": "fit", "steps": [{"name": "fit", "env": {"OMP_NUM_THREADS": "1"}}]},
        {"run_id": "predict", "steps": [{"name": "predict", "env": {}}]},
    ]

    receipt = KERNEL.bind_portable_registry(specs)

    assert receipt == {
        "path": str(registry),
        "sha256": _sha256(registry.read_bytes()),
        "child_steps": 2,
    }
    for spec in specs:
        for step in spec["steps"]:
            assert step["env"]["WNN_PORTABLE_DATA_MANIFEST"] == str(registry)


def test_bind_portable_registry_fails_closed_on_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "repo/research/campaign_2026-10-04/portable_inputs.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(b'{"version":1}\n')
    monkeypatch.setattr(KERNEL, "REPO", tmp_path / "repo")
    monkeypatch.setattr(KERNEL, "PORTABLE_REGISTRY_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="portable data registry changed"):
        KERNEL.bind_portable_registry([{"run_id": "fit", "steps": [{"name": "fit"}]}])
