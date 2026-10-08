"""Offline checks for the scoped H232 R7 payload-manifest correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
R6_ROOT = PACKAGE_ROOT / "asset_transport_r6"
sys.path.insert(0, str(HERE))

import build_r7_packet as builder  # noqa: E402


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a frozen packet module directly from its source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    """Hash a fixture file through bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def r7_artifact() -> tuple[dict[str, Any], Path, dict[str, str]]:
    """Load the locally built R7 packet and exact R6 payload identity."""
    receipt = builder.read_json(HERE / "H232_V7B_EP3_ASSET_TRANSPORT_R7_BUILD_RECEIPT_R1.json")
    payload_path = PACKAGE_ROOT / str(receipt["r6_payload_path"])
    members = cast(dict[str, str], receipt["payload_member_sha256"])
    return receipt, payload_path, members


def test_r7_keeps_the_r6_payload_worker_checkpoint_and_source_bytes(
    r7_artifact: tuple[dict[str, Any], Path, dict[str, str]],
) -> None:
    """The successor only changes the transport manifest basename and packet ID."""
    receipt, payload_path, member_hashes = r7_artifact
    r6 = builder.read_json(builder.R6_BUILD_RECEIPT)
    r6_members = cast(dict[str, str], r6["payload_member_sha256"])

    assert receipt["r6_payload_sha256"] == builder.EXPECTED_R6_PAYLOAD_SHA256
    assert _sha256(payload_path) == builder.EXPECTED_R6_PAYLOAD_SHA256
    assert member_hashes == r6_members
    assert receipt["checkpoint_sha256"] == builder.EXPECTED_R6_CHECKPOINT_SHA256
    assert receipt["source_archive_sha256"] == builder.EXPECTED_R6_SOURCE_ARCHIVE_SHA256
    assert receipt["scientific_source_worker_checkpoint_payload_unchanged"] is True
    assert receipt["root_kernel_push_authorized"] is False


def test_provider_inventory_is_audited_against_every_worker_basename(
    r7_artifact: tuple[dict[str, Any], Path, dict[str, str]],
) -> None:
    """Only R6's old manifest collides; the proposed R7 name clears all lookups."""
    receipt, _, _ = r7_artifact
    inventory = builder.read_json(
        HERE / "reference/H232_R6_PROVIDER_INPUT_INVENTORY_20261008_R1.json"
    )
    inventories = cast(dict[str, list[dict[str, Any]]], inventory["dataset_file_inventories"])
    assert set(inventories) == {
        "vladkochetov07/wunder-funder",
        "vladkochetov07/wnn-c20261004-fold0-assets",
        "vladkochetov07/h232-v7b-ep3-payload-r6-dda776b3f9b7/1",
    }
    old_names = [entry["name"] for rows in inventories.values() for entry in rows]
    assert sorted(name for name, count in Counter(old_names).items() if count > 1) == [
        "assets-manifest.json"
    ]
    consumer_names = cast(list[str], inventory["consumer_global_lookup_names"])
    old_counts = Counter(old_names)
    assert all(
        old_counts[name] == (2 if name == "assets-manifest.json" else 1) for name in consumer_names
    )

    r7_upload = PACKAGE_ROOT / str(receipt["asset_upload_path"])
    proposed_names = [
        path.name
        for path in r7_upload.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    ]
    expected_proposed_names = {receipt["asset_manifest_filename"], receipt["asset_filename"]}
    if receipt["asset_filename"] not in proposed_names:
        assert _sha256(PACKAGE_ROOT / str(receipt["r6_payload_path"])) == receipt["asset_sha256"]
        proposed_names.append(str(receipt["asset_filename"]))
    assert set(proposed_names) == expected_proposed_names
    assert "assets-manifest.json" not in proposed_names
    proposed_inventories = [
        inventories["vladkochetov07/wunder-funder"],
        inventories["vladkochetov07/wnn-c20261004-fold0-assets"],
        [{"name": name} for name in proposed_names],
    ]
    proposed_counts = Counter(entry["name"] for rows in proposed_inventories for entry in rows)
    assert all(proposed_counts[name] == 1 for name in consumer_names)
    assert not [name for name, count in proposed_counts.items() if count > 1]


def _extract_frozen_worker(payload_path: Path, destination: Path) -> Path:
    """Extract only the reviewed verifier files needed by the mount fixture."""
    r6 = builder.read_json(builder.R6_BUILD_RECEIPT)
    members = cast(dict[str, str], r6["payload_member_sha256"])
    requested = (
        "profile_worker.py",
        "cache_pressure_recovery.py",
        "gpu_memory_domains.py",
        "verified_digest_handoff.py",
        "prior_input_identity_v2.json",
    )
    destination.mkdir()
    with zipfile.ZipFile(payload_path) as archive:
        for name in requested:
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == members[name]
            (destination / name).write_bytes(data)
    return destination / "profile_worker.py"


def _mount_directory(input_root: Path, dataset: str) -> Path:
    """Create a nested Kaggle-style dataset mount path."""
    mount = input_root / "datasets" / "vladkochetov07" / dataset
    mount.mkdir(parents=True, exist_ok=True)
    return mount


def test_unchanged_worker_selects_fold_manifest_after_r7_rename(
    r7_artifact: tuple[dict[str, Any], Path, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the frozen verifier's real recursive discovery on three mounts."""
    receipt, payload_path, _ = r7_artifact
    worker_path = _extract_frozen_worker(payload_path, tmp_path / "frozen_payload")
    sys.path.insert(0, str(worker_path.parent))
    try:
        worker = _load_module("h232_r7_frozen_profile_worker", worker_path)
    finally:
        sys.path.remove(str(worker_path.parent))
    fold_manifest = HERE / "reference/H232_R6_FOLD0_ASSETS_MANIFEST_R1.json"
    assert (
        _sha256(worker_path)
        == builder.read_json(builder.R6_BUILD_RECEIPT)["payload_member_sha256"]["profile_worker.py"]
    )
    assert _sha256(fold_manifest) == worker.ASSETS_SHA256
    assert fold_manifest.stat().st_size == worker.ASSETS_BYTES

    r6_upload = R6_ROOT / "asset_upload_v1"
    r7_upload = PACKAGE_ROOT / str(receipt["asset_upload_path"])
    r6_asset = builder.read_json(builder.R6_BUILD_RECEIPT)["asset_filename"]
    r7_asset = str(receipt["asset_filename"])

    def make_mounts(parent: Path, *, use_r7_name: bool) -> tuple[Path, Path]:
        input_root = parent / "input"
        _mount_directory(input_root, "wunder-funder")
        fold_mount = _mount_directory(input_root, "wnn-c20261004-fold0-assets")
        shutil.copyfile(fold_manifest, fold_mount / "assets-manifest.json")
        payload_slug = (
            builder.ASSET_DATASET_ID.split("/")[1]
            if use_r7_name
            else "h232-v7b-ep3-payload-r6-dda776b3f9b7"
        )
        payload_mount = _mount_directory(input_root, payload_slug)
        if use_r7_name:
            shutil.copyfile(
                r7_upload / builder.ASSET_MANIFEST_FILENAME,
                payload_mount / builder.ASSET_MANIFEST_FILENAME,
            )
            asset_source = r7_upload / r7_asset
            if not asset_source.is_file():
                asset_source = payload_path
            (payload_mount / r7_asset).symlink_to(asset_source)
        else:
            shutil.copyfile(
                r6_upload / "assets-manifest.json", payload_mount / "assets-manifest.json"
            )
            r6_asset_path = R6_ROOT / "r6_payload.wnn"
            (payload_mount / str(r6_asset)).symlink_to(r6_asset_path)
        return input_root, fold_mount

    old_root, _ = make_mounts(tmp_path / "r6_layout", use_r7_name=False)
    old_error_receipt = tmp_path / "r6_identity.json"
    with pytest.raises(worker.ProfileAbort, match="identity mismatch for assets-manifest.json"):
        worker.verify_and_stage_inputs(
            old_root,
            tmp_path / "r6_staged",
            old_error_receipt,
            worker_path.parent / "prior_input_identity_v2.json",
        )
    duplicate_identity = json.loads(old_error_receipt.read_text())
    assert "found 2" in duplicate_identity["observed"]["assets-manifest.json"]["diagnostic"]

    new_root, expected_fold_mount = make_mounts(tmp_path / "r7_layout", use_r7_name=True)
    r7_identity_receipt = tmp_path / "r7_identity.json"
    selected_manifest_paths: list[Path] = []
    original_hash = worker.sha256_file
    original_persist = worker.persist_identity_check

    class StopAfterFoldManifestIdentity(Exception):
        """Stop after proving which manifest passed the unchanged worker lookup."""

    def record_hash(path: Path) -> str:
        selected_manifest_paths.append(path.resolve())
        return original_hash(path)

    def persist_then_stop(
        path: Path,
        identity: dict[str, Any],
        logical_name: str,
        expected: dict[str, Any],
        observed: dict[str, Any],
    ) -> None:
        original_persist(path, identity, logical_name, expected, observed)
        if logical_name == "assets-manifest.json":
            raise StopAfterFoldManifestIdentity

    monkeypatch.setattr(worker, "sha256_file", record_hash)
    monkeypatch.setattr(worker, "persist_identity_check", persist_then_stop)
    with pytest.raises(StopAfterFoldManifestIdentity):
        worker.verify_and_stage_inputs(
            new_root,
            tmp_path / "r7_staged",
            r7_identity_receipt,
            worker_path.parent / "prior_input_identity_v2.json",
        )

    assert selected_manifest_paths == [(expected_fold_mount / "assets-manifest.json").resolve()]
    selected_identity = json.loads(r7_identity_receipt.read_text())
    assert selected_identity["observed"]["assets-manifest.json"] == {
        "bytes": worker.ASSETS_BYTES,
        "sha256": worker.ASSETS_SHA256,
    }
    assert json.loads((expected_fold_mount / "assets-manifest.json").read_text())["dataset"] == (
        "vladkochetov07/wnn-c20261004-fold0-assets"
    )


def test_r7_bootstrap_opens_its_manifest_as_a_payload_sibling(
    r7_artifact: tuple[dict[str, Any], Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    """The bootstrap uses a scoped filename even when another assets manifest mounts."""
    receipt, _, _ = r7_artifact
    packet = PACKAGE_ROOT / str(receipt["kernel_packet_path"])
    module = _load_module("h232_r7_bootstrap_test", packet / "kernel.py")
    input_root = tmp_path / "input"
    fold_mount = _mount_directory(input_root, "wnn-c20261004-fold0-assets")
    (fold_mount / "assets-manifest.json").write_text("{}\n")
    payload_mount = _mount_directory(input_root, builder.ASSET_DATASET_ID.split("/")[1])
    asset_upload = PACKAGE_ROOT / str(receipt["asset_upload_path"])
    shutil.copyfile(
        asset_upload / builder.ASSET_MANIFEST_FILENAME,
        payload_mount / builder.ASSET_MANIFEST_FILENAME,
    )
    asset = payload_mount / str(receipt["asset_filename"])
    asset_source = asset_upload / str(receipt["asset_filename"])
    if not asset_source.is_file():
        asset_source = PACKAGE_ROOT / str(receipt["r6_payload_path"])
    asset.symlink_to(asset_source)

    found = module.find_verified_asset(
        input_root,
        builder.ASSET_DATASET_ID,
        str(receipt["asset_filename"]),
        int(receipt["asset_bytes"]),
        str(receipt["asset_sha256"]),
    )
    assert found.resolve() == asset.resolve()
    assert module.ASSET_DATASET_VERSION == 1
    assert receipt["payload_member_sha256"] == module.PAYLOAD_MEMBER_SHA256
