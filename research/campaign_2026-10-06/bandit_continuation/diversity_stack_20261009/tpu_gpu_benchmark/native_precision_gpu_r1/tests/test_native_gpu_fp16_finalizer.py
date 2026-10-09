"""Verify root-admission finalization against the actual native worker consumer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "native_fp16_admission_finalizer", PACKAGE / "finalize_native_gpu_fp16_admission.py"
)
assert spec is not None and spec.loader is not None
finalizer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = finalizer
spec.loader.exec_module(finalizer)


def _synthetic_admission(*, source_manifest_sha256: str) -> dict[str, object]:
    return {
        "schema": "wnn.native-gpu-fp16.root-admission.v1",
        "state": "admitted",
        "source_manifest_sha256": source_manifest_sha256,
        "r5_training_contract_sha256": finalizer.R5_CONTRACT_SHA256,
        "r5_root_review_sha256": finalizer.R5_REVIEW_SHA256,
        "compute_precision": "fp16",
        "amp_initial_scale": 1024.0,
        "push_allowed": True,
        "profile_allowed": True,
        "conditional_fit_allowed": True,
        "training_contract_sha256": finalizer.R5_CONTRACT_SHA256,
        "test_only_synthetic_receipt": True,
    }


def _load_generated_kernel(path: Path):
    generated_spec = importlib.util.spec_from_file_location("finalized_native_bootstrap", path)
    assert generated_spec is not None and generated_spec.loader is not None
    module = importlib.util.module_from_spec(generated_spec)
    sys.modules[generated_spec.name] = module
    generated_spec.loader.exec_module(module)
    return module


def test_finalizer_emits_actual_worker_consumable_copy_and_preserves_pending(tmp_path: Path):
    pending_packet = PACKAGE / "native_gpu_fp16.packet.json"
    pending_manifest = PACKAGE / "source_asset/assets-manifest.json"
    packet_bytes_before = pending_packet.read_bytes()
    manifest_bytes_before = pending_manifest.read_bytes()
    pending_packet_value = json.loads(packet_bytes_before)
    source_sha = pending_packet_value["source_manifest_sha256"]
    admission = _synthetic_admission(source_manifest_sha256=source_sha)
    admission_path = tmp_path / "synthetic-root-admission.json"
    admission_path.write_bytes(finalizer.canonical(admission))
    output = tmp_path / "finalized"

    receipt = finalizer.finalize(
        pending_package=PACKAGE,
        admission_path=admission_path,
        output_package=output,
    )

    assert pending_packet.read_bytes() == packet_bytes_before
    assert pending_manifest.read_bytes() == manifest_bytes_before
    assert receipt["state"] == "finalized_pending_provider_action"
    assert receipt["device_fit"] is False
    assert receipt["provider_mutation"] is False

    packet = json.loads((output / "native_gpu_fp16.packet.json").read_bytes())
    assert packet["source_commit"] == pending_packet_value["source_commit"]
    assert packet["authorized_for_execution"] is True
    assert packet["root_admission"] == admission
    assert packet["root_admission_required_before_asset_upload_or_kernel_push"] is False
    packet_without_self_pin = dict(packet)
    self_pin = packet_without_self_pin.pop("packet_sha256_before_self_pin")
    assert self_pin == finalizer.sha256_bytes(finalizer.canonical(packet_without_self_pin))
    asset_manifest_path = output / "source_asset/assets-manifest.json"
    asset_manifest = json.loads(asset_manifest_path.read_bytes())
    assert finalizer.sha256_file(asset_manifest_path) == packet["source_asset_manifest_sha256"]
    rows_by_path = {row["path"]: row for row in asset_manifest["files"]}
    inventory_by_path = {row["path"]: row for row in packet["source_asset_upload_inventory"]}
    assert len(inventory_by_path) == len(packet["source_asset_upload_inventory"])
    assert set(inventory_by_path) == set(rows_by_path) | {
        "assets-manifest.json",
        "dataset-metadata.json",
    }
    for path, row in rows_by_path.items():
        inventory_row = inventory_by_path[path]
        assert inventory_row["filename"] == row["filename"]
        assert inventory_row["sha256"] == row["sha256"]
        assert inventory_row["size_bytes"] == row["size"]
    assert {path.name for path in (output / "source_asset").iterdir()} == {
        row["filename"] for row in asset_manifest["files"]
    } | {
        "assets-manifest.json",
        "dataset-metadata.json",
    }
    auth_rows = [
        row
        for row in asset_manifest["files"]
        if row["path"] == "native_controls/root_admission.json"
    ]
    assert len(auth_rows) == 1
    auth_path = output / "source_asset" / auth_rows[0]["filename"]
    assert auth_path.read_bytes() == admission_path.read_bytes()

    kernel_path = output / "kernel" / packet["kernel"]["code_file"]
    assert finalizer.sha256_file(kernel_path) == packet["kernel"]["code_file_sha256"]
    kernel = _load_generated_kernel(kernel_path)
    kernel.INPUT = output / "source_asset"
    extraction_root = tmp_path / "runtime"
    kernel.REPO = extraction_root / "wnn"
    native_root, discovered = kernel.discover_native_asset_manifest()
    kernel.unpack_native_overlay(native_root, discovered)
    scripts = kernel.REPO / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        worker_spec = importlib.util.spec_from_file_location(
            "finalized_native_worker", scripts / "native_gpu_fp16_worker.py"
        )
        assert worker_spec is not None and worker_spec.loader is not None
        worker = importlib.util.module_from_spec(worker_spec)
        sys.modules[worker_spec.name] = worker
        worker_spec.loader.exec_module(worker)
        actual_admission, _, controls_root = worker._validate_admission(
            {
                "native_root_admission_path": str(
                    kernel.REPO / "native_controls/root_admission.json"
                ),
                "native_controls_root": str(kernel.REPO / "native_controls"),
                "native_source_manifest_sha256": source_sha,
            }
        )
        assert actual_admission == admission
        assert controls_root == kernel.REPO / "native_controls"
    finally:
        sys.path.remove(str(scripts))


def test_finalizer_rejects_wrong_source_binding_before_creating_output(tmp_path: Path):
    packet = json.loads((PACKAGE / "native_gpu_fp16.packet.json").read_bytes())
    admission = _synthetic_admission(source_manifest_sha256="0" * 64)
    admission_path = tmp_path / "wrong-root-admission.json"
    admission_path.write_bytes(finalizer.canonical(admission))
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="mismatches a required packet pin"):
        finalizer.finalize(
            pending_package=PACKAGE,
            admission_path=admission_path,
            output_package=output,
        )

    assert packet["root_admission"]["state"] == "pending"
    assert not output.exists()
