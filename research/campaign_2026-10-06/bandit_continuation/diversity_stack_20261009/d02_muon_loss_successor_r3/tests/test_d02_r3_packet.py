from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "gpu_loss_pair.packet.json"
R5_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
R5_CONTRACT_SHA = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_packet_is_pending_and_binds_reviewed_r5_controls() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    admission = json.loads((ROOT / "root_admission.json").read_text(encoding="utf-8"))
    parent = packet["r5_control_parent"]
    kernel_path = ROOT / "kernel" / packet["kernel"]["code_file"]
    metadata_path = ROOT / "kernel" / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    kernel_text = kernel_path.read_text(encoding="utf-8")

    assert packet["schema"] == "wnn.d02.kaggle-muon-loss-packet.v3"
    assert packet["packet_id"] == "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v3"
    assert packet["control_gpu_kernel"] == R5_KERNEL
    assert packet["authorized_for_execution"] is False
    assert admission["push_allowed"] is False
    assert admission["conditional_fit_allowed"] is False
    assert admission["gpu_control_contract_sha256"] == R5_CONTRACT_SHA
    assert admission["control_gpu_kernel"] == R5_KERNEL
    assert admission["source_code_manifest_sha256"] == packet["source"]["code_manifest_sha256"]
    assert metadata["id"] == packet["packet_id"]
    assert metadata["title"] == packet["kernel"]["title"]
    assert metadata["kernel_sources"] == [
        "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1",
        R5_KERNEL,
    ]
    assert "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v3" in kernel_text
    assert R5_KERNEL in kernel_text
    assert R5_CONTRACT_SHA in kernel_text
    assert parent["training_contract"]["sha256"] == R5_CONTRACT_SHA
    assert parent["training_contract"]["path_under_kernel_output"] == "out/training_contract.json"
    assert parent["root_terminal_review"]["sha256"] == (
        "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
    )
    assert parent["operator_terminal_receipt"]["files_verified"] == 149
    assert parent["fixed_grade"]["evaluation_groups"] == 468
    assert parent["fixed_grade"]["components"] == 141
    assert parent["fixed_grade"]["wmse_wp"] == 0.6060019230032425
    assert parent["fixed_grade"]["huber_d1_wp"] == 0.6012391352737978
    assert parent["fixed_grade"]["equal_blend_wp"] == 0.6054802014069163
    assert packet["experiment"]["control_basis"]["science_recipe_change"] is False


def test_source_asset_contains_exact_r3_worker_and_reader() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    manifest_path = ROOT / "source_asset" / "assets-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    assert _sha256(manifest_raw) == packet["source"]["asset_manifest_sha256"]
    asset_manifest = json.loads(manifest_raw)
    entries = {row["path"]: row for row in asset_manifest["files"]}
    source_manifest_path = ROOT / "source_asset" / entries["source-code-manifest.json"]["filename"]
    source_manifest_raw = source_manifest_path.read_bytes()
    assert _sha256(source_manifest_raw) == packet["source"]["code_manifest_sha256"]
    source_manifest = json.loads(source_manifest_raw)
    archive_path = ROOT / "source_asset" / entries["wnn_loss_pair_src.tar.gz"]["filename"]
    with tarfile.open(archive_path, "r:gz") as archive:
        reader = archive.extractfile("src/wnn/parsed_sequence_cache_reader.py")
        worker = archive.extractfile("scripts/loss_pair_gpu_worker.py")
        controls = archive.extractfile("src/wnn/d02_controls.py")
        assert reader is not None and worker is not None and controls is not None
        reader_bytes = reader.read()
        worker_text = worker.read().decode("utf-8")
        controls_text = controls.read().decode("utf-8")

    source_rows = {row["path"]: row for row in source_manifest["files"]}
    assert _sha256(reader_bytes) == "35431e48583b1523188b3d6084fa12be23e7379560a323411655bf561507a693"
    assert source_rows["src/wnn/parsed_sequence_cache_reader.py"]["sha256"] == _sha256(reader_bytes)
    assert source_rows["scripts/loss_pair_gpu_worker.py"]["sha256"] == _sha256(worker_text.encode())
    assert source_rows["src/wnn/d02_controls.py"]["sha256"] == _sha256(controls_text.encode())
    assert "wnn-d02-muon-loss-code-20261009-r3" in worker_text
    assert "reviewed matched R5 AdamW producer" in controls_text
    assert "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7" in controls_text


def test_r3_identity_and_fit_cells_match_frozen_r2_recipe() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    packet_experiment = dict(packet["experiment"])
    packet_experiment.pop("control_basis")
    reference_packet = json.loads(
        (ROOT / "tests" / "fixtures" / "d02_r2_packet.json").read_text(encoding="utf-8")
    )
    assert packet_experiment == reference_packet["experiment"]

    recipe = json.loads((ROOT / "exact_argv_recipe_r3.json").read_text(encoding="utf-8"))
    reference_path = ROOT / "tests" / "fixtures" / "d02_r2_exact_argv_recipe.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    assert recipe["controls"]["kernel"] == R5_KERNEL
    assert recipe["controls"]["completed_training_contract_sha256"] == R5_CONTRACT_SHA
    assert recipe["controls"]["retrain"] is False
    for key in (
        "cells",
        "membership_sha256",
        "fit_count",
        "unused_fit_valid_count",
        "evaluation_count",
        "attempts_per_cell",
        "planned_new_fit_count",
        "nominal_lr_change_from_adamw",
        "muon_parameter_policy",
    ):
        assert recipe[key] == reference[key], key
