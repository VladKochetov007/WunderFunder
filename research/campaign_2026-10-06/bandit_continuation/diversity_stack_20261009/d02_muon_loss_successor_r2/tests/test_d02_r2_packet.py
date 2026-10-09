from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = CANDIDATE_ROOT / "gpu_loss_pair.packet.json"
R4_READER_PATH = (
    CANDIDATE_ROOT.parent
    / "tpu_gpu_benchmark"
    / "native_device_comparison_r1"
    / "sealed_candidate_r2"
    / "offline_boot"
    / "tpu_source"
    / "src"
    / "wnn"
    / "parsed_sequence_cache_reader.py"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_packet_is_pending_and_binds_verified_r4_parent() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    admission = json.loads((CANDIDATE_ROOT / "root_admission.json").read_text(encoding="utf-8"))
    parent = packet["r4_control_parent"]
    kernel = CANDIDATE_ROOT / "kernel" / packet["kernel"]["code_file"]
    kernel_metadata_path = CANDIDATE_ROOT / "kernel" / "kernel-metadata.json"
    kernel_metadata = json.loads(kernel_metadata_path.read_text(encoding="utf-8"))
    kernel_text = kernel.read_text(encoding="utf-8")

    assert packet["schema"] == "wnn.d02.kaggle-muon-loss-packet.v2"
    assert packet["control_gpu_kernel"] == "vladkochetov07/wnn-loss-pair-gpu-20261009-v4"
    assert admission["push_allowed"] is False
    assert admission["conditional_fit_allowed"] is False
    assert admission["gpu_control_contract_sha256"] is None
    assert admission["source_code_manifest_sha256"] == packet["source"]["code_manifest_sha256"]
    assert admission["kernel_id"] == packet["packet_id"]
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == packet["kernel"]["code_file_sha256"]
    assert (
        hashlib.sha256(kernel_metadata_path.read_bytes()).hexdigest()
        == packet["kernel"]["metadata_sha256"]
    )
    assert kernel_metadata["id"] == packet["packet_id"]
    assert "vladkochetov07/wnn-loss-pair-gpu-20261009-v4" in kernel_metadata["kernel_sources"]
    assert "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v2" in kernel_text
    assert "vladkochetov07/wnn-loss-pair-gpu-20261009-v4" in kernel_text

    assert parent["state"] == "fits_verified_export_recovery_pending"
    assert parent["training_contract_sha256"] is None
    assert parent["control_contract_admitted"] is False
    assert (
        parent["completed_fits_review"]["sha256"]
        == "701987320ad045a4688e633cef19d59fb8b940cb1a75f53b089deaeb0a070e47"
    )
    assert parent["operator_terminal_collection_receipt_sha256"] == (
        "1bed2d7befbdd25b487f15f0b8788b8e514783058a2c7d5f683b31ff70a7500c"
    )
    assert parent["arms"]["wmse"]["checkpoint_sha256"] == (
        "71fcba3d9eaff3710f8fd42f3778ef74cd425f11977668a5575c21563a82c309"
    )
    assert parent["arms"]["huber_d1"]["checkpoint_sha256"] == (
        "b61b5026386273b11db871f06f09eda7654af5814c066e850274e32d9cdfca77"
    )
    assert all(
        row["prediction_export"] == "pending R5 export-only recovery"
        for row in parent["arms"].values()
    )


def test_source_archive_contains_exact_r4_reader_and_export_argv_repair() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    asset_manifest_path = CANDIDATE_ROOT / "source_asset" / "assets-manifest.json"
    asset_manifest_raw = asset_manifest_path.read_bytes()
    assert sha256(asset_manifest_raw) == packet["source"]["asset_manifest_sha256"]
    asset_manifest = json.loads(asset_manifest_raw)
    entries = {row["path"]: row for row in asset_manifest["files"]}
    source_manifest_path = (
        CANDIDATE_ROOT / "source_asset" / entries["source-code-manifest.json"]["filename"]
    )
    source_manifest_raw = source_manifest_path.read_bytes()
    assert sha256(source_manifest_raw) == packet["source"]["code_manifest_sha256"]
    source_manifest = json.loads(source_manifest_raw)

    archive_path = CANDIDATE_ROOT / "source_asset" / entries["wnn_loss_pair_src.tar.gz"]["filename"]
    with tarfile.open(archive_path, "r:gz") as archive:
        reader = archive.extractfile("src/wnn/parsed_sequence_cache_reader.py")
        worker = archive.extractfile("scripts/loss_pair_gpu_worker.py")
        assert reader is not None and worker is not None
        reader_bytes = reader.read()
        worker_text = worker.read().decode("utf-8")
    r4_reader = R4_READER_PATH.read_bytes()
    assert reader_bytes == r4_reader
    assert sha256(reader_bytes) == packet["r4_control_parent"]["reader_module_sha256"]
    export_start = worker_text.index("def _export_predictions(")
    export_end = worker_text.index("def _collect_prediction(", export_start)
    assert '"--device"' not in worker_text[export_start:export_end]
    assert "wnn-d02-muon-loss-code-20261009-r2" in worker_text
    source_rows = {row["path"]: row for row in source_manifest["files"]}
    assert source_rows["src/wnn/parsed_sequence_cache_reader.py"]["sha256"] == sha256(reader_bytes)
    assert source_rows["scripts/loss_pair_gpu_worker.py"]["sha256"] == sha256(worker_text.encode())
