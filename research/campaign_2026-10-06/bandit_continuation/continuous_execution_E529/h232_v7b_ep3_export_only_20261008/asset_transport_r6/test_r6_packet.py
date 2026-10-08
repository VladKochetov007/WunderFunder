"""Focused offline tests for the corrected H232 R6 private asset transport."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_r6_packet as builder  # noqa: E402


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a packet module by path without running its CLI entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r6_artifact() -> tuple[dict[str, object], Path, dict[str, str]]:
    """Load the locally built R6 package and its exact member hash map."""
    receipt_path = HERE / "H232_V7B_EP3_ASSET_TRANSPORT_R6_BUILD_RECEIPT_R1.json"
    if not receipt_path.is_file():
        pytest.skip("R6 packet has not been built")
    receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
    payload_path = PACKAGE_ROOT / receipt["payload_path"]
    members = cast(dict[str, str], receipt["payload_member_sha256"])
    return receipt, payload_path, members


def test_r6_keeps_all_r5_members_and_adds_exact_parent_metadata(
    r6_artifact: tuple[dict[str, object], Path, dict[str, str]],
) -> None:
    receipt, payload_path, member_hashes = r6_artifact
    r5 = builder.read_json(builder.R5_BUILD_RECEIPT)
    r5_archive = builder.R5_ROOT / "asset_upload_v1" / r5["asset_file"]

    assert receipt["payload_member_count"] == 18
    assert len(member_hashes) == 18
    assert member_hashes["parent_packet_meta.json"] == builder.EXPECTED_PARENT_METADATA_SHA256
    assert (HERE / "basis/parent_packet_meta.json").stat().st_size == 4_204
    assert member_hashes["parent_packet_meta.json"] == (
        "97e01d0c09a4e7dd67c13da6757e6f5bc72e2906c4232ec52415af8fdbc95e20"
    )
    assert builder.sha256_file(payload_path) == receipt["payload_zip_sha256"]
    assert builder.sha256_file(r5_archive) == receipt["r5_payload_sha256"]
    assert receipt["export_worker_sha256"] == r5["payload_member_sha256"]["export_worker.py"]
    assert receipt["checkpoint_sha256"] == r5["payload_member_sha256"]["model_ep3.pt"]
    for name, digest in r5["payload_member_sha256"].items():
        assert member_hashes[name] == digest


def test_actual_full_r6_extraction_passes_unchanged_worker_payload_verifier(
    r6_artifact: tuple[dict[str, object], Path, dict[str, str]],
) -> None:
    receipt, payload_path, member_hashes = r6_artifact
    bootstrap_path = PACKAGE_ROOT / receipt["kernel_packet_path"] / "kernel.py"
    bootstrap = _load_module("h232_r6_bootstrap_test", bootstrap_path)
    assert bootstrap.KERNEL_ID == builder.KERNEL_ID
    assert bootstrap.KERNEL_TITLE == builder.KERNEL_TITLE
    assert len(bootstrap.PAYLOAD_MEMBER_SHA256) == 18

    with tempfile.TemporaryDirectory(prefix="h232-r6-verify-") as temporary:
        extracted = Path(temporary) / "payload"
        observed = bootstrap.extract_verified_payload(
            payload_path,
            extracted,
            receipt["payload_zip_bytes"],
            receipt["payload_zip_sha256"],
            member_hashes,
        )
        assert observed == member_hashes
        packet_meta = json.loads((extracted / "packet_meta.json").read_text())
        inner_hashes = cast(dict[str, str], packet_meta["payload_file_sha256"])
        extracted_files = {path.name for path in extracted.iterdir() if path.is_file()}
        assert set(inner_hashes) == extracted_files - {"packet_meta.json"}
        parent_metadata = extracted / "parent_packet_meta.json"
        assert parent_metadata.stat().st_size == 4_204
        assert parent_metadata.read_bytes() == (HERE / "basis/parent_packet_meta.json").read_bytes()
        for name, expected in inner_hashes.items():
            assert builder.sha256_file(extracted / name) == expected

        before_torch = "torch" in sys.modules
        sys.path.insert(0, str(extracted))
        try:
            worker = _load_module("h232_r6_export_worker_test", extracted / "export_worker.py")
            assert packet_meta["packet_id"] == worker.PACKET_ID
            worker._verify_payload(extracted, packet_meta)
        finally:
            sys.path.remove(str(extracted))
        assert ("torch" in sys.modules) is before_torch


def test_r6_kernel_metadata_and_environment_precede_export_worker_import(
    r6_artifact: tuple[dict[str, object], Path, dict[str, str]],
) -> None:
    receipt, _, _ = r6_artifact
    kernel_packet = PACKAGE_ROOT / receipt["kernel_packet_path"]
    source_path = kernel_packet / "kernel.py"
    metadata = json.loads((kernel_packet / "kernel-metadata.json").read_text())
    source = source_path.read_text()
    tree = ast.parse(source)
    import_line = source.index("from export_worker import main as run_export")

    assert builder.sha256_file(source_path) == receipt["kernel_source_sha256"]
    assert source_path.stat().st_size < 1_000_000
    assert metadata["id"] == builder.KERNEL_ID
    assert metadata["title"] == builder.KERNEL_TITLE
    assert metadata["dataset_sources"][-1] == receipt["asset_dataset_version_pin"]
    builder.validate_kernel_identity(metadata["id"], metadata["title"])
    assert len(tree.body) > 0
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "0",
        "WNN_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }.items():
        assert source.index(f'os.environ["{name}"] = "{value}"') < import_line


def test_small_archive_fixture_rejects_a_parent_metadata_hash_mismatch(tmp_path: Path) -> None:
    r5 = builder.read_json(builder.R5_BUILD_RECEIPT)
    r5_archive = builder.R5_ROOT / "asset_upload_v1" / r5["asset_file"]
    parent = tmp_path / "parent_packet_meta.json"
    parent.write_text('{"fixture":true}\n')
    with pytest.raises(ValueError, match="frozen parent packet metadata"):
        builder.repackage_payload(
            r5_archive,
            parent,
            r5["payload_zip_bytes"],
            r5["payload_zip_sha256"],
            r5["payload_member_sha256"],
        )
