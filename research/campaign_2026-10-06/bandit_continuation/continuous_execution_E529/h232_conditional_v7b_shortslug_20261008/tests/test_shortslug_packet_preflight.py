from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "packet"
PARENT = ROOT.parent / "h232_conditional_v7b_reviewed_20261008"
sys.path.insert(0, str(PACKET))

from profile_worker import REVIEWED_PACKET_FILES, validate_external_review  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shortslug_changes_only_kernel_metadata_and_keeps_actual_review_valid() -> None:
    parent_receipt = json.loads((PARENT / "V7B_BUILD_RECEIPT.json").read_text())
    receipt_path = PACKET / "external_review.json"
    receipt_bytes = receipt_path.read_bytes()
    review_digest = hashlib.sha256(receipt_bytes).hexdigest()
    packet_meta = json.loads((PACKET / "packet_meta.json").read_text())
    build_receipt = json.loads((ROOT / "V7B_SHORTSLUG_BUILD_RECEIPT.json").read_text())
    kernel_metadata = json.loads((PACKET / "kernel-metadata.json").read_text())
    basis_hashes = {
        path.name: _sha256(path) for path in sorted((ROOT / "basis").iterdir()) if path.is_file()
    }
    delivery_authorization_path = ROOT / "basis/V7B_ROOT_DELIVERY_AUTHORIZATION_20261008_R1.json"
    delivery_authorization = json.loads(delivery_authorization_path.read_text())
    prior_push_admission = json.loads(
        (ROOT / "basis/H232_V7B_ROOT_PUSH_ADMISSION_20261008_R1.json").read_text()
    )
    push_failure = json.loads(
        (ROOT / "basis/H232_V7B_PROVIDER_PUSH_FAILURE_20261008_R1.json").read_text()
    )

    assert build_receipt["external_review_exactly_preserved"] is True
    assert build_receipt["packet_meta_exactly_preserved"] is True
    assert build_receipt["embedded_wrapper_exactly_preserved"] is True
    assert build_receipt["builder_sha256"] == _sha256(
        ROOT / "prepare_h232_v7b_shortslug.py"
    )
    assert build_receipt["focused_preflight_test_sha256"] == _sha256(Path(__file__))
    assert build_receipt["basis_files_sha256"] == basis_hashes
    assert build_receipt["external_review_sha256"] == review_digest
    assert packet_meta["external_review_sha256"] == review_digest
    assert packet_meta["provider_push_authorized"] is False
    assert external_review_root_approval(receipt_bytes)["v7b_delivery_authorization_sha256"] == _sha256(
        delivery_authorization_path
    )
    assert delivery_authorization["provider_push_authorized"] is False
    assert prior_push_admission["canonical_expected"].endswith(
        "v7b-proactive-cache-recovery"
    )
    assert prior_push_admission["push_count_authorized"] == 1
    assert build_receipt["v7b_push_failure_receipt_sha256"] == _sha256(
        ROOT / "basis/H232_V7B_PROVIDER_PUSH_FAILURE_20261008_R1.json"
    )
    assert push_failure["http_status"] == 400
    assert push_failure["operation"] == "SaveKernel"
    assert push_failure["response_body_captured"] is False
    assert push_failure["accepted_execution_proven"] is False
    assert kernel_metadata["id"] == "vladkochetov07/h232-gru384-b8-amp4096-v7b-shortslug"
    assert kernel_metadata["title"] == "H232 GRU384 B8 AMP4096 V7B Shortslug"
    assert len(kernel_metadata["id"].split("/", maxsplit=1)[1]) < 50
    assert "-".join(kernel_metadata["title"].lower().split()) == kernel_metadata["id"].split(
        "/", maxsplit=1
    )[1]

    parent_hashes = parent_receipt["packet_files_14"]
    observed_hashes = {
        name: _sha256(PACKET / name) for name in sorted(build_receipt["packet_files_14"])
    }
    source_manifest = json.loads((PACKET / "source_archive_manifest.json").read_text())
    source_tree_hashes = {
        path.relative_to(ROOT / "source").as_posix(): _sha256(path)
        for path in sorted((ROOT / "source").rglob("*"))
        if path.is_file()
    }
    assert observed_hashes == build_receipt["packet_files_14"]
    assert source_tree_hashes == build_receipt["source_tree_members_sha256"]
    assert len(source_tree_hashes) == source_manifest["member_count"]
    for member in source_manifest["members"]:
        source_path = ROOT / "source" / member["path"]
        assert source_path.stat().st_size == member["bytes"]
        assert source_tree_hashes[member["path"]] == member["sha256"]
    assert set(observed_hashes) == set(REVIEWED_PACKET_FILES) | {
        "external_review.json",
        "h232_profile_kernel_bundle.py",
        "kernel-metadata.json",
        "packet_meta.json",
    }
    assert {
        name for name, digest in observed_hashes.items() if digest != parent_hashes[name]
    } == {"kernel-metadata.json"}
    assert observed_hashes["source_archive.tar"] == parent_hashes["source_archive.tar"]
    assert observed_hashes["source_archive_manifest.json"] == parent_hashes[
        "source_archive_manifest.json"
    ]

    loaded_before_validation = set(sys.modules)
    review = validate_external_review(receipt_bytes, review_digest, "h232_cloud_operator")
    imported_by_preflight = set(sys.modules) - loaded_before_validation
    assert review["reviewed_hashes"] == packet_meta["reviewed_hashes"]
    for name in REVIEWED_PACKET_FILES:
        assert _sha256(PACKET / name) == packet_meta["reviewed_hashes"][name]
    assert "torch" not in imported_by_preflight
    assert not any(
        name == "pyarrow" or name.startswith("pyarrow.") for name in imported_by_preflight
    )


def external_review_root_approval(receipt_bytes: bytes) -> dict[str, object]:
    """Return a typed view of the review receipt's root approval binding."""
    receipt = json.loads(receipt_bytes)
    approval = receipt.get("root_approval")
    assert isinstance(approval, dict)
    return approval
