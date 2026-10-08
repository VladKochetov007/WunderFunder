from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1] / "packet"
ROOT = PACKET.parent
sys.path.insert(0, str(PACKET))

from profile_worker import REVIEWED_PACKET_FILES, validate_external_review  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_review_preflight_binds_inputs_before_provider_use() -> None:
    """Validate the actual review receipt without importing Torch or Parquet."""
    receipt_path = PACKET / "external_review.json"
    receipt_bytes = receipt_path.read_bytes()
    meta = json.loads((PACKET / "packet_meta.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(receipt_bytes).hexdigest()

    assert digest == meta["external_review_sha256"]
    loaded_before_validation = set(sys.modules)
    receipt = validate_external_review(receipt_bytes, digest, "h232_cloud_operator")
    imported_by_preflight = set(sys.modules) - loaded_before_validation
    assert receipt["reviewed_hashes"] == meta["reviewed_hashes"]
    assert set(REVIEWED_PACKET_FILES) == set(meta["reviewed_hashes"])
    for name in REVIEWED_PACKET_FILES:
        assert _sha256(PACKET / name) == meta["reviewed_hashes"][name]

    root_auth_path = ROOT / "V7B_ROOT_DELIVERY_AUTHORIZATION_20261008_R1.json"
    root_auth = json.loads(root_auth_path.read_text(encoding="utf-8"))
    approval = receipt["root_approval"]
    assert approval["v7b_delivery_authorization_sha256"] == _sha256(root_auth_path)
    assert root_auth["provider_push_authorized"] is False
    assert approval["provider_push_authorized"] is False
    assert receipt["qualified_independent_review"]["focused_tests_passed"] == 24

    build_receipt = json.loads((ROOT / "V7B_BUILD_RECEIPT.json").read_text(encoding="utf-8"))
    test_receipt_path = ROOT / "V7B_TEST_RECEIPT.json"
    test_receipt = json.loads(test_receipt_path.read_text(encoding="utf-8"))
    assert build_receipt["test_results_receipt_sha256"] == _sha256(test_receipt_path)
    assert test_receipt["focused_suite"]["status"] == "passed"
    assert test_receipt["focused_suite"]["passed"] == 25
    assert test_receipt["isolated_review_preflight"]["status"] == "passed"
    assert len(build_receipt["packet_files_14"]) == 14
    assert build_receipt["provider_push_authorized"] is False
    assert build_receipt["status"] == "built_pending_fresh_root_push_admission"
    for name, expected in build_receipt["packet_files_14"].items():
        assert _sha256(PACKET / name) == expected

    assert "torch" not in imported_by_preflight
    assert not any(
        name == "pyarrow" or name.startswith("pyarrow.") for name in imported_by_preflight
    )
