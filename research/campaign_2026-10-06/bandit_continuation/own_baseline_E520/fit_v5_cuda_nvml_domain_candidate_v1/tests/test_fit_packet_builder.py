"""Tests for deterministic transport and review-set coverage in the fit builder."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from fit_guards import REQUIRED_ARTIFACTS
from prepare_fit_packet import (
    ADMISSION_NAME,
    BUNDLE_PAYLOAD_FILES,
    KAGGLE_TITLE,
    META_NAME,
    PACKET_ID,
    REVIEW_NAME,
    render_fit_kernel_bundle,
)


def test_fit_bundle_covers_reviewed_worker_and_both_external_receipts() -> None:
    assert set(REQUIRED_ARTIFACTS).issubset(BUNDLE_PAYLOAD_FILES)
    assert {
        "E520_FIT_STAGE_PLAN.json",
        "tests/test_training_target_semantics.py",
        "support/verified_digest_handoff.py",
        "tests/test_digest_handoff.py",
    }.issubset(REQUIRED_ARTIFACTS)
    assert {ADMISSION_NAME, META_NAME, REVIEW_NAME}.issubset(BUNDLE_PAYLOAD_FILES)


def test_fit_packet_metadata_matches_the_new_candidate_identity() -> None:
    packet_root = Path(__file__).resolve().parents[1]
    config = json.loads((packet_root / "FIT_STAGE_CONFIG.json").read_text())
    assert config["packet_id"] == PACKET_ID
    author, slug = PACKET_ID.split("/", maxsplit=1)
    assert author == "vladkochetov07"
    assert KAGGLE_TITLE.lower().replace(" ", "-") == slug


def test_fit_bundle_is_deterministic_and_embeds_exact_hash_map() -> None:
    payloads = {name: f"synthetic:{name}".encode() for name in BUNDLE_PAYLOAD_FILES}
    source = render_fit_kernel_bundle(payloads)
    assert source == render_fit_kernel_bundle(dict(reversed(tuple(payloads.items()))))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"PAYLOAD_B64", "PAYLOAD_FILE_SHA256", "PAYLOAD_ZIP_SHA256"}
    }
    archive_bytes = base64.b64decode(assignments["PAYLOAD_B64"], validate=True)
    assert hashlib.sha256(archive_bytes).hexdigest() == assignments["PAYLOAD_ZIP_SHA256"]
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == sorted(payloads)
        assert {name: archive.read(name) for name in archive.namelist()} == payloads
    assert assignments["PAYLOAD_FILE_SHA256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(payloads.items())
    }


def test_fit_bundle_rejects_missing_members_and_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="payload mismatch"):
        render_fit_kernel_bundle({"fit_worker.py": b"x"})
    payloads = {name: b"x" for name in BUNDLE_PAYLOAD_FILES}
    payloads["../escape.py"] = payloads.pop(BUNDLE_PAYLOAD_FILES[0])
    with pytest.raises(ValueError, match="unsafe E520 bundle path"):
        render_fit_kernel_bundle(payloads)
