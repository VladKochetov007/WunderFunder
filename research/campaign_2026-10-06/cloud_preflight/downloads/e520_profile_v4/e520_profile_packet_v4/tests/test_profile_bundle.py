"""Synthetic verification for the deterministic single-file Kaggle wrapper."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import zipfile

import pytest
from prepare_profile_packet import (
    BUNDLE_PAYLOAD_FILES,
    REQUIRED_REVIEWED,
    render_profile_kernel_bundle,
)
from profile_worker import REVIEWED_PACKET_FILES


def test_bundle_members_cover_every_worker_review_gate() -> None:
    assert set(REVIEWED_PACKET_FILES) == set(REQUIRED_REVIEWED)
    assert set(REVIEWED_PACKET_FILES).issubset(BUNDLE_PAYLOAD_FILES)
    assert {"external_review.json", "kernel-metadata.json", "packet_meta.json"}.issubset(
        BUNDLE_PAYLOAD_FILES
    )


def test_profile_bundle_is_deterministic_self_contained_and_hash_checked() -> None:
    payloads = {
        "profile_worker.py": b"def run_profile(root): return {'status': 'synthetic'}\n",
        "profile_kernel.py": b"from profile_worker import run_profile\n",
    }
    source = render_profile_kernel_bundle(payloads)
    assert source == render_profile_kernel_bundle(dict(reversed(tuple(payloads.items()))))
    assert 'Path("/kaggle/working/e520_profile_packet_v4")' in source
    assert "from profile_worker import run_profile" in source

    assignments = {
        item.targets[0].id: ast.literal_eval(item.value)
        for item in ast.parse(source).body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id in {"PAYLOAD_B64", "PAYLOAD_FILE_SHA256", "PAYLOAD_ZIP_SHA256"}
    }
    archive_bytes = base64.b64decode(assignments["PAYLOAD_B64"], validate=True)
    assert hashlib.sha256(archive_bytes).hexdigest() == assignments["PAYLOAD_ZIP_SHA256"]
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == sorted(payloads)
        embedded = {name: archive.read(name) for name in archive.namelist()}
    assert embedded == payloads
    assert assignments["PAYLOAD_FILE_SHA256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(payloads.items())
    }

    with pytest.raises(ValueError, match="unsafe profile bundle path"):
        render_profile_kernel_bundle({"../escape.py": b"invalid"})
