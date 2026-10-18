"""Cover interrupted evidence packaging without replacing any prior archive."""

from __future__ import annotations

import hashlib
import json
import tarfile

import package_evidence
import pytest


def _fixture(tmp_path):
    root = tmp_path / "project"
    run = root / "runs" / "cell-a"
    artifact = run / "artifacts"
    artifact.mkdir(parents=True)
    evidence = run / "producer_receipt.json"
    evidence.write_text('{"status":"completed"}\n', encoding="utf-8")
    weight = artifact / "model_ep5.pt"
    weight.write_bytes(b"frozen-checkpoint")
    row = {
        "id": "cell-a",
        "receipt_sha256": hashlib.sha256(b"grade").hexdigest(),
        "evidence_paths": [str(evidence)],
        "weights_paths": [str(weight)],
    }
    return root, row


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resumes_weights_after_evidence_archive_was_committed(tmp_path, monkeypatch) -> None:
    root, row = _fixture(tmp_path)
    create = package_evidence._create_archive

    def fail_weights(root_arg, directory, kind, paths):
        if kind == "weights":
            raise OSError("simulated disk interruption")
        return create(root_arg, directory, kind, paths)

    monkeypatch.setattr(package_evidence, "_create_archive", fail_weights)
    with pytest.raises(OSError, match="simulated"):
        package_evidence.ensure_archives(root, row)
    archive_root = next((root / "artifacts/github-reporting-v1").iterdir())
    evidence_archive = archive_root / "evidence.tar"
    evidence_digest = _sha(evidence_archive)
    assert (archive_root / "evidence.json").is_file()
    assert not (archive_root / "MANIFEST.json").exists()

    monkeypatch.setattr(package_evidence, "_create_archive", create)
    result = package_evidence.ensure_archives(root, row)
    assert _sha(evidence_archive) == evidence_digest
    assert result["archive_manifest_path"].endswith("MANIFEST.json")
    assert (archive_root / "weights.json").is_file()


def test_reconciles_archive_committed_before_kind_manifest(tmp_path, monkeypatch) -> None:
    root, row = _fixture(tmp_path)
    publish = package_evidence._publish_json_noreplace
    failed = False

    def interrupt(path, value):
        nonlocal failed
        if path.name == "evidence.json" and not failed:
            failed = True
            raise OSError("simulated receipt-write interruption")
        publish(path, value)

    monkeypatch.setattr(package_evidence, "_publish_json_noreplace", interrupt)
    with pytest.raises(OSError, match="receipt-write"):
        package_evidence.ensure_archives(root, row)
    archive_root = next((root / "artifacts/github-reporting-v1").iterdir())
    evidence_archive = archive_root / "evidence.tar"
    evidence_digest = _sha(evidence_archive)
    assert not (archive_root / "evidence.json").exists()

    monkeypatch.setattr(package_evidence, "_publish_json_noreplace", publish)
    package_evidence.ensure_archives(root, row)
    assert _sha(evidence_archive) == evidence_digest
    with tarfile.open(evidence_archive, "r") as archive:
        assert archive.getnames() == ["runs/cell-a/producer_receipt.json"]
    assert json.loads((archive_root / "evidence.json").read_text())["archive"]["sha256"]


def test_partial_tar_is_preserved_and_never_replaced(tmp_path) -> None:
    root, row = _fixture(tmp_path)
    identity = hashlib.sha256(
        json.dumps(
            {"id": row["id"], "receipt_sha256": row["receipt_sha256"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    directory = root / "artifacts/github-reporting-v1" / identity
    directory.mkdir(parents=True)
    partial = directory / "evidence.partial.tar"
    partial.write_bytes(b"retain-for-manual-review")

    with pytest.raises(FileExistsError, match="Partial archive"):
        package_evidence.ensure_archives(root, row)
    assert partial.read_bytes() == b"retain-for-manual-review"
