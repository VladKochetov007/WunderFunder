from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

PACKET = Path(__file__).resolve().parent / "r5_packet"


@pytest.fixture
def packet(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(PACKET))
    worker = importlib.import_module("worker")
    monkeypatch.setattr(worker, "PACKET", PACKET)
    return worker


def test_sealed_authority_refuses_pending_before_any_fit(packet) -> None:
    with pytest.raises(PermissionError, match="pending"):
        packet.verify_authority()


def test_actual_successful_profile_producer_passes_unchanged_consumer(packet) -> None:
    module = importlib.import_module("capacity_fit_worker")
    evidence = module._verify_profile_evidence(PACKET)
    assert set(evidence["profiles"]) == {"muon512", "muon768"}


def test_actual_source_extract_and_six_file_binding(
    packet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "inputs"
    staged.mkdir()
    spec = json.loads((PACKET / "worker_spec.json").read_text())
    identities = {}
    for logical in spec["inputs"]:
        path = staged / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        content = logical.encode()
        path.write_bytes(content)
        identities[logical] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    monkeypatch.setenv("WNN_INPUT_DIR", str(staged))
    output = tmp_path / "outputs"
    output.mkdir()
    project, source_hashes, receipt = packet.prepare_project(output, {"inputs": identities})
    assert len(source_hashes) == 56
    assert receipt == project / "input_identity_receipt.json"
    binding = packet.runtime.validate_shared_input_binding(
        project,
        Path(__import__("os").environ["WNN_INPUT_BINDING_PATH"]),
        __import__("os").environ["WNN_INPUT_BINDING_SHA256"],
    )
    assert len(binding["sources"]) == 6
    (staged / "cache/valid_targets.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        packet.runtime.validate_shared_input_binding(
            project,
            Path(__import__("os").environ["WNN_INPUT_BINDING_PATH"]),
            __import__("os").environ["WNN_INPUT_BINDING_SHA256"],
        )


def test_missing_staged_cache_fails_before_child_launch(
    packet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "inputs"
    staged.mkdir()
    monkeypatch.setenv("WNN_INPUT_DIR", str(staged))
    output = tmp_path / "outputs"
    output.mkdir()
    with pytest.raises(ValueError, match="regular staged"):
        packet.prepare_project(
            output, {"inputs": {"cache/train_targets.npz": {"bytes": 1, "sha256": "0" * 64}}}
        )
