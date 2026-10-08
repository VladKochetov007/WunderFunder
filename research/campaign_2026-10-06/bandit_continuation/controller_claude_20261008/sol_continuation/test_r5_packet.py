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


def test_terminal_failure_still_produces_exact_collection_manifest(
    packet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = importlib.import_module("terminal")
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "worker.log").write_text("preserved failure log\n")
    expected = ["artifact_bundle.tar", "result.json", "worker.log", "image_preflight.json"]
    monkeypatch.setenv("WNN_OUTPUT_DIR", str(output))
    monkeypatch.setenv("WNN_ATTEMPT_ID", "r5-fixture")
    monkeypatch.setenv("WNN_PACKET_SHA256", "a" * 64)
    monkeypatch.setenv("WNN_EXPECTED_ARTIFACTS", json.dumps(expected))
    monkeypatch.setenv("WNN_WORKER_EXIT", "1")
    terminal.main()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["worker_terminal"] is True
    assert manifest["attempt_id"] == "r5-fixture"
    assert {row["path"] for row in manifest["files"]} == set(expected)
    for row in manifest["files"]:
        path = output / row["path"]
        assert row["size_bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_resource_monitor_sums_worker_and_descendants(packet) -> None:
    supervisor = importlib.import_module("resource_supervisor")

    class Process:
        def __init__(self, rss, children=()):
            self.rss, self.descendants = rss, children

        def children(self, recursive):
            assert recursive is True
            return self.descendants

        def memory_info(self):
            from types import SimpleNamespace

            return SimpleNamespace(rss=self.rss)

    assert supervisor.aggregate_rss(Process(10, [Process(20), Process(30)])) == 60


def test_resource_stop_terminates_nested_trainer_sessions(packet, monkeypatch) -> None:
    supervisor = importlib.import_module("resource_supervisor")
    from types import SimpleNamespace

    sent = []
    monkeypatch.setattr(
        supervisor.psutil,
        "Process",
        lambda _: SimpleNamespace(children=lambda recursive: [SimpleNamespace(pid=200)]),
    )
    monkeypatch.setattr(supervisor.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(supervisor.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(supervisor.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    supervisor.stop_group(SimpleNamespace(pid=100, wait=lambda timeout: 0))
    assert set(sent) == {
        (pid, sig)
        for pid in (100, 200)
        for sig in (supervisor.signal.SIGTERM, supervisor.signal.SIGKILL)
    }
