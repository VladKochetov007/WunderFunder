from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import heterogeneous_six_evidence as evidence
from scripts import heterogeneous_six_support as support
from scripts.heterogeneous_six_evidence import external_status, profile_gate
from scripts.heterogeneous_six_support import source_binding


def test_source_binding_checks_real_frozen_files(tmp_path: Path) -> None:
    source = tmp_path / "source-r3"
    source.mkdir()
    path = source / "code.py"
    path.write_text("x = 1\n")
    files = {"code.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    identity = source.with_name("source-r3.identity.json")
    identity.write_text(
        json.dumps({"schema": "wnn.heterogeneous-source.v1", "files": files, "sha256": digest})
    )
    assert source_binding(source)["sha256"] == digest
    path.write_text("x = 2\n")
    with pytest.raises(ValueError, match="immutable source file differs"):
        source_binding(source)


def test_profile_gate_requires_full_finite_eighty_updates(tmp_path: Path) -> None:
    path = tmp_path / "profile.log"
    record = {
        "valid_for_throughput": True,
        "optimizer_attempts": 80,
        "optimizer_completed_updates": 80,
        "gradient_stats_steps": 80,
        "loss_checks": 80,
        "loss_finite": True,
        "gradient_finite": True,
    }
    path.write_text("PROFILE_UPDATE_EVIDENCE " + json.dumps(record) + "\n")
    assert profile_gate(path)["optimizer_completed_updates"] == 80
    record["optimizer_completed_updates"] = 79
    path.write_text("PROFILE_UPDATE_EVIDENCE " + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        profile_gate(path)


@pytest.mark.parametrize("source_encoding", ["binding", "aggregate"])
def test_external_phase_cannot_adopt_changed_request(tmp_path: Path, source_encoding: str) -> None:
    (tmp_path / "requests").mkdir()
    (tmp_path / "status").mkdir()
    binding = {"path": "pinned-identity.json", "sha256": "a" * 64}
    source_identity = binding if source_encoding == "binding" else binding["sha256"]
    request_path = tmp_path / "requests/cache-gru-only-r2.json"
    request_path.write_text(json.dumps({"source_identity": source_identity}))
    status = {
        "status": "complete",
        "return_code": 0,
        "source_identity": source_identity,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "status/cache-gru-only-r2.json").write_text(json.dumps(status))
    assert external_status(tmp_path, "cache-gru-only-r2", binding)["status"]["status"] == "complete"
    request_path.write_text("{}")
    with pytest.raises(ValueError, match="request/source differs"):
        external_status(tmp_path, "cache-gru-only-r2", binding)


def test_runtime_adopts_original_owned_export_without_relaunching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "requests").mkdir()
    (tmp_path / "status").mkdir()
    source, python = tmp_path / "source-r3", tmp_path / "python"
    binding = {"path": "source-r3.identity.json", "sha256": "a" * 64}
    key = "export-tcn_s1202-train-r1"
    tokens = ["het6_tcn_s1202", "train"]
    command = [
        "taskset",
        "-c",
        "8-15",
        str(python),
        "-u",
        str(source / "scripts/predict_run.py"),
        *tokens,
    ]
    request = {
        "command": command,
        "cwd": str(source),
        "phase": "train_export",
        "role": "tcn_s1202",
        "source_identity": binding,
        "gpu_uuid": "GPU-test",
        "env": {"WNN_PROJECT_ROOT": str(tmp_path), "CUDA_VISIBLE_DEVICES": "GPU-test"},
    }
    request_path = tmp_path / "requests" / f"{key}.json"
    request_path.write_text(json.dumps(request))
    status = {
        "status": "complete",
        "return_code": 0,
        "runner_pid": 123,
        "runner_start_ticks": "42",
        "phase": "train_export",
        "role": "tcn_s1202",
        "source_identity": binding,
        "gpu_uuid": "GPU-test",
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    status_path = tmp_path / "status" / f"{key}.json"
    status_path.write_text(json.dumps(status))
    original = request_path.read_bytes(), status_path.read_bytes()
    monkeypatch.setattr(evidence, "gpu_snapshot", lambda: {1: {"uuid": "GPU-test"}})

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an adopted export must never launch another process")

    monkeypatch.setattr(support.subprocess, "Popen", forbidden)
    runtime = support.Runtime(tmp_path, source, python, binding, time.monotonic() + 60)
    assert (
        runtime.job(
            key,
            "predict_run.py",
            tokens,
            gpu=1,
            phase="train_export",
            role="tcn_s1202",
            seconds=1800,
        )
        == "complete"
    )
    assert runtime.jobs[key]["adopted"] is True and runtime.handles[key] is None
    assert original == (request_path.read_bytes(), status_path.read_bytes())


def test_arithmetic_fixture_is_serializable_and_detects_wrong_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = [np.full((3, 20000, 2), value, dtype=np.float32) for value in (3, -1)]
    records = [{"coefficient": 0.5, "index": index} for index in (0, 1)]
    monkeypatch.setattr(
        evidence,
        "validate_source",
        lambda record, split: (arrays[record["index"]], {2: 0, 7: 1, 8: 2}),
    )
    path = tmp_path / "main_source_train.npy"
    np.save(path, np.ones((3, 20000, 2), dtype=np.float32), allow_pickle=False)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "profile": "gru-only",
                "sources": records,
                "files": {"main_source_train": {"row_groups": [2, 7, 8]}},
                "source_coefficients": records,
            }
        )
    )
    output = tmp_path / "fixture.json"
    result = evidence.arithmetic_fixture(tmp_path, output, SimpleNamespace())
    assert result["passed"] is True and len(json.loads(output.read_text())["samples"]) == 15
    values = np.load(path, allow_pickle=False)
    values[0, 0, 0] += 1
    np.save(path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="independent cache arithmetic"):
        evidence.arithmetic_fixture(tmp_path, tmp_path / "bad-fixture.json", SimpleNamespace())
