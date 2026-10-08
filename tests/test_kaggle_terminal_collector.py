from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-06/cloud_preflight"
)
sys.path.insert(0, str(MODULE_DIR))
import collect_terminal_outputs as collector  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture_config(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    repo_root = tmp_path
    config_dir = repo_root / "cloud_preflight"
    config_dir.mkdir()
    cli_path = repo_root / ".venv/bin/kaggle"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/bin/sh\nexit 0\n")
    cli_path.chmod(0o700)

    pins: dict[str, dict[str, object]] = {}
    targets = []
    for kernel, resource in (
        ("test-owner/smoke", "gpu"),
        ("test-owner/tpu", "tpu"),
    ):
        metadata_id = kernel + "-metadata"
        admission_bindings = {"code_sha256": f"{kernel}-code"}
        admission = {
            "authorized": True,
            "kernel_id": metadata_id,
            "code_sha256": admission_bindings["code_sha256"],
        }
        admission_path = repo_root / "admissions" / f"{resource}.json"
        _write_json(admission_path, admission)
        admission_sha = hashlib.sha256(admission_path.read_bytes()).hexdigest()
        target = {
            "kernel": kernel,
            "resource": resource,
            "metadata_id": metadata_id,
            "root_admission": f"../admissions/{resource}.json",
            "root_admission_sha256": admission_sha,
            "artifact_dir": f"../outputs/{resource}",
            "terminal_receipt": f"../receipts/{resource}.json",
            "push_receipt": f"../receipts/{resource}_push.json",
            "admission_bindings": admission_bindings,
        }
        _write_json(
            repo_root / "receipts" / f"{resource}_push.json",
            {
                "canonical_provider_slug": kernel,
                "root_admission_sha256": admission_sha,
                "push_count": 1,
                "kernel_version": 1,
                "packet": admission_bindings,
            },
        )
        pins[kernel] = {
            key: target[key]
            for key in (
                "resource",
                "metadata_id",
                "root_admission",
                "root_admission_sha256",
                "artifact_dir",
                "terminal_receipt",
                "push_receipt",
                "admission_bindings",
            )
        }
        targets.append(target)

    config_path = config_dir / "targets.json"
    _write_json(
        config_path,
        {
            "schema": "wnn.kaggle-terminal-collector-targets.v1",
            "kaggle_cli": str(cli_path),
            "expected_cli_version": "Kaggle CLI 2.2.4",
            "minimum_free_bytes": 2 * collector.GIB,
            "maximum_download_bytes": 3 * collector.GIB,
            "max_runtime_seconds": 800,
            "max_attempts": 3,
            "poll_log": "poll.jsonl",
            "targets": targets,
        },
    )
    return config_path, repo_root, pins


def test_validate_config_binds_exact_root_admissions_and_single_push(
    tmp_path: Path,
) -> None:
    config_path, repo_root, pins = _fixture_config(tmp_path)

    config, targets = collector.validate_config(
        config_path, repo_root, pinned_targets=pins
    )

    assert len(targets) == 2
    assert config["max_attempts"] == 3


def test_validate_config_rejects_an_unpinned_kernel(tmp_path: Path) -> None:
    config_path, repo_root, pins = _fixture_config(tmp_path)
    config = json.loads(config_path.read_text())
    config["targets"][0]["kernel"] = "test-owner/other"
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="exactly the two pinned kernels"):
        collector.validate_config(config_path, repo_root, pinned_targets=pins)


def test_validate_config_rejects_changed_root_admission(tmp_path: Path) -> None:
    config_path, repo_root, pins = _fixture_config(tmp_path)
    admission = repo_root / "admissions/gpu.json"
    data = json.loads(admission.read_text())
    data["authorized"] = False
    _write_json(admission, data)

    with pytest.raises(ValueError, match="root admission hash mismatch"):
        collector.validate_config(config_path, repo_root, pinned_targets=pins)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("KernelWorkerStatus.QUEUED", False),
        ("KernelWorkerStatus.RUNNING", False),
        ("KernelWorkerStatus.COMPLETE", True),
        ("KernelWorkerStatus.ERROR", True),
    ],
)
def test_terminal_state(status: str, expected: bool) -> None:
    assert collector.terminal_state(status) is expected


def test_file_inventory_streams_hashes_and_keeps_relative_names(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "attempt_001/output.bin"
    artifact.parent.mkdir(parents=True)
    payload = b"model output" * 4096
    artifact.write_bytes(payload)

    inventory = collector.file_inventory(tmp_path)

    assert inventory["attempt_001/output.bin"] == {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_collection_receipt_requires_existing_files_and_both_downloads(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    (artifact_dir / "attempt_001").mkdir(parents=True)
    (artifact_dir / "attempt_001/provider_logs.txt").write_text("")
    receipt_path = tmp_path / "receipt.json"
    _write_json(
        receipt_path,
        {
            "terminal_status": "KernelWorkerStatus.COMPLETE",
            "terminal_collection": {
                "output_exit_code": 0,
                "log_exit_code": 0,
                "files": {"attempt_001/provider_logs.txt": {}},
            },
        },
    )

    assert collector.collection_is_complete(receipt_path, artifact_dir)
    (artifact_dir / "attempt_001/provider_logs.txt").unlink()
    assert not collector.collection_is_complete(receipt_path, artifact_dir)


def test_collection_receipt_rejects_paths_outside_artifact_directory(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    attempt_dir = artifact_dir / "attempt_001"
    attempt_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside")
    (attempt_dir / "escape.txt").symlink_to(outside_file)
    receipt_path = tmp_path / "receipt.json"
    _write_json(
        receipt_path,
        {
            "terminal_status": "KernelWorkerStatus.COMPLETE",
            "terminal_collection": {
                "output_exit_code": 0,
                "log_exit_code": 0,
                "files": {"attempt_001/escape.txt": {}},
            },
        },
    )

    assert not collector.collection_is_complete(receipt_path, artifact_dir)


def test_next_attempt_preserves_prior_partial_attempts(tmp_path: Path) -> None:
    (tmp_path / "attempt_001").mkdir()
    (tmp_path / "attempt_003").mkdir()

    assert collector.next_attempt_number(tmp_path) == 4


def test_bounded_command_captures_output(tmp_path: Path) -> None:
    monitor_dir = tmp_path / "attempt"
    monitor_dir.mkdir()
    result = collector.run_bounded_command(
        [sys.executable, "-c", "print('bounded')"],
        stdout_path=monitor_dir / "stdout.txt",
        stderr_path=monitor_dir / "stderr.txt",
        monitor_dir=monitor_dir,
        min_free_bytes=1,
        max_bytes=1024,
        deadline=time.monotonic() + 20,
    )

    assert result["exit_code"] == 0
    assert (monitor_dir / "stdout.txt").read_text().strip() == "bounded"


def test_bounded_command_rejects_fast_download_over_byte_cap(
    tmp_path: Path,
) -> None:
    monitor_dir = tmp_path / "attempt"
    monitor_dir.mkdir()
    payload_path = monitor_dir / "download.bin"
    result = collector.run_bounded_command(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'x'*4096)",
            str(payload_path),
        ],
        stdout_path=monitor_dir / "stdout.txt",
        stderr_path=monitor_dir / "stderr.txt",
        monitor_dir=monitor_dir,
        min_free_bytes=1,
        max_bytes=512,
        deadline=time.monotonic() + 20,
    )

    assert result["exit_code"] == 125
    assert result["stop_reason"] == "download_byte_limit_reached_after_process_exit"
    assert payload_path.exists()
