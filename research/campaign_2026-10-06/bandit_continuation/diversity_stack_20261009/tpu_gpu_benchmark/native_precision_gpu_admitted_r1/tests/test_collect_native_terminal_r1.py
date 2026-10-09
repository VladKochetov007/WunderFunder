"""Check the exact admitted native target and terminal-only collector handoff."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

NATIVE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE_DIR))

import collect_native_terminal_r1 as native_collector  # noqa: E402

KERNEL = "vladkochetov07/wnn-native-gpu-fp16-20261009-r1"
RELATIVE_NATIVE_DIR = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "tpu_gpu_benchmark/native_precision_gpu_admitted_r1"
)
COPY_FILES = (
    "native_terminal_collection_targets_r1.json",
    "ROOT_NATIVE_GPU_FP16_DISPATCH_R1.json",
    "SOURCE_READY_REMOTE_VERIFICATION_R1.json",
    "FRESH_GPU_QUOTA_ADMISSION_R1.json",
    "NATIVE_GPU_FP16_PROVIDER_PUSH_R1.json",
    "source_manifest.json",
    "source_asset/assets-manifest.json",
    "kernel/wnn-native-gpu-fp16-20261009-r1.py",
    "kernel/kernel-metadata.json",
)


def _find_repo_root(start: Path) -> Path:
    """Find the project root containing the authoritative specifications."""
    for candidate in start.parents:
        if (candidate / "specs/README.md").is_file():
            return candidate
    raise ValueError("could not locate repository root")


def _copy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the real admitted receipts and code into an isolated small fixture."""
    repo_root = tmp_path / "repo"
    native_dir = repo_root / RELATIVE_NATIVE_DIR
    native_dir.mkdir(parents=True)
    for relative_name in COPY_FILES:
        source = NATIVE_DIR / relative_name
        target = native_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    dispatch = json.loads((native_dir / "ROOT_NATIVE_GPU_FP16_DISPATCH_R1.json").read_text())
    admission_source = _find_repo_root(NATIVE_DIR) / (
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "tpu_gpu_benchmark/native_precision_gpu_r1/ROOT_NATIVE_GPU_FP16_ADMISSION_R1.json"
    )
    admission_target = repo_root / (
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "tpu_gpu_benchmark/native_precision_gpu_r1/ROOT_NATIVE_GPU_FP16_ADMISSION_R1.json"
    )
    admission_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(admission_source, admission_target)
    packet = repo_root / dispatch["packet"]
    packet.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_find_repo_root(NATIVE_DIR) / dispatch["packet"], packet)
    return native_dir, repo_root


def test_native_binding_matches_actual_dispatch_and_provider_receipts() -> None:
    """Verify the live target binds the exact accepted private T4 kernel."""
    repo_root = _find_repo_root(NATIVE_DIR)
    config, target = native_collector.validate_binding(NATIVE_DIR, repo_root)

    assert config["expected_cli_version"] == "Kaggle CLI 2.2.4"
    assert target["kernel"] == KERNEL
    assert target["provider_timeout_seconds"] == 3600
    assert target["admission_bindings"]["packet_sha256"] == (
        "52ef723bc159773627d0d89ff6923fd62f8615bf012cc7ed14de148aa54807ac"
    )


def test_native_binding_rejects_actual_provider_device_drift(tmp_path: Path) -> None:
    """Reject a provider receipt changed to claim a different accelerator."""
    native_dir, repo_root = _copy_fixture(tmp_path)
    push_path = native_dir / "NATIVE_GPU_FP16_PROVIDER_PUSH_R1.json"
    push = json.loads(push_path.read_text())
    push["provider_observation"]["pulled_metadata"]["machine_shape"] = "NvidiaA100"
    push_path.write_text(json.dumps(push, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="actual provider kernel metadata"):
        native_collector.validate_binding(native_dir, repo_root)


def test_nonterminal_status_never_calls_terminal_download_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the durable downloader idle while the provider worker is running."""
    native_dir, repo_root = _copy_fixture(tmp_path)
    snapshot = {
        "observed_at_utc": "2026-10-09T03:31:28+00:00",
        "quota": {},
        "status": [
            {
                "kernel": KERNEL,
                "status": "KernelWorkerStatus.RUNNING",
                "monitoring_state": "polling_until_terminal_evidence_downloaded",
            }
        ],
    }
    monkeypatch.setattr(native_collector, "collect_provider_snapshot", lambda _path: snapshot)

    def fail_if_called(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("terminal output downloader ran before terminal status")

    monkeypatch.setattr(native_collector, "collect_terminal_target", fail_if_called)
    result = native_collector.run_once(native_dir, repo_root)

    assert result["targets"] == [
        {"kernel": KERNEL, "state": "provider_not_terminal", "status": "KernelWorkerStatus.RUNNING"}
    ]


def test_terminal_status_delegates_to_existing_bounded_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route terminal results through the repository's bounded collector helper."""
    native_dir, repo_root = _copy_fixture(tmp_path)
    snapshot = {
        "observed_at_utc": "2026-10-09T04:00:00+00:00",
        "quota": {},
        "status": [
            {
                "kernel": KERNEL,
                "status": "KernelWorkerStatus.COMPLETE",
                "monitoring_state": "polling_until_terminal_evidence_downloaded",
            }
        ],
    }
    monkeypatch.setattr(native_collector, "collect_provider_snapshot", lambda _path: snapshot)
    captured: dict[str, Any] = {}

    def collect(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"kernel": KERNEL, "state": "terminal_outputs_and_logs_collected"}

    monkeypatch.setattr(native_collector, "collect_terminal_target", collect)
    result = native_collector.run_once(native_dir, repo_root)

    assert result["targets"][0]["state"] == "terminal_outputs_and_logs_collected"
    assert captured["target"]["kernel"] == KERNEL
    assert captured["config"]["maximum_download_bytes"] == 8 * 1024**3
