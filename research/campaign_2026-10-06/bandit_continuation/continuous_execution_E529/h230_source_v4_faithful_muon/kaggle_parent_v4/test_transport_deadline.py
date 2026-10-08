from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

PACKET_ROOT = Path(__file__).resolve().parent
PAYLOAD_SCRIPTS = PACKET_ROOT / "payload_root/scripts"
sys.path.insert(0, str(PAYLOAD_SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sealed transport module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_kernel_and_runner_expose_the_authorized_time_window(tmp_path: Path) -> None:
    kernel = _load("h230_v4_kaggle_kernel", PAYLOAD_SCRIPTS / "kaggle_kernel.py")
    runner = _load(
        "h230_v4_kaggle_gpu_runner", PAYLOAD_SCRIPTS / "kaggle_gpu_runner.py"
    )

    assert kernel.KERNEL_PROCESS_WALL_SECONDS_MAX == 10_000
    assert kernel.RUNNER_FINALIZATION_RESERVE_SECONDS == 300
    assert runner.WORKER_TIMEOUT_CEILING_SECONDS == 9_000

    now = 1_000.0
    assert kernel.worker_timeout_for_deadline(9_000, now + 9_301, now_monotonic=now) == 9_000
    assert (
        runner.resolve_worker_timeout_seconds(
            9_000, str(now + 9_301), now_monotonic=now
        )
        == 9_000
    )
    assert (
        runner.resolve_worker_timeout_seconds(
            9_000, str(now + 8_301), now_monotonic=now
        )
        == 8_001
    )

    source = tmp_path / "spec.json"
    output = tmp_path / "runtime.json"
    source.write_text(json.dumps({"limits": {"timeout_seconds": 9_000}}), encoding="utf-8")
    deadline = time.monotonic() + 9_301
    assert kernel.write_runtime_spec(source, output, deadline) == 9_000
    runtime_spec = json.loads(output.read_text(encoding="utf-8"))
    assert runtime_spec["limits"]["timeout_seconds"] == 9_000


def test_v4_worker_timeout_rejects_invalid_or_exhausted_deadlines() -> None:
    runner = _load(
        "h230_v4_kaggle_gpu_runner", PAYLOAD_SCRIPTS / "kaggle_gpu_runner.py"
    )
    now = time.monotonic()
    with pytest.raises(ValueError):
        runner.resolve_worker_timeout_seconds(True, str(now + 10_000), now_monotonic=now)
    with pytest.raises(TimeoutError):
        runner.resolve_worker_timeout_seconds(9_000, str(now + 299), now_monotonic=now)
