from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from wnn import campaign, campaign_local, campaign_spec


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    for name in campaign.RUNTIME_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source\n")
    (root / "data.bin").write_bytes(b"training input")
    contract = root / "contract.md"
    contract.write_text("authorized campaign")
    state = tmp_path / "state"
    campaign.initialize(state, root, contract, "block1")
    return root, state


def specification(root: Path, kind: str = "gpu") -> campaign_spec.ExperimentSpec:
    return campaign_spec.seal_spec(
        root,
        {
            "version": 1,
            "name": "experiment",
            "kind": kind,
            "argv": ["{python}", "scripts/run_campaign.py", "--example"],
            "code": ["scripts/run_campaign.py"],
            "inputs": ["data.bin"],
            "artifacts": ["result.json"],
            "environment": {"WNN_LOADER_WORKERS": "2"},
            "limits": {
                "cpus": 8 if kind == "gpu" else 1,
                "ram_bytes": (24 if kind == "gpu" else 2) * 1024**3,
                "timeout_seconds": 100,
                "gpu_memory_bytes": 4_200_000_000 if kind == "gpu" else 0,
                "gpu_stop_bytes": 4_800_000_000 if kind == "gpu" else 0,
            },
        },
    )


def parent_properties() -> dict[str, str]:
    return {
        "CPUQuotaPerSecUSec": "12s",
        "MemoryMax": str(campaign_spec.RAM_MAX),
        "MemorySwapMax": "0",
        "ControlGroup": "/test/wnn-campaign.slice",
    }


def reserve(
    root: Path, state: Path, attempt: str = "attempt1", kind: str = "gpu"
) -> dict[str, object]:
    with campaign.locked(state):
        return campaign.reserve_attempt(state, root, specification(root, kind), attempt)


def test_content_identity_survives_copy_and_mtime(
    workspace: tuple[Path, Path], tmp_path: Path
) -> None:
    root, _ = workspace
    spec = specification(root)
    copied = tmp_path / "copy"
    copied.mkdir()
    for entry in spec["code"] + spec["inputs"]:
        path = copied / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / entry["path"]).read_bytes())
    campaign_spec.verify_files(copied, spec["code"] + spec["inputs"])
    assert spec["code_sha256"] == campaign_spec.digest(
        sorted(spec["code"], key=lambda x: x["path"])
    )
    (copied / "data.bin").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="content mismatch"):
        campaign_spec.verify_files(copied, spec["inputs"])


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "./data.bin", "."])
def test_spec_rejects_nonportable_paths(path: str) -> None:
    with pytest.raises(ValueError):
        campaign_spec.relative_path(path)


def test_symlink_escape_rejected(workspace: tuple[Path, Path], tmp_path: Path) -> None:
    root, _ = workspace
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes root"):
        campaign_spec.rooted(root, "escape/file")


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"code_sha256": "0" * 64},
        {"unknown": "field"},
        {"argv": []},
        {"artifacts": ["data.bin"]},
        {"environment": {"WNN_GPU_MEMORY_BYTES": "999999999999"}},
        {"environment": {"API_TOKEN": "secret"}},
    ],
)
def test_spec_rejects_identity_and_environment_bypasses(
    workspace: tuple[Path, Path], change: dict[str, object]
) -> None:
    root, _ = workspace
    with pytest.raises(ValueError):
        campaign_spec.validate_spec({**specification(root), **change})


@pytest.mark.parametrize(
    "key,value", [("cpus", 9), ("ram_bytes", 25 * 1024**3), ("gpu_stop_bytes", 5_000_000_000)]
)
def test_local_limits_cannot_be_widened(workspace: tuple[Path, Path], key: str, value: int) -> None:
    spec = specification(workspace[0])
    spec["limits"][key] = value
    with pytest.raises(ValueError):
        campaign_spec.validate_local(spec)


def test_remote_memory_spec_does_not_widen_local(workspace: tuple[Path, Path]) -> None:
    spec = specification(workspace[0])
    spec["limits"].update(gpu_memory_bytes=12 * 1024**3, gpu_stop_bytes=13 * 1024**3)
    campaign_spec.validate_spec(spec)
    with pytest.raises(ValueError):
        campaign_spec.validate_local(spec)


def test_reservations_serialize_gpu_and_preserve_failed_time(workspace: tuple[Path, Path]) -> None:
    root, state = workspace
    record = reserve(root, state)
    assert campaign.budget(state, "block1")["reserved_seconds"] == 1030
    with pytest.raises(RuntimeError, match="occupied"):
        reserve(root, state, "second")
    record.update(status="failed", charged_seconds=37.0)
    campaign.atomic_json(state / "attempts/attempt1.json", record)
    with pytest.raises(FileExistsError):
        reserve(root, state)
    reserve(root, state, "second")
    assert campaign.budget(state, "block1")["used_seconds"] == 37


def test_only_two_cpu_slots(workspace: tuple[Path, Path]) -> None:
    root, state = workspace
    reserve(root, state, "one", "cpu")
    reserve(root, state, "two", "cpu")
    with pytest.raises(RuntimeError, match="occupied"):
        reserve(root, state, "three", "cpu")
    assert campaign.budget(state, "block1")["remaining_seconds"] == 86400


def test_one_six_gib_helper_keeps_gpu_notebook_and_support_reserves(
    workspace: tuple[Path, Path],
) -> None:
    root, state = workspace
    reserve(root, state, "gpu")
    large = specification(root, "cpu")
    large["limits"]["ram_bytes"] = 6 * 1024**3
    campaign_spec.validate_local(large)
    with campaign.locked(state):
        campaign.reserve_attempt(state, root, large, "large")
        with pytest.raises(campaign.AdmissionBusy, match="larger-memory"):
            campaign.reserve_attempt(state, root, large, "secondlarge")
    assert not (state / "attempts/secondlarge.json").exists()
    record = reserve(root, state, "small", "cpu")
    allocation = record["admission_resources"]
    assert allocation["worker_ram_bytes"] == 32 * 1024**3
    assert allocation["worker_cpus"] == 10
    assert allocation["worker_ram_bytes"] + allocation["support_ram_bytes"] == campaign_spec.RAM_MAX
    assert allocation["worker_cpus"] + allocation["support_cpus"] == campaign_spec.CPU_MAX
    assert not (state / "PAUSE.json").exists()


def test_large_cpu_slot_released_only_after_attempt_ends(workspace: tuple[Path, Path]) -> None:
    root, state = workspace
    large = specification(root, "cpu")
    large["limits"]["ram_bytes"] = 6 * 1024**3
    with campaign.locked(state):
        previous = campaign.reserve_attempt(state, root, large, "first")
        previous["status"] = "failed"
        campaign.atomic_json(state / "attempts/first.json", previous)
        campaign.reserve_attempt(state, root, large, "second")
    assert campaign.budget(state, "block1")["remaining_seconds"] == 86400


@pytest.mark.parametrize("limits", [{"ram_bytes": 6 * 1024**3 + 1}, {"cpus": 2}])
def test_cpu_helper_rejects_larger_envelopes(
    workspace: tuple[Path, Path], limits: dict[str, int]
) -> None:
    root, state = workspace
    spec = specification(root, "cpu")
    spec["limits"].update(limits)
    with pytest.raises(ValueError):
        campaign_spec.validate_local(spec)
    with campaign.locked(state), pytest.raises(ValueError):
        campaign.reserve_attempt(state, root, spec, "rejected")
    assert not (state / "attempts/rejected.json").exists()


def test_block_renewal_retains_history(workspace: tuple[Path, Path]) -> None:
    root, state = workspace
    record = reserve(root, state)
    record.update(status="failed", charged_seconds=86_000)
    campaign.atomic_json(state / "attempts/attempt1.json", record)
    with pytest.raises(RuntimeError, match="allowance"):
        reserve(root, state, "cannot_fit")
    campaign.open_block(state, root, "block2", "reviewed outcomes and reassigned budget")
    reserve(root, state, "second")
    assert campaign.budget(state, "block1")["used_seconds"] == 86_000
    assert campaign.read_json(state / "blocks/block2.json")["previous_block"] == "block1"


def test_readiness_rejects_source_drift_and_pause(workspace: tuple[Path, Path]) -> None:
    root, state = workspace
    (root / "src/wnn/resources.py").write_text("changed")
    with pytest.raises(ValueError, match="runner changed"):
        campaign.ready(state, root)
    campaign.pause(state, "source changed; must review")
    with pytest.raises(RuntimeError, match="paused"):
        reserve(root, state)
    incident = str(campaign.read_json(state / "PAUSE.json")["id"])
    with pytest.raises(ValueError):
        campaign.acknowledge(state, root, "wrong", "review")
    campaign.acknowledge(state, root, incident, "reviewed correction, retained previous version")
    campaign.ready(state, root)
    assert (state / "incidents" / f"{incident}.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"CPUQuotaPerSecUSec": "13s"},
        {"MemoryMax": str(40 * 1024**3)},
        {"MemorySwapMax": "infinity"},
        {"ControlGroup": ""},
        {"CPUQuotaPerSecUSec": "infinity"},
    ],
)
def test_parent_must_be_verified(change: dict[str, str]) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        campaign_local.check_parent({**parent_properties(), **change})


def test_gpu_monitor_uses_decimal_bytes_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign_local, "capture", lambda _: "4700\n")
    assert campaign_local.device_memory() == 4701 * 1024**2
    monkeypatch.setattr(campaign_local, "capture", lambda _: "unknown\n")
    with pytest.raises(RuntimeError):
        campaign_local.device_memory()


def test_service_is_bounded_and_does_not_forward_secrets(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = workspace
    record = reserve(root, state)
    command = campaign_local.service_command(root, state, record, "/test/python")
    assert "--slice=wnn-campaign.slice" in command
    assert "MemorySwapMax=0" in command
    assert "KillMode=control-group" in command
    assert "OOMPolicy=kill" in command
    assert "MemoryOOMGroup=yes" not in command
    assert "Restart=no" in command
    assert "RuntimeMaxSec=120" in command
    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.setenv("WNN_GPU_FRACTION", "1")
    environment = campaign_local.worker_environment(specification(root))
    assert "API_TOKEN" not in environment and "WNN_GPU_FRACTION" not in environment
    assert environment["WNN_GPU_MEMORY_BYTES"] == "4200000000"


def test_launch_uncertainty_is_charged_and_blocks_retries(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = workspace
    monkeypatch.setattr(campaign_local, "properties", lambda _: parent_properties())
    monkeypatch.setattr(campaign_local, "check_orphans", lambda _: None)
    monkeypatch.setattr(campaign_local, "device_memory", lambda: 100_000_000)

    def failed(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("systemd-run", 10)

    monkeypatch.setattr(campaign_local.subprocess, "run", failed)
    with pytest.raises(subprocess.TimeoutExpired):
        campaign_local.launch(root, state, specification(root), "launch", "/test/python")
    record = campaign.read_json(state / "attempts/launch.json")
    assert record["status"] == "launch_uncertain"
    assert record["charged_seconds"] == 1030
    assert (state / "PAUSE.json").is_file()


def test_stale_monitor_cannot_be_replaced(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = workspace
    record = reserve(root, state)
    record.update(status="running", heartbeat_unix=time.time() - 60)
    campaign.atomic_json(state / "attempts/attempt1.json", record)
    monkeypatch.setattr(
        campaign_local, "capture", lambda _: "wnn-c04-attempt1.service loaded active running"
    )
    monkeypatch.setattr(campaign_local, "properties", lambda _: {"ActiveState": "active"})
    with pytest.raises(RuntimeError, match="heartbeat"):
        campaign_local.check_orphans(state)


def test_busy_launch_does_not_pause_existing_work(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = workspace
    reserve(root, state)
    monkeypatch.setattr(campaign_local, "properties", lambda _: parent_properties())
    monkeypatch.setattr(campaign_local, "check_orphans", lambda _: None)
    monkeypatch.setattr(campaign_local, "device_memory", lambda: 100_000_000)
    with pytest.raises(campaign.AdmissionBusy):
        campaign_local.launch(root, state, specification(root), "second", "/test/python")
    assert not (state / "PAUSE.json").exists()
    assert not (state / "attempts/second.json").exists()


def test_recent_finished_service_waits_for_teardown(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = workspace
    record = reserve(root, state)
    record.update(status="finished", finished_unix=time.time())
    campaign.atomic_json(state / "attempts/attempt1.json", record)
    monkeypatch.setattr(
        campaign_local, "capture", lambda _: "wnn-c04-attempt1.service loaded active running"
    )
    with pytest.raises(campaign.AdmissionBusy, match="teardown"):
        campaign_local.check_orphans(state)


def test_reconcile_lost_monitor_preserves_failure_evidence(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = workspace
    reserve(root, state)
    evidence = {"ActiveState": "failed", "Result": "oom-kill", "ExecMainStatus": "9"}
    monkeypatch.setattr(campaign_local, "properties", lambda _: evidence)
    campaign_local.reconcile(state)
    record = campaign.read_json(state / "attempts/attempt1.json")
    assert record["status"] == "lost"
    assert record["systemd_terminal"] == evidence
    assert campaign.budget(state, "block1")["used_seconds"] == 1030
    assert (state / "PAUSE.json").exists()


@pytest.mark.parametrize("mode", ["success", "vram", "monitor", "nonzero", "missing_output"])
def test_worker_lifecycle_without_launching_processes(
    workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root, state = workspace
    record = reserve(root, state)
    group = "/test/wnn-campaign.slice/wnn-c04-attempt1.service"
    original_read = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/self/cgroup":
            return f"0::{group}\n"
        if str(path).endswith("/memory.current"):
            return "2000000"
        if str(path).endswith("/memory.oom.group"):
            return "1"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        campaign_local,
        "properties",
        lambda unit: (
            parent_properties() if unit == campaign_spec.SLICE else {"ControlGroup": group}
        ),
    )
    monkeypatch.setattr(campaign_local, "memory_events", lambda _: {"oom_kill": 0})
    monkeypatch.setattr(campaign_local, "notify_ready", lambda: None)
    monkeypatch.setattr(campaign_local, "stop_child", lambda _: None)
    monkeypatch.setattr(campaign_local.time, "sleep", lambda _: None)
    readings = iter([100_000_000, 4_900_000_000 if mode == "vram" else 100_000_000])

    def observed() -> int:
        value = next(readings)
        if mode == "monitor" and value == 100_000_000 and (root / "result.json").exists():
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return value

    monkeypatch.setattr(campaign_local, "device_memory", observed)

    class Child:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode = 7 if mode == "nonzero" else 0
            self.polls = 0
            if mode != "missing_output":
                (root / "result.json").write_text(json.dumps({"ok": True}))

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else self.returncode

    monkeypatch.setattr(campaign_local.subprocess, "Popen", Child)
    result = campaign_local.run_worker(root, state, "attempt1", "/test/python")
    final = campaign.read_json(state / "attempts/attempt1.json")
    assert result == (0 if mode == "success" else 1)
    assert final["status"] == ("finished" if mode == "success" else "failed")
    assert final["spec_sha256"] == record["spec_sha256"]
    assert final["charged_seconds"] >= 0
    assert (state / "PAUSE.json").exists() == (mode != "success")
    if mode == "success":
        campaign_spec.verify_files(root, final["artifacts"])
