"""Mocked tests for the single-invocation CPU thermal observer."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

OBSERVER_SCRIPT = Path(__file__).parent.parent / "scripts/observe_online_panel_cpu_temperature.py"


@dataclass(frozen=True)
class FakeServiceState:
    """Minimal systemd state returned by the mocked CLI."""

    invocation_id: str
    active_state: str
    freezer_state: str


def load_observer() -> ModuleType:
    """Load the observer module without running its CLI."""
    spec = importlib.util.spec_from_file_location(
        "online_panel_cpu_observer_under_test", OBSERVER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_systemctl_mock(
    observer: ModuleType, monkeypatch: pytest.MonkeyPatch, states: list[FakeServiceState]
) -> list[list[str]]:
    """Return recorded commands and answer show requests from a state sequence."""
    calls: list[list[str]] = []
    last_state = states[-1]
    remaining = iter(states)

    def run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[2] != "show":
            return subprocess.CompletedProcess(args, 0, "", "")
        try:
            state = next(remaining)
        except StopIteration:
            state = last_state
        stdout = (
            f"InvocationID={state.invocation_id}\n"
            f"ActiveState={state.active_state}\n"
            f"FreezerState={state.freezer_state}\n"
        )
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(observer.subprocess, "run", run)
    return calls


def service_state(
    observer: ModuleType,
    *,
    freezer_state: str = "running",
    invocation_id: str | None = None,
    active_state: str = "active",
) -> FakeServiceState:
    """Create the requested fake systemd state."""
    return FakeServiceState(
        observer.TARGET_INVOCATION if invocation_id is None else invocation_id,
        active_state,
        freezer_state,
    )


def test_package_reader_uses_hottest_package_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer = load_observer()
    hwmon = tmp_path / "hwmon0"
    hwmon.mkdir()
    (hwmon / "temp1_label").write_text("Package id 0\n")
    (hwmon / "temp1_input").write_text("92000\n")
    (hwmon / "temp2_label").write_text("Core 0\n")
    (hwmon / "temp2_input").write_text("99000\n")
    monkeypatch.setattr(observer, "CPU_HWMON_ROOT", tmp_path)

    assert observer.read_cpu_package_temperature() == 92.0


def test_package_reader_fails_clearly_when_no_package_sensor_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer = load_observer()
    monkeypatch.setattr(observer, "CPU_HWMON_ROOT", tmp_path)

    with pytest.raises(observer.CpuSensorError, match="no readable Package id sensor"):
        observer.read_cpu_package_temperature()


def test_hot_threshold_freezes_only_after_exact_invocation_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = load_observer()
    calls = install_systemctl_mock(
        observer,
        monkeypatch,
        [service_state(observer), service_state(observer), service_state(observer)],
    )
    temperatures = iter((96.999, 97.0))
    monkeypatch.setattr(observer, "read_cpu_package_temperature", lambda: next(temperatures))

    assert observer.poll_once(False) is False
    assert observer.poll_once(False) is True
    assert [call[2] for call in calls] == ["show", "show", "show", "freeze"]
    assert all(call[-1] == observer.TARGET_UNIT for call in calls)
    assert all("--property=FreezerState" in call for call in calls[:2])


def test_external_freeze_is_not_adopted_or_thawed(monkeypatch: pytest.MonkeyPatch) -> None:
    observer = load_observer()
    calls = install_systemctl_mock(
        observer,
        monkeypatch,
        [service_state(observer, freezer_state="frozen")],
    )
    temperatures = iter((98.0, 90.0))
    monkeypatch.setattr(observer, "read_cpu_package_temperature", lambda: next(temperatures))

    assert observer.poll_once(False) is False
    assert observer.poll_once(False) is False
    assert [call[2] for call in calls] == ["show", "show"]


def test_owned_freeze_thaws_only_at_or_below_90(monkeypatch: pytest.MonkeyPatch) -> None:
    observer = load_observer()
    calls = install_systemctl_mock(
        observer, monkeypatch, [service_state(observer, freezer_state="frozen")]
    )
    temperatures = iter((90.001, 90.0))
    monkeypatch.setattr(observer, "read_cpu_package_temperature", lambda: next(temperatures))

    assert observer.poll_once(True) is True
    assert observer.poll_once(True) is False
    assert [call[2] for call in calls] == ["show", "show", "show", "thaw"]


def test_sensor_failure_freezes_matching_active_target_then_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = load_observer()
    calls = install_systemctl_mock(
        observer,
        monkeypatch,
        [service_state(observer), service_state(observer), service_state(observer)],
    )

    def missing_sensor() -> float:
        raise observer.CpuSensorError("no readable Package id sensor")

    monkeypatch.setattr(observer, "read_cpu_package_temperature", missing_sensor)

    with pytest.raises(observer.CpuSensorError, match="requested a systemd freeze"):
        observer.poll_once(False)

    assert [call[2] for call in calls] == ["show", "show", "show", "freeze"]
    assert all(call[-1] == observer.TARGET_UNIT for call in calls)


@pytest.mark.parametrize(
    ("invocation_id", "active_state"),
    [("different-invocation", "active"), (None, "inactive")],
)
def test_changed_or_inactive_target_exits_without_read_or_action(
    monkeypatch: pytest.MonkeyPatch, invocation_id: str | None, active_state: str
) -> None:
    observer = load_observer()
    calls = install_systemctl_mock(
        observer,
        monkeypatch,
        [service_state(observer, invocation_id=invocation_id, active_state=active_state)],
    )
    monkeypatch.setattr(
        observer,
        "read_cpu_package_temperature",
        lambda: pytest.fail("sensor read after target invocation ended or changed"),
    )

    assert observer.poll_once(False) is None
    assert [call[2] for call in calls] == ["show"]


def test_changed_invocation_before_action_prevents_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = load_observer()
    calls = install_systemctl_mock(
        observer,
        monkeypatch,
        [
            service_state(observer),
            service_state(observer, invocation_id="replacement-invocation"),
        ],
    )
    monkeypatch.setattr(observer, "read_cpu_package_temperature", lambda: 97.0)

    assert observer.poll_once(False) is None
    assert [call[2] for call in calls] == ["show", "show"]
