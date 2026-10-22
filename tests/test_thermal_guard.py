"""Unit tests for the optional CPU-package thermal guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

GUARD_SCRIPT = Path(__file__).parent.parent / "scripts/thermal_guard.py"


def load_guard() -> ModuleType:
    """Load the standalone guard without invoking its CLI."""
    spec = importlib.util.spec_from_file_location("thermal_guard_under_test", GUARD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_sensor(root: Path, hwmon: str, index: int, label: str, milli_celsius: int) -> None:
    """Create one hwmon temperature label and reading."""
    sensor = root / hwmon
    sensor.mkdir(exist_ok=True)
    (sensor / "name").write_text("coretemp\n")
    (sensor / f"temp{index}_label").write_text(f"{label}\n")
    (sensor / f"temp{index}_input").write_text(f"{milli_celsius}\n")


def test_reads_max_package_temperature_and_ignores_core_sensor(tmp_path: Path) -> None:
    guard = load_guard()
    write_sensor(tmp_path, "hwmon0", 1, "Package id 0", 93_000)
    write_sensor(tmp_path, "hwmon0", 2, "Core 0", 99_000)
    write_sensor(tmp_path, "hwmon1", 1, "Package id 1", 95_000)

    assert guard.read_cpu_package_temperature(tmp_path) == 95.0


def test_cpu_temperature_hysteresis_pauses_at_97_and_resumes_at_90() -> None:
    guard = load_guard()

    assert guard.pause_reason(guard.Reading(70, 30_000_000_000, 0.0, 96.999), False) is None
    assert (
        guard.pause_reason(guard.Reading(70, 30_000_000_000, 0.0, 97.0), False)
        == "cpu_package_temperature"
    )
    assert (
        guard.pause_reason(guard.Reading(70, 30_000_000_000, 0.0, 90.001), True)
        == "cpu_package_temperature"
    )
    assert guard.pause_reason(guard.Reading(70, 30_000_000_000, 0.0, 90.0), True) is None


def test_gpu_and_cpu_reasons_remain_independent() -> None:
    guard = load_guard()

    both_hot = guard.Reading(80, 30_000_000_000, 0.0, 98.0)
    assert guard.pause_reason(both_hot, False) == "temperature"
    assert (
        guard.pause_reason(guard.Reading(70, 30_000_000_000, 0.0, 98.0), True)
        == "cpu_package_temperature"
    )
    assert guard.pause_reason(guard.Reading(80, 30_000_000_000, 0.0, 90.0), False) == "temperature"


def test_enabled_cpu_guard_fails_when_package_sensor_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    guard = load_guard()
    monkeypatch.setattr(guard, "CPU_HWMON_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="no readable Package id sensor"):
        guard.read_resources(include_cpu_package_temperature=True)


def test_unreadable_cpu_package_input_fails_clearly(tmp_path: Path) -> None:
    guard = load_guard()
    sensor = tmp_path / "hwmon0"
    sensor.mkdir()
    (sensor / "temp1_label").write_text("Package id 0\n")

    with pytest.raises(RuntimeError, match="cannot read CPU package temperature"):
        guard.read_cpu_package_temperature(tmp_path)


def test_default_resource_read_does_not_probe_cpu_sensor(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = load_guard()
    monkeypatch.setattr(
        guard,
        "read_cpu_package_temperature",
        lambda *_args, **_kwargs: pytest.fail("CPU sensor read without explicit opt-in"),
    )
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="72\n"),
    )

    def read_proc(path: Path) -> str:
        if path == Path("/proc/meminfo"):
            return "MemAvailable: 1000 kB\n"
        if path == Path("/proc/pressure/memory"):
            return "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(Path, "read_text", read_proc)
    reading = guard.read_resources()

    assert reading == guard.Reading(72, 1_024_000, 0.0, None)


def test_cli_without_cpu_flag_keeps_default_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = load_guard()
    calls: list[tuple[Path, bool]] = []

    def run_command(path: Path, *, cpu_package_temperature_guard: bool = False) -> int:
        calls.append((path, cpu_package_temperature_guard))
        return 0

    monkeypatch.setattr(guard, "run_command", run_command)
    monkeypatch.setattr(sys, "argv", [str(GUARD_SCRIPT), "--spec", "guard.json"])
    with pytest.raises(SystemExit) as error:
        guard.main()

    assert error.value.code == 0
    assert calls == [(Path("guard.json"), False)]
