"""Fail-closed CPU package observer for one admitted online-panel invocation."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TARGET_UNIT = "wnn-online-panel-r1-20261023.service"
TARGET_INVOCATION = "1e64dd46f5294caba4977275e6c91feb"
CPU_HWMON_ROOT = Path("/sys/class/hwmon")
POLL_SECONDS = 1.0
FREEZE_CELSIUS = 97.0
THAW_CELSIUS = 90.0
LOGGER = logging.getLogger(__name__)


class CpuSensorError(RuntimeError):
    """Raised when no trustworthy CPU package temperature can be read."""


@dataclass(frozen=True)
class ServiceState:
    """Identity and systemd freezer state for the target unit."""

    invocation_id: str
    active_state: str
    freezer_state: str


def read_cpu_package_temperature() -> float:
    """Read the hottest readable package temperature from Linux hwmon sysfs."""
    readings: list[float] = []
    try:
        label_paths = sorted(CPU_HWMON_ROOT.glob("hwmon*/temp*_label"))
    except OSError as error:
        raise CpuSensorError(f"cannot inspect CPU sensor directory {CPU_HWMON_ROOT}") from error
    for label_path in label_paths:
        try:
            label = label_path.read_text().strip()
        except OSError as error:
            raise CpuSensorError(f"cannot read CPU sensor label {label_path}") from error
        if not label.startswith("Package id "):
            continue
        input_path = label_path.with_name(f"{label_path.name.removesuffix('_label')}_input")
        try:
            readings.append(int(input_path.read_text().strip()) / 1000.0)
        except (OSError, ValueError) as error:
            raise CpuSensorError(f"cannot read CPU package temperature {input_path}") from error
    if not readings:
        raise CpuSensorError(f"no readable Package id sensor found under {CPU_HWMON_ROOT}")
    return max(readings)


def read_service_state() -> ServiceState:
    """Read target identity, active state, and freezer state in one systemd query."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--property=InvocationID",
            "--property=ActiveState",
            "--property=FreezerState",
            TARGET_UNIT,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"cannot inspect {TARGET_UNIT}: {detail}")
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    required = {"InvocationID", "ActiveState", "FreezerState"}
    if not required.issubset(properties):
        missing = ", ".join(sorted(required.difference(properties)))
        raise RuntimeError(f"systemd omitted required target properties: {missing}")
    return ServiceState(
        properties["InvocationID"],
        properties["ActiveState"],
        properties["FreezerState"],
    )


def _is_target_active(state: ServiceState) -> bool:
    return state.invocation_id == TARGET_INVOCATION and state.active_state == "active"


def _request_action(
    action: Literal["freeze", "thaw"], allowed_freezer_states: tuple[str, ...]
) -> bool | None:
    """Act only after rechecking exact invocation, active state, and freezer state."""
    state = read_service_state()
    if not _is_target_active(state):
        return None
    if state.freezer_state not in allowed_freezer_states:
        return False
    result = subprocess.run(
        ["systemctl", "--user", action, TARGET_UNIT],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"systemd {action} failed for {TARGET_UNIT}: {detail}")
    LOGGER.warning("%s %s for invocation %s", action, TARGET_UNIT, TARGET_INVOCATION)
    return True


def _freeze_after_sensor_error() -> bool | None:
    """Freeze an active matching unit on sensor failure without claiming old freezes."""
    for _attempt in range(3):
        state = read_service_state()
        if not _is_target_active(state):
            return None
        if state.freezer_state in {"frozen", "freezing"}:
            return False
        requested = _request_action("freeze", (state.freezer_state,))
        if requested is None or requested:
            return requested
    raise RuntimeError("target freezer state kept changing; fail-closed freeze was not issued")


def poll_once(owned_freeze: bool) -> bool | None:
    """Check one sample; return owned state, or None when the target has changed."""
    state = read_service_state()
    if not _is_target_active(state):
        return None
    if state.freezer_state not in {"running", "freezing", "frozen", "thawing"}:
        raise RuntimeError(f"unexpected FreezerState for {TARGET_UNIT}: {state.freezer_state}")

    try:
        temperature = read_cpu_package_temperature()
    except CpuSensorError as error:
        try:
            freeze_requested = _freeze_after_sensor_error()
        except (RuntimeError, subprocess.TimeoutExpired) as action_error:
            raise RuntimeError(
                f"{error}; unable to verify or freeze the matching target: {action_error}"
            ) from action_error
        if freeze_requested is None:
            return None
        if freeze_requested:
            detail = "requested a systemd freeze"
        else:
            detail = "left an already frozen or transitioning unit unchanged"
        raise CpuSensorError(f"{error}; fail-closed observer {detail} and will exit") from error

    if owned_freeze and state.freezer_state in {"running", "thawing"}:
        owned_freeze = False

    if temperature >= FREEZE_CELSIUS and not owned_freeze:
        if state.freezer_state == "running":
            action = _request_action("freeze", ("running",))
            if action is None:
                return None
            return action
        return owned_freeze

    if temperature <= THAW_CELSIUS and owned_freeze:
        if state.freezer_state == "frozen":
            action = _request_action("thaw", ("frozen",))
            if action is None:
                return None
            return False if action else owned_freeze
        if state.freezer_state == "freezing":
            return owned_freeze
    return owned_freeze


def watch_target() -> None:
    """Poll only the frozen target invocation, at one-second intervals."""
    owned_freeze = False
    LOGGER.info("watching %s invocation %s", TARGET_UNIT, TARGET_INVOCATION)
    while True:
        owned = poll_once(owned_freeze)
        if owned is None:
            LOGGER.info("target invocation ended or changed; observer exiting")
            return
        owned_freeze = owned
        time.sleep(POLL_SECONDS)


def main() -> int:
    """Run the single-invocation observer and report fail-closed errors."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        watch_target()
    except (CpuSensorError, RuntimeError, subprocess.TimeoutExpired) as error:
        LOGGER.critical("observer stopped: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
