"""Preview, prepare and identity-checked freeze of the original R01 controller."""

from __future__ import annotations

import hashlib
import os
import signal
import time
from pathlib import Path
from typing import Any

from adoption import reconcile_state, validate_initial_takeover
from durability import write_json_fsync
from frozen_panel import read_object, validate_panel
from proc_snapshot import read_process, require_controller_identity


def reconcile_authorized(authority: dict[str, Any], *, stopped: bool) -> dict[str, Any]:
    """Read the exact old unit and filesystem without changing either."""
    root = Path(authority["root"])
    packet = Path(authority["packet_root"])
    state = Path(authority["old_state"])
    return reconcile_state(
        root=root,
        packet=packet,
        state=state,
        unit=authority["old_unit"],
        main_pid=authority["old_main_pid"],
        main_start_ticks=authority["old_main_start_ticks"],
        main_cgroup=authority["old_main_cgroup_path"],
        main_cmdline_sha256=authority["main_cmdline_sha256"],
        main_cmd_marker=str(packet / "execute.py"),
        must_be_stopped=stopped,
    )


def prepare_state(authority: dict[str, Any]) -> dict[str, Any]:
    """Create a fresh isolated successor state after read-only initial-wave checks."""
    destination = Path(authority["successor_state_path"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"successor state already exists: {destination}")
    snapshot = reconcile_authorized(authority, stopped=False)
    jobs = validate_panel(
        read_object(Path(authority["packet_root"]) / "PANEL.json"),
        Path(authority["root"]),
    )
    validate_initial_takeover(snapshot, jobs, require_stopped=False)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "receipts", "tmp"):
        (destination / name).mkdir()
    receipt = {
        "schema": "tabdata-r01-density4-prepare.v1",
        "status": "prepared_read_only",
        "prepared_unix": time.time(),
        "old_unit": snapshot["unit"],
        "old_main": snapshot["main"],
        "old_journal_sha256": snapshot["journal_sha256"],
        "panel_sha256": snapshot["panel_sha256"],
        "claimed": snapshot["claimed"],
        "pending": snapshot["pending"],
        "live_ids": sorted(snapshot["live"]),
        "successor_state_path": str(destination),
    }
    write_json_fsync(destination / "PREPARE.json", receipt)
    return receipt


def freeze_old_controller(authority: dict[str, Any]) -> dict[str, Any]:
    """Stop only the exact original main PID after rechecking the full seed-1003 wave."""
    state = Path(authority["successor_state_path"])
    prepared = read_object(state / "PREPARE.json")
    if prepared.get("status") != "prepared_read_only" or prepared.get(
        "successor_state_path"
    ) != str(state):
        raise ValueError("successor PREPARE receipt does not authorize this state directory")
    before = reconcile_authorized(authority, stopped=False)
    jobs = validate_panel(
        read_object(Path(authority["packet_root"]) / "PANEL.json"),
        Path(authority["root"]),
    )
    validate_initial_takeover(before, jobs, require_stopped=False)
    intent = {
        "schema": "tabdata-r01-density4-freeze-intent.v1",
        "time_unix": time.time(),
        "main": before["main"],
        "journal_sha256": before["journal_sha256"],
        "live_ids": sorted(before["live"]),
    }
    write_json_fsync(state / "FREEZE_INTENT.json", intent)
    pid = int(authority["old_main_pid"])
    stopped = False
    signal_sent = False
    pidfd: int | None = None
    try:
        current = read_process(pid)
        require_controller_identity(
            current,
            expected_pid=pid,
            expected_start_ticks=authority["old_main_start_ticks"],
            expected_unit=authority["old_unit"],
            command_marker=str(Path(authority["packet_root"]) / "execute.py"),
            must_be_stopped=False,
        )
        if (
            current.cgroup_path != authority["old_main_cgroup_path"]
            or current.cmdline_sha256 != authority["main_cmdline_sha256"]
        ):
            raise ValueError("old R01 controller changed after the pre-freeze snapshot")
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            pidfd = os.pidfd_open(pid, 0)
            current = read_process(pid)
            require_controller_identity(
                current,
                expected_pid=pid,
                expected_start_ticks=authority["old_main_start_ticks"],
                expected_unit=authority["old_unit"],
                command_marker=str(Path(authority["packet_root"]) / "execute.py"),
                must_be_stopped=False,
            )
            if (
                current.cgroup_path != authority["old_main_cgroup_path"]
                or current.cmdline_sha256 != authority["main_cmdline_sha256"]
            ):
                raise ValueError("old R01 controller changed after pidfd was opened")
            signal.pidfd_send_signal(pidfd, signal.SIGSTOP)
        else:
            os.kill(pid, signal.SIGSTOP)
        signal_sent = True
        limit = time.monotonic() + 10
        while time.monotonic() < limit:
            current = read_process(pid)
            if current.identity.state in {"T", "t"}:
                stopped = True
                break
            time.sleep(0.05)
        if not stopped:
            raise TimeoutError("old R01 controller did not enter stopped state")
        after = reconcile_authorized(authority, stopped=True)
        validate_initial_takeover(after, jobs, require_stopped=True)
        if after["main"]["cmdline_sha256"] != intent["main"]["cmdline_sha256"]:
            raise ValueError("old main command line changed across the freeze operation")
        receipt = {
            "schema": "tabdata-r01-density4-freeze-receipt.v1",
            "status": "old_main_stopped_exactly",
            "freeze_intent_sha256": hashlib.sha256(
                (state / "FREEZE_INTENT.json").read_bytes()
            ).hexdigest(),
            "frozen_unix": time.time(),
            "snapshot": after,
        }
        write_json_fsync(state / "FREEZE.json", receipt)
        return receipt
    except BaseException:
        if signal_sent:
            try:
                if pidfd is not None:
                    signal.pidfd_send_signal(pidfd, signal.SIGCONT)
                else:
                    current = read_process(pid)
                    require_controller_identity(
                        current,
                        expected_pid=pid,
                        expected_start_ticks=authority["old_main_start_ticks"],
                        expected_unit=authority["old_unit"],
                        command_marker=str(Path(authority["packet_root"]) / "execute.py"),
                        must_be_stopped=True,
                    )
                    if (
                        current.cgroup_path == authority["old_main_cgroup_path"]
                        and current.cmdline_sha256 == authority["main_cmdline_sha256"]
                    ):
                        os.kill(pid, signal.SIGCONT)
            except (OSError, ValueError, ProcessLookupError):
                pass
        raise
    finally:
        if pidfd is not None:
            os.close(pidfd)
