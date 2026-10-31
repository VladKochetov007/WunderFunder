"""Stop only this cohort if live validation fails, preserving the last safe epoch."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def main() -> None:
    """Prevent unattended fitting without a working per-epoch evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    selected = args.root / "SELECTED_PLAN.json"
    if not selected.exists():
        return
    plan_path = Path(json.loads(selected.read_text())["path"])
    plan = json.loads(plan_path.read_text())
    state_path = Path(plan["run_root"]) / "STATE.json"
    if not state_path.exists() or json.loads(state_path.read_text()).get("status") != "fit":
        return
    claim = json.loads((args.root / "FIT_ONCE.json").read_text())
    pid = int(claim["pid"])
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return
    command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    if "compact-reg-20261026/run_panel.py" not in command or str(plan_path) not in command:
        return
    monitor = args.root / "observer-b128/MONITOR.json"
    reason = None
    if (args.root / "METRIC_VERIFICATION_FAILED.json").exists():
        reason = "independent metric verification failed"
    if not monitor.exists():
        if time.time() - claim["created_unix"] > 120:
            reason = "validation monitor did not initialize"
    elif time.time() - monitor.stat().st_mtime > 120:
        reason = "validation monitor heartbeat stale for over120seconds"
    else:
        observation = json.loads(monitor.read_text())
        failures = [
            (name, epoch, item["status"])
            for name, job in observation["jobs"].items()
            for epoch, item in job["epochs"].items()
            if item["status"]
            in {"failed_preserved", "blocked_preserved", "terminated_at_monitor_deadline"}
        ]
        if failures:
            reason = "validation failed: " + str(failures)
    if reason is not None:
        receipt = {
            "unix": time.time(),
            "reason": reason,
            "controller_pid": pid,
            "action": "SIGTERM owned controller; retain safe checkpoints and all evidence",
        }
        (args.root / "WATCHDOG_STOP.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.kill(pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
