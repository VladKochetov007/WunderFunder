"""Probe sibling scope shutdown using bounded CPU-only transient units."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Bound each systemd operation independently."""
    return subprocess.run(args, check=check, text=True, capture_output=True, timeout=5)


def properties(unit: str) -> dict[str, str]:
    """Read only this probe's transient unit properties."""
    result = command(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "ControlGroup",
            "-p",
            "RuntimeMaxUSec",
            "-p",
            "TimeoutStopUSec",
        ],
        check=False,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def main() -> None:
    started = time.monotonic()
    suffix = str(time.time_ns())
    service = f"wnn-independent-service-{suffix}.service"
    scope = f"wnn-independent-scope-{suffix}.scope"
    report: dict = {"service": service, "scope": scope, "scope_kind": "CPU-only sleep"}
    try:
        command(
            [
                "systemd-run",
                "--user",
                "--quiet",
                f"--unit={service}",
                "--slice=wnn-campaign.slice",
                "-p",
                "RuntimeMaxSec=1",
                "-p",
                "TimeoutStopSec=1",
                "-p",
                "MemoryMax=64M",
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                f"--unit={scope}",
                "--slice=wnn-campaign.slice",
                "-p",
                "RuntimeMaxSec=3",
                "-p",
                "MemoryMax=64M",
                "-p",
                "MemorySwapMax=0",
                "-p",
                "CPUQuota=10%",
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)",
            ]
        )
        time.sleep(0.25)
        report["initial_service"] = properties(service)
        report["initial_scope"] = properties(scope)
        time.sleep(4)
        report["after_service_and_scope_runtime"] = {
            "service": properties(service),
            "scope": properties(scope),
        }
        report["scope_survives_service_and_runtime"] = report["after_service_and_scope_runtime"][
            "scope"
        ].get("ActiveState") in {"active", "deactivating"}
    finally:
        for unit in (scope, service):
            command(["systemctl", "--user", "kill", "--signal=KILL", unit], check=False)
        report["elapsed_seconds"] = time.monotonic() - started
        Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
