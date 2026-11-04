from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from scripts import tl3_finite17_dashboard_core as _core
    from scripts import tl3_finite17_dashboard_process as _process
    from scripts.tl3_finite17_dashboard_publish import publish_once
except ModuleNotFoundError:
    import tl3_finite17_dashboard_core as _core
    import tl3_finite17_dashboard_process as _process
    from tl3_finite17_dashboard_publish import publish_once

BridgeError = _core.BridgeError
PLAN_SCHEMA = _core.PLAN_SCHEMA
STATE_SCHEMA = _core.STATE_SCHEMA
_canonical_sha256 = _core._canonical_sha256
_read_json = _core._read_json
_sha256_file = _core._sha256_file
_owned_process = _process._owned_process


def run_bridge(root: Path, state: Path, runs: Path, poll: int, startup: int, hours: int) -> int:
    """Poll until terminal driver status or a fixed lifetime bound."""
    if not sys.platform.startswith("linux"):
        raise BridgeError("native finite17 bridge requires Linux /proc")
    if not 1 <= poll <= 60 or not 1 <= hours <= 168 or startup < 1:
        raise BridgeError("poll interval or bridge lifetime is outside its bound")
    started, startup_limit = time.monotonic(), time.monotonic() + startup
    while time.monotonic() - started < hours * 3600:
        result = publish_once(root, state, runs)
        status = result["status"]
        if status == "completed":
            return 0
        if status in {
            "failed",
            "budget_exhausted",
            "interrupted",
            "finished_with_failures",
            "disk_capacity_blocked",
        }:
            return 1
        if status not in {"waiting_for_plan", "running"}:
            raise BridgeError(f"finite driver returned an unknown status: {status}")
        if status == "waiting_for_plan" and time.monotonic() >= startup_limit:
            raise BridgeError("finite plan/status did not appear before startup timeout")
        time.sleep(poll)
    raise BridgeError("native finite17 bridge reached its runtime bound")


def main() -> None:
    """Run the report-only native finite17 dashboard broker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--dashboard-runs-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-runtime-hours", type=int, default=168)
    args = parser.parse_args()
    try:
        code = run_bridge(
            args.root,
            args.state,
            args.dashboard_runs_root,
            args.poll_seconds,
            args.startup_timeout_seconds,
            args.max_runtime_hours,
        )
    except (BridgeError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"native TL3 dashboard bridge blocked: {error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
