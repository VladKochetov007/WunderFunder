from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from b300_wave_evidence import load_panel, snapshot
from b300_wave_report import render
from l4_wave_publish import publish

from scripts.sequence_monitor_mirror_io import atomic_bytes, atomic_json
from scripts.sequence_progress_github import ACCOUNT, REPOSITORY, ProgressGitHub
from wnn.multistage_search import file_lock

SCHEMA = "wnn.objective-b300-github-progress.v1"


def main() -> int:
    """Publish only ROOT-pinned four-cell progress, using the tested existing API/pending journal."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-config", required=True, type=Path)
    parser.add_argument("--receiver-plan", required=True, type=Path)
    parser.add_argument("--receiver-plan-sha256", required=True)
    parser.add_argument("--pr", type=int, default=70)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.pr <= 0
        or not 300 <= args.interval_seconds <= 3600
        or not 5 <= args.poll_seconds <= 60
        or not 1 <= args.max_seconds <= 86400
    ):
        parser.error("positive PR; interval300..3600, poll5..60, max1..86400 required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    panel = load_panel(args.receiver_plan, args.receiver_plan_sha256)
    root = Path(json.loads(args.mirror_config.read_bytes())["native_root"]).resolve(strict=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        body = render(snapshot(root, panel, time.time()))
        atomic_bytes(args.state.with_name(f"b300-wave-pr{args.pr}-dry-run.md"), body.encode())
        os.write(1, (body + "\n").encode())
        return 0
    lock = args.state.with_suffix(".lock")
    if args.state.is_symlink() or lock.is_symlink():
        raise ValueError("report state/lock cannot be a symlink")
    with file_lock(lock, blocking=False):
        state = (
            json.loads(args.state.read_bytes())
            if args.state.exists()
            else {
                "schema": SCHEMA,
                "repository": REPOSITORY,
                "account": ACCOUNT,
                "pr": args.pr,
                "panel_sha256": panel["sha256"],
                "started_unix": time.time(),
            }
        )
        if (
            state.get("schema") != SCHEMA
            or state.get("repository") != REPOSITORY
            or state.get("account") != ACCOUNT
            or state.get("pr") != args.pr
            or state.get("panel_sha256") != panel["sha256"]
        ):
            raise ValueError("state belongs to another publication scope or frozen panel")
        started = state["started_unix"]
        if (
            type(started) not in (int, float)
            or not math.isfinite(started)
            or started > time.time() + 30
        ):
            raise ValueError("persistent reporting start clock is invalid")
        deadline = time.monotonic() + max(
            0, min(args.max_seconds, started + args.max_seconds - time.time())
        )
        client = ProgressGitHub({args.pr}, deadline)
        atomic_json(args.state, state)
        while time.monotonic() < deadline:
            now = time.time()
            value = snapshot(root, panel, now)
            publish(
                value,
                args.pr,
                state,
                args.state,
                client,
                now,
                args.interval_seconds,
                body=render(value),
            )
            if args.once:
                return int(
                    bool(state.get("pull_requests", {}).get(str(args.pr), {}).get("last_error"))
                )
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
