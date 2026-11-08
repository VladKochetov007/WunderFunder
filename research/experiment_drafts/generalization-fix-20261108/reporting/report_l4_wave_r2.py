from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from l4_readout_mirror_r2 import ReadoutMirror
from l4_wave_publish import publish
from l4_wave_readouts import admission
from l4_wave_report import render, snapshot

from scripts.sequence_monitor_mirror_io import atomic_bytes, atomic_json
from scripts.sequence_progress_github import ACCOUNT, REPOSITORY, ProgressGitHub
from wnn.multistage_search import file_lock

SCHEMA = "wnn.generalization-l4-github-progress.v1"


def main() -> int:
    """Report the fixed six-cell wave; dry run creates no credential/API/RPC client."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-config", type=Path, required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--readout-root", type=Path)
    parser.add_argument("--readout-admission", type=Path)
    parser.add_argument("--rpc-root", type=Path)
    parser.add_argument("--native-run-root")
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
    if bool(args.rpc_root) != bool(args.native_run_root) or (
        args.rpc_root and not args.readout_root
    ):
        parser.error(
            "RPC collection requires --rpc-root, --native-run-root and --readout-root together"
        )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    config = json.loads(args.mirror_config.read_bytes())
    root = Path(config["native_root"]).resolve(strict=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    approved = admission(args.readout_admission)
    if args.dry_run:
        body = render(snapshot(root, args.readout_root, approved, time.time()))
        atomic_bytes(args.state.with_name(f"l4-wave-pr{args.pr}-dry-run.md"), body.encode())
        os.write(1, (body + "\n").encode())
        return 0
    lock = args.state.with_suffix(".lock")
    if args.state.is_symlink() or lock.is_symlink():
        raise ValueError("reporter state/lock cannot be a symlink")
    with file_lock(lock, blocking=False):
        state = (
            json.loads(args.state.read_bytes())
            if args.state.exists()
            else {
                "schema": SCHEMA,
                "repository": REPOSITORY,
                "account": ACCOUNT,
                "pr": args.pr,
                "started_unix": time.time(),
            }
        )
        if (
            state.get("schema") != SCHEMA
            or state.get("repository") != REPOSITORY
            or state.get("account") != ACCOUNT
            or state.get("pr") != args.pr
        ):
            raise ValueError("state belongs to another publication scope")
        started = state["started_unix"]
        if (
            type(started) not in (int, float)
            or not math.isfinite(started)
            or started > time.time() + 30
        ):
            raise ValueError("reporter persistent start clock is invalid")
        deadline = time.monotonic() + max(
            0, min(args.max_seconds, started + args.max_seconds - time.time())
        )
        client = ProgressGitHub({args.pr}, deadline)
        mirror = (
            ReadoutMirror(
                args.rpc_root,
                args.native_run_root,
                args.readout_root,
                args.state.with_name("l4-wave-readout-mirror.json"),
            )
            if args.rpc_root
            else None
        )
        atomic_json(args.state, state)
        while time.monotonic() < deadline:
            now = time.time()
            if mirror is not None:
                try:
                    mirror.tick(now)
                    state.pop("readout_mirror_error", None)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    state["readout_mirror_error"] = {
                        "at_unix": now,
                        "type": type(error).__name__,
                        "message": str(error)[:200],
                    }
            value = snapshot(root, args.readout_root, approved, now)
            publish(value, args.pr, state, args.state, client, now, args.interval_seconds)
            atomic_json(args.state, state)
            if args.once:
                return int(
                    bool(
                        state.get("readout_mirror_error")
                        or state.get("pull_requests", {}).get(str(args.pr), {}).get("last_error")
                    )
                )
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
