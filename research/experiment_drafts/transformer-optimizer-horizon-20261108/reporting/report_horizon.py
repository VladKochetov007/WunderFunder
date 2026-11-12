from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from horizon_progress import PLAN_SHA256, load_panel, object_file, render, snapshot

PACKET = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_WORKTREE = PACKET.parents[3] / "WunderFunder-generalization-fix-20261108"


def _authority(path: Path, expected_sha256: str) -> dict:
    """Bind reports to the existing first-profile clock; never start a new timer."""
    import hashlib

    if path.is_symlink() or not path.is_file():
        raise ValueError("ROOT authority must be a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("ROOT authority bytes differ from the frozen receiving plan")
    value = json.loads(payload)
    first = 1794168764.8507051
    if (
        value.get("approved_by") != "ROOT"
        or value.get("source_commit") != "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
        or value.get("first_profile_epoch") != first
        or value.get("fit_deadline_epoch") - first != 25200
        or value.get("deadline_epoch") - first != 28800
        or value.get("fit_cap") != 6
        or value.get("profile_cap") != 6
        or value.get("collection_reserve_seconds") != 3600
    ):
        raise ValueError("ROOT authority is not the fixed source-bound eight-hour cohort")
    return value


def _shared_helpers(root: Path):
    """Load only the existing Semyon-verified client and pending-journal publisher."""
    root = root.resolve(strict=True)
    reporting = root / "research/experiment_drafts/generalization-fix-20261108/reporting"
    required = (
        root / "scripts/sequence_progress_github.py",
        root / "scripts/report_sequence_progress.py",
        reporting / "l4_wave_publish.py",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("existing shared GitHub reporting helpers are incomplete")
    for item in (str(root / "src"), str(root), str(reporting)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from l4_wave_publish import publish
    from scripts.sequence_monitor_mirror_io import atomic_json
    from scripts.sequence_progress_github import ACCOUNT, REPOSITORY, ProgressGitHub
    from wnn.multistage_search import file_lock

    return publish, atomic_json, ACCOUNT, REPOSITORY, ProgressGitHub, file_lock


def main() -> int:
    """Render cached progress or publish bounded updates without owning fit/RPC lifecycles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-config", type=Path, required=True)
    parser.add_argument("--receiver-plan", type=Path, required=True)
    parser.add_argument("--receiver-plan-sha256", default=PLAN_SHA256)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--readout-bundle", type=Path)
    parser.add_argument("--shared-worktree", type=Path, default=DEFAULT_SHARED_WORKTREE)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.pr <= 0 or not 300 <= args.interval_seconds <= 3600 or not 5 <= args.poll_seconds <= 60:
        parser.error("positive PR, interval300..3600 and poll5..60 required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    panel = load_panel(args.receiver_plan, args.receiver_plan_sha256)
    if args.authority.is_symlink() or not args.authority.is_file():
        raise ValueError("ROOT authority must be a regular file")
    authority = _authority(args.authority, panel["plan"]["authority_sha256"])
    mirror = object_file(args.mirror_config)
    native_root = Path(mirror["native_root"]).resolve(strict=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        value = snapshot(native_root, panel, args.readout_bundle, time.time())
        body = render(value)
        destination = args.state.with_name(f"horizon-pr{args.pr}-dry-run.md")
        if destination.is_symlink():
            raise ValueError("dry-run report path cannot be a symlink")
        destination.write_text(body, encoding="utf-8")
        os.write(1, (body + "\n").encode("utf-8"))
        return 0

    if args.state.is_symlink():
        raise ValueError("report state path cannot be a symlink")
    publish, atomic_json, account, repository, client_type, file_lock = _shared_helpers(
        args.shared_worktree
    )
    lock = args.state.with_suffix(".lock")
    if lock.is_symlink():
        raise ValueError("report lock path cannot be a symlink")
    with file_lock(lock, blocking=False):
        state = (
            json.loads(args.state.read_bytes())
            if args.state.exists()
            else {
                "schema": "wnn.transformer-optimizer-horizon-progress.v1",
                "repository": repository,
                "account": account,
                "pr": args.pr,
                "panel_sha256": panel["sha256"],
                "authority_sha256": panel["plan"]["authority_sha256"],
                "started_unix": time.time(),
            }
        )
        if (
            state.get("schema") != "wnn.transformer-optimizer-horizon-progress.v1"
            or state.get("repository") != repository
            or state.get("account") != account
            or state.get("pr") != args.pr
            or state.get("panel_sha256") != panel["sha256"]
            or state.get("authority_sha256") != panel["plan"]["authority_sha256"]
        ):
            raise ValueError("publication state belongs to another PR, panel or authority")
        deadline = time.monotonic() + max(0.0, authority["deadline_epoch"] - time.time())
        client = client_type({args.pr}, deadline)
        atomic_json(args.state, state)
        while time.monotonic() < deadline:
            now = time.time()
            value = snapshot(native_root, panel, args.readout_bundle, now)
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
            time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
