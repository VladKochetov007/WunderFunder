from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts import tl3_finite17_dashboard_core as _core
    from scripts import tl3_finite17_github_posts as _posts
    from scripts import tl3_finite17_results as _results
    from scripts.galam_l4_quality8 import common as _common
    from scripts.galam_l4_quality8 import publisher as _publisher
except ModuleNotFoundError:
    import tl3_finite17_dashboard_core as _core
    import tl3_finite17_github_posts as _posts
    import tl3_finite17_results as _results
    from galam_l4_quality8 import common as _common
    from galam_l4_quality8 import publisher as _publisher

atomic_write_json = _common.atomic_write_json
GitHubClient = _publisher.GitHubClient
PublisherError = _publisher.PublisherError
TransientGitHubError = _publisher.TransientGitHubError
read_token = _publisher.read_token
BridgeError = _core.BridgeError
_canonical_sha256 = _core._canonical_sha256
_plan_and_split = _core._plan_and_split
MAX_HOURS = _posts.MAX_HOURS
POLL_SECONDS = _posts.POLL_SECONDS
RECEIPT_SCHEMA = _posts.RECEIPT_SCHEMA
_hourly_due = _posts._hourly_due
_hourly_event_key = _posts._hourly_event_key
_load_state = _posts._load_state
_local_json = _posts._local_json
_persist_state = _posts._persist_state
_post_event = _posts._post_event
_private_directory = _posts._private_directory
_status_record = _posts._status_record
_terminal_receipt = _posts._terminal_receipt
_write_status = _posts._write_status
build_snapshot = _results.build_snapshot


def _run_loop(
    root: Path,
    state_dir: Path,
    pr_number: int,
    output: Path,
    client: Any,
    poll_seconds: int,
    max_hours: int,
    clock: Any,
    sleeper: Any,
) -> None:
    _private_directory(output)
    receipt_path = output / "watcher_receipt.json"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "pr_number": pr_number,
        "root": str(root),
        "state_dir": str(state_dir),
        "created_epoch": clock(),
        "max_runtime_hours": max_hours,
    }
    if receipt_path.exists():
        previous = _local_json(receipt_path)
        for key in ("schema", "pr_number", "root", "state_dir", "max_runtime_hours"):
            if previous.get(key) != receipt[key]:
                raise PublisherError("existing watcher receipt belongs to another launch")
        receipt = previous
    else:
        atomic_write_json(receipt_path, receipt)
    started = float(receipt["created_epoch"])
    deadline = started + max_hours * 3600
    state_path = output / "watcher_state.json"
    state = _load_state(state_path, pr_number, started)
    if state["started_epoch"] != started:
        raise PublisherError("watcher receipt and mutable state start times differ")
    account_verified = False
    while clock() < deadline:
        now = clock()
        retry_at = state.get("retry_not_before")
        if isinstance(retry_at, (int, float)) and not isinstance(retry_at, bool) and now < retry_at:
            sleeper(min(poll_seconds, max(1.0, min(deadline, float(retry_at)) - now)))
            continue
        state.pop("retry_not_before", None)
        if not account_verified:
            try:
                client.verify_account(deadline)
                account_verified = True
            except TransientGitHubError as error:
                state["retry_not_before"] = now + max(float(error.retry_after), poll_seconds)
                _persist_state(state_path, state)
                _write_status(
                    output,
                    _status_record(
                        pr_number,
                        state,
                        "retry_wait",
                        now,
                        reason="GitHub account check transiently failed",
                    ),
                )
                sleeper(min(poll_seconds, max(1.0, deadline - now)))
                continue
            except PublisherError:
                _write_status(
                    output,
                    _status_record(
                        pr_number, state, "blocked", now, reason="GitHub account guard failed"
                    ),
                )
                raise
        loaded = None
        if state_dir.is_dir() and not state_dir.is_symlink():
            loaded = _plan_and_split(root, state_dir.resolve(strict=True))
        if loaded is None:
            _write_status(
                output,
                _status_record(pr_number, state, "waiting_for_plan", now),
            )
            sleeper(min(poll_seconds, max(1.0, deadline - now)))
            continue
        plan, driver_status, train, valid = loaded
        plan_sha256 = str(plan["sha256"])
        if state.get("plan_sha256") not in (None, plan_sha256):
            raise BridgeError("watcher state plan identity changed")
        state["plan_sha256"] = plan_sha256
        snapshot = build_snapshot(root, plan, driver_status, train, valid)
        _write_status(output, _status_record(pr_number, state, "running", now, snapshot))
        try:
            start_key = f"start:{plan_sha256}"
            if start_key not in state["posted_event_keys"]:
                _post_event(
                    client,
                    state_path,
                    state,
                    pr_number,
                    plan_sha256,
                    snapshot,
                    start_key,
                    "initial status",
                    deadline,
                    now,
                )
                state["last_hourly_epoch"] = now
                state["last_hourly_snapshot_sha256"] = _canonical_sha256(snapshot)
                _persist_state(state_path, state)
            elif state.get("last_hourly_epoch") is None:
                state["last_hourly_epoch"] = now
                state["last_hourly_snapshot_sha256"] = _canonical_sha256(snapshot)
                _persist_state(state_path, state)
            for node in snapshot["nodes"]:
                metrics = node.get("metrics")
                is_fit_finished = node.get("state") == "complete" or node.get("phase") in {
                    "export_pending",
                    "train_export_running",
                }
                if (
                    not node.get("fit_verified")
                    or not is_fit_finished
                    or not isinstance(metrics, dict)
                ):
                    continue
                event_key = f"fit-complete:{node['id']}:{metrics['model_sha256']}"
                _post_event(
                    client,
                    state_path,
                    state,
                    pr_number,
                    plan_sha256,
                    snapshot,
                    event_key,
                    f"fit verified: {node['id']}",
                    deadline,
                    now,
                    only_node=str(node["id"]),
                )
            # A terminal receipt may arrive after the snapshot was built. Wait
            # one poll so the final comment includes its verified blend score.
            terminal_sha = (
                None
                if snapshot.get("terminal_receipt_pending")
                else _terminal_receipt(state_dir, driver_status, plan)
            )
            if terminal_sha is not None:
                terminal_key = f"terminal:{driver_status['status']}:{terminal_sha}"
                _post_event(
                    client,
                    state_path,
                    state,
                    pr_number,
                    plan_sha256,
                    snapshot,
                    terminal_key,
                    f"DAG terminal: {driver_status['status']}",
                    deadline,
                    now,
                )
                _write_status(
                    output,
                    _status_record(pr_number, state, "terminal", now, snapshot),
                )
                return
            if _hourly_due(state, snapshot, now):
                snapshot_sha = _canonical_sha256(snapshot)
                key = _hourly_event_key(snapshot, now)
                _post_event(
                    client,
                    state_path,
                    state,
                    pr_number,
                    plan_sha256,
                    snapshot,
                    key,
                    "hourly changed status",
                    deadline,
                    now,
                )
                state["last_hourly_epoch"] = now
                state["last_hourly_snapshot_sha256"] = snapshot_sha
                _persist_state(state_path, state)
            _write_status(output, _status_record(pr_number, state, "running", now, snapshot))
        except TransientGitHubError as error:
            state["retry_not_before"] = now + max(float(error.retry_after), poll_seconds)
            _persist_state(state_path, state)
            _write_status(
                output,
                _status_record(pr_number, state, "retry_wait", now, snapshot),
            )
        sleeper(min(poll_seconds, max(1.0, deadline - now)))
    _write_status(
        output,
        _status_record(
            pr_number, state, "observer_expired", clock(), reason="168-hour bound reached"
        ),
    )


def run_watcher(
    root: Path,
    state_dir: Path,
    pr_number: int,
    token_path: Path,
    output: Path,
    poll_seconds: int = POLL_SECONDS,
    max_hours: int = MAX_HOURS,
    opener: Any = None,
    sleeper: Any = time.sleep,
    clock: Any = time.time,
) -> None:
    """Verify the authorized account and watch one exact finite17 plan for at most 168 hours."""
    if (
        not 1 <= pr_number <= 999_999
        or not 1 <= poll_seconds <= 300
        or not 1 <= max_hours <= MAX_HOURS
    ):
        raise PublisherError("PR number, poll interval or watcher lifetime is outside its bound")
    if sys.platform != "linux":
        raise PublisherError("finite17 GitHub watcher is available only on Linux")
    root = root.resolve(strict=True)
    state_dir = state_dir.expanduser().absolute()
    output = output.expanduser().absolute()
    token, _token_identity = read_token(token_path)
    client = GitHubClient(token, opener=opener, sleeper=sleeper, clock=clock)
    _run_loop(root, state_dir, pr_number, output, client, poll_seconds, max_hours, clock, sleeper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--max-runtime-hours", type=int, default=MAX_HOURS)
    args = parser.parse_args()
    try:
        run_watcher(
            args.root,
            args.state,
            args.pr_number,
            args.token_file,
            args.output,
            args.poll_seconds,
            args.max_runtime_hours,
        )
    except (
        BridgeError,
        PublisherError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        parser.exit(2, f"finite17 watcher stopped: {error}\n")


if __name__ == "__main__":
    main()
