from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from scripts.sequence_monitor_mirror_io import atomic_bytes, atomic_json
from scripts.sequence_progress_github import ACCOUNT, REPOSITORY, ProgressGitHub
from scripts.sequence_progress_panel import load_panel_manifest
from scripts.sequence_progress_report import canonical_sha, observe_host, render_report
from wnn.multistage_search import file_lock

# Controller state contains bounded native summaries and publication receipts, never credentials.
JsonObject = dict[str, Any]
LOGGER = logging.getLogger(__name__)
SCHEMA = "wnn.sequence-github-progress.v1"
PANEL_SECONDS = {"l4": 86400, "b300": 777600}


def _receipt(path: Path, value: JsonObject) -> None:
    """Retain a small append-only publication/error receipt without API headers."""
    if path.is_symlink():
        raise ValueError("report receipt path cannot be a symlink")
    with path.open("ab") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _record_post(
    state: JsonObject,
    entry: JsonObject,
    pending: JsonObject,
    comment: JsonObject,
    state_path: Path,
    receipts: Path,
    now: float,
    reconciled: bool,
) -> None:
    number = pending["pr"]
    record = {
        "kind": "reconciled" if reconciled else "posted",
        "at_unix": now,
        "pr": number,
        "comment_id": comment["id"],
        "verified_login": ACCOUNT,
        "comment_url": f"https://github.com/{REPOSITORY}/pull/{number}#issuecomment-{comment['id']}",
        "report_id": pending["report_id"],
        "body_sha256": pending["body_sha256"],
        "source_identities": pending.get("source_identities", {}),
        "panel": pending.get("panel"),
    }
    history = entry.setdefault("last_comments_by_panel", {})
    previous = entry.get("last_comment")
    if previous:
        previous_panel = previous.get("panel") or {}
        history[previous_panel.get("manifest_sha256", "original")] = previous
    panel = pending.get("panel") or {}
    history[panel.get("manifest_sha256", "original")] = record
    entry.update(
        last_posted_unix=now,
        last_terminal_signature=pending["terminal_signature"],
        last_comment=record,
    )
    entry.pop("pending", None)
    entry.pop("last_error", None)
    atomic_json(state_path, state)
    _receipt(receipts, record)


def report_once(
    root: Path,
    targets: dict[str, int],
    state: JsonObject,
    state_path: Path,
    client: ProgressGitHub,
    now: float,
    interval: int = 1800,
    panels: JsonObject | None = None,
) -> None:
    """Batch due terminal/periodic updates per PR, reconciling every pending marker."""
    receipts = state_path.with_name("github-report-receipts.jsonl")
    for host, number in targets.items():
        entry = state.setdefault("pull_requests", {}).setdefault(str(number), {"host": host})
        if entry["host"] != host:
            raise ValueError("report state binds this PR to another host panel")
        snapshot = observe_host(root, host, now, (panels or {}).get(host))
        pending = entry.get("pending")
        if pending is None:
            terminal_changed = (
                entry.get("last_terminal_signature") != snapshot["terminal_signature"]
            )
            if not terminal_changed and now - entry.get("last_posted_unix", 0) < interval:
                continue
            body = render_report(snapshot)
            report_id = canonical_sha({"repository": REPOSITORY, "pr": number, "body": body})
            marker = f"<!-- wnn-progress:{number}:{report_id} -->"
            pending = {
                "pr": number,
                "host": host,
                "prepared_at_unix": now,
                "report_id": report_id,
                "terminal_signature": snapshot["terminal_signature"],
                "marker": marker,
                "body": body + "\n\n" + marker,
                "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "source_identities": {
                    row["cell"]: row["ids"] for row in snapshot["cells"] if row.get("ids")
                },
                "panel": snapshot.get("panel"),
            }
            entry["pending"] = pending
            atomic_json(state_path, state)
            atomic_bytes(
                state_path.with_name(f"github-report-pr{number}.md"), pending["body"].encode()
            )
        if now < pending.get("next_reconcile_unix", 0):
            continue
        try:
            existing = client.find_report(number, pending["marker"], pending["prepared_at_unix"])
            if existing is not None:
                _record_post(state, entry, pending, existing, state_path, receipts, now, True)
                continue
            if pending.get("post_attempted_unix") is not None:
                pending["next_reconcile_unix"] = now + 300
                entry["last_error"] = "Previous POST remains unconfirmed; no automatic repeat POST."
                atomic_json(state_path, state)
                continue

            def before_post(pending_report: JsonObject = pending) -> None:
                """Durably record the irreversible attempt immediately before its HTTP write."""
                pending_report["post_attempted_unix"] = now
                atomic_json(state_path, state)

            comment = client.post_report(number, pending["body"], before_post=before_post)
            _record_post(state, entry, pending, comment, state_path, receipts, now, False)
            entry.pop("last_error", None)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            # The client exposes fixed-origin status errors, never response bodies or auth.
            entry["last_error"] = {
                "at_unix": now,
                "type": type(exc).__name__,
                "message": str(exc)[:200],
            }
            pending["next_reconcile_unix"] = (
                now + 300 if pending.get("post_attempted_unix") else now + 30
            )
            _receipt(
                receipts, {"kind": "error", "at_unix": now, "pr": number, **entry["last_error"]}
            )
            atomic_json(state_path, state)
            LOGGER.error("PR %s report failed (%s)", number, type(exc).__name__)


def main() -> int:
    """Run a bounded local reporter; dry runs never create a credential/API client."""
    parser = argparse.ArgumentParser(
        description="Report cached fit progress to ROOT-selected GitHub PRs"
    )
    parser.add_argument("--mirror-config", required=True, type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--l4-pr", type=int)
    parser.add_argument("--b300-pr", type=int)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=86400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    targets = {
        host: number
        for host, number in (("l4", args.l4_pr), ("b300", args.b300_pr))
        if number is not None
    }
    if (
        not targets
        or any(number <= 0 for number in targets.values())
        or len(set(targets.values())) != len(targets)
        or not 300 <= args.interval_seconds <= 3600
        or not 5 <= args.poll_seconds <= 60
        or not 1 <= args.max_seconds <= 777600
    ):
        parser.error("supply distinct positive PR IDs; interval300..3600, poll5..60, max1..777600")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    config = json.loads(args.mirror_config.read_bytes())
    panels = load_panel_manifest(args.panel_manifest) if args.panel_manifest else {}
    if set(panels) - set(targets):
        parser.error("mapped panels require their corresponding --l4-pr/--b300-pr")
    root = Path(config["native_root"]).resolve(strict=True)
    state_path = args.state or args.mirror_config.with_name("github-report-state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        reports = {
            host: render_report(observe_host(root, host, time.time(), panels.get(host)))
            for host in targets
        }
        for host, body in reports.items():
            atomic_bytes(
                state_path.with_name(f"github-report-pr{targets[host]}-dry-run.md"), body.encode()
            )
            sys_body = f"PR {targets[host]} (dry run; no network)\n{body}\n"
            os.write(1, sys_body.encode())
        atomic_json(
            state_path.with_name("github-report-dry-run.json"),
            {"dry_run": True, "targets": targets, "reports": reports},
        )
        return 0
    lock_path = state_path.with_suffix(".lock")
    if state_path.is_symlink() or lock_path.is_symlink():
        raise ValueError("report state/lock cannot be symlinks")
    with file_lock(lock_path, blocking=False):
        state = (
            json.loads(state_path.read_bytes())
            if state_path.exists()
            else {"schema": SCHEMA, "repository": REPOSITORY, "account": ACCOUNT}
        )
        if (
            state.get("schema") != SCHEMA
            or state.get("repository") != REPOSITORY
            or state.get("account") != ACCOUNT
        ):
            raise ValueError("report state belongs to another publication scope")
        started = state.setdefault("started_unix", time.time())
        if not isinstance(started, (int, float)):
            raise ValueError("report state has an invalid persistent start clock")
        deadline = time.monotonic() + max(
            0, min(args.max_seconds, started + args.max_seconds - time.time())
        )
        atomic_json(state_path, state)
        client = ProgressGitHub(set(targets.values()), deadline)
        while time.monotonic() < deadline:
            now = time.time()
            active = {
                host: number
                for host, number in targets.items()
                if now < started + min(PANEL_SECONDS[host], args.max_seconds)
            }
            if not active:
                return 0
            report_once(root, active, state, state_path, client, now, args.interval_seconds, panels)
            if args.once:
                return int(
                    any(
                        entry.get("last_error") for entry in state.get("pull_requests", {}).values()
                    )
                )
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
