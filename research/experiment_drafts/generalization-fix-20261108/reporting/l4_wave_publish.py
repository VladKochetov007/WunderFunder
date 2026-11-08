from __future__ import annotations

import hashlib
from pathlib import Path

from l4_wave_report import render

from scripts.report_sequence_progress import _receipt, _record_post
from scripts.sequence_monitor_mirror_io import atomic_bytes, atomic_json
from scripts.sequence_progress_github import REPOSITORY, ProgressGitHub
from scripts.sequence_progress_report import canonical_sha


def publish(
    value: dict,
    number: int,
    state: dict,
    path: Path,
    client: ProgressGitHub,
    now: float,
    interval: int = 1800,
    *,
    body: str | None = None,
) -> None:
    """Reuse account-verified publication and durable pending-marker deduplication.

    An uncertain POST is reconciled by marker, never automatically sent again.
    This separate state and receipt directory never modifies old PR reporters.
    """
    entry = state.setdefault("pull_requests", {}).setdefault(str(number), {"host": value["host"]})
    if entry["host"] != value["host"]:
        raise ValueError("this reporter state belongs to another panel")
    receipts = path.with_name("l4-wave-publication-receipts.jsonl")
    pending = entry.get("pending")
    if pending is None:
        if (
            entry.get("last_terminal_signature") == value["terminal_signature"]
            and now - entry.get("last_posted_unix", 0) < interval
        ):
            return
        body = render(value) if body is None else body
        report_id = canonical_sha({"repository": REPOSITORY, "pr": number, "body": body})
        marker = f"<!-- wnn-generalization-progress:{number}:{report_id} -->"
        pending = {
            "pr": number,
            "host": value["host"],
            "prepared_at_unix": now,
            "report_id": report_id,
            "terminal_signature": value["terminal_signature"],
            "marker": marker,
            "body": body + "\n\n" + marker,
            "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "source_identities": {
                row["cell"]: row["ids"] for row in value["cells"] if row.get("ids")
            },
            "panel": value["panel"],
        }
        entry["pending"] = pending
        atomic_json(path, state)
        atomic_bytes(path.with_name(f"l4-wave-pr{number}.md"), pending["body"].encode())
    if now < pending.get("next_reconcile_unix", 0):
        return
    try:
        existing = client.find_report(number, pending["marker"], pending["prepared_at_unix"])
        if existing is not None:
            _record_post(state, entry, pending, existing, path, receipts, now, True)
            return
        if pending.get("post_attempted_unix") is not None:
            pending["next_reconcile_unix"] = now + 300
            entry["last_error"] = "Previous POST remains unconfirmed; no automatic repeat POST."
            atomic_json(path, state)
            return

        def before_post() -> None:
            pending["post_attempted_unix"] = now
            atomic_json(path, state)

        comment = client.post_report(number, pending["body"], before_post=before_post)
        _record_post(state, entry, pending, comment, path, receipts, now, False)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        entry["last_error"] = {
            "at_unix": now,
            "type": type(error).__name__,
            "message": str(error)[:200],
        }
        pending["next_reconcile_unix"] = now + (300 if pending.get("post_attempted_unix") else 30)
        atomic_json(path, state)
        _receipt(receipts, {"kind": "error", "pr": number, **entry["last_error"]})
