from __future__ import annotations

import hashlib
import os
import re
import stat
import urllib.parse
from pathlib import Path
from typing import Any

try:
    from scripts.galam_l4_quality8.common import atomic_write_json
    from scripts.galam_l4_quality8.publisher import REPO, PublisherError
    from scripts.tl3_finite17_dashboard_core import (
        BridgeError,
        _canonical_sha256,
        _read_json,
        _sha256_file,
    )
    from scripts.tl3_finite17_results import TERMINAL_STATUSES, format_update
except ModuleNotFoundError:
    from galam_l4_quality8.common import atomic_write_json
    from galam_l4_quality8.publisher import REPO, PublisherError
    from tl3_finite17_dashboard_core import BridgeError, _canonical_sha256, _read_json, _sha256_file
    from tl3_finite17_results import TERMINAL_STATUSES, format_update

STATE_SCHEMA = "wnn.tl3-finite17-github-watcher.v1"
RECEIPT_SCHEMA = "wnn.tl3-finite17-github-watcher-receipt.v1"
POLL_SECONDS = 60
MAX_HOURS = 168
COMMENT_ID = re.compile(r"issuecomment-[0-9]+\Z")


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise PublisherError("watcher output cannot be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PublisherError("watcher output must be owned and private (mode 0700)")


def _local_json(path: Path, maximum: int = 1 << 20) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise PublisherError(f"watcher receipt is missing, unsafe or oversized: {path.name}")
    return _read_json(path, maximum)


def _initial_state(pr_number: int, started: float) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "pr_number": pr_number,
        "started_epoch": started,
        "plan_sha256": None,
        "last_hourly_epoch": None,
        "last_hourly_snapshot_sha256": None,
        "posted_event_keys": [],
        "pending_event_keys": [],
        "posts": [],
    }


def _load_state(path: Path, pr_number: int, started: float) -> dict[str, Any]:
    if not path.exists():
        return _initial_state(pr_number, started)
    state = _local_json(path)
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("pr_number") != pr_number
        or not isinstance(state.get("posted_event_keys"), list)
        or not isinstance(state.get("pending_event_keys"), list)
        or not isinstance(state.get("posts"), list)
        or type(state.get("started_epoch")) not in (int, float)
        or len(state["posted_event_keys"]) > 256
        or len(state["pending_event_keys"]) > 16
        or len(state["posts"]) > 256
        or len(set(state["posted_event_keys"])) != len(state["posted_event_keys"])
        or len(set(state["pending_event_keys"])) != len(state["pending_event_keys"])
    ):
        raise PublisherError("watcher state belongs to another plan or has invalid fields")
    return state


def _write_status(output: Path, status: dict[str, Any]) -> None:
    status_path = output / "watcher_status.json"
    if status_path.is_symlink():
        raise PublisherError("watcher status path cannot be a symlink")
    atomic_write_json(status_path, status)


def _status_record(
    pr_number: int,
    state: dict[str, Any],
    watcher_status: str,
    now: float,
    snapshot: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "pr_number": pr_number,
        "status": watcher_status,
        "plan_sha256": state.get("plan_sha256"),
        "updated_epoch": now,
        "posted_events": len(state.get("posted_event_keys", [])),
        "pending_events": len(state.get("pending_event_keys", [])),
    }
    if snapshot is not None:
        record.update(
            {
                "driver_status": snapshot["driver_status"],
                "fits_used": snapshot["fits_used"],
                "planned_fits": snapshot["planned_fits"],
                "fit_states": snapshot["fit_states"],
                "snapshot_sha256": _canonical_sha256(snapshot),
            }
        )
    if reason is not None:
        record["reason"] = reason
    return record


def _event_marker(plan_sha256: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"<!-- tl3-finite17:{plan_sha256}:{digest} -->"


def _issue_comments_path(pr_number: int, page: int) -> str:
    query = urllib.parse.urlencode(
        {"per_page": 100, "sort": "created", "direction": "desc", "page": page}
    )
    return f"/repos/{REPO}/issues/{pr_number}/comments?{query}"


def _comment_url(value: object, pr_number: int) -> str:
    if not isinstance(value, str):
        raise PublisherError("GitHub returned a malformed PR comment URL")
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.path not in (f"/{REPO}/pull/{pr_number}", f"/{REPO}/issues/{pr_number}")
        or COMMENT_ID.fullmatch(parsed.fragment) is None
    ):
        raise PublisherError("GitHub comment URL is outside the requested PR")
    return value


def _find_posted_comment(client: Any, pr_number: int, marker: str, deadline: float) -> str | None:
    """Search a bounded comment history to recover a POST interrupted before local receipt."""
    for page in range(1, 11):
        result = client.request("GET", _issue_comments_path(pr_number, page), None, deadline)
        if not isinstance(result, list):
            raise PublisherError("GitHub returned a malformed PR comment page")
        for comment in result:
            if isinstance(comment, dict) and marker in str(comment.get("body", "")):
                return _comment_url(comment.get("html_url"), pr_number)
        if len(result) < 100:
            break
    return None


def _persist_state(path: Path, state: dict[str, Any]) -> None:
    if (
        len(state.get("posted_event_keys", [])) > 256
        or len(state.get("pending_event_keys", [])) > 16
        or len(state.get("posts", [])) > 256
    ):
        raise PublisherError("watcher event receipt exceeded its fixed bound")
    atomic_write_json(path, state)


def _post_event(
    client: Any,
    state_path: Path,
    state: dict[str, Any],
    pr_number: int,
    plan_sha256: str,
    snapshot: dict[str, Any],
    key: str,
    title: str,
    deadline: float,
    now: float,
    only_node: str | None = None,
) -> bool:
    posted = state["posted_event_keys"]
    pending = state["pending_event_keys"]
    if key in posted:
        return False
    if len(key) > 200:
        raise PublisherError("watcher event key exceeds its bound")
    marker = _event_marker(plan_sha256, key)
    body = marker + "\n" + format_update(snapshot, title, only_node)
    resumed = key in pending
    if not resumed:
        pending.append(key)
        _persist_state(state_path, state)
    else:
        recovered = _find_posted_comment(client, pr_number, marker, deadline)
        if recovered is not None:
            url = recovered
            posted.append(key)
            pending.remove(key)
            state["posts"].append({"event": key, "posted_epoch": now, "url": url})
            del state["posts"][:-256]
            _persist_state(state_path, state)
            return True
    endpoint = f"/repos/{REPO}/issues/{pr_number}/comments"
    result = client.request("POST", endpoint, {"body": body}, deadline)
    if not isinstance(result, dict):
        raise PublisherError("GitHub returned a malformed PR comment receipt")
    url = _comment_url(result.get("html_url"), pr_number)
    posted.append(key)
    pending.remove(key)
    state["posts"].append({"event": key, "posted_epoch": now, "url": url})
    del state["posts"][:-256]
    _persist_state(state_path, state)
    return True


def _terminal_receipt(state_dir: Path, status: dict[str, Any], plan: dict[str, Any]) -> str | None:
    if status.get("status") not in TERMINAL_STATUSES:
        return None
    path = state_dir / "terminal.json"
    if not path.exists():
        return None
    terminal = _read_json(path)
    status_sha = _sha256_file(state_dir / "status.json")
    if (
        terminal.get("schema") != "wnn.tl3-finite17-terminal.v1"
        or terminal.get("status") != status.get("status")
        or terminal.get("plan_sha256") != plan.get("sha256")
        or terminal.get("state_sha256") != status_sha
        or terminal.get("fits_used") != status.get("fits_used")
        or terminal.get("score") != status.get("score")
    ):
        raise BridgeError("finite DAG terminal receipt differs from its durable status")
    return _sha256_file(path)


def _hourly_due(state: dict[str, Any], snapshot: dict[str, Any], now: float) -> bool:
    last = state.get("last_hourly_epoch")
    return (
        isinstance(last, (int, float))
        and not isinstance(last, bool)
        and now - float(last) >= 3600
        and state.get("last_hourly_snapshot_sha256") != _canonical_sha256(snapshot)
    )


def _hourly_event_key(snapshot: dict[str, Any], now: float) -> str:
    """Give each changed hourly snapshot a stable, retry-safe event identity."""
    return f"hourly:{int(now // 3600)}:{_canonical_sha256(snapshot)}"
