from __future__ import annotations

import email.utils
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .common import LaunchError, atomic_write_json, read_json

API = "https://api.github.com"
LOGIN = "SemyonEpanov"
REPO = "VladKochetov007/WunderFunder"
ISSUE = 46
POLL = 60
LIFETIME = 96 * 60 * 60
RETRIES = 3
STATE_SCHEMA = "wnn.galam-l4-quality8-publisher.v1"


class PublisherError(LaunchError):
    """Publisher validation or API failure."""


class AuthenticationError(PublisherError):
    """GitHub credentials do not match the authorized account."""


class TransientGitHubError(PublisherError):
    """Temporary GitHub failure after bounded retries."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> Response: ...

    def __exit__(self, *args: object) -> object: ...


class GitHubClient:
    """Call only the fixed GitHub account and issue endpoints."""

    def __init__(
        self,
        token: str,
        opener: Callable[..., Response] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.token = token
        self.opener = opener or urllib.request.urlopen
        self.sleep = sleeper
        self.clock = clock

    def request(
        self, method: str, path: str, data: dict[str, object] | None, deadline: float
    ) -> object:
        last_delay = 1.0
        for attempt in range(RETRIES):
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "WunderFunder-GALAM-quality8-monitor",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            payload = None if data is None else json.dumps(data, allow_nan=False).encode("utf-8")
            request = urllib.request.Request(API + path, payload, headers, method=method)
            try:
                with self.opener(request, timeout=20) as response:
                    status, response_headers = response.status, response.headers
                    raw = response.read(1 << 20)
            except urllib.error.HTTPError as error:
                status, response_headers, raw = error.code, error.headers or {}, b""
            except (urllib.error.URLError, TimeoutError, OSError):
                status, response_headers, raw = 503, {}, b""
            if status in (401, 403):
                raise AuthenticationError(f"GitHub rejected credentials (HTTP {status})")
            if status == 429 or 500 <= status < 600:
                last_delay = _retry_after(response_headers.get("Retry-After"), self.clock())
                if attempt + 1 == RETRIES or self.clock() + last_delay >= deadline:
                    raise TransientGitHubError(
                        f"GitHub temporary failure (HTTP {status})", last_delay
                    )
                self.sleep(last_delay)
                continue
            if not 200 <= status < 300:
                raise PublisherError(f"GitHub request failed (HTTP {status})")
            if not raw:
                return None
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise PublisherError("GitHub returned invalid JSON") from None
        raise TransientGitHubError("GitHub retry limit reached", last_delay)

    def verify_account(self, deadline: float) -> None:
        result = self.request("GET", "/user", None, deadline)
        if not isinstance(result, dict) or result.get("login") != LOGIN:
            raise AuthenticationError("GitHub token does not identify SemyonEpanov")

    def post(self, body: str, marker: str, deadline: float, pending: bool) -> str:
        endpoint = f"/repos/{REPO}/issues/{ISSUE}/comments"
        if pending:
            query = urllib.parse.urlencode({"per_page": 20, "sort": "created", "direction": "desc"})
            comments = self.request("GET", f"{endpoint}?{query}", None, deadline)
            if not isinstance(comments, list):
                raise PublisherError("GitHub returned a malformed comment list")
            for comment in comments:
                if isinstance(comment, dict) and marker in str(comment.get("body", "")):
                    return _comment_url(comment.get("html_url"))
        result = self.request("POST", endpoint, {"body": body}, deadline)
        if not isinstance(result, dict):
            raise PublisherError("GitHub returned a malformed comment receipt")
        return _comment_url(result.get("html_url"))


def require_linux_host(platform_name: str | None = None) -> None:
    """Allow token handling only on the Linux receiving host."""
    if (platform_name or sys.platform) != "linux" or not hasattr(os, "O_NOFOLLOW"):
        raise PublisherError("publisher is available only on the Linux receiving host")


def read_token(path: Path) -> tuple[str, tuple[int, int, int]]:
    """Read one current-user mode-0600 token without following its final path component."""
    require_linux_host()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as error:
        raise PublisherError("could not open private token file safely") from error
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= 16_384
        ):
            raise PublisherError("token must be owned, regular, mode 0600, and bounded")
        raw = os.read(fd, 16_385)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise PublisherError("token file changed while being read")
        token = raw.decode("utf-8").strip()
        if not token or any(char.isspace() for char in token):
            raise PublisherError("token file must contain one token")
        return token, (before.st_dev, before.st_ino, before.st_uid)
    except UnicodeDecodeError:
        raise PublisherError("token file is not valid UTF-8") from None
    finally:
        os.close(fd)


def remove_owned_token(path: Path, identity: tuple[int, int, int]) -> bool:
    """Remove only the exact private token inode read by this process."""
    try:
        item = path.lstat()
    except FileNotFoundError:
        return False
    if (
        stat.S_ISREG(item.st_mode)
        and stat.S_IMODE(item.st_mode) == 0o600
        and (item.st_dev, item.st_ino, item.st_uid) == identity
    ):
        path.unlink()
        return True
    return False


def validate_snapshot(receipt: dict[str, object], summary: dict[str, object]) -> None:
    """Require exact launch/source/split identity and honest masks before reporting."""
    if (
        receipt.get("schema") != "wnn.galam-l4-quality8-launch.v1"
        or summary.get("schema") != "wnn.galam-l4-quality8-summary.v1"
    ):
        raise PublisherError("unsupported launch receipt or summary")
    for key in ("launch_id", "panel_sha256", "source_sha256", "split_sha256"):
        if not _valid_id(key, receipt.get(key)) or summary.get(key) != receipt.get(key):
            raise PublisherError(f"summary {key} differs from launch receipt")
    receipt_cells, summary_cells = receipt.get("cells"), summary.get("cells")
    if (
        not isinstance(receipt_cells, dict)
        or not isinstance(summary_cells, dict)
        or len(receipt_cells) != 8
        or set(receipt_cells) != set(summary_cells)
    ):
        raise PublisherError("summary does not cover the receipt's exact eight cells")
    if (
        summary.get("status") not in ("running", "completed", "failed")
        or summary.get("official_valid_mask") != "is_scored"
        or summary.get("train_eval_mask") != "all_after_warmup"
    ):
        raise PublisherError("summary state or evaluation masks are unsupported")
    gpus: set[int] = set()
    for cell_id, record in receipt_cells.items():
        entry = summary_cells[cell_id]
        if (
            not isinstance(cell_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", cell_id) is None
            or not isinstance(record, dict)
            or not isinstance(entry, dict)
        ):
            raise PublisherError("summary has a malformed cell")
        gpu, digest = record.get("gpu"), record.get("identity_sha256")
        if type(gpu) is not int or not 0 <= gpu < 8 or gpu in gpus or not _sha(digest):
            raise PublisherError(f"{cell_id}: invalid GPU or run identity in receipt")
        gpus.add(gpu)
        terminal = entry.get("terminal_status")
        if isinstance(terminal, dict):
            identity = terminal.get("identity")
            if (
                not isinstance(identity, dict)
                or identity.get("source_sha256") != receipt.get("source_sha256")
                or _canonical(identity) != digest
            ):
                raise PublisherError(f"{cell_id}: terminal identity differs from receipt")


def format_update(
    receipt: dict[str, object], summary: dict[str, object], title: str, only_cell: str | None = None
) -> str:
    """Render cell states and only complete full-mask evaluation metrics."""
    validate_snapshot(receipt, summary)
    cells, records = summary["cells"], receipt["cells"]
    assert isinstance(cells, dict) and isinstance(records, dict)
    selected = [only_cell] if only_cell else sorted(cells)
    lines = [
        f"## GALAM L4 fixed-eight {title}",
        f"Launch `{receipt['launch_id']}` | status `{summary['status']}`",
        f"Source `{receipt['source_sha256']}` | PANEL `{receipt['panel_sha256']}` | split `{receipt['split_sha256']}`",
        "Official validation: full `is_scored`; train-eval: diagnostic `all_after_warmup`.",
        "",
        "| Cell | GPU | State | Epoch | Best valid WP | Latest valid WP (t0 / t1) | Train-eval WP (t0 / t1) |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for cell_id in selected:
        entry, record = cells.get(cell_id), records.get(cell_id)
        if not isinstance(entry, dict) or not isinstance(record, dict):
            raise PublisherError("requested cell is absent from the verified summary")
        metrics = _metrics(entry, receipt.get("split_sha256"), record.get("identity_sha256"))
        epoch = metrics.get("epoch")
        terminal, pointers = entry.get("terminal_status"), entry.get("checkpoint_pointers")
        if (
            isinstance(terminal, dict)
            and type(terminal.get("completed_epochs")) is int
            and 1 <= terminal["completed_epochs"] <= 20
        ):
            epoch = terminal["completed_epochs"]
        best = "-"
        if isinstance(pointers, dict):
            pointer = pointers.get("best")
            if (
                isinstance(pointer, dict)
                and pointers.get("run_identity_sha256") == record.get("identity_sha256")
                and _finite(pointer.get("valid_wp"))
                and pointer.get("valid_wp") == pointers.get("best_raw_wp")
                and type(pointer.get("epoch")) is int
                and 1 <= pointer["epoch"] <= 20
            ):
                best = f"{float(pointer['valid_wp']):.6f} (ep {pointer['epoch']})"
        lines.append(
            f"| {cell_id} | {record['gpu']} | {entry.get('state', 'unknown')} | {epoch if epoch is not None else 'not reported'}/20 | {best} | {_pair(metrics.get('valid'))} | {_pair(metrics.get('train_eval'))} |"
        )
    return "\n".join(lines)


def run_publisher(
    receipt_path: Path,
    token_path: Path,
    output_dir: Path,
    cleanup_owned_token: bool = False,
    opener: Callable[..., Response] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> None:
    """Poll one receipt every minute and publish start/hourly/cell/final updates."""
    require_linux_host()
    token, token_id = read_token(token_path)
    try:
        _run(receipt_path, output_dir, token, opener, sleeper, clock)
    finally:
        if cleanup_owned_token:
            remove_owned_token(token_path, token_id)


def _run(
    receipt_path: Path,
    output_dir: Path,
    token: str,
    opener: Callable[..., Response] | None,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    _private_dir(output_dir)
    receipt = _json_file(receipt_path, 2 << 20)
    deadline = _created_at(receipt.get("created_at")) + LIFETIME
    state_path = output_dir / "publisher_state.json"
    state = _state(state_path, receipt.get("launch_id"), clock())
    client = GitHubClient(token, opener, sleeper, clock)
    client.verify_account(deadline)
    while clock() < deadline:
        state = _state(state_path, receipt.get("launch_id"), clock())
        retry_at = state.get("retry_not_before")
        if _finite(retry_at) and clock() < float(retry_at):
            sleeper(min(POLL, float(retry_at) - clock()))
            continue
        receipt = _json_file(receipt_path, 2 << 20)
        summary_path = receipt_path.parent / "summary.json"
        if not summary_path.exists():
            if receipt.get("status") in ("preflight_failed", "launch_failed") and not receipt.get(
                "source_sha256"
            ):
                raise PublisherError("launch failed before source identity was admitted")
            sleeper(POLL)
            continue
        summary = _json_file(summary_path, 8 << 20)
        validate_snapshot(receipt, summary)
        try:
            if not state.get("startup_posted"):
                _publish(
                    client, state, state_path, receipt, summary, "start", "start", deadline, clock()
                )
                state["startup_posted"] = True
                state["last_hourly_epoch"] = clock()
                atomic_write_json(state_path, state)
            cells = summary["cells"]
            assert isinstance(cells, dict)
            for cell_id, cell in cells.items():
                if _terminal(cell):
                    key = f"terminal:{cell_id}:{cell['state']}"
                    _publish(
                        client,
                        state,
                        state_path,
                        receipt,
                        summary,
                        key,
                        f"finished cell {cell_id}",
                        deadline,
                        clock(),
                        cell_id,
                    )
            if all(_terminal(cell) for cell in cells.values()):
                _publish(
                    client, state, state_path, receipt, summary, "final", "final", deadline, clock()
                )
                return
            last = state.get("last_hourly_epoch")
            if state.get("startup_posted") and _finite(last) and clock() - float(last) >= 3600:
                event_key = _hourly_event_key(float(last))
                _publish(
                    client,
                    state,
                    state_path,
                    receipt,
                    summary,
                    event_key,
                    "hourly",
                    deadline,
                    clock(),
                )
                state["last_hourly_epoch"] = clock()
                atomic_write_json(state_path, state)
        except TransientGitHubError as error:
            state["retry_not_before"] = clock() + max(error.retry_after, POLL)
            atomic_write_json(state_path, state)
        sleeper(POLL)
    raise PublisherError("publisher reached the 96-hour lifetime without a final update")


def _publish(
    client: GitHubClient,
    state: dict[str, object],
    state_path: Path,
    receipt: dict[str, object],
    summary: dict[str, object],
    key: str,
    title: str,
    deadline: float,
    now: float,
    only_cell: str | None = None,
) -> None:
    posted = state.setdefault("posted_event_keys", [])
    pending = state.setdefault("pending_event_keys", [])
    posts = state.setdefault("posts", [])
    if (
        not isinstance(posted, list)
        or not isinstance(pending, list)
        or not isinstance(posts, list)
        or key in posted
    ):
        return
    marker = f"<!-- galam-l4-quality8:{receipt['launch_id']}:{hashlib.sha256(key.encode()).hexdigest()[:24]} -->"
    body = f"{marker}\n{format_update(receipt, summary, title, only_cell)}"
    resumed = key in pending
    if not resumed:
        pending.append(key)
        atomic_write_json(state_path, state)
    url = client.post(body, marker, deadline, resumed)
    posted.append(key)
    pending.remove(key)
    posts.append(
        {"event": key, "posted_at": datetime.fromtimestamp(now, UTC).isoformat(), "url": url}
    )
    del posts[:-128]
    atomic_write_json(state_path, state)


def _metrics(cell: dict[str, object], split_sha: object, identity_sha: object) -> dict[str, object]:
    event = cell.get("latest_official_validation")
    if (
        not isinstance(event, dict)
        or event.get("schema") != "wnn.monitor.v1"
        or event.get("kind") != "evaluation"
    ):
        return {}
    epoch, context, metrics = event.get("epoch"), event.get("context"), event.get("metrics")
    if (
        type(epoch) is not int
        or not 1 <= epoch <= 20
        or not isinstance(context, dict)
        or not isinstance(metrics, dict)
    ):
        return {}
    valid, train = context.get("valid"), context.get("train_eval")
    if context.get("run_identity_sha256") != identity_sha:
        return {}
    good_valid = (
        isinstance(valid, dict)
        and valid.get("row_mask") == "is_scored"
        and valid.get("coverage_complete") is True
        and valid.get("split_sha256") == split_sha
        and _positive(valid.get("member_count"))
        and _positive(valid.get("selected_rows"))
        and _positive(metrics.get("valid/sequence_count"))
        and metrics.get("valid/sequence_count") == valid.get("member_count")
        and _positive(metrics.get("valid/selected_rows"))
        and metrics.get("valid/selected_rows") == valid.get("selected_rows")
    )
    good_train = (
        isinstance(train, dict)
        and train.get("row_mask") == "all_after_warmup"
        and train.get("coverage_complete") is True
        and _positive(train.get("member_count"))
        and _positive(metrics.get("train_eval/sequence_count"))
        and metrics.get("train_eval/sequence_count") == train.get("member_count")
    )
    result: dict[str, object] = {"epoch": epoch} if good_valid and good_train else {}
    for key, prefix, good in (
        ("valid", "valid", good_valid),
        ("train_eval", "train_eval", good_train),
    ):
        values = [metrics.get(f"{prefix}/wp{suffix}") for suffix in ("", "/t0", "/t1")]
        if good_valid and good_train and good and all(_finite(value) for value in values):
            result[key] = tuple(float(value) for value in values)
    return result


def _pair(value: object) -> str:
    if not isinstance(value, tuple) or len(value) != 3:
        return "-"
    return f"{value[0]:.6f} ({value[1]:.6f} / {value[2]:.6f})"


def _terminal(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("process_state") != "running"
        and value.get("state") in ("completed", "early_stopped", "failed")
    )


def _private_dir(path: Path) -> None:
    if path.is_symlink():
        raise PublisherError("publisher output cannot be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PublisherError("publisher output must be a private directory owned by this user")


def _json_file(path: Path, limit: int) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise PublisherError(f"expected a bounded regular file: {path.name}")
    return read_json(path)


def _state(path: Path, launch_id: object, now: float) -> dict[str, object]:
    if path.exists():
        value = _json_file(path, 1 << 20)
        if value.get("schema") != STATE_SCHEMA or value.get("launch_id") != launch_id:
            raise PublisherError("publisher state belongs to another launch")
        if any(
            not isinstance(value.get(key), list)
            for key in ("posted_event_keys", "pending_event_keys", "posts")
        ):
            raise PublisherError("publisher state lists are malformed")
        return value
    return {
        "schema": STATE_SCHEMA,
        "launch_id": launch_id,
        "created_epoch": now,
        "posted_event_keys": [],
        "pending_event_keys": [],
        "posts": [],
    }


def _created_at(value: object) -> float:
    if not isinstance(value, str):
        raise PublisherError("receipt has no launch creation time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise PublisherError("receipt launch creation time is invalid") from None
    if parsed.tzinfo is None:
        raise PublisherError("receipt launch time must include a timezone")
    return parsed.timestamp()


def _retry_after(value: object, now: float) -> float:
    if not isinstance(value, str):
        return 1.0
    try:
        return max(1.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            return max(1.0, parsed.timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return 1.0


def _comment_url(value: object) -> str:
    if isinstance(value, str):
        parsed = urllib.parse.urlparse(value)
        path = f"/{REPO}/issues/{ISSUE}"
        if (
            parsed.scheme == "https"
            and parsed.netloc == "github.com"
            and parsed.path in (path, f"/{REPO}/pull/{ISSUE}")
            and parsed.fragment.startswith("issuecomment-")
        ):
            return value
    raise PublisherError("GitHub returned a comment URL outside the fixed issue")


def _canonical(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_id(key: str, value: object) -> bool:
    if key == "launch_id":
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None
    return _sha(value)


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _hourly_event_key(last_hourly_epoch: float) -> str:
    """Use the due-hour timestamp as a stable retry and deduplication key."""
    return f"hourly:{int(last_hourly_epoch + 3600)}"
