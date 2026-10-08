"""Hash-bound, bounded signed-object prefetch for receipt-owned Vast workers.

The URL plan is private and is sent only through pinned SSH stdin. The public
admission binds that plan to one attempt and its frozen worker-input identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

MAX_PLAN_BYTES = 64_000
MAX_URL_CHARS = 16_384
MAX_REMOTE_PAYLOAD_BYTES = 48_000
MAX_PREFETCH_SECONDS = 1_200
ALLOWED_REMOTE_PATHS = frozenset(
    {
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }
)
_ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_LABEL_RE = re.compile(r"wnn-e529-[a-z0-9][a-z0-9-]{6,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PrefetchPlanError(ValueError):
    """A private plan or public admission does not match the frozen attempt."""


@dataclass(frozen=True)
class PrefetchItem:
    """One signed URL bound to a frozen worker-input identity."""

    remote_path: str
    sha256: str
    size_bytes: int
    url: str = field(repr=False)


@dataclass(frozen=True)
class PreparedPrefetch:
    """Validated private items and nonsecret protocol hashes."""

    items: tuple[PrefetchItem, ...]
    admission_sha256: str
    private_plan_sha256: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PrefetchPlanError("prefetch document contains a duplicate key")
        result[key] = value
    return result


def _read_bound_json(path: Path, runtime_root: Path, *, private: bool) -> tuple[bytes, object]:
    absolute = Path(os.path.abspath(path))
    runtime = Path(os.path.abspath(runtime_root))
    if not absolute.is_relative_to(runtime) or absolute == runtime:
        raise PrefetchPlanError("prefetch documents must be inside the runtime directory")
    current = runtime
    try:
        runtime_info = current.lstat()
        if (
            not stat.S_ISDIR(runtime_info.st_mode)
            or runtime_info.st_uid != os.geteuid()
            or stat.S_IMODE(runtime_info.st_mode) & 0o022
        ):
            raise PrefetchPlanError("runtime directory is not owned and protected")
        for part in absolute.relative_to(runtime).parts[:-1]:
            current = current / part
            part_info = current.lstat()
            if (
                not stat.S_ISDIR(part_info.st_mode)
                or part_info.st_uid != os.geteuid()
                or stat.S_IMODE(part_info.st_mode) & 0o022
            ):
                raise PrefetchPlanError("prefetch document parent is not owned and protected")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(absolute, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise PrefetchPlanError("prefetch document ownership or file type is invalid")
            if private and stat.S_IMODE(info.st_mode) != 0o600:
                raise PrefetchPlanError("private prefetch plan must have mode 0600")
            if not private and stat.S_IMODE(info.st_mode) & 0o022:
                raise PrefetchPlanError("public prefetch admission is group/world writable")
            if info.st_size <= 0 or info.st_size > MAX_PLAN_BYTES:
                raise PrefetchPlanError("prefetch document exceeds its bounded size")
            chunks: list[bytes] = []
            remaining = MAX_PLAN_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
    except PrefetchPlanError:
        raise
    except OSError:
        raise PrefetchPlanError("prefetch document is missing or unsafe") from None
    if len(raw) > MAX_PLAN_BYTES:
        raise PrefetchPlanError("prefetch document exceeds its bounded size")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PrefetchPlanError("prefetch document is not valid JSON") from None
    return raw, value


def _identity_rows(value: object, *, field_name: str) -> dict[str, tuple[str, int]]:
    if not isinstance(value, list) or not value:
        raise PrefetchPlanError(f"{field_name} must be a nonempty list")
    rows: dict[str, tuple[str, int]] = {}
    for candidate in value:
        if not isinstance(candidate, dict):
            raise PrefetchPlanError(f"{field_name} row is malformed")
        if set(candidate) != {"remote_path", "sha256", "size_bytes"}:
            raise PrefetchPlanError(f"{field_name} row is malformed")
        path = candidate.get("remote_path")
        digest = candidate.get("sha256")
        size = candidate.get("size_bytes")
        if (
            not isinstance(path, str)
            or path not in ALLOWED_REMOTE_PATHS
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path in rows
        ):
            raise PrefetchPlanError(f"{field_name} row is invalid")
        rows[path] = (digest, size)
    return rows


def _validate_url(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_URL_CHARS:
        raise PrefetchPlanError("private prefetch URL is malformed")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise PrefetchPlanError("private prefetch URL contains a control character")
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise PrefetchPlanError("private prefetch URL is malformed") from None
    if (
        parsed.scheme != "https"
        or hostname != "storage.googleapis.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise PrefetchPlanError("private prefetch URL is outside the approved storage host")
    return value


def prepare_prefetch(
    private_plan_path: Path,
    admission_path: Path,
    *,
    attempt_id: str,
    campaign_label: str,
    frozen_inputs: list[dict[str, object]],
    runtime_root: Path,
) -> PreparedPrefetch:
    """Validate private URLs against an explicit public admission and frozen inputs."""
    if _ATTEMPT_RE.fullmatch(attempt_id) is None or _LABEL_RE.fullmatch(campaign_label) is None:
        raise PrefetchPlanError("prefetch attempt identity is malformed")
    private_raw, private_value = _read_bound_json(private_plan_path, runtime_root, private=True)
    admission_raw, admission_value = _read_bound_json(admission_path, runtime_root, private=False)
    if not isinstance(private_value, dict) or set(private_value) != {"schema_version", "inputs"}:
        raise PrefetchPlanError("private prefetch plan schema is invalid")
    if private_value.get("schema_version") != 1:
        raise PrefetchPlanError("private prefetch plan version is unsupported")
    if not isinstance(admission_value, dict) or set(admission_value) != {
        "schema_version",
        "attempt_id",
        "campaign_label",
        "private_plan_sha256",
        "source_binding",
    }:
        raise PrefetchPlanError("public prefetch admission schema is invalid")
    if admission_value.get("schema_version") != 1:
        raise PrefetchPlanError("public prefetch admission version is unsupported")
    if (
        admission_value.get("attempt_id") != attempt_id
        or admission_value.get("campaign_label") != campaign_label
        or admission_value.get("private_plan_sha256") != hashlib.sha256(private_raw).hexdigest()
    ):
        raise PrefetchPlanError("public prefetch admission does not bind this attempt and plan")

    frozen_rows: dict[str, tuple[str, int]] = {}
    for row in frozen_inputs:
        path = row.get("remote_path")
        digest = row.get("source_sha256")
        size = row.get("size_bytes")
        if not isinstance(path, str) or path not in ALLOWED_REMOTE_PATHS:
            continue
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path in frozen_rows
        ):
            raise PrefetchPlanError("frozen prefetch input identity is malformed")
        frozen_rows[path] = (digest, size)
    plan_rows_value = private_value.get("inputs")
    admission_rows_value = admission_value.get("source_binding")
    if not isinstance(plan_rows_value, list) or not isinstance(admission_rows_value, list):
        raise PrefetchPlanError("prefetch input bindings must be lists")
    plan_rows: dict[str, tuple[str, int, str]] = {}
    for row in plan_rows_value:
        if not isinstance(row, dict) or set(row) != {"remote_path", "sha256", "size_bytes", "url"}:
            raise PrefetchPlanError("private prefetch input row is malformed")
        identities = _identity_rows(
            [{key: row[key] for key in ("remote_path", "sha256", "size_bytes")}],
            field_name="private prefetch inputs",
        )
        path, (digest, size) = next(iter(identities.items()))
        if path in plan_rows:
            raise PrefetchPlanError("private prefetch plan contains a duplicate path")
        plan_rows[path] = (digest, size, _validate_url(row.get("url")))
    admission_rows = _identity_rows(admission_rows_value, field_name="public source binding")
    if not frozen_rows or plan_rows.keys() != frozen_rows.keys() or admission_rows != frozen_rows:
        raise PrefetchPlanError("prefetch plan and admission differ from frozen prefetch inputs")
    for path, (digest, size, _) in plan_rows.items():
        if frozen_rows[path] != (digest, size):
            raise PrefetchPlanError("prefetch source binding differs from frozen input identity")
    items = tuple(
        PrefetchItem(path, digest, size, url)
        for path, (digest, size, url) in sorted(plan_rows.items())
    )
    return PreparedPrefetch(
        items=items,
        admission_sha256=hashlib.sha256(admission_raw).hexdigest(),
        private_plan_sha256=hashlib.sha256(private_raw).hexdigest(),
    )


def make_remote_payload(
    prepared: PreparedPrefetch,
    *,
    attempt_id: str,
    campaign_label: str,
    output_root: str,
    budget_seconds: int,
) -> bytes:
    """Create the bounded stdin payload; callers must never log or argv-encode it."""
    if _ATTEMPT_RE.fullmatch(attempt_id) is None or _LABEL_RE.fullmatch(campaign_label) is None:
        raise PrefetchPlanError("prefetch attempt identity is malformed")
    if budget_seconds <= 0 or budget_seconds > MAX_PREFETCH_SECONDS:
        raise PrefetchPlanError("prefetch budget is outside its hard limit")
    if not output_root.startswith("/") or ".." in Path(output_root).parts:
        raise PrefetchPlanError("worker output root is unsafe")
    payload = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "campaign_label": campaign_label,
        "output_root": output_root,
        "budget_seconds": budget_seconds,
        "inputs": [
            {
                "remote_path": item.remote_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "url": item.url,
            }
            for item in prepared.items
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_REMOTE_PAYLOAD_BYTES:
        raise PrefetchPlanError("private prefetch payload exceeds its bounded size")
    return encoded


def validate_remote_result(raw_stdout: str, prepared: PreparedPrefetch) -> dict[str, object]:
    """Validate only the remote's nonsecret result fields for ledger storage."""
    try:
        result = json.loads(raw_stdout, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, PrefetchPlanError):
        raise PrefetchPlanError("remote prefetch returned an invalid receipt") from None
    if not isinstance(result, dict) or set(result) != {
        "schema_version",
        "status",
        "completed",
        "error_code",
    }:
        raise PrefetchPlanError("remote prefetch returned an invalid receipt")
    if result.get("schema_version") != 1 or result.get("status") not in {"complete", "error"}:
        raise PrefetchPlanError("remote prefetch returned an invalid receipt")
    completed_value = result.get("completed")
    if not isinstance(completed_value, list) or len(completed_value) > len(prepared.items):
        raise PrefetchPlanError("remote prefetch returned an invalid receipt")
    expected = {item.remote_path: item for item in prepared.items}
    completed: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in completed_value:
        if not isinstance(row, dict) or set(row) != {
            "remote_path",
            "sha256",
            "size_bytes",
            "elapsed_ms",
            "status",
        }:
            raise PrefetchPlanError("remote prefetch returned an invalid receipt")
        path = row.get("remote_path")
        item = expected.get(path) if isinstance(path, str) else None
        if (
            item is None
            or path in seen
            or row.get("sha256") != item.sha256
            or row.get("size_bytes") != item.size_bytes
            or isinstance(row.get("elapsed_ms"), bool)
            or not isinstance(row.get("elapsed_ms"), int)
            or row["elapsed_ms"] < 0
            or row.get("status") not in {"downloaded", "already_present"}
        ):
            raise PrefetchPlanError("remote prefetch returned an invalid receipt")
        seen.add(path)
        completed.append(dict(row))
    if result.get("status") == "complete" and (
        result.get("error_code") is not None or seen != expected.keys()
    ):
        raise PrefetchPlanError("remote prefetch returned an invalid receipt")
    error_code = result.get("error_code")
    if result.get("status") == "error" and (
        not isinstance(error_code, str) or re.fullmatch(r"[a-z_]{1,32}", error_code) is None
    ):
        raise PrefetchPlanError("remote prefetch returned an invalid receipt")
    if result.get("status") == "complete" and error_code is not None:
        raise PrefetchPlanError("remote prefetch returned an invalid receipt")
    return {
        "status": cast(str, result["status"]),
        "completed": completed,
        "error_code": error_code,
    }


REMOTE_PROGRAM_PATH = Path(__file__).with_name("kaggle_vast_prefetch_remote.py")
REMOTE_PROGRAM = REMOTE_PROGRAM_PATH.read_text(encoding="utf-8")
REMOTE_PROGRAM_SHA256 = hashlib.sha256(REMOTE_PROGRAM.encode("utf-8")).hexdigest()
