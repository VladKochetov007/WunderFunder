"""Standalone, stdin-driven downloader executed only through pinned SSH."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, BinaryIO, cast

CHUNK_BYTES = 1024 * 1024
MAX_RUNTIME_SECONDS = 1_200
READ_TIMEOUT_SECONDS = 30
MAX_PAYLOAD_BYTES = 48_000
ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
LABEL_RE = re.compile(r"wnn-e529-[a-z0-9][a-z0-9-]{6,63}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ALLOWED_PATHS = frozenset(
    {
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }
)


class SafeFailure(ValueError):
    """A sanitized remote-transfer failure without any signed URL material."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow HTTPS redirects only when they remain on storage.googleapis.com."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: BinaryIO,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        validate_url(new_url)
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


def validate_url(value: object) -> str:
    """Validate the signed storage host without including the URL in errors."""
    if not isinstance(value, str) or not 1 <= len(value) <= 16_384:
        raise SafeFailure("invalid_url")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise SafeFailure("invalid_url")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        raise SafeFailure("invalid_url") from None
    if (
        parsed.scheme != "https"
        or parsed.hostname != "storage.googleapis.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise SafeFailure("invalid_url")
    return value


def open_directory(path: str) -> int:
    """Open/create an absolute path one component at a time without following links."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise SafeFailure("invalid_destination")
    parts = path.split("/")[1:]
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise SafeFailure("invalid_destination")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open("/", flags)
    try:
        for part in parts:
            with suppress(FileExistsError):
                os.mkdir(part, 0o700, dir_fd=directory_fd)
            try:
                child_fd = os.open(part, flags, dir_fd=directory_fd)
            except OSError:
                raise SafeFailure("unsafe_destination") from None
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def hash_existing(directory_fd: int, filename: str, expected_size: int) -> str | None:
    """Return the hash of an existing regular file, rejecting symlink targets."""
    try:
        info = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise SafeFailure("unsafe_destination")
    if info.st_size != expected_size:
        return None
    file_fd = os.open(filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        digest = hashlib.sha256()
        while True:
            block = os.read(file_fd, CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(file_fd)


def set_read_timeout(response: Any, seconds: float) -> None:
    """Constrain the next single socket read to the remaining absolute deadline."""
    try:
        response.fp.raw._sock.settimeout(max(0.001, min(READ_TIMEOUT_SECONDS, seconds)))
    except (AttributeError, OSError):
        raise SafeFailure("deadline_control") from None


def fetch_one(
    opener: urllib.request.OpenerDirector,
    payload: Mapping[str, object],
    item: Mapping[str, object],
    deadline: float,
) -> dict[str, object]:
    """Stream one exact data object to a private temporary file, then atomically replace."""
    started = time.monotonic()
    remote_path = item["remote_path"]
    if (
        not isinstance(remote_path, str)
        or remote_path not in ALLOWED_PATHS
        or remote_path.startswith("/")
        or ".." in remote_path.split("/")
    ):
        raise SafeFailure("invalid_destination")
    expected_hash = item["sha256"]
    expected_size = item["size_bytes"]
    if (
        not isinstance(expected_hash, str)
        or SHA256_RE.fullmatch(expected_hash) is None
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise SafeFailure("invalid_identity")
    output_root = payload["output_root"]
    campaign_label = payload["campaign_label"]
    attempt_id = payload["attempt_id"]
    if not all(isinstance(value, str) for value in (output_root, campaign_label, attempt_id)):
        raise SafeFailure("invalid_payload")
    base = f"{output_root}/.worker/{campaign_label}/inputs/"
    parent_parts = "/".join(remote_path.split("/")[:-1])
    directory_fd = open_directory(base + parent_parts)
    filename = remote_path.split("/")[-1]
    temporary_name = f".{filename}.prefetch-{attempt_id}-{secrets.token_hex(8)}.part"
    temporary_fd: int | None = None
    temporary_created = False
    try:
        if hash_existing(directory_fd, filename, expected_size) == expected_hash:
            return {
                "remote_path": remote_path,
                "sha256": expected_hash,
                "size_bytes": expected_size,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "status": "already_present",
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SafeFailure("deadline")
        url = validate_url(item.get("url"))
        request = urllib.request.Request(
            url,
            headers={"Accept-Encoding": "identity", "User-Agent": "wnn-owned-prefetch/1"},
            method="GET",
        )
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        digest = hashlib.sha256()
        total_bytes = 0
        try:
            with opener.open(request, timeout=min(READ_TIMEOUT_SECONDS, remaining)) as response:
                if getattr(response, "status", response.getcode()) != 200:
                    raise SafeFailure("http_status")
                validate_url(response.geturl())
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length != str(expected_size):
                    raise SafeFailure("size_mismatch")
                while True:
                    # HTTPResponse closes its stream at the declared body length.
                    # Hash/size checks below still decide whether the file is valid.
                    if total_bytes == expected_size and response.fp is None:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SafeFailure("deadline")
                    set_read_timeout(response, remaining)
                    block = response.read1(min(CHUNK_BYTES, expected_size - total_bytes + 1))
                    if not block:
                        break
                    total_bytes += len(block)
                    if total_bytes > expected_size:
                        raise SafeFailure("size_mismatch")
                    digest.update(block)
                    view = memoryview(block)
                    while view:
                        written = os.write(temporary_fd, view)
                        if written <= 0:
                            raise SafeFailure("storage_write")
                        view = view[written:]
                if total_bytes != expected_size:
                    raise SafeFailure("size_mismatch")
                if digest.hexdigest() != expected_hash:
                    raise SafeFailure("hash_mismatch")
                os.fsync(temporary_fd)
        except SafeFailure:
            raise
        except urllib.error.HTTPError:
            raise SafeFailure("http_error") from None
        except Exception:
            raise SafeFailure("download_error") from None
        os.close(temporary_fd)
        temporary_fd = None
        try:
            existing = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise SafeFailure("unsafe_destination")
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        return {
            "remote_path": remote_path,
            "sha256": expected_hash,
            "size_bytes": expected_size,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "status": "downloaded",
        }
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def validate_payload(value: object) -> dict[str, object]:
    """Validate the strict remote stdin schema before creating any destination."""
    required = {
        "schema_version",
        "attempt_id",
        "campaign_label",
        "output_root",
        "budget_seconds",
        "inputs",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema_version") != 1:
        raise SafeFailure("invalid_payload")
    attempt_id = value.get("attempt_id")
    campaign_label = value.get("campaign_label")
    output_root = value.get("output_root")
    budget = value.get("budget_seconds")
    inputs = value.get("inputs")
    if (
        not isinstance(attempt_id, str)
        or ATTEMPT_RE.fullmatch(attempt_id) is None
        or not isinstance(campaign_label, str)
        or LABEL_RE.fullmatch(campaign_label) is None
        or not isinstance(output_root, str)
        or not output_root.startswith("/")
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or not 1 <= budget <= MAX_RUNTIME_SECONDS
        or not isinstance(inputs, list)
        or not 1 <= len(inputs) <= 4
    ):
        raise SafeFailure("invalid_payload")
    seen: set[str] = set()
    normalized_inputs: list[dict[str, object]] = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {
            "remote_path",
            "sha256",
            "size_bytes",
            "url",
        }:
            raise SafeFailure("invalid_payload")
        path = item.get("remote_path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(path, str)
            or path not in ALLOWED_PATHS
            or path in seen
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise SafeFailure("invalid_payload")
        validate_url(item.get("url"))
        seen.add(path)
        normalized_inputs.append(dict(item))
    return {
        "attempt_id": attempt_id,
        "campaign_label": campaign_label,
        "output_root": output_root,
        "budget_seconds": budget,
        "inputs": normalized_inputs,
    }


def main() -> int:
    """Run a bounded stdin request and print only a sanitized public receipt."""
    completed: list[dict[str, object]] = []
    try:
        raw_payload = sys.stdin.buffer.read(MAX_PAYLOAD_BYTES + 1)
        if len(raw_payload) > MAX_PAYLOAD_BYTES:
            raise SafeFailure("payload_too_large")
        payload = validate_payload(json.loads(raw_payload))
        deadline = time.monotonic() + cast_int(payload["budget_seconds"])
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), SameHostRedirectHandler()
        )
        for item in cast(list[dict[str, object]], payload["inputs"]):
            completed.append(fetch_one(opener, payload, item, deadline))
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "completed": completed,
            "error_code": None,
        }
        code = 0
    except SafeFailure as error:
        result = {
            "schema_version": 1,
            "status": "error",
            "completed": completed,
            "error_code": error.code,
        }
        code = 1
    except Exception:
        result = {
            "schema_version": 1,
            "status": "error",
            "completed": completed,
            "error_code": "internal_error",
        }
        code = 1
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return code


def cast_int(value: object) -> int:
    """Narrow a validated integer payload field for static type checking."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafeFailure("invalid_payload")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
