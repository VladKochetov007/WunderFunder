from __future__ import annotations

import base64
import hashlib
import os
import re
import stat
import time
from pathlib import Path
from typing import BinaryIO

CHUNK_BYTES = 262_144
MAX_RESPONSE_BYTES = 8 * 1_048_576
FILES = (
    "run_metadata.json",
    "monitor_evaluations.jsonl",
    "monitor_progress.json",
    "monitor_observer.jsonl",
    "monitor_events.jsonl",
    "rank0.jsonl",
    "epoch_index.json",
    "terminal_status.json",
    "score.json",
)
STAGES = ("fit", "profile", "profile-one", "profile-two", "profile-peak")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _marker(stream: BinaryIO, offset: int) -> str:
    """Identify the short byte prefix immediately before a resumable offset."""
    stream.seek(max(0, offset - 64))
    return stream.read(min(64, offset)).hex()


def _collect_file(path: Path, old: dict[str, object]) -> dict[str, object]:
    """Return at most 256 KiB of unchanged source bytes and an exact continuation."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("monitor source must be a regular non-link file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("monitor source changed during open")
        journal = path.suffix == ".jsonl"
        identity = f"{info.st_dev}:{info.st_ino}"
        version = identity if journal else f"{identity}:{info.st_size}:{info.st_mtime_ns}"
        offset = old.get("offset", 0)
        matched = (
            old.get("version") == version
            and type(offset) is int
            and 0 <= offset <= info.st_size
            and old.get("marker") == _marker(stream, offset)
        )
        if not matched:
            offset = 0
        stream.seek(offset)
        payload = stream.read(min(CHUNK_BYTES, info.st_size - offset))
        if journal:
            end = payload.rfind(b"\n") + 1
            if not end and len(payload) == CHUNK_BYTES:
                raise ValueError("monitor source line exceeds the collector chunk bound")
            payload = payload[:end]
        stop = offset + len(payload)
        marker = _marker(stream, stop)
        after = os.fstat(stream.fileno())
        if not journal and (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
            raise ValueError("monitor snapshot changed while being collected")
        return {
            "name": path.name,
            "offset": offset,
            "reset": not matched,
            "data_base64": base64.b64encode(payload).decode("ascii"),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "source_size": info.st_size,
            "source_mtime_ns": info.st_mtime_ns,
            "eof": stop == info.st_size,
            "cursor": {"version": version, "offset": stop, "marker": marker},
        }


def collect_runs(
    run_root: str, host: str, cursors: dict[str, object] | None = None
) -> dict[str, object]:
    """Read only explicit run artifacts below one root for the existing file RPC.

    Compatible with the host's stdlib Python 3.9. A declared root symlink is
    resolved once; child directories/files may not escape that physical root.
    No process, model, CUDA, network, checkpoint payload or credential is read.
    """
    if not RUN_ID.fullmatch(host):
        raise ValueError("host label must be a safe dashboard identifier")
    root = Path(run_root).resolve(strict=True)
    previous = cursors or {}
    runs: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    total = 0
    cells = sorted(root.iterdir(), key=lambda path: path.name)
    if len(cells) > 128:
        raise ValueError("collector root exceeds its 128-entry discovery bound")
    for cell in cells:
        if cell.is_symlink() or not cell.is_dir():
            continue
        for stage in STAGES:
            source = cell / stage
            if source.is_symlink() or not source.is_dir():
                continue
            source = source.resolve(strict=True)
            if root not in source.parents:
                raise ValueError("run directory escaped the declared physical root")
            run_id = f"{host}-{cell.name}-{stage}"
            if not RUN_ID.fullmatch(run_id) or ".." in run_id:
                raise ValueError("run identifier exceeds dashboard limits")
            chunks: list[dict[str, object]] = []
            for name in FILES:
                if total + CHUNK_BYTES > MAX_RESPONSE_BYTES:
                    break
                path = source / name
                if not path.exists() and not path.is_symlink():
                    continue
                key = f"{run_id}/{name}"
                old = previous.get(key, {})
                try:
                    chunk = _collect_file(path, old if isinstance(old, dict) else {})
                    chunks.append(chunk)
                    total += int(chunk["bytes"])
                except (OSError, ValueError) as exc:
                    errors.append({"file": key, "error": str(exc)[:200]})
            runs.append(
                {"run_id": run_id, "host": host, "source_directory": str(source), "files": chunks}
            )
    return {
        "schema": "wnn.sequence-monitor-rpc.v1",
        "host": host,
        "root": str(root),
        "collected_at_unix": time.time(),
        "chunk_bytes": CHUNK_BYTES,
        "payload_bytes": total,
        "runs": runs,
        "errors": errors,
    }
