"""Durable reporter state and a private, credential-free publication outbox."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def read_json(path: Path, default: dict | None = None) -> dict:
    """Read one JSON object without following a symbolic-link state file."""
    if path.is_symlink():
        raise ValueError("Reporter state file is a symbolic link")
    if not path.exists() and default is not None:
        return default
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Reporter state must be a JSON object")
    return result


def write_json(path: Path, value: dict) -> None:
    """Publish an fsynced state file atomically on the approved physical disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Reporter destination is a symbolic link")
    descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_event(path: Path, event: str, **fields: object) -> None:
    """Retain immutable publishing acknowledgments and bounded error diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(time_unix=time.time(), event=event, **fields), sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        raw = line.encode()
        while raw:
            raw = raw[os.write(descriptor, raw) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
