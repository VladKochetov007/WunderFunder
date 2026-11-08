from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import time
from pathlib import Path

CELLS = tuple(
    f"{arm}_s{seed}"
    for arm in ("raw112_d0", "raw112_d02", "event211_d02")
    for seed in (101101, 101102)
)
ENDPOINTS = ("swa3456", "epoch6")
MAX_FILE = 1_048_576
MAX_RESPONSE = 8 * 1_048_576


def collect(run_root: str, known: dict) -> dict:
    """Read only twelve exact fixed-readout receipts through the existing ROOT RPC.

    This stdlib-only function writes nothing and never inspects process, model,
    prediction array or credential files. A declared root symlink resolves once.
    """
    root = Path(run_root).resolve(strict=True)
    result = {
        "schema": "wnn.l4-fixed-readout-bytes.v1",
        "root": str(root),
        "collected_at_unix": time.time(),
        "files": [],
        "errors": [],
    }
    encoded_bytes = 0
    for cell in CELLS:
        for endpoint in ENDPOINTS:
            relative = f"{cell}/{endpoint}/readout.json"
            path = root / cell / "readouts" / endpoint / "readout.json"
            parents = (path.parent, path.parent.parent, path.parent.parent.parent)
            if any(parent.is_symlink() for parent in parents) or path.is_symlink():
                raise ValueError("readout receipt cannot follow child symlinks")
            if not path.exists():
                continue
            before = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > MAX_FILE
                or root not in path.resolve(strict=True).parents
            ):
                raise ValueError("readout receipt escapes its bounded regular-file root")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                raw = stream.read(MAX_FILE + 1)
                after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev,
                opened.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or len(raw) != before.st_size:
                result["errors"].append({"file": relative, "error": "receipt changed during read"})
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if known.get(relative) == digest:
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                result["errors"].append({"file": relative, "error": "receipt JSON not complete"})
                continue
            if not isinstance(value, dict):
                raise ValueError("readout receipt must be a JSON object")
            if not raw.endswith(b"\n") or value.get("complete") is not True:
                result["errors"].append(
                    {"file": relative, "error": "receipt publication not complete"}
                )
                continue
            data = base64.b64encode(raw).decode("ascii")
            if encoded_bytes + len(data) > MAX_RESPONSE - 65536:
                continue
            encoded_bytes += len(data)
            result["files"].append(
                {
                    "name": relative,
                    "sha256": digest,
                    "bytes": len(raw),
                    "data_base64": data,
                    "source_mtime_ns": before.st_mtime_ns,
                }
            )
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESPONSE:
        raise ValueError("readout response exceeds its byte bound")
    return result
