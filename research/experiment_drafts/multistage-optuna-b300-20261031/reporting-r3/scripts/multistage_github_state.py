"""Private restart-safe state for the node-side publisher."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

STATE_SCHEMA = "wnn.multistage-github-publisher-state.v1"


def save_state(path: Path, value: dict[str, object]) -> None:
    """Durably save nonsecret publisher progress for restart-safe deduplication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("publisher state path cannot be a symlink")
    temporary.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def load_state(path: Path) -> dict[str, object]:
    """Read restart state without following a symlink; state never contains credentials."""
    if not path.exists():
        return {"schema": STATE_SCHEMA, "terminal_seen": []}
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > 1_048_576:
        raise ValueError("publisher state file is unsafe or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        raise ValueError("publisher state schema is unsupported")
    if not isinstance(value.get("terminal_seen"), list):
        raise ValueError("publisher terminal state is malformed")
    return value
