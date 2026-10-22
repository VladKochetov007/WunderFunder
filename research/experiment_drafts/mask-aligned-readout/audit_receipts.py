"""Atomic immutable JSON writes for receiver audit evidence."""

from __future__ import annotations

# Any represents dynamic receipt objects validated by stage-specific checks.
import json
from pathlib import Path
from typing import Any


def write_receipt(path: Path, record: dict[str, Any]) -> None:
    """Write one audit receipt atomically and refuse to replace existing evidence."""
    if path.exists():
        raise FileExistsError(f"audit receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
