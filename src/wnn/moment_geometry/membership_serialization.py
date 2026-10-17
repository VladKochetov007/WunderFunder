"""Write source-qualified membership rows in a compact reviewable JSON format."""

from __future__ import annotations

import json
from pathlib import Path


def _is_membership_rows(value: list[object]) -> bool:
    """Identify arrays of source, row-group, component and fold tuples."""
    return bool(value) and all(
        isinstance(row, (list, tuple))
        and len(row) == 4
        and isinstance(row[0], str)
        and row[0] in {"train", "valid"}
        and all(isinstance(item, int) and not isinstance(item, bool) for item in row[1:])
        for row in value
    )


def _render(value: object, depth: int) -> str:
    """Render containers with one inline source-qualified membership tuple per line."""
    padding = "  " * depth
    child_padding = "  " * (depth + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = [
            f"{child_padding}{json.dumps(key)}: {_render(item, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{padding}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if _is_membership_rows(value):
            entries = [f"{child_padding}{json.dumps(row, separators=(',', ':'))}" for row in value]
        else:
            entries = [f"{child_padding}{_render(item, depth + 1)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{padding}]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_membership_json(path: Path, value: dict[str, object]) -> None:
    """Write a reviewable JSON artifact with compact fixed-width membership rows."""
    path.write_text(_render(value, 0) + "\n")
