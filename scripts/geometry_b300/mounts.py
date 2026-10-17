"""Validate the Linux mount modes used by the B300 handoff."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class MountEntry(NamedTuple):
    """One decoded mount-table entry."""

    target: Path
    filesystem: str
    options: frozenset[str]


def read_mount_table(path: Path = Path("/proc/mounts")) -> list[MountEntry]:
    """Read mount targets, filesystems and option flags without reading input data."""
    entries = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        target = fields[1]
        for escaped, decoded in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            target = target.replace(escaped, decoded)
        entries.append(MountEntry(Path(target), fields[2], frozenset(fields[3].split(","))))
    return entries


def mount_for_path(path: Path, entries: list[MountEntry]) -> MountEntry:
    """Return the most specific mount that contains an existing or future path."""
    resolved = path.resolve()
    matches = [
        entry for entry in entries if resolved == entry.target or entry.target in resolved.parents
    ]
    if not matches:
        raise OSError(f"no mount-table entry covers path: {resolved}")
    return max(matches, key=lambda entry: len(str(entry.target)))


def require_mount_mode(path: Path, entries: list[MountEntry], mode: str) -> MountEntry:
    """Require a path's most-specific mount to be read-only or read-write."""
    if mode not in {"ro", "rw"}:
        raise ValueError(f"unsupported mount mode: {mode}")
    entry = mount_for_path(path, entries)
    if mode not in entry.options:
        raise PermissionError(f"{path} must be mounted {mode}; found {sorted(entry.options)}")
    return entry
