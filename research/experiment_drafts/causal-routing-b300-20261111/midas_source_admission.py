"""Pinned source manifest validation for Signal Routes MIDAS evidence intake."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any

from midas_evidence_io import _canonical, _hash_file, _read

SHA = re.compile(r"^[0-9a-f]{64}$")
SOURCE_MANIFEST_SHA256 = {
    "transformer": "b6c2004cd5b916a8976973217b944d44f79464a557309b31c42f8912ef7711bf",
}


def _packet_source_files(packet: Path) -> dict[str, str]:
    """Hash every regular Python file in the packet runtime, rejecting symlinks."""
    runtime = packet / "runtime"
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("packet runtime source tree is missing or unsafe")

    files: dict[str, str] = {}

    def fail_walk(error: OSError) -> None:
        """Do not let an unreadable directory silently shrink the source map."""
        raise error

    for current, directories, names in os.walk(runtime, followlinks=False, onerror=fail_walk):
        parent = Path(current)
        for name in (*directories, *names):
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"packet runtime contains a symlink: {path.relative_to(packet)}")
        for name in names:
            path = parent / name
            if path.suffix != ".py":
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"packet runtime source is not a regular file: {path.name}")
            digest, _ = _hash_file(path, 64 << 20)
            key = f"signal_routes/{path.relative_to(packet).as_posix()}"
            files[key] = digest
    return files


def validate_source_map(
    packet: Path,
    producer_files: dict[str, Any],
    runtime_commit: str,
    bindings_sha256: str,
) -> None:
    """Require the pinned family runtime and complete current packet source map."""
    sources = packet / "sources"
    if packet.is_symlink() or not packet.is_dir() or sources.is_symlink() or not sources.is_dir():
        raise ValueError("packet source-manifest directory is missing or unsafe")
    filename = "runtime_source_files.json"
    manifest, raw = _read(sources / filename, 1 << 20)
    if hashlib.sha256(raw).hexdigest() != SOURCE_MANIFEST_SHA256["transformer"]:
        raise ValueError("canonical family runtime source manifest differs from its packet pin")
    if set(manifest) != {"schema", "runtime_commit", "source_sha256", "files"}:
        raise ValueError("canonical runtime source manifest schema is invalid")
    files, source_sha = manifest.get("files"), manifest.get("source_sha256")
    if (
        manifest.get("schema") != "impact81.canonical-runtime-source-files.v1"
        or manifest.get("runtime_commit") != runtime_commit
        or not isinstance(files, dict)
        or not files
        or not isinstance(source_sha, str)
        or not SHA.fullmatch(source_sha)
        or hashlib.sha256(_canonical(files)).hexdigest() != source_sha
    ):
        raise ValueError("canonical family runtime manifest does not bind its source files")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or "\\" in name
            or name.startswith("/")
            or ".." in name.split("/")
            or not name.endswith(".py")
            or not isinstance(digest, str)
            or not SHA.fullmatch(digest)
            or not (
                name.startswith("runtime/src/")
                or name in {"runtime/scripts/train_sequence.py", "runtime/utils.py"}
            )
        ):
            raise ValueError("canonical runtime manifest contains an invalid source entry")

    expected = dict(files)
    packet_files = _packet_source_files(packet)
    if expected.keys() & packet_files.keys():
        raise ValueError("canonical and packet source paths unexpectedly overlap")
    expected.update(packet_files)
    if expected.get("signal_routes/runtime/signal_routes/bindings.py") != bindings_sha256:
        raise ValueError("manifest-bound packet source pin differs from current bindings")
    if producer_files != expected:
        missing = len(expected.keys() - producer_files.keys())
        unexpected = len(producer_files.keys() - expected.keys())
        changed = sum(
            producer_files[name] != expected[name]
            for name in expected.keys() & producer_files.keys()
        )
        raise ValueError(
            "producer source map is incomplete or differs from pinned sources "
            f"(missing={missing}, unexpected={unexpected}, changed={changed})"
        )
