from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

MAX_EPOCH_INDEX_BYTES = 16_000_000


def canonical_index_bytes(value: Mapping[str, object]) -> bytes:
    """Encode canonical index JSON with its required final newline."""
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (encoded + "\n").encode("utf-8")


def index_sha256(value: Mapping[str, object]) -> str:
    """Hash canonical index content without the self-referential digest field."""
    payload = dict(value)
    payload.pop("index_sha256", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_index_json(path: Path) -> tuple[dict[str, object], bytes]:
    """Read one bounded regular index file and reject unstable or ambiguous JSON."""
    if path.is_symlink():
        raise ValueError("epoch index cannot be a symlink")
    try:
        before_path = path.lstat()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_ino != before_path.st_ino
                or before.st_dev != before_path.st_dev
                or before.st_size > MAX_EPOCH_INDEX_BYTES
            ):
                raise ValueError("epoch index is not a bounded regular file")
            encoded = stream.read(MAX_EPOCH_INDEX_BYTES + 1)
            after = os.fstat(stream.fileno())
    except FileNotFoundError:
        raise
    if (
        len(encoded) > MAX_EPOCH_INDEX_BYTES
        or before.st_size != after.st_size
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
        or before.st_size != len(encoded)
    ):
        raise ValueError("epoch index changed while it was being read")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("epoch index is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("epoch index must contain an object")
    return value, encoded


def write_immutable(path: Path, encoded: bytes) -> None:
    """Create one immutable file, accepting only an exact existing byte match."""
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != len(encoded)
            or path.read_bytes() != encoded
        ):
            raise FileExistsError(f"refusing to replace conflicting epoch index: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_replace(path: Path, encoded: bytes) -> None:
    """Durably replace the mutable latest-index pointer."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"refusing to replace non-regular index pointer: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    """Flush directory metadata after publishing an immutable or replaced file."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    """Hash a checkpoint in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate epoch-index key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def valid_counter(value: object) -> bool:
    """Require a nonnegative integer counter and reject Boolean values."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def epoch_output_root(path: Path) -> Path:
    """Resolve the fit root of a latest index or immutable index snapshot."""
    if path.name == "epoch_index.json" and path.parent.name != "epoch_indexes":
        return path.parent
    if path.parent.name == "epoch_indexes":
        return path.parent.parent
    raise ValueError("epoch index must be the run's latest pointer or immutable snapshot")
