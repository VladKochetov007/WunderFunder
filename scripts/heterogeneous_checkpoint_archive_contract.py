"""Stdlib-only identities and completed-fit checks for seven owned checkpoint roles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# Scientific/worker metadata are heterogeneous JSON and are checked before archiving.
JsonObject = dict[str, Any]
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
NEW_RUNS = (
    "het6_lstm_s1201",
    "het6_tcn_s1202",
    "het6_control_s1301",
    "het6_control_s1302",
    "het6_heterogeneous_s1301",
    "het6_heterogeneous_s1302",
)
W18 = "tl3c_w18final_distilled_s5"
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|ghp_[A-Za-z0-9]{30,})")


def canonical(value: object) -> str:
    """Hash the trainer's canonical JSON encoding without importing its runtime."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    """Hash an original file incrementally without deserializing checkpoints."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value: object) -> bytes:
    """Encode bounded evidence while refusing apparent embedded credential values."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if SECRET.search(raw.decode()):
        raise ValueError("apparent credential in evidence metadata")
    return raw


def safe_name(name: str) -> None:
    """Reject archive traversal, absolute paths and platform-dependent separators."""
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError("unsafe archive entry name")


@dataclass(frozen=True)
class FileEntry:
    """An immutable byte identity captured before any ZIP is created."""

    source: Path
    destination: str
    size: int
    sha256: str
    stat_key: tuple[int, int, int, int]
    raw: bytes | None = None

    def manifest(self) -> JsonObject:
        """Return the portable identity, excluding local filesystem handles."""
        return {"path": self.destination, "bytes": self.size, "sha256": self.sha256}


def capture(
    path: Path, destination: str, *, text: bool = False, expected: str | None = None
) -> FileEntry:
    """Pin an original regular file and optionally retain one metadata-byte snapshot."""
    safe_name(destination)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked archive input: {path.name}")
    before = path.stat()
    limit = 16 << 20 if text else 64 << 20
    if not 0 < before.st_size <= limit:
        raise ValueError(f"archive input outside its byte bound: {path.name}")
    raw = path.read_bytes() if text else None
    digest = hashlib.sha256(raw).hexdigest() if raw is not None else sha256(path)
    after = path.stat()

    def key(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if key(before) != key(after) or expected is not None and digest != expected:
        raise ValueError(f"archive input identity changed: {path.name}")
    if raw is not None and SECRET.search(raw.decode("utf-8")):
        raise ValueError(f"apparent credential in archive input: {path.name}")
    return FileEntry(path.resolve(), destination, before.st_size, digest, key(before), raw)


def read_json(path: Path) -> tuple[JsonObject, bytes]:
    """Read one stable bounded JSON object without altering original bytes."""
    item = capture(path, path.name, text=True)
    value = json.loads(item.raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value, item.raw


def fit_receipt(root: Path, run: str) -> tuple[JsonObject, list[FileEntry]] | None:
    """Find the exact completed owned fit; profiles and failed attempts are not fits."""
    matches = []
    for path in sorted((root / "requests").glob("*.json")):
        request, _ = read_json(path)
        command = request.get("command", [])
        if request.get("phase") not in {"fit", "scientific_fit"} or "--name" not in command:
            continue
        position = command.index("--name")
        if position + 1 >= len(command) or command[position + 1] != run:
            continue
        status_path = root / "status" / path.name
        if not status_path.is_file():
            continue
        status, _ = read_json(status_path)
        if status.get("status") != "complete" or status.get("return_code") != 0:
            continue
        if (
            status.get("schema") != "wnn.owned-command.v1"
            or status.get("request_sha256") != sha256(path)
            or status.get("command_sha256") != canonical(command)
        ):
            raise ValueError(f"completed fit receipt identity mismatch: {run}")
        index = next(
            (
                i
                for i, part in enumerate(command)
                if str(part).replace("\\", "/").endswith("/scripts/train_rnn.py")
            ),
            None,
        )
        if index is None:
            raise ValueError(f"fit receipt does not name the canonical trainer: {run}")
        matches.append(
            (
                {
                    "request": request,
                    "status": status,
                    "source_root": str(Path(command[index]).parent.parent),
                },
                [
                    capture(path, f"{run}/metadata/request/{path.name}", text=True),
                    capture(status_path, f"{run}/metadata/status/{path.name}", text=True),
                ],
            )
        )
    if len(matches) > 1:
        raise ValueError(f"multiple completed fit attempts share one run identity: {run}")
    return matches[0] if matches else None


def validate_run(
    run: str,
    epochs: int,
    provenance: JsonObject,
    history: list[JsonObject],
    split: JsonObject,
    source_root: Path,
) -> None:
    """Check actual fixed selection, complete primary history and trainer source bytes."""
    args, fit = provenance.get("args", {}), provenance.get("fit_row_groups", {})

    def same_members(actual: object, expected: list[int]) -> bool:
        return (
            isinstance(actual, list)
            and all(type(group) is int for group in actual)
            and len(actual) == len(set(actual)) == len(expected)
            and set(actual) == set(expected)
        )

    if (
        args.get("name") != run
        or args.get("epochs") != epochs
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epochs - 1
        or args.get("precision") != "fp32"
        or provenance.get("selected_epoch") != epochs - 1
        or provenance.get("split_manifest_sha256") != SPLIT_SHA256
        or not same_members(fit.get("train"), split["fit"]["train"])
        or fit.get("valid") != []
        or not same_members(provenance.get("evaluation_row_groups"), split["evaluation"]["valid"])
    ):
        raise ValueError(f"fixed recipe or official membership differs: {run}")
    if len(history) != epochs or any(
        row.get("epoch") != index
        or row.get("selected_rows") != 3528273
        or row.get("split") != "vfold1873"
        or not isinstance(row.get("weighted_pearson"), (int, float))
        or not math.isfinite(row["weighted_pearson"])
        for index, row in enumerate(history)
    ):
        raise ValueError(f"full fixed-epoch validation history is incomplete: {run}")
    identity = provenance.get("trainer_implementation", {})
    files = identity.get("files")
    if not isinstance(files, dict) or identity.get("sha256") != canonical(files):
        raise ValueError(f"trainer source identity is invalid: {run}")
    source_root = source_root.resolve(strict=True)
    for relative, expected in files.items():
        safe_name(relative)
        path = source_root / relative
        if not path.resolve(strict=True).is_relative_to(source_root):
            raise ValueError("trainer source escapes its actual request-owned snapshot")
        capture(path, f"source-check/{relative}", text=True, expected=expected)


def replay_files(root: Path, source_roots: list[Path]) -> tuple[list[FileEntry], list[JsonObject]]:
    """Inventory original runtime Python/dependency files and small explicit replay evidence."""
    files, sources = [], []
    unique_roots = list(dict.fromkeys(path.resolve(strict=True) for path in source_roots))
    for index, source in enumerate(unique_roots):
        prefix = f"sources/{index:02d}-{source.name}"
        paths = set(source.glob("*.py"))
        for name in ("src", "scripts"):
            paths.update((source / name).rglob("*.py"))
        paths.update(
            path for name in ("pyproject.toml", "uv.lock") if (path := source / name).is_file()
        )
        snapshot = [
            capture(path, f"{prefix}/{path.relative_to(source).as_posix()}", text=True)
            for path in sorted(paths)
        ]
        identity = source.with_name(source.name + ".identity.json")
        if identity.is_file():
            snapshot.append(capture(identity, f"{prefix}/original-source-identity.json", text=True))
        files.extend(snapshot)
        sources.append(
            {
                "original_root": str(source),
                "archive_prefix": prefix,
                "files": [item.manifest() for item in snapshot],
                "scope": "archived Python/dependency subset only; original identity JSON may describe unarchived capsule files; semantic trainer digest is separate",
            }
        )
    tool_paths = set(root.glob("*.py"))
    for directory in root.glob("tools*"):
        if directory.is_dir() and not directory.is_symlink():
            tool_paths.update(directory.rglob("*.py"))
    files.extend(
        capture(path, f"tools/{path.relative_to(root).as_posix()}", text=True)
        for path in sorted(tool_paths)
    )
    metadata_names = (
        "nodes-r1.json",
        "w18-node-r1.json",
        "galam-model-source-admission-r2.json",
        "galam-fit-export-r1/source-record.json",
        "galam-fit-export-r1/qualification.json",
        "galam-fit-export-r1/predictions.json",
        "gru-only-targets/metadata.json",
        "heterogeneous-targets/metadata.json",
    )
    metadata = [root / name for name in metadata_names if (root / name).is_file()]
    metadata.extend(sorted((root / "requests").glob("*.json")))
    metadata.extend(sorted((root / "status").glob("*.json")))
    files.extend(
        capture(path, f"evidence/{path.relative_to(root).as_posix()}", text=True)
        for path in metadata
    )
    return files, sources
