"""Immutable campaign inputs, hashes, and JSON-only file operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKET_ROOT.parents[3]
R3_ROOT = REPO_ROOT / "research/experiment_drafts/b300-transformer-distillation-runtime"
CAMPAIGN_SCHEMA = "wnn.high-compute-r4-campaign.v1"
INNER_SCHEMA = "wnn.sequence-inner-split.v1"
R3_INPUTS = (
    "large_teacher_r3_muon30/recipe.json",
    "large_teacher_r3_muon30/configs/t1280_f0.json",
    "large_teacher_r3_muon30/configs/t1280_f2.json",
    "large_teacher_r3_muon30/configs/t1792_f0.json",
    "large_teacher_r3_muon30/configs/t1792_f2.json",
)
_FILE_HASH_CACHE: dict[tuple[str, int, int, int, int], str] = {}

# JSON is heterogeneous at this boundary; Any stays within these manifest helpers.


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks without loading its payload into memory."""
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    signature = (
        str(resolved),
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_ino,
    )
    cached = _FILE_HASH_CACHE.get(signature)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    after = resolved.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino),
        f"file changed while hashing: {resolved}",
    )
    result = digest.hexdigest()
    _FILE_HASH_CACHE[signature] = result
    return result


def canonical_sha256(value: object) -> str:
    """Return the stable JSON identity used by the R3 runtime and configs."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_bytes(value: object) -> bytes:
    """Serialize finite JSON with deterministic key ordering and a final newline."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def read_object(path: Path, maximum_bytes: int = 32_000_000) -> dict[str, Any]:
    """Read a bounded JSON object from an immutable input or receipt."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"JSON size outside the accepted bound: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_bytes_new(path: Path, contents: bytes) -> None:
    """Publish new bytes atomically, refusing to replace any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json_new(path: Path, value: object) -> str:
    """Publish one immutable JSON file and return its raw-byte SHA256."""
    encoded = json_bytes(value)
    write_bytes_new(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def relative_campaign_path(root: Path, value: str | Path) -> Path:
    """Resolve a relative artifact path against the campaign root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def require(condition: bool, reason: str) -> None:
    """Raise a descriptive integrity error at a campaign boundary."""
    if not condition:
        raise ValueError(reason)


def create_campaign(
    campaign_root: Path,
    data_root: Path,
    outer_paths: dict[int, Path],
    inner_paths: dict[int, Path],
) -> dict[str, Any]:
    """Create the immutable source-bound campaign manifest and its hash receipt."""
    from source_bindings import (
        packet_source_files,
        r3_source_files,
        runtime_source_files,
        source_identity,
    )
    from split_inputs import validate_outer_inner_pair

    root = campaign_root.expanduser().resolve()
    data = data_root.expanduser().resolve()
    require(data.is_dir(), f"data-root must be an existing directory: {data}")
    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("campaign-root must be outside the repository")
    require(
        set(outer_paths) == {0, 2} and set(inner_paths) == {0, 2},
        "exact F0 and F2 splits are required",
    )
    runtime_files = runtime_source_files()
    packet_files = packet_source_files()
    r3_files = r3_source_files()
    roles: dict[str, dict[str, Any]] = {}
    outer_values: dict[int, dict[str, Any]] = {}
    outer_hashes: dict[int, str] = {}
    for fold in (0, 2):
        outer_path = outer_paths[fold].expanduser().resolve(strict=True)
        outer_value = read_object(outer_path)
        require(
            outer_value.get("version") == 1 and int(outer_value.get("fold", -1)) == fold,
            f"bad outer F{fold} manifest",
        )
        outer_values[fold] = outer_value
        outer_hashes[fold] = sha256_file(outer_path)
        roles[f"outer_f{fold}"] = {
            "source_path": str(outer_path),
            "sha256": outer_hashes[fold],
            "fold": fold,
            "schema": "wnn.sequence-campaign-split.v1",
        }
    for fold in (0, 2):
        inner_path = inner_paths[fold].expanduser().resolve(strict=True)
        inner_value = read_object(inner_path)
        validate_outer_inner_pair(
            outer_paths[fold].expanduser().resolve(strict=True),
            outer_values[fold],
            inner_path,
            inner_value,
            fold,
        )
        roles[f"inner_f{fold}"] = {
            "source_path": str(inner_path),
            "sha256": sha256_file(inner_path),
            "fold": fold,
            "schema": INNER_SCHEMA,
            "parent_role": f"outer_f{fold}",
            "parent_path": roles[f"outer_f{fold}"]["source_path"],
            "parent_sha256": outer_hashes[fold],
        }
    manifest: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "data_root": str(data),
        "runtime_source": source_identity(runtime_files),
        "runtime_source_sha256": canonical_sha256(runtime_files),
        "packet_source": source_identity(packet_files),
        "r3_source": source_identity(r3_files),
        "outer_splits": {f"F{fold}": roles[f"outer_f{fold}"] for fold in (0, 2)},
        "inner_splits": {f"F{fold}": roles[f"inner_f{fold}"] for fold in (0, 2)},
    }
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    for relative, value in (("campaign.json", manifest),):
        path = root / relative
        encoded = json_bytes(value)
        write_bytes_new(
            root / "campaign.sha256", hashlib.sha256(encoded).hexdigest().encode() + b"\n"
        )
        write_bytes_new(path, encoded)
    return manifest


def load_campaign(campaign_root: Path) -> dict[str, Any]:
    """Load a campaign only while all runtime, packet, R3, and split inputs match."""
    from source_bindings import (
        _validate_source_identity,
        packet_source_files,
        r3_source_files,
        runtime_source_files,
    )
    from split_inputs import validate_role_pair

    root = campaign_root.expanduser().resolve(strict=True)
    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("campaign-root must be outside the repository")
    manifest_path = root / "campaign.json"
    stored_bytes = manifest_path.read_bytes()
    digest_path = root / "campaign.sha256"
    require(
        digest_path.read_text(encoding="utf-8").strip() == hashlib.sha256(stored_bytes).hexdigest(),
        "campaign manifest hash mismatch",
    )
    manifest = read_object(manifest_path)
    manifest["campaign_sha256"] = hashlib.sha256(stored_bytes).hexdigest()
    require(manifest.get("schema") == CAMPAIGN_SCHEMA, "unsupported campaign manifest schema")
    _validate_source_identity("runtime source", manifest["runtime_source"], runtime_source_files())
    require(
        manifest.get("runtime_source_sha256") == manifest["runtime_source"]["sha256"],
        "runtime source hash alias differs from its file map",
    )
    _validate_source_identity("packet source", manifest["packet_source"], packet_source_files())
    _validate_source_identity("R3 source", manifest["r3_source"], r3_source_files())
    for fold in (0, 2):
        outer_role = manifest["outer_splits"][f"F{fold}"]
        outer_path = Path(outer_role["source_path"]).resolve(strict=True)
        outer = validate_role_pair(
            f"outer_f{fold}", outer_role, fold, "wnn.sequence-campaign-split.v1"
        )
        require(sha256_file(outer_path) == outer_role["sha256"], "outer split hash mismatch")
        inner_role = manifest["inner_splits"][f"F{fold}"]
        require(
            inner_role.get("parent_role") == f"outer_f{fold}", "inner split parent role changed"
        )
        require(
            inner_role.get("parent_path") == str(outer_path),
            "inner split parent path binding changed",
        )
        require(
            inner_role.get("parent_sha256") == outer_role["sha256"],
            "inner split parent hash binding changed",
        )
        validate_role_pair(f"inner_f{fold}", inner_role, fold, INNER_SCHEMA, outer, outer_path)
    return manifest


def campaign_digest(campaign_root: Path) -> str:
    """Return the raw immutable campaign manifest hash after validation."""
    load_campaign(campaign_root)
    return sha256_file(campaign_root.resolve() / "campaign.json")
