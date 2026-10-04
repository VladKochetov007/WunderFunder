from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

SECRET_NAME = re.compile(
    r"(^|[/_.-])(token|secret|credentials?|password|kaggle\.json|\.env)([/_.-]|$)", re.I
)


def relative_path(value: str) -> str:
    """Reject escaping or credential-like artifact names before transport."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    if SECRET_NAME.search(value) or value.startswith(".git/"):
        raise ValueError(f"Forbidden bundle path: {value!r}")
    return str(path)


def sha256_file(path: Path) -> str:
    """Stream file identity without loading checkpoints or data into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Match the campaign canonical JSON hash convention."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_spec(path: Path) -> dict:
    """Validate the frozen controller spec at the worker trust boundary."""
    spec = json.loads(path.read_text())
    required = {
        "version",
        "name",
        "kind",
        "argv",
        "code",
        "code_sha256",
        "inputs",
        "artifacts",
        "limits",
        "environment",
    }
    if set(spec) != required or spec["version"] != 1:
        raise ValueError("Unsupported experiment spec schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", spec["name"]):
        raise ValueError("Invalid experiment name")
    if (
        not spec["argv"]
        or spec["argv"][0] != "{python}"
        or not all(isinstance(x, str) for x in spec["argv"])
    ):
        raise ValueError("Worker accepts explicit Python argv only")
    code = sorted(spec["code"], key=lambda item: item["path"])
    if canonical_sha256(code) != spec["code_sha256"]:
        raise ValueError("Code aggregate hash mismatch")
    seen = set()
    for item in spec["code"] + spec["inputs"]:
        name = relative_path(item["path"])
        if name in seen or set(item) != {"path", "size", "sha256"}:
            raise ValueError("Duplicate or invalid input descriptor")
        seen.add(name)
        if (
            not isinstance(item["size"], int)
            or item["size"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid content identity")
    for name in spec["artifacts"]:
        relative_path(name)
    for key, value in spec["environment"].items():
        if SECRET_NAME.search(key) or not isinstance(value, str):
            raise ValueError("Credential-like or non-string environment setting")
    limits = spec["limits"]
    for key in ("cpus", "ram_bytes", "timeout_seconds", "gpu_memory_bytes", "gpu_stop_bytes"):
        if not isinstance(limits.get(key), int) or limits[key] <= 0:
            raise ValueError(f"Missing positive resource limit: {key}")
    if limits["gpu_memory_bytes"] > 12 * 1024**3 or limits["gpu_stop_bytes"] > 13 * 1024**3:
        raise ValueError("GPU limit exceeds campaign admission")
    if limits["gpu_memory_bytes"] >= limits["gpu_stop_bytes"]:
        raise ValueError("GPU operating target must be below stop limit")
    return spec


def verify_files(root: Path, descriptors: list[dict]) -> None:
    """Require exact bytes for every declared dependency before launch."""
    for item in descriptors:
        path = root / relative_path(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != item["size"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Dependency identity mismatch: {item['path']}")


def write_json(path: Path, value: object) -> None:
    """Atomically persist finite, machine-readable worker evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
