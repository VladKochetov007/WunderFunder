from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict, cast

RAM_MAX = 36 * 1024**3
CPU_MAX = 12
CPU_SMALL_RAM_MAX = 2 * 1024**3
CPU_HELPER_RAM_MAX = 6 * 1024**3
SUPPORT_RAM_RESERVE = 4 * 1024**3
SUPPORT_CPU_RESERVE = 2
VRAM_MAX = 5_000_000_000
BLOCK_SECONDS = 24 * 3600
SLICE = "wnn-campaign.slice"
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
ENVIRONMENT = re.compile(r"(?:WNN_|OMP_|MKL_|OPENBLAS_|NUMBA_|POLARS_|ARROW_|PYTORCH_)[A-Z0-9_]+\Z")
RESERVED_ENV = {
    "WNN_GPU_MEMORY_BYTES",
    "WNN_VRAM_GB",
    "WNN_GPU_FRACTION",
    "WNN_THREADS",
    "WNN_CPU_QUOTA",
    "WNN_MEM_MAX",
    "WNN_SLICE",
    "WNN_SCOPE_UNIT",
    "WNN_RUNTIME_MAX",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "ARROW_NUM_THREADS",
}


class ContentFile(TypedDict):
    path: str
    size: int
    sha256: str


class Limits(TypedDict):
    cpus: int
    ram_bytes: int
    timeout_seconds: int
    gpu_memory_bytes: int
    gpu_stop_bytes: int


class ExperimentSpec(TypedDict):
    version: int
    name: str
    kind: str
    argv: list[str]
    code: list[ContentFile]
    code_sha256: str
    inputs: list[ContentFile]
    artifacts: list[str]
    limits: Limits
    environment: dict[str, str]


def canonical(value: object) -> bytes:
    """Encode the bytes used for scientific identities, independent of formatting."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    """Hash a JSON identity."""
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    """Hash large dependencies without loading them into RAM."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def relative_path(value: object) -> str:
    """Reject paths that could escape the experiment root."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("file path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix() or value == ".":
        raise ValueError(f"expected normalized relative path: {value}")
    return value


def rooted(root: Path, value: str) -> Path:
    """Resolve a declared artifact without following a symlink outside the root."""
    path = (root / relative_path(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"dependency escapes root: {value}")
    return path


def content_file(root: Path, value: str) -> ContentFile:
    """Describe a file by portable content identity."""
    path = rooted(root, value)
    before = path.stat()
    identity = file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {value}")
    return {"path": value, "size": after.st_size, "sha256": identity}


def _integer(value: object, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _files(value: object) -> list[ContentFile]:
    if not isinstance(value, list):
        raise ValueError("file manifests must be lists")
    result: list[ContentFile] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("manifest entries require path, size and sha256")
        name = relative_path(entry["path"])
        size = _integer(entry["size"], 0, 2**63 - 1, "file size")
        if not isinstance(entry["sha256"], str) or not HASH.fullmatch(entry["sha256"]):
            raise ValueError(f"invalid file hash: {name}")
        result.append({"path": name, "size": size, "sha256": entry["sha256"]})
    if len({entry["path"] for entry in result}) != len(result):
        raise ValueError("duplicate manifest paths")
    return sorted(result, key=lambda entry: entry["path"])


def validate_spec(value: object) -> ExperimentSpec:
    """Validate the portable spec; local resource restrictions are separate."""
    required = set(ExperimentSpec.__annotations__)
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"spec requires exactly {sorted(required)}")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported spec version")
    if not isinstance(value["name"], str) or not NAME.fullmatch(value["name"]):
        raise ValueError("invalid experiment name")
    if value["kind"] not in ("cpu", "gpu", "tpu"):
        raise ValueError("kind must be cpu, gpu or tpu")
    argv = value["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(x, str) or not x or "\0" in x for x in argv)
    ):
        raise ValueError("argv must be nonempty literal arguments")
    code, inputs = _files(value["code"]), _files(value["inputs"])
    if not code or digest(code) != value["code_sha256"]:
        raise ValueError("code_sha256 does not match the sorted code manifest")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("at least one required artifact must be declared")
    artifacts = [relative_path(path) for path in artifacts]
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("duplicate artifact paths")
    if set(artifacts) & {entry["path"] for entry in code + inputs}:
        raise ValueError("outputs cannot overwrite inputs or source")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != set(Limits.__annotations__):
        raise ValueError("invalid limit fields")
    _integer(limits["cpus"], 1, 1024, "cpus")
    _integer(limits["ram_bytes"], 1, 2**63 - 1, "ram_bytes")
    _integer(limits["timeout_seconds"], 1, 12 * 3600, "timeout_seconds")
    _integer(limits["gpu_memory_bytes"], 0, 2**63 - 1, "gpu_memory_bytes")
    _integer(limits["gpu_stop_bytes"], 0, 2**63 - 1, "gpu_stop_bytes")
    if value["kind"] == "cpu":
        if limits["gpu_memory_bytes"] or limits["gpu_stop_bytes"]:
            raise ValueError("CPU jobs cannot reserve device memory")
    elif not 0 < limits["gpu_memory_bytes"] < limits["gpu_stop_bytes"]:
        raise ValueError("allocator bytes must be below the observed device stop threshold")
    env = value["environment"]
    if not isinstance(env, dict) or any(
        not isinstance(k, str)
        or not ENVIRONMENT.fullmatch(k)
        or k in RESERVED_ENV
        or not isinstance(v, str)
        or "\0" in v
        for k, v in env.items()
    ):
        raise ValueError("invalid or reserved environment override")
    return cast(ExperimentSpec, {**value, "code": code, "inputs": inputs, "artifacts": artifacts})


def validate_local(spec: ExperimentSpec) -> None:
    """Enforce the authorized local worker ceiling, independently of remote limits."""
    limits = spec["limits"]
    if spec["kind"] not in ("cpu", "gpu"):
        raise ValueError("local adapter does not support TPU")
    cpu_max, ram_max = (8, 24 * 1024**3) if spec["kind"] == "gpu" else (1, CPU_HELPER_RAM_MAX)
    if limits["cpus"] > cpu_max or limits["ram_bytes"] > ram_max:
        raise ValueError("local worker exceeds its CPU/RAM envelope")
    if limits["gpu_stop_bytes"] > 4_800_000_000:
        raise ValueError("local operational VRAM threshold must be <=4.8 decimal GB")


def seal_spec(root: Path, value: dict[str, object]) -> ExperimentSpec:
    """Turn source/input path lists into a content-bound execution spec."""
    result = dict(value)
    for key in ("code", "inputs"):
        paths = result.get(key)
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError(f"unsealed {key} must be a list of paths")
        result[key] = sorted(
            (content_file(root, path) for path in paths), key=lambda entry: entry["path"]
        )
    result["code_sha256"] = digest(result["code"])
    return validate_spec(result)


def verify_files(root: Path, entries: list[ContentFile]) -> None:
    """Fail before execution when any required content changed or disappeared."""
    for entry in entries:
        if content_file(root, entry["path"]) != entry:
            raise ValueError(f"content mismatch: {entry['path']}")
