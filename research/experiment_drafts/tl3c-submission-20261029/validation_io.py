"""Receiver-only archive, input identity and resource constraints; no model imports."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def sha256(path: Path) -> str:
    """Hash immutable input bytes, rejecting a change during the read."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """Read a bounded regular JSON object."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 << 20:
        raise ValueError(f"expected bounded regular JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def constrain_receiver() -> dict:
    """Require the target OS/ABI and an offline container, then cap one CPU and RAM."""
    if sys.version_info[:2] != (3, 11) or platform.system() != "Linux":
        raise RuntimeError("run on Linux in the offline Python 3.11 receiver container")
    if platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("native runtime requires x86-64")
    if os.environ.get("PYTHONPATH") or os.environ.get("LD_LIBRARY_PATH"):
        raise RuntimeError("unset PYTHONPATH and LD_LIBRARY_PATH in the receiver")
    interfaces = {path.name for path in Path("/sys/class/net").iterdir()}
    if interfaces - {"lo"}:
        raise RuntimeError("use a --network none container; non-loopback interfaces remain")
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("no allowed CPU")
    os.sched_setaffinity(0, {available[0]})
    maximum = 16_000_000_000
    cpu_limit = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
    memory_limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if (
        cpu_limit[0] == "max"
        or int(cpu_limit[0]) / int(cpu_limit[1]) > 1
        or memory_limit == "max"
        or int(memory_limit) > maximum
    ):
        raise RuntimeError("require cgroup v2 container limits --cpus 1 --memory 16000000000")
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
    return {
        "python": sys.version,
        "machine": platform.machine(),
        "cpu": available[0],
        "affinity": sorted(os.sched_getaffinity(0)),
        "address_space_limit_bytes": ceiling,
        "network_interfaces": sorted(interfaces),
        "cgroup_cpu_max": cpu_limit,
        "cgroup_memory_max_bytes": int(memory_limit),
        "thread_environment": {
            key: os.environ[key]
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def extract_candidate(path: Path, receipt: dict, destination: Path) -> dict:
    """Bind the exact ZIP and copy only safe regular entries into an empty directory."""
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("extraction requires a fresh empty directory")
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("extraction destination has a symlink ancestor")
    artifact = receipt.get("artifact", {})
    if receipt.get("schema") != "tl3c-submission-candidate-v1":
        raise ValueError("unsupported build receipt schema")
    digest = sha256(path)
    if artifact.get("sha256") != digest or artifact.get("bytes") != path.stat().st_size:
        raise ValueError("build receipt does not bind the ZIP bytes")
    if path.stat().st_size > 20_000_000:
        raise ValueError("ZIP exceeds 20 MB")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not 1 <= len(entries) <= 64 or sum(x.file_size for x in entries) > 100 << 20:
            raise ValueError("oversized archive contents")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        names = set()
        for item in entries:
            name = PurePosixPath(item.filename)
            mode = stat.S_IFMT(item.external_attr >> 16)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in item.filename
                or not name.parts
                or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                or name.as_posix().rstrip("/") in names
            ):
                raise ValueError(f"unsafe/duplicate archive entry: {item.filename}")
            names.add(name.as_posix().rstrip("/"))
            target = destination.joinpath(*name.parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    for key in ("solution.py", "members.json"):
        if key not in names:
            raise ValueError(f"missing {key}")
    members = read_json(destination / "members.json")
    if members.get("mode") != receipt.get("mode"):
        raise ValueError("runtime mode differs from build receipt")
    shipped = members.get("members")
    declared = receipt.get("members")
    if (
        not isinstance(shipped, list)
        or not isinstance(declared, list)
        or len(shipped) != len(declared)
    ):
        raise ValueError("member inventory mismatch")
    for actual, expected in zip(shipped, declared, strict=True):
        if any(actual.get(key) != expected.get(key) for key in ("run", "coef", "family")):
            raise ValueError("member recipe mismatch")
        weight = actual.get("weights")
        if weight != expected.get("weight_file") or weight not in names:
            raise ValueError("member weight path mismatch")
        if sha256(destination / weight) != expected.get("weight_sha256"):
            raise ValueError("member weight hash mismatch")
    return {"sha256": digest, "bytes": path.stat().st_size, "mode": receipt["mode"]}


def selected_groups(split: Path, maximum: int | None, full: bool) -> list[int]:
    """Require the immutable official train-only/full-valid split."""
    if sha256(split) != SPLIT_SHA:
        raise ValueError("unexpected chronological split bytes")
    manifest = read_json(split)
    train = manifest.get("fit", {}).get("train", [])
    valid = manifest.get("evaluation", {}).get("valid", [])
    if (
        len(train) != 10325
        or len(set(train)) != 10325
        or manifest.get("fit", {}).get("valid", [])
        or len(valid) != 1873
        or len(set(valid)) != 1873
        or any(type(group) is not int or group < 0 for group in valid)
    ):
        raise ValueError("unexpected fit/evaluation membership")
    if full:
        return valid
    if maximum is None or not 2 <= maximum <= len(valid):
        raise ValueError("profile requires --max-sequences between 2 and 1873")
    return valid[:maximum]
