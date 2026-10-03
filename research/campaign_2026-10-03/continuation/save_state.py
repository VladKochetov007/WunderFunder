from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(__file__).resolve().parent
CONTINUATION = "research/campaign_2026-10-03/continuation"
SOURCE_ROOTS = (
    "src",
    "scripts",
    "native",
    "submission",
    "notebooks",
    CONTINUATION,
    "research/campaign_2026-10-03/eda",
    "research/campaign_2026-10-03/scores",
    "research/campaign_2026-10-03/opt2",
    "research/bandit",
    "research/campaign_2026-09-29/jobs",
    "research/campaign_2026-09-29/artifacts/splits",
    "research/campaign_2026-10-01/jobs",
)
ROOT_FILES = (
    ".gitignore",
    "utils.py",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    ".python-version",
    "README.md",
    "METRIC.md",
    "Makefile",
    ".gitmodules",
    "research/experiments.jsonl",
    "research/hypotheses.jsonl",
    "research/frontier.md",
    "research/campaign_2026-09-29/STOP_NEW_JOBS",
    "research/campaign_2026-10-01/QUEUE_PAUSE",
)
REQUIRED = (
    "scripts/limited",
    "scripts/campaign_job.py",
    "utils.py",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    ".python-version",
    "native/Cargo.toml",
    "native/Cargo.lock",
    *(
        f"{CONTINUATION}/{name}"
        for name in (
            "CONTRACT.txt",
            "resource_contract.json",
            "run_cpu.sh",
            "run_gpu.sh",
            "gpu_queue.sh",
            "watch.py",
            "save_state.py",
            "RESTORE.md",
        )
    ),
)
REFERENCE_ROOTS = ("artifacts", "cache", "datasets", "submissions", "research")
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".codex",
    ".claude",
    ".ruff_cache",
    "node_modules",
    "target",
    "snapshots",
    "locks",
    "review",
    "briefs",
}
SUFFIXES = {
    ".py",
    ".sh",
    ".rs",
    ".toml",
    ".lock",
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".yml",
    ".yaml",
    ".cfg",
    ".ini",
    ".log",
    ".patch",
    ".sha256",
    ".resource",
}
SECRET_NAMES = {"claude_agents.json", "auth.json", "credentials.json", "credentials", ".netrc"}
SECRET_PATTERN = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b|"
    rb"(?i:(?:api_key|password|access_token|refresh_token|client_secret)\s*[=:]\s*[\"']?)[A-Za-z0-9_+/.-]{16,}|"
    rb"https?://[^\s/:]+:[^\s/@]+@"
)
ENV_NAMES = {
    "WNN_SLICE",
    "WNN_MEM_MAX",
    "WNN_THREADS",
    "WNN_CPU_QUOTA",
    "WNN_CPU_PCT",
    "WNN_GPU_FRACTION",
    "WNN_VRAM_GB",
    "WNN_CAMPAIGN_VRAM_BYTES",
    "WNN_LOADER_WORKERS",
    "WNN_LOADER_RAM_GB",
    "WNN_RUNTIME_MAX",
    "WNN_SCOPE_UNIT",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "ARROW_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
}
UNIT_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Slice",
    "ControlGroup",
    "FragmentPath",
    "DropInPaths",
    "CPUQuotaPerSecUSec",
    "MemoryMax",
    "MemoryHigh",
    "MemorySwapMax",
    "MemoryCurrent",
    "RuntimeMaxUSec",
    "TimeoutStopUSec",
    "KillMode",
    "Restart",
    "RestartUSec",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "Environment",
)
MAX_FILE_BYTES = 8 * 1024**2
MAX_PAYLOAD_BYTES = 128 * 1024**2
MAX_ARCHIVE_BYTES = 256 * 1024**2
MAX_MEMBERS = 20000
EXTERNAL_SUFFIXES = {
    ".tar",
    ".zip",
    ".gz",
    ".tgz",
    ".xz",
    ".bz2",
    ".zst",
    ".7z",
    ".rar",
    ".pt",
    ".npy",
    ".npz",
    ".pkl",
    ".parquet",
    ".png",
    ".ipynb",
}
CONTROL_PATHS = {
    f"{CONTINUATION}/READY",
    f"{CONTINUATION}/RESOURCE_FAILURE",
    "research/campaign_2026-09-29/STOP_NEW_JOBS",
    "research/campaign_2026-10-01/QUEUE_PAUSE",
}
EVIDENCE_ONLY_PATHS = CONTROL_PATHS | {f"{CONTINUATION}/LATEST_SNAPSHOT"}


def stamp() -> str:
    """Return a timezone-aware snapshot timestamp."""
    return datetime.now().astimezone().isoformat()


def atomic_write(path: Path, data: bytes) -> None:
    """Publish complete files, fsyncing before replacement."""
    if SECRET_PATTERN.search(data):
        raise ValueError(f"possible credential in metadata; not written: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Persist directory entries before publishing a recovery pointer."""
    directory = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path: Path, value: object) -> None:
    """Write structured metadata transactionally."""
    atomic_write(path, (json.dumps(value, indent=2) + "\n").encode())


def capture(*args: str, directory: Path) -> dict[str, object]:
    """Bound diagnostic memory; retain return codes and missing-tool errors."""
    try:
        with tempfile.TemporaryFile(dir=directory) as output:
            process = subprocess.run(
                args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=20
            )
            output.seek(0)
            data = output.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            return {"returncode": process.returncode, "error": "diagnostic exceeded size bound"}
        return {"returncode": process.returncode, "output": data.decode(errors="replace")}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error": str(error)}


def safe_file(path: Path) -> bool:
    """Exclude links, private session trees, credential files and build output."""
    return (
        not path.is_symlink()
        and path.is_file()
        and path.resolve().is_relative_to(ROOT.resolve())
        and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
        and path.name not in SECRET_NAMES
        and not path.name.startswith(".env")
        and path.suffix not in {".pem", ".key", ".p12", ".pfx"}
    )


def walk(directory: Path) -> list[Path]:
    """Enumerate eligible regular files without traversing links or excluded trees."""
    found = []
    if directory.is_symlink():
        return found
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name not in SKIP_DIRS and not (Path(parent) / name).is_symlink()
        )
        found.extend(
            Path(parent) / name for name in sorted(files) if safe_file(Path(parent) / name)
        )
    return found


def signature(path: Path) -> tuple[int, int, int, int, int]:
    """Detect replacement or mutation while copying/hashing."""
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest(path: Path, deadline: float = float("inf")) -> str:
    """Hash with one MiB working buffers and an optional wall-time limit."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024**2):
            if time.monotonic() >= deadline:
                raise TimeoutError("hash budget exhausted")
            value.update(block)
    return value.hexdigest()


def inventory(
    output: Path, paths: list[Path], budget_bytes: int, seconds: float, reuse: Path | None
) -> dict[str, object]:
    """Stream bounded external hashes; mark all skipped/reused references honestly."""
    prior: dict[str, dict[str, object]] = {}
    if reuse is not None:
        if reuse.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("reuse manifest exceeds metadata size limit")
        with reuse.open() as source:
            for line in source:
                entry = json.loads(line)
                prior[entry["path"]] = entry
    deadline = time.monotonic() + seconds
    used = 0
    counts: dict[str, int] = {}
    with output.open("x") as manifest:
        for path in sorted(set(paths)):
            try:
                before = signature(path)
            except OSError as error:
                manifest.write(
                    json.dumps(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "external": True,
                            "verification": "unverified",
                            "reason": str(error),
                        }
                    )
                    + "\n"
                )
                counts["unverified"] = counts.get("unverified", 0) + 1
                continue
            relative = str(path.relative_to(ROOT))
            entry: dict[str, object] = {
                "path": relative,
                "bytes": before[2],
                "mtime_ns": before[3],
                "external": True,
            }
            old = prior.get(relative, {})
            if (
                old.get("bytes") == before[2]
                and old.get("mtime_ns") == before[3]
                and re.fullmatch(r"[0-9a-f]{64}", str(old.get("sha256", "")))
            ):
                entry.update(
                    sha256=old["sha256"],
                    verification="reused_unverified",
                    reason="matching size/mtime does not revalidate content",
                )
            elif before[2] <= budget_bytes - used and time.monotonic() < deadline:
                used += before[2]
                try:
                    value = digest(path, deadline)
                    if signature(path) == before:
                        entry.update(sha256=value, verification="hashed_now")
                    else:
                        entry.update(verification="unverified", reason="changed during hashing")
                except (OSError, TimeoutError) as error:
                    entry.update(verification="unverified", reason=str(error))
            else:
                entry.update(verification="unverified", reason="byte/time budget; hash before use")
            status = str(entry["verification"])
            counts[status] = counts.get(status, 0) + 1
            manifest.write(json.dumps(entry) + "\n")
        manifest.flush()
        os.fsync(manifest.fileno())
    return {
        "statuses": counts,
        "hash_bytes_charged": used,
        "hash_budget_bytes": budget_bytes,
        "hash_seconds_limit": seconds,
        "missing_roots": [name for name in REFERENCE_ROOTS if not (ROOT / name).exists()],
    }


def sanitized_unit(text: str) -> str:
    """Strip private environment and credential directives from systemd evidence."""
    clean = []
    # Resolve unit-file line continuations before dropping private directives.
    for line in text.replace("\\\n", "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("Environment="):
            values = re.findall(r'(?:"[^"\n]*"|[^\s]+)', stripped.partition("=")[2])
            values = [value for value in values if value.strip('"').partition("=")[0] in ENV_NAMES]
            line = "Environment=" + " ".join(values)
        elif stripped.startswith(
            ("EnvironmentFile=", "PassEnvironment=", "LoadCredential", "SetCredential")
        ):
            line = "# private systemd directive omitted"
        line = re.sub(
            r"(?i)(--(?:token|password|api-key|secret)(?:=|\s+))\S+", r"\1[REDACTED]", line
        )
        clean.append(line)
    return SECRET_PATTERN.sub(b"[REDACTED]", "\n".join(clean).encode()).decode() + "\n"


def unit_evidence(directory: Path) -> dict[str, object]:
    """Capture effective campaign units and definitions; never install/replay them."""
    listed = capture(
        "systemctl",
        "--user",
        "list-units",
        "--all",
        "--plain",
        "--no-legend",
        "wnn*",
        directory=directory,
    )
    installed = capture(
        "systemctl", "--user", "list-unit-files", "--no-legend", "wnn*", directory=directory
    )
    names = {"wnn.slice", "wnn-campaign.slice"}
    for result in (listed, installed):
        for line in str(result.get("output", "")).splitlines():
            candidate = line.split()[0] if line.split() else ""
            if re.fullmatch(r"wnn[-\w]*\.(?:service|scope|slice)", candidate):
                names.add(candidate)
        result["output"] = sanitized_unit(str(result.get("output", "")))
    units = {}
    for name in sorted(names):
        properties = capture(
            "systemctl",
            "--user",
            "show",
            name,
            *[f"--property={key}" for key in UNIT_PROPERTIES],
            directory=directory,
        )
        definition = capture("systemctl", "--user", "cat", name, directory=directory)
        for result in (properties, definition):
            result["output"] = sanitized_unit(str(result.get("output", "")))
        units[name] = {"properties": properties, "definition": definition}
    return {
        "list_units": listed,
        "list_unit_files": installed,
        "units": units,
        "policy": "evidence only; paths require rebasing; never replay services or environment",
    }


def source_paths() -> list[Path]:
    """Include runtime/configuration, incident acknowledgements and dependencies."""
    return sorted(
        set(
            [path for name in SOURCE_ROOTS for path in walk(ROOT / name)]
            + [ROOT / name for name in ROOT_FILES if safe_file(ROOT / name)]
            + [
                path
                for path in (STATE / "review").glob("*")
                if safe_file(path) and path.suffix in SUFFIXES
            ]
        )
    )


def eligible(path: Path) -> bool:
    """Keep small source/metadata and extensionless runtime admission markers."""
    name = str(path.relative_to(ROOT))
    if path.suffix in EXTERNAL_SUFFIXES:
        return False
    return path.stat().st_size <= MAX_FILE_BYTES and (
        path.suffix in SUFFIXES
        or name in ROOT_FILES
        or name in CONTROL_PATHS
        or (path.parent == STATE and path.name not in SECRET_NAMES)
        or path.name in {"limited", "Dockerfile.build"}
    )


def live_evidence(path: Path) -> bool:
    """Runtime observations change during healthy training; source must remain stable."""
    relative = str(path.relative_to(ROOT))
    return relative.startswith(
        (
            "research/campaign_2026-09-29/jobs/",
            "research/campaign_2026-09-29/logs/",
            f"{CONTINUATION}/logs/",
        )
    ) or relative in {
        f"{CONTINUATION}/watch_status.json",
        f"{CONTINUATION}/watch_memory.json",
        f"{CONTINUATION}/ETA.json",
    }


def archive(destination: Path, metadata: Path, paths: list[Path]) -> None:
    """Archive exact bounded bytes; include a per-member integrity manifest."""
    manifest = []
    total = 0
    with tarfile.open(destination, "w", dereference=False) as bundle:
        pairs = [(path, str(path.relative_to(ROOT))) for path in paths]
        pairs += [
            (path, "_recovery/" + path.name)
            for path in sorted(metadata.iterdir())
            if path.is_file()
        ]
        for path, name in pairs:
            before = signature(path)
            if before[2] > MAX_FILE_BYTES:
                raise ValueError(f"metadata/source grew beyond file bound: {name}")
            data = path.read_bytes()
            if signature(path) != before or len(data) != before[2]:
                raise RuntimeError(f"source changed during snapshot: {name}")
            if SECRET_PATTERN.search(data):
                raise RuntimeError(f"possible credential in snapshot input; not published: {name}")
            total += len(data)
            if total > MAX_PAYLOAD_BYTES or len(manifest) >= MAX_MEMBERS:
                raise ValueError("snapshot payload/member bound exceeded")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), path.stat().st_mode & 0o777, 0
            bundle.addfile(info, io.BytesIO(data))
            manifest.append(
                {
                    "path": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "captured_at": stamp(),
                    "live_evidence": path in paths and live_evidence(path),
                }
            )
        evidence = json.dumps(
            {"version": 1, "members": manifest, "standalone_training_restore": False}, indent=2
        ).encode()
        if len(evidence) > MAX_FILE_BYTES:
            raise ValueError("archive integrity manifest exceeded file bound")
        info = tarfile.TarInfo("_recovery/archive_manifest.json")
        info.size, info.mode = len(evidence), 0o644
        bundle.addfile(info, io.BytesIO(evidence))


def snapshot(
    output: Path, with_archive: bool, hash_bytes: int, hash_seconds: float, reuse: Path | None
) -> Path:
    """Build privately and publish only after successful metadata/archive verification."""
    if output.exists():
        raise FileExistsError(f"snapshot destination already exists: {output}")
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, mode=0o700)
    try:
        paths = source_paths()
        included = [path for path in paths if eligible(path)]
        included_names = {str(path.relative_to(ROOT)) for path in included}
        original_signatures = {
            path: signature(path) for path in included if not live_evidence(path)
        }
        missing = [name for name in REQUIRED if name not in included_names]
        if (
            missing
            or not any(name.startswith("native/src/") for name in included_names)
            or not any(name.startswith("submission/") for name in included_names)
            or not any(name.startswith("src/") for name in included_names)
        ):
            raise ValueError(
                f"missing required recovery sources: {missing}; require src/native/src/submission trees"
            )
        head = capture("git", "rev-parse", "HEAD", directory=staging)
        if head.get("returncode") != 0:
            raise RuntimeError("cannot record repository revision")
        write_json(staging / "git-head.json", head)
        relative = sorted(included_names)
        for name, arguments in (
            ("worktree.patch", ("diff", "--binary", "--")),
            ("index.patch", ("diff", "--cached", "--binary", "--")),
            ("git-status.txt", ("status", "--short", "--")),
        ):
            result = capture("git", *arguments, *relative, directory=staging)
            if result.get("returncode") != 0 or "error" in result:
                raise RuntimeError(
                    f"cannot capture complete {name}: {result.get('error', result.get('returncode'))}"
                )
            atomic_write(staging / name, str(result.get("output", "")).encode())
        write_json(staging / "systemd.json", unit_evidence(staging))
        packages = sorted(
            {
                (dist.metadata.get("Name", "unknown"), dist.version)
                for dist in importlib.metadata.distributions()
            }
        )
        write_json(
            staging / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "resources": {key: os.environ[key] for key in ENV_NAMES if key in os.environ},
                "policy": "no credentials, direct install URLs, venv binaries or private sessions",
            },
        )
        external = [
            path
            for name in REFERENCE_ROOTS
            for path in walk(ROOT / name)
            if str(path.relative_to(ROOT)) not in included_names
        ]
        external += [path for path in paths if path not in included]
        references = inventory(
            staging / "artifact_inventory.jsonl", external, hash_bytes, hash_seconds, reuse
        )
        write_json(
            staging / "snapshot.json",
            {
                "version": 1,
                "created_at": stamp(),
                "original_root": str(ROOT),
                "paused_restore": True,
                "standalone_training_restore": False,
                "external_references": references,
                "exclusions": sorted(SKIP_DIRS | SECRET_NAMES),
                "metadata_only": not with_archive,
                "live_evidence_policy": "Each mutable job/log/watcher file is a individually stable observation at its member timestamp, not one globally atomic runtime state; restoration is paused and reconciles units. Source/configuration must remain stable across capture.",
                "required_sources": list(REQUIRED),
                "patch_scope": "archived paths only; files already reflect index/worktree changes; do not reapply patches",
            },
        )
        if with_archive:
            holding = staging / "metadata"
            holding.mkdir()
            for file in list(staging.iterdir()):
                if file != holding:
                    file.rename(holding / file.name)
            path = staging / "research-session.tar"
            archive(path, holding, included)
            if path.stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("archive disk-size bound exceeded")
            with tarfile.open(path, "r:") as bundle:
                checked_members(bundle)
            with path.open("rb") as stored:
                os.fsync(stored.fileno())
            atomic_write(staging / "SHA256SUMS", f"{digest(path)}  research-session.tar\n".encode())
            for file in holding.iterdir():
                file.rename(staging / file.name)
            holding.rmdir()
        if capture("git", "rev-parse", "HEAD", directory=staging) != head:
            raise RuntimeError("repository revision changed during snapshot")
        if any(signature(path) != original for path, original in original_signatures.items()):
            raise RuntimeError(
                "source/configuration changed during snapshot; retry when writers are quiescent"
            )
        sync_directory(staging)
        staging.rename(output)
        sync_directory(output.parent)
        return output
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write_json(
            staging / "FAILURE.json", {"status": "incomplete", "error": str(error), "time": stamp()}
        )
        raise


def checked_members(bundle: tarfile.TarFile) -> tuple[dict[str, tarfile.TarInfo], int]:
    """Validate bounded source/metadata names, required dependencies and every payload."""
    members: dict[str, tarfile.TarInfo] = {}
    total = 0
    for member in bundle:
        parts = PurePosixPath(member.name)
        if (
            parts.is_absolute()
            or ".." in parts.parts
            or "\\" in member.name
            or str(parts) != member.name
            or not member.isfile()
            or member.name in members
        ):
            raise ValueError(f"unsafe/duplicate archive member: {member.name}")
        if parts.suffix in EXTERNAL_SUFFIXES or not (
            member.name.startswith("_recovery/")
            or member.name in ROOT_FILES
            or member.name in CONTROL_PATHS
            or any(member.name.startswith(name + "/") for name in SOURCE_ROOTS)
        ):
            raise ValueError(f"member outside recovery source/metadata policy: {member.name}")
        total += member.size
        if (
            len(members) >= MAX_MEMBERS + 1
            or member.size > MAX_FILE_BYTES
            or total > MAX_PAYLOAD_BYTES + MAX_FILE_BYTES
        ):
            raise ValueError("restore payload/member bound exceeded")
        members[member.name] = member
    manifest_member = members.get("_recovery/archive_manifest.json")
    if manifest_member is None:
        raise ValueError("archive lacks integrity manifest")
    source = bundle.extractfile(manifest_member)
    assert source is not None
    evidence = json.load(source)
    expected = {entry["path"]: entry for entry in evidence["members"]}
    if (
        evidence.get("version") != 1
        or len(expected) != len(evidence["members"])
        or set(expected) != set(members) - {manifest_member.name}
    ):
        raise ValueError("archive manifest/member mismatch")
    if not set(REQUIRED) <= set(members) or not all(
        any(name.startswith(tree + "/") for name in members)
        for tree in ("src", "native/src", "submission")
    ):
        raise ValueError("archive missing required recovery dependencies")
    for name, member in members.items():
        if name == manifest_member.name:
            continue
        source = bundle.extractfile(member)
        assert source is not None
        data = source.read(MAX_FILE_BYTES + 1)
        if (
            len(data) != expected[name]["bytes"]
            or hashlib.sha256(data).hexdigest() != expected[name]["sha256"]
        ):
            raise ValueError(f"payload checksum mismatch: {name}")
    return members, total


def restore_dry_run(
    archive_path: Path, destination: Path, expected_sha256: str
) -> dict[str, object]:
    """Verify/extract metadata and source into a new paused tree; execute nothing."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected SHA256 must have 64 lowercase hexadecimal digits")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES or digest(archive_path) != expected_sha256:
        raise ValueError("archive checksum/size verification failed")
    if destination.exists():
        raise FileExistsError("restore requires a new destination")
    destination.mkdir(parents=True, mode=0o700)
    reason = b"RESTORE DRY RUN: paused; revalidate references, environment, quotas and incidents before controller resumption.\n"
    for name in CONTROL_PATHS - {f"{CONTINUATION}/READY"}:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(reason)
    locks = destination / CONTINUATION / "locks"
    locks.mkdir()
    for slot in range(4):
        (locks / f"cpu-{slot}.lock").touch()
    with tarfile.open(archive_path, "r:") as bundle:
        members, total = checked_members(bundle)
        for name, member in members.items():
            source = bundle.extractfile(member)
            assert source is not None
            data = source.read(MAX_FILE_BYTES + 1)
            restored = (
                "_recovery/original_control_state/" + name if name in EVIDENCE_ONLY_PATHS else name
            )
            target = destination / restored
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(member.mode & 0o777)
    result = {
        "status": "PASS",
        "paused": True,
        "members": len(members),
        "payload_bytes": total,
        "sha256": expected_sha256,
        "services_started": 0,
        "external_artifacts_restored": False,
    }
    write_json(destination / "_recovery/dry_run.json", result)
    return result


def main() -> None:
    """Save portable recovery evidence or verify a paused restoration copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="new snapshot directory; does not update live state"
    )
    parser.add_argument("--hash-budget-mib", type=int, default=256)
    parser.add_argument("--hash-seconds", type=float, default=60)
    parser.add_argument(
        "--hash-manifest", type=Path, help="reuse hashes as explicitly unverified references"
    )
    parser.add_argument("--restore-dry-run", type=Path)
    parser.add_argument("--restore-dir", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.restore_dry_run:
        if not args.restore_dir or not args.expected_sha256:
            parser.error("dry-run requires --restore-dir and --expected-sha256")
        print(
            json.dumps(
                restore_dry_run(args.restore_dry_run, args.restore_dir, args.expected_sha256)
            )
        )
        return
    if not 0 <= args.hash_budget_mib <= 1024 or not 0 < args.hash_seconds <= 120:
        parser.error("hash budget must be 0..1024 MiB and time limit 0..120 seconds")
    output = args.output or STATE / "snapshots" / (
        datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    snapshot(
        output, args.archive, args.hash_budget_mib * 1024**2, args.hash_seconds, args.hash_manifest
    )
    if args.output is None:
        atomic_write(STATE / "LATEST_SNAPSHOT", (str(output.relative_to(ROOT)) + "\n").encode())
    print(
        json.dumps(
            {"snapshot": str(output), "archive": args.archive, "standalone_training_restore": False}
        )
    )


if __name__ == "__main__":
    main()
