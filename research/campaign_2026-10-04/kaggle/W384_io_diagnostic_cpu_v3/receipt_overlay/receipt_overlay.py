from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

CHUNK_BYTES = 8 * 1024**2
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
MFD_CLOEXEC = getattr(os, "MFD_CLOEXEC", 0x0001)
MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
REQUIRED_SEALS = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL


class ReceiptError(RuntimeError):
    """Raised when a receipt cannot be bound to verified mounted content."""


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class DatasetBinding:
    slug: str
    version: int
    mount_root: Path


@dataclass(frozen=True)
class ReceiptRequest:
    path: Path
    expected_sha256: str
    expected_size: int
    dataset: DatasetBinding
    logical_paths: tuple[str, ...]


@dataclass(frozen=True)
class MountIdentity:
    mount_id: int
    mount_point: str
    device: int
    readonly: bool


MountProbe = Callable[[Path], MountIdentity]


def _mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def mount_identity(path: Path) -> MountIdentity:
    """Return the containing mount and require agreement with statvfs RO state."""
    resolved = path.resolve(strict=True)
    candidates: list[tuple[int, int, str, bool]] = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_point = Path(_mount_field(fields[4])).resolve(strict=True)
            resolved.relative_to(mount_point)
            options = set(fields[5].split(",")) | set(fields[separator + 3].split(","))
            candidates.append(
                (len(str(mount_point)), int(fields[0]), str(mount_point), "ro" in options)
            )
        except (ValueError, IndexError, FileNotFoundError):
            continue
    if not candidates:
        raise ReceiptError(f"no containing mount found for {resolved}")
    _, mount_id, mount_point, mountinfo_readonly = max(candidates)
    statvfs_readonly = bool(os.statvfs(resolved).f_flag & getattr(os, "ST_RDONLY", 1))
    readonly = mountinfo_readonly and statvfs_readonly
    return MountIdentity(mount_id, mount_point, resolved.stat().st_dev, readonly)


def strict_sha256_file(path: Path, deadline_monotonic: float | None = None) -> str:
    """Stream a normal full content hash for receipt misses or invalidations."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("strict receipt fallback reached its wall deadline")
            digest.update(block)
    return digest.hexdigest()


def load_input_manifest(
    manifest_path: Path,
    experiment_root: Path,
    expected_kernel_id: str,
    expected_dataset_versions: dict[str, int],
    expected_root_manifest_sha256: str,
) -> list[ReceiptRequest]:
    """Load root-bound source descriptors without treating them as content proof."""
    payload = json.loads(manifest_path.read_text())
    recorded_sha256 = payload.pop("root_manifest_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    observed_sha256 = hashlib.sha256(canonical).hexdigest()
    if observed_sha256 != recorded_sha256 or observed_sha256 != expected_root_manifest_sha256:
        raise ReceiptError("input root manifest self-hash mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("proposed_kernel_id") != expected_kernel_id
    ):
        raise ReceiptError("input root manifest kernel/schema mismatch")
    versions = payload.get("provider_dataset_versions")
    if versions != dict(sorted(expected_dataset_versions.items())):
        raise ReceiptError("input root manifest provider versions mismatch")
    prototype = payload.get("overlay_prototype")
    if not isinstance(prototype, dict) or prototype.get("sha256") != strict_sha256_file(
        Path(__file__)
    ):
        raise ReceiptError("receipt overlay source differs from the root-pinned prototype")
    root = experiment_root.resolve(strict=True)
    requests = []
    for source in payload.get("sources", []):
        dataset = source.get("dataset_slug")
        version = source.get("dataset_version")
        if dataset not in versions or versions[dataset] != version:
            raise ReceiptError("source dataset version is not pinned by provider inventory")
        expected_mount_root = Path("/kaggle/input/datasets") / dataset
        mount_root = Path(source.get("dataset_mount_root", ""))
        if mount_root != expected_mount_root:
            raise ReceiptError("source mount root differs from Kaggle dataset slug")
        filename = PurePosixPath(source.get("filename", ""))
        if filename.is_absolute() or len(filename.parts) != 1 or filename.name in {"", ".", ".."}:
            raise ReceiptError("source filename is not a single safe component")
        size = source.get("expected_size")
        sha256 = source.get("expected_sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReceiptError("source size is invalid")
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
            raise ReceiptError("source SHA-256 is invalid")
        logical_paths = source.get("logical_paths")
        if not isinstance(logical_paths, list) or not logical_paths:
            raise ReceiptError("source logical paths are missing")
        aliases = []
        for logical_path in logical_paths:
            relative = PurePosixPath(logical_path)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ReceiptError("source logical path is unsafe")
            alias = (root / Path(*relative.parts)).resolve(strict=False)
            if not alias.is_relative_to(root):
                raise ReceiptError("source logical path escapes experiment root")
            aliases.append(str(alias))
        binding = DatasetBinding(dataset, version, mount_root)
        requests.append(
            ReceiptRequest(
                path=mount_root / filename.name,
                expected_sha256=sha256,
                expected_size=size,
                dataset=binding,
                logical_paths=tuple(aliases),
            )
        )
    if not requests:
        raise ReceiptError("input root manifest has no source descriptors")
    return requests


def _fd_sha256(fd: int, deadline_monotonic: float | None = None) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("initial receipt full hash reached its wall deadline")
        block = os.read(fd, CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _fd_is_readonly(fd: int) -> bool:
    return fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY


def _memfd_create(name: str) -> int:
    """Call Linux memfd_create even when this Python build omits its os wrapper."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, MFD_CLOEXEC | MFD_ALLOW_SEALING)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        create = libc.memfd_create
    except AttributeError as exc:
        raise ReceiptError("sealed memfd receipts are unavailable") from exc
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(name.encode(), MFD_CLOEXEC | MFD_ALLOW_SEALING)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _identity_from_path(path: Path) -> FileIdentity:
    return FileIdentity.from_stat(path.stat())


def _same_file_identity(left: FileIdentity, right: FileIdentity) -> bool:
    return left == right


def _validate_request(
    request: ReceiptRequest,
    dataset_versions: dict[str, int],
    allowed_mount_roots: tuple[Path, ...],
) -> tuple[Path, Path]:
    if not SHA256_RE.fullmatch(request.expected_sha256):
        raise ReceiptError("invalid expected SHA-256")
    if (
        not isinstance(request.expected_size, int)
        or isinstance(request.expected_size, bool)
        or request.expected_size < 0
        or not request.logical_paths
    ):
        raise ReceiptError("invalid size or missing logical path")
    if dataset_versions.get(request.dataset.slug) != request.dataset.version:
        raise ReceiptError("dataset version differs from the frozen provider inventory")
    root = request.dataset.mount_root.resolve(strict=True)
    resolved = request.path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReceiptError("resolved source escapes its bound dataset mount") from exc
    if not any(root.is_relative_to(allowed.resolve()) for allowed in allowed_mount_roots):
        raise ReceiptError("dataset mount root is outside the allowed roots")
    return root, resolved


@dataclass
class OpenReceiptSet:
    """A verified receipt bundle whose source descriptors remain open."""

    kernel_id: str
    root_manifest_sha256: str
    dataset_versions: dict[str, int]
    entries: list[dict]
    source_fds: list[int]
    manifest_fd: int
    reused_count: int = 0
    fallback_count: int = 0

    def pass_fds(self) -> tuple[int, ...]:
        return (self.manifest_fd, *self.source_fds)

    def close(self) -> None:
        for fd in [self.manifest_fd, *self.source_fds]:
            with contextlib.suppress(OSError):
                os.close(fd)


def build_receipt_set(
    kernel_id: str,
    dataset_versions: dict[str, int],
    requests: Iterable[ReceiptRequest],
    mount_probe: MountProbe = mount_identity,
    allowed_mount_roots: tuple[Path, ...] = (Path("/kaggle/input"),),
    deadline_monotonic: float | None = None,
) -> OpenReceiptSet:
    """Hash unique immutable sources once and seal the exact identities for children."""
    if (
        not kernel_id
        or not dataset_versions
        or any(version <= 0 for version in dataset_versions.values())
    ):
        raise ReceiptError("kernel identity and positive dataset versions are required")
    grouped: dict[str, tuple[ReceiptRequest, list[str], Path]] = {}
    for request in requests:
        root, resolved = _validate_request(request, dataset_versions, allowed_mount_roots)
        key = str(resolved)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = (request, list(request.logical_paths), root)
            continue
        original, aliases, original_root = previous
        same_binding = (
            original.expected_sha256 == request.expected_sha256
            and original.expected_size == request.expected_size
            and original.dataset == request.dataset
            and original_root == root
        )
        if not same_binding:
            raise ReceiptError("one resolved file has conflicting descriptor identities")
        aliases.extend(request.logical_paths)

    entries: list[dict] = []
    fds: list[int] = []
    try:
        for resolved_value, (request, logical_paths, root) in sorted(grouped.items()):
            resolved = Path(resolved_value)
            before = _identity_from_path(resolved)
            if before.size != request.expected_size:
                raise ReceiptError("source size differs from frozen descriptor")
            mount_before = mount_probe(resolved)
            if (
                not resolved.is_relative_to(Path(mount_before.mount_point).resolve(strict=True))
                or not root.is_relative_to(Path(mount_before.mount_point).resolve(strict=True))
                or not mount_before.readonly
                or mount_before.device != before.device
            ):
                raise ReceiptError("source mount is writable or device identity disagrees")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(resolved, flags)
            fds.append(fd)
            if not _fd_is_readonly(fd):
                raise ReceiptError("held source descriptor is not read-only")
            fd_before = FileIdentity.from_stat(os.fstat(fd))
            if not _same_file_identity(before, fd_before):
                raise ReceiptError("source path changed while opening held descriptor")
            observed_sha256 = _fd_sha256(fd, deadline_monotonic)
            fd_after = FileIdentity.from_stat(os.fstat(fd))
            path_after = _identity_from_path(resolved)
            mount_after = mount_probe(resolved)
            if not (
                _same_file_identity(before, fd_after)
                and _same_file_identity(before, path_after)
                and mount_before == mount_after
                and mount_after.readonly
            ):
                raise ReceiptError("source identity or read-only mount changed during full hash")
            if observed_sha256 != request.expected_sha256:
                raise ReceiptError("full-read digest differs from frozen expected SHA-256")
            aliases = sorted(set(logical_paths))
            for alias in aliases:
                if Path(alias).is_absolute() and Path(alias).resolve(strict=True) != resolved:
                    raise ReceiptError("logical alias does not resolve to the hashed file")
            entries.append(
                {
                    "kernel_id": kernel_id,
                    "resolved_path": str(resolved),
                    "dataset_slug": request.dataset.slug,
                    "dataset_version": request.dataset.version,
                    "dataset_mount_root": str(root),
                    "logical_paths": aliases,
                    "expected_sha256": request.expected_sha256,
                    "observed_sha256": observed_sha256,
                    "identity": asdict(before),
                    "mount": asdict(mount_before),
                    "source_fd": fd,
                }
            )
        payload = {
            "schema_version": 1,
            "kernel_id": kernel_id,
            "dataset_versions": dict(sorted(dataset_versions.items())),
            "entries": entries,
        }
        payload_bytes = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        root_manifest_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        payload["root_manifest_sha256"] = root_manifest_sha256
        manifest_fd = _sealed_json_memfd(payload)
        return OpenReceiptSet(
            kernel_id,
            root_manifest_sha256,
            dict(dataset_versions),
            entries,
            fds,
            manifest_fd,
        )
    except BaseException:
        for fd in fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        raise


def _sealed_json_memfd(payload: dict) -> int:
    """Create an immutable inherited receipt manifest; unsupported sealing fails closed."""
    if REQUIRED_SEALS == 0:
        raise ReceiptError("sealed memfd receipts are unavailable")
    fd = _memfd_create("wnn-readonly-hash-receipts")
    try:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, F_ADD_SEALS, REQUIRED_SEALS)
        if fcntl.fcntl(fd, F_GET_SEALS) & REQUIRED_SEALS != REQUIRED_SEALS:
            raise ReceiptError("receipt manifest memfd did not retain all required seals")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_sealed_receipts(
    fd: int,
    kernel_id: str,
    dataset_versions: dict[str, int],
    expected_root_manifest_sha256: str,
) -> list[dict]:
    """Read and validate an inherited immutable receipt manifest."""
    seals = fcntl.fcntl(fd, F_GET_SEALS)
    if seals & REQUIRED_SEALS != REQUIRED_SEALS:
        raise ReceiptError("receipt manifest is not fully sealed")
    size = os.fstat(fd).st_size
    if size <= 0 or size > 16 * 1024**2:
        raise ReceiptError("receipt manifest size is invalid")
    chunks = bytearray()
    while len(chunks) < size:
        block = os.pread(fd, min(CHUNK_BYTES, size - len(chunks)), len(chunks))
        if not block:
            raise ReceiptError("short sealed receipt manifest read")
        chunks.extend(block)
    payload = json.loads(chunks)
    recorded_root_sha256 = payload.pop("root_manifest_sha256", None)
    root_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    observed_root_sha256 = hashlib.sha256(root_payload).hexdigest()
    if (
        recorded_root_sha256 != observed_root_sha256
        or observed_root_sha256 != expected_root_manifest_sha256
    ):
        raise ReceiptError("receipt root manifest digest mismatch")
    if payload.get("schema_version") != 1 or payload.get("kernel_id") != kernel_id:
        raise ReceiptError("receipt manifest kernel/schema identity mismatch")
    if payload.get("dataset_versions") != dict(sorted(dataset_versions.items())):
        raise ReceiptError("receipt manifest dataset versions mismatch")
    if not isinstance(payload.get("entries"), list):
        raise ReceiptError("receipt entries are malformed")
    return payload["entries"]


def validate_receipt_entry(
    entry: dict,
    path: Path,
    expected_sha256: str,
    kernel_id: str,
    dataset_versions: dict[str, int],
    mount_probe: MountProbe = mount_identity,
    allowed_mount_roots: tuple[Path, ...] = (Path("/kaggle/input"),),
) -> tuple[bool, str]:
    """Check held FD, current path, expected digest, dataset version, and RO mount."""
    try:
        resolved = path.resolve(strict=True)
        identity = FileIdentity(**entry["identity"])
        fd = int(entry["source_fd"])
        if entry["resolved_path"] != str(resolved):
            return False, "resolved_path_changed"
        if (
            entry["expected_sha256"] != expected_sha256
            or entry["observed_sha256"] != expected_sha256
        ):
            return False, "expected_digest_mismatch"
        if entry["dataset_version"] != dataset_versions.get(entry["dataset_slug"]):
            return False, "dataset_version_mismatch"
        if entry.get("kernel_id", kernel_id) != kernel_id:
            return False, "kernel_id_mismatch"
        if not _fd_is_readonly(fd):
            return False, "source_fd_not_readonly"
        fd_identity = FileIdentity.from_stat(os.fstat(fd))
        path_identity = _identity_from_path(resolved)
        if not _same_file_identity(identity, fd_identity) or not _same_file_identity(
            identity, path_identity
        ):
            return False, "file_identity_changed"
        root = Path(entry["dataset_mount_root"]).resolve(strict=True)
        if not any(root.is_relative_to(allowed.resolve()) for allowed in allowed_mount_roots):
            return False, "dataset_mount_root_not_allowed"
        resolved.relative_to(root)
        mount = mount_probe(resolved)
        if not mount.readonly or asdict(mount) != entry["mount"]:
            return False, "mount_identity_or_readonly_changed"
        mount_point = Path(mount.mount_point).resolve(strict=True)
        if not resolved.is_relative_to(mount_point) or not root.is_relative_to(mount_point):
            return False, "dataset_root_outside_mount"
        fd_target = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve(strict=True)
        if fd_target != resolved:
            return False, "held_fd_target_changed"
        aliases = entry.get("logical_paths", [])
        if any(
            Path(alias).is_absolute() and Path(alias).resolve(strict=True) != resolved
            for alias in aliases
        ):
            return False, "logical_alias_changed"
    except (OSError, ValueError, KeyError, TypeError, ReceiptError):
        return False, "receipt_validation_error"
    return True, "verified_receipt_reuse"


class ReceiptOverlay:
    """Isolated hash adapter; invalid receipts trigger the strict full-hash path."""

    def __init__(
        self,
        entries: list[dict],
        kernel_id: str,
        dataset_versions: dict[str, int],
        mount_probe: MountProbe = mount_identity,
        allowed_mount_roots: tuple[Path, ...] = (Path("/kaggle/input"),),
        deadline_monotonic: float | None = None,
    ) -> None:
        self.entries = {entry["resolved_path"]: entry for entry in entries}
        self.kernel_id = kernel_id
        self.dataset_versions = dict(dataset_versions)
        self.mount_probe = mount_probe
        self.allowed_mount_roots = allowed_mount_roots
        self.deadline_monotonic = deadline_monotonic
        self.reused_count = 0
        self.fallback_count = 0
        self.fallback_reasons: list[str] = []

    def sha256_file(self, path: Path) -> str:
        """Return verified digest reuse or compute a fresh strict content hash."""
        candidate = Path(path)
        entry = self.entries.get(str(candidate.resolve(strict=True)))
        if entry is not None:
            valid, reason = validate_receipt_entry(
                entry,
                candidate,
                entry["expected_sha256"],
                self.kernel_id,
                self.dataset_versions,
                self.mount_probe,
                self.allowed_mount_roots,
            )
            if valid:
                self.reused_count += 1
                return entry["observed_sha256"]
            self.fallback_reasons.append(reason)
        else:
            self.fallback_reasons.append("no_exact_receipt")
        self.fallback_count += 1
        return strict_sha256_file(candidate, self.deadline_monotonic)


def install_hash_overlay(
    overlay: ReceiptOverlay,
    modules: Iterable[ModuleType],
    allowed_module_hashes: dict[str, str],
) -> Callable[[], None]:
    """Install wrappers only into hash-verified modules inside a diagnostic process."""
    originals: list[tuple[ModuleType, object]] = []
    for module in modules:
        module_path = Path(str(module.__file__)).resolve(strict=True)
        expected = allowed_module_hashes.get(str(module_path))
        if expected is None or strict_sha256_file(module_path) != expected:
            raise ReceiptError("refusing to patch an unpinned verifier module")
        original = getattr(module, "sha256_file", None)
        if not callable(original):
            raise ReceiptError("pinned module has no sha256_file function")
        originals.append((module, original))

    for module, _ in originals:
        module.__dict__["sha256_file"] = overlay.sha256_file

    def uninstall() -> None:
        for module, original in reversed(originals):
            module.__dict__["sha256_file"] = original

    return uninstall
