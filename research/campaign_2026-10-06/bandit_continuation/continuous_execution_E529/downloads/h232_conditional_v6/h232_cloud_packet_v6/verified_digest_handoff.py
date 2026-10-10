"""Reuse an in-process SHA-256 only for the exact file identity that produced it."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

DigestProgress = Callable[[int, int], None]
PortableDigest = Callable[[str, int, int, int, int, int], str]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK_BYTES = 1 << 20


class DigestHandoffError(RuntimeError):
    """Base error for invalid or unstable digest handoffs."""


class SourceChangedError(DigestHandoffError):
    """Raised when the file identity changes during hashing or authorization."""


class DigestMismatchError(DigestHandoffError):
    """Raised when observed bytes disagree with the frozen source digests."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Filesystem identity fields used to scope a single-process digest."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        """Build a stable identity snapshot from an open descriptor or path stat."""
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
            mode=value.st_mode,
        )

    def portable_key_matches(
        self,
        device: int,
        inode: int,
        size: int,
        modified_ns: int,
        changed_ns: int,
    ) -> bool:
        """Match the five stat fields supplied by the pinned portability verifier."""
        return (
            self.device,
            self.inode,
            self.size,
            self.modified_ns,
            self.changed_ns,
        ) == (device, inode, size, modified_ns, changed_ns)


@dataclass(frozen=True, slots=True)
class DigestReceipt:
    """Unapproved candidate digest produced by one complete stable file read."""

    canonical_path: str
    identity: FileIdentity
    sha256: str
    bytes_hashed: int
    reused_from_prior: bool = False


class VerifiedDigestHandoff:
    """Hold verified raw-source digests for exact allowed paths and stat tuples.

    The handoff is intentionally memory-only and worker-local. A digest is
    reusable only after the bytes match both the campaign's expected SHA-256
    and the portable registry SHA-256, and only while the full filesystem
    identity remains unchanged. A miss falls through to the pinned verifier's
    original content hash.
    """

    def __init__(self, allowed_paths: tuple[Path, ...]) -> None:
        """Create a handoff whose reads and lookups are restricted to exact files."""
        if not allowed_paths:
            raise ValueError("at least one exact source path is required")
        canonical_paths = tuple(self._resolve(path) for path in allowed_paths)
        if len(set(canonical_paths)) != len(canonical_paths):
            raise ValueError("allowed source paths must be unique after resolution")
        self._allowed_paths = frozenset(canonical_paths)
        self._pending: dict[int, DigestReceipt] = {}
        self._authorized: dict[str, DigestReceipt] = {}
        self._metrics = {
            "verifier_lookups": 0,
            "digest_reuse_hits": 0,
            "digest_reuse_misses": 0,
            "fallback_digest_calls": 0,
        }
        self._path_metrics: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def hash_file(self, path: Path, on_progress: DigestProgress | None = None) -> DigestReceipt:
        """Hash exact allowed bytes once, rejecting any stat change during the read.

        Args:
            path: Exact raw-source path previously declared in ``allowed_paths``.
            on_progress: Optional callback given file descriptor and cumulative
                bytes hashed after each block. A page-cache adviser may use it.

        Returns:
            An unapproved receipt bound to the file descriptor and path identity.

        Raises:
            DigestHandoffError: If the path is outside scope, unreadable, or
                changes while being hashed.
        """
        resolved = self._resolve(path)
        if resolved not in self._allowed_paths:
            raise DigestHandoffError(f"source path is outside handoff scope: {resolved}")
        with self._lock:
            self._authorized.pop(str(resolved), None)

        try:
            before_path = FileIdentity.from_stat(resolved.stat())
            if not stat.S_ISREG(before_path.mode):
                raise DigestHandoffError(f"source is not a regular file: {resolved}")
            digest = hashlib.sha256()
            bytes_hashed = 0
            with resolved.open("rb", buffering=0) as stream:
                before_fd = FileIdentity.from_stat(os.fstat(stream.fileno()))
                if before_fd != before_path:
                    raise SourceChangedError(f"source changed before hashing: {resolved}")
                while block := stream.read(_CHUNK_BYTES):
                    digest.update(block)
                    bytes_hashed += len(block)
                    if on_progress is not None:
                        on_progress(stream.fileno(), bytes_hashed)
                after_fd = FileIdentity.from_stat(os.fstat(stream.fileno()))
            after_path = FileIdentity.from_stat(resolved.stat())
        except OSError as error:
            raise DigestHandoffError(f"cannot hash source file {resolved}: {error}") from error

        if before_path != before_fd or before_fd != after_fd or after_fd != after_path:
            raise SourceChangedError(f"source changed while hashing: {resolved}")
        if bytes_hashed != before_path.size:
            raise SourceChangedError(
                f"source byte count changed while hashing: {resolved} "
                f"({bytes_hashed} != {before_path.size})"
            )
        receipt = DigestReceipt(
            canonical_path=str(resolved),
            identity=after_path,
            sha256=digest.hexdigest(),
            bytes_hashed=bytes_hashed,
        )
        with self._lock:
            self._pending[id(receipt)] = receipt
        return receipt

    def authorize(
        self,
        receipt: DigestReceipt,
        *,
        expected_sha256: str,
        registry_sha256: str,
    ) -> None:
        """Authorize a receipt only after both frozen digests match its bytes."""
        with self._lock:
            issued = self._pending.pop(id(receipt), None)
        if issued is not receipt:
            raise DigestHandoffError("digest receipt was not issued by this handoff")
        if not _SHA256_PATTERN.fullmatch(expected_sha256) or not _SHA256_PATTERN.fullmatch(
            registry_sha256
        ):
            raise DigestMismatchError("expected and registry digests must be lowercase SHA-256")
        if expected_sha256 != registry_sha256 or receipt.sha256 != expected_sha256:
            raise DigestMismatchError("observed source digest disagrees with frozen identity")
        path = Path(receipt.canonical_path)
        try:
            current = FileIdentity.from_stat(path.stat())
        except OSError as error:
            raise SourceChangedError(
                f"source became unavailable before digest authorization: {path}"
            ) from error
        if current != receipt.identity:
            raise SourceChangedError(f"source changed before digest authorization: {path}")
        with self._lock:
            self._authorized[receipt.canonical_path] = receipt

    def candidate_from_prior_identity(
        self,
        path: Path,
        *,
        prior_identity: FileIdentity,
        prior_sha256: str,
    ) -> DigestReceipt | None:
        """Prepare a prior digest only when every current stat field is identical."""
        resolved = self._resolve(path)
        if resolved not in self._allowed_paths:
            raise DigestHandoffError(f"source path is outside handoff scope: {resolved}")
        try:
            current = FileIdentity.from_stat(resolved.stat())
        except OSError as error:
            raise SourceChangedError(f"cannot stat prior-digest source: {resolved}") from error
        if current != prior_identity:
            return None
        if not _SHA256_PATTERN.fullmatch(prior_sha256):
            raise DigestMismatchError("prior digest must be lowercase SHA-256")
        receipt = DigestReceipt(
            canonical_path=str(resolved),
            identity=current,
            sha256=prior_sha256,
            bytes_hashed=0,
            reused_from_prior=True,
        )
        with self._lock:
            self._pending[id(receipt)] = receipt
        return receipt

    def lookup(
        self,
        path: str,
        *,
        device: int,
        inode: int,
        size: int,
        modified_ns: int,
        changed_ns: int,
    ) -> str | None:
        """Return an authorized digest only for a matching current stat identity."""
        try:
            resolved = self._resolve(Path(path))
            current = FileIdentity.from_stat(resolved.stat())
        except (DigestHandoffError, OSError):
            self._record_lookup(hit=False, canonical_path=path)
            return None
        if resolved not in self._allowed_paths:
            self._record_lookup(hit=False, canonical_path=str(resolved))
            return None
        key = str(resolved)
        with self._lock:
            receipt = self._authorized.get(key)
            if receipt is None:
                self._record_lookup(hit=False, canonical_path=key)
                return None
            if receipt.identity != current or not current.portable_key_matches(
                device, inode, size, modified_ns, changed_ns
            ):
                self._authorized.pop(key, None)
                self._record_lookup(hit=False, canonical_path=key)
                return None
            self._record_lookup(hit=True, canonical_path=key)
            return receipt.sha256

    def metrics(self) -> dict[str, int]:
        """Return a copy of worker-local lookup and pinned-verifier fallback counts."""
        with self._lock:
            return self._metrics.copy()

    def metrics_for(self, path: Path) -> dict[str, int]:
        """Return handoff lookup and fallback counts for one canonical file path."""
        try:
            key = str(self._resolve(path))
        except DigestHandoffError:
            key = str(path)
        with self._lock:
            return self._path_metrics.get(key, self._empty_metrics()).copy()

    def _record_lookup(self, *, hit: bool, canonical_path: str) -> None:
        with self._lock:
            self._metrics["verifier_lookups"] += 1
            path_metrics = self._path_metrics.setdefault(canonical_path, self._empty_metrics())
            path_metrics["verifier_lookups"] += 1
            event = "digest_reuse_hits" if hit else "digest_reuse_misses"
            self._metrics[event] += 1
            path_metrics[event] += 1

    def _record_fallback(self, path: str) -> None:
        try:
            key = str(self._resolve(Path(path)))
        except DigestHandoffError:
            key = path
        with self._lock:
            self._metrics["fallback_digest_calls"] += 1
            self._path_metrics.setdefault(key, self._empty_metrics())["fallback_digest_calls"] += 1

    @staticmethod
    def _empty_metrics() -> dict[str, int]:
        return {
            "verifier_lookups": 0,
            "digest_reuse_hits": 0,
            "digest_reuse_misses": 0,
            "fallback_digest_calls": 0,
        }

    def install_portable_verifier(self, module: ModuleType) -> None:
        """Wrap only ``_content_digest``; all non-hit calls use the pinned original.

        Install this before the first ``load_campaign_split`` call. The pinned
        verifier continues to check original metadata and compare the returned
        digest to the portable registry. ``wnn.splits.file_sha256`` is untouched.
        """
        original_value = getattr(module, "_content_digest", None)
        if not callable(original_value):
            raise DigestHandoffError("portable module has no callable _content_digest")
        # The private verifier is dynamically supplied by the frozen source module.
        original_digest = cast(PortableDigest, original_value)

        def content_digest(
            path: str,
            device: int,
            inode: int,
            size: int,
            modified_ns: int,
            changed_ns: int,
        ) -> str:
            reused = self.lookup(
                path,
                device=device,
                inode=inode,
                size=size,
                modified_ns=modified_ns,
                changed_ns=changed_ns,
            )
            if reused is not None:
                return reused
            self._record_fallback(path)
            return original_digest(path, device, inode, size, modified_ns, changed_ns)

        module._content_digest = content_digest

    @staticmethod
    def _resolve(path: Path) -> Path:
        try:
            return path.resolve(strict=True)
        except OSError as error:
            raise DigestHandoffError(f"source path cannot be resolved: {path}") from error
