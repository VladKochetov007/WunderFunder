"""Measure logical bytes and wall time for duplicate hashing versus a handoff."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path
from types import ModuleType

from verified_digest_handoff import VerifiedDigestHandoff

SAMPLE_BYTES = 8 * 1024 * 1024
BLOCK_BYTES = 1 << 20


def _write_sample(path: Path) -> None:
    """Create a deterministic temporary file without a wide in-memory buffer."""
    block = bytes(range(256)) * (BLOCK_BYTES // 256)
    with path.open("wb") as stream:
        remaining = SAMPLE_BYTES
        while remaining:
            piece = block[: min(remaining, len(block))]
            stream.write(piece)
            remaining -= len(piece)


def _stream_hash(path: Path) -> tuple[str, int, float]:
    """Hash one file pass and report user-space bytes consumed and elapsed time."""
    started = time.perf_counter()
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
            bytes_read += len(block)
    return digest.hexdigest(), bytes_read, time.perf_counter() - started


def _portable_arguments(path: Path) -> tuple[str, int, int, int, int, int]:
    """Build the pinned verifier's exact content-digest argument tuple."""
    current = path.stat()
    return (
        str(path.resolve()),
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )


def run_measurement() -> dict[str, object]:
    """Compare the measured one-file verification paths on a bounded sample."""
    with tempfile.TemporaryDirectory(prefix="e522-io-") as temporary:
        path = Path(temporary) / "train.parquet"
        _write_sample(path)

        baseline_start = time.perf_counter()
        first_digest, first_bytes, _ = _stream_hash(path)
        second_digest, second_bytes, _ = _stream_hash(path)
        baseline_seconds = time.perf_counter() - baseline_start
        if first_digest != second_digest:
            raise RuntimeError("baseline repeated digest changed unexpectedly")

        handoff = VerifiedDigestHandoff((path,))
        optimized_start = time.perf_counter()
        receipt = handoff.hash_file(path)
        handoff.authorize(
            receipt,
            expected_sha256=first_digest,
            registry_sha256=second_digest,
        )
        fallback_bytes = 0
        fallback_calls = 0
        module = ModuleType("synthetic_portability")

        def original_digest(
            source: str,
            _device: int,
            _inode: int,
            _size: int,
            _modified_ns: int,
            _changed_ns: int,
        ) -> str:
            nonlocal fallback_bytes, fallback_calls
            fallback_calls += 1
            digest = hashlib.sha256()
            with Path(source).open("rb", buffering=0) as stream:
                while block := stream.read(BLOCK_BYTES):
                    fallback_bytes += len(block)
                    digest.update(block)
            return digest.hexdigest()

        module._content_digest = original_digest
        handoff.install_portable_verifier(module)
        lookup_digests = [module._content_digest(*_portable_arguments(path)) for _ in range(2)]
        optimized_seconds = time.perf_counter() - optimized_start
        if lookup_digests != [first_digest, first_digest]:
            raise RuntimeError("portable verifier did not receive the authorized digest")

        return {
            "schema_version": 1,
            "measurement": "small-file logical hash-read count and elapsed wall time",
            "sample_bytes": SAMPLE_BYTES,
            "chunk_bytes": BLOCK_BYTES,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "baseline_two_pass": {
                "hash_passes": 2,
                "bytes_read": first_bytes + second_bytes,
                "elapsed_seconds": baseline_seconds,
            },
            "authorized_handoff": {
                "hash_passes": 1,
                "bytes_read": receipt.bytes_hashed + fallback_bytes,
                "fallback_digest_calls": fallback_calls,
                "verifier_lookups": len(lookup_digests),
                "elapsed_seconds": optimized_seconds,
            },
            "logical_bytes_eliminated": first_bytes
            + second_bytes
            - receipt.bytes_hashed
            - fallback_bytes,
            "timing_limit": "synthetic small local file; timing is not a parquet/Kaggle throughput forecast",
            "identity": {
                "device": receipt.identity.device,
                "inode": receipt.identity.inode,
                "size": receipt.identity.size,
                "modified_ns": receipt.identity.modified_ns,
                "changed_ns": receipt.identity.changed_ns,
                "sha256": receipt.sha256,
            },
        }


if __name__ == "__main__":
    print(json.dumps(run_measurement(), indent=2, sort_keys=True))
