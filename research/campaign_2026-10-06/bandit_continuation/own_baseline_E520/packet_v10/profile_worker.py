"""Fail-closed, profile-only E520 Kaggle worker.

This module never has a fit or prediction-export mode. It performs identity and
fold preflight, invokes the pinned trainer's bounded early-return profile, and
measures one real-input inference chunk without persisting model outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import signal
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from gpu_memory_domains import (
    memory_domain_snapshot,
    select_nvml_device,
)
from verified_digest_handoff import DigestReceipt, VerifiedDigestHandoff

PACKET_ID = "E520_PROFILE_GPU_IDENTITY_LIVE_BINDING_V10"
REVIEWED_PACKET_FILES = (
    "profile_worker.py",
    "profile_kernel.py",
    "prepare_profile_packet.py",
    "gpu_memory_domains.py",
    "export_guards.py",
    "tests/conftest.py",
    "tests/test_profile_guards.py",
    "tests/test_portable_manifest_binding.py",
    "tests/test_profile_bundle.py",
    "tests/test_gpu_memory_domains.py",
    "tests/test_loader_hash_advice.py",
    "verified_digest_handoff.py",
    "tests/test_digest_handoff_integration.py",
    "E520_PACKET_PLAN.json",
    "source_archive.tar",
)
PROFILE_WARMUP = 3
PROFILE_STEPS = 20
PROFILE_WALL_CAP_S = 1_450
HASH_BLOCK_BYTES = 1 << 20
HASH_PROGRESS_RECEIPT_BYTES = 1 << 28
MEMORY_STAT_MAX_BYTES = 65_536
GPU_QUERY_RAW_MAX_BYTES = 65_536
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
PINNED_PORTABILITY_SOURCE_SHA256 = (
    "51ef5c1462bf6fa18b5235b4616e178251394022db9f88e4163282524f2bf81a"
)
EXPECTED = {
    "train": {
        "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "bytes": 29_240_219_011,
        "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
        "footer_bytes": 60_437_285,
    },
    "valid": {
        "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        "bytes": 5_615_509_128,
        "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
        "footer_bytes": 10_686_926,
    },
}
ASSETS_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
ASSETS_BYTES = 6_268
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
TARGET_HASHES = {
    "train": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    "valid": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
}
ASSET_HASHES = {
    "cache/chain_folds.npz": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    "cache/chains.npz": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
    "cache/train_targets.npz": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    "cache/valid_targets.npz": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    "research/campaign_2026-09-29/artifacts/splits/fold0.json": SPLIT_SHA256,
    "research/campaign_2026-10-04/portable_inputs.json": PORTABLE_SHA256,
}
TARGET_COUNTS = {"train": 10_606, "valid": 1_873}
FIT_COUNTS = {"train": 10_535, "valid": 1_405, "evaluation": 468}
CORRUPT_TRAIN_GROUP = 528
CAPS = {
    "rss_bytes": 8_000_000_000,
    "cuda_allocated_bytes": 2_000_000_000,
    "cuda_reserved_bytes": 2_500_000_000,
    "gpu_process_bytes": 3_000_000_000,
    "headroom_bytes": 4_294_967_296,
    "headroom_fraction": 0.20,
    "disk_free_bytes": 8_000_000_000,
    "visible_gpus": 1,
}


class ProfileAbort(RuntimeError):
    """Raised when a preflight, external-review, or live resource gate fails."""


def _record_hash_advice(receipt_path: Path, logical_name: str, record: dict[str, Any]) -> None:
    """Durably update one file's small page-cache-advice record."""
    payload: dict[str, Any] = {
        "phase": "raw_parquet_hash_pagecache_advice",
        "files": {},
    }
    if receipt_path.exists():
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileAbort(f"cannot read existing hash-advice receipt: {error}") from error
        if payload.get("phase") != "raw_parquet_hash_pagecache_advice" or not isinstance(
            payload.get("files"), dict
        ):
            raise ProfileAbort("existing hash-advice receipt has an unexpected schema")
    payload["files"][logical_name] = record.copy()
    atomic_json(receipt_path, payload)


def sha256_file(
    path: Path,
    *,
    pagecache_receipt_path: Path | None = None,
    logical_name: str | None = None,
) -> str:
    """Hash a file; raw parquet calls can require recorded POSIX cache advice."""
    if pagecache_receipt_path is not None:
        if logical_name is None:
            raise ProfileAbort("page-cache-safe hashing requires a stable logical name")
        return _sha256_file_with_pagecache_advice(path, pagecache_receipt_path, logical_name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file_with_pagecache_advice(path: Path, receipt_path: Path, logical_name: str) -> str:
    """Hash sequentially and advise the kernel to drop each completed page range."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        file_size = path.stat().st_size
    except (OSError, ValueError, TypeError, AttributeError) as error:
        record = {
            "status": "failed_preflight",
            "path": str(path),
            "posix_fadvise_available": callable(fadvise),
            "dontneed_constant_available": dontneed is not None,
            "diagnostic": f"{type(error).__name__}: {error}",
        }
        _record_hash_advice(receipt_path, logical_name, record)
        raise ProfileAbort(f"page-cache hash preflight failed for {logical_name}") from error
    record: dict[str, Any] = {
        "status": "starting",
        "path": str(path),
        "expected_file_bytes": file_size,
        "hash_block_bytes": HASH_BLOCK_BYTES,
        "page_size_bytes": page_size,
        "advice": "POSIX_FADV_DONTNEED",
        "posix_fadvise_available": callable(fadvise),
        "dontneed_constant_available": dontneed is not None,
        "bytes_hashed": 0,
        "bytes_advised": 0,
        "advice_calls": 0,
        "advice_failures": 0,
        "advice_semantics": "successful calls are kernel hints, not proof pages were evicted",
    }
    _record_hash_advice(receipt_path, logical_name, record)
    if (
        not callable(fadvise)
        or dontneed is None
        or page_size <= 0
        or HASH_BLOCK_BYTES % page_size != 0
    ):
        record.update(
            {
                "status": "unsupported",
                "diagnostic": "POSIX_FADV_DONTNEED or a compatible page size is unavailable",
            }
        )
        _record_hash_advice(receipt_path, logical_name, record)
        raise ProfileAbort("required POSIX page-cache advice is unavailable")

    digest = hashlib.sha256()
    bytes_hashed = 0
    bytes_advised = 0
    next_receipt_bytes = HASH_PROGRESS_RECEIPT_BYTES
    try:
        with path.open("rb", buffering=0) as stream:
            while block := stream.read(HASH_BLOCK_BYTES):
                digest.update(block)
                bytes_hashed += len(block)
                advised_end = bytes_hashed - (bytes_hashed % page_size)
                if advised_end > bytes_advised:
                    fadvise(
                        stream.fileno(),
                        bytes_advised,
                        advised_end - bytes_advised,
                        dontneed,
                    )
                    record["advice_calls"] += 1
                    bytes_advised = advised_end
                record["bytes_hashed"] = bytes_hashed
                record["bytes_advised"] = bytes_advised
                if bytes_hashed >= next_receipt_bytes:
                    record["status"] = "in_progress"
                    _record_hash_advice(receipt_path, logical_name, record)
                    next_receipt_bytes += HASH_PROGRESS_RECEIPT_BYTES
    except (OSError, NotImplementedError) as error:
        record.update(
            {
                "status": "failed",
                "bytes_hashed": bytes_hashed,
                "bytes_advised": bytes_advised,
                "advice_failures": int(record["advice_failures"]) + 1,
                "diagnostic": f"{type(error).__name__}: {error}",
            }
        )
        _record_hash_advice(receipt_path, logical_name, record)
        raise ProfileAbort(f"page-cache advice/hash failed for {logical_name}") from error
    if bytes_hashed != file_size:
        record.update(
            {
                "status": "file_size_changed",
                "bytes_hashed": bytes_hashed,
                "bytes_advised": bytes_advised,
            }
        )
        _record_hash_advice(receipt_path, logical_name, record)
        raise ProfileAbort(f"file size changed while hashing {logical_name}")
    record.update(
        {
            "status": "complete",
            "bytes_hashed": bytes_hashed,
            "bytes_advised": bytes_advised,
            "sha256": digest.hexdigest(),
        }
    )
    _record_hash_advice(receipt_path, logical_name, record)
    return str(record["sha256"])


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a small JSON receipt after writing its staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _record_memory_stat_phase(
    receipt_path: Path,
    phase: str,
    boundary: str,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist one bounded cgroup memory.stat snapshot for an I/O phase."""
    payload: dict[str, Any] = {"schema_version": 1, "phases": {}}
    if receipt_path.exists():
        try:
            loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileAbort(f"cannot read memory.stat receipt: {error}") from error
        if (
            not isinstance(loaded, dict)
            or loaded.get("schema_version") != 1
            or not isinstance(loaded.get("phases"), dict)
        ):
            raise ProfileAbort("memory.stat receipt has an unexpected schema")
        payload = loaded
    phase_record = payload["phases"].setdefault(phase, {})
    if boundary in phase_record:
        raise ProfileAbort(f"duplicate memory.stat phase boundary: {phase}/{boundary}")
    phase_record[boundary] = {
        "time_unix": time.time(),
        "snapshot": cgroup_memory_stat_snapshot(),
    }
    if details is not None:
        phase_record.setdefault("details", {}).update(details)
    atomic_json(receipt_path, payload)


def _hash_with_memory_stat(
    path: Path,
    *,
    hash_receipt_path: Path,
    memory_stat_receipt_path: Path,
    logical_name: str,
    diagnostic_phase: str,
) -> str:
    """Hash with required page-cache advice and durable before/after cgroup samples."""
    started = time.monotonic()
    _record_memory_stat_phase(memory_stat_receipt_path, diagnostic_phase, "before")
    status = "failed"
    result: str | None = None
    diagnostic: str | None = None
    try:
        result = sha256_file(
            path,
            pagecache_receipt_path=hash_receipt_path,
            logical_name=logical_name,
        )
        status = "complete"
        return result
    except Exception as error:
        causes: list[str] = []
        cause: BaseException | None = error
        while cause is not None:
            causes.append(f"{type(cause).__name__}: {cause}")
            cause = cause.__cause__
        diagnostic = " <- ".join(causes)
        raise
    finally:
        details: dict[str, Any] = {
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "bytes": path.stat().st_size if path.exists() else None,
        }
        if result is not None:
            details["sha256"] = result
        if diagnostic is not None:
            details["diagnostic"] = diagnostic
        _record_memory_stat_phase(
            memory_stat_receipt_path,
            diagnostic_phase,
            "after",
            details=details,
        )


def _hash_raw_with_handoff(
    path: Path,
    *,
    handoff: VerifiedDigestHandoff,
    hash_receipt_path: Path,
    memory_stat_receipt_path: Path,
    logical_name: str,
    diagnostic_phase: str,
) -> DigestReceipt:
    """Hash once through the worker-local handoff and preserve v6 I/O receipts."""
    started = time.monotonic()
    _record_memory_stat_phase(memory_stat_receipt_path, diagnostic_phase, "before")
    receipt: DigestReceipt | None = None
    record: dict[str, Any] | None = None
    status = "failed"
    diagnostic: str | None = None
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    try:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            file_size = path.stat().st_size
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise ProfileAbort(f"page-cache hash preflight failed for {logical_name}") from error
        record = {
            "status": "starting",
            "path": str(path),
            "expected_file_bytes": file_size,
            "hash_block_bytes": HASH_BLOCK_BYTES,
            "page_size_bytes": page_size,
            "advice": "POSIX_FADV_DONTNEED",
            "posix_fadvise_available": callable(fadvise),
            "dontneed_constant_available": dontneed is not None,
            "bytes_hashed": 0,
            "bytes_advised": 0,
            "advice_calls": 0,
            "advice_failures": 0,
            "advice_semantics": "successful calls are kernel hints, not proof pages were evicted",
        }
        _record_hash_advice(hash_receipt_path, logical_name, record)
        if (
            not callable(fadvise)
            or dontneed is None
            or page_size <= 0
            or HASH_BLOCK_BYTES % page_size != 0
        ):
            record.update(
                {
                    "status": "unsupported",
                    "diagnostic": "POSIX_FADV_DONTNEED or a compatible page size is unavailable",
                }
            )
            _record_hash_advice(hash_receipt_path, logical_name, record)
            raise ProfileAbort("required POSIX page-cache advice is unavailable")

        advised_bytes = 0
        next_receipt_bytes = HASH_PROGRESS_RECEIPT_BYTES

        def advise_completed_bytes(file_descriptor: int, bytes_hashed: int) -> None:
            nonlocal advised_bytes, next_receipt_bytes
            advised_end = bytes_hashed - (bytes_hashed % page_size)
            if advised_end > advised_bytes:
                try:
                    fadvise(
                        file_descriptor,
                        advised_bytes,
                        advised_end - advised_bytes,
                        dontneed,
                    )
                except (OSError, NotImplementedError) as error:
                    record.update(
                        {
                            "status": "failed",
                            "bytes_hashed": bytes_hashed,
                            "bytes_advised": advised_bytes,
                            "advice_failures": int(record["advice_failures"]) + 1,
                            "diagnostic": f"{type(error).__name__}: {error}",
                        }
                    )
                    _record_hash_advice(hash_receipt_path, logical_name, record)
                    raise
                record["advice_calls"] = int(record["advice_calls"]) + 1
                advised_bytes = advised_end
            record["bytes_hashed"] = bytes_hashed
            record["bytes_advised"] = advised_bytes
            if bytes_hashed >= next_receipt_bytes:
                record["status"] = "in_progress"
                _record_hash_advice(hash_receipt_path, logical_name, record)
                next_receipt_bytes += HASH_PROGRESS_RECEIPT_BYTES

        receipt = handoff.hash_file(path, on_progress=advise_completed_bytes)
        if receipt.bytes_hashed != file_size:
            raise ProfileAbort(f"file size changed while hashing {logical_name}")
        record.update(
            {
                "status": "complete",
                "bytes_hashed": receipt.bytes_hashed,
                "bytes_advised": advised_bytes,
                "sha256": receipt.sha256,
            }
        )
        _record_hash_advice(hash_receipt_path, logical_name, record)
        status = "complete"
        return receipt
    except Exception as error:
        if record is not None and record.get("status") != "unsupported":
            record.update(
                {
                    "status": "failed",
                    "diagnostic": f"{type(error).__name__}: {error}",
                }
            )
            _record_hash_advice(hash_receipt_path, logical_name, record)
        causes: list[str] = []
        cause: BaseException | None = error
        while cause is not None:
            causes.append(f"{type(cause).__name__}: {cause}")
            cause = cause.__cause__
        diagnostic = " <- ".join(causes)
        raise
    finally:
        details: dict[str, Any] = {
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "bytes": path.stat().st_size if path.exists() else None,
        }
        if receipt is not None:
            details["sha256"] = receipt.sha256
            details["bytes_hashed"] = receipt.bytes_hashed
        if diagnostic is not None:
            details["diagnostic"] = diagnostic
        _record_memory_stat_phase(
            memory_stat_receipt_path,
            diagnostic_phase,
            "after",
            details=details,
        )


def install_loader_portable_hash_advice(
    source_root: Path,
    hash_receipt_path: Path,
    memory_stat_receipt_path: Path,
    portability_module: Any | None = None,
) -> dict[str, Any]:
    """Advise the one pinned portable-registry hash pass over raw parquet inputs.

    Only ``wnn.portability._content_digest`` is wrapped. All registry, stat,
    size, digest comparison, and cache semantics remain in the pinned verifier.

    Args:
        source_root: Root containing the verified extracted source and raw symlinks.
        hash_receipt_path: Durable receipt of page-aligned POSIX_FADV_DONTNEED calls.
        memory_stat_receipt_path: Durable cgroup memory.stat phase observations.
        portability_module: Optional pinned module object for focused synthetic tests.

    Returns:
        Identity details for the exact patched pinned module and raw paths.

    Raises:
        ProfileAbort: If the imported module is not the pinned source or advice fails.
    """
    if portability_module is None:
        import wnn.portability as portability_module

    expected_module = source_root / "src" / "wnn" / "portability.py"
    try:
        observed_module = Path(portability_module.__file__).resolve(strict=True)
        expected_module = expected_module.resolve(strict=True)
    except (OSError, AttributeError) as error:
        raise ProfileAbort("pinned portability module path is unavailable") from error
    if observed_module != expected_module:
        raise ProfileAbort("portable verifier import did not resolve from the pinned source tree")
    module_sha = sha256_file(observed_module)
    if module_sha != PINNED_PORTABILITY_SOURCE_SHA256:
        raise ProfileAbort("pinned portability verifier source hash mismatch")
    original_digest = getattr(portability_module, "_content_digest", None)
    if not callable(original_digest):
        raise ProfileAbort("pinned portable verifier lacks _content_digest")

    raw_sources: dict[Path, str] = {}
    for split_name in ("train", "valid"):
        raw_path = source_root / "datasets" / f"{split_name}.parquet"
        if not raw_path.is_file():
            raise ProfileAbort(f"verified raw input is missing before loader patch: {split_name}")
        raw_sources[raw_path.resolve(strict=True)] = f"datasets/{split_name}.parquet"

    @lru_cache(maxsize=64)
    def content_digest(
        path: str,
        device: int,
        inode: int,
        size: int,
        modified_ns: int,
        changed_ns: int,
    ) -> str:
        """Preserve pinned cache keys and route only raw files through advice."""
        source = Path(path)
        try:
            resolved = source.resolve(strict=True)
            stat = source.stat()
        except OSError as error:
            raise ProfileAbort(f"portable source unavailable before hash: {path}") from error
        logical_name = raw_sources.get(resolved)
        if logical_name is None:
            return str(original_digest(path, device, inode, size, modified_ns, changed_ns))
        observed_identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        expected_identity = (device, inode, size, modified_ns, changed_ns)
        if observed_identity != expected_identity:
            raise ProfileAbort(
                f"portable source identity changed before advised hash: {logical_name}"
            )
        phase = f"loader_portable_content_digest:{logical_name}"
        return _hash_with_memory_stat(
            source,
            hash_receipt_path=hash_receipt_path,
            memory_stat_receipt_path=memory_stat_receipt_path,
            logical_name=f"loader_portable:{logical_name}",
            diagnostic_phase=phase,
        )

    portability_module._content_digest = content_digest
    return {
        "module_path": str(observed_module),
        "module_sha256": module_sha,
        "patched_symbol": "wnn.portability._content_digest",
        "raw_source_names": sorted(raw_sources.values()),
        "cache_maxsize": 64,
        "original_registry_verification_retained": True,
        "splits_file_sha256_unmodified": True,
    }


def install_loader_verified_digest_handoff(
    source_root: Path,
    handoff: VerifiedDigestHandoff,
    portability_module: Any | None = None,
) -> dict[str, Any]:
    """Reuse only exact raw digests after authenticating the pinned verifier path."""
    if portability_module is None:
        import wnn.portability as portability_module

    expected_module = source_root / "src" / "wnn" / "portability.py"
    try:
        observed_module = Path(portability_module.__file__).resolve(strict=True)
        expected_module = expected_module.resolve(strict=True)
    except (OSError, AttributeError) as error:
        raise ProfileAbort("pinned portability module path is unavailable") from error
    if observed_module != expected_module:
        raise ProfileAbort("portable verifier import did not resolve from the pinned source tree")
    module_sha = sha256_file(observed_module)
    if module_sha != PINNED_PORTABILITY_SOURCE_SHA256:
        raise ProfileAbort("pinned portability verifier source hash mismatch")
    try:
        handoff.install_portable_verifier(portability_module)
    except Exception as error:
        raise ProfileAbort("verified digest handoff could not bind the pinned verifier") from error
    return {
        "module_path": str(observed_module),
        "module_sha256": module_sha,
        "patched_symbol": "wnn.portability._content_digest",
        "raw_source_names": ["datasets/train.parquet", "datasets/valid.parquet"],
        "cache_maxsize": 0,
        "handoff_scope": "exact canonical path plus device/inode/size/mtime_ns/ctime_ns/mode",
        "fallback": "original pinned full content verifier on every miss",
        "registry_comparison_retained": True,
        "splits_file_sha256_unmodified": True,
    }


def require_profile_only(argv: list[str]) -> None:
    """Reject any command that could fit, save weights, or export predictions."""
    forbidden = {
        "--save-training-state",
        "--resume-state",
        "--parity-dump",
        "--checkpoint-path",
        "--prediction-output",
        "--export-predictions",
    }
    if any(item in forbidden for item in argv):
        raise ProfileAbort("profile-only command contains a fit/save/export option")
    pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    if pairs.get("--profile-steps") != str(PROFILE_STEPS):
        raise ProfileAbort("profile must request exactly 20 timed outer batches")
    if pairs.get("--profile-warmup") != str(PROFILE_WARMUP):
        raise ProfileAbort("profile must request exactly 3 warmup batches")


def validate_external_review(
    receipt_bytes: bytes, expected_sha256: str, actor_id: str
) -> dict[str, Any]:
    """Require a controller-supplied, externally hashed approval receipt."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ProfileAbort("independent review hash was not supplied by the controller")
    observed = hashlib.sha256(receipt_bytes).hexdigest()
    if observed != expected_sha256:
        raise ProfileAbort("external review receipt hash mismatch")
    receipt = json.loads(receipt_bytes)
    reviewer = receipt.get("reviewer", {})
    if (
        receipt.get("experiment_id") != PACKET_ID
        or receipt.get("scope") != "profile-only"
        or receipt.get("status") != "approved"
        or not reviewer.get("id")
        or reviewer.get("id") == actor_id
        or reviewer.get("role") != "independent-controller"
    ):
        raise ProfileAbort("external review receipt is missing independent profile-only approval")
    hashes = receipt.get("reviewed_hashes", {})
    if not set(REVIEWED_PACKET_FILES).issubset(hashes):
        raise ProfileAbort("external review does not bind all executable/source inputs")
    return receipt


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    """Return deterministic nearest-rank p50/p90 and count for finite samples."""
    cleaned = sorted(value for value in values if math.isfinite(value))
    if not cleaned:
        raise ProfileAbort("no finite timing samples were recorded")

    def nearest_rank(q: float) -> float:
        return cleaned[max(0, math.ceil(q * len(cleaned)) - 1)]

    return {"count": len(cleaned), "p50": nearest_rank(0.50), "p90": nearest_rank(0.90)}


def validate_export_shape(batch: int, rows: int, features: int, outputs: int) -> dict[str, int]:
    """Validate the fixed probe and account for input/output bytes without allocating them."""
    if (batch, rows, outputs) != (16, 4_000, 2) or features <= 0:
        raise ProfileAbort("exporter probe must use batch=16, chunk_rows=4000, outputs=2")
    return {
        "input_float32_bytes": batch * rows * features * 4,
        "output_float32_bytes_if_written": batch * rows * outputs * 4,
    }


def validate_export_tensors(
    raw_shape: tuple[int, ...],
    model_input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    *,
    raw_features: int,
    model_features: int,
    raw_dtype: str,
    model_input_dtype: str,
    output_dtype: str,
    outputs_finite: bool,
) -> None:
    """Fail unless the real export probe matches the exact shape/dtype/value contract."""
    if raw_shape != (16, 4_000, raw_features):
        raise ProfileAbort(f"raw exporter input shape mismatch: {raw_shape}")
    if model_input_shape != (16, 4_000, model_features):
        raise ProfileAbort(f"model exporter input shape mismatch: {model_input_shape}")
    if output_shape != (16, 4_000, 2):
        raise ProfileAbort(f"exporter output shape mismatch: {output_shape}")
    if raw_dtype != "torch.float32" or model_input_dtype != "torch.float32":
        raise ProfileAbort("exporter raw/model input dtype must remain float32")
    if output_dtype != "torch.float32":
        raise ProfileAbort("exporter two-target output dtype must be float32")
    if not outputs_finite:
        raise ProfileAbort("exporter output contains a non-finite value")


def _read_meminfo() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, tail = line.partition(":")
        field = tail.strip().split()
        if field and field[0].isdigit():
            values[key] = int(field[0]) * (1024 if len(field) > 1 and field[1] == "kB" else 1)
    return values.get("MemAvailable", 0), values.get("MemTotal", 0)


def _cgroup_headroom() -> tuple[int, int]:
    mount = Path("/sys/fs/cgroup")
    candidates = [mount]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, _, relative = line.partition(":")
            if hierarchy == "0" and relative:
                current = (mount / relative.lstrip("/")).resolve()
                while current == mount or mount in current.parents:
                    candidates.append(current)
                    if current == mount:
                        break
                    current = current.parent
                break
    except OSError:
        pass
    observations: list[tuple[int, int]] = []
    for root in dict.fromkeys(candidates):
        current_path, max_path = root / "memory.current", root / "memory.max"
        if current_path.exists() and max_path.exists():
            current = int(current_path.read_text().strip())
            raw_max = max_path.read_text().strip()
            if raw_max != "max":
                maximum = int(raw_max)
                observations.append((max(0, maximum - current), maximum))
    if observations:
        return min(observations, key=lambda item: item[0])
    return 0, 0


def _read_memory_stat_file(stat_path: Path) -> dict[str, Any]:
    """Read and parse at most the bounded prefix needed to reject oversized input."""
    try:
        with stat_path.open("rb") as stream:
            raw = stream.read(MEMORY_STAT_MAX_BYTES + 1)
    except OSError as error:
        return {"status": "unavailable", "diagnostic": f"{type(error).__name__}: {error}"}
    if len(raw) > MEMORY_STAT_MAX_BYTES:
        return {
            "status": "unavailable",
            "bytes_observed": len(raw),
            "diagnostic": f"memory.stat exceeds {MEMORY_STAT_MAX_BYTES} bytes",
        }
    try:
        values: dict[str, int] = {}
        for line in raw.decode("ascii").splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ValueError("malformed memory.stat line")
            value = int(fields[1])
            if value < 0:
                raise ValueError("negative memory.stat counter")
            values[fields[0]] = value
    except (UnicodeError, ValueError) as error:
        return {"status": "unavailable", "diagnostic": f"{type(error).__name__}: {error}"}
    return {"status": "available", "memory_stat": values}


def cgroup_memory_stat_snapshot() -> dict[str, Any]:
    """Read bounded memory.stat data from the most constrained visible cgroup."""
    mount = Path("/sys/fs/cgroup")
    candidates = [mount]
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, _, relative = line.partition(":")
            if hierarchy == "0" and relative:
                current = (mount / relative.lstrip("/")).resolve()
                while current == mount or mount in current.parents:
                    candidates.append(current)
                    if current == mount:
                        break
                    current = current.parent
                break
    except OSError as error:
        return {"status": "unavailable", "diagnostic": f"{type(error).__name__}: {error}"}

    observations: list[tuple[int, Path, int, int]] = []
    try:
        for root in dict.fromkeys(candidates):
            current_path, max_path = root / "memory.current", root / "memory.max"
            if not current_path.is_file() or not max_path.is_file():
                continue
            current = int(current_path.read_text(encoding="ascii").strip())
            max_text = max_path.read_text(encoding="ascii").strip()
            if max_text != "max":
                maximum = int(max_text)
                observations.append((max(0, maximum - current), root, current, maximum))
        if not observations:
            return {"status": "unavailable", "diagnostic": "no finite cgroup-v2 memory limit"}
        available, root, current, maximum = min(observations, key=lambda item: item[0])
        stat_path = root / "memory.stat"
        stat_observation = _read_memory_stat_file(stat_path)
        if stat_observation["status"] != "available":
            return {"cgroup_path": str(root), **stat_observation}
    except (OSError, UnicodeError, ValueError) as error:
        return {"status": "unavailable", "diagnostic": f"{type(error).__name__}: {error}"}
    return {
        "status": "available",
        "cgroup_path": str(root),
        "memory_current_bytes": current,
        "memory_limit_bytes": maximum,
        "memory_available_bytes": available,
        "memory_stat": stat_observation["memory_stat"],
    }


def _rss_bytes() -> tuple[int, int]:
    current = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1]) * 1024
            break
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    return current, peak


def resource_snapshot(
    torch_module: Any | None,
    working: Path,
    *,
    expected_physical_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Collect host and GPU usage without treating an empty process table as zero."""
    rss_current, rss_peak = _rss_bytes()
    host_available, host_total = _read_meminfo()
    cgroup_available, cgroup_limit = _cgroup_headroom()
    usage = shutil.disk_usage(working)
    result: dict[str, Any] = {
        "rss_current_bytes": rss_current,
        "rss_peak_bytes": rss_peak,
        "host_available_bytes": host_available,
        "host_total_bytes": host_total,
        "host_available_fraction": host_available / host_total if host_total else 0.0,
        "cgroup_available_bytes": cgroup_available,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_available_fraction": cgroup_available / cgroup_limit if cgroup_limit else 0.0,
        "disk_free_bytes": usage.free,
        "cuda_initialized": False,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_allocated_peak_bytes": None,
        "cuda_reserved_peak_bytes": None,
        "gpu_process_bytes": None,
        "gpu_process_query_status": "pre_cuda_idle",
        "gpu_process_query_raw": None,
        "gpu_process_query_raw_byte_length": None,
        "gpu_process_query_raw_sha256": None,
        "gpu_process_query_raw_truncated": None,
        "gpu_process_query_stderr": None,
        "gpu_process_query_stderr_byte_length": None,
        "gpu_process_query_stderr_sha256": None,
        "gpu_process_query_stderr_truncated": None,
        "gpu_process_query_return_code": None,
        "gpu_process_query_diagnostic": None,
        "gpu_total_bytes": None,
        "gpu_memory_info_total_bytes": None,
        "gpu_memory_totals_match": None,
        "gpu_cuda_usable_total_bytes": None,
        "gpu_cuda_free_bytes": None,
        "gpu_nvml_physical_total_bytes": None,
        "gpu_nvml_memory_used_bytes": None,
        "gpu_nvml_free_bytes": None,
        "gpu_selected_physical_gpu_uuid": None,
        "gpu_bound_physical_gpu_uuid": expected_physical_gpu_uuid,
        "gpu_physical_inventory_status": "pre_cuda_idle",
        "gpu_physical_inventory_raw_sha256": None,
        "gpu_device_used_upper_bound_bytes": None,
        "gpu_allocation_measurement_basis": "pre_cuda_idle",
        "gpu_resource_query_status": "pre_cuda_idle",
        "gpu_resource_query_diagnostic": None,
        "gpu_free_bytes": None,
    }
    if torch_module is None:
        return result
    if not torch_module.cuda.is_available():
        result.update(
            {
                "gpu_process_query_status": "cuda_unavailable",
                "gpu_allocation_measurement_basis": "unavailable",
                "gpu_resource_query_status": "query_error",
                "gpu_resource_query_diagnostic": "Torch reports CUDA unavailable",
            }
        )
        return result
    if torch_module is not None:
        result["cuda_initialized"] = True
        process_measurement = _nvidia_process_measurement(os.getpid())
        process_bytes = process_measurement["bytes"]
        result.update(
            {
                "gpu_process_bytes": process_bytes,
                "gpu_process_query_status": process_measurement["status"],
                "gpu_process_query_raw": process_measurement["raw"],
                "gpu_process_query_raw_byte_length": process_measurement["raw_byte_length"],
                "gpu_process_query_raw_sha256": process_measurement["raw_sha256"],
                "gpu_process_query_raw_truncated": process_measurement["raw_truncated"],
                "gpu_process_query_stderr": process_measurement["stderr"],
                "gpu_process_query_stderr_byte_length": process_measurement["stderr_byte_length"],
                "gpu_process_query_stderr_sha256": process_measurement["stderr_sha256"],
                "gpu_process_query_stderr_truncated": process_measurement["stderr_truncated"],
                "gpu_process_query_return_code": process_measurement["return_code"],
                "gpu_process_query_diagnostic": process_measurement.get("diagnostic"),
                "gpu_allocation_measurement_basis": "process_table_pid"
                if process_bytes is not None
                else "device_total_minus_free_upper_bound",
            }
        )
        try:
            properties = torch_module.cuda.get_device_properties(0)
            physical_inventory = nvidia_smi_gpu_sample(timeout_seconds=2)
            result["gpu_physical_inventory_status"] = physical_inventory.get("status")
            result["gpu_physical_inventory_raw_sha256"] = physical_inventory.get(
                "raw_output_sha256"
            )
            matching_rows = [
                row
                for row in physical_inventory.get("rows", [])
                if row.get("uuid") == expected_physical_gpu_uuid
            ]
            if not expected_physical_gpu_uuid or len(matching_rows) != 1:
                raise ValueError(
                    "worker-bound physical GPU UUID is absent or ambiguous in live inventory"
                )
            selected_row = matching_rows[0]
            free_bytes, memory_info_total_bytes = torch_module.cuda.mem_get_info(0)
            total_bytes = int(properties.total_memory)
            free_bytes = int(free_bytes)
            memory_info_total_bytes = int(memory_info_total_bytes)
            totals_match = total_bytes == memory_info_total_bytes
            domain = memory_domain_snapshot(
                cuda_property_total_bytes=total_bytes,
                cuda_memory_info_total_bytes=memory_info_total_bytes,
                cuda_free_bytes=free_bytes,
                nvml_physical_total_mib=(
                    selected_row.get("memory_total_mib") if selected_row else None
                ),
                nvml_memory_used_mib=(
                    selected_row.get("memory_used_mib") if selected_row else None
                ),
                observed_uuid=selected_row.get("uuid") if selected_row else None,
                expected_uuid=expected_physical_gpu_uuid,
            )
            result.update(
                {
                    "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
                    "cuda_reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
                    "cuda_allocated_peak_bytes": int(torch_module.cuda.max_memory_allocated(0)),
                    "cuda_reserved_peak_bytes": int(torch_module.cuda.max_memory_reserved(0)),
                    "gpu_total_bytes": total_bytes,
                    "gpu_memory_info_total_bytes": memory_info_total_bytes,
                    "gpu_memory_totals_match": totals_match,
                    "gpu_cuda_usable_total_bytes": domain["cuda_usable_total_bytes"],
                    "gpu_cuda_free_bytes": domain["cuda_free_bytes"],
                    "gpu_nvml_physical_total_bytes": domain["nvml_physical_total_bytes"],
                    "gpu_nvml_memory_used_bytes": domain["nvml_memory_used_bytes"],
                    "gpu_nvml_free_bytes": domain["nvml_free_bytes"],
                    "gpu_selected_physical_gpu_uuid": domain["selected_physical_gpu_uuid"],
                    "gpu_device_used_upper_bound_bytes": domain["device_used_upper_bound_bytes"],
                    "gpu_free_bytes": domain["conservative_free_bytes"],
                    "gpu_allocation_measurement_basis": "process_table_pid"
                    if process_bytes is not None
                    else "device_total_minus_free_upper_bound"
                    if totals_match
                    else "unavailable",
                    "gpu_resource_query_status": "ok",
                    "gpu_resource_query_diagnostic": None,
                }
            )
        except ValueError as error:
            result.update(
                {
                    "gpu_resource_query_status": "invalid_memory_domains",
                    "gpu_resource_query_diagnostic": f"{type(error).__name__}: {error}",
                }
            )
        except (AttributeError, RuntimeError, TypeError) as error:
            result.update(
                {
                    "gpu_resource_query_status": "query_error",
                    "gpu_allocation_measurement_basis": "unavailable",
                    "gpu_resource_query_diagnostic": f"{type(error).__name__}: {error}",
                }
            )
    return result


def _process_output_receipt(value: str | bytes | None) -> dict[str, Any]:
    """Retain bounded raw subprocess text, its exact digest, and truncation status."""
    raw = b"" if value is None else value if isinstance(value, bytes) else value.encode("utf-8")
    return {
        "text": raw[:GPU_QUERY_RAW_MAX_BYTES].decode("utf-8", errors="replace"),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > GPU_QUERY_RAW_MAX_BYTES,
    }


def parse_nvidia_process_bytes(pid: int, output: str) -> int | None:
    """Parse a process allocation; no matching row means unavailable, never zero."""
    import csv
    import io

    matches = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if (
            len(row) >= 2
            and row[0].strip().isdigit()
            and int(row[0]) == pid
            and row[1].strip().isdigit()
        ):
            matches.append(int(row[1]) * 1024 * 1024)
    return sum(matches) if matches else None


def _nvidia_process_measurement(pid: int) -> dict[str, Any]:
    """Read the process table while preserving bounded raw output on every outcome."""
    import subprocess

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as error:
        stdout = _process_output_receipt(getattr(error, "stdout", None))
        stderr = _process_output_receipt(getattr(error, "stderr", None))
        return {
            "bytes": None,
            "status": "query_error",
            "raw": stdout["text"],
            "raw_byte_length": stdout["byte_length"],
            "raw_sha256": stdout["sha256"],
            "raw_truncated": stdout["truncated"],
            "stderr": stderr["text"],
            "stderr_byte_length": stderr["byte_length"],
            "stderr_sha256": stderr["sha256"],
            "stderr_truncated": stderr["truncated"],
            "return_code": getattr(error, "returncode", None),
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    stdout = _process_output_receipt(completed.stdout)
    stderr = _process_output_receipt(completed.stderr)
    if completed.returncode != 0:
        return {
            "bytes": None,
            "status": "query_error",
            "raw": stdout["text"],
            "raw_byte_length": stdout["byte_length"],
            "raw_sha256": stdout["sha256"],
            "raw_truncated": stdout["truncated"],
            "stderr": stderr["text"],
            "stderr_byte_length": stderr["byte_length"],
            "stderr_sha256": stderr["sha256"],
            "stderr_truncated": stderr["truncated"],
            "return_code": completed.returncode,
            "diagnostic": f"nvidia-smi exited with return code {completed.returncode}",
        }
    used = parse_nvidia_process_bytes(pid, stdout["text"])
    return {
        "bytes": used,
        "status": "matched_pid" if used is not None else "pid_not_reported",
        "raw": stdout["text"],
        "raw_byte_length": stdout["byte_length"],
        "raw_sha256": stdout["sha256"],
        "raw_truncated": stdout["truncated"],
        "stderr": stderr["text"],
        "stderr_byte_length": stderr["byte_length"],
        "stderr_sha256": stderr["sha256"],
        "stderr_truncated": stderr["truncated"],
        "return_code": completed.returncode,
        "diagnostic": None,
    }


def runtime_identity(torch_module: Any) -> dict[str, Any]:
    """Capture actual selected-GPU, Torch, CUDA, cuDNN, driver, and Python identity."""
    properties = torch_module.cuda.get_device_properties(0)
    nvidia_smi = nvidia_smi_gpu_sample(timeout_seconds=5)
    try:
        cuda_free_bytes, cuda_memory_info_total_bytes = torch_module.cuda.mem_get_info(0)
        cuda_memory_info_status = "ok"
        cuda_memory_info_diagnostic = None
    except (AttributeError, RuntimeError, TypeError) as error:
        cuda_free_bytes = None
        cuda_memory_info_total_bytes = None
        cuda_memory_info_status = "query_error"
        cuda_memory_info_diagnostic = f"{type(error).__name__}: {error}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_row: dict[str, object] | None = None
    try:
        selected_row = select_nvml_device(nvidia_smi.get("rows", []), visible_devices)
        selected_binding_status = "bound_from_cuda_visibility_token"
        selected_binding_diagnostic = None
    except ValueError as error:
        selected_binding_status = "unavailable"
        selected_binding_diagnostic = f"{type(error).__name__}: {error}"
    return {
        "visible_gpu_count": int(torch_module.cuda.device_count()),
        "cuda_visible_devices": visible_devices,
        "selected_physical_gpu_binding_status": selected_binding_status,
        "selected_physical_gpu_binding_diagnostic": selected_binding_diagnostic,
        "selected_gpu_name": str(torch_module.cuda.get_device_name(0)),
        "selected_gpu_total_bytes": int(properties.total_memory),
        "selected_cuda_free_bytes": (int(cuda_free_bytes) if cuda_free_bytes is not None else None),
        "selected_cuda_memory_info_total_bytes": (
            int(cuda_memory_info_total_bytes) if cuda_memory_info_total_bytes is not None else None
        ),
        "cuda_memory_info_status": cuda_memory_info_status,
        "cuda_memory_info_diagnostic": cuda_memory_info_diagnostic,
        "selected_physical_gpu_uuid": selected_row.get("uuid") if selected_row else None,
        "selected_physical_gpu_index": selected_row.get("index") if selected_row else None,
        "selected_physical_gpu_memory_used_mib": (
            selected_row.get("memory_used_mib") if selected_row else None
        ),
        "selected_physical_gpu_memory_total_mib": (
            selected_row.get("memory_total_mib") if selected_row else None
        ),
        "selected_gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": str(torch_module.__version__),
        "cuda_runtime_version": torch_module.version.cuda,
        "cudnn_version": torch_module.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "physical_gpu_inventory_and_utilization": nvidia_smi,
    }


def require_physical_gpu_inventory(identity: dict[str, Any]) -> None:
    """Require the startup CUDA visibility token to bind to one physical GPU."""
    inventory = identity.get("physical_gpu_inventory_and_utilization", {})
    if inventory.get("status") != "ok":
        raise ProfileAbort("physical GPU inventory query is unavailable or malformed")
    rows = inventory.get("rows", [])
    if not rows or any(
        row.get("uuid") is None or row.get("memory_total_mib") is None for row in rows
    ):
        raise ProfileAbort("physical GPU UUID/memory inventory is incomplete")
    selected_uuid = identity.get("selected_physical_gpu_uuid")
    selected_index = identity.get("selected_physical_gpu_index")
    selected_total_mib = identity.get("selected_physical_gpu_memory_total_mib")
    selected_used_mib = identity.get("selected_physical_gpu_memory_used_mib")
    if (
        identity.get("selected_physical_gpu_binding_status") != "bound_from_cuda_visibility_token"
        or selected_uuid is None
        or selected_index is None
        or selected_total_mib is None
        or selected_used_mib is None
    ):
        raise ProfileAbort("selected CUDA device UUID/memory usage could not be confirmed")
    selected_rows = [row for row in rows if row.get("uuid") == selected_uuid]
    if len(selected_rows) != 1:
        raise ProfileAbort("selected CUDA device UUID is absent or ambiguous in NVML inventory")
    selected_row = selected_rows[0]
    if (
        selected_row.get("index") != selected_index
        or selected_row.get("memory_total_mib") != selected_total_mib
        or selected_row.get("memory_used_mib") != selected_used_mib
    ):
        raise ProfileAbort("selected CUDA/NVML physical identity changed during startup")
    selected_total_bytes = identity.get("selected_gpu_total_bytes")
    try:
        memory_domain_snapshot(
            cuda_property_total_bytes=selected_total_bytes,
            cuda_memory_info_total_bytes=identity.get("selected_cuda_memory_info_total_bytes"),
            cuda_free_bytes=identity.get("selected_cuda_free_bytes"),
            nvml_physical_total_mib=selected_total_mib,
            nvml_memory_used_mib=selected_used_mib,
            observed_uuid=selected_uuid,
            expected_uuid=selected_uuid,
        )
    except (TypeError, ValueError) as error:
        raise ProfileAbort(str(error)) from error


GPU_QUERY_FIELDS = (
    "index,name,driver_version,uuid,memory.used,memory.total,utilization.gpu,utilization.memory"
)


def _optional_gpu_integer(value: str) -> int | None:
    """Parse an observed integer metric while preserving N/A as unavailable."""
    normalized = value.strip()
    if not normalized or normalized.lower() in {"n/a", "na", "not supported"}:
        return None
    numeric = normalized.split(maxsplit=1)[0]
    try:
        return int(numeric)
    except ValueError:
        return None


def parse_nvidia_smi_gpu_rows(output: str) -> list[dict[str, Any]]:
    """Parse every physical GPU row, retaining unavailable metrics as null."""
    import csv
    import io

    rows: list[dict[str, Any]] = []
    for fields in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) != 8:
            raise ValueError(f"expected 8 nvidia-smi columns, got {len(fields)}")
        rows.append(
            {
                "index": _optional_gpu_integer(fields[0]),
                "name": fields[1].strip() or None,
                "driver_version": fields[2].strip() or None,
                "uuid": fields[3].strip() or None,
                "memory_used_mib": _optional_gpu_integer(fields[4]),
                "memory_total_mib": _optional_gpu_integer(fields[5]),
                "utilization_gpu_percent": _optional_gpu_integer(fields[6]),
                "utilization_memory_percent": _optional_gpu_integer(fields[7]),
            }
        )
    return rows


def nvidia_smi_gpu_sample(*, timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Collect bounded physical-GPU inventory/utilization; never invent zero values."""
    import subprocess

    empty_output = _process_output_receipt(None)
    sample: dict[str, Any] = {
        "sampled_at_unix": time.time(),
        "query_fields": GPU_QUERY_FIELDS,
        "status": "unavailable",
        "rows": [],
        "return_code": None,
        "raw_output": empty_output["text"],
        "raw_output_byte_length": empty_output["byte_length"],
        "raw_output_sha256": empty_output["sha256"],
        "raw_output_truncated": empty_output["truncated"],
        "raw_stderr": empty_output["text"],
        "raw_stderr_byte_length": empty_output["byte_length"],
        "raw_stderr_sha256": empty_output["sha256"],
        "raw_stderr_truncated": empty_output["truncated"],
    }
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raw_output = _process_output_receipt(getattr(error, "stdout", None))
        raw_stderr = _process_output_receipt(getattr(error, "stderr", None))
        sample.update(
            {
                "status": "query_error",
                "diagnostic": f"{type(error).__name__}: {error}",
                "raw_output": raw_output["text"],
                "raw_output_byte_length": raw_output["byte_length"],
                "raw_output_sha256": raw_output["sha256"],
                "raw_output_truncated": raw_output["truncated"],
                "raw_stderr": raw_stderr["text"],
                "raw_stderr_byte_length": raw_stderr["byte_length"],
                "raw_stderr_sha256": raw_stderr["sha256"],
                "raw_stderr_truncated": raw_stderr["truncated"],
                "return_code": getattr(error, "returncode", None),
            }
        )
        return sample
    sample["return_code"] = completed.returncode
    raw_output = _process_output_receipt(completed.stdout)
    raw_stderr = _process_output_receipt(completed.stderr)
    sample.update(
        {
            "raw_output": raw_output["text"],
            "raw_output_byte_length": raw_output["byte_length"],
            "raw_output_sha256": raw_output["sha256"],
            "raw_output_truncated": raw_output["truncated"],
            "raw_stderr": raw_stderr["text"],
            "raw_stderr_byte_length": raw_stderr["byte_length"],
            "raw_stderr_sha256": raw_stderr["sha256"],
            "raw_stderr_truncated": raw_stderr["truncated"],
        }
    )
    if completed.returncode != 0:
        sample.update(
            {
                "status": "query_error",
                "diagnostic": f"nvidia-smi exited with return code {completed.returncode}",
            }
        )
        return sample
    try:
        rows = parse_nvidia_smi_gpu_rows(completed.stdout)
    except ValueError as error:
        sample.update(
            {
                "status": "malformed_output",
                "diagnostic": str(error),
            }
        )
        return sample
    sample["status"] = "ok" if rows else "unavailable"
    sample["rows"] = rows
    if not rows:
        sample["diagnostic"] = "nvidia-smi returned no physical GPU rows"
    return sample


class GpuTelemetrySampler:
    """Collect capped, timestamped all-device GPU telemetry during trainer.profile."""

    def __init__(
        self,
        receipt_path: Path,
        *,
        interval_seconds: float = 5.0,
        max_samples: int = 24,
        sample_fn: Callable[[], dict[str, Any]] = nvidia_smi_gpu_sample,
    ) -> None:
        if interval_seconds <= 0 or max_samples < 2:
            raise ValueError("GPU telemetry interval and sample cap must be positive")
        self.receipt_path = receipt_path
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self.sample_fn = sample_fn
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="e520-gpu-telemetry",
            daemon=True,
        )
        self._started = False
        self._status = "not_started"

    def _record(self, sample_phase: str) -> None:
        if len(self.samples) >= self.max_samples:
            self._status = "sample_limit_reached"
            self._persist()
            return
        try:
            sample = self.sample_fn()
        except Exception as error:
            sample = {
                "status": "query_error",
                "diagnostic": f"{type(error).__name__}: {error}",
                "rows": [],
            }
        sample["sample_phase"] = sample_phase
        sample.setdefault("sampled_at_unix", time.time())
        self.samples.append(sample)
        self._status = "sampling"
        self._persist()

    def _persist(self) -> None:
        atomic_json(
            self.receipt_path,
            {
                "schema_version": 1,
                "phase": "trainer_profile_gpu_telemetry",
                "sampling_interval_seconds": self.interval_seconds,
                "sample_limit": self.max_samples,
                "status": self._status,
                "samples": self.samples,
            },
        )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            if len(self.samples) >= self.max_samples - 1:
                self._status = "sample_limit_reached"
                self._persist()
                return
            self._record("during_trainer_main")

    def start(self) -> None:
        """Record a bounded pre-profile sample, then poll in a background thread."""
        if self._started:
            raise RuntimeError("GPU telemetry sampler cannot be started twice")
        self._started = True
        self._record("before_trainer_main")
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and record one bounded post-profile sample."""
        if not self._started:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 2.5)
        self._record("after_trainer_main")
        if self._status != "sample_limit_reached":
            self._status = "complete"
        self._persist()


def enforce_resource_caps(snapshot: dict[str, Any], elapsed_s: float) -> None:
    """Raise immediately on a hard worker limit or missing resource observation."""
    if elapsed_s > PROFILE_WALL_CAP_S:
        raise ProfileAbort("profile worker exceeded its 1,450-second admitted wall cap")
    if snapshot["rss_peak_bytes"] > CAPS["rss_bytes"]:
        raise ProfileAbort("worker RSS high-water exceeded 8,000,000,000 bytes")
    for field, cap in (
        ("cuda_allocated_bytes", CAPS["cuda_allocated_bytes"]),
        ("cuda_reserved_bytes", CAPS["cuda_reserved_bytes"]),
        ("cuda_allocated_peak_bytes", CAPS["cuda_allocated_bytes"]),
        ("cuda_reserved_peak_bytes", CAPS["cuda_reserved_bytes"]),
        ("gpu_process_bytes", CAPS["gpu_process_bytes"]),
    ):
        value = snapshot[field]
        if value is not None and value > cap:
            raise ProfileAbort(f"{field} exceeded its E520 profile cap")
    for prefix in ("host", "cgroup"):
        available = snapshot[f"{prefix}_available_bytes"]
        fraction = snapshot[f"{prefix}_available_fraction"]
        failed_gates = []
        if available < CAPS["headroom_bytes"]:
            failed_gates.append(f"available_bytes<{CAPS['headroom_bytes']}")
        if fraction < CAPS["headroom_fraction"]:
            failed_gates.append(f"available_fraction<{CAPS['headroom_fraction']}")
        if failed_gates:
            triggers = " or ".join(failed_gates)
            raise ProfileAbort(
                f"{prefix} available memory fell below at least one E520 headroom gate: {triggers}"
            )
    if snapshot["disk_free_bytes"] < CAPS["disk_free_bytes"]:
        raise ProfileAbort("working disk fell below 8,000,000,000 free bytes")
    process_bytes = snapshot.get("gpu_process_bytes")
    process_status = snapshot.get("gpu_process_query_status")
    cuda_active = snapshot.get("cuda_allocated_bytes") is not None
    pre_cuda_idle = process_status == "pre_cuda_idle" and snapshot.get("cuda_initialized") is False
    if process_status == "pre_cuda_idle" and not pre_cuda_idle:
        raise ProfileAbort("pre-CUDA idle marker is invalid after CUDA initialization")
    if not pre_cuda_idle and snapshot.get("cuda_initialized") is not True:
        raise ProfileAbort("active CUDA initialization state is unavailable")
    if not pre_cuda_idle and snapshot.get("gpu_resource_query_status") != "ok":
        raise ProfileAbort("active CUDA device-memory measurement is unavailable or inconsistent")
    if not pre_cuda_idle:
        selected_uuid = snapshot.get("gpu_selected_physical_gpu_uuid")
        bound_uuid = snapshot.get("gpu_bound_physical_gpu_uuid")
        if not bound_uuid or selected_uuid != bound_uuid:
            raise ProfileAbort("selected physical GPU UUID changed or is unavailable")
    if process_bytes is None and not pre_cuda_idle:
        if snapshot.get("gpu_memory_totals_match") is not True:
            raise ProfileAbort("device-wide GPU memory total/free measurement is unavailable")
        device_used_upper_bound = snapshot.get("gpu_device_used_upper_bound_bytes")
        if not isinstance(device_used_upper_bound, int) or device_used_upper_bound < 0:
            raise ProfileAbort("device-wide GPU allocation upper bound is unavailable")
        if device_used_upper_bound > CAPS["gpu_process_bytes"]:
            raise ProfileAbort(
                "device-wide GPU usage upper bound exceeded the 3,000,000,000-byte cap"
            )
    if not pre_cuda_idle and any(
        snapshot.get(field) is None
        for field in (
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "cuda_allocated_peak_bytes",
            "cuda_reserved_peak_bytes",
        )
    ):
        raise ProfileAbort(
            "CUDA allocator current/peak APIs are unavailable during active profiling"
        )
    if cuda_active:
        gpu_free_bytes = snapshot.get("gpu_free_bytes")
        if not isinstance(gpu_free_bytes, int):
            raise ProfileAbort("selected GPU free-memory measurement is unavailable")
        if gpu_free_bytes < 6 * 1024**3:
            raise ProfileAbort("selected GPU free memory fell below 6 GiB")


def parquet_footer(path: Path) -> dict[str, Any]:
    """Stream-hash only a Parquet footer and trailer, then report size and digest."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes > size - 8:
            raise ProfileAbort(f"invalid Parquet trailer: {path.name}")
        count = footer_bytes + 8
        stream.seek(-count, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = count
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ProfileAbort(f"truncated Parquet footer: {path.name}")
            digest.update(block)
            remaining -= len(block)
    return {"bytes": size, "footer_bytes": footer_bytes, "footer_sha256": digest.hexdigest()}


def _unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise ProfileAbort(f"expected one mounted {name}, found {len(matches)}")
    return matches[0]


def persist_identity_check(
    receipt_path: Path,
    identity: dict[str, Any],
    logical_name: str,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    """Atomically persist both sides of an identity check before raising mismatch."""
    identity.setdefault("expected", {})[logical_name] = expected
    identity.setdefault("observed", {})[logical_name] = observed
    atomic_json(receipt_path, identity)
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise ProfileAbort(f"identity mismatch for {logical_name}: fields={','.join(mismatches)}")


def bind_verified_portable_manifest(
    source_root: Path, receipt_path: Path, identity: dict[str, Any]
) -> Path:
    """Bind the registry after its staged path and SHA-256 match.

    Args:
        source_root: Root containing the already identity-checked staged inputs.
        receipt_path: Durable input identity receipt to update before binding.
        identity: Mutable input identity receipt payload.

    Returns:
        The exact staged portable-manifest path used in the environment.

    Raises:
        ProfileAbort: If the staged registry is unreadable or its hash differs.
    """
    manifest_path = source_root / PORTABLE_RELATIVE
    expected = {"relative_path": PORTABLE_RELATIVE.as_posix(), "sha256": PORTABLE_SHA256}
    try:
        observed = {
            "relative_path": PORTABLE_RELATIVE.as_posix(),
            "staged_path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    except OSError as error:
        observed = {
            "relative_path": PORTABLE_RELATIVE.as_posix(),
            "staged_path": str(manifest_path),
            "status": "unreadable",
            "diagnostic": f"{type(error).__name__}: {error}",
        }
        persist_identity_check(
            receipt_path, identity, "portable_manifest_binding", expected, observed
        )
        raise ProfileAbort(
            "verified portable registry could not be read from its staged path"
        ) from error
    persist_identity_check(receipt_path, identity, "portable_manifest_binding", expected, observed)
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(manifest_path)
    identity["portable_manifest_env"] = os.environ["WNN_PORTABLE_DATA_MANIFEST"]
    atomic_json(receipt_path, identity)
    return manifest_path


def _index_digest(groups: Any, seq_ix: Any) -> str:
    digest = hashlib.sha256()
    digest.update(groups.astype("<i8", copy=False).tobytes())
    digest.update(seq_ix.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def verify_and_stage_inputs(
    input_root: Path, source_root: Path, receipt_path: Path
) -> dict[str, Any]:
    """Record expected/observed mount identities, stage symlinks, and validate fold membership."""
    os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    hash_receipt_path = receipt_path.with_name("hash_io_advice.json")
    memory_stat_receipt_path = receipt_path.with_name("memory_stat_phases.json")
    identity: dict[str, Any] = {
        "phase": "input_identity",
        "status": "in_progress",
        "expected": {
            "datasets/train.parquet": EXPECTED["train"],
            "datasets/valid.parquet": EXPECTED["valid"],
            "assets-manifest.json": {"sha256": ASSETS_SHA256, "bytes": ASSETS_BYTES},
            **{relative: {"sha256": digest} for relative, digest in ASSET_HASHES.items()},
        },
        "observed": {},
        "mount_version": None,
    }
    atomic_json(receipt_path, identity)
    try:
        assets_path = _unique(input_root, "assets-manifest.json")
    except ProfileAbort as error:
        persist_identity_check(
            receipt_path,
            identity,
            "assets-manifest.json",
            {"sha256": ASSETS_SHA256, "bytes": ASSETS_BYTES},
            {"status": "unavailable", "diagnostic": str(error)},
        )
        raise
    manifest_observed = {"bytes": assets_path.stat().st_size, "sha256": sha256_file(assets_path)}
    persist_identity_check(
        receipt_path,
        identity,
        "assets-manifest.json",
        {"sha256": ASSETS_SHA256, "bytes": ASSETS_BYTES},
        manifest_observed,
    )
    assets = json.loads(assets_path.read_text())
    if assets.get("dataset") != "vladkochetov07/wnn-c20261004-fold0-assets":
        persist_identity_check(
            receipt_path,
            identity,
            "assets-manifest.json.dataset",
            {"dataset": "vladkochetov07/wnn-c20261004-fold0-assets"},
            {"dataset": assets.get("dataset")},
        )
        raise ProfileAbort("fold assets manifest dataset identity mismatch")
    by_path = {entry["path"]: entry for entry in assets.get("files", [])}
    required = [
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        SPLIT_RELATIVE.as_posix(),
        PORTABLE_RELATIVE.as_posix(),
    ]
    staged: dict[str, Path] = {}
    for relative in required:
        entry = by_path.get(relative)
        if not isinstance(entry, dict):
            persist_identity_check(
                receipt_path,
                identity,
                relative,
                {"sha256": ASSET_HASHES[relative]},
                {"status": "unavailable", "diagnostic": "asset missing from verified manifest"},
            )
        if "filename" not in entry or "size" not in entry:
            persist_identity_check(
                receipt_path,
                identity,
                relative,
                {"sha256": ASSET_HASHES[relative]},
                {"status": "malformed", "manifest_entry": entry},
            )
        expected_asset = {
            "sha256": ASSET_HASHES[relative],
            "bytes": int(entry["size"]),
        }
        try:
            path = _unique(input_root, str(entry["filename"]))
        except ProfileAbort as error:
            persist_identity_check(
                receipt_path,
                identity,
                relative,
                expected_asset,
                {"status": "unavailable", "diagnostic": str(error)},
            )
            raise
        observed_asset = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        persist_identity_check(receipt_path, identity, relative, expected_asset, observed_asset)
        staged[relative] = path
    portable = json.loads(staged[PORTABLE_RELATIVE.as_posix()].read_text())
    raw: dict[str, Path] = {}
    for split_name in ("train", "valid"):
        logical_name = f"datasets/{split_name}.parquet"
        try:
            raw[split_name] = _unique(input_root, f"{split_name}.parquet")
        except ProfileAbort as error:
            persist_identity_check(
                receipt_path,
                identity,
                logical_name,
                EXPECTED[split_name],
                {"status": "unavailable", "diagnostic": str(error)},
            )
            raise
    handoff = VerifiedDigestHandoff(tuple(raw.values()))
    for split_name, path in raw.items():
        logical_name = f"datasets/{split_name}.parquet"
        expected = EXPECTED[split_name]
        candidate = _hash_raw_with_handoff(
            path,
            handoff=handoff,
            hash_receipt_path=hash_receipt_path,
            memory_stat_receipt_path=memory_stat_receipt_path,
            logical_name=logical_name,
            diagnostic_phase=f"initial_raw_hash:{logical_name}",
        )
        observed: dict[str, Any] = {
            "sha256": candidate.sha256,
            "bytes": candidate.bytes_hashed,
        }
        identity.setdefault("observed", {})[logical_name] = observed
        atomic_json(receipt_path, identity)
        try:
            observed.update(parquet_footer(path))
        except ProfileAbort as error:
            observed["footer_status"] = "invalid"
            observed["footer_diagnostic"] = str(error)
            atomic_json(receipt_path, identity)
            raise
        persist_identity_check(receipt_path, identity, logical_name, expected, observed)
        registry_entry = portable.get("sources", {}).get(logical_name)
        registry_digest = registry_entry.get("sha256") if isinstance(registry_entry, dict) else None
        persist_identity_check(
            receipt_path,
            identity,
            f"{logical_name}.portable_registry_sha256",
            {"sha256": candidate.sha256},
            {"sha256": registry_digest},
        )
        try:
            handoff.authorize(
                candidate,
                expected_sha256=expected["sha256"],
                registry_sha256=str(registry_digest),
            )
        except Exception as error:
            raise ProfileAbort(f"raw digest could not be authorized for {split_name}") from error
    for logical, path in raw.items():
        destination = source_root / "datasets" / f"{logical}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ProfileAbort(f"refusing to replace staged input {destination}")
        destination.symlink_to(path)
    for relative, path in staged.items():
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ProfileAbort(f"refusing to replace staged asset {destination}")
        destination.symlink_to(path)
    bind_verified_portable_manifest(source_root, receipt_path, identity)
    _record_memory_stat_phase(memory_stat_receipt_path, "load_campaign_split", "before")
    sys.path.insert(0, str(source_root / "src"))
    import wnn.portability as portability

    loader_handoff = install_loader_verified_digest_handoff(
        source_root,
        handoff,
        portability_module=portability,
    )
    identity["loader_verified_digest_handoff"] = loader_handoff
    atomic_json(receipt_path, identity)
    try:
        from wnn.splits import load_campaign_split

        split = load_campaign_split(source_root / SPLIT_RELATIVE, source_root)
    finally:
        raw_lookup_metrics = {
            split_name: handoff.metrics_for(path) for split_name, path in raw.items()
        }
        identity["portable_verifier_raw_digest_metrics"] = raw_lookup_metrics
        atomic_json(receipt_path, identity)
        _record_memory_stat_phase(
            memory_stat_receipt_path,
            "load_campaign_split",
            "after",
            details={
                "portable_verification_hashes_advised": False,
                "raw_digest_handoff_metrics": raw_lookup_metrics,
            },
        )
    train, valid, evaluation = (
        set(map(int, split.train_groups)),
        set(map(int, split.valid_groups)),
        set(map(int, split.evaluation_groups)),
    )
    if (len(train), len(valid), len(evaluation)) != (
        FIT_COUNTS["train"],
        FIT_COUNTS["valid"],
        FIT_COUNTS["evaluation"],
    ):
        raise ProfileAbort("fold-0 exact fit/evaluation membership counts changed")
    if CORRUPT_TRAIN_GROUP in train:
        raise ProfileAbort("corrupt train row group 528 entered the fit membership")
    if split.sha256 != SPLIT_SHA256:
        raise ProfileAbort("loaded split fingerprint differs from E508-v5")
    orders: dict[str, list[int]] = {}
    import numpy as np

    for split_name in ("train", "valid"):
        with np.load(staged[f"cache/{split_name}_targets.npz"], allow_pickle=False) as cache:
            groups = np.asarray(cache["row_group"], dtype=np.int64)
            seq_ix = np.asarray(cache["seq_ix"], dtype=np.int64)
        if (
            groups.shape != (TARGET_COUNTS[split_name],)
            or len(set(map(int, groups))) != TARGET_COUNTS[split_name]
        ):
            raise ProfileAbort(f"{split_name} target row-group order dimensions changed")
        expected_index_hash = portable["sources"][f"cache/{split_name}_targets.npz"][
            "original_identity"
        ]["index_sha256"]
        observed_index_hash = _index_digest(groups, seq_ix)
        persist_identity_check(
            receipt_path,
            identity,
            f"cache/{split_name}_targets.npz.index",
            {"index_sha256": expected_index_hash},
            {"index_sha256": observed_index_hash},
        )
        target_hash = sha256_file(staged[f"cache/{split_name}_targets.npz"])
        persist_identity_check(
            receipt_path,
            identity,
            f"cache/{split_name}_targets.npz.v5_hash",
            {"sha256": TARGET_HASHES[split_name]},
            {"sha256": target_hash},
        )
        eligible = train if split_name == "train" else valid
        if not eligible.issubset(set(map(int, groups))):
            raise ProfileAbort(f"{split_name} target order omits an eligible fit group")
        orders[split_name] = list(map(int, groups[:16]))
    if CORRUPT_TRAIN_GROUP in set(
        map(int, np.load(staged["cache/train_targets.npz"], allow_pickle=False)["row_group"])
    ):
        raise ProfileAbort("corrupt train row group entered target-cache order")
    identity.update(
        {
            "status": "matched",
            "split_sha256": split.sha256,
            "fit_membership_counts": {"train": len(train), "valid": len(valid)},
            "evaluation_group_count": len(evaluation),
            "target_counts": TARGET_COUNTS,
            "target_index_hashes": {
                name: json.loads(staged[PORTABLE_RELATIVE.as_posix()].read_text())["sources"][
                    f"cache/{name}_targets.npz"
                ]["original_identity"]["index_sha256"]
                for name in orders
            },
            "dataset_version_from_runtime_mount": None,
        }
    )
    atomic_json(receipt_path, identity)
    return {"identity": identity, "orders": orders, "split": split}


class LiveGuard:
    """Poll hard resource limits while the trainer or chunk probe is active."""

    def __init__(
        self,
        torch_module: Any | None,
        working: Path,
        receipts: Path,
        start: float,
        selected_physical_gpu_uuid: str | None = None,
    ) -> None:
        self.torch = torch_module
        self.selected_physical_gpu_uuid = selected_physical_gpu_uuid
        self.working = working
        self.receipts = receipts
        self.start = start
        self.stop = threading.Event()
        self.failure: str | None = None
        self.latest: dict[str, Any] = {}
        self._check_lock = threading.Lock()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self._started = False

    def check(self, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._check_lock:
            snap = (
                snapshot
                if snapshot is not None
                else resource_snapshot(
                    self.torch,
                    self.working,
                    expected_physical_gpu_uuid=self.selected_physical_gpu_uuid,
                )
            )
            self.latest = snap
            phase = (
                "cuda_initialized_pre_profile_resource_gate"
                if snapshot is not None
                else "profile_active"
            )
            receipt = {
                "phase": phase,
                "time_unix": time.time(),
                "snapshot": snap,
            }
            atomic_json(self.receipts / "resource.json", receipt)
            try:
                enforce_resource_caps(snap, time.monotonic() - self.start)
            except ProfileAbort as error:
                self.failure = str(error)
                atomic_json(
                    self.receipts / "resource_abort.json",
                    {
                        "status": "aborted",
                        "phase": phase,
                        "reason": self.failure,
                        "snapshot": snap,
                        "time_unix": time.time(),
                    },
                )
                raise
            return snap

    def _poll(self) -> None:
        while not self.stop.wait(1.0):
            try:
                self.check()
            except (
                Exception
            ) as error:  # the main thread cannot safely continue after a gate failure
                self.failure = str(error)
                atomic_json(
                    self.receipts / "resource_abort.json",
                    {
                        "status": "aborted",
                        "reason": self.failure,
                        "snapshot": self.latest,
                        "time_unix": time.time(),
                    },
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def start_polling(self) -> None:
        if self._started:
            raise RuntimeError("resource guard polling cannot be started twice")
        self._started = True
        self.thread.start()

    def stop_polling(self) -> None:
        self.stop.set()
        if self._started:
            self.thread.join(timeout=3)

    def raise_if_failed(self) -> None:
        if self.failure:
            raise ProfileAbort(self.failure)


def run_profile(packet_root: Path) -> dict[str, Any]:
    """Run mounted-data preflight, frozen trainer profile, and real-input 4,000-row probe."""
    work = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    receipts = work / "e520_profile_v10"
    receipts.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    phase: dict[str, Any] = {
        "packet": PACKET_ID,
        "status": "active",
        "phase": "review_gate",
        "worker_wall_cap_seconds": PROFILE_WALL_CAP_S,
        "start_unix": time.time(),
    }
    phase_path = receipts / "phase.json"
    atomic_json(phase_path, phase)
    guard: LiveGuard | None = None
    gpu_sampler: GpuTelemetrySampler | None = None
    identity_path = receipts / "input_identity.json"

    def handle_termination(signum: int, frame: Any) -> None:
        phase.update(
            {
                "phase": "failed",
                "status": "aborted",
                "reason": "SIGTERM resource/runtime gate",
                "signal": signum,
                "updated_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
            }
        )
        atomic_json(phase_path, phase)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_termination)
    try:
        if not work.is_dir():
            raise ProfileAbort(f"working directory is missing: {work}")
        guard = LiveGuard(None, work, receipts, start)
        guard.start_polling()
        meta = json.loads((packet_root / "packet_meta.json").read_text())
        receipt_bytes = (packet_root / "external_review.json").read_bytes()
        receipt = validate_external_review(
            receipt_bytes, meta.get("external_review_sha256", ""), "weak_diversity_kaggle_actor"
        )
        if receipt.get("reviewed_hashes") != meta.get("reviewed_hashes"):
            raise ProfileAbort("controller receipt and packet review hash map differ")
        for name in REVIEWED_PACKET_FILES[:-1]:
            actual_hash = sha256_file(packet_root / name)
            if actual_hash != meta["reviewed_hashes"].get(name):
                raise ProfileAbort(f"reviewed code hash mismatch: {name}")
        source_tar = packet_root / "source_archive.tar"
        if sha256_file(source_tar) != meta.get("source_archive_sha256"):
            raise ProfileAbort("pinned source archive hash mismatch")
        if sha256_file(source_tar) != meta["reviewed_hashes"].get("source_archive.tar"):
            raise ProfileAbort("source archive is not the reviewed archive")
        source_root = work / "e520_source_v10"
        source_root.mkdir(parents=True, exist_ok=False)
        with tarfile.open(source_tar, "r:") as archive:
            members = archive.getmembers()
            for member in members:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or member.issym()
                    or member.islnk()
                ):
                    raise ProfileAbort(f"unsafe path in pinned source archive: {member.name}")
            archive.extractall(source_root, members=members)
        trainer = source_root / "scripts/train_rnn.py"
        if sha256_file(trainer) != meta.get("trainer_sha256"):
            raise ProfileAbort("pinned trainer source hash mismatch")
        phase.update({"phase": "mount_identity_and_fold", "updated_unix": time.time()})
        atomic_json(phase_path, phase)
        mounted = verify_and_stage_inputs(
            Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")), source_root, identity_path
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WNN_LOADER_WORKERS"] = "0"
        os.environ["WNN_THREADS"] = "2"
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        guard.check()
        guard.stop_polling()
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != CAPS["visible_gpus"]:
            raise ProfileAbort("profile requires exactly one visible CUDA GPU")
        device_identity = runtime_identity(torch)
        selected_physical_gpu_uuid = device_identity.get("selected_physical_gpu_uuid")
        startup_resources = resource_snapshot(
            torch, work, expected_physical_gpu_uuid=selected_physical_gpu_uuid
        )
        atomic_json(
            receipts / "environment.json",
            {
                "phase": "cuda_initialized_pre_profile_resource_identity",
                "identity": device_identity,
                "startup_resource_snapshot": startup_resources,
                "gpu_name_is_observed_not_hardcoded": True,
                "time_unix": time.time(),
            },
        )
        atomic_json(
            receipts / "resource.json",
            {
                "phase": "cuda_initialized_pre_profile_resource_identity",
                "time_unix": time.time(),
                "snapshot": startup_resources,
            },
        )
        guard = LiveGuard(
            torch, work, receipts, start, selected_physical_gpu_uuid=selected_physical_gpu_uuid
        )
        guard.check(startup_resources)
        require_physical_gpu_inventory(device_identity)
        guard.start_polling()
        torch.cuda.reset_peak_memory_stats(0)
        profile_capture: dict[str, Any] = {}
        sys.path.insert(0, str(source_root))
        import importlib.util

        spec = importlib.util.spec_from_file_location("e520_train_rnn", trainer)
        if spec is None or spec.loader is None:
            raise ProfileAbort("could not load pinned train_rnn.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_make_model = module.make_model

        def capture_model(args: dict[str, Any], n_in: int, n_out: int) -> Any:
            model = original_make_model(args, n_in, n_out)
            profile_capture.update(
                {"model": model, "n_in": int(n_in), "n_out": int(n_out), "args": args.copy()}
            )
            return model

        module.make_model = capture_model
        original_stop = module.StageTimer.stop

        def capture_stop(timer: Any) -> None:
            if timer.cur is not None:
                timer.cur["_e520_stop_wall"] = time.perf_counter()
            original_stop(timer)
            guard.check()
            guard.raise_if_failed()

        module.StageTimer.stop = capture_stop
        original_report = module.StageTimer.report

        def report(timer: Any, warmup: int, batch: int) -> dict[str, Any]:
            summary = original_report(timer, warmup, batch)
            training_resources = resource_snapshot(
                torch, work, expected_physical_gpu_uuid=selected_physical_gpu_uuid
            )
            guard.latest = training_resources
            training_peaks = {
                "allocated_peak_bytes": training_resources["cuda_allocated_peak_bytes"],
                "reserved_peak_bytes": training_resources["cuda_reserved_peak_bytes"],
                "process_gpu_bytes_at_capture": training_resources["gpu_process_bytes"],
                "process_gpu_query_status": training_resources["gpu_process_query_status"],
                "process_gpu_query_raw": training_resources["gpu_process_query_raw"],
                "process_gpu_query_stderr": training_resources["gpu_process_query_stderr"],
                "process_gpu_query_return_code": training_resources[
                    "gpu_process_query_return_code"
                ],
                "gpu_total_bytes": training_resources["gpu_total_bytes"],
                "gpu_free_bytes": training_resources["gpu_free_bytes"],
                "gpu_device_used_upper_bound_bytes": training_resources[
                    "gpu_device_used_upper_bound_bytes"
                ],
                "gpu_allocation_measurement_basis": training_resources[
                    "gpu_allocation_measurement_basis"
                ],
            }
            profile_capture["training_cuda_peaks"] = training_peaks
            atomic_json(
                receipts / "training_profile_peak.json",
                {
                    "phase": "training_profile_peak_captured_before_trainer_early_return",
                    "snapshot": training_resources,
                    "peaks": training_peaks,
                    "time_unix": time.time(),
                },
            )
            enforce_resource_caps(training_resources, time.monotonic() - start)
            guard.raise_if_failed()
            timed = timer.steps[warmup:]
            if len(timed) != PROFILE_STEPS:
                raise ProfileAbort(
                    f"trainer recorded {len(timed)} timed outer batches, expected {PROFILE_STEPS}"
                )
            intervals_ms = [
                (timed[i + 1]["_wall"] - timed[i]["_wall"]) * 1000 for i in range(len(timed) - 1)
            ]
            step_ms = [(item["_e520_stop_wall"] - item["_wall"]) * 1000 for item in timed]
            loader_ms = [value * 1000 for value in timer.loader_wait[warmup:]]
            profile_capture["timing"] = {
                "timed_outer_batches": len(timed),
                "end_to_end_inter_start_ms": percentile_summary(intervals_ms),
                "host_batch_span_ms": percentile_summary(step_ms),
                "loader_wait_ms": percentile_summary(loader_ms),
                "cuda_stage_median_ms": summary,
                "timing_note": "p50/p90 of inter-start throughput intervals (19 observations from 20 timed starts); separate host spans and loader waits are not added.",
            }
            print(
                "E520_PROFILE_TIMINGS " + json.dumps(profile_capture["timing"], sort_keys=True),
                flush=True,
            )
            return summary

        module.StageTimer.report = report
        argv = [
            "--name",
            "e520_profile_s3_f0",
            "--device",
            "cuda",
            "--arch",
            "gru",
            "--cell",
            "gru",
            "--hidden",
            "128",
            "--layers",
            "2",
            "--dropout",
            "0",
            "--seed",
            "3",
            "--raw-only",
            "--loss",
            "wmse",
            "--loss-norm",
            "chunk",
            "--train-source",
            "train+valid",
            "--train-seqs",
            "10607",
            "--split-manifest",
            SPLIT_RELATIVE.as_posix(),
            "--valid-fold",
            "0",
            "--epochs",
            "4",
            "--checkpoint-policy",
            "fixed",
            "--checkpoint-epoch",
            "3",
            "--no-eval-during-fit",
            "--batch",
            "16",
            "--chunk",
            "2000",
            "--lr",
            "0.001",
            "--wd",
            "0.01",
            "--optim",
            "adamw",
            "--sched",
            "onecycle",
            "--precision",
            "fp16",
            "--init",
            "",
            "--distill",
            "",
            "--distill-boot-h",
            "0",
            "--propensity",
            "",
            "--prop-input",
            "",
            "--prop-power",
            "1",
            "--prop-eps",
            "0",
            "--valid-true-mask",
            "-1",
            "--ema-teacher",
            "0",
            "--aux-struct",
            "0",
            "--aux-next",
            "0",
            "--aux-tier",
            "0",
            "--aux-mask",
            "0",
            "--aux-sign",
            "0",
            "--aux-quote",
            "0",
            "--aux",
            "",
            "--aux-weight",
            "0",
            "--swa-frac",
            "0",
            "--fast",
            "--profile-warmup",
            "3",
            "--profile-steps",
            "20",
        ]
        require_profile_only(argv)
        phase.update(
            {
                "phase": "trainer_profile",
                "updated_unix": time.time(),
                "profile_warmup": 3,
                "profile_steps": 20,
            }
        )
        atomic_json(phase_path, phase)
        sys.argv = [str(trainer), *argv]
        gpu_sampler = GpuTelemetrySampler(receipts / "gpu_telemetry.json")
        gpu_sampler.start()
        try:
            module.main()
        finally:
            gpu_sampler.stop()
            module.StageTimer.stop = original_stop
            module.StageTimer.report = original_report
        guard.raise_if_failed()
        run_dir = source_root / "artifacts/runs/e520_profile_s3_f0"
        forbidden = [
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and (
                path.name.endswith(".pt")
                or path.name.endswith(".npy")
                or "pred" in path.name.lower()
            )
        ]
        if forbidden:
            raise ProfileAbort(
                f"profile trainer wrote forbidden checkpoint/prediction artifact: {forbidden[0].name}"
            )
        phase.update({"phase": "export_shape_probe", "updated_unix": time.time()})
        atomic_json(phase_path, phase)
        torch.cuda.empty_cache()
        model = profile_capture.get("model")
        if model is None:
            raise ProfileAbort("trainer profile did not instantiate its registered model")
        model.eval()
        model.zero_grad(set_to_none=True)
        from wnn.data import SequenceFile

        export_metrics = []
        for split_name in ("train", "valid"):
            file_path = source_root / "datasets" / f"{split_name}.parquet"
            reader = SequenceFile(file_path)
            chunks = []
            import numpy as np

            for row_group in mounted["orders"][split_name]:
                seq = reader.read(row_group, targets=False, mask=False)
                feature_chunk = np.asarray(seq.features[:4_000], dtype=np.float32)
                if feature_chunk.shape != (4_000, module.N_FEATURES):
                    raise ProfileAbort(
                        "raw input feature width/shape differs from pinned data schema"
                    )
                chunks.append(torch.from_numpy(feature_chunk.copy()))
                del seq, feature_chunk
            cpu_batch = torch.stack(chunks, dim=0)
            del chunks
            shape_bytes = validate_export_shape(16, 4_000, profile_capture["n_in"], 2)
            torch.cuda.reset_peak_memory_stats(0)
            before = resource_snapshot(
                torch, work, expected_physical_gpu_uuid=selected_physical_gpu_uuid
            )
            raw_batch = cpu_batch.to("cuda", non_blocking=False)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                model_inputs = module.build_inputs(raw_batch)
                outputs, _ = module.forward_chunk(model, model_inputs, slice(0, 4_000), None)
                outputs = (
                    module.readout(outputs, *module.READOUT, tilted=module.READOUT_TILTED)
                    if module.READOUT is not None
                    else outputs[..., :2].float()
                )
            output_finite = bool(torch.isfinite(outputs).all().item())
            validate_export_tensors(
                tuple(raw_batch.shape),
                tuple(model_inputs.shape),
                tuple(outputs.shape),
                raw_features=module.N_FEATURES,
                model_features=profile_capture["n_in"],
                raw_dtype=str(raw_batch.dtype),
                model_input_dtype=str(model_inputs.dtype),
                output_dtype=str(outputs.dtype),
                outputs_finite=output_finite,
            )
            torch.cuda.synchronize(0)
            after = resource_snapshot(
                torch, work, expected_physical_gpu_uuid=selected_physical_gpu_uuid
            )
            guard.check()
            export_metrics.append(
                {
                    "split": split_name,
                    "raw_input_shape": list(raw_batch.shape),
                    "model_input_shape": list(model_inputs.shape),
                    "input_dtype": str(model_inputs.dtype),
                    "output_shape": list(outputs.shape),
                    "output_dtype": str(outputs.dtype),
                    "output_finite": output_finite,
                    **shape_bytes,
                    "predictions_written": False,
                    "torch_allocated_peak_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "torch_reserved_peak_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "rss_peak_bytes": after["rss_peak_bytes"],
                    "rss_before_bytes": before["rss_current_bytes"],
                    "rss_after_bytes": after["rss_current_bytes"],
                    "gpu_free_after_bytes": after["gpu_free_bytes"],
                }
            )
            del outputs, model_inputs, raw_batch, cpu_batch, reader
            torch.cuda.empty_cache()
        guard.raise_if_failed()
        final_resource = guard.check()
        guard.stop_polling()
        phase.update(
            {
                "phase": "complete",
                "status": "profile_only_complete",
                "updated_unix": time.time(),
                "end_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
            }
        )
        result = {
            "packet": PACKET_ID,
            "status": "profile_only_complete",
            "fit_started": False,
            "checkpoint_written": False,
            "predictions_written": False,
            "source_archive_sha256": meta["source_archive_sha256"],
            "trainer_sha256": meta["trainer_sha256"],
            "external_review_sha256": meta["external_review_sha256"],
            "input_identity": mounted["identity"],
            "hash_io_advice": json.loads(
                (receipts / "hash_io_advice.json").read_text(encoding="utf-8")
            ),
            "timing": profile_capture.get("timing"),
            "training_cuda_peaks": profile_capture.get("training_cuda_peaks"),
            "environment_identity": device_identity,
            "gpu_telemetry_receipt": str(receipts / "gpu_telemetry.json"),
            "export_probe": export_metrics,
            "resources_final": final_resource,
            "caps": CAPS,
            "worker_wall_cap_seconds": PROFILE_WALL_CAP_S,
            "wall_seconds": time.monotonic() - start,
        }
        atomic_json(receipts / "profile_result.json", result)
        atomic_json(phase_path, phase)
        return result
    except BaseException as error:
        if guard is not None:
            guard.stop_polling()
        if identity_path.exists():
            try:
                identity = json.loads(identity_path.read_text())
                if identity.get("status") == "in_progress":
                    identity.update(
                        {"status": "failed", "failure": f"{type(error).__name__}: {error}"}
                    )
                    atomic_json(identity_path, identity)
            except (OSError, json.JSONDecodeError):
                pass
        phase.update(
            {
                "phase": "failed",
                "status": "aborted",
                "reason": f"{type(error).__name__}: {error}",
                "updated_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
            }
        )
        atomic_json(phase_path, phase)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise ProfileAbort(str(error)) from error


if __name__ == "__main__":  # pragma: no cover - runtime-only entry point
    raise SystemExit("Use profile_kernel.py; it validates the external review gate first.")
