"""Fail-closed H232 profile-gated fit/export worker.

The worker verifies mounted inputs once, profiles the frozen H232 recipe, and
continues to fit/export in the same process only when the strict measured
profile, resource, identity-handoff, and lifetime gates all pass.
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
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from cache_pressure_recovery import (
    MAX_ADVICE_BYTES,
    MAX_RELIEF_EVENTS,
    MAX_TOTAL_ADVICE_BYTES,
    MIN_RELIEF_INTERVAL_SECONDS,
    ReliefBudget,
    advise_recent_rowgroups,
    has_reclaimable_file_pressure,
    host_headroom_passes,
    memory_snapshot_matches,
    relief_target_bytes,
)
from gpu_memory_domains import (
    memory_domain_snapshot,
    select_nvml_device,
)
from verified_digest_handoff import DigestReceipt, FileIdentity, VerifiedDigestHandoff

PACKET_ID = "E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V6"
REVIEWED_PACKET_FILES = (
    "profile_worker.py",
    "profile_kernel.py",
    "gpu_memory_domains.py",
    "verified_digest_handoff.py",
    "prior_input_identity_v2.json",
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "source_archive_manifest.json",
    "cache_pressure_recovery.py",
    "source_archive.tar",
)
PROFILE_WARMUP = 3
PROFILE_STEPS = 20
WORKER_WALL_CAP_S = 34_000
PREDICTION_EXPORT_CAP_S = 1_800
MAX_PROJECTED_FIT_SECONDS = 28_000
FIT_PROJECTION_BATCHES_PER_EPOCH = (10_535 + 1_405) // 8
FIT_PROJECTION_EPOCHS = 4
FIT_PROJECTION_SAFETY_FACTOR = 1.20
FIT_SETUP_OVERHEAD_SECONDS = 600
EXPECTED_RAW_FEATURES = 112
EXPECTED_EVENT_FEATURES = 99
EXPECTED_MODEL_INPUT_WIDTH = EXPECTED_RAW_FEATURES + EXPECTED_EVENT_FEATURES
EXPECTED_LOADER_BATCH_BYTES = 8 * 20_000 * ((EXPECTED_RAW_FEATURES + 2) * 4 + 1)
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
PRIOR_INPUT_IDENTITY_SHA256 = "cbea42868303968f7ee84105990a25848ce17add175b7161e116c80dfa63e46e"
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
    "rss_bytes": 25_000_000_000,
    "cuda_allocated_bytes": 11_500_000_000,
    "cuda_reserved_bytes": 11_800_000_000,
    "gpu_process_bytes": 12_000_000_000,
    "headroom_bytes": 6_500_000_000,
    "headroom_fraction": 0.25,
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


def staged_split_manifest_argument(source_root: Path) -> str:
    """Return the verified staged fold path as an absolute trainer argument."""
    staged_path = source_root / SPLIT_RELATIVE
    if not staged_path.is_absolute() or not staged_path.is_file():
        raise ProfileAbort(f"staged fold-0 manifest is missing: {staged_path}")
    return str(staged_path)


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
    if pairs.get("--amp-init-scale") != "4096.0":
        raise ProfileAbort("V6 profile must use the declared 4096.0 GradScaler initial scale")
    if "--profile-trace-output" not in pairs:
        raise ProfileAbort("profile must persist the per-update trace")


def _metric_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: int(after.get(name, 0)) - int(before.get(name, 0)) for name in after}


def _path_metric_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Return per-raw-file digest handoff counts observed during one profile window."""
    names = (
        "verifier_lookups",
        "digest_reuse_hits",
        "digest_reuse_misses",
        "fallback_digest_calls",
    )
    return {
        split_name: {
            name: int(after.get(split_name, {}).get(name, 0))
            - int(before.get(split_name, {}).get(name, 0))
            for name in names
        }
        for split_name in ("train", "valid")
    }


def _probe_source_input_width(module: Any) -> dict[str, Any]:
    """Measure raw plus event-feature width from the pinned source implementation."""
    import numpy as np

    from wnn.features import N_EVENT_FEATURES, event_features

    raw_count = int(module.N_FEATURES)
    raw = np.zeros((2, 3, raw_count), dtype=np.float32)
    event = event_features(raw)
    if event.shape[:2] != (2, 3) or event.dtype != np.float32:
        raise ProfileAbort("pinned event_features smoke probe returned an invalid shape or dtype")
    event_count = int(event.shape[-1])
    if event_count != int(N_EVENT_FEATURES):
        raise ProfileAbort(
            "pinned event_features smoke width differs from its declared feature count"
        )
    return {
        "probe_input_shape": list(raw.shape),
        "probe_event_output_shape": list(event.shape),
        "raw_feature_count": raw_count,
        "event_feature_count": event_count,
        "declared_event_feature_count": int(N_EVENT_FEATURES),
        "derived_model_input_width": raw_count + event_count,
        "event_features_source": "pinned source archive: src/wnn/features.py:event_features",
    }


def derive_fit_argv(profile_argv: list[str], run_name: str) -> list[str]:
    """Remove profile-only flags and change only the operational run identity."""
    profile_only_pairs = {"--profile-warmup", "--profile-steps", "--profile-trace-output"}
    result: list[str] = []
    index = 0
    while index < len(profile_argv):
        token = profile_argv[index]
        if token in profile_only_pairs:
            if index + 1 >= len(profile_argv):
                raise ProfileAbort(f"profile-only argument lacks a value: {token}")
            index += 2
            continue
        if token == "--name":
            if index + 1 >= len(profile_argv):
                raise ProfileAbort("profile run name lacks a value")
            result.extend(("--name", run_name))
            index += 2
            continue
        result.append(token)
        index += 1
    if any(token in profile_only_pairs for token in result):
        raise ProfileAbort("conditional fit argv retained a profile-only option")
    # Rebuild the expected vector independently; any other argv difference fails closed.
    expected: list[str] = []
    index = 0
    while index < len(profile_argv):
        token = profile_argv[index]
        if token in profile_only_pairs:
            index += 2
            continue
        if token == "--name":
            expected.extend(("--name", run_name))
            index += 2
            continue
        expected.append(token)
        index += 1
    if result != expected:
        raise ProfileAbort(
            "conditional fit argv differs beyond run identity and profile-only flags"
        )
    if "--profile-trace-output" in result or "--profile-steps" in result:
        raise ProfileAbort("fit argv would early-return before model training")
    return result


def _serializable_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Match the trainer's explicit Path-to-string JSON conversion."""
    return {
        name: str(value) if isinstance(value, Path) else value for name, value in arguments.items()
    }


def verify_fit_artifacts(
    run_dir: Path,
    *,
    expected_args: dict[str, Any],
    mounted: dict[str, Any],
    trainer_sha256: str,
) -> dict[str, Any]:
    """Verify fixed epoch-3 checkpoints, trainer provenance, and exact fold membership."""
    if not run_dir.is_dir():
        raise ProfileAbort(f"conditional fit output directory is missing: {run_dir}")
    checkpoint_paths = {name: run_dir / name for name in ("model.pt", "model_ep3.pt")}
    required = {
        **checkpoint_paths,
        "provenance.json": run_dir / "provenance.json",
        "history.json": run_dir / "history.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ProfileAbort(f"conditional fit outputs are incomplete: {missing}")
    checkpoint_receipts = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in checkpoint_paths.items()
    }
    provenance = json.loads(required["provenance.json"].read_text(encoding="utf-8"))
    history = json.loads(required["history.json"].read_text(encoding="utf-8"))
    observed_args = provenance.get("args")
    if not isinstance(observed_args, dict):
        raise ProfileAbort("fit provenance has no parsed trainer arguments")
    expected_clean = _serializable_args(expected_args)
    observed_clean = _serializable_args(observed_args)
    for key in ("name", "profile_steps", "profile_warmup", "profile_trace_output", "max_batches"):
        expected_clean.pop(key, None)
        observed_clean.pop(key, None)
    if expected_clean != observed_clean:
        mismatches = {
            key: {"expected": expected_clean.get(key), "observed": observed_clean.get(key)}
            for key in sorted(set(expected_clean) | set(observed_clean))
            if expected_clean.get(key) != observed_clean.get(key)
        }
        raise ProfileAbort(
            f"fit provenance arguments differ from profiled scientific argv: {mismatches}"
        )
    if provenance.get("split_manifest_sha256") != SPLIT_SHA256:
        raise ProfileAbort("fit provenance split-manifest digest mismatch")
    if provenance.get("seed") != 3 or provenance.get("selected_epoch") != 3:
        raise ProfileAbort("fit provenance does not select the frozen seed-3 epoch-3 checkpoint")
    recorded_hashes = provenance.get("checkpoint_sha256", {})
    for name, receipt in checkpoint_receipts.items():
        if recorded_hashes.get(name) != receipt["sha256"]:
            raise ProfileAbort(f"trainer provenance checkpoint digest mismatch: {name}")
    expected_fit = {
        "train": set(map(int, mounted["split"].train_groups)),
        "valid": set(map(int, mounted["split"].valid_groups)),
    }
    observed_fit = provenance.get("fit_row_groups", {})
    if not isinstance(observed_fit, dict) or any(
        set(map(int, observed_fit.get(name, []))) != groups for name, groups in expected_fit.items()
    ):
        raise ProfileAbort(
            "fit provenance fitting groups differ from frozen train/valid membership"
        )
    if set(map(int, provenance.get("evaluation_row_groups", []))) != set(
        map(int, mounted["split"].evaluation_groups)
    ):
        raise ProfileAbort("fit provenance evaluation groups differ from frozen fold-0 mask")
    if any(
        expected_fit[split_name] & set(map(int, mounted["split"].evaluation_groups))
        for split_name in expected_fit
    ):
        raise ProfileAbort("frozen fit membership overlaps the fold-0 evaluation mask")
    if (
        not isinstance(history, list)
        or len(history) != 4
        or [row.get("epoch") for row in history] != [0, 1, 2, 3]
    ):
        raise ProfileAbort("fit history is incomplete or does not contain epochs 0 through 3")
    if any(row.get("evaluation") != "withheld" for row in history):
        raise ProfileAbort("fit history contains evaluation output despite the no-eval recipe")
    return {
        "status": "verified",
        "trainer_sha256": trainer_sha256,
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "fit_membership_counts": {name: len(groups) for name, groups in expected_fit.items()},
        "evaluation_group_count": len(set(map(int, mounted["split"].evaluation_groups))),
        "evaluation_overlap_groups": 0,
        "selected_epoch": provenance["selected_epoch"],
        "fitting_seconds": provenance.get("fitting_seconds"),
        "checkpoint_receipts": checkpoint_receipts,
        "provenance_sha256": sha256_file(required["provenance.json"]),
        "history_sha256": sha256_file(required["history.json"]),
        "history_epochs": [row.get("epoch") for row in history],
        "predictions_or_scores_written_by_this_worker": False,
    }


def _raw_input_stats_unchanged(mounted: dict[str, Any]) -> dict[str, Any]:
    """Compare every stat field with the identity bound before the first trainer call."""
    observations = mounted["identity"]["observed"]
    current: dict[str, Any] = {}
    for split_name, path in mounted["raw_paths"].items():
        logical_name = f"datasets/{split_name}.parquet"
        binding = observations[logical_name]["read_only_file_stat_binding"]
        observed_identity = FileIdentity.from_stat(path.stat())
        expected_identity = FileIdentity(
            device=int(binding["device"]),
            inode=int(binding["inode"]),
            size=int(binding["size"]),
            modified_ns=int(binding["mtime_ns"]),
            changed_ns=int(binding["ctime_ns"]),
            mode=int(binding["mode"]),
        )
        canonical_path = str(path.resolve(strict=True))
        matches = (
            canonical_path == binding["canonical_path"] and observed_identity == expected_identity
        )
        current[logical_name] = {
            "canonical_path": canonical_path,
            "identity_matches_initial_verified_binding": matches,
            "device": observed_identity.device,
            "inode": observed_identity.inode,
            "size": observed_identity.size,
            "mtime_ns": observed_identity.modified_ns,
            "ctime_ns": observed_identity.changed_ns,
            "mode": observed_identity.mode,
        }
    return {
        "all_unchanged": bool(current)
        and all(row["identity_matches_initial_verified_binding"] for row in current.values()),
        "files": current,
    }


def validate_valid_prediction_export(
    prediction_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path,
    target_cache_path: Path,
    predictor_path: Path,
    input_identity_path: Path,
    mounted: dict[str, Any],
    *,
    run_name: str,
) -> dict[str, Any]:
    """Verify a fixed-epoch valid export by streaming/memmapping its values."""
    import numpy as np

    if not prediction_path.is_file() or not sidecar_path.is_file():
        raise ProfileAbort("fixed-epoch valid prediction or sidecar is missing")
    if prediction_path.with_name(prediction_path.name + ".partial").exists():
        raise ProfileAbort("partial valid prediction file remains after export")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(sidecar_path)
    predictor_sha = sha256_file(predictor_path)
    expected_groups: list[int] = []
    with np.load(target_cache_path, allow_pickle=False) as cache:
        expected_groups = list(map(int, np.asarray(cache["row_group"], dtype=np.int64)))
    row_groups = sidecar.get("row_groups")
    if (
        sidecar.get("run") != run_name
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 3
        or sidecar.get("checkpoint_sha256") != checkpoint_sha
        or sidecar.get("prediction_sha256") != prediction_sha
        or sidecar.get("split_manifest_sha256") != mounted["identity"].get("split_sha256")
        or sidecar.get("inference_precision") != "fp32"
        or row_groups != expected_groups
    ):
        raise ProfileAbort("valid prediction sidecar differs from fixed epoch-3 source identity")
    expected_shape = (len(expected_groups), 20_000, 2)
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if prediction_path.stat().st_size != expected_bytes:
        raise ProfileAbort("valid prediction file has an unexpected size")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != expected_shape or prediction.dtype != np.float32:
        raise ProfileAbort("valid prediction array has an unexpected shape or dtype")
    finite_rows = 0
    for start in range(0, len(expected_groups), 16):
        batch = prediction[start : start + 16]
        if not np.isfinite(batch).all():
            raise ProfileAbort(f"valid prediction contains non-finite values near group {start}")
        finite_rows += len(batch)
    row_group_digest = hashlib.sha256(
        json.dumps(row_groups, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "status": "complete_ungraded",
        "run_name": run_name,
        "split": "valid",
        "epoch": 3,
        "inference_precision": "fp32",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "prediction_bytes": expected_bytes,
        "prediction_shape": list(expected_shape),
        "prediction_dtype": str(prediction.dtype),
        "finite_groups_checked": finite_rows,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": metadata_sha,
        "row_group_count": len(row_groups),
        "row_group_order_sha256": row_group_digest,
        "valid_target_index_sha256": mounted["identity"]["target_index_hashes"]["valid"],
        "input_identity_sha256": sha256_file(input_identity_path),
        "predictor_source_path": str(predictor_path),
        "predictor_source_sha256": predictor_sha,
        "official_metric_computed": False,
        "public_score_claim": None,
    }


def _input_digest_measurements(mounted: dict[str, Any], receipts: Path) -> dict[str, Any]:
    """Report observed raw-file digest source, bytes read, stat binding, and hash time."""
    memory_path = receipts / "memory_stat_phases.json"
    memory = (
        json.loads(memory_path.read_text(encoding="utf-8"))
        if memory_path.exists()
        else {"phases": {}}
    )
    phases = memory.get("phases", {})
    observed = mounted["identity"].get("observed", {})
    prior = mounted["identity"].get("prior_digest_reuse", {})
    result: dict[str, Any] = {}
    for split_name in ("train", "valid"):
        logical_name = f"datasets/{split_name}.parquet"
        row = observed.get(logical_name, {})
        hash_phase = phases.get(f"initial_raw_hash:{logical_name}", {})
        hash_details = hash_phase.get("details", {})
        source = prior.get(split_name, {})
        result[split_name] = {
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
            "bytes_hashed_this_attempt": row.get("bytes_hashed_this_attempt"),
            "digest_source": row.get("digest_source"),
            "current_digest_resolution_seconds": source.get("current_digest_resolution_seconds"),
            "full_content_hash_elapsed_seconds": hash_details.get("elapsed_seconds"),
            "full_hash_fallback_used": source.get("status") == "full_hash_fallback",
            "prior_identity_status": source.get("status"),
            "stat_binding": row.get("read_only_file_stat_binding"),
        }
    result["total_current_digest_resolution_seconds"] = sum(
        float(result[name]["current_digest_resolution_seconds"] or 0.0)
        for name in ("train", "valid")
    )
    result["total_full_content_hash_seconds"] = sum(
        float(result[name]["full_content_hash_elapsed_seconds"] or 0.0)
        for name in ("train", "valid")
    )
    return result


def evaluate_conditional_fit_gate(
    *,
    trace: dict[str, Any],
    timing: dict[str, Any],
    loader_budget: dict[str, Any],
    trainer_args: dict[str, Any],
    n_in: int,
    feature_width_probe: dict[str, Any],
    resources: dict[str, Any],
    export_probe: list[dict[str, Any]],
    raw_input_identity: dict[str, Any],
    profile_handoff_delta: dict[str, int],
    raw_digest_path_profile_delta: dict[str, dict[str, int]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Authorize same-worker fit only from complete finite profile and live gates."""
    reasons: list[str] = []
    attempts = trace.get("attempts")
    timed_rows = (
        [row for row in attempts if row.get("phase") == "timed"]
        if isinstance(attempts, list)
        else []
    )
    warmup_rows = (
        [row for row in attempts if row.get("phase") == "warmup"]
        if isinstance(attempts, list)
        else []
    )
    expected_timed = PROFILE_STEPS * math.ceil(20_000 / 2_000)
    expected_warmup = PROFILE_WARMUP * math.ceil(20_000 / 2_000)
    timed = trace.get("timed", {})
    warmup = trace.get("warmup", {})
    numeric = trace.get("numeric_policy", {})
    if trace.get("schema") != "h232-profile-update-trace-v1":
        reasons.append("profile_trace_schema_mismatch")
    if trace.get("expected_timed_attempts") != expected_timed or len(timed_rows) != expected_timed:
        reasons.append("timed_update_attempt_count_mismatch")
    if trace.get("valid_for_throughput") is not True or timed.get("valid_for_throughput") is False:
        reasons.append("trainer_profile_validity_flag_false")
    if (
        timed.get("attempts") != expected_timed
        or timed.get("completed_updates") != expected_timed
        or timed.get("amp_skipped_updates") != 0
        or timed.get("loss_checks") != expected_timed
        or timed.get("loss_finite") is not True
        or timed.get("gradient_finite") is not True
    ):
        reasons.append("timed_updates_not_all_finite_and_completed")
    if any(
        row.get("loss_finite") is not True
        or row.get("unscaled_found_inf") is not False
        or row.get("optimizer_update_completed") is not True
        or row.get("scale_schedule_consistent") is not True
        for row in timed_rows
    ):
        reasons.append("timed_attempt_trace_contains_nonfinite_skip_or_scale_mismatch")
    if (
        len(warmup_rows) != expected_warmup
        or warmup.get("attempts") != expected_warmup
        or warmup.get("loss_checks") != expected_warmup
        or warmup.get("loss_finite") is not True
    ):
        reasons.append("warmup_trace_incomplete_or_nonfinite")
    if any(row.get("scale_schedule_consistent") is not True for row in [*warmup_rows, *timed_rows]):
        reasons.append("scale_schedule_trace_inconsistent")
    if numeric.get("initial_scale") != 4096.0:
        reasons.append("initial_amp_scale_differs_from_declared_4096")
    required_args = {
        "device": "cuda",
        "arch": "gru",
        "cell": "gru",
        "hidden": 384,
        "layers": 2,
        "seed": 3,
        "train_source": "train+valid",
        "train_seqs": 10_607,
        "batch": 8,
        "chunk": 2_000,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "precision": "fp16",
        "amp_init_scale": 4096.0,
        "raw_only": False,
        "heuristics": False,
    }
    changed_args = {
        name: {"expected": value, "observed": trainer_args.get(name)}
        for name, value in required_args.items()
        if trainer_args.get(name) != value
    }
    if changed_args:
        reasons.append("profile_recipe_arguments_differ_from_frozen_h232")
    expected_probe = {
        "probe_input_shape": [2, 3, EXPECTED_RAW_FEATURES],
        "probe_event_output_shape": [2, 3, EXPECTED_EVENT_FEATURES],
        "raw_feature_count": EXPECTED_RAW_FEATURES,
        "event_feature_count": EXPECTED_EVENT_FEATURES,
        "declared_event_feature_count": EXPECTED_EVENT_FEATURES,
        "derived_model_input_width": EXPECTED_MODEL_INPUT_WIDTH,
    }
    if any(feature_width_probe.get(name) != value for name, value in expected_probe.items()):
        reasons.append("pinned_source_feature_width_probe_mismatch")
    if (
        n_in != EXPECTED_MODEL_INPUT_WIDTH
        or feature_width_probe.get("derived_model_input_width") != n_in
    ):
        reasons.append("profile_input_width_differs_from_pinned_source_probe")
    if (
        loader_budget.get("effective_workers") != 2
        or loader_budget.get("effective_prefetch_factor") != 1
    ):
        reasons.append("observed_loader_budget_differs_from_loader2_prefetch1")
    if int(loader_budget.get("batch_tensor_bytes") or 0) != EXPECTED_LOADER_BATCH_BYTES:
        reasons.append("observed_loader_tensor_bytes_differ_from_frozen_batch_shape")
    if not raw_input_identity.get("all_unchanged"):
        reasons.append("verified_readonly_input_stat_tuple_changed_before_fit")
    lookup_count = int(profile_handoff_delta.get("verifier_lookups", 0))
    hit_count = int(profile_handoff_delta.get("digest_reuse_hits", 0))
    miss_count = int(profile_handoff_delta.get("digest_reuse_misses", 0))
    fallback_count = int(profile_handoff_delta.get("fallback_digest_calls", 0))
    if lookup_count != hit_count + miss_count or miss_count != fallback_count:
        reasons.append("profile_digest_lookup_accounting_or_fallback_mismatch")
    raw_path_failures: dict[str, dict[str, int | str]] = {}
    for split_name in ("train", "valid"):
        observed = raw_digest_path_profile_delta.get(split_name)
        if not isinstance(observed, dict):
            raw_path_failures[split_name] = {"reason": "raw_path_metrics_missing"}
            continue
        path_lookups = int(observed.get("verifier_lookups", 0))
        path_hits = int(observed.get("digest_reuse_hits", 0))
        path_misses = int(observed.get("digest_reuse_misses", 0))
        path_fallbacks = int(observed.get("fallback_digest_calls", 0))
        if (
            path_lookups != 1
            or path_lookups != path_hits + path_misses
            or path_misses != path_fallbacks
            or path_hits + path_fallbacks != 1
        ):
            raw_path_failures[split_name] = {
                "verifier_lookups": path_lookups,
                "digest_reuse_hits": path_hits,
                "digest_reuse_misses": path_misses,
                "fallback_digest_calls": path_fallbacks,
            }
    if raw_path_failures:
        reasons.append("raw_train_valid_digest_verification_not_fail_closed")
    if (
        resources.get("gpu_resource_query_status") != "ok"
        or resources.get("cuda_initialized") is not True
    ):
        reasons.append("actual_cuda_resource_identity_unavailable")
    if not resources.get("gpu_selected_physical_gpu_uuid") or resources.get(
        "gpu_selected_physical_gpu_uuid"
    ) != resources.get("gpu_bound_physical_gpu_uuid"):
        reasons.append("selected_physical_gpu_binding_missing_or_changed")
    resource_checks = {
        "rss_peak_bytes": (resources.get("rss_peak_bytes"), CAPS["rss_bytes"], "le"),
        "cuda_allocated_peak_bytes": (
            resources.get("cuda_allocated_peak_bytes"),
            CAPS["cuda_allocated_bytes"],
            "le",
        ),
        "cuda_reserved_peak_bytes": (
            resources.get("cuda_reserved_peak_bytes"),
            CAPS["cuda_reserved_bytes"],
            "le",
        ),
        "gpu_free_bytes": (resources.get("gpu_free_bytes"), 6 * 1024**3, "ge"),
        "host_available_bytes": (
            resources.get("host_available_bytes"),
            CAPS["headroom_bytes"],
            "ge",
        ),
        "cgroup_available_bytes": (
            resources.get("cgroup_available_bytes"),
            CAPS["headroom_bytes"],
            "ge",
        ),
        "disk_free_bytes": (resources.get("disk_free_bytes"), CAPS["disk_free_bytes"], "ge"),
    }
    resource_failures = {}
    for name, (observed, threshold, comparison) in resource_checks.items():
        good = isinstance(observed, int) and (
            observed <= threshold if comparison == "le" else observed >= threshold
        )
        if not good:
            resource_failures[name] = {
                "observed": observed,
                "required_comparison": comparison,
                "threshold": threshold,
            }
    if resource_failures:
        reasons.append("actual_profile_resource_headroom_or_peak_gate_failed")
    intervals = timing.get("end_to_end_inter_start_ms", {})
    p50_ms, p90_ms = intervals.get("p50"), intervals.get("p90")
    if (
        not isinstance(p50_ms, (int, float))
        or not isinstance(p90_ms, (int, float))
        or p50_ms <= 0
        or p90_ms <= 0
    ):
        reasons.append("profile_timing_summary_missing")
        p50_ms, p90_ms = 0.0, 0.0
    batches = FIT_PROJECTION_BATCHES_PER_EPOCH * FIT_PROJECTION_EPOCHS
    projected_fit_p50 = (
        p50_ms * batches / 1000 * FIT_PROJECTION_SAFETY_FACTOR + FIT_SETUP_OVERHEAD_SECONDS
    )
    projected_fit_p90 = (
        p90_ms * batches / 1000 * FIT_PROJECTION_SAFETY_FACTOR + FIT_SETUP_OVERHEAD_SECONDS
    )
    if projected_fit_p50 > MAX_PROJECTED_FIT_SECONDS:
        reasons.append("p50_projected_fit_lifetime_exceeds_bound")
    if elapsed_seconds + projected_fit_p90 > WORKER_WALL_CAP_S:
        reasons.append("p90_projected_total_lifetime_exceeds_worker_wall_cap")
    if len(export_probe) != 2 or any(row.get("output_finite") is not True for row in export_probe):
        reasons.append("actual_train_valid_shape_probe_missing_or_nonfinite")
    return {
        "status": "pass" if not reasons else "reject_no_fit",
        "fit_authorized_by_gate": not reasons,
        "reasons": reasons,
        "trace": {
            "expected_timed_outer_batches": PROFILE_STEPS,
            "expected_warmup_outer_batches": PROFILE_WARMUP,
            "expected_timed_optimizer_updates": expected_timed,
            "observed_timed_optimizer_updates": timed.get("attempts"),
            "timed_loss_finite": timed.get("loss_finite"),
            "timed_gradient_finite": timed.get("gradient_finite"),
            "timed_amp_skipped_updates": timed.get("amp_skipped_updates"),
            "warmup_optimizer_updates": warmup.get("attempts"),
            "warmup_amp_skipped_updates": warmup.get("amp_skipped_updates"),
            "numeric_policy": numeric,
        },
        "recipe_argument_mismatches": changed_args,
        "feature_width_probe": feature_width_probe,
        "loader_budget": loader_budget,
        "loader_tensor_bytes_policy": {
            "observed_batch_tensor_bytes": loader_budget.get("batch_tensor_bytes"),
            "expected_batch_tensor_bytes": EXPECTED_LOADER_BATCH_BYTES,
            "formula": "8 * 20000 * ((112 raw features + 2 targets) * 4 bytes + 1 bool mask byte); event/normalization features are computed on device",
        },
        "resource_failures": resource_failures,
        "raw_input_identity": raw_input_identity,
        "profile_digest_handoff_delta": profile_handoff_delta,
        "raw_digest_path_profile_delta": raw_digest_path_profile_delta,
        "raw_digest_path_failures": raw_path_failures,
        "timing": timing,
        "lifetime_projection": {
            "fit_sequences": 10_535 + 1_405,
            "fit_batches_per_epoch": FIT_PROJECTION_BATCHES_PER_EPOCH,
            "epochs": FIT_PROJECTION_EPOCHS,
            "total_fit_batches": batches,
            "p50_batch_interval_ms": p50_ms,
            "p90_batch_interval_ms": p90_ms,
            "safety_factor": FIT_PROJECTION_SAFETY_FACTOR,
            "fit_setup_overhead_seconds": FIT_SETUP_OVERHEAD_SECONDS,
            "projected_fit_seconds_p50": projected_fit_p50,
            "projected_fit_seconds_p90": projected_fit_p90,
            "worker_elapsed_before_fit_seconds": elapsed_seconds,
            "worker_wall_cap_seconds": WORKER_WALL_CAP_S,
            "max_projected_fit_seconds": MAX_PROJECTED_FIT_SECONDS,
        },
        "export_shape_probe": export_probe,
    }


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
        or receipt.get("scope") != "conditional-profile-gated-fit"
        or receipt.get("status") != "approved"
        or receipt.get("conditional_fit_authorized") is not True
        or receipt.get("fit_admission") != "same-worker-only-after-profile-gate"
        or not reviewer.get("id")
        or reviewer.get("id") == actor_id
        or reviewer.get("role") != "independent-controller"
    ):
        raise ProfileAbort("external review receipt lacks conditional fit authorization")
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
    cgroup_memory = cgroup_memory_stat_snapshot()
    if cgroup_memory.get("status") == "available":
        cgroup_available = int(cgroup_memory["memory_available_bytes"])
        cgroup_limit = int(cgroup_memory["memory_limit_bytes"])
    else:
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
        "cgroup_memory_stat_status": cgroup_memory.get("status", "unavailable"),
        "cgroup_memory_stat_path": cgroup_memory.get("cgroup_path"),
        "cgroup_memory_available_bytes": cgroup_memory.get("memory_available_bytes"),
        "cgroup_memory_limit_bytes": cgroup_memory.get("memory_limit_bytes"),
        "cgroup_memory_current_bytes": cgroup_memory.get("memory_current_bytes"),
        "cgroup_memory_stat": cgroup_memory.get("memory_stat"),
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
    if elapsed_s > WORKER_WALL_CAP_S:
        raise ProfileAbort(f"conditional worker exceeded its {WORKER_WALL_CAP_S}-second wall cap")
    if snapshot["rss_peak_bytes"] > CAPS["rss_bytes"]:
        raise ProfileAbort("worker RSS high-water exceeded the observed-cloud RAM cap")
    for field, cap in (
        ("cuda_allocated_bytes", CAPS["cuda_allocated_bytes"]),
        ("cuda_reserved_bytes", CAPS["cuda_reserved_bytes"]),
        ("cuda_allocated_peak_bytes", CAPS["cuda_allocated_bytes"]),
        ("cuda_reserved_peak_bytes", CAPS["cuda_reserved_bytes"]),
        ("gpu_process_bytes", CAPS["gpu_process_bytes"]),
    ):
        value = snapshot[field]
        if value is not None and value > cap:
            raise ProfileAbort(f"{field} exceeded its H232 cloud profile cap")
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
                f"{prefix} available memory fell below at least one H232 cloud headroom gate: {triggers}"
            )
    if snapshot["disk_free_bytes"] < CAPS["disk_free_bytes"]:
        raise ProfileAbort("working disk fell below the H232 profile free-space reserve")
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
                "device-wide GPU usage upper bound exceeded the H232 cloud device cap"
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


def _read_only_mount_for(path: Path) -> str | None:
    """Return the longest covering mount point only when its mount flags are read-only."""
    try:
        resolved = path.resolve(strict=True)
        rows = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    matches = []
    for row in rows:
        fields = row.split()
        if len(fields) < 7:
            continue
        mountpoint = fields[4]
        for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
            mountpoint = mountpoint.replace(encoded, decoded)
        root = Path(mountpoint)
        if resolved == root or root in resolved.parents:
            matches.append((len(root.parts), root, fields[5].split(",")))
    if not matches:
        return None
    _, mountpoint, options = max(matches, key=lambda item: item[0])
    return str(mountpoint) if "ro" in options else None


def verify_and_stage_inputs(
    input_root: Path, source_root: Path, receipt_path: Path, prior_identity_path: Path
) -> dict[str, Any]:
    """Record expected/observed mount identities, stage symlinks, and validate fold membership."""
    os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    prior_bytes = prior_identity_path.read_bytes()
    if hashlib.sha256(prior_bytes).hexdigest() != PRIOR_INPUT_IDENTITY_SHA256:
        raise ProfileAbort("prior V2 input-identity receipt digest mismatch")
    prior_identity = json.loads(prior_bytes)
    if prior_identity.get("status") != "matched":
        raise ProfileAbort("prior V2 input-identity receipt is not a matched source")
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
        "prior_verified_identity": {
            "parent_attempt": "H232_PROFILE_V2",
            "receipt_path": "prior_input_identity_v2.json",
            "receipt_sha256": PRIOR_INPUT_IDENTITY_SHA256,
            "reuse_condition": "exact canonical path and full FileIdentity stat tuple; otherwise full content hash fallback",
        },
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
    prior_reuse: dict[str, dict[str, Any]] = {}
    for split_name, path in raw.items():
        digest_resolution_started = time.monotonic()
        logical_name = f"datasets/{split_name}.parquet"
        expected = EXPECTED[split_name]
        prior_observed = prior_identity.get("observed", {}).get(logical_name, {})
        prior_binding = prior_observed.get("read_only_file_stat_binding", {})
        prior_registry = (
            prior_identity.get("observed", {})
            .get(f"{logical_name}.portable_registry_sha256", {})
            .get("sha256")
        )
        candidate = None
        reuse_reason = "prior stat identity unavailable; full hash fallback"
        read_only_mount = _read_only_mount_for(path)
        if (
            prior_observed.get("sha256") == expected["sha256"]
            and prior_observed.get("bytes") == expected["bytes"]
            and prior_registry == expected["sha256"]
            and prior_binding.get("canonical_path") == str(path.resolve(strict=True))
            and path.resolve(strict=True).is_relative_to(input_root.resolve(strict=True))
            and read_only_mount is not None
        ):
            try:
                bound_identity = FileIdentity(
                    device=int(prior_binding["device"]),
                    inode=int(prior_binding["inode"]),
                    size=int(prior_binding["size"]),
                    modified_ns=int(prior_binding["mtime_ns"]),
                    changed_ns=int(prior_binding["ctime_ns"]),
                    mode=int(prior_binding["mode"]),
                )
                candidate = handoff.candidate_from_prior_identity(
                    path,
                    prior_identity=bound_identity,
                    prior_sha256=str(prior_observed["sha256"]),
                )
            except (KeyError, TypeError, ValueError):
                candidate = None
            if candidate is not None:
                reuse_reason = "exact V2 full-stat tuple match on read-only Kaggle input mount"
        if candidate is None:
            candidate = _hash_raw_with_handoff(
                path,
                handoff=handoff,
                hash_receipt_path=hash_receipt_path,
                memory_stat_receipt_path=memory_stat_receipt_path,
                logical_name=logical_name,
                diagnostic_phase=f"initial_raw_hash:{logical_name}",
            )
            if prior_binding:
                reuse_reason = (
                    "V6 current stat/path did not match V2; original full hash fallback used"
                )
        else:
            _record_hash_advice(
                hash_receipt_path,
                logical_name,
                {
                    "status": "prior_verified_digest_reused_after_exact_stat_match",
                    "path": str(path),
                    "bytes_hashed": 0,
                    "bytes_bound_to_prior_full_hash": candidate.identity.size,
                    "prior_attempt": "H232_PROFILE_V2",
                    "prior_input_identity_sha256": PRIOR_INPUT_IDENTITY_SHA256,
                    "sha256": candidate.sha256,
                    "reuse_basis": "canonical path and all device/inode/size/mtime_ns/ctime_ns/mode fields matched",
                },
            )
            diagnostic_phase = f"initial_raw_digest_reuse:{logical_name}"
            _record_memory_stat_phase(memory_stat_receipt_path, diagnostic_phase, "before")
            _record_memory_stat_phase(
                memory_stat_receipt_path,
                diagnostic_phase,
                "after",
                details={"status": "reused_prior_verified_digest", "bytes_read_for_hash": 0},
            )
        prior_reuse[split_name] = {
            "status": "reused_prior_verified_digest"
            if candidate.reused_from_prior
            else "full_hash_fallback",
            "reason": reuse_reason,
            "prior_sha256": prior_observed.get("sha256"),
            "prior_identity_sha256": PRIOR_INPUT_IDENTITY_SHA256,
            "current_bytes_hashed": candidate.bytes_hashed,
            "read_only_mount": read_only_mount,
            "current_digest_resolution_seconds": round(
                time.monotonic() - digest_resolution_started, 6
            ),
        }
        identity["prior_digest_reuse"] = prior_reuse.copy()
        observed: dict[str, Any] = {
            "sha256": candidate.sha256,
            "bytes": candidate.identity.size,
            "bytes_hashed_this_attempt": candidate.bytes_hashed,
            "digest_source": "prior_verified_stat_bound_receipt"
            if candidate.reused_from_prior
            else "current_full_content_hash",
            "read_only_file_stat_binding": {
                "canonical_path": candidate.canonical_path,
                "device": candidate.identity.device,
                "inode": candidate.identity.inode,
                "size": candidate.identity.size,
                "mtime_ns": candidate.identity.modified_ns,
                "ctime_ns": candidate.identity.changed_ns,
                "mode": candidate.identity.mode,
                "mode_has_write_bits": bool(candidate.identity.mode & 0o222),
            },
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
    return {
        "identity": identity,
        "orders": orders,
        "split": split,
        "digest_handoff": handoff,
        "raw_paths": raw,
    }


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
        self._previous_memory_stat: dict[str, int] | None = None
        self._previous_memory_stat_path: str | None = None
        self._cache_paths: dict[str, Path] = {}
        self._cache_bindings: dict[str, dict[str, Any]] = {}
        self._cache_trace_path: Path | None = None
        self._cache_relief_budget = ReliefBudget(
            maximum_events=MAX_RELIEF_EVENTS,
            maximum_episode_bytes=MAX_ADVICE_BYTES,
            maximum_total_bytes=MAX_TOTAL_ADVICE_BYTES,
            minimum_interval_seconds=MIN_RELIEF_INTERVAL_SECONDS,
        )
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self._started = False

    def bind_cache_inputs(self, mounted: dict[str, Any], trace_path: Path) -> None:
        """Bind relief to the exact read-only train/valid files already identity-checked."""
        paths = mounted.get("raw_paths", {})
        observed = mounted.get("identity", {}).get("observed", {})
        if set(paths) != {"train", "valid"}:
            raise ProfileAbort("cache relief requires both verified train and valid inputs")
        bindings: dict[str, dict[str, Any]] = {}
        for split, path in paths.items():
            logical = f"datasets/{split}.parquet"
            binding = observed.get(logical, {}).get("read_only_file_stat_binding")
            if not isinstance(binding, dict):
                raise ProfileAbort(f"missing identity binding for cache input {logical}")
            bindings[split] = binding.copy()
            if str(path.resolve(strict=True)) != binding.get("canonical_path"):
                raise ProfileAbort(f"cache input path differs from verified identity: {logical}")
        with self._check_lock:
            self._cache_paths = {name: Path(path) for name, path in paths.items()}
            self._cache_bindings = bindings
            self._cache_trace_path = trace_path

    def _append_cache_relief(self, record: dict[str, Any]) -> None:
        path = self.receipts / "cache_pressure_relief.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()

    def _try_cache_relief(self, before: dict[str, Any]) -> dict[str, Any] | None:
        stat_before = before.get("cgroup_memory_stat")
        cgroup_path = before.get("cgroup_memory_stat_path")
        record: dict[str, Any] = {
            "time_unix": time.time(),
            "packet_id": PACKET_ID,
            "status": "skipped",
            "before": before,
        }
        if not memory_snapshot_matches(before) or not isinstance(stat_before, dict):
            record["reason"] = "cgroup memory.stat snapshot is unavailable or inconsistent"
            self._append_cache_relief(record)
            return None
        if self._previous_memory_stat_path != cgroup_path:
            record["reason"] = "no prior sample from the same constrained cgroup"
            self._append_cache_relief(record)
            return None
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        eligible, reasons = has_reclaimable_file_pressure(
            stat_before, self._previous_memory_stat, page_size=page_size
        )
        record["pressure_evidence"] = {
            "eligible": eligible,
            "reasons": reasons,
            "anon_bytes": stat_before.get("anon"),
            "file_bytes": stat_before.get("file"),
            "active_file_bytes": stat_before.get("active_file"),
            "inactive_file_bytes": stat_before.get("inactive_file"),
            "workingset_refault_file_pages": stat_before.get("workingset_refault_file"),
        }
        if not eligible:
            record["reason"] = "file-cache relief not justified by current memory.stat evidence"
            self._append_cache_relief(record)
            return None
        if not self._cache_paths or not self._cache_bindings or self._cache_trace_path is None:
            record["reason"] = "verified raw inputs or rowgroup trace are not bound"
            self._append_cache_relief(record)
            return None
        host_available = int(before.get("host_available_bytes") or 0)
        host_total = int(before.get("host_total_bytes") or 0)
        if not host_headroom_passes(
            host_available,
            host_total,
            minimum_bytes=CAPS["headroom_bytes"],
            minimum_fraction=CAPS["headroom_fraction"],
        ):
            record["reason"] = (
                "host hard gate is already crossed; cgroup cache relief is insufficient"
            )
            self._append_cache_relief(record)
            return None
        record["budget_before"] = self._cache_relief_budget.snapshot()
        allowed, budget_reason = self._cache_relief_budget.reserve(
            now=time.monotonic(), requested_bytes=MAX_ADVICE_BYTES
        )
        if not allowed:
            record["reason"] = f"cache-relief budget blocked this event: {budget_reason}"
            record["budget_after"] = self._cache_relief_budget.snapshot()
            self._append_cache_relief(record)
            return None
        record["budget_reservation_bytes"] = MAX_ADVICE_BYTES
        record["budget_after_reservation"] = self._cache_relief_budget.snapshot()
        relief = advise_recent_rowgroups(
            self._cache_paths,
            self._cache_bindings,
            self._cache_trace_path,
            page_size=page_size,
        )
        after = resource_snapshot(
            self.torch,
            self.working,
            expected_physical_gpu_uuid=self.selected_physical_gpu_uuid,
        )
        target = relief_target_bytes(
            int(after.get("cgroup_limit_bytes") or 0),
            CAPS["headroom_bytes"],
            CAPS["headroom_fraction"],
        )
        after_host_available = int(after.get("host_available_bytes") or 0)
        after_host_total = int(after.get("host_total_bytes") or 0)
        after_host_ok = host_headroom_passes(
            after_host_available,
            after_host_total,
            minimum_bytes=CAPS["headroom_bytes"],
            minimum_fraction=CAPS["headroom_fraction"],
        )
        recovered = (
            relief.get("status") == "advised"
            and memory_snapshot_matches(after)
            and int(after.get("cgroup_available_bytes") or 0) >= target
            and after_host_ok
        )
        record.update(
            {
                "status": "recovered" if recovered else "failed_closed",
                "relief": relief,
                "recovery_target_bytes": target,
                "host_hard_gates_passed_after_advice": after_host_ok,
                "after": after,
                "recovered": recovered,
                "budget_after_advice": self._cache_relief_budget.snapshot(),
            }
        )
        self._append_cache_relief(record)
        if not recovered:
            raise ProfileAbort(
                "bounded raw-input page-cache advice did not restore host and cgroup headroom"
            )
        return after

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
            host_available = int(snap.get("host_available_bytes") or 0)
            host_total = int(snap.get("host_total_bytes") or 0)
            cgroup_available = int(snap.get("cgroup_available_bytes") or 0)
            cgroup_limit = int(snap.get("cgroup_limit_bytes") or 0)
            cgroup_low = cgroup_available < CAPS["headroom_bytes"] or (
                cgroup_limit > 0 and cgroup_available / cgroup_limit < CAPS["headroom_fraction"]
            )
            host_ok = host_headroom_passes(
                host_available,
                host_total,
                minimum_bytes=CAPS["headroom_bytes"],
                minimum_fraction=CAPS["headroom_fraction"],
            )
            if cgroup_low and host_ok:
                try:
                    recovered = self._try_cache_relief(snap)
                    if recovered is not None:
                        snap = recovered
                except ProfileAbort as error:
                    self.failure = str(error)
                    self.latest = snap
                    atomic_json(
                        self.receipts / "resource.json",
                        {"phase": "cache_pressure", "time_unix": time.time(), "snapshot": snap},
                    )
                    atomic_json(
                        self.receipts / "resource_abort.json",
                        {
                            "status": "aborted",
                            "phase": "cache_pressure",
                            "reason": self.failure,
                            "snapshot": snap,
                            "time_unix": time.time(),
                        },
                    )
                    raise
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
            stat_after = snap.get("cgroup_memory_stat")
            if isinstance(stat_after, dict):
                self._previous_memory_stat = stat_after.copy()
                self._previous_memory_stat_path = snap.get("cgroup_memory_stat_path")
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
    """Run exact H232 profile, measured fit gate, and a same-worker scratch fit only on pass."""
    work = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    receipts = work / "h232_conditional_v6"
    receipts.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    timing: dict[str, Any] = {
        "worker_start_unix": time.time(),
        "worker_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stage_durations: dict[str, float] = {}
    stage_name = "packet_review_and_source_extract"
    stage_started = start

    def enter_stage(name: str, **fields: Any) -> None:
        nonlocal stage_name, stage_started
        now = time.monotonic()
        stage_durations[stage_name] = round(now - stage_started, 6)
        stage_name = name
        stage_started = now
        phase.update({"phase": name, "updated_unix": time.time(), **fields})
        atomic_json(phase_path, phase)

    def finish_stage() -> None:
        now = time.monotonic()
        stage_durations[stage_name] = round(now - stage_started, 6)
        timing["phase_durations_seconds"] = stage_durations.copy()
        timing["worker_wall_seconds"] = round(now - start, 6)

    phase: dict[str, Any] = {
        "packet": PACKET_ID,
        "status": "active",
        "phase": "review_gate",
        "worker_wall_cap_seconds": WORKER_WALL_CAP_S,
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
                "stage_durations_seconds": {
                    **stage_durations,
                    stage_name: round(time.monotonic() - stage_started, 6),
                },
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
        successor_recipe_path = packet_root / "H232_AMP4096_SUCCESSOR_RECIPE.json"
        if sha256_file(successor_recipe_path) != meta.get("source_recipe_sha256"):
            raise ProfileAbort("embedded AMP4096 successor recipe hash mismatch")
        successor_recipe = json.loads(successor_recipe_path.read_text(encoding="utf-8"))
        if (
            successor_recipe.get("recipe_id") != "E529-H232-FULLFEATURE-GRU384-B8-AMP4096-S3-V1"
            or successor_recipe.get("scientific_recipe", {}).get("amp_init_scale") != 4096.0
            or successor_recipe.get("historical_recipe", {}).get("fit_amp_initial_scale") != 65536.0
        ):
            raise ProfileAbort(
                "embedded recipe does not bind the declared AMP4096 successor and historical default"
            )
        receipt_bytes = (packet_root / "external_review.json").read_bytes()
        receipt = validate_external_review(
            receipt_bytes, meta.get("external_review_sha256", ""), "h232_cloud_operator"
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
        source_root = work / "h232_source_v6"
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
        source_manifest = json.loads(
            (packet_root / "source_archive_manifest.json").read_text(encoding="utf-8")
        )
        expected_source_members = source_manifest.get("members")
        if not isinstance(expected_source_members, list) or not expected_source_members:
            raise ProfileAbort("source archive manifest has no member list")
        archived_members = {
            PurePosixPath(item["path"]).as_posix(): item
            for item in expected_source_members
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        if len(archived_members) != len(expected_source_members):
            raise ProfileAbort("source archive manifest contains duplicate or malformed members")
        if set(archived_members) != {member.name for member in members if member.isfile()}:
            raise ProfileAbort("source archive member names differ from reviewed manifest")
        for relative, member_manifest in archived_members.items():
            staged_path = source_root / relative
            if staged_path.stat().st_size != member_manifest.get("bytes") or sha256_file(
                staged_path
            ) != member_manifest.get("sha256"):
                raise ProfileAbort(
                    f"extracted source member differs from reviewed manifest: {relative}"
                )
        timing["source_archive_member_count"] = len(archived_members)
        timing["source_archive_manifest_sha256"] = sha256_file(
            packet_root / "source_archive_manifest.json"
        )
        timing["source_archive_sha256"] = meta["source_archive_sha256"]
        timing["trainer_sha256"] = meta["trainer_sha256"]
        enter_stage("input_identity_and_membership")
        input_identity_started = time.monotonic()
        mounted = verify_and_stage_inputs(
            Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")),
            source_root,
            identity_path,
            packet_root / "prior_input_identity_v2.json",
        )
        cache_trace_path = receipts / "rowgroup_reads.tsv"
        guard.bind_cache_inputs(mounted, cache_trace_path)
        os.environ["WNN_CACHE_TRACE_PATH"] = str(cache_trace_path)
        for split, path in mounted["raw_paths"].items():
            os.environ[f"WNN_CACHE_{split.upper()}_PATH"] = str(path.resolve(strict=True))
        timing["input_identity_and_membership_seconds"] = round(
            time.monotonic() - input_identity_started, 6
        )
        timing["input_digest_measurements"] = _input_digest_measurements(mounted, receipts)
        timing["raw_digest_handoff_before_profile"] = mounted["digest_handoff"].metrics()
        enter_stage("cuda_startup_and_device_identity")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WNN_LOADER_WORKERS"] = "2"
        os.environ["WNN_LOADER_RAM_GB"] = "0.30"
        os.environ["WNN_GPU_FRACTION"] = "0.75"
        os.environ["WNN_THREADS"] = "4"
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        guard.check()
        guard.stop_polling()
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != CAPS["visible_gpus"]:
            raise ProfileAbort("profile requires exactly one visible CUDA GPU")
        if len(os.sched_getaffinity(0)) < 4:
            raise ProfileAbort(
                "profile requires the four observed Kaggle CPU slots for loader workers"
            )
        device_identity = runtime_identity(torch)
        gpu_rows = device_identity.get("physical_gpu_inventory_and_utilization", {}).get("rows", [])
        if len(gpu_rows) != 2 or any("T4" not in str(row.get("name", "")) for row in gpu_rows):
            raise ProfileAbort(
                "cloud resource caps require the previously observed two-T4 Kaggle host"
            )
        selected_physical_gpu_uuid = device_identity.get("selected_physical_gpu_uuid")
        startup_resources = resource_snapshot(
            torch, work, expected_physical_gpu_uuid=selected_physical_gpu_uuid
        )
        timing["worker_start_to_cuda_initialized_seconds"] = round(time.monotonic() - start, 6)
        if int(startup_resources.get("cgroup_limit_bytes") or 0) < 32_212_254_720:
            raise ProfileAbort("cloud host cgroup is smaller than the measured V12 profile host")
        if int(startup_resources.get("gpu_total_bytes") or 0) < 15_636_037_632:
            raise ProfileAbort("selected T4 CUDA capacity is below the measured V12 device")
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
        guard.bind_cache_inputs(mounted, cache_trace_path)
        guard.check(startup_resources)
        require_physical_gpu_inventory(device_identity)
        guard.start_polling()
        torch.cuda.reset_peak_memory_stats(0)
        profile_capture: dict[str, Any] = {}
        sys.path.insert(0, str(source_root))
        import importlib.util

        spec = importlib.util.spec_from_file_location("h232_train_rnn", trainer)
        if spec is None or spec.loader is None:
            raise ProfileAbort("could not load pinned train_rnn.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        profile_capture["feature_width_probe"] = _probe_source_input_width(module)
        original_loader_budget = module.dataloader_budget

        def capture_loader_budget(
            batch_bytes: int, ram_budget_bytes: int | None = None, max_workers: int = 4
        ) -> tuple[int, int]:
            budget = original_loader_budget(
                batch_bytes, ram_budget_bytes=ram_budget_bytes, max_workers=max_workers
            )
            profile_capture["loader_budget"] = {
                "batch_tensor_bytes": int(batch_bytes),
                "ram_budget_bytes": int(ram_budget_bytes) if ram_budget_bytes is not None else None,
                "max_workers": int(max_workers),
                "configured_worker_cap": int(os.environ["WNN_LOADER_WORKERS"]),
                "observed_threads": int(os.environ["WNN_THREADS"]),
                "effective_workers": int(budget[0]),
                "effective_prefetch_factor": int(budget[1]),
            }
            return budget

        module.dataloader_budget = capture_loader_budget
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
                timer.cur["_h232_stop_wall"] = time.perf_counter()
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
            profile_capture["training_resource_snapshot"] = training_resources.copy()
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
            step_ms = [(item["_h232_stop_wall"] - item["_wall"]) * 1000 for item in timed]
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
                "H232_PROFILE_TIMINGS " + json.dumps(profile_capture["timing"], sort_keys=True),
                flush=True,
            )
            return summary

        module.StageTimer.report = report
        raw_digest_paths_before_profile = {
            split_name: mounted["digest_handoff"].metrics_for(path)
            for split_name, path in mounted["raw_paths"].items()
        }
        timing["raw_digest_path_metrics_before_profile"] = raw_digest_paths_before_profile
        argv = [
            "--name",
            "bd26_h232_full_s3_f0_profile_b8_v6",
            "--device",
            "cuda",
            "--arch",
            "gru",
            "--cell",
            "gru",
            "--hidden",
            "384",
            "--layers",
            "2",
            "--dropout",
            "0",
            "--seed",
            "3",
            "--loss",
            "wmse",
            "--loss-norm",
            "chunk",
            "--train-source",
            "train+valid",
            "--train-seqs",
            "10607",
            "--split-manifest",
            staged_split_manifest_argument(source_root),
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
            "8",
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
            "--profile-trace-output",
            str(receipts / "profile_update_trace.json"),
            "--amp-init-scale",
            "4096.0",
        ]
        require_profile_only(argv)
        enter_stage("trainer_profile", profile_warmup=PROFILE_WARMUP, profile_steps=PROFILE_STEPS)
        sys.argv = [str(trainer), *argv]
        profile_started = time.monotonic()
        gpu_sampler = GpuTelemetrySampler(receipts / "gpu_telemetry.json")
        gpu_sampler.start()
        try:
            module.main()
        finally:
            gpu_sampler.stop()
            module.StageTimer.stop = original_stop
            module.StageTimer.report = original_report
            raw_digest_paths_after_profile = {
                split_name: mounted["digest_handoff"].metrics_for(path)
                for split_name, path in mounted["raw_paths"].items()
            }
            timing["raw_digest_path_metrics_after_profile"] = raw_digest_paths_after_profile
            raw_digest_path_profile_delta = _path_metric_delta(
                raw_digest_paths_before_profile, raw_digest_paths_after_profile
            )
            timing["raw_digest_path_profile_delta"] = raw_digest_path_profile_delta
        guard.raise_if_failed()
        timing["trainer_profile_seconds"] = round(time.monotonic() - profile_started, 6)
        trace_path = receipts / "profile_update_trace.json"
        if not trace_path.is_file():
            raise ProfileAbort("trainer did not write the required per-update profile trace")
        profile_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        timing["profile_update_trace_sha256"] = sha256_file(trace_path)
        profile_handoff_after = mounted["digest_handoff"].metrics()
        profile_handoff_delta = _metric_delta(
            timing["raw_digest_handoff_before_profile"], profile_handoff_after
        )
        timing["raw_digest_handoff_after_profile"] = profile_handoff_after
        timing["profile_digest_handoff_delta"] = profile_handoff_delta
        run_dir = source_root / "artifacts/runs/bd26_h232_full_s3_f0_profile_b8_v6"
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
        enter_stage("actual_train_valid_shape_probe")
        export_started = time.monotonic()
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
        timing["actual_shape_probe_seconds"] = round(time.monotonic() - export_started, 6)
        enter_stage("conditional_fit_gate")
        raw_identity_before_fit = _raw_input_stats_unchanged(mounted)
        current_resources = resource_snapshot(
            torch, work, expected_physical_gpu_uuid=selected_physical_gpu_uuid
        )
        profile_resources = profile_capture.get("training_resource_snapshot", {})
        fit_gate_resources = dict(profile_resources)
        for name in (
            "rss_peak_bytes",
            "gpu_free_bytes",
            "host_available_bytes",
            "cgroup_available_bytes",
            "disk_free_bytes",
            "gpu_resource_query_status",
            "cuda_initialized",
            "gpu_selected_physical_gpu_uuid",
            "gpu_bound_physical_gpu_uuid",
        ):
            fit_gate_resources[name] = current_resources.get(name, fit_gate_resources.get(name))
        fit_gate_resources["rss_peak_bytes"] = max(
            int(profile_resources.get("rss_peak_bytes") or 0),
            int(current_resources.get("rss_peak_bytes") or 0),
        )
        fit_gate_resources["cuda_allocated_peak_bytes"] = max(
            int(profile_resources.get("cuda_allocated_peak_bytes") or 0),
            *(int(row.get("torch_allocated_peak_bytes") or 0) for row in export_metrics),
        )
        fit_gate_resources["cuda_reserved_peak_bytes"] = max(
            int(profile_resources.get("cuda_reserved_peak_bytes") or 0),
            *(int(row.get("torch_reserved_peak_bytes") or 0) for row in export_metrics),
        )
        profile_loader_budget = dict(profile_capture.get("loader_budget") or {})
        gate_result = evaluate_conditional_fit_gate(
            trace=profile_trace,
            timing=profile_capture.get("timing") or {},
            loader_budget=profile_loader_budget,
            trainer_args=profile_capture.get("args") or {},
            n_in=int(profile_capture.get("n_in") or 0),
            feature_width_probe=profile_capture.get("feature_width_probe") or {},
            resources=fit_gate_resources,
            export_probe=export_metrics,
            raw_input_identity=raw_identity_before_fit,
            profile_handoff_delta=profile_handoff_delta,
            raw_digest_path_profile_delta=raw_digest_path_profile_delta,
            elapsed_seconds=time.monotonic() - start,
        )
        gate_result.update(
            {
                "packet_id": PACKET_ID,
                "external_review_sha256": meta["external_review_sha256"],
                "source_archive_sha256": meta["source_archive_sha256"],
                "trainer_sha256": meta["trainer_sha256"],
                "input_identity_sha256": sha256_file(identity_path),
                "profile_trace_sha256": sha256_file(trace_path),
            }
        )
        atomic_json(receipts / "profile_gate_result.json", gate_result)
        if not gate_result["fit_authorized_by_gate"]:
            final_resource = guard.check()
            guard.stop_polling()
            finish_stage()
            phase.update(
                {
                    "phase": "profile_gate_rejected",
                    "status": "profile_gate_rejected_no_fit",
                    "fit_started": False,
                    "updated_unix": time.time(),
                    "end_unix": time.time(),
                    "wall_seconds": time.monotonic() - start,
                    "stage_durations_seconds": stage_durations.copy(),
                }
            )
            result = {
                "packet": PACKET_ID,
                "status": "profile_gate_rejected_no_fit",
                "fit_started": False,
                "checkpoint_written": False,
                "predictions_written": False,
                "profile_gate": gate_result,
                "source_archive_sha256": meta["source_archive_sha256"],
                "trainer_sha256": meta["trainer_sha256"],
                "external_review_sha256": meta["external_review_sha256"],
                "input_identity": mounted["identity"],
                "timing": {
                    **timing,
                    "phase_durations_seconds": stage_durations.copy(),
                    "worker_wall_seconds": time.monotonic() - start,
                },
                "hash_io_advice": json.loads(
                    (receipts / "hash_io_advice.json").read_text(encoding="utf-8")
                ),
                "raw_digest_handoff": mounted["digest_handoff"].metrics(),
                "loader_environment": {
                    "WNN_LOADER_WORKERS": os.environ.get("WNN_LOADER_WORKERS"),
                    "WNN_LOADER_RAM_GB": os.environ.get("WNN_LOADER_RAM_GB"),
                    "WNN_GPU_FRACTION": os.environ.get("WNN_GPU_FRACTION"),
                    "WNN_THREADS": os.environ.get("WNN_THREADS"),
                    "effective_profile": profile_loader_budget,
                },
                "environment_identity": device_identity,
                "gpu_telemetry_receipt": str(receipts / "gpu_telemetry.json"),
                "export_probe": export_metrics,
                "resources_final": final_resource,
                "caps": CAPS,
                "worker_wall_cap_seconds": WORKER_WALL_CAP_S,
                "wall_seconds": time.monotonic() - start,
            }
            atomic_json(receipts / "profile_result.json", result)
            atomic_json(phase_path, phase)
            return result

        fit_run_name = "bd26_h232_full_s3_f0_amp4096_v6"
        fit_argv = derive_fit_argv(argv, fit_run_name)
        expected_fit_args = dict(profile_capture.get("args") or {})
        if not _raw_input_stats_unchanged(mounted).get("all_unchanged"):
            raise ProfileAbort(
                "raw input stat tuple changed between profile gate and fit admission"
            )
        run_dir = source_root / "artifacts/runs" / fit_run_name
        if run_dir.exists():
            raise ProfileAbort(f"refusing to overwrite existing conditional fit output: {run_dir}")
        profile_model = profile_capture.pop("model", None)
        if profile_model is not None:
            profile_model.zero_grad(set_to_none=True)
            del profile_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        module.make_model = original_make_model
        timing["profile_to_fit_cuda_cleanup_seconds"] = 0.0
        enter_stage("conditional_fit_and_fixed_checkpoint_writes", fit_run_name=fit_run_name)
        fit_started = time.monotonic()
        sys.argv = [str(trainer), *fit_argv]
        fit_sampler = GpuTelemetrySampler(receipts / "gpu_telemetry_fit.json")
        fit_sampler.start()
        try:
            module.main()
        except BaseException as error:
            timing["conditional_fit_seconds"] = round(time.monotonic() - fit_started, 6)
            fit_failure = {
                "status": "failed",
                "fit_started": True,
                "run_name": fit_run_name,
                "reason": f"{type(error).__name__}: {error}",
                "traceback": __import__("traceback").format_exc()[-16_384:],
                "timing": timing.copy(),
            }
            atomic_json(receipts / "fit_result.json", fit_failure)
            raise
        finally:
            fit_sampler.stop()
        timing["conditional_fit_seconds"] = round(time.monotonic() - fit_started, 6)
        guard.raise_if_failed()
        enter_stage("fit_output_and_membership_validation")
        validation_started = time.monotonic()
        fit_result = verify_fit_artifacts(
            run_dir,
            expected_args=expected_fit_args,
            mounted=mounted,
            trainer_sha256=meta["trainer_sha256"],
        )
        timing["fit_output_validation_seconds"] = round(time.monotonic() - validation_started, 6)
        enter_stage("fixed_epoch3_valid_prediction_export", cap_seconds=PREDICTION_EXPORT_CAP_S)
        export_started = time.monotonic()
        output_name = "h232_v6_fixed_ep3"
        prediction_path = source_root / "cache" / f"pred_{output_name}_valid.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        predictor_path = source_root / "scripts" / "predict_run.py"
        checkpoint_path = run_dir / "model_ep3.pt"
        stdout_path = receipts / "valid_prediction_export.stdout.log"
        stderr_path = receipts / "valid_prediction_export.stderr.log"
        if (
            prediction_path.exists()
            or sidecar_path.exists()
            or prediction_path.with_name(prediction_path.name + ".partial").exists()
        ):
            raise ProfileAbort("refusing to overwrite an existing V6 valid prediction export")
        prediction_argv = [
            sys.executable,
            str(predictor_path),
            fit_run_name,
            "valid",
            "--epoch",
            "3",
            "--output-name",
            output_name,
            "--inference-precision",
            "fp32",
        ]
        prediction_process: subprocess.CompletedProcess[bytes] | None = None
        try:
            with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
                prediction_process = subprocess.run(
                    prediction_argv,
                    cwd=source_root,
                    env=os.environ.copy(),
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    check=False,
                    timeout=PREDICTION_EXPORT_CAP_S,
                )
        except subprocess.TimeoutExpired as error:
            export_failure = {
                "status": "timeout",
                "cap_seconds": PREDICTION_EXPORT_CAP_S,
                "elapsed_seconds": round(time.monotonic() - export_started, 6),
                "argv": prediction_argv,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "diagnostic": str(error),
            }
            atomic_json(receipts / "valid_prediction_export.json", export_failure)
            raise ProfileAbort(
                "fixed epoch-3 valid prediction export exceeded its 1800-second cap"
            ) from error
        timing["valid_prediction_export_seconds"] = round(time.monotonic() - export_started, 6)
        if prediction_process is None or prediction_process.returncode != 0:
            export_failure = {
                "status": "failed",
                "return_code": prediction_process.returncode if prediction_process else None,
                "cap_seconds": PREDICTION_EXPORT_CAP_S,
                "elapsed_seconds": timing["valid_prediction_export_seconds"],
                "argv": prediction_argv,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            atomic_json(receipts / "valid_prediction_export.json", export_failure)
            raise ProfileAbort("fixed epoch-3 valid prediction export subprocess failed")
        valid_prediction_export = validate_valid_prediction_export(
            prediction_path,
            sidecar_path,
            checkpoint_path,
            source_root / "cache" / "valid_targets.npz",
            predictor_path,
            identity_path,
            mounted,
            run_name=fit_run_name,
        )
        valid_prediction_export.update(
            {
                "argv": prediction_argv,
                "return_code": prediction_process.returncode,
                "elapsed_seconds": timing["valid_prediction_export_seconds"],
                "cap_seconds": PREDICTION_EXPORT_CAP_S,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        atomic_json(receipts / "valid_prediction_export.json", valid_prediction_export)
        guard.raise_if_failed()
        fit_handoff_after = mounted["digest_handoff"].metrics()
        timing["raw_digest_handoff_after_fit"] = fit_handoff_after
        timing["fit_digest_handoff_delta"] = _metric_delta(profile_handoff_after, fit_handoff_after)
        final_resource = guard.check()
        guard.stop_polling()
        finish_stage()
        phase.update(
            {
                "phase": "complete",
                "status": "conditional_fit_and_prediction_export_complete_ungraded",
                "fit_started": True,
                "updated_unix": time.time(),
                "end_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
                "stage_durations_seconds": stage_durations.copy(),
            }
        )
        fit_result.update(
            {
                "packet_id": PACKET_ID,
                "run_name": fit_run_name,
                "fit_started": True,
                "profile_gate_sha256": sha256_file(receipts / "profile_gate_result.json"),
                "external_review_sha256": meta["external_review_sha256"],
                "source_archive_sha256": meta["source_archive_sha256"],
                "input_identity_sha256": sha256_file(identity_path),
                "fit_digest_handoff_delta": timing["fit_digest_handoff_delta"],
                "profile_amp_initial_scale": 4096.0,
                "fit_amp_initial_scale": 4096.0,
                "fit_scale_basis": "explicit H232 AMP4096 numeric-successor recipe; historical default65536 recipe is preserved separately",
                "profile_timing": profile_capture.get("timing"),
                "worker_phase_timing": timing,
                "prediction_export_or_grade_performed": True,
                "valid_prediction_export": valid_prediction_export,
                "official_metric_computed": False,
                "score_claim": None,
                "gpu_telemetry_fit": str(receipts / "gpu_telemetry_fit.json"),
                "resources_final": final_resource,
            }
        )
        atomic_json(receipts / "fit_result.json", fit_result)
        result = {
            "packet": PACKET_ID,
            "status": "conditional_fit_and_prediction_export_complete_ungraded",
            "fit_started": True,
            "checkpoint_written": True,
            "predictions_written": True,
            "valid_prediction_export": valid_prediction_export,
            "fit_result": fit_result,
            "profile_gate": gate_result,
            "source_archive_sha256": meta["source_archive_sha256"],
            "trainer_sha256": meta["trainer_sha256"],
            "external_review_sha256": meta["external_review_sha256"],
            "input_identity": mounted["identity"],
            "hash_io_advice": json.loads(
                (receipts / "hash_io_advice.json").read_text(encoding="utf-8")
            ),
            "raw_digest_handoff": fit_handoff_after,
            "timing": {
                **timing,
                "phase_durations_seconds": stage_durations.copy(),
                "worker_wall_seconds": time.monotonic() - start,
            },
            "loader_environment": {
                "WNN_LOADER_WORKERS": os.environ.get("WNN_LOADER_WORKERS"),
                "WNN_LOADER_RAM_GB": os.environ.get("WNN_LOADER_RAM_GB"),
                "WNN_GPU_FRACTION": os.environ.get("WNN_GPU_FRACTION"),
                "WNN_THREADS": os.environ.get("WNN_THREADS"),
                "effective_profile": profile_loader_budget,
                "effective_fit": profile_capture.get("loader_budget"),
            },
            "environment_identity": device_identity,
            "gpu_telemetry_receipt": str(receipts / "gpu_telemetry.json"),
            "export_probe": export_metrics,
            "resources_final": final_resource,
            "caps": CAPS,
            "worker_wall_cap_seconds": WORKER_WALL_CAP_S,
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
                "stage_durations_seconds": {
                    **stage_durations,
                    stage_name: round(time.monotonic() - stage_started, 6),
                },
            }
        )
        atomic_json(phase_path, phase)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise ProfileAbort(str(error)) from error


if __name__ == "__main__":  # pragma: no cover - runtime-only entry point
    raise SystemExit("Use profile_kernel.py; it validates the external review gate first.")
