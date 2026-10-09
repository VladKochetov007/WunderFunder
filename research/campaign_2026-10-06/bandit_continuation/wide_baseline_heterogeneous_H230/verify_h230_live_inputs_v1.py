"""Refresh H230's exact input identities before the one-shot cache stage.

This read-only gate hashes the raw parquet and portable source files in bounded
chunks, advising completed pages away. It never opens model arrays or writes a
cache. A durable receipt is written before work and after each input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DEFAULT = Path(__file__).resolve().parents[4]
H230_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230"
)
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
HASH_BLOCK_BYTES = 8 << 20
PROGRESS_INTERVAL_BYTES = 256 << 20
MINIMUM_FREE_BYTES = 40_216_345_600
CACHE_RESERVE_BYTES = 6_500_000_000
CONCURRENT_SUPPORT_RESERVE_BYTES = 2_000_000_000
PARENT_HIGH_BYTES = 18_000_000_000
WORKER_MAX_BYTES = 1_500_000_000

EXPECTED_PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_FILES: dict[str, tuple[int, str]] = {
    "datasets/train.parquet": (
        29_240_219_011,
        "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    ),
    "datasets/valid.parquet": (
        5_615_509_128,
        "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    ),
    "cache/chain_folds.npz": (
        550_072,
        "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    ),
    "cache/chains.npz": (
        650_142,
        "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
    ),
    "cache/train_targets.npz": (
        1_909_335_790,
        "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    ),
    "cache/valid_targets.npz": (
        337_186_198,
        "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    ),
}
FROZEN_ARTIFACTS: dict[str, str] = {
    "H230_SOURCE_HASH_BUNDLE_V2.json": "c9f4235994e9fc85f0f732fd74852e1c28b1d5c7efc8a1ebefbc71f6dc673340",
    "H230_SOURCE_SEAL_PROGRESS_V4.json": "b21ac766f66dfe662b0f9315462257bb6b850dc50fac500c7d95c4070f33c411",
    "H230_ACTUAL_SOURCE_REVIEW_TARGETS_V1.json": "14b86be4d1e9c4944af1e8dcfd9d3d63c73ba0edd97a510d9c7be3c987a3485a",
    "H230_ACTUAL_SOURCE_INDEPENDENT_REVIEW_V1_20261007_R1.json": "b7272d096d793ccf9252e98fcbbac495a51c202b916c1a3fa1d85018ec9d9124",
}
EXPECTED_ROW_GROUPS = {
    "train": {
        "count": 10_606,
        "fit_count": 10_535,
        "evaluation_count": 0,
        "ordered_sha256": "658498ef9fb3ad7f4dab9dea249da809168be595171099406795f72b63554fbe",
    },
    "valid": {
        "count": 1_873,
        "fit_count": 1_405,
        "evaluation_count": 468,
        "ordered_sha256": "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7",
    },
}


class IdentityError(RuntimeError):
    """Raised when a current H230 input or resource identity gate fails."""


def _sha256_small(path: Path) -> str:
    """Hash a small manifest or receipt without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(path: Path) -> dict[str, int]:
    """Capture file identity fields used to detect mutation during hashing."""
    info = path.stat()
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
        "mode": int(info.st_mode),
    }


def hash_with_page_advice(
    path: Path, progress: dict[str, Any], checkpoint: Callable[[], None]
) -> str:
    """Hash exact bytes with bounded memory and fail-closed page advice."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if not callable(fadvise) or dontneed is None or page_size <= 0:
        raise IdentityError("POSIX_FADV_DONTNEED support is required for raw hashing")
    if HASH_BLOCK_BYTES % page_size:
        raise IdentityError("hash block size must be page aligned")

    before = _stat_identity(path)
    digest = hashlib.sha256()
    bytes_hashed = 0
    bytes_advised = 0
    advice_calls = 0
    next_update = PROGRESS_INTERVAL_BYTES
    progress.update(
        status="hashing",
        path=str(path),
        file_before=before,
        bytes_hashed=0,
        bytes_advised=0,
        advice_calls=0,
        advice="POSIX_FADV_DONTNEED hint; digest input/order unchanged",
    )
    try:
        with path.open("rb", buffering=0) as stream:
            while block := stream.read(HASH_BLOCK_BYTES):
                digest.update(block)
                bytes_hashed += len(block)
                advised_end = bytes_hashed - bytes_hashed % page_size
                if advised_end > bytes_advised:
                    fadvise(stream.fileno(), bytes_advised, advised_end - bytes_advised, dontneed)
                    advice_calls += 1
                    bytes_advised = advised_end
                progress.update(
                    bytes_hashed=bytes_hashed,
                    bytes_advised=bytes_advised,
                    advice_calls=advice_calls,
                )
                if bytes_hashed >= next_update:
                    next_update += PROGRESS_INTERVAL_BYTES
                    checkpoint()
    except (OSError, NotImplementedError) as error:
        progress.update(
            status="hash_failed",
            bytes_hashed=bytes_hashed,
            bytes_advised=bytes_advised,
            advice_calls=advice_calls,
            error=f"{type(error).__name__}: {error}",
        )
        checkpoint()
        raise IdentityError(f"hash/advice failed for {path}") from error

    after = _stat_identity(path)
    progress.update(
        status="hashed",
        bytes_hashed=bytes_hashed,
        bytes_advised=bytes_advised,
        advice_calls=advice_calls,
        file_after=after,
        observed_sha256=digest.hexdigest(),
    )
    checkpoint()
    if before != after or bytes_hashed != before["bytes"]:
        raise IdentityError(f"input identity changed during hash: {path}")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object, *, replace: bool) -> None:
    """Durably write one small JSON receipt with an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() and not replace:
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_cgroup(path: Path) -> dict[str, Any]:
    """Read a bounded set of cgroup-v2 memory fields."""
    result: dict[str, Any] = {}
    for name in (
        "memory.current",
        "memory.peak",
        "memory.high",
        "memory.max",
        "memory.swap.current",
    ):
        try:
            raw = (path / name).read_text(encoding="ascii").strip()
            result[name] = raw if raw == "max" else int(raw)
        except OSError as error:
            result[name] = {"error": f"{type(error).__name__}: {error}"}
    for name, selected in (
        ("memory.stat", {"anon", "file", "inactive_file", "active_file"}),
        ("memory.events", {"high", "max", "oom", "oom_kill", "oom_group_kill"}),
    ):
        try:
            values = dict(line.split() for line in (path / name).read_text().splitlines())
            result[name] = {key: int(value) for key, value in values.items() if key in selected}
        except (OSError, ValueError) as error:
            result[name] = {"error": f"{type(error).__name__}: {error}"}
    return result


def _resource_snapshot(root: Path) -> dict[str, Any]:
    """Record process, worker, campaign-parent, and disk headroom."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    groups = [
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    ]
    if len(groups) != 1:
        raise IdentityError("cannot resolve a unique worker cgroup")
    worker_cgroup = Path("/sys/fs/cgroup") / groups[0].lstrip("/")
    campaign_cgroup = next(
        (
            parent
            for parent in (worker_cgroup, *worker_cgroup.parents)
            if parent.name == "wnn-campaign.slice"
        ),
        None,
    )
    if campaign_cgroup is None:
        raise IdentityError("worker is not running inside wnn-campaign.slice")
    disk = shutil.disk_usage(root / "cache")
    return {
        "captured_utc": datetime.now(UTC).isoformat(),
        "process_ru_maxrss_bytes": int(max_rss * 1024),
        "worker_cgroup": str(worker_cgroup),
        "worker_memory": _read_cgroup(worker_cgroup),
        "campaign_cgroup": str(campaign_cgroup),
        "campaign_memory": _read_cgroup(campaign_cgroup),
        "disk_total_bytes": int(disk.total),
        "disk_used_bytes": int(disk.used),
        "disk_free_bytes": int(disk.free),
    }


def _verify_frozen_source(root: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    """Verify one previously reviewed, small H230 identity artifact."""
    path = root / H230_RELATIVE / name
    observed = _sha256_small(path)
    return {
        "path": str(path),
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "matched": observed == expected_sha256,
    }


def verify_inputs(root: Path, receipt_path: Path, progress_path: Path) -> dict[str, Any]:
    """Hash and validate the portable data inputs bound to the sealed source."""
    if receipt_path.exists() or progress_path.exists():
        raise FileExistsError("refusing to overwrite an H230 live-identity receipt")
    cache_output = root / "cache/bd26_h230_w384_family_v1"
    initial = _resource_snapshot(root)
    receipt: dict[str, Any] = {
        "id": "H230-live-input-identity-v1",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "read-only data/input identity and resource gate; no cache/profile/fit",
        "worker": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_small(Path(__file__).resolve()),
        },
        "expected": {
            "portable_manifest_sha256": EXPECTED_PORTABLE_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "portable_files": {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in EXPECTED_FILES.items()
            },
            "frozen_h230_artifacts": FROZEN_ARTIFACTS,
            "row_groups": EXPECTED_ROW_GROUPS,
            "disk_free_floor_bytes": MINIMUM_FREE_BYTES,
            "cache_memory_reservation_bytes": CACHE_RESERVE_BYTES,
            "concurrent_support_reservation_bytes": CONCURRENT_SUPPORT_RESERVE_BYTES,
            "parent_memory_high_bytes": PARENT_HIGH_BYTES,
            "worker_memory_max_bytes": WORKER_MAX_BYTES,
            "cache_output_absent": True,
        },
        "observed": {},
        "resource_before": initial,
        "cache_output_exists_before": cache_output.exists(),
    }

    def checkpoint() -> None:
        receipt["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
        receipt["resource_during"] = _resource_snapshot(root)
        _atomic_json(receipt_path, receipt, replace=True)

    receipt["started_utc"] = datetime.now(UTC).isoformat()
    _atomic_json(receipt_path, receipt, replace=False)
    progress: dict[str, Any] = {"inputs": {}}
    _atomic_json(
        progress_path, {"id": "H230-live-input-hash-progress-v1", "inputs": {}}, replace=False
    )

    def save_progress() -> None:
        _atomic_json(
            progress_path,
            {"id": "H230-live-input-hash-progress-v1", "inputs": progress["inputs"]},
            replace=True,
        )

    try:
        if cache_output.exists():
            raise IdentityError("cache output path already exists")
        initial_parent = initial["campaign_memory"]
        initial_parent_current = initial_parent.get("memory.current")
        initial_parent_high = initial_parent.get("memory.high")
        initial_disk_free = int(initial["disk_free_bytes"])
        if (
            not isinstance(initial_parent_current, int)
            or not isinstance(initial_parent_high, int)
            or initial_parent_current + CACHE_RESERVE_BYTES + CONCURRENT_SUPPORT_RESERVE_BYTES
            > initial_parent_high
        ):
            raise IdentityError(
                "initial parent headroom cannot cover cache plus concurrent support"
            )
        if initial_disk_free < MINIMUM_FREE_BYTES:
            raise IdentityError("initial free disk is below cache payload plus 25 GiB floor")

        portable_path = root / PORTABLE_RELATIVE
        split_path = root / SPLIT_RELATIVE
        portable_sha = _sha256_small(portable_path)
        split_sha = _sha256_small(split_path)
        receipt["observed"]["portable_manifest"] = {
            "path": str(portable_path),
            "sha256": portable_sha,
            "bytes": portable_path.stat().st_size,
        }
        receipt["observed"]["split_manifest"] = {
            "path": str(split_path),
            "sha256": split_sha,
            "bytes": split_path.stat().st_size,
        }
        if portable_sha != EXPECTED_PORTABLE_SHA256 or split_sha != EXPECTED_SPLIT_SHA256:
            raise IdentityError("portable registry or fold manifest hash mismatch")

        portable = json.loads(portable_path.read_text(encoding="utf-8"))
        if portable.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
            raise IdentityError("portable registry is bound to another split manifest")
        sources = portable.get("sources")
        if not isinstance(sources, dict) or set(sources) != set(EXPECTED_FILES):
            raise IdentityError("portable registry input set differs from the frozen expected set")

        frozen_artifacts = {
            name: _verify_frozen_source(root, name, expected)
            for name, expected in FROZEN_ARTIFACTS.items()
        }
        receipt["observed"]["frozen_h230_artifacts"] = frozen_artifacts
        if not all(value["matched"] for value in frozen_artifacts.values()):
            raise IdentityError("sealed H230 source or independent-review artifact changed")
        bundle = json.loads((root / H230_RELATIVE / "H230_SOURCE_HASH_BUNDLE_V2.json").read_text())
        groups = bundle.get("actual_target_row_groups")
        if groups != EXPECTED_ROW_GROUPS:
            raise IdentityError("sealed ordered row-group or fold membership identity changed")
        receipt["observed"]["source_group_binding"] = {
            "source_bundle_sha256": FROZEN_ARTIFACTS["H230_SOURCE_HASH_BUNDLE_V2.json"],
            "fold": bundle["fold"],
            "target_row_groups": groups,
            "evaluation_rows_exposed": 468,
        }

        for relative, (expected_size, expected_sha) in EXPECTED_FILES.items():
            path = root / relative
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root.resolve())
            except (OSError, ValueError) as error:
                raise IdentityError(
                    f"input path is missing or escaped project root: {relative}"
                ) from error
            source_record = sources[relative]
            if source_record.get("sha256") != expected_sha:
                raise IdentityError(f"portable registry expected hash differs for {relative}")
            before = _stat_identity(path)
            if before["bytes"] != expected_size:
                receipt["observed"].setdefault("inputs", {})[relative] = {
                    "expected_bytes": expected_size,
                    "observed_before": before,
                    "expected_sha256": expected_sha,
                    "status": "size_mismatch_before_hash",
                }
                checkpoint()
                raise IdentityError(f"input byte size mismatch for {relative}")

            file_progress: dict[str, Any] = {
                "expected_bytes": expected_size,
                "expected_sha256": expected_sha,
                "status": "starting",
            }

            def progress_checkpoint(
                relative_name: str = relative,
                record: dict[str, Any] = file_progress,
            ) -> None:
                progress["inputs"][relative_name] = record.copy()
                save_progress()
                checkpoint()

            observed_sha = hash_with_page_advice(path, file_progress, progress_checkpoint)
            observed_input = {
                "path": str(path),
                "expected_bytes": expected_size,
                "expected_sha256": expected_sha,
                "observed_sha256": observed_sha,
                "before": before,
                "after": _stat_identity(path),
                "bytes_hashed": file_progress["bytes_hashed"],
                "bytes_advised": file_progress["bytes_advised"],
                "advice_calls": file_progress["advice_calls"],
                "status": "matched" if observed_sha == expected_sha else "hash_mismatch",
            }
            receipt["observed"].setdefault("inputs", {})[relative] = observed_input
            progress["inputs"][relative] = file_progress.copy()
            save_progress()
            checkpoint()
            if observed_sha != expected_sha:
                raise IdentityError(f"input SHA-256 mismatch for {relative}")

        final = _resource_snapshot(root)
        parent_memory = final["campaign_memory"]
        worker_memory = final["worker_memory"]
        disk_free = int(final["disk_free_bytes"])
        parent_current = parent_memory.get("memory.current")
        parent_high = parent_memory.get("memory.high")
        worker_max = worker_memory.get("memory.max")
        worker_peak = worker_memory.get("memory.peak")
        if not isinstance(parent_current, int) or not isinstance(parent_high, int):
            raise IdentityError("campaign parent memory current/high unavailable")
        if not isinstance(worker_max, int) or not isinstance(worker_peak, int):
            raise IdentityError("worker cgroup memory max/peak unavailable")
        if worker_max > WORKER_MAX_BYTES or worker_peak > WORKER_MAX_BYTES:
            raise IdentityError("worker memory cgroup exceeded the 1.5 GB verification cap")
        if parent_current + CACHE_RESERVE_BYTES + CONCURRENT_SUPPORT_RESERVE_BYTES > parent_high:
            raise IdentityError("live parent headroom cannot cover cache plus concurrent support")
        if disk_free < MINIMUM_FREE_BYTES:
            raise IdentityError("free disk is below cache payload plus 25 GiB floor")
        initial_events = initial["campaign_memory"].get("memory.events", {})
        final_events = parent_memory.get("memory.events", {})
        for event in ("oom", "oom_kill", "oom_group_kill"):
            if (
                not isinstance(initial_events, dict)
                or not isinstance(final_events, dict)
                or final_events.get(event) != initial_events.get(event)
            ):
                raise IdentityError(
                    f"campaign parent {event} counter changed during identity check"
                )

        receipt.update(
            status="passed",
            finished_utc=datetime.now(UTC).isoformat(),
            resource_after=final,
            admission_calculation={
                "parent_current_bytes": parent_current,
                "cache_reserve_bytes": CACHE_RESERVE_BYTES,
                "concurrent_support_reserve_bytes": CONCURRENT_SUPPORT_RESERVE_BYTES,
                "combined_projected_parent_bytes": parent_current
                + CACHE_RESERVE_BYTES
                + CONCURRENT_SUPPORT_RESERVE_BYTES,
                "parent_high_bytes": parent_high,
                "free_disk_bytes": disk_free,
                "required_free_disk_bytes": MINIMUM_FREE_BYTES,
            },
            cache_output_absent_after=not cache_output.exists(),
            cache_output_created=False,
        )
        _atomic_json(receipt_path, receipt, replace=True)
        return receipt
    except Exception as error:
        receipt.update(
            status="failed",
            finished_utc=datetime.now(UTC).isoformat(),
            error=f"{type(error).__name__}: {error}",
            resource_after=_resource_snapshot(root),
            cache_output_created=cache_output.exists(),
        )
        _atomic_json(receipt_path, receipt, replace=True)
        raise


def main() -> int:
    """Run the one-shot H230 live identity gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--receipt", type=Path, default=H230_RELATIVE / "H230_LIVE_INPUT_IDENTITY_V1.json"
    )
    parser.add_argument(
        "--progress", type=Path, default=H230_RELATIVE / "H230_LIVE_INPUT_HASH_PROGRESS_V1.json"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = args.receipt if args.receipt.is_absolute() else root / args.receipt
    progress = args.progress if args.progress.is_absolute() else root / args.progress
    try:
        verify_inputs(root, receipt, progress)
    except Exception as error:
        print(f"H230 live identity gate failed: {type(error).__name__}: {error}")
        return 1
    print(f"H230 live identity gate passed: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
