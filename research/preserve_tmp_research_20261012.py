from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

INVENTORY = Path(
    "research/campaign_2026-10-12/research_archive_inventory_r1/TMP_PRESERVATION_INVENTORY_R1.json"
)
RECEIPT = INVENTORY.with_name("TMP_DURABLE_COPY_RECEIPT_R1.json")
DESTINATION = Path("artifacts/preservation_20261012/tmp").resolve()
BLOCK_BYTES = 1024 * 1024
FLUSH_BYTES = 16 * BLOCK_BYTES


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        read_bytes = 0
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
            read_bytes += len(block)
            if read_bytes >= FLUSH_BYTES:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                read_bytes = 0
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> dict[str, object]:
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".preserving")
    digest = hashlib.sha256()
    with source.open("rb") as src, partial.open("xb") as dst:
        written = 0
        while block := src.read(BLOCK_BYTES):
            dst.write(block)
            digest.update(block)
            written += len(block)
            if written >= FLUSH_BYTES:
                dst.flush()
                os.fsync(dst.fileno())
                os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                written = 0
        dst.flush()
        os.fsync(dst.fileno())
        os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed during preservation: {source}")
    expected = digest.hexdigest()
    if partial.stat().st_size != before.st_size or digest_file(partial) != expected:
        raise RuntimeError(f"durable copy verification failed: {source}")
    shutil.copystat(source, partial)
    os.replace(partial, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": before.st_size,
        "sha256": expected,
    }


def main() -> None:
    if INVENTORY.stat().st_size > 8 * 1024 * 1024:
        raise RuntimeError("inventory exceeds fixed input bound")
    inventory = json.loads(INVENTORY.read_text())
    roots = inventory["roots"]
    expected_bytes = sum(root["bytes"] for root in roots)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DESTINATION).free < expected_bytes + 4 * 1024**3:
        raise RuntimeError("insufficient durable disk space")
    rows: list[dict[str, object]] = []
    links: list[dict[str, str]] = []
    for root in roots:
        for name in root["regular_files"]:
            source = Path(name)
            if not source.is_relative_to("/tmp") or source.is_symlink():
                raise RuntimeError(f"unexpected inventory source: {source}")
            target = DESTINATION / source.relative_to("/tmp")
            if target.exists():
                raise RuntimeError(f"create-only preservation target exists: {target}")
            rows.append(copy_file(source, target))
        for entry in root.get("symlinks", []):
            source = Path(entry["path"])
            target = DESTINATION / source.relative_to("/tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            original_target = entry["target"]
            resolved = source.resolve()
            # Rewrite links into preserved roots to their durable counterpart.
            new_target = str(resolved)
            if resolved.is_relative_to("/tmp") and any(
                resolved.is_relative_to(Path(item["source"])) for item in roots
            ):
                new_target = str(DESTINATION / resolved.relative_to("/tmp"))
            target.symlink_to(new_target)
            links.append(
                {
                    "source": str(source),
                    "destination": str(target),
                    "original_target": original_target,
                    "preserved_target": new_target,
                }
            )
    receipt = {
        "recorded_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "all_inventoried_regular_files_copied_and_sha256_verified",
        "durable_root": str(DESTINATION),
        "regular_files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "symlinks": links,
        "files": rows,
        "originals_retained_until_current_run_owner_finishes": True,
        "model_data_binary_payloads_git_ignored": True,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key not in {"files", "symlinks"}}
        )
    )


if __name__ == "__main__":
    main()
