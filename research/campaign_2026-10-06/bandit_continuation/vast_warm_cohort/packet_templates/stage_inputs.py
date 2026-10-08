"""Startup validator and input materializer for the warm smoke packet.

Runs after the supervisor has SHA-verified every rsync upload and the signed
prefetch has SHA-verified the raw Parquet files. This script checks the exact
frozen input set, the fold-0 split manifest and its 468-group evaluation panel,
copies containment-sensitive inputs into the project tree and writes a receipt.
It never reads a full Parquet file and never prints signed URLs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PARQUET_PATHS = frozenset({"datasets/train.parquet", "datasets/valid.parquet"})
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def _digest_copy(source: Path, destination: Path, expected_sha: str, expected_size: int) -> str:
    """Copy a regular staged file through a temporary name and verify its bytes."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace a packet path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while block := reader.read(8 << 20):
            digest.update(block)
            writer.write(block)
            copied += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if copied != expected_size or digest.hexdigest() != expected_sha:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"staged bytes differ from the frozen identity: {source}")
    os.replace(temporary, destination)
    return digest.hexdigest()


def _digest_only(source: Path, expected_sha: str, expected_size: int) -> str:
    """Hash a staged regular file without copying it."""
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader:
        while block := reader.read(8 << 20):
            digest.update(block)
            copied += len(block)
    if copied != expected_size or digest.hexdigest() != expected_sha:
        raise ValueError(f"staged bytes differ from the frozen identity: {source}")
    return digest.hexdigest()


def main() -> int:
    """Validate the frozen inputs, materialize the project tree and write a receipt.

    With --verify-only the same identity, size and split checks run against the
    staged input root without copying or linking anything; it is used offline to
    rehearse the startup validator on the producer-side files before admission.
    """
    verify_only = "--verify-only" in sys.argv[1:]
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    input_root = Path(os.environ["WNN_INPUT_DIR"])
    spec = json.loads(Path("worker_spec.json").read_text(encoding="utf-8"))
    project = Path("project")
    receipt: dict[str, object] = {"status": "failed", "inputs": [], "split_check": None}
    try:
        expected = {row["path"]: row for row in spec["inputs"]}
        if len(expected) != spec["expected_input_count"]:
            raise ValueError("frozen input set differs from its expected count")
        for relative, row in sorted(expected.items()):
            source = input_root / relative
            if source.is_symlink() or not source.is_file():
                raise FileNotFoundError(f"staged input is absent or linked: {relative}")
            if source.stat().st_size != row["size_bytes"]:
                raise ValueError(f"staged input has the wrong size: {relative}")
            destination = project / relative
            if verify_only and relative in PARQUET_PATHS:
                observed = row["sha256"]
                mode = "verify_only_parquet_size_checked_prefetch_sha_verified"
            elif verify_only:
                observed = _digest_only(source, row["sha256"], row["size_bytes"])
                mode = "verify_only"
            elif relative in PARQUET_PATHS:
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f"refusing to replace a packet path: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source.resolve())
                mode = "raw_parquet_symlink_size_checked_prefetch_sha_verified"
                observed = row["sha256"]
            else:
                observed = _digest_copy(source, destination, row["sha256"], row["size_bytes"])
                mode = "regular_copy_sha256_verified"
            receipt["inputs"].append(  # type: ignore[union-attr]
                {"path": relative, "mode": mode, "size_bytes": row["size_bytes"], "sha256": observed}
            )
        if SPLIT_PATH not in expected:
            raise ValueError("fold-0 split manifest is absent from the frozen input set")
        sys.path.insert(0, str(project / "src"))
        from wnn.splits import load_campaign_split  # noqa: E402

        # Dataset identities are checked against the portable registry, which the
        # run script exports; cache and split identities resolve under this root.
        root = input_root if verify_only else project
        split = load_campaign_split(root / SPLIT_PATH, root)
        if split.fold != 0 or len(split.evaluation_groups) != 468:
            raise ValueError("split manifest is not the frozen fold-0 468-group panel")
        receipt["split_check"] = {
            "path": SPLIT_PATH,
            "fold": split.fold,
            "evaluation_groups": len(split.evaluation_groups),
        }
        receipt["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - receipt must capture every startup failure
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        temporary = output / "input_receipt.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output / "input_receipt.json")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
