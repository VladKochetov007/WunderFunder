from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def stats(path: Path) -> dict[str, int]:
    """Identify a file generation without loading checkpoint tensors."""
    value = path.stat()
    return {"inode": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def sha(path: Path) -> str:
    """Hash a checkpoint through a bounded streaming reader."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(arm: str, epoch: int) -> dict[str, object]:
    """Archive one stable continuation boundary without touching active training."""
    base = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
    run = f"rr18_r01_{arm}_f0_s1001_e6"
    source = base / "data/artifacts/runs" / run
    destination = base / "checkpoint_snapshots" / run / f"epoch{epoch}"
    names = (f"model_ep{epoch}.pt", "provenance.json", "training_state.pt")
    before = {name: stats(source / name) for name in names}
    if sum(row["size"] for row in before.values()) > 100_000_000:
        raise ValueError("unexpected checkpoint collection size")
    group = subprocess.check_output(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"],
        text=True,
    ).strip()
    memory = int((Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.current").read_text())
    if memory + 8_000_000_000 + 512_000_000 > 17_000_000_000:
        raise ValueError("insufficient conservative parent memory headroom")
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance["cumulative_updates"]["optimizer_completed_updates"] != (epoch + 1) * 14_920:
        raise ValueError("provenance does not belong to the requested boundary")
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name in names:
        shutil.copy2(source / name, destination / name)
        digest = sha(destination / name)
        if digest != sha(source / name):
            raise ValueError(f"source/copy mismatch: {name}")
        artifacts[name] = {
            "path": str(destination / name),
            "bytes": (destination / name).stat().st_size,
            "sha256": digest,
        }
    if artifacts[names[0]]["sha256"] != provenance["checkpoint_sha256"][names[0]]:
        raise ValueError("model hash differs from producer provenance")
    after = {name: stats(source / name) for name in names}
    if before != after:
        raise ValueError("source generation changed during collection")
    return {
        "kind": "continuation_epoch_checkpoint_collection",
        "created_utc": datetime.now(UTC).isoformat(),
        "run": run,
        "epoch_index": epoch,
        "source_directory": str(source),
        "snapshot_directory": str(destination),
        "source_stats_before": before,
        "source_stats_after": after,
        "artifacts": artifacts,
        "source_snapshot_hash_match": True,
        "producer_model_hash_matches": True,
        "resumed_from": provenance["resumed_from"],
        "parent_control_group": group,
        "parent_memory_at_collection_bytes": memory,
        "support_limits": {
            "cpu_quota": "25%",
            "memory_max_bytes": 512_000_000,
            "memory_swap_max_bytes": 0,
            "runtime_max_seconds": 120,
            "gpu": False,
        },
        "collector_sha256": sha(Path(__file__)),
        "scientific_status": "intermediate continuation boundary; no quality or GPU resume-parity result",
    }


def main() -> None:
    """Collect one explicitly selected rank1 continuation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--epoch", type=int, choices=range(2, 6), required=True)
    args = parser.parse_args()
    receipt = collect(args.arm, args.epoch)
    output = (
        Path("/home/grinchenkoeg/computer-vision-homework-20261018/receipts")
        / f"RANK01_{args.arm.upper()}_F0_S1001_E6_EP{args.epoch}_COLLECTION.json"
    )
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
