from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def stats(path: Path) -> dict[str, int]:
    """Identify a source generation without reading checkpoint content."""
    value = path.stat()
    return {"inode": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def sha(path: Path) -> str:
    """Hash a bounded checkpoint through a streaming reader."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Preserve the first deep epoch without modifying the running fit."""
    base = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
    run = "rr18_r01_deep_f0_s1001"
    source = base / "data/artifacts/runs" / run
    destination = base / "checkpoint_snapshots" / run / "epoch0"
    names = ("model_ep0.pt", "provenance.json", "training_state.pt")
    before = {name: stats(source / name) for name in names}
    if sum(row["size"] for row in before.values()) > 100_000_000:
        raise ValueError("unexpected checkpoint collection size")
    group = subprocess.check_output(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"],
        text=True,
    ).strip()
    cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
    memory = int((cgroup / "memory.current").read_text())
    if memory + 8_000_000_000 + 512_000_000 > 17_000_000_000:
        raise ValueError("insufficient conservative parent memory headroom")
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
    after = {name: stats(source / name) for name in names}
    if before != after:
        raise ValueError("source generation changed during collection")
    receipt = {
        "kind": "epoch_boundary_checkpoint_collection",
        "created_utc": datetime.now(UTC).isoformat(),
        "run": run,
        "epoch_index": 0,
        "source_directory": str(source),
        "snapshot_directory": str(destination),
        "source_stats_before": before,
        "source_stats_after": after,
        "artifacts": artifacts,
        "source_snapshot_hash_match": True,
        "parent_control_group": group,
        "parent_memory_at_collection_bytes": memory,
        "support_limits": {
            "cpu_quota": "25%",
            "memory_max_bytes": 512_000_000,
            "runtime_max_seconds": 120,
            "gpu": False,
        },
        "collector_sha256": sha(Path(__file__)),
        "scientific_status": "partial fit; no endpoint predictions or quality comparison",
    }
    output = base / "receipts/RANK01_DEEP_F0_S1001_EP0_COLLECTION.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
