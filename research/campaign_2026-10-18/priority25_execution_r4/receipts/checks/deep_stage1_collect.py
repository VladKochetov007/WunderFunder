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
    """Preserve terminal deep stage1 without modifying the following fit."""
    base = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
    run = "rr18_r01_deep_f0_s1001"
    source = base / "data/artifacts/runs" / run
    destination = base / "checkpoint_snapshots" / run / "stage1_epoch1"
    names = ("model_ep0.pt", "model_ep1.pt", "provenance.json", "training_state.pt", "history.json")
    attempt = "computer-vision-homework-r01-deep-f0s1001-ep01-v1"
    attempt_path = base / "runtime/attempts" / f"{attempt}.json"
    terminal = json.loads(attempt_path.read_text())
    if terminal["status"] != "finished" or terminal["exit_status"] != 0:
        raise ValueError("deep stage1 is not successfully terminal")
    evidence = (attempt_path, base / "runtime/logs" / f"{attempt}.log")
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
        "kind": "terminal_stage1_checkpoint_collection",
        "attempt": attempt,
        "attempt_status": terminal["status"],
        "attempt_exit_status": terminal["exit_status"],
        "finished_utc": terminal["finished_utc"],
        "created_utc": datetime.now(UTC).isoformat(),
        "run": run,
        "epoch_indices": [0, 1],
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
        "attempt_evidence": {
            f.name: {"path": str(f), "bytes": f.stat().st_size, "sha256": sha(f)} for f in evidence
        },
        "scientific_status": "terminal two-epoch stage1; six-epoch endpoint and quality comparison pending",
    }
    output = base / "receipts/RANK01_DEEP_F0_S1001_STAGE1_COLLECTION.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
