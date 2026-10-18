from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import collect_epoch_checkpoint as checkpoint

COLLECTOR_DEPENDENCY_SHA = "4bbeea5e7c6f5ff617dbff43cbb24c531686ce010b749e5988c8cd15171e014c"
ARTIFACT_NAMES = (
    "model_ep5.pt",
    "model.pt",
    "provenance.json",
    "training_state.pt",
    "history.json",
)


def archive_terminal(base: Path, arm: str, memory: int, group: str) -> dict[str, object]:
    """Preserve a successful fixed endpoint with stable attempt and log evidence.

    Args:
        base: Campaign root on physical storage.
        arm: The declared wide or deep F0/seed1001 arm.
        memory: Observed parent cgroup memory usage in bytes.
        group: Actual parent ControlGroup returned by systemd.

    Raises:
        ValueError: The endpoint, memory allowance or producer identity is invalid.
        FileExistsError: An archive already occupies the intended destination.
    """
    if arm not in ("wide", "deep"):
        raise ValueError("unknown rank1 arm")
    dependency = Path(checkpoint.__file__)
    if checkpoint.sha(dependency) != COLLECTOR_DEPENDENCY_SHA:
        raise ValueError("collector dependency hash differs")
    if memory < 0 or memory + 8_000_000_000 + 512_000_000 > 17_000_000_000:
        raise ValueError("insufficient conservative parent memory headroom")
    run = f"rr18_r01_{arm}_f0_s1001_e6"
    attempt = f"computer-vision-homework-r01-{arm}-f0s1001-ep25-v1"
    source = base / "data/artifacts/runs" / run
    destination = base / "checkpoint_snapshots" / run / "epoch5_terminal"
    paths = {name: source / name for name in ARTIFACT_NAMES}
    paths["attempt.json"] = base / "runtime/attempts" / f"{attempt}.json"
    paths["training.log"] = base / "runtime/logs" / f"{attempt}.log"
    before = {name: checkpoint.stats(path) for name, path in paths.items()}
    if sum(row["size"] for row in before.values()) > 100_000_000:
        raise ValueError("unexpected terminal archive size")
    if any(
        before[name]["size"] > 2_000_000
        for name in ("attempt.json", "provenance.json", "history.json")
    ):
        raise ValueError("unexpected JSON evidence size")
    terminal = json.loads(paths["attempt.json"].read_text())
    if (
        terminal["id"] != attempt
        or terminal["status"] != "finished"
        or terminal.get("exit_status") != 0
    ):
        raise ValueError("requested endpoint is not successfully terminal")
    provenance = json.loads(paths["provenance.json"].read_text())
    if provenance["cumulative_updates"]["optimizer_completed_updates"] != 89_520:
        raise ValueError("producer has not completed the fixed six-epoch schedule")
    history = json.loads(paths["history.json"].read_text())
    if [row["epoch"] for row in history] != list(range(6)):
        raise ValueError("final history does not contain epochs0 through5")
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, path in paths.items():
        target = destination / name
        shutil.copy2(path, target)
        digest = checkpoint.sha(target)
        if digest != checkpoint.sha(path):
            raise ValueError(f"source/copy mismatch: {name}")
        copied[name] = {"path": str(target), "bytes": target.stat().st_size, "sha256": digest}
    for name in ("model_ep5.pt", "model.pt"):
        if copied[name]["sha256"] != provenance["checkpoint_sha256"][name]:
            raise ValueError(f"model hash differs from producer: {name}")
    after = {name: checkpoint.stats(path) for name, path in paths.items()}
    if before != after:
        raise ValueError("terminal source generation changed during collection")
    return {
        "kind": "fixed_epoch5_terminal_checkpoint_collection",
        "created_utc": datetime.now(UTC).isoformat(),
        "run": run,
        "epoch_index": 5,
        "source_directory": str(source),
        "snapshot_directory": str(destination),
        "source_stats_before": before,
        "source_stats_after": after,
        "artifacts": {name: copied[name] for name in ARTIFACT_NAMES},
        "attempt_evidence": {
            **copied["attempt.json"],
            "id": attempt,
            "status": terminal["status"],
            "exit_status": terminal["exit_status"],
        },
        "log_evidence": copied["training.log"],
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
        "collector_sha256": checkpoint.sha(Path(__file__)),
        "collector_dependency_sha256": checkpoint.sha(dependency),
        "scientific_status": "fixed epoch5 endpoint archived; independent structural verification and prediction quality grading pending",
    }


def main() -> None:
    """Collect one successful endpoint without altering the running sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    args = parser.parse_args()
    base = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
    output = base / "receipts" / f"RANK01_{args.arm.upper()}_F0_S1001_E6_TERMINAL_COLLECTION.json"
    if output.exists():
        raise FileExistsError(output)
    group = subprocess.check_output(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"],
        text=True,
    ).strip()
    memory = int((Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.current").read_text())
    receipt = archive_terminal(base, args.arm, memory, group)
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
