from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from scripts.bandit_resources import ROOT, atomic_json, authorized_cpu_cap, read_json
else:
    from bandit_resources import ROOT, atomic_json, authorized_cpu_cap, read_json


def memory_snapshot(cgroup: Path) -> dict[str, int]:
    """Observe charged memory; do not alter any limit or event counter."""
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    return {
        "current": int((cgroup / "memory.current").read_text()),
        "file": int(stat["file"]),
        "anon": int(stat["anon"]),
    }


def reclaim_completed_export(
    runtime: Path,
    predecessor: Path,
    receipt: Path,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> None:
    """Reclaim bounded file cache after a successful export, under the GPU lock."""
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite {receipt}")
    record = read_json(predecessor)
    if (
        record.get("status") != "finished"
        or record.get("exit_status") != 0
        or record.get("kind") != "gpu"
        or not {"scripts/predict_run.py", str(ROOT / "scripts/predict_run.py")}.intersection(
            record.get("command", [])
        )
    ):
        raise ValueError("predecessor must be a successfully completed guarded export")
    cgroup = cgroup_root / record["parent_properties_at_start"]["ControlGroup"].lstrip("/")
    if record["parent_slice"] != "wnn-campaign.slice" or cgroup.name != "wnn-campaign.slice":
        raise ValueError("only the campaign parent slice may be reclaimed")
    if (
        int((cgroup / "memory.max").read_text()) > 20_000_000_000
        or int((cgroup / "memory.high").read_text()) > 18_000_000_000
        or int((cgroup / "memory.swap.max").read_text()) != 0
    ):
        raise ValueError("campaign memory limits differ from the authorized bounds")
    quota, period = map(int, (cgroup / "cpu.max").read_text().split())
    if quota > authorized_cpu_cap() * period:
        raise ValueError("campaign CPU limit exceeds the current authorized cores")
    with (runtime / "gpu.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = memory_snapshot(cgroup)
        requested = min(
            6_000_000_000,
            max(0, before["current"] - 8_500_000_000),
            max(0, before["file"] - 6_000_000_000),
        )
        reclaim_warning = None
        if requested:
            try:
                (cgroup / "memory.reclaim").write_text(f"{requested} swappiness=0\n")
            except OSError as error:
                if error.errno != errno.EAGAIN:
                    raise
                reclaim_warning = str(error)
        after = memory_snapshot(cgroup)
        ready = after["current"] <= 10_000_000_000
        atomic_json(
            receipt,
            {
                "status": "ready" if ready else "needs_review",
                "observed_utc": datetime.now(UTC).isoformat(),
                "predecessor": str(predecessor),
                "predecessor_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
                "cgroup": str(cgroup),
                "before": before,
                "after": after,
                "requested_file_reclaim_bytes": requested,
                "kernel_partial_reclaim_warning": reclaim_warning,
                "swappiness": 0,
                "limits_changed": False,
                "next_export_reserve_bytes": 8_000_000_000,
            },
        )
        if not ready:
            raise RuntimeError("file reclaim did not restore export admission headroom")


def main() -> None:
    """Read controller-supplied paths and perform one bounded cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    reclaim_completed_export(args.runtime, args.predecessor, args.receipt)


if __name__ == "__main__":
    main()
