from __future__ import annotations

import json
import subprocess
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
CONTINUATION = ROOT / "research/campaign_2026-10-03/continuation"
PARENT = Path(
    "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice"
)


def events(path: Path) -> dict[str, int]:
    """Read existing kernel resource counters, without changing limits."""
    return {
        name: int(value)
        for name, value in (
            line.split() for line in (path / "memory.events").read_text().splitlines()
        )
    }


def main() -> None:
    """Record current terminal units, guards and OOM counters read-only."""
    watch = json.loads((CONTINUATION / "watch_status.json").read_text())
    parent, gpu = events(PARENT), events(PARENT / "wnn-campaign-gpu.slice")
    units = {}
    for name in (
        "wnn-codex-research-queue-v2.service",
        "wnn-codex-fold23-scores.service",
        "wnn-top1-c20261003_capture_v4_default_2120r.service",
    ):
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                name,
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "Result",
                "-p",
                "MainPID",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        units[name] = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    barriers = {
        name: (ROOT / name).exists()
        for name in (
            "research/campaign_2026-10-03/continuation/RESOURCE_FAILURE",
            "research/campaign_2026-09-29/STOP_NEW_JOBS",
            "research/campaign_2026-10-01/QUEUE_PAUSE",
        )
    }
    assert not any(barriers.values())
    assert (CONTINUATION / "READY").exists() and not watch["barrier"]
    assert not watch["monitoring_degraded"]
    assert parent["oom"] == parent["oom_kill"] == 11
    assert gpu["oom"] == gpu["oom_kill"] == 0
    for name in ("wnn-codex-research-queue-v2.service", "wnn-codex-fold23-scores.service"):
        assert units[name]["ActiveState"] == "inactive" and units[name]["Result"] == "success"
    job_unit = units["wnn-top1-c20261003_capture_v4_default_2120r.service"]
    if job_unit.get("LoadState") != "not-found":
        assert job_unit["ActiveState"] == "inactive" and job_unit["Result"] == "success"
    result = {
        "all_pass": True,
        "watch": watch,
        "units": units,
        "parent_memory_events": parent,
        "gpu_family_memory_events": gpu,
        "barriers": barriers,
        "parent_oom_comparison_basis": "controller's previously reported cumulative11; not attributed to this successful job",
        "GPU_probe_or_launch": False,
    }
    (OWN / "current_guard_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "current_guards_pass": True,
                "watch_time": watch["time"],
                "parent_oom": parent["oom"],
                "gpu_oom": gpu["oom"],
                "active_gpu_jobs_at_watch_snapshot": watch["active_gpu_jobs"],
            }
        )
    )


if __name__ == "__main__":
    main()
