from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    """Record real service containment without training or dataset access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group_name = Path("/proc/self/cgroup").read_text().splitlines()[0].split(":")[-1]
    group = Path("/sys/fs/cgroup") / group_name.lstrip("/")
    if "/wnn-campaign.slice/wnn-c04-" not in group_name:
        raise RuntimeError("smoke is outside the declared scientific service")
    result = {
        "utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "cgroup": group_name,
        "worker_ram_max": int((group / "memory.max").read_text()),
        "worker_swap_max": int((group / "memory.swap.max").read_text()),
        "parent_ram_max": int((group.parent / "memory.max").read_text()),
        "parent_ram_high": int((group.parent / "memory.high").read_text()),
        "scientific_fits": 0,
        "dataset_rows_loaded": 0,
    }
    if result["worker_swap_max"] != 0 or result["parent_ram_max"] > 40_000_000_000:
        raise RuntimeError("resource containment differs from the contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
