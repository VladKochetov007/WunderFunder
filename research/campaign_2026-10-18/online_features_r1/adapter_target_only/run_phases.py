"""Run the frozen target-only fit, export and official score serially."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]


def record(event: str, **fields: object) -> None:
    """Append a timestamped phase receipt without replacing prior evidence."""
    with (PACKET / "progress.jsonl").open("a") as output:
        output.write(
            json.dumps({"event": event, "utc": datetime.now(UTC).isoformat(), **fields}) + "\n"
        )


def main() -> None:
    """Execute only the declared commands within one admitted GPU reservation."""
    for key in (
        "WNN_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "ARROW_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ.update(WNN_LOADER_WORKERS="0", WNN_LOADER_RAM_GB="1.5")
    recipe = json.loads((PACKET / "execution_phases.json").read_text())
    for phase in recipe["phases"]:
        name = phase["name"]
        record("start", phase=name, command=phase["command"])
        with (PACKET / f"{name}.log").open("x") as output:
            result = subprocess.run(
                phase["command"],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=phase["timeout_seconds"],
                check=False,
            )
        record("end", phase=name, returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f"{name} failed with return code {result.returncode}")
    record("complete")


if __name__ == "__main__":
    main()
