"""Run one bounded warm feature/control comparison, then export and grade it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def record(event: str, **fields: object) -> None:
    """Append phase receipts without replacing previous evidence."""
    with (PACKET / "progress.jsonl").open("a") as output:
        output.write(
            json.dumps({"event": event, "timestamp_utc": datetime.now(UTC).isoformat(), **fields})
            + "\n"
        )


def execute(arm: str, phase: str, command: list[str], timeout: int) -> None:
    """Execute one serial phase within the shared task resource boundary."""
    record("phase_start", arm=arm, phase=phase, command=command)
    with (PACKET / f"{arm}.{phase}.log").open("x") as output:
        result = subprocess.run(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    record("phase_end", arm=arm, phase=phase, returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f"{arm}/{phase} failed: {result.returncode}")


def main() -> None:
    """Fit the feature arm first to ensure the requested model is delivered."""
    os.environ.update(
        {
            "WNN_THREADS": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "1.5",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    recipe = json.loads((PACKET / "recipe.json").read_text())
    split = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    for arm in ("feature", "control"):
        execute(arm, "train", recipe["arms"][arm], 2400)
    for arm in ("feature", "control"):
        name = f"geometry_quick_f0_s3_{arm}"
        execute(
            arm,
            "predict",
            [
                sys.executable,
                "-u",
                "scripts/predict_run.py",
                name,
                "valid",
                "--campaign-split-manifest",
                split,
                "--campaign-evaluation-only",
                "--device",
                "cuda",
                "--epoch",
                "0",
            ],
            600,
        )
        execute(
            arm,
            "score",
            [
                sys.executable,
                "-u",
                "scripts/score_run.py",
                "--run",
                name,
                "--split",
                split,
                "--epoch",
                "0",
            ],
            300,
        )
    record("pair_complete")


if __name__ == "__main__":
    main()
