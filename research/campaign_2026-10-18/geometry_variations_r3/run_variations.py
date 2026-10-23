"""Execute two bounded, matched feature-block probes using existing train tools."""

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
    """Preserve phase receipts independently of child process output."""
    with (PACKET / "progress.jsonl").open("a") as output:
        output.write(
            json.dumps({"event": event, "timestamp_utc": datetime.now(UTC).isoformat(), **fields})
            + "\n"
        )


def execute(arm: str, phase: str, command: list[str], timeout: int) -> None:
    """Run a frozen command serially within the one-worker resource reservation."""
    record("phase_start", arm=arm, phase=phase, command=command)
    with (PACKET / f"{arm}.{phase}.log").open("x") as output:
        result = subprocess.run(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    record("phase_end", arm=arm, phase=phase, returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f"{arm}/{phase} failed: {result.returncode}")


def main() -> None:
    """Fit both blocks, then export and score their fixed checkpoints."""
    os.environ.update(
        {
            name: "1"
            for name in [
                "WNN_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "POLARS_MAX_THREADS",
                "ARROW_NUM_THREADS",
            ]
        }
    )
    os.environ.update(
        {"WNN_LOADER_WORKERS": "0", "WNN_LOADER_RAM_GB": "1.5", "PYTHONUNBUFFERED": "1"}
    )
    recipe = json.loads((PACKET / "recipe.json").read_text())
    for arm, command in recipe["arms"].items():
        execute(arm, "train", command, 1200)
    split = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    for arm in recipe["arms"]:
        name = f"geometry_variation_f0_s3_r3_{arm}"
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
    record("variations_complete")


if __name__ == "__main__":
    main()
