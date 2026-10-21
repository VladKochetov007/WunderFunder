"""Export the four predeclared inference-only correction scales serially."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-18/geometry_variations_r3"


def execute(arm: str, phase: str, command: list[str]) -> None:
    """Keep exact commands and terminal statuses beside the frozen scale plan."""

    def record(event: str, **fields: object) -> None:
        with (PACKET / "alpha_progress.jsonl").open("a") as output:
            output.write(
                json.dumps(
                    {
                        "event": event,
                        "utc": datetime.now(UTC).isoformat(),
                        "arm": arm,
                        "phase": phase,
                        **fields,
                    }
                )
                + "\n"
            )

    record("start", command=command)
    with (PACKET / f"{arm}.{phase}.log").open("x") as output:
        result = subprocess.run(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=600, check=False
        )
    record("end", returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f"{arm}/{phase} failed: {result.returncode}")


def main() -> None:
    """Create hash-bound aliases, then forward and score all predeclared scales."""
    for name in (
        "WNN_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "ARROW_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.update(WNN_LOADER_WORKERS="0", WNN_LOADER_RAM_GB="1.5")
    execute(
        "aliases",
        "create",
        [
            sys.executable,
            "scripts/create_geometry_alpha_aliases.py",
            "--checkpoint",
            "artifacts/runs/geometry_adapter_f0_s3_r2/model_ep0.pt",
            "--runs-root",
            "artifacts/runs",
            "--manifest",
            str(PACKET / "r2_alpha_aliases.json"),
            "--expected-source-sha256",
            "f03785c3dbc364d3d12a0ad0cbe1969a50a95bdc3a95d1f4bd0f4758f654df19",
        ],
    )
    manifest = json.loads((PACKET / "r2_alpha_aliases.json").read_text())
    split = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    for alias in manifest["aliases"]:
        run = alias["run"]
        execute(
            run,
            "predict",
            [
                sys.executable,
                "-u",
                "scripts/predict_run.py",
                run,
                "valid",
                "--campaign-split-manifest",
                split,
                "--campaign-evaluation-only",
                "--device",
                "cuda",
                "--epoch",
                "0",
            ],
        )
        execute(
            run,
            "score",
            [
                sys.executable,
                "-u",
                "scripts/score_run.py",
                "--run",
                run,
                "--split",
                split,
                "--epoch",
                "0",
            ],
        )


if __name__ == "__main__":
    main()
