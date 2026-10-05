from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
CHECKS = ROOT / "research/campaign_2026-10-05/checks"
PYTHON = str(ROOT / ".venv/bin/python")


def main() -> None:
    """Run only two preregistered seeds; stop on any failure without a retry."""
    state: dict[str, object] = {"status": "running", "completed_seeds": []}
    path = CHECKS / "E469_REPLICATION_STATUS.json"
    for seed in (271829, 271830):
        identity = f"e469-ten-arm-seed{seed}-v1"
        state.update(utc=datetime.now(UTC).isoformat(), active_attempt=identity)
        path.write_text(json.dumps(state, indent=2) + "\n")
        result = subprocess.run(
            [
                PYTHON,
                "scripts/run_campaign.py",
                "launch",
                str(FOLDER / f"seed{seed}.sealed.json"),
                "--attempt",
                identity,
                "--python",
                PYTHON,
            ],
            cwd=ROOT,
            check=False,
        )
        attempt = json.loads(
            (ROOT / "research/campaign_2026-10-04/runner/attempts" / f"{identity}.json").read_text()
        )
        if (
            result.returncode
            or attempt.get("status") != "finished"
            or (attempt.get("exit_code") != 0 or not attempt.get("teardown_verified"))
        ):
            state.update(status="failed", failed_attempt=identity)
            path.write_text(json.dumps(state, indent=2) + "\n")
            raise RuntimeError(f"Fixed seed failed: {identity}; no retry")
        environment = {
            **os.environ,
            "WNN_SLICE": "wnn-campaign.slice",
            "WNN_CPU_QUOTA": "100%",
            "WNN_THREADS": "1",
            "WNN_MEM_MAX": "2G",
            "WNN_RUNTIME_MAX": "180",
        }
        run = FOLDER / "runs" / f"seed{seed}"
        subprocess.run(
            [str(ROOT / "scripts/limited"), PYTHON, str(FOLDER / "score_replication.py"), str(run)],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=200,
        )
        summary = {
            "fit": json.loads((run / "result.json").read_text()),
            "paired": json.loads((run / "paired_components.json").read_text()),
            "attempt": attempt,
        }
        (FOLDER / f"SEED{seed}_RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
        state["completed_seeds"].append(seed)
    state.update(status="complete", active_attempt=None, utc=datetime.now(UTC).isoformat())
    path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
