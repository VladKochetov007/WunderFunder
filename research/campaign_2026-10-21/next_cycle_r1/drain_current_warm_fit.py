from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
UNIT = "wnn-warm-i1-panel-r2-20261022.service"
INVOCATION = "04c1ee399eb84328a0da28c10cf8102d"
PID = 3384439
START_TICKS = "18816570"
OUTPUT = ROOT / "research/campaign_2026-10-21/next_cycle_r1/WARM_GRACEFUL_DRAIN_RESULT_R1.json"


def observe_exit() -> int | None:
    """Wait for only the pinned training child; never resume its supervisor."""
    while True:
        try:
            fields = Path(f"/proc/{PID}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return None
        if fields[19] != START_TICKS:
            return None
        if fields[0] == "Z":
            return int(fields[49])
        time.sleep(15)


def main() -> None:
    """Collect terminal checkpoint identities and release the paused worker unit."""
    exit_status = observe_exit()
    artifacts = {}
    run = ROOT / "artifacts/runs/i1warm_f0_control_seed3_e2_cp1_r1"
    for name in ("model_ep0.pt", "model_ep1.pt", "training_state.pt", "provenance.json"):
        path = run / name
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    invocation = subprocess.run(
        ["systemctl", "--user", "show", UNIT, "-p", "InvocationID", "--value"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    result = {
        "status": "FIT_LANDED_PAUSE_REQUESTED"
        if exit_status == 0
        else "FIT_EXIT_UNKNOWN_OR_FAILED",
        "wait_status": exit_status,
        "unit": UNIT,
        "expected_invocation": INVOCATION,
        "observed_invocation": invocation,
        "artifacts": artifacts,
        "next_steps": "Paused by user; no exports, further fits, profiles or new admission.",
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if invocation == INVOCATION:
        # The child has exited; the suspended supervisor must never admit its next step.
        subprocess.run(
            ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", UNIT],
            check=True,
        )


if __name__ == "__main__":
    main()
