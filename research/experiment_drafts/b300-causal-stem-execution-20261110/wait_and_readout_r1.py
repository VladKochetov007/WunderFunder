"""Wait for the admitted fit panel, then invoke the immutable reviewed readout once."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata-causal-stem-b300-20261110-r1")


def main() -> int:
    """Keep the original wave deadline; start no profile, fit or retry."""
    control = ROOT / "control/b300-driver-r1"
    output = ROOT / "control/readout-wait-r1"
    output.mkdir(exist_ok=False)
    birth = int(
        Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    )
    (output / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "birth_ticks": birth, "new_fits": 0})
    )
    clock = json.loads((ROOT / "receipts/first-profile-r1.json").read_bytes())
    deadline = clock["wave_deadline_epoch"]
    state = {
        "status": "waiting_for_fixed_E4_panel",
        "new_fits": 0,
        "wave_deadline_epoch": deadline,
    }

    def publish() -> None:
        state["at_epoch"] = time.time()
        temporary = output / "state.pending"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / "state.json")

    while time.time() < deadline:
        publish()
        summary = control / "fit-r2-complete.json"
        if summary.is_file():
            value = json.loads(summary.read_bytes())
            if value.get("status") != "complete":
                state.update(
                    status="fit_panel_incomplete", fit_status=value.get("status")
                )
                publish()
                return 2
            with (control / "phase.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(min(5, max(0, deadline - time.time())))
                    continue
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            break
        time.sleep(min(15, max(0, deadline - time.time())))
    else:
        state.update(status="original_wave_deadline_expired")
        publish()
        return 124

    script = ROOT / "control/readout-receiver-r2/b300_readout_r2.py"
    if (
        hashlib.sha256(script.read_bytes()).hexdigest()
        != "ed54c4174acffe2d94f0de2f4c17d504d2bd5099d7407df3bd7b4131b7891b49"
    ):
        raise ValueError("reviewed readout receiver changed")
    command = [
        sys.executable,
        "-u",
        str(script),
        "--root",
        str(ROOT),
        "--source",
        str(ROOT / "source-r1"),
        "--manifest",
        str(ROOT / "source-r1/source-manifest.json"),
        "--manifest-sha256",
        "1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936",
        "--source-commit",
        "8a8afec964985a6ba8b13d1c9af1ab00e14c6504",
        "--python",
        "/data/epanov_s/tabdata/.venv/bin/python",
        "--authority",
        str(ROOT / "control/AUTHORITY_R2.json"),
        "--authority-sha256",
        "74663454d5c8ace8f7c1cdeb6c651e3d9a93fb9b2e1c4cc756bd03fb45df3f77",
        "--fit-admission-sha256",
        "fa608f409499b67ef6c37b6d0e6a07f9c1a32392d321edbf669effd870c8ba18",
        "--r1-receiver",
        str(ROOT / "control/receiver-r1/receiving"),
        "--r2-receiver",
        str(ROOT / "control/receiver-r2"),
    ]
    state.update(status="readout_running", command=command)
    publish()
    with (output / "readout.log").open("xb") as log:
        code = subprocess.call(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log
        )
    state.update(
        status="readout_finished" if code == 0 else "readout_failed", returncode=code
    )
    publish()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
