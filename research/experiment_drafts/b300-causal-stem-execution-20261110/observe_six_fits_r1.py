"""Attach the frozen read-only observer to six admitted fits; never start or stop fits."""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata-causal-stem-b300-20261110-r1")
SOURCE = ROOT / "source-r1"


def main() -> None:
    """Observe actual resources and rank-zero progress without rewriting producer status."""
    runpy.run_path(str(SOURCE / "src/wnn/training/sequence_source_capsule.py"))[
        "verify_source_capsule"
    ](
        SOURCE,
        SOURCE / "source-manifest.json",
        "1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936",
        "8a8afec964985a6ba8b13d1c9af1ab00e14c6504",
    )
    sys.path.insert(0, str(SOURCE / "src"))
    from wnn.training.sequence_observer import SequenceObserver
    from wnn.training.sequence_observer_resources import read_process

    class FitObserver(SequenceObserver):
        """Label sampled rank-log progress explicitly; do not invent a trainer heartbeat."""

        def _append(self, event: dict[str, object]) -> None:
            terminal = self.output / "terminal_status.json"
            if terminal.exists():
                return
            log = self.output / "rank0.jsonl"
            try:
                with log.open("rb") as stream:
                    size = os.fstat(stream.fileno()).st_size
                    stream.seek(max(0, size - 65_536))
                    complete = stream.read().split(b"\n")[:-1]
                if complete:
                    row = json.loads(complete[-1])
                    event["step"] = int(row["step"])
                    event["epoch"] = int(row["step"]) / 820
                    event["metrics"]["observer/rank0_last_update"] = int(row["step"])
                    event["metrics"]["observer/rank0_file_age_seconds"] = max(
                        0, time.time() - log.stat().st_mtime
                    )
                    event["context"]["observed_step_source"] = (
                        "actual completed rank0.jsonl record; not producer heartbeat"
                    )
            except (OSError, ValueError, KeyError):
                pass
            if not terminal.exists():
                super()._append(event)

    control = ROOT / "control/external-observer-r1"
    control.mkdir(exist_ok=False)
    (control / "owner.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "birth_ticks": read_process(os.getpid()).start_ticks,
                "new_fits": 0,
                "signals": 0,
                "source_commit": "8a8afec964985a6ba8b13d1c9af1ab00e14c6504",
            }
        )
    )
    admission = json.loads((ROOT / "control/ROOT-fit-admission-r1.json").read_bytes())
    deadline = json.loads((ROOT / "receipts/first-profile-r1.json").read_bytes())[
        "wave_deadline_epoch"
    ]
    observers, closed = {}, set()
    while time.time() < deadline and len(closed) < 6:
        for cell in admission["profiles"]:
            output = ROOT / "runs" / cell / "fit"
            if cell in closed:
                continue
            if (output / "terminal_status.json").exists():
                closed.add(cell)
                observers.pop(cell, None)
                continue
            receipt = ROOT / "control/b300-driver-r1/owned" / f"{cell}-fit-r2.json"
            if not receipt.is_file():
                continue
            owned = json.loads(receipt.read_bytes())
            launcher = owned.get("launcher", {})
            pid = launcher.get("pid")
            process = read_process(pid) if isinstance(pid, int) else None
            if (
                process is None
                or not process.is_alive
                or process.start_ticks != launcher.get("birth_ticks")
            ):
                if owned.get("status") in ("completed", "failed"):
                    closed.add(cell)
                continue
            if cell not in observers:
                observers[cell] = FitObserver(output, pid, 10)
            observers[cell].sample()
        state = {
            "at_epoch": time.time(),
            "attached": sorted(observers),
            "closed": sorted(closed),
            "new_fits": 0,
            "signals": 0,
        }
        temporary = control / "status.pending"
        temporary.write_text(json.dumps(state))
        temporary.replace(control / "status.json")
        time.sleep(min(10, max(0, deadline - time.time())))


if __name__ == "__main__":
    main()
