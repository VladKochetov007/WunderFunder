"""Run the unchanged cohort directly in a new unit within the original deadline."""

from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import threading
import time
from pathlib import Path

from cohort import monitor, verify
from continue_cohort import family
from delivery import PACKET, STATE, UNIT, read, write


def main() -> None:
    """Reuse completed calibration and retain the original absolute phase cutoffs."""
    with (STATE / "REMAINING_ONESHOT.json").open("x") as claim:
        claim.write(str(os.getpid()))
    stop = threading.Event()
    try:
        plan = read(PACKET / "PLAN.json")
        verify(plan)
        start = read(STATE / "START.json")["start_unix"]
        assert start == 1792693855.4927282 and time.time() < start + 11100
        assert UNIT == "tabdata-next4h-20261022-remaining.service"
        assert "/" + UNIT in Path("/proc/self/cgroup").read_text()
        actual = subprocess.check_output(
            ["systemctl", "--user", "show", UNIT, "-pRuntimeMaxUSec", "--value"],
            text=True,
            timeout=10,
        ).strip()
        for fold in (0, 2):
            assert (STATE / f"numeric/preprocessing_r2_e3/f{fold}.json").exists()
        write(
            STATE / "REMAINING_STARTED.json",
            {
                "pid": os.getpid(),
                "unit": UNIT,
                "runtime_max_observed": actual,
                "original_start_unix": start,
                "deadline_unix": start + 14400,
                "started_unix": time.time(),
                "plan_sha256": os.environ["TABDATA_PLAN_SHA256"],
            },
        )
        watcher = threading.Thread(target=monitor, args=(stop,), daemon=True)
        watcher.start()
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(family, name, [j for j in plan["jobs"] if j["family"] == name], start)
                for name in ("numeric", "nll")
            ]
            for done in futures.as_completed(pending):
                done.result()
        verify(plan)
        write(
            STATE / "COMPLETE.json",
            {"stage": "cohort_terminal", "elapsed_seconds": time.time() - start},
        )
    except BaseException as error:
        write(STATE / "REMAINING_FAILURE.json", {"error": repr(error), "time_unix": time.time()})
        raise
    finally:
        stop.set()
        subprocess.run(["systemctl", "--user", "stop", "--no-block", UNIT], check=True, timeout=10)


if __name__ == "__main__":
    main()
