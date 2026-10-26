"""Execute exactly two approved cohorts with mandatory live epoch validation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stage import OPERATORS, WAVES, read, sha


def save(path: Path, value: object) -> None:
    """Atomically publish operator state for monitoring."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def spawn(command: list[str], log_path: Path) -> subprocess.Popen:
    """Start an owned subprocess and preserve its entire console log."""
    with log_path.open("xb") as stream:
        return subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )


def stop(process: subprocess.Popen) -> None:
    """Signal only a child we created and still own."""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def validation_failure(folder: Path, started: float) -> str | None:
    """Treat stale observers and any broken epoch admission as fit-stopping failures."""
    monitor = folder / "observer-b128/MONITOR.json"
    if not monitor.exists():
        return "observer did not initialize" if time.time() - started > 120 else None
    if time.time() - monitor.stat().st_mtime > 120:
        return "observer heartbeat stale"
    state = read(monitor)
    if state.get("status") in {"partial", "partial_deadline", "failed"}:
        return "observer terminated incompletely"
    for name, job in state["jobs"].items():
        if job.get("status") not in {"running", "complete"}:
            return f"{name}: {job.get('status')}"
        for epoch, item in job["epochs"].items():
            if item.get("status") not in {"running", "complete"}:
                return f"{name} epoch{epoch}: {item.get('status')}"
    return None


def memory_failure() -> str | None:
    """Yield our cohort if co-resident work consumes the remaining physical margin."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    for line in result.stdout.splitlines():
        index, free_mib = line.split(",")
        if float(free_mib) < 16 * 1024:
            return f"GPU{index.strip()} physical freeVRAM below16GiB; yield only our cohort"
    return None


def run_wave(root: Path, wave: str) -> None:
    """Run four fits and their observers; preserve any failure without retries."""
    folder = root / wave
    pointer = read(folder / "SELECTED_PLAN.json")
    plan = Path(pointer["path"]).resolve(strict=True)
    if plan.parent != folder.resolve() or plan.is_symlink():
        raise ValueError("selected plan must stay inside its wave folder")
    if pointer["sha256"] != sha(plan):
        raise ValueError("approved plan changed")
    runtime = Path(__file__).resolve().parent
    children = []
    try:
        monitor = spawn(
            [
                sys.executable,
                str(runtime / "monitor.py"),
                "--plan",
                str(plan),
                "--output",
                str(folder / "observer-b128"),
                "--max-hours",
                "3",
                "--batch-size",
                "4",
                "--evaluation-seconds",
                "900",
            ],
            folder / "VALIDATION_MONITOR.log",
        )
        children.append(monitor)
        hardware = spawn(
            [
                sys.executable,
                str(OPERATORS / "telemetry.py"),
                "--root",
                str(folder),
                "--hours",
                "3",
            ],
            folder / "HARDWARE_MONITOR.log",
        )
        children.append(hardware)
        controller = spawn(
            [sys.executable, str(runtime / "run_panel.py"), "--plan", str(plan)],
            folder / "FIT_CONTROLLER.log",
        )
        children.append(controller)
        started = time.time()
        save(
            folder / "FIT_ONCE.json",
            {
                "pid": controller.pid,
                "created_unix": started,
                "command": [str(runtime / "run_panel.py"), "--plan", str(plan)],
            },
        )
        save(
            root / "CAMPAIGN_STATE.json",
            {
                "status": "running",
                "wave": wave,
                "started_unix": started,
                "controller_pid": controller.pid,
            },
        )
        while controller.poll() is None or monitor.poll() is None:
            reason = validation_failure(folder, started)
            if reason is None:
                reason = memory_failure()
            if controller.poll() not in {None, 0}:
                reason = f"fit controller exit{controller.returncode}"
            if monitor.poll() not in {None, 0}:
                reason = f"validation observer exit{monitor.returncode}"
            if (root / "METRIC_VERIFICATION_FAILED.json").exists():
                reason = "independent metric verification failed"
            if time.time() - started > 10800:
                reason = "three-hour wave deadline"
            if reason:
                save(folder / "WATCHDOG_STOP.json", {"unix": time.time(), "reason": reason})
                raise RuntimeError(reason)
            time.sleep(5)
        points = list((folder / "cohort-b128").glob("*/validation/epoch-*/RESULT.json"))
        if controller.returncode != 0 or monitor.returncode != 0:
            raise RuntimeError("fit or observer terminated with a nonzero exit")
        if read(folder / "cohort-b128/STATE.json").get("status") != "completed":
            raise RuntimeError("canonical cohort did not complete successfully")
        if len(points) != 20 or any(read(p).get("status") != "complete" for p in points):
            raise ValueError("wave ended without twenty complete validation points")
        save(folder / "DELIVERY_COMPLETE.json", {"unix": time.time(), "validation_points": 20})
    finally:
        # Controller first: its handler drains its receipt-proven training child groups.
        for process in reversed(children):
            stop(process)


def main() -> None:
    """Claimed by native cron once; never recreate failed or completed attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum: int, frame: object) -> None:
        raise InterruptedError(f"campaign received signal{signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if (args.root / "CAMPAIGN_STATE.json").exists():
        raise FileExistsError("campaign has an existing attempt")
    try:
        for wave in WAVES:
            run_wave(args.root, wave)
        save(
            args.root / "CAMPAIGN_STATE.json",
            {"status": "completed", "unix": time.time(), "new_fits": 8, "validation_points": 40},
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        save(
            args.root / "CAMPAIGN_STATE.json",
            {"status": "failed_preserved", "unix": time.time(), "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
