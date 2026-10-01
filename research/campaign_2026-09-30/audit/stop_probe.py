"""Exercise durable CPU stop propagation without launching any GPU work."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "research/campaign_2026-09-29"
MARKER = CAMPAIGN / "STOP_NEW_JOBS"
JOBS = CAMPAIGN / "jobs"


def write(path: Path, value: object) -> None:
    """Save unique probe evidence."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(command: list[str]) -> subprocess.CompletedProcess:
    """Run a bounded command and retain its output on error."""
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                          text=True, timeout=20)


def wait_file(path: Path) -> None:
    """Wait briefly for a worker startup artifact."""
    deadline = time.monotonic() + 15
    while not path.exists():
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.1)


def worker(mode: str, evidence: Path) -> None:
    """Create an idle descendant, or wait inside the queue polling code."""
    if mode.startswith("wait_"):
        sys.path.insert(0, str(ROOT / "scripts"))
        import campaign_controls_queue as queue
        queue.JOBS = AUDIT / evidence.stem
        queue.JOBS.mkdir()
        if mode == "wait_running":
            write(queue.JOBS / "fake.json", {"status": "running"})
        write(evidence, {"pid": os.getpid(), "mode": mode})
        try:
            queue.wait_success("fake", AUDIT / "never_created", 10)
        except queue.QueueStopped as exc:
            print(str(exc), flush=True)
            return
        raise AssertionError("queue wait ignored persistent stop")
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)"
    child = subprocess.Popen([sys.executable, "-c", code])
    time.sleep(0.2)
    write(evidence, {"pid": os.getpid(), "descendant": child.pid, "mode": mode})
    if mode == "orphan":
        return
    time.sleep(90)


def driver() -> None:
    """Launch unique one-core durable jobs, signal them, and verify cleanup."""
    original = MARKER.read_bytes()
    (AUDIT / "original_stop_marker.txt").write_bytes(original)
    historical = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for directory in (JOBS, CAMPAIGN / "logs")
                  for p in directory.iterdir() if p.is_file()}
    results: list[dict] = []
    prefix = f"c20260930_stopprobe_{time.time_ns()}"
    try:
        for mode in ("term", "int", "orphan", "wait_missing", "wait_running"):
            name = f"{prefix}_{mode}"
            unit = f"wnn-top1-{name}"
            evidence = AUDIT / f"{name}.json"
            MARKER.unlink(missing_ok=True)
            command = [sys.executable, "scripts/campaign_job.py", "launch",
                       "--name", name, "--kind", "cpu", "--phase", "stop-regression",
                       "--timeout", "30", "--threads", "1", "--", "scripts/limited",
                       sys.executable, str(Path(__file__).resolve()), mode, str(evidence)]
            launched = run(command)
            wait_file(evidence)
            pids = json.loads(evidence.read_text())
            if mode == "term":
                run(["systemctl", "--user", "stop", unit])
            elif mode == "int":
                run(["systemctl", "--user", "kill", "--kill-whom=main",
                     "--signal=SIGINT", unit])
            elif mode.startswith("wait_"):
                MARKER.write_bytes(original)
            record_path = JOBS / f"{name}.json"
            deadline = time.monotonic() + 20
            while True:
                record = json.loads(record_path.read_text())
                if record["status"] != "running":
                    break
                assert time.monotonic() < deadline, record
                time.sleep(0.1)
            assert record["status"] == ("interrupted" if mode in ("term", "int") else "finished"), record
            active = subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                                     record["scope_unit"]], check=False).returncode == 0
            assert not active, record
            for pid in [pids["pid"], *([pids["descendant"]] if "descendant" in pids else [])]:
                stat = Path(f"/proc/{pid}/stat")
                assert not stat.exists() or stat.read_text().split()[2] == "Z", pid
            if mode in ("term", "int"):
                assert MARKER.exists(), "signal did not leave durable marker"
                blocked_name = f"{name}_blocked"
                blocked = subprocess.run([sys.executable, "scripts/campaign_job.py", "launch",
                                          "--name", blocked_name, "--kind", "cpu",
                                          "--timeout", "5", "--threads", "1", "--", "true"],
                                         capture_output=True, text=True, check=False)
                assert blocked.returncode != 0 and "campaign launch blocked" in blocked.stderr
                assert not (JOBS / f"{blocked_name}.json").exists()
            results.append({"mode": mode, "status": "pass", "job": name,
                            "record_status": record["status"], "exit_code": record["exit_code"],
                            "elapsed_seconds": record["elapsed_seconds"], "pids": pids,
                            "scope_inactive": True, "launch_stdout": launched.stdout.strip()})
            print(json.dumps(results[-1]), flush=True)
    finally:
        MARKER.write_bytes(original)
        unchanged = all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                        for p, h in historical.items())
        write(AUDIT / "stop_probe_results.json", {"results": results,
               "historical_job_and_log_bytes_unchanged": unchanged,
               "historical_file_count": len(historical),
               "original_marker_restored": MARKER.read_bytes() == original})
    assert unchanged


if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver()
    else:
        worker(sys.argv[1], Path(sys.argv[2]))
