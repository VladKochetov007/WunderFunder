"""Verify and acknowledge one diagnosed pre-GPU false commandline alarm."""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONT = Path(__file__).resolve().parent
CPU_UNIT = "wnn-cpu-1-controller-1791051156-3427787"
JOB = "c20261003_capture_v4_default_2113"


def query(arguments: list[str]) -> str:
    """Read authoritative state under the bounded maintenance scope."""
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=10).stdout


def main() -> None:
    """Test the actual corrected predicate before clearing only explained incidents."""
    source = (CONT / "run_cpu.sh").read_text()
    assignment = next(line for line in source.splitlines() if line.startswith("failure_message="))
    terminal = (CONT / f"logs/{CPU_UNIT}.terminal.log").read_text()
    commands = (CONT / "logs/cpu_commands.log").read_text()
    if f"unit={CPU_UNIT} exit=0\n" not in commands or "--timeout 600" not in terminal:
        raise ValueError("successful CPU launcher with timeout argument not established")
    cases = [
        ("actual_successful_launcher", terminal, False),
        ("timeout", f"systemd: {CPU_UNIT}.scope: Failed with result 'timeout'.\n", True),
        (
            "oom",
            f"systemd: {CPU_UNIT}.scope: A process of this unit has been killed by the OOM killer.\n",
            True,
        ),
        ("signal", f"systemd: {CPU_UNIT}.scope: Failed with result 'signal'.\n", True),
        ("resources", f"systemd: {CPU_UNIT}.scope: Failed with result 'resources'.\n", True),
        (
            "commandline_resource_word",
            f"systemd: Started {CPU_UNIT}.scope - command --timeout 600 --note resources.\n",
            False,
        ),
    ]
    checked = []
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "terminal.log"
        for name, text, expected in cases:
            fixture.write_text(text)
            code = 'unit="$1"\n' + assignment + '\nrg -qi "$failure_message" "$2"\n'
            result = subprocess.run(
                ["bash", "-c", code, "predicate", CPU_UNIT, str(fixture)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode not in (0, 1) or (result.returncode == 0) is not expected:
                raise ValueError(f"predicate regression failed: {name}: {result.stderr}")
            checked.append({"case": name, "detected": expected})
    job_path = ROOT / f"research/campaign_2026-09-29/jobs/{JOB}.json"
    job = json.loads(job_path.read_text())
    if (
        job["status"] != "failed"
        or job.get("started_at") is not None
        or job["elapsed_seconds"] != 0
        or job["peak_observed_vram_bytes"] != 0
        or "RESOURCE_FAILURE" not in job["failure"]
    ):
        raise ValueError("only failed pre-execution admission may be acknowledged")
    group = query(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"]
    ).strip()
    memory_events = (Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.events").read_text()
    if int(dict(line.split() for line in memory_events.splitlines())["oom_kill"]) != 11:
        raise ValueError("new OOM needs separate diagnosis")
    if query(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip():
        raise ValueError("GPU device not idle")
    with (CONT / "locks/admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        acknowledgements = json.loads((CONT / "acknowledgements.json").read_text())
        incidents = [
            json.loads(line)
            for line in (CONT / "logs/incidents.jsonl").read_text().splitlines()
            if line.strip()
        ]
        pending = [
            event for event in incidents if event["id"] not in acknowledgements["incident_ids"]
        ]
        for event in pending:
            if not (
                (event["type"] == "gpu_job_failed" and event["record"].endswith(f"/{JOB}.json"))
                or (
                    event["type"] == "journal_fault"
                    and f"{JOB}.service: Failed with result 'exit-code'" in event["message"]
                )
            ):
                raise ValueError(f"unexplained incident: {event['id']}")
        if len(pending) != 2:
            raise ValueError("expected exactly two diagnosed admission incidents")
        output = CONT / "runtime_incidents/false_timeout_2112"
        output.mkdir(parents=True, exist_ok=False)
        for name, path in (
            ("RESOURCE_FAILURE", CONT / "RESOURCE_FAILURE"),
            ("QUEUE_PAUSE", ROOT / "research/campaign_2026-10-01/QUEUE_PAUSE"),
            ("failed_job.json", job_path),
            ("launcher_terminal.log", CONT / f"logs/{CPU_UNIT}.terminal.log"),
        ):
            (output / name).write_bytes(path.read_bytes())
        report = {
            "status": "PASS",
            "time": datetime.now().astimezone().isoformat(),
            "diagnosis": "Old unanchored timeout regex matched successful Started command --timeout600, publishing a false CPU barrier before GPU admission",
            "predicate_cases": checked,
            "CPU_launcher_exit": 0,
            "GPU_started": False,
            "new_OOM": False,
            "historical_OOM_kill": 11,
            "run_cpu_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "incidents_acknowledged": pending,
            "failed_job_preserved": True,
            "retry_condition": "fresh hash-bound live receipt and new job/output name, unchanged scientific recipe and resource ceilings",
        }
        (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
        acknowledgements["incident_ids"] += [event["id"] for event in pending]
        acknowledgements.setdefault("historical_job_records", []).append(f"{JOB}.json")
        acknowledgements.setdefault("diagnosed_retries", []).append(
            str((output / "REPORT.json").relative_to(ROOT))
        )
        temporary = CONT / "acknowledgements.json.tmp"
        temporary.write_text(json.dumps(acknowledgements, indent=2) + "\n")
        temporary.replace(CONT / "acknowledgements.json")
        (CONT / "RESOURCE_FAILURE").unlink()
        (ROOT / "research/campaign_2026-10-01/QUEUE_PAUSE").unlink()
    print(
        json.dumps(
            {
                "status": "PASS",
                "predicate_checks": len(checked),
                "GPU_started": False,
                "report": str(output / "REPORT.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
