from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
P = B / "research/campaign_2026-10-18/priority25_execution_r4"
SUPPORT = P / "f3_checkpoint_support_r1"
PLAN = P / "f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
PLAN_SHA = "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, timeout=15)


def write_new(path: Path, value: object) -> bytes:
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return raw


def main() -> None:
    """Admit one exact F3 stage1 support operation after native successful exit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["wide", "deep"], required=True)
    parser.add_argument("--operation", choices=["collector", "verifier"], required=True)
    parser.add_argument("--pins-json", required=True)
    parser.add_argument("--wrapper-sha", required=True)
    parser.add_argument("--wrapper-review-sha", required=True)
    args = parser.parse_args()
    for value in [args.wrapper_sha, args.wrapper_review_sha]:
        assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    pins = json.loads(args.pins_json)
    assert set(pins) == {
        "collect_f3_checkpoint.py",
        "verify_f3_checkpoint.py",
        "f3_checkpoint_common.py",
        "f3_checkpoint_evidence.py",
        "f3_checkpoint_lineage.py",
        "f3_checkpoint_membership.py",
        "f3_checkpoint_semantics.py",
        "f3_checkpoint_tensors.py",
    }
    for name, expected in pins.items():
        assert digest(SUPPORT / name) == expected, name
    assert digest(PLAN) == PLAN_SHA
    assert (
        digest(B / "splits/fold3.json")
        == "21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f"
    )
    registry = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
    archives = B / "support-source/terminal-v1/PORTABLE_ARCHIVES.json"
    assert digest(registry) == "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
    assert digest(archives) == "9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341"
    attempt_path = (
        B / f"runtime/attempts/computer-vision-homework-r01-{args.arm}-f3s1001-ep01-v1.json"
    )
    attempt = json.loads(attempt_path.read_bytes())
    if (attempt.get("status"), attempt.get("exit_status")) != ("finished", 0):
        print(
            json.dumps(
                {
                    "status": "WAITING_NATIVE_FINISH",
                    "attempt_status": attempt.get("status"),
                    "exit_status": attempt.get("exit_status"),
                }
            )
        )
        return

    controller = "computer-vision-homework-r01-f3-sequence.service"
    sequence = json.loads((B / "runtime/receipts/RANK01_F3_S1001_SEQUENCE.json").read_bytes())
    assert sequence["plan_sha256"] == PLAN_SHA and sequence["status"] == "running"
    active_worker = f"wnn-bandit-{sequence['attempt']}.service"
    properties = dict(
        line.split("=", 1)
        for line in command(
            [
                "systemctl",
                "--user",
                "show",
                "wnn-campaign.slice",
                "-p",
                "ControlGroup,MemoryCurrent,MemoryHigh,MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec",
            ]
        ).splitlines()
    )
    assert (
        properties["CPUQuotaPerSecUSec"],
        properties["MemoryHigh"],
        properties["MemoryMax"],
        properties["MemorySwapMax"],
    ) == ("2s", "18000000000", "20000000000", "0")
    parent = Path("/sys/fs/cgroup") / properties["ControlGroup"].lstrip("/")
    children = []
    for child in parent.iterdir():
        if not child.is_dir() or not (child / "cgroup.procs").exists():
            continue
        if not (child / "cgroup.procs").read_text().strip():
            continue
        assert child.name in {controller, active_worker}, child.name
        quota, period = (child / "cpu.max").read_text().split()
        assert quota != "max"
        cores = int(quota) / int(period)
        maximum = int((child / "memory.max").read_text())
        expected = (0.1, 512000000) if child.name == controller else (1.0, 8000000000)
        assert (cores, maximum) == expected
        assert (child / "memory.swap.max").read_text().strip() == "0"
        children.append(
            {
                "unit": child.name,
                "cpu_cores": cores,
                "memory_max": maximum,
                "memory_current": int((child / "memory.current").read_text()),
            }
        )
    assert {child["unit"] for child in children} == {controller, active_worker}
    collecting = args.operation == "collector"
    extra_cpu, extra_ram = (0.25, 512000000) if collecting else (0.5, 1500000000)
    committed = (
        max(
            0, int(properties["MemoryCurrent"]) - sum(child["memory_current"] for child in children)
        )
        + sum(child["memory_max"] for child in children)
        + extra_ram
    )
    assert sum(child["cpu_cores"] for child in children) + extra_cpu <= 2
    assert committed < 17000000000
    scratch = command(["findmnt", "-T", str(B / "research_tmp"), "-no", "FSTYPE,SOURCE,TARGET"])
    assert scratch.split()[0] not in {"tmpfs", "ramfs"}
    assert os.statvfs(B).f_bavail * os.statvfs(B).f_frsize > 1000000000

    unit = f"computer-vision-homework-r01-{args.arm}-f3-stage1-{args.operation}-v1.service"
    assert (
        command(["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"]).strip()
        == "not-found"
    )
    prefix = f"RANK01_{args.arm.upper()}_F3_S1001_STAGE1"
    records = B / "receipts"
    collect = records / f"{prefix}_COLLECTION.json"
    verify = records / f"{prefix}_VERIFICATION.json"
    snapshot = B / f"checkpoint_snapshots/rr18_r01_{args.arm}_f3_s1001/stage1_epoch1"
    output = collect if collecting else verify
    assert not os.path.lexists(output)
    if collecting:
        assert not os.path.lexists(snapshot)
    else:
        assert collect.is_file() and snapshot.is_dir()
    admission_path = records / f"{prefix}_{args.operation.upper()}_ADMISSION_V1.json"
    launch_path = records / f"{prefix}_{args.operation.upper()}_LAUNCH_V1.json"
    assert not os.path.lexists(admission_path) and not os.path.lexists(launch_path)
    py = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"
    argv = [
        py,
        str(SUPPORT / ("collect_f3_checkpoint.py" if collecting else "verify_f3_checkpoint.py")),
        "--arm",
        args.arm,
        "--phase",
        "stage1",
        "--plan",
        str(PLAN),
        "--output",
        str(output),
    ]
    if collecting:
        argv += ["--campaign-root", str(B)]
    else:
        argv += [
            "--snapshot-dir",
            str(snapshot),
            "--source-root",
            str(B / "source-a266b036"),
            "--collection-receipt",
            str(collect),
            "--fold-manifest",
            str(B / "splits/fold3.json"),
            "--data-registry",
            str(registry),
            "--archive-manifest",
            str(archives),
        ]
    cmd = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        f"--property=CPUQuota={int(extra_cpu * 100)}%",
        f"--property=MemoryMax={extra_ram}",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=120",
        "--property=TimeoutStopSec=15",
        "--working-directory=" + str(B),
    ]
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(B / "research_tmp"),
    }.items():
        cmd.append(f"--setenv={key}={value}")
    cmd += argv
    admission = {
        "schema": "priority25.f3_stage1_support_admission.v1",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "ADMITTED",
        "unit": unit,
        "arm": args.arm,
        "operation": args.operation,
        "attempt_sha256": digest(attempt_path),
        "helper_pins": pins,
        "wrapper_sha256": args.wrapper_sha,
        "wrapper_review_sha256": args.wrapper_review_sha,
        "parent": properties,
        "active_children": children,
        "committed_with_support_bytes": committed,
        "memory_events_before": (parent / "memory.events").read_text(),
        "physical_scratch": scratch,
        "command": cmd,
    }
    admission_raw = write_new(admission_path, admission)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    launch = {
        "schema": "priority25.f3_stage1_support_launch.v1",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "unit": unit,
        "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
        "launch_returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    write_new(launch_path, launch)
    print(
        json.dumps(
            {
                "status": "LAUNCHED" if result.returncode == 0 else "LAUNCH_FAILED",
                "admission": admission,
                "launch": launch,
            },
            indent=2,
        )
    )
    assert result.returncode == 0


if __name__ == "__main__":
    main()
