"""Create fresh execution attempts after user-manager logout, preserving the clock."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
OLD = "next4h-20261022-r2"
NEW = "next4h-20261022-recovery"
ORIGINAL_START = 1792693855.4927282
BASE_DIGEST = "d76774e58d8d8c06c52bac701bde97ea2aa704f2536ac2f4815775e6a5e1eefd"


def sha(path: Path) -> str:
    """Hash immutable small source and manifest files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    """Read one local operator receipt."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """Write a new receipt without replacing a previous attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def source_copy(old: str, new: str) -> Path:
    """Copy prior code; all prior source and state remain byte-for-byte intact."""
    target = ROOT / "runtime" / new
    shutil.copytree(ROOT / "runtime" / old, target, ignore=shutil.ignore_patterns("__pycache__"))
    replacements = (
        ("next4h-20261022-r2", NEW),
        ("tabdata-next4h-20261022-remaining.service", "tabdata-" + NEW + ".service"),
        ("next4h-20261022-scaleout", "next4h-20261022-recovery-scaleout"),
        ("next4h-20261022-wide", "next4h-20261022-recovery-wide"),
    )
    for file in target.glob("*.py"):
        value = file.read_text()
        for before, after in replacements:
            value = value.replace(before, after)
        value = value.replace("start + 3600", "start + 5400")
        value = value.replace("PROFILE_CUTOFF = 3600", "PROFILE_CUTOFF = 5400")
        value = value.replace("profiles +3600s", "profiles +5400s")
        value = value.replace("systemd RuntimeMaxSec=13500,", "systemd remaining-time watchdog,")
        if file.name == "delivery.py":
            assert value.count('"--grad-stats",') == 1
            value = value.replace('"--grad-stats",', '"--grad-stats", "--save-training-state",')
        file.write_text(value)
    return target


def project(state: Path, packet: Path) -> None:
    """Mount checked immutable inputs beside clean output directories."""
    destination = state / "nll_project"
    destination.mkdir(parents=True)
    for name, target in (
        ("src", packet / "sources/nll/src"),
        ("scripts", packet / "sources/nll/scripts"),
        ("datasets", ROOT / "data/datasets"),
        ("splits", ROOT / "splits"),
    ):
        (destination / name).symlink_to(target, target_is_directory=True)
    (destination / "cache").mkdir()
    for name in ("train_targets.npz", "valid_targets.npz", "chains.npz", "chain_folds.npz"):
        (destination / "cache" / name).symlink_to(ROOT / "data/cache" / name)
    (destination / "artifacts/runs").mkdir(parents=True)
    (state / "numeric").mkdir()
    (state / "numeric/preprocessing_r2_e3").symlink_to(
        ROOT / "state" / OLD / "numeric/preprocessing_r2_e3", target_is_directory=True
    )


def main() -> None:
    """Verify linger, stage fresh attempts, and launch only the remaining allowance."""
    linger = subprocess.check_output(
        ["loginctl", "show-user", str(os.getuid()), "-pLinger", "--value"], text=True
    ).strip()
    if linger != "yes":
        raise RuntimeError("Linger must be verified yes before recovery admission")
    old_packet = ROOT / "runtime" / OLD
    assert sha(old_packet / "PLAN.json") == BASE_DIGEST
    origin = read(ROOT / "state" / OLD / "START.json")
    assert origin["start_unix"] == ORIGINAL_START and time.time() < ORIGINAL_START + 10000
    packet = source_copy(OLD, NEW)
    state = ROOT / "state" / NEW
    state.mkdir()
    project(state, packet)
    assert (packet / "sources/nll/src/wnn/training_state.py").is_file()
    write(
        state / "START.json",
        {
            **origin,
            "service": "tabdata-" + NEW + ".service",
            "recovery_started_unix": time.time(),
            "linger_verified": linger,
        },
    )
    shutil.copy2(ROOT / "state" / OLD / "INPUTS.json", state / "INPUTS.json")
    incident = {
        "reason": "user systemd stopped at SSH logout with Linger=no; Linger now verified yes",
        "original_start_unix": ORIGINAL_START,
        "hard_deadline_unix": ORIGINAL_START + 14400,
        "original_attempts_preserved": True,
        "new_attempts": "fresh initialization with original fold/seed/config; not extra independent replications",
        "saved_previous_inference_checkpoints": "four H256 model_ep0.pt checkpoints remain in original state",
        "optimizer_resume": False,
        "checkpoint_improvement": "canonical --save-training-state now saves complete epoch-boundary state",
        "changed_model_or_optimizer_source": False,
    }
    write(state / "RECOVERY_AUTHORITY.json", incident)
    old_plan = read(packet / "PLAN.json")
    (packet / "PLAN.json").rename(packet / "PLAN_PRIOR_ATTEMPT.json")
    runtime = int(ORIGINAL_START + 13500 - time.time()) - 30
    unit = "tabdata-" + NEW + ".service"
    plan = dict(old_plan, unit=unit, runtime_max_seconds=runtime, recovery=incident)
    plan["files"] = {
        str(p): sha(p)
        for p in packet.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".json", ".toml", ".lock")
        and "__pycache__" not in str(p)
    }
    write(packet / "PLAN.json", plan)
    digest = sha(packet / "PLAN.json")
    env = dict(
        TABDATA_SERVICE_UNIT=unit,
        TABDATA_PLAN_SHA256=digest,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONPATH=str(packet / "sources/numeric/src"),
    )
    os.environ.update(env)
    sys.path.insert(0, str(packet))
    import cohort

    cohort.verify(plan)
    args = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--working-directory=" + str(ROOT),
        "--property=RuntimeMaxSec=" + str(runtime),
        "--property=TimeoutStopSec=120",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=ExecStopPost=/usr/bin/timeout 110 "
        + str(ROOT / ".venv/bin/python")
        + " "
        + str(packet / "publish.py")
        + " --final",
    ]
    args += ["--setenv=" + key + "=" + value for key, value in env.items()]
    args += [str(ROOT / ".venv/bin/python"), "-u", str(packet / "continue_remaining.py")]
    subprocess.run(args, check=True, timeout=20)
    print(json.dumps({"unit": unit, "plan_sha256": digest, "runtime_seconds": runtime}), flush=True)
    for old, new, helper in (
        ("next4h-20261022-scaleout", "next4h-20261022-recovery-scaleout", "prepare_scaleout.py"),
        ("next4h-20261022-wide", "next4h-20261022-recovery-wide", "prepare_wide.py"),
    ):
        side = source_copy(old, new)
        (side / "SIDEPLAN.json").rename(side / "SIDEPLAN_PRIOR_ATTEMPT.json")
        setup = side / helper
        value = setup.read_text().replace(BASE_DIGEST, digest)
        setup.write_text(value)
        subprocess.run([str(ROOT / ".venv/bin/python"), str(setup)], check=True, timeout=40)
    print("THREE_RECOVERY_SERVICES_STARTED; ORIGINAL_DEADLINE_UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
