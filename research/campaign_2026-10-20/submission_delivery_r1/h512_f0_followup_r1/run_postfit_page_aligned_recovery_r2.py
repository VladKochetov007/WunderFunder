"""Wait for the exact admitted fit, then launch its finite export/grade sequence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from record_h512_fit_terminal import service_properties

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = Path(__file__).resolve().parent
ADMISSION = PACKET / "POSTFIT_EXECUTION_ADMISSION_R1.json"
UNIT = "wnn-h512-postfit-r1-20261021.service"
CGROUP = Path(
    "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice"
)


def sha256(path: Path) -> str:
    """Hash source and metadata without loading model/data payloads."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reclaim_inactive_cache() -> None:
    """Reclaim only inactive file pages charged to the bounded campaign slice."""
    statistics = dict(line.split() for line in (CGROUP / "memory.stat").read_text().splitlines())
    current = int((CGROUP / "memory.current").read_text())
    amount = min(max(0, current - 3_800_000_000), int(statistics["inactive_file"]))
    if amount:
        try:
            (CGROUP / "memory.reclaim").write_text(f"{amount} swappiness=0\n")
        except OSError as error:
            print(json.dumps({"cache_reclaim_warning": str(error)}), flush=True)


def main() -> None:
    """Admit no fits; launch one bounded derivation/export/official-score worker."""
    admission = json.loads(ADMISSION.read_text())
    if admission.get("status") != "ADMITTED_CONDITIONAL_ON_EXACT_FIT_SUCCESS":
        raise RuntimeError("post-fit execution has not been admitted")
    fit = admission["fit_unit"]
    while True:
        if (PACKET / "STOP_POST_FIT").exists():
            return
        properties = service_properties(fit)
        if properties.get("InvocationID") != admission["fit_invocation_id"]:
            raise RuntimeError("the admitted fit invocation changed or disappeared")
        if properties.get("MainPID") == "0":
            break
        time.sleep(15)
    if properties.get("Result") != "success" or properties.get("ExecMainStatus") != "0":
        raise RuntimeError("fit did not finish successfully; no post-fit execution")
    elapsed = (
        int(properties["ExecMainExitTimestampMonotonic"])
        - int(properties["ExecMainStartTimestampMonotonic"])
    ) / 1e6
    spent = admission["previous_completed_gpu_job_seconds"] + elapsed
    cap = admission["postfit_cap_seconds"]
    if elapsed <= 0 or spent + cap > admission["campaign_cap_seconds"]:
        raise RuntimeError("post-fit sequence exceeds the cumulative GPU-job budget")
    for name, digest in admission["execution_pins"].items():
        if sha256(Path(name)) != digest:
            raise RuntimeError(f"frozen execution source/plan changed: {name}")
    for name in admission["must_not_exist"]:
        if os.path.lexists(ROOT / name):
            raise FileExistsError(ROOT / name)
    subprocess.run(
        [
            "git",
            "-C",
            admission["producer_root"],
            "diff",
            "--exit-code",
            admission["producer_revision"],
            "--",
            "src",
            "scripts",
            "utils.py",
        ],
        check=True,
    )
    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            admission["producer_root"],
            "ls-files",
            "--others",
            "--",
            "src",
            "scripts",
            "utils.py",
        ],
        text=True,
    )
    if any(name.endswith(".py") for name in untracked.splitlines()):
        raise RuntimeError("untracked Python execution source in the frozen producer")
    page_size = os.sysconf("SC_PAGE_SIZE")
    expected_limit = (5_000_000_000 // page_size) * page_size
    if int((CGROUP / "memory.max").read_text()) != expected_limit:
        raise RuntimeError("aggregate task memory ceiling differs")
    if int((CGROUP / "memory.swap.current").read_text()):
        raise RuntimeError("task swap is nonzero")
    reclaim_inactive_cache()
    start = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "admission_sha256": sha256(ADMISSION),
        "fit_service": properties,
        "fit_elapsed_seconds": elapsed,
        "completed_gpu_job_seconds_before_postfit": spent,
        "postfit_cap_seconds": cap,
        "root_acceptance_policy": "Terminal producer is first; any mismatch stops the sequence.",
    }
    with (PACKET / "POSTFIT_START_R1.json").open("x") as stream:
        json.dump(start, stream, indent=2)
        stream.write("\n")
    command = [
        "systemd-run",
        "--user",
        f"--unit={UNIT}",
        "--slice=wnn-campaign.slice",
        f"--working-directory={ROOT}",
        "-p",
        "CPUQuota=100%",
        "-p",
        "MemoryHigh=3400000000",
        "-p",
        "MemoryMax=4000000000",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"RuntimeMaxSec={cap}",
        "-p",
        "TimeoutStopSec=30",
        "-p",
        "RemainAfterExit=yes",
        "-p",
        "Nice=10",
        "-p",
        f"StandardOutput=append:{PACKET / 'postfit_r1.log'}",
        "-p",
        "StandardError=inherit",
    ]
    for key, value in admission["environment"].items():
        command.extend(["--setenv", f"{key}={value}"])
    command.extend(
        [
            "/usr/bin/flock",
            "--nonblock",
            str(ROOT / admission["gpu_lock"]),
            "/usr/bin/time",
            "-v",
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/kaggle_sequence.py"),
            "--plan",
            str(PACKET / "POSTFIT_SEQUENCE_R1.json"),
        ]
    )
    if (PACKET / "STOP_POST_FIT").exists():
        return
    subprocess.run(command, check=True)
    while service_properties(UNIT).get("MainPID") != "0":
        reclaim_inactive_cache()
        time.sleep(15)
    properties = service_properties(UNIT)
    with (PACKET / "POSTFIT_SERVICE_TERMINAL_R1.json").open("x") as stream:
        json.dump(properties, stream, indent=2)
        stream.write("\n")
    if properties.get("Result") != "success" or properties.get("ExecMainStatus") != "0":
        raise RuntimeError("post-fit sequence failed; inspect its finite step report")


if __name__ == "__main__":
    main()
