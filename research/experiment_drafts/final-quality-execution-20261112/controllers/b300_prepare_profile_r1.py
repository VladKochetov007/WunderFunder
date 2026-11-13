"""Prepare and profile the frozen PR91 input-interaction study on owned B300s."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata-final-quality-20261112-r1")
PREVIOUS = Path("/data/epanov_s/tabdata-all-20261112-r1")
WARM = Path("/data/epanov_s/tabdata-qknorm-b300-20261111-r1")
RUNTIME = PREVIOUS / "runtimes/pr75"
PACKET = ROOT / "intake/pr91-source/research/experiment_drafts/bilinear-entmax-b300-20261112"
PYTHON = "/data/epanov_s/tabdata/.venv/bin/python"
CELLS = (
    "be_input_linear_s206201",
    "be_input_bilinear_s206201",
    "be_input_linear_s206202",
    "be_input_bilinear_s206202",
)


def main() -> None:
    """Run acceptance or four profiles, without automatic approval or fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare-check", "profiles"))
    args = parser.parse_args()
    sys.path.insert(0, str(WARM / "control/profile-receiver-r1"))
    import b300_driver_core_r1 as core
    import b300_driver_support_r1 as support
    import b300_owned_runner_r2 as runner

    control = ROOT / "control" / args.phase
    control.mkdir(parents=True, exist_ok=False)
    clock_path = ROOT / "control/CLOCK_R1.json"
    if args.phase == "prepare-check":
        now = time.time()
        core.atomic_json(
            clock_path,
            {
                "schema": "wnn.final-quality-wave-clock.r1",
                "started_epoch": now,
                "fit_deadline_epoch": now + 7 * 3600,
                "wave_deadline_epoch": now + 8 * 3600,
                "basis": "Fresh user instruction; closed previous wave clocks are not reused",
            },
        )
    clock = json.loads(clock_path.read_bytes())
    assert time.time() < clock["fit_deadline_epoch"]
    split = WARM / "project/chronological_primary.json"
    assert hashlib.sha256(split.read_bytes()).hexdigest() == (
        "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
    )
    owner = support._identity(ROOT)
    core.atomic_json(control / "owner.json", owner)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    devices: dict[int, str] = {}
    for line in query.splitlines():
        index, uuid, used = [part.strip() for part in line.split(",")]
        if int(used) > 256:
            raise RuntimeError(f"B300 GPU {index} is occupied; no co-launch was admitted")
        devices[int(index)] = uuid
    assert len(devices) == 8
    guard = runner.ResourceGuard(
        ROOT,
        control,
        {
            "owned_rss_bytes": 512 * 1024**3,
            "output_bytes": 128 * 1024**3,
            "minimum_free_bytes": 128 * 1024**3,
        },
    )

    def stage(cell: str, mode: str, indices: list[int], extra: list[str]) -> dict[str, object]:
        timeout = min(1800, int(clock["fit_deadline_epoch"] - time.time()))
        if timeout <= 0:
            raise TimeoutError("Fresh fitting admission window has expired")
        command = [
            PYTHON,
            str(PACKET / "runtime/launch.py"),
            mode,
            "--runtime-root",
            str(RUNTIME),
            "--cell",
            str(PACKET / "configs" / (cell + ".json")),
            "--data-root",
            str(WARM / "project"),
            "--split-manifest",
            str(split),
            "--owner",
            "SemyonEpanov-root-final-quality-20261112",
            "--timeout-seconds",
            str(timeout),
            *extra,
        ]
        uuids = [devices[index] for index in indices]
        environment = support._environment(RUNTIME, WARM, uuids)
        environment["PATH"] = (
            str(PREVIOUS / "tools/git-r2/usr/bin") + ":" + environment.get("PATH", "")
        )
        environment["WNN_MONITOR_HEARTBEAT_SECONDS"] = "10"
        result = runner.run_owned(
            command,
            root=ROOT,
            control=control,
            cell=cell,
            stage=mode,
            gpu_uuids=uuids,
            owner=owner,
            environment=environment,
            deadline=clock["fit_deadline_epoch"],
            stage_timeout=timeout,
            guard=guard,
            progress=lambda: None,
        )
        core.atomic_json(control / (cell + "-" + mode + ".json"), result)
        if result["status"] != "completed":
            raise RuntimeError(f"{cell} {mode} did not complete; inspect retained logs")
        return result

    preparation = ROOT / "runs/pr91-input/preprocessing/preprocessing.json"
    checks = ROOT / "runs/pr91-input/acceptance/checks.json"
    if args.phase == "prepare-check":
        stage(CELLS[0], "prepare", [], ["--output", str(preparation.parent)])
        stage(
            CELLS[0],
            "check",
            [0],
            ["--preprocessing", str(preparation), "--output", str(checks), "--devices", "0"],
        )
    else:
        assert preparation.is_file() and checks.is_file()

        def profile(index: int) -> dict[str, object]:
            cell = CELLS[index]
            return stage(
                cell,
                "profile",
                [index * 2, index * 2 + 1],
                [
                    "--preprocessing",
                    str(preparation),
                    "--checks-receipt",
                    str(checks),
                    "--output",
                    str(ROOT / "runs/pr91-input" / cell / "profile"),
                    "--devices",
                    f"{index * 2},{index * 2 + 1}",
                    "--profile-updates",
                    "50",
                ],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(profile, index) for index in range(4)]
            results = [future.result() for future in futures]
        core.atomic_json(control / "all_profiles.json", results)
    core.atomic_json(
        control / "complete.json", {"phase": args.phase, "completed_epoch": time.time()}
    )


if __name__ == "__main__":
    main()
