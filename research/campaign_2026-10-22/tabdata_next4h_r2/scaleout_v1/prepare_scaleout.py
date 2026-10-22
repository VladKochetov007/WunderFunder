"""Seal a separate twelve-fit scaleout under the original node deadline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
BASE = ROOT / "runtime/next4h-20261022-r2"
PACKET = ROOT / "runtime/next4h-20261022-scaleout"
STATE = ROOT / "state/next4h-20261022-scaleout"
UNIT = "tabdata-next4h-20261022-scaleout.service"
BASE_HASH = "d76774e58d8d8c06c52bac701bde97ea2aa704f2536ac2f4815775e6a5e1eefd"


def sha(path: Path) -> str:
    """Return the exact small code/manifest content digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Seal before executing; the timer is only the remaining original allowance."""
    assert sha(BASE / "PLAN.json") == BASE_HASH
    start = json.loads((ROOT / "state/next4h-20261022-r2/START.json").read_text())["start_unix"]
    assert start == 1792693855.4927282
    STATE.mkdir(parents=True, exist_ok=True)
    jobs = []
    base = json.loads((BASE / "PLAN.json").read_text())
    devices = {j["gpu_index"]: j["gpu_uuid"] for j in base["jobs"]}
    for pair, (fold, seed, gpu) in enumerate(
        [(0, 1301, 2), (0, 1302, 3), (0, 1303, 4), (2, 1401, 5), (2, 1402, 6), (2, 1403, 7)]
    ):
        for index, arm in enumerate(("wmse", "nll")):
            cpu = 64 + 4 * (pair * 2 + index)
            jobs.append(
                {
                    "id": f"scaleout_f{fold}_s{seed}_{arm}",
                    "family": "nll",
                    "fold": fold,
                    "seed": seed,
                    "arm": arm,
                    "gpu_index": gpu,
                    "gpu_uuid": devices[gpu],
                    "cpus": list(range(cpu, cpu + 4)),
                }
            )
    runtime = int(start + 13500 - time.time()) - 30
    assert runtime > 600
    code = [PACKET / name for name in ("scaleout.py", "scaleout_publish.py", "prepare_scaleout.py")]
    assert all(p.is_file() for p in code)
    assert not (PACKET / "SIDEPLAN.json").exists()
    plan = {
        "jobs": jobs,
        "original_start_unix": start,
        "deadline_unix": start + 14400,
        "unit": UNIT,
        "base_plan_sha256": BASE_HASH,
        "runtime_max_seconds": runtime,
        "epochs": 3,
        "fixed_epoch": 2,
        "fit_cutoff_unix": start + 11100,
        "export_cutoff_unix": start + 12900,
        "grade_cutoff_unix": start + 13140,
        "files": {str(p): sha(p) for p in code},
        "initial_density": "GPUs0-1:2 base NLL fits; GPUs2-7:1 base Transformer +2 scaleout NLL fits",
        "evaluation_exposure": "Two previously exposed development folds; no new holdout",
        "scientific_claim": "Six additional paired seed cells, not six new mechanisms or confirmed quality results",
        "comparison": "Cold label NLL versus WMSE, identical common auxiliary losses, initialization, membership and fixed budget",
        "decision": "Evaluate the official paired deltas and seed dispersion; report incomplete endpoints as progress only",
    }
    for name in ("PLAN.json", "delivery.py", "nll_profile.py", "nll_grade.py"):
        plan["files"][str(BASE / name)] = sha(BASE / name)
    path = PACKET / "SIDEPLAN.json"
    with path.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    digest = sha(path)
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--working-directory=" + str(ROOT),
        "--property=RuntimeMaxSec=" + str(runtime),
        "--property=TimeoutStopSec=120",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=SendSIGKILL=yes",
        "--property=ExecStopPost=/usr/bin/timeout 110 "
        + str(ROOT / ".venv/bin/python")
        + " "
        + str(PACKET / "scaleout_publish.py")
        + " --final",
    ]
    env = {
        "TABDATA_SERVICE_UNIT": UNIT,
        "TABDATA_BASE_PLAN_SHA256": BASE_HASH,
        "TABDATA_SCALEOUT_PLAN_SHA256": digest,
        "TABDATA_PLAN_SHA256": BASE_HASH,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONPATH": str(BASE / "sources/nll/src"),
    }
    for key, value in env.items():
        argv.append("--setenv=" + key + "=" + value)
    os.environ.update(env)
    sys.path.insert(0, str(PACKET))
    import scaleout

    scaleout.verify(base, plan)
    argv += [str(ROOT / ".venv/bin/python"), "-u", str(PACKET / "scaleout.py")]
    subprocess.run(argv, check=True, timeout=20)
    print(
        json.dumps(
            {
                "unit": UNIT,
                "side_plan_sha256": digest,
                "jobs": len(jobs),
                "remaining_runtime_seconds": runtime,
                "hard_deadline_unix": start + 14400,
            }
        )
    )


if __name__ == "__main__":
    main()
