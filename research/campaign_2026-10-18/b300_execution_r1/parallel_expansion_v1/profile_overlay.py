"""Profile native feature successors beside the unchanged active B300 fits."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
PACKET = ROOT / "runtime/parallel-expansion-v1"
STATE = ROOT / "state/parallel-expansion-profile-v1"
PR15 = ROOT / "runtime/pr15-host-v2"
R01 = ROOT / "runtime/b300-r01-v1"
SOURCE = ROOT / "runtime/source-a266-lf-subset-v1"
PATTERN = re.compile(r"ep (\d+) batch (\d+)/(\d+).*?seq/s ([0-9.]+)")


def save(path: Path, value: dict) -> None:
    """Retain this bounded profile's current observation atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def progress() -> dict:
    """Read existing training logs without signals or GPU initialization."""
    rows = {}
    for state in ("r01-execution-v4", "r01-density4-v1", "pr15-execution-v2"):
        for path in (ROOT / "state" / state / "logs").glob("*.log"):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 8000))
                matches = PATTERN.findall(stream.read().decode(errors="replace"))
            if matches:
                ep, batch, total, _ = matches[-1]
                rows[path.stem] = dict(
                    batch_index=int(ep) * int(total) + int(batch), log_mtime=path.stat().st_mtime
                )
    return rows


def run_profiles() -> None:
    """Prepare four exact native modes and profile two extra actors at a time."""
    sys.path[:0] = [str(R01 / "code"), str(PR15)]
    import hashlib

    from adapter_common import _RECIPE, _TREE_MANIFEST, _json, verify_deployed_tree_manifest
    from fit_recipe import build_training_argv
    from fit_workspace import prepare_workspace
    from worker import _make_request, _phase_environment

    verify_deployed_tree_manifest(
        SOURCE,
        _json(_TREE_MANIFEST),
        manifest_sha256=hashlib.sha256(_TREE_MANIFEST.read_bytes()).hexdigest(),
    )
    STATE.mkdir(exist_ok=False)
    (STATE / "logs").mkdir()
    (STATE / "work").mkdir()
    devices = dict(
        line.strip().split(", ")
        for line in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
    )
    save(
        STATE / "INITIAL.json",
        dict(time_unix=time.time(), progress=progress(), source_manifest=str(_TREE_MANIFEST)),
    )
    deadline = time.monotonic() + 1800
    cohort = [(1, [84, 86], "field_age", "none"), (5, [85, 87], "causal_innovations", "none")]
    following = [(1, [84, 86], "group_age", "none"), (5, [85, 87], "causal_innovations", "newest")]
    with (STATE / "observations.jsonl").open("x", buffering=1) as observations:
        baseline_end = time.monotonic() + 130
        while time.monotonic() < baseline_end:
            observations.write(
                json.dumps(dict(time_unix=time.time(), phase="baseline", progress=progress()))
                + "\n"
            )
            time.sleep(5)
        for phase, jobs in enumerate((cohort, following)):
            processes = []
            handles = []
            try:
                for gpu, cpus, mode, control in jobs:
                    name = f"expansion_profile_{mode}_{control}_g{gpu}"
                    work = STATE / "work" / name
                    work.mkdir()
                    split = ROOT / "splits/fold0.json"
                    ancestry = prepare_workspace(work, ROOT / "data", split, 0)
                    args = build_training_argv(
                        _json(_RECIPE),
                        arm="wide",
                        fold=0,
                        seed=1199,
                        run_name=name,
                        split_manifest=split,
                        mode="profile",
                    )
                    args[args.index("--profile-steps") + 1] = "600"
                    args[args.index("--profile-warmup") + 1] = "100"
                    args += ["--research-features", mode, "--research-feature-control", control]
                    request = work / "worker_request.json"
                    _make_request(
                        request,
                        mode="profile",
                        name=name,
                        source_root=SOURCE,
                        work=work,
                        train_argv=args,
                        predict_argv=None,
                    )
                    save(work / "INPUTS.json", ancestry)
                    env = _phase_environment(
                        dict(os.environ), work=work, gpu_uuid=devices[str(gpu)], phase="profile"
                    )
                    env["PYTHONPATH"] = str(SOURCE / "scripts")
                    argv = [
                        sys.executable,
                        "-B",
                        "-u",
                        str(PR15 / "gpu_entrypoint.py"),
                        str(PR15 / "train_cell.py"),
                        "--request",
                        str(request),
                    ]
                    log = STATE / "logs" / (name + ".log")
                    handle = log.open("x")
                    handles.append(handle)
                    child = subprocess.Popen(
                        argv,
                        env=env,
                        cwd=SOURCE,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        preexec_fn=lambda ids=cpus: os.sched_setaffinity(0, set(ids)),
                    )
                    processes.append((name, child))
                while any(p.poll() is None for _, p in processes):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Finite native feature profile window expired")
                    observations.write(
                        json.dumps(
                            dict(
                                time_unix=time.time(), phase=f"overlay{phase}", progress=progress()
                            )
                        )
                        + "\n"
                    )
                    time.sleep(5)
                outcomes = {name: p.returncode for name, p in processes}
                save(
                    STATE / f"COHORT_{phase}.json", dict(time_unix=time.time(), exit_codes=outcomes)
                )
                if any(outcomes.values()):
                    raise RuntimeError(
                        "A feature profile failed; retained its exact request and log"
                    )
            finally:
                for _, child in processes:
                    if child.poll() is None:
                        os.killpg(child.pid, 15)
                for handle in handles:
                    handle.close()
    save(
        STATE / "COMPLETE.json",
        dict(time_unix=time.time(), status="profiles_completed", scientific_fits=0),
    )


if __name__ == "__main__":
    run_profiles()
