"""Stage a sealed canonical inference source and admit one fit-only shape profile."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import subprocess
import zlib
from pathlib import Path

PAYLOAD = json.loads(base64.b64decode("__PAYLOAD_B64__", validate=True))


def main() -> None:
    """Reuse reviewed gates and launch exactly one create-only engineering profile."""
    old = zlib.decompress(base64.b64decode(PAYLOAD["gates_zlib"], validate=True))
    assert hashlib.sha256(old).hexdigest() == PAYLOAD["gates_sha256"]
    gates = {"__name__": "reused_readonly_inference_gates"}
    exec(compile(old, "reviewed_profile_gates", "exec"), gates)
    gates["GPU"] = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
    gates["MIN_GPU_FREE"] = 4_294_967_296 + 1_000_000_000
    b = gates["B"]
    source = Path(PAYLOAD["source_root"])
    output = b / "research_tmp/fg19_teacher_cache_profile_v1"
    unit = "computer-vision-homework-fg19-teacher-cache-profile-v1.service"
    assert source.parent == b and not source.exists() and not source.is_symlink()
    assert not output.exists() and not output.is_symlink()
    assert not (b / "runtime/STOP_ADMISSION.json").exists()
    assert gates["_unit"](unit, ["LoadState"])["LoadState"] == "not-found"
    assert gates["_sha"](b / "splits/fold0.json") == PAYLOAD["contract"]["split_sha256"]
    data = gates["_data_identity"]()
    parent, populated, cgroup = gates["_parent"]()
    gpu, foreign = gates["_gpu"]()
    scratch = gates["_run"](["findmnt", "-T", str(b / "research_tmp"), "-n", "-o", "FSTYPE"])
    assert scratch.strip() not in ("tmpfs", "ramfs")
    disk = os.statvfs(b / "research_tmp")
    assert disk.f_bavail * disk.f_frsize > 40 * (1 << 30)
    teachers = []
    for teacher in PAYLOAD["contract"]["teachers"]:
        run = b / "data/artifacts/runs" / teacher["run"]
        observed = {}
        for name, key in (
            ("model_ep2.pt", "checkpoint_sha256"),
            ("provenance.json", "provenance_sha256"),
        ):
            path = run / name
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size < 32_000_000
            observed[name] = gates["_sha"](path)
            assert observed[name] == teacher[key]
        acceptance = base64.b64decode(PAYLOAD["acceptances"][teacher["run"]], validate=True)
        assert hashlib.sha256(acceptance).hexdigest() == teacher["acceptance_sha256"]
        accepted = json.loads(acceptance)
        assert (
            accepted["status"] == "ACCEPTED"
            and accepted["checkpoint_sha256"] == teacher["checkpoint_sha256"]
        )
        teachers.append(
            {
                "run": teacher["run"],
                "fresh_dependencies": observed,
                "acceptance_sha256": teacher["acceptance_sha256"],
            }
        )
    files = PAYLOAD["files"]
    assert 164 <= len(files) <= 180
    assert (
        sum(len(base64.b64decode(item["base64"], validate=True)) for item in files.values())
        < 8_000_000
    )
    for name, item in files.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        raw = base64.b64decode(item["base64"], validate=True)
        assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
    # All observations and payload validation precede the first mutation.
    source.mkdir()
    for name, item in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(base64.b64decode(item["base64"], validate=True))
    assert {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()} == set(files)
    for name, item in files.items():
        assert not (source / name).is_symlink() and gates["_sha"](source / name) == item["sha256"]
    output.mkdir()
    argv = [
        str(gates["PYTHON"]),
        "-u",
        str(source / "scripts/predict_run.py"),
        "fg19_t3_h384x2_e3_s3_v1",
        "train",
        "--epoch",
        "2",
        "--campaign-split-manifest",
        str(b / "splits/fold0.json"),
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
        "--loader-workers",
        "0",
        "--profile-only",
        "--profile-output",
        str(output / "PROFILE.json"),
    ]
    command = ["systemd-run", "--user", "--unit=" + unit]
    properties = [
        "Type=exec",
        "RemainAfterExit=yes",
        "Slice=wnn-campaign.slice",
        "CPUQuota=100%",
        "MemoryHigh=7500000000",
        "MemoryMax=8000000000",
        "MemorySwapMax=0",
        "RuntimeMaxSec=600",
        "StandardOutput=append:" + str(output / "unit.log"),
        "StandardError=append:" + str(output / "unit.log"),
    ]
    command += ["--property=" + item for item in properties]
    environment = dict(
        CUDA_VISIBLE_DEVICES=gates["GPU"],
        WNN_PROJECT_ROOT=str(b / "data"),
        PYTHONPATH=str(source / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        WNN_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        WNN_GPU_MEMORY_BYTES="4294967296",
        WNN_GPU_MEMORY_RESERVE_BYTES="1000000000",
        WNN_LOADER_WORKERS="0",
        WNN_LOADER_RAM_GB="1",
        TMPDIR=str(b / "research_tmp"),
    )
    command += ["--setenv=" + key + "=" + value for key, value in environment.items()]
    command += ["--working-directory=" + str(source), *argv]
    admission = dict(
        utc=dt.datetime.now(dt.UTC).isoformat(),
        unit=unit,
        source_commit=PAYLOAD["source_commit"],
        source_files={name: item["sha256"] for name, item in files.items()},
        teachers=teachers,
        contract_sha256=PAYLOAD["contract_sha256"],
        data=data,
        parent=parent,
        populated=populated,
        gpu=gpu,
        foreign_preserved=foreign,
        memory_events=(cgroup / "memory.events").read_text(),
        memory_events_local=(cgroup / "memory.events.local").read_text(),
        command=command,
    )
    gates["_write"](output / "ADMISSION.json", admission)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    launch = dict(
        utc=dt.datetime.now(dt.UTC).isoformat(),
        unit=unit,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        post=gates["_unit"](
            unit, ["InvocationID", "ActiveState", "SubState", "MainPID", "ControlGroup"]
        ),
        unknown_outcome_policy="Never retry; inspect exact unit and durable receipts.",
    )
    gates["_write"](output / "LAUNCH.json", launch)
    print(json.dumps(dict(admission=admission, launch=launch)))


if __name__ == "__main__":
    main()
