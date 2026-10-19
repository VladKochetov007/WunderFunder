"""One exact profile admission; FILES is supplied by the local source binder."""

import base64
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

FILES: dict[str, dict[str, str]] = globals()["FILES"]  # Injected exact-source mapping.

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
SOURCE = B / "source-fg19-propensity-235db75"
PACKET = B / "research/full_gru_tweaks_r1/profile_v1"
OUTPUT = B / "research_tmp/fg19_propensity_profile_v1"
UNIT = "computer-vision-homework-fg19-propensity-profile-v1.service"
GPU = "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c"
PYTHON = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"


def run(args, timeout=15):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((args, result.returncode, result.stderr))
    return result.stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path in [SOURCE, PACKET, OUTPUT]:
    assert not os.path.lexists(path), str(path)
assert not (B / "runtime/STOP_ADMISSION.json").exists()
assert (
    run(["systemctl", "--user", "show", "--all", UNIT, "-p", "LoadState", "--value"]).strip()
    == "not-found"
)
props = dict(
    line.split("=", 1)
    for line in run(
        [
            "systemctl",
            "--user",
            "show",
            "--all",
            "wnn-campaign.slice",
            "-p",
            "ControlGroup",
            "-p",
            "MemoryCurrent",
            "-p",
            "MemoryHigh",
            "-p",
            "MemoryMax",
            "-p",
            "MemorySwapMax",
            "-p",
            "CPUQuotaPerSecUSec",
        ]
    ).splitlines()
)
assert props["ControlGroup"] and props["CPUQuotaPerSecUSec"] == "2s"
assert (
    props["MemoryHigh"] == "18000000000"
    and props["MemoryMax"] == "20000000000"
    and props["MemorySwapMax"] == "0"
)
assert int(props["MemoryCurrent"]) + 8000000000 < 17000000000
cg = Path("/sys/fs/cgroup") / props["ControlGroup"].lstrip("/")
proc_paths = [cg / "cgroup.procs", *cg.rglob("cgroup.procs")]
assert len(proc_paths) < 128
populated = {str(p): p.read_text().strip() for p in proc_paths if p.read_text().strip()}
assert not populated, populated
scratch = run(["findmnt", "-T", str(B / "research_tmp"), "-n", "-o", "FSTYPE,SOURCE,TARGET"])
assert scratch.split()[0] not in ["tmpfs", "ramfs"]
assert os.statvfs(B).f_bavail * os.statvfs(B).f_frsize > 20000000000
gpu_text = run(
    [
        "nvidia-smi",
        "-i",
        GPU,
        "--query-gpu=uuid,name,memory.free,memory.total",
        "--format=csv,noheader",
    ]
)
assert int(gpu_text.strip().split(",")[2].split()[0]) >= 4500
gpu_processes = []
for line in run(
    [
        "nvidia-smi",
        "-i",
        GPU,
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader",
    ]
).splitlines():
    cols = [value.strip() for value in line.split(",")]
    assert len(cols) == 3
    pid_dir = Path("/proc") / cols[0]
    try:
        owner_cg = (pid_dir / "cgroup").read_text()
        uid = pid_dir.stat().st_uid
        cmd = (pid_dir / "cmdline").read_bytes().decode(errors="replace")
    except FileNotFoundError:
        gpu_processes.append({"pid": cols[0], "exited_during_observation": True})
        continue
    assert "wnn-campaign.slice" not in owner_cg and not (uid == os.getuid() and str(B) in cmd)
    gpu_processes.append(
        {"pid": cols[0], "uid": uid, "cgroup": owner_cg, "used_memory": cols[2], "preserved": True}
    )
split = B / "splits/fold0.json"
assert sha(split) == "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
registry = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
assert sha(registry) == "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
registered = json.loads(registry.read_bytes())
data_stats = {}
for relative, entry in registered["sources"].items():
    path = B / "data" / relative
    stat = path.stat()
    assert stat.st_size == entry["original_identity"]["bytes"]
    data_stats[relative] = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "registered_sha256": entry["sha256"],
        "evidence": "fresh stat; full payload validation remains inside capped profile load_campaign_split",
    }
for filename, item in FILES.items():
    path = Path(filename)
    assert path.is_relative_to(SOURCE) or path.is_relative_to(PACKET)
    assert ".." not in path.parts and not os.path.lexists(path)
    for ancestor in path.parents:
        if ancestor == B:
            break
        assert not ancestor.is_symlink(), str(ancestor)
    payload = base64.b64decode(item["base64"], validate=True)
    assert hashlib.sha256(payload).hexdigest() == item["sha256"]
# First mutation follows all resource, ownership and output-absence gates.
for filename, item in FILES.items():
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(base64.b64decode(item["base64"], validate=True))
        stream.flush()
        os.fsync(stream.fileno())
    assert sha(path) == item["sha256"]
OUTPUT.mkdir()
admission = {
    "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "unit": UNIT,
    "source_files": {k: v["sha256"] for k, v in FILES.items()},
    "parent": props,
    "populated": populated,
    "parent_events": (cg / "memory.events").read_text(),
    "parent_events_local": (cg / "memory.events.local").read_text(),
    "gpu": gpu_text,
    "gpu_processes": gpu_processes,
    "scratch": scratch,
    "source_data_stats": data_stats,
    "scope": "engineering profile only; no outer data/labels/quality selection; no campaign model retained",
}
with (OUTPUT / "ADMISSION.json").open("x") as stream:
    json.dump(admission, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
cmd = [
    "systemd-run",
    "--user",
    "--unit=" + UNIT,
    "--property=Type=exec",
    "--property=RemainAfterExit=yes",
    "--property=Slice=wnn-campaign.slice",
    "--property=CPUQuota=100%",
    "--property=MemoryHigh=7500000000",
    "--property=MemoryMax=8000000000",
    "--property=MemorySwapMax=0",
    "--property=RuntimeMaxSec=1200",
    "--property=TimeoutStopSec=45",
    "--property=StandardOutput=append:" + str(OUTPUT / "unit.log"),
    "--property=StandardError=append:" + str(OUTPUT / "unit.log"),
    "--working-directory=" + str(SOURCE),
]
env = {
    "PYTHONPATH": str(SOURCE / "src") + ":" + str(SOURCE / "scripts"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "WNN_PROJECT_ROOT": str(B / "data"),
    "WNN_PORTABLE_DATA_MANIFEST": str(registry),
    "WNN_GPU_MEMORY_BYTES": "3000000000",
    "WNN_LOADER_WORKERS": "0",
    "WNN_LOADER_RAM_GB": "1",
    "WNN_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": GPU,
    "NVIDIA_TF32_OVERRIDE": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TMPDIR": str(B / "research_tmp"),
    "WNN_TMP_ROOT": str(B / "research_tmp"),
}
for key, value in env.items():
    cmd += ["--setenv=" + key + "=" + value]
cmd += [
    PYTHON,
    str(PACKET / "propensity_profile.py"),
    "--source-root",
    str(SOURCE),
    "--data-root",
    str(B / "data"),
    "--split",
    str(split),
    "--output",
    str(OUTPUT),
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
launch = {
    "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "unit": UNIT,
    "command": cmd,
    "returncode": result.returncode,
    "stdout": result.stdout,
    "stderr": result.stderr,
    "unknown_outcome_policy": "never resubmit; inspect exact unit and durable receipt",
}
if result.returncode == 0:
    observed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--all",
            UNIT,
            "-p",
            "InvocationID",
            "-p",
            "ExecStart",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "ControlGroup",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    launch["post_launch_systemctl"] = {
        "returncode": observed.returncode,
        "stdout": observed.stdout,
        "stderr": observed.stderr,
    }

with (OUTPUT / "LAUNCH.json").open("x") as stream:
    json.dump(launch, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(
    json.dumps(
        {
            name: base64.b64encode((OUTPUT / name).read_bytes()).decode("ascii")
            for name in ["ADMISSION.json", "LAUNCH.json"]
        }
    )
)
assert result.returncode == 0
