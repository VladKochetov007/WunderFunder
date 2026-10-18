"""Hash eight exact W18 ancestor inputs and launch one reviewed bounded metadata extraction."""

import argparse
import base64
import hashlib
import json
import pathlib
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--review-sha", required=True)
p.add_argument("--manifest-sha", required=True)
p.add_argument("--wrapper-review-sha", required=True)
a = p.parse_args()
R = pathlib.Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
Q = R / "research/campaign_2026-10-18/quality_gap_r1"
D = Q / "tl3_w18_checkpoint_metadata_r1"
E = Q / "tl3_w18_checkpoint_execution_r1"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


mraw = (D / "PREPARATION_MANIFEST.json").read_bytes()
assert sha(mraw) == a.manifest_sha
m = json.loads(mraw)
for name, spec in m["artifacts"].items():
    assert sha((D / name).read_bytes()) == spec["sha256"]
reviewraw = (D / "INDEPENDENT_REVIEW.json").read_bytes()
assert sha(reviewraw) == a.review_sha
review = json.loads(reviewraw)
assert review.get("blockers") == []
assert review["decision"].startswith("PASS")
assert review["preparation_manifest_sha256"] == sha(mraw) and review["blockers"] == []
wrapper_path = pathlib.Path(__file__)
wrapper_raw = wrapper_path.read_bytes()
wrapper_review_raw = (E / "ROOT_WRAPPER_REVIEW_V1.json").read_bytes()
assert sha(wrapper_review_raw) == a.wrapper_review_sha
wrapper_review = json.loads(wrapper_review_raw)
assert wrapper_review["decision"].startswith("PASS") and wrapper_review["blockers"] == []
assert wrapper_review["launch_sha256"] == sha(wrapper_raw)
assert wrapper_review["capture_sha256"] == sha((E / "root_capture_v1.py").read_bytes())
assert (
    sha((E / "ROOT_PATH_PREFLIGHT_V1.json").read_bytes())
    == "b555f6e282e45129be74d7614da8a77fc42b2a684da06c6c7aa6d83ae8acc5ba"
)
template = json.loads((D / "INPUT_MANIFEST_TEMPLATE_R1.json").read_bytes())
helper_names = [
    "extract_w18_checkpoint_metadata.py",
    "w18_metadata_core.py",
    "w18_metadata_summary.py",
]
helper_files = {name: base64.b64encode((D / name).read_bytes()).decode() for name in helper_names}
assert all(name in m["artifacts"] for name in helper_names)
script = (
    "wrapper_sha="
    + repr(sha(wrapper_raw))
    + "\nwrapper_review_sha="
    + repr(a.wrapper_review_sha)
    + "\ntemplate="
    + repr(template)
    + "\nhelper_files="
    + repr(helper_files)
    + "\nreview_sha="
    + repr(a.review_sha)
    + "\nmanifest_sha="
    + repr(a.manifest_sha)
    + "\n"
    + r"""
import pathlib, subprocess, json, hashlib, datetime, os, base64, stat, time

B = pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018")
unit = "computer-vision-homework-tl3-w18-checkpoint-metadata-v1.service"
E = B / "quality-gap-r1/tl3-w18-checkpoint-metadata-v1"
records = B / "receipts"
start = time.monotonic()
names = [
    "TL3_W18_CHECKPOINT_METADATA_ADMISSION_V1.json",
    "TL3_W18_CHECKPOINT_METADATA_INPUTS_R1.json",
    "TL3_W18_CHECKPOINT_METADATA_R1.json",
    "TL3_W18_CHECKPOINT_METADATA_LAUNCH_V1.json",
]
for n in names:
    assert not os.path.lexists(records / n), n
assert not os.path.lexists(E)
for ancestor in E.parents:
    if ancestor == B:
        break
    assert not ancestor.is_symlink()
p = subprocess.run(
    ["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"],
    capture_output=True,
    text=True,
    timeout=5,
)
assert p.stdout.strip() == "not-found"


def cmd(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert p.returncode == 0, (args, p.stderr)
    return p.stdout


props = dict(
    line.split("=", 1)
    for line in cmd(
        [
            "systemctl",
            "--user",
            "show",
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
assert (
    props["CPUQuotaPerSecUSec"] == "2s"
    and props["MemoryMax"] == "20000000000"
    and props["MemoryHigh"] == "18000000000"
    and props["MemorySwapMax"] == "0"
)
cg = pathlib.Path("/sys/fs/cgroup") / props["ControlGroup"].lstrip("/")
assert (cg / "cgroup.procs").read_text().strip() == ""
active = []
child_usage = 0
reserved = 0
quota = 0
sequence = json.loads((B / "runtime/receipts/RANK01_F3_S1001_SEQUENCE.json").read_bytes())
assert (
    sequence["status"] == "running"
    and sequence["plan_sha256"]
    == "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"
)
assert (
    sequence["attempt"].startswith("computer-vision-homework-r01-")
    and "-f3s1001-ep" in sequence["attempt"]
)
expected = {
    "computer-vision-homework-r01-f3-sequence.service": (0.1, 512000000),
    "wnn-bandit-" + sequence["attempt"] + ".service": (1.0, 8000000000),
}
for child in cg.iterdir():
    if not child.is_dir():
        continue
    if (child / "cgroup.events").exists() and dict(line.split() for line in (child / "cgroup.events").read_text().splitlines()).get("populated") == "1":
        assert child.name in expected, child.name
        cap, period = (child / "cpu.max").read_text().split()
        assert cap != "max"
        cores = int(cap) / int(period)
        memory = int((child / "memory.max").read_text())
        usage = int((child / "memory.current").read_text())
        assert (cores, memory) == expected[child.name]
        assert (child / "memory.swap.max").read_text().strip() == "0"
        active.append(
            {"unit": child.name, "cpu_cores": cores, "memory_max": memory, "memory_current": usage}
        )
        child_usage += usage
        reserved += memory
        quota += cores
assert {x["unit"] for x in active} == set(expected)
assert quota + 0.5 <= 2
committed = max(0, int(props["MemoryCurrent"]) - child_usage) + reserved + 1500000000
assert committed < 17000000000, committed
scratch = cmd(["findmnt", "-T", str(B / "research_tmp"), "-no", "FSTYPE,SOURCE,TARGET"])
assert scratch.split()[0] not in ["tmpfs", "ramfs"]
assert os.statvfs(B).f_bavail * os.statvfs(B).f_frsize > 2000000000


def signature(s):
    return {
        "device": s.st_dev,
        "inode": s.st_ino,
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "ctime_ns": s.st_ctime_ns,
        "mode": s.st_mode,
    }


bound = []
aliases = []
assert len(template["inputs"]) == 8
for item in template["inputs"]:
    p = pathlib.Path(item["logical_path"])
    assert p.resolve(strict=True) == p
    assert not p.is_symlink() and all(not parent.is_symlink() for parent in p.parents)
    s = p.lstat()
    assert (
        stat.S_ISREG(s.st_mode)
        and s.st_size == item["expected_bytes"]
        and s.st_size <= 2 * 1024 * 1024
    )
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    e = p.lstat()
    assert signature(s) == signature(e)
    item["expected_sha256"] = h.hexdigest()
    item["expected_generation"] = signature(s)
    item["source_resolution"] = {"resolved_path": str(p), "symlinks": []}
    bound.append(
        {
            "id": item["id"],
            "logical_path": str(p),
            "resolved_path": str(p),
            "bytes": s.st_size,
            "sha256": h.hexdigest(),
            "generation": signature(s),
        }
    )
assert sum(x["bytes"] for x in bound) == 8433688
assert time.monotonic() - start < 60

admission = {
    "schema": "quality_gap.tl3_w18_checkpoint_metadata_admission.v1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "ADMITTED_METADATA_ONLY",
    "manifest_sha256": manifest_sha,
    "review_sha256": review_sha,
    "helper_sha256": hashlib.sha256(base64.b64decode(helper_files["extract_w18_checkpoint_metadata.py"])).hexdigest(),
    "helper_dependency_sha256": {name:hashlib.sha256(base64.b64decode(data)).hexdigest() for name,data in helper_files.items()},
    "bound_inputs": bound,
    "root_launch_wrapper_sha256": wrapper_sha,
    "root_wrapper_review_sha256": wrapper_review_sha,
    "fresh_parent_properties": props,
    "active_children": active,
    "committed_with_support_bytes": committed,
    "total_reserved_cpu_with_support": quota + 0.5,
    "memory_events_before": (cg / "memory.events").read_text(),
    "physical_scratch": scratch,
    "limits": template["resource_contract"],
    "no_training_or_inference": True,
    "restricted_loader_only": True,
}


def write(n, obj):
    data = json.dumps(obj, indent=2, sort_keys=True).encode() + b"\n"
    with (records / n).open("xb") as f:
        f.write(data)
    return data


ad = write(names[0], admission)
template["status"] = "ROOT_HASH_BOUND"
template["root_admission_receipt"] = {
    "path": str(records / names[0]),
    "sha256": hashlib.sha256(ad).hexdigest(),
}
inp = write(names[1], template)
E.mkdir()
assert set(helper_files)=={"extract_w18_checkpoint_metadata.py", "w18_metadata_core.py", "w18_metadata_summary.py"}
for name,data in helper_files.items():
 raw=base64.b64decode(data)
 with (E/name).open("xb") as f:f.write(raw)
 assert hashlib.sha256((E/name).read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
with (E / names[1]).open("xb") as f:
    f.write(inp)
assert sorted(p.name for p in E.iterdir()) == sorted(list(helper_files)+[names[1]])
command = [
    "systemd-run",
    "--user",
    "--unit=" + unit,
    "--property=Type=exec",
    "--property=RemainAfterExit=yes",
    "--property=Slice=wnn-campaign.slice",
    "--property=CPUQuota=50%",
    "--property=MemoryMax=1500000000",
    "--property=MemorySwapMax=0",
    "--property=RuntimeMaxSec=120",
    "--property=TimeoutStopSec=15",
    "--working-directory=" + str(E),
]
for k, v in {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TMPDIR": str(B / "research_tmp"),
}.items():
    command += ["--setenv=" + k + "=" + v]
command += [
    "/home/grinchenkoeg/WunderFunder/.venv/bin/python",
    str(E / "extract_w18_checkpoint_metadata.py"),
    "--manifest",
    str(E / names[1]),
    "--output",
    str(records / names[2]),
]
p = subprocess.run(command, capture_output=True, text=True, timeout=20)
write(
    names[3],
    {
        "schema": "quality_gap.tl3_w18_checkpoint_metadata_launch.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "unit": unit,
        "command": command,
        "exit_status": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "admission_sha256": hashlib.sha256(ad).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(inp).hexdigest(),
    },
)
print(
    json.dumps(
        {
            n: base64.b64encode((records / n).read_bytes()).decode()
            for n in [names[0], names[1], names[3]]
        }
    )
)
assert p.returncode == 0
"""
)
assert E.is_dir()
local_outputs = [
    "transport_stdout.json",
    "transport_stderr.txt",
    "TL3_W18_CHECKPOINT_METADATA_ADMISSION_V1.json",
    "TL3_W18_CHECKPOINT_METADATA_INPUTS_R1.json",
    "TL3_W18_CHECKPOINT_METADATA_LAUNCH_V1.json",
    "TL3_W18_CHECKPOINT_METADATA_R1.json",
    "EXECUTION_V1.json",
    "STOP_V1.json",
]
for name in local_outputs:
    assert not (E / name).exists() and not (E / name).is_symlink(), name
p = subprocess.run(
    [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "beleriand",
        "python3 -B -",
    ],
    input=script.encode(),
    capture_output=True,
    timeout=100,
)
with (E / "transport_stdout.json").open("xb") as stream:
    stream.write(p.stdout)
with (E / "transport_stderr.txt").open("xb") as stream:
    stream.write(p.stderr)
assert p.returncode == 0, (p.returncode, p.stderr.decode())
for name, value in json.loads(p.stdout).items():
    with (E / name).open("xb") as f:
        f.write(base64.b64decode(value))
print(json.dumps({"status": "launched", "records": str(E)}))
