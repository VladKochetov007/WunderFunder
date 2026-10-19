B = pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DEST = B / "support-source/eda19-r9-v1"
OUT = B / "research_tmp/nf19-r9-v1"
UNIT = "computer-vision-homework-eda19-r9-v1.service"
PY = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    with p.open("x") as f:
        json.dump(x, f, indent=2)
        f.write("\n")


def props(unit):
    p = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "ControlGroup",
            "-p",
            "CPUQuotaPerSecUSec",
            "-p",
            "MemoryHigh",
            "-p",
            "MemoryMax",
            "-p",
            "MemorySwapMax",
            "-p",
            "MemoryCurrent",
            "-p",
            "ExecMainStatus",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return dict(s.split("=", 1) for s in p.stdout.splitlines() if "=" in s)


def age(s):
    return (
        datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(s)
    ).total_seconds()


assert not DEST.exists() and not OUT.exists(), "create-only destination/output collision"
assert props(UNIT).get("LoadState") == "not-found", "support unit already exists"
planpath = (
    B
    / "research/campaign_2026-10-18/priority25_execution_r4/f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
)
assert sha(planpath) == "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"
plan = json.loads(planpath.read_bytes())
# Reuse only pure pointer/resource validation from the already-reviewed F3 helper.
P = B / "research/campaign_2026-10-18/priority25_execution_r4/f3_terminal_execution_r1"
helper_pins = __GATE_HELPER_PINS_REPR__
for name, digest in helper_pins.items():
    assert sha(P / name) == digest, "existing resource helper bytes differ"
import importlib.util, sys
sys.dont_write_bytecode = True
assert {x.name for x in P.iterdir()} == set(helper_pins), "existing resource helper inventory differs"
spec = importlib.util.spec_from_file_location("nf19_existing_f3_resource_gate", P / "terminal_gate.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
controller = "computer-vision-homework-r01-f3-sequence.service"

def observe():
    sequence = json.loads((B / "runtime/receipts/RANK01_F3_S1001_SEQUENCE.json").read_bytes())
    current = gate.validate_pointer(plan, sequence, "wide")
    if current is not None:
        assert 0 <= age(sequence["heartbeat_utc"]) < 120
        assert 0 <= age(sequence["worker_heartbeat"]) < 120
    parent = props("wnn-campaign.slice")
    cg = pathlib.Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
    direct = (cg / "cgroup.procs").read_text().strip()
    populated = {}
    for path in cg.rglob("cgroup.procs"):
        text = path.read_text().strip()
        if text:
            rel = path.parent.relative_to(cg)
            assert rel.parts
            populated[str(rel)] = text.split()
    units = {name: props(name) for name in sorted({x.split("/")[0] for x in populated})}
    top = []
    for name, unit_props in units.items():
        root = cg / name
        quota, period = (root / "cpu.max").read_text().split()
        assert quota != "max"
        top.append(dict(unit=name, cpu_cores=int(quota)/int(period),
            memory_max=int(unit_props["MemoryMax"]),
            kernel_memory_max=(root / "memory.max").read_text().strip(),
            memory_current=int((root / "memory.current").read_text()),
            swap_max=int((root / "memory.swap.max").read_text()), populated=True))
        assert unit_props["ActiveState"] == "active" and unit_props["SubState"] == "running"
    snapshot = dict(parent_properties=parent,parent_cgroup_procs=direct,
                    top_units=top,populated_paths=sorted(populated))
    # 'verifier' here reserves the identical .5CPU/1.5GB envelope only.
    # No terminal operation, native validation, collection or verification is launched.
    admission = gate.validate_resource_snapshot(snapshot,current,"verifier")
    return sequence,current,parent,cg,populated,units,dict(snapshot=snapshot,admission=admission)

seq,current,parent,cg,populated,units,resource_admission = observe()
worker = "wnn-bandit-"+seq["attempt"]+".service" if current else None
events = (cg / "memory.events").read_text()
events_local = (cg / "memory.events.local").read_text()
assert dict(x.split() for x in events.splitlines())["oom"] == "0"
assert dict(x.split() for x in events.splitlines())["oom_kill"] == "0"
scratch = B / "research_tmp"
fs = subprocess.run(
    ["findmnt", "-n", "-o", "FSTYPE", "-T", str(scratch)], capture_output=True, text=True, timeout=5
)
assert fs.returncode == 0 and fs.stdout.strip() not in ("tmpfs", "ramfs")
assert shutil.disk_usage(scratch).free > 2000000000
meminfo = dict(
    line.replace(":", "").split()[:2]
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
)
assert int(meminfo["MemAvailable"]) * 1024 > 5000000000
gpu = subprocess.run(
    [
        "nvidia-smi",
        "-i",
        "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
    timeout=10,
)
assert gpu.returncode == 0
gpu_rows = []
for line in gpu.stdout.splitlines():
    cols = [x.strip() for x in line.split(",")]
    if len(cols) != 3:
        continue
    proc = pathlib.Path("/proc") / cols[0]
    try:
        pcg = (proc / "cgroup").read_text()
    except FileNotFoundError:
        continue
    gpu_rows.append(
        dict(
            pid=int(cols[0]),
            uuid=cols[1],
            memory=cols[2],
            cgroup=pcg,
            campaign="wnn-campaign.slice" in pcg,
        )
    )
ours = [d for d in gpu_rows if d["campaign"]]
gpu_expected = current is not None and current["kind"] != "paired_official_grade"
assert len(ours) == (1 if gpu_expected else 0), gpu_rows
if gpu_expected:
    assert worker in ours[0]["cgroup"], gpu_rows
sample = B / "research_tmp/nf19-sample-r3-v2/SAMPLE.json"
assert sha(sample) == "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
checkpoint = B / "checkpoint_snapshots/rr18_r01_wide_f0_s1001_e6/epoch5_terminal/model_ep5.pt"
assert checkpoint.stat().st_size == 8905959
assert sha(checkpoint) == "78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f"
canonical = __CANONICAL_REPR__
for name, digest in canonical.items():
    assert sha(B / "source-a266b036/src/wnn" / name) == digest
DEST.mkdir()
for name, entry in CFG["files"].items():
    rel = pathlib.PurePosixPath(name)
    assert not rel.is_absolute() and ".." not in rel.parts
    assert len(rel.parts) >= 2 and rel.parts[0] in {
        "idea_screens_r1",
        "idea_screens_r2",
        "idea_screens_r3",
        "idea_screens_r6",
        "idea_screens_r9",
    }
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(entry["base64"], validate=True)
    assert len(payload) == entry["bytes"] and hashlib.sha256(payload).hexdigest() == entry["sha256"]
    with target.open("xb") as f:
        f.write(payload)
    assert sha(target) == entry["sha256"]
# Independent pre-invocation full byte check, before importing any staged module.
for name, entry in CFG["files"].items():
    assert sha(DEST / name) == entry["sha256"]
for name, digest in canonical.items():
    assert sha(B / "source-a266b036/src/wnn" / name) == digest
latest,fresh_current,parent,cg,populated,units,fresh_resource = observe()
assert latest["index"] == seq["index"] and latest["attempt"] == seq["attempt"]
assert latest["status"] == seq["status"] and fresh_current == current
resource_admission = dict(initial=resource_admission,prelaunch=fresh_resource)
events_initial, events_local_initial = events, events_local
events = (cg / "memory.events").read_text()
events_local = (cg / "memory.events.local").read_text()
assert dict(x.split() for x in events.splitlines())["oom"] == "0"
assert dict(x.split() for x in events.splitlines())["oom_kill"] == "0"
assert not OUT.exists()
inline = __INLINE_REPR__
command = [
    "systemd-run",
    "--user",
    "--unit=" + UNIT,
    "--slice=wnn-campaign.slice",
    "--property=Type=exec",
    "--property=CPUQuota=50%",
    "--property=MemoryHigh=1400000000",
    "--property=MemoryMax=1500000000",
    "--property=MemorySwapMax=0",
    "--property=RuntimeMaxSec=600",
    "--property=RemainAfterExit=yes",
    "--property=StandardOutput=append:" + str(DEST / "probe.log"),
    "--property=StandardError=append:" + str(DEST / "probe.log"),
    "--working-directory=" + str(DEST),
    "/usr/bin/env",
    "CUDA_VISIBLE_DEVICES=",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
    "PYTHONPATH=" + str(B / "source-a266b036/src") + ":" + str(DEST),
    "TMPDIR=" + str(scratch),
    PY,
    "-B",
    "-c",
    inline,
]
record = dict(
    schema="nf19.r9_probe_launch.v1",
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    unit=UNIT,
    sequence=latest,
    source_sha256={n: v["sha256"] for n, v in CFG["files"].items()},
    canonical_source_sha256=canonical,
    sample_sha256=sha(sample),
    admission=dict(
        parent=parent,
        resource_admission=resource_admission,
        helper_pins=helper_pins,
        units=units,
        populated=populated,
        memory_events=events,
        memory_events_initial=events_initial,
        memory_events_local_initial=events_local_initial,
        memory_events_local=events_local,
        gpu_processes=gpu_rows,
        scratch_fs=fs.stdout.strip(),
        scratch_free=shutil.disk_usage(scratch).free,
        host_mem_available=int(meminfo["MemAvailable"]) * 1024,
    ),
    command=command,
    output=str(OUT),
)
write(DEST / "ROOT_ADMISSION.json", record)
p = subprocess.run(command, capture_output=True, text=True, timeout=20)
record.update(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
write(DEST / "ROOT_LAUNCH.json", record)
print(json.dumps(record, indent=2))
