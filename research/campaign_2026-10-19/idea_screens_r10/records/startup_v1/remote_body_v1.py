import base64
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

B = pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DEST = B / "support-source/eda19-r10-v1"
OUT = B / "research_tmp/nf19-r10-v1"
UNIT = "computer-vision-homework-eda19-r10-v1.service"
PY = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"
SAMPLE = B / "research_tmp/nf19-sample-r3-v2"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
MANIFEST_SHA256 = "__MANIFEST_SHA__"
CANONICAL_DATA_SHA256 = __CANONICAL_PIN_REPR__
RUNTIME_PY = __MODULES_REPR__
EXPECTED_RUNTIME_PY = {
    "idea_screens_r10": "idea_screens_r10/__init__.py",
    "idea_screens_r10.geometry": "idea_screens_r10/geometry.py",
    "idea_screens_r10.panel_io": "idea_screens_r10/panel_io.py",
    "idea_screens_r10.run_probe": "idea_screens_r10/run_probe.py",
}
MANIFEST_REL = "idea_screens_r10/PREPARATION_MANIFEST.json"
EXTRA_REL = {
    MANIFEST_REL,
    "idea_screens_r10/IMPLEMENTATION_REVIEW.json",
    "idea_screens_r10/TEST_RECEIPT.json",
    "idea_screens_r10/README.md",
}
CGROOT = pathlib.Path("/sys/fs/cgroup")
PARENT_UNIT = "wnn-campaign.slice"
PARENT_HIGH = 18_000_000_000
PARENT_MAX = 20_000_000_000
PARENT_RESERVATION_TARGET = 17_000_000_000
NEW_MEMORY_RESERVATION = 1_500_000_000
MAX_FILE_BYTES = 16_000_000
MAX_BUNDLE_BYTES = 64_000_000
GPU_UUID = "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c"


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_create_only(path, value):
    with path.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def read_small_json(path, expected_sha256, limit=16_000_000):
    if path.is_symlink():
        raise RuntimeError("metadata path must be a regular non-symlink file: " + str(path))
    before = path.stat()
    if not path.is_file() or before.st_size > limit:
        raise RuntimeError("metadata file is missing or exceeds the fixed read bound: " + str(path))
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError("metadata file changed during bounded read: " + str(path))
    if sha_bytes(payload) != expected_sha256:
        raise RuntimeError("metadata SHA-256 differs from the frozen R3 pin: " + str(path))
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("metadata root must be a JSON object: " + str(path))
    return payload, value


def systemd_show(unit, fields):
    command = ["systemctl", "--user", "show", unit]
    for field in fields:
        command.extend(["-p", field])
    result = subprocess.run(command, capture_output=True, timeout=10)
    raw = result.stdout
    text = raw.decode("utf-8", "replace")
    stderr_text = result.stderr.decode("utf-8", "replace")
    parsed = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
    return {
        "returncode": result.returncode,
        "stderr": stderr_text,
        "stderr_base64": base64.b64encode(result.stderr).decode("ascii"),
        "stderr_bytes": len(result.stderr),
        "stderr_sha256": sha_bytes(result.stderr),
        "stdout": text,
        "stdout_base64": base64.b64encode(raw).decode("ascii"),
        "stdout_bytes": len(raw),
        "stdout_sha256": sha_bytes(raw),
        "properties": parsed,
    }


def parse_key_values(text):
    parsed = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            parsed[fields[0]] = int(fields[1])
    return parsed


def read_text(path):
    return path.read_text(encoding="ascii").strip()


def walk_cgroup_procs(root):
    checked = []
    populated = []

    def fail_walk(error):
        raise error

    for raw_dir, child_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=fail_walk
    ):
        current = pathlib.Path(raw_dir)
        for name in child_names:
            if (current / name).is_symlink():
                raise RuntimeError("unexpected symlink in campaign cgroup tree")
        if "cgroup.procs" not in file_names:
            raise RuntimeError("campaign cgroup descendant lacks cgroup.procs")
        raw = read_text(current / "cgroup.procs")
        processes = [line for line in raw.splitlines() if line.strip()]
        rel = str(current.relative_to(root))
        checked.append(rel or ".")
        if processes:
            populated.append({"path": rel or ".", "pids": processes})
    return checked, populated


def gpu_snapshot():
    identity = subprocess.run(
        ["nvidia-smi", "-i", GPU_UUID, "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if identity.returncode != 0 or identity.stdout.strip() != GPU_UUID:
        raise RuntimeError("selected GPU identity could not be verified")
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            GPU_UUID,
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError("selected GPU process inventory failed")
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError("unexpected selected GPU process row")
        pid = int(fields[0])
        proc_cgroup = pathlib.Path("/proc") / str(pid) / "cgroup"
        try:
            cgroup_text = proc_cgroup.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        campaign_owned = any(
            "wnn-campaign.slice" in pathlib.PurePosixPath(line.split(":", 2)[-1]).parts
            for line in cgroup_text.splitlines()
            if ":" in line
        )
        rows.append(
            {
                "pid": pid,
                "gpu_uuid": fields[1],
                "used_memory": fields[2],
                "cgroup": cgroup_text,
                "campaign_owned": campaign_owned,
            }
        )
    if any(row["campaign_owned"] for row in rows):
        raise RuntimeError("campaign-owned process already occupies the selected GPU")
    return {"identity_stdout": identity.stdout, "compute_processes": rows}


def observe_admission():
    if not B.is_dir() or B.resolve() != B:
        raise RuntimeError("physical B root is missing or resolves through another path")
    scratch = B / "research_tmp"
    if not scratch.is_dir() or scratch.resolve() != scratch:
        raise RuntimeError("physical B research_tmp is missing or resolves through another path")
    mount = subprocess.run(
        ["findmnt", "-n", "-o", "TARGET,SOURCE,FSTYPE", "-T", str(scratch)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if mount.returncode != 0:
        raise RuntimeError("could not identify the research_tmp filesystem")
    mount_fields = mount.stdout.strip().split()
    if len(mount_fields) < 3 or mount_fields[-1] in {"tmpfs", "ramfs", "devtmpfs"}:
        raise RuntimeError("research_tmp is not on an accepted persistent filesystem")
    disk_free = shutil.disk_usage(scratch).free
    if disk_free <= 2_000_000_000:
        raise RuntimeError("research_tmp has less than 2 GB free")

    meminfo = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        fields = line.replace(":", "").split()
        if len(fields) >= 2:
            meminfo[fields[0]] = int(fields[1]) * 1024
    host_available = meminfo.get("MemAvailable", 0)
    if host_available <= 5_000_000_000:
        raise RuntimeError("host available memory is at or below the 5 GB admission floor")

    parent = systemd_show(
        PARENT_UNIT,
        [
            "LoadState",
            "ActiveState",
            "SubState",
            "ControlGroup",
            "CPUQuotaPerSecUSec",
            "MemoryHigh",
            "MemoryMax",
            "MemorySwapMax",
            "MemoryCurrent",
        ],
    )
    p = parent["properties"]
    if (
        parent["returncode"] != 0
        or p.get("LoadState") != "loaded"
        or p.get("ActiveState") != "active"
    ):
        raise RuntimeError("persistent campaign parent slice is not active and loaded")
    control_group = p.get("ControlGroup", "")
    control_parts = pathlib.PurePosixPath(control_group).parts
    expected_user_manager = "user@{}.service".format(os.getuid())
    if (
        not control_group.startswith("/")
        or control_group == "/"
        or ".." in control_parts
        or control_parts[-1:] != (PARENT_UNIT,)
        or "user.slice" not in control_parts
        or expected_user_manager not in control_parts
    ):
        raise RuntimeError("parent slice has an invalid ControlGroup property")
    parent_cgroup = CGROOT / control_group.lstrip("/")
    if not parent_cgroup.is_dir() or not parent_cgroup.resolve().is_relative_to(CGROOT.resolve()):
        raise RuntimeError("parent cgroup path is missing or outside the unified cgroup root")
    checked_cgroups, populated_cgroups = walk_cgroup_procs(parent_cgroup)
    if populated_cgroups:
        raise RuntimeError("campaign parent or descendant cgroup is populated")
    cpu_max = read_text(parent_cgroup / "cpu.max").split()
    if len(cpu_max) != 2 or cpu_max[0] == "max" or int(cpu_max[0]) != 2 * int(cpu_max[1]):
        raise RuntimeError("campaign parent CPU quota is not exactly 2 cores")
    memory_high = int(read_text(parent_cgroup / "memory.high"))
    memory_max = int(read_text(parent_cgroup / "memory.max"))
    memory_swap_max = int(read_text(parent_cgroup / "memory.swap.max"))
    memory_current = int(read_text(parent_cgroup / "memory.current"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    expected_kernel_high = (PARENT_HIGH // page_size) * page_size
    expected_kernel_max = (PARENT_MAX // page_size) * page_size
    if memory_high != expected_kernel_high or memory_max != expected_kernel_max or memory_swap_max != 0:
        raise RuntimeError("campaign parent kernel memory caps differ from page-rounded 18/20 GB and zero swap")
    if int(p.get("MemoryHigh", "-1")) != PARENT_HIGH or int(p.get("MemoryMax", "-1")) != PARENT_MAX:
        raise RuntimeError("campaign parent systemd memory properties differ from 18/20 GB")
    if int(p.get("MemorySwapMax", "-1")) != 0:
        raise RuntimeError("campaign parent systemd swap property is not zero")
    if memory_current + NEW_MEMORY_RESERVATION > PARENT_RESERVATION_TARGET:
        raise RuntimeError("current parent memory plus the complete new reservation exceeds 17 GB")
    events = read_text(parent_cgroup / "memory.events")
    events_local = read_text(parent_cgroup / "memory.events.local")
    gpu = gpu_snapshot()
    return {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scratch_mount": mount.stdout.strip(),
        "scratch_free_bytes": disk_free,
        "host_mem_available_bytes": host_available,
        "parent_systemd": parent,
        "parent_cgroup": str(parent_cgroup),
        "parent_cpu_max": " ".join(cpu_max),
        "parent_memory": {
            "current_bytes": memory_current,
            "high_bytes": memory_high,
            "max_bytes": memory_max,
            "configured_high_bytes": PARENT_HIGH,
            "configured_max_bytes": PARENT_MAX,
            "kernel_page_size_bytes": page_size,
            "swap_max_bytes": memory_swap_max,
            "new_reservation_bytes": NEW_MEMORY_RESERVATION,
            "reservation_target_bytes": PARENT_RESERVATION_TARGET,
            "events": events,
            "events_local": events_local,
        },
        "cgroups_checked_empty": checked_cgroups,
        "populated_cgroups": populated_cgroups,
        "selected_gpu": gpu,
    }


def assert_no_unit_and_paths():
    unit = systemd_show(UNIT, ["LoadState", "InvocationID", "ActiveState", "SubState"])
    if unit["properties"].get("LoadState") != "not-found":
        raise RuntimeError("exact R10 support unit is not absent")
    if DEST.exists() or OUT.exists():
        raise RuntimeError("create-only R10 source or output path already exists")
    if not DEST.parent.is_dir() or not OUT.parent.is_dir():
        raise RuntimeError("create-only R10 parent directory is missing")
    if DEST.parent.resolve() != B / "support-source" or OUT.parent.resolve() != B / "research_tmp":
        raise RuntimeError("R10 source/output parent resolves outside the designated B paths")


def prepare_cfg():
    if (
        "CFG" not in globals()
        or not isinstance(CFG, dict)
        or not isinstance(CFG.get("files"), dict)
    ):
        raise RuntimeError("sealed CFG files payload is missing")
    if RUNTIME_PY != EXPECTED_RUNTIME_PY:
        raise RuntimeError(
            "assembled R10 runtime module map differs from the frozen four-module closure"
        )
    cfg_files = CFG["files"]
    manifest_entry = cfg_files.get(MANIFEST_REL)
    if not isinstance(manifest_entry, dict):
        raise RuntimeError("sealed CFG lacks PREPARATION_MANIFEST.json")
    manifest_payload = decode_entry(MANIFEST_REL, manifest_entry)
    if sha_bytes(manifest_payload) != MANIFEST_SHA256:
        raise RuntimeError("preparation manifest differs from the separately frozen SHA-256")
    manifest = json.loads(manifest_payload)
    if not isinstance(manifest, dict) or manifest.get("schema") != "nf19.r10_preparation.v1":
        raise RuntimeError("unexpected R10 preparation manifest schema")
    if manifest.get("runtime_py") != EXPECTED_RUNTIME_PY:
        raise RuntimeError("manifest runtime_py differs from the frozen import closure")
    file_pins = manifest.get("files")
    if (
        not isinstance(file_pins, dict)
        or file_pins.get("canonical:wnn/data.py") != CANONICAL_DATA_SHA256
    ):
        raise RuntimeError("manifest canonical semantic pin differs from the separately frozen pin")
    if any(not isinstance(name, str) for name in file_pins):
        raise RuntimeError("manifest file pin names must be strings")
    canonical_names = {name for name in file_pins if name.startswith("canonical:")}
    if canonical_names != {"canonical:wnn/data.py"}:
        raise RuntimeError("R10 may carry only the canonical wnn/data.py semantic pin")
    noncanonical = {name for name in file_pins if not name.startswith("canonical:")}
    if any(
        not name.startswith("idea_screens_r10/")
        or ".." in pathlib.PurePosixPath(name).parts
        or pathlib.PurePosixPath(name).is_absolute()
        or pathlib.PurePosixPath(name).as_posix() != name
        or not isinstance(file_pins[name], str)
        or len(file_pins[name]) != 64
        or any(char not in "0123456789abcdef" for char in file_pins[name])
        for name in noncanonical
    ):
        raise RuntimeError("manifest noncanonical source path escapes the R10 allowlist")
    expected_names = noncanonical | EXTRA_REL
    if set(cfg_files) != expected_names:
        raise RuntimeError(
            "CFG paths do not equal manifest noncanonical pins plus the four frozen extras"
        )
    total = 0
    decoded = {}
    for name in sorted(cfg_files):
        rel = pathlib.PurePosixPath(name)
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or not rel.parts
            or rel.parts[0] != "idea_screens_r10"
        ):
            raise RuntimeError("sealed CFG path escapes idea_screens_r10: " + name)
        entry = cfg_files[name]
        if not isinstance(entry, dict):
            raise RuntimeError("sealed CFG file entry is not an object: " + name)
        payload = decode_entry(name, entry)
        total += len(payload)
        if total > MAX_BUNDLE_BYTES:
            raise RuntimeError("sealed R10 source bundle exceeds the fixed 64 MB staging bound")
        decoded[name] = payload
        if name in file_pins and sha_bytes(payload) != file_pins[name]:
            raise RuntimeError("sealed CFG bytes differ from manifest pin: " + name)
    if sha_bytes(decoded[MANIFEST_REL]) != MANIFEST_SHA256:
        raise RuntimeError("decoded manifest bytes changed during CFG validation")
    canonical_path = B / "source-a266b036/src/wnn/data.py"
    if not canonical_path.is_file() or sha_file(canonical_path) != CANONICAL_DATA_SHA256:
        raise RuntimeError("canonical data.py semantic source pin differs")
    return manifest, decoded, file_pins


def decode_entry(name, entry):
    encoded = entry.get("base64")
    expected_bytes = entry.get("bytes")
    expected_sha256 = entry.get("sha256")
    if (
        not isinstance(encoded, str)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
    ):
        raise RuntimeError("sealed CFG entry lacks bounded base64/byte metadata: " + name)
    if (
        expected_bytes < 0
        or expected_bytes > MAX_FILE_BYTES
        or len(encoded) > (MAX_FILE_BYTES * 4 // 3 + 8)
    ):
        raise RuntimeError("sealed CFG entry exceeds the fixed per-file bound: " + name)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("sealed CFG entry lacks SHA-256: " + name)
    payload = base64.b64decode(encoded, validate=True)
    if len(payload) != expected_bytes or sha_bytes(payload) != expected_sha256:
        raise RuntimeError(
            "sealed CFG transport bytes fail their exact length/SHA binding: " + name
        )
    return payload


def main():
    if os.getuid() != 1061 or B.stat().st_uid != 1061:
        raise RuntimeError("R10 must run as the B-owned UID1061 user")
    assert_no_unit_and_paths()
    if (
        not SAMPLE.is_dir()
        or SAMPLE.is_symlink()
        or SAMPLE.resolve().parent != B / "research_tmp"
    ):
        raise RuntimeError("accepted R3 sample directory is missing or resolves outside research_tmp")
    initial = observe_admission()
    sample_bytes, sample = read_small_json(SAMPLE / "SAMPLE.json", SAMPLE_SHA256)
    membership_bytes, membership = read_small_json(SAMPLE / "MEMBERSHIP.json", MEMBERSHIP_SHA256)
    selection = sample.get("selection")
    if not isinstance(selection, dict) or selection.get("selected_component_counts") != {
        "fit_full64": 64,
        "fit_nested_low32": 32,
        "eval32": 32,
    }:
        raise RuntimeError("accepted R3 sample metadata does not declare the fixed 64/32/32 panel")
    if len(sample.get("records", [])) != 96 or len(membership.get("records", [])) != 96:
        raise RuntimeError("accepted R3 metadata does not contain exactly 96 panel records")
    if membership.get("selection") != selection:
        raise RuntimeError("accepted R3 sample and membership selection metadata disagree")
    manifest, decoded, file_pins = prepare_cfg()

    DEST.mkdir(mode=0o755, parents=False, exist_ok=False)
    for name, payload in sorted(decoded.items()):
        target = DEST / pathlib.PurePosixPath(name)
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    staged = {}
    for name, payload in sorted(decoded.items()):
        target = DEST / pathlib.PurePosixPath(name)
        if not target.is_file() or target.is_symlink() or target.stat().st_size != len(payload):
            raise RuntimeError("staged R10 file has unexpected type or length: " + name)
        observed_sha = sha_file(target)
        if observed_sha != sha_bytes(payload):
            raise RuntimeError("staged R10 file differs from its sealed transport bytes: " + name)
        staged[name] = {"bytes": len(payload), "sha256": observed_sha}
    actual_names = {path.relative_to(DEST).as_posix() for path in DEST.rglob("*") if path.is_file()}
    if actual_names != set(decoded):
        raise RuntimeError("staged R10 source inventory differs from the exact CFG allowlist")
    if any(path.is_symlink() for path in DEST.rglob("*")):
        raise RuntimeError("staged R10 source contains an unexpected symlink")

    assert_no_unit_before_second_gate()
    prelaunch = observe_admission()
    if (
        prelaunch["parent_memory"]["events"] != initial["parent_memory"]["events"]
        or prelaunch["parent_memory"]["events_local"] != initial["parent_memory"]["events_local"]
    ):
        raise RuntimeError("campaign parent memory event counters changed during R10 staging")
    if not OUT.parent.is_dir() or OUT.exists():
        raise RuntimeError("R10 output appeared or physical research_tmp disappeared before launch")
    for name, expected in staged.items():
        target = DEST / pathlib.PurePosixPath(name)
        if target.stat().st_size != expected["bytes"] or sha_file(target) != expected["sha256"]:
            raise RuntimeError("staged R10 bytes changed before launch: " + name)
    if sha_file(B / "source-a266b036/src/wnn/data.py") != CANONICAL_DATA_SHA256:
        raise RuntimeError("canonical data.py semantic pin changed before launch")

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
        "PYTHONPATH=" + str(DEST),
        "TMPDIR=" + str(B / "research_tmp"),
        PY,
        "-B",
        "-c",
        inline,
    ]
    admission = {
        "schema": "nf19.r10_support_launch_admission.v1",
        "uid": os.getuid(),
        "base_owner_uid": B.stat().st_uid,
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "unit": UNIT,
        "sample": {
            "path": str(SAMPLE),
            "sample_sha256": sha_bytes(sample_bytes),
            "membership_sha256": sha_bytes(membership_bytes),
            "split_sha256": SPLIT_SHA256,
            "fit_component_count": 64,
            "nested_fit_component_count": 32,
            "eval_metadata_component_count": 32,
            "array_payloads_read": 0,
        },
        "preparation_manifest_sha256": MANIFEST_SHA256,
        "canonical_data_py_sha256": CANONICAL_DATA_SHA256,
        "runtime_py": RUNTIME_PY,
        "staged_files": staged,
        "manifest_noncanonical_pins": {
            name: digest for name, digest in file_pins.items() if not name.startswith("canonical:")
        },
        "resource_admission_initial": initial,
        "resource_admission_prelaunch": prelaunch,
        "command": command,
        "output": str(OUT),
        "caps": {
            "cpu_cores": 0.5,
            "memory_high_bytes": 1_400_000_000,
            "memory_max_bytes": 1_500_000_000,
            "memory_swap_max_bytes": 0,
            "runtime_max_s": 600,
            "cuda_visible_devices": "",
            "blas_threads": 1,
            "tmpdir": str(B / "research_tmp"),
        },
    }
    write_create_only(DEST / "ROOT_ADMISSION.json", admission)
    try:
        launched = subprocess.run(command, capture_output=True, timeout=30)
        launch_returncode = launched.returncode
        launch_stdout = launched.stdout
        launch_stderr = launched.stderr
        launch_timeout = None
    except subprocess.TimeoutExpired as exc:
        launch_returncode = None
        launch_stdout = exc.stdout or b""
        launch_stderr = exc.stderr or b""
        launch_timeout = 30
    try:
        launch_props = systemd_show(
            UNIT,
            [
                "LoadState",
                "ActiveState",
                "SubState",
                "Result",
                "InvocationID",
                "ControlGroup",
                "ExecStart",
                "ExecMainPID",
                "ExecMainCode",
                "ExecMainStatus",
                "CPUQuotaPerSecUSec",
                "CPUUsageNSec",
                "MemoryHigh",
                "MemoryMax",
                "MemorySwapMax",
                "MemoryCurrent",
                "MemoryPeak",
                "RuntimeMaxUSec",
                "RemainAfterExit",
            ],
        )
    except Exception as exc:
        launch_props = {
            "returncode": None,
            "stdout": "",
            "stdout_base64": "",
            "stdout_bytes": 0,
            "stdout_sha256": sha_bytes(b""),
            "stderr": repr(exc),
            "stderr_base64": "",
            "stderr_bytes": 0,
            "stderr_sha256": sha_bytes(b""),
            "properties": {},
            "capture_error_type": type(exc).__name__,
            "capture_error": str(exc),
        }
    launch_record = {
        "schema": "nf19.r10_support_root_launch.v1",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "unit": UNIT,
        "command": command,
        "systemd_run_returncode": launch_returncode,
        "systemd_run_timeout_s": launch_timeout,
        "systemd_run_stdout_base64": base64.b64encode(launch_stdout).decode("ascii"),
        "systemd_run_stdout_bytes": len(launch_stdout),
        "systemd_run_stdout_sha256": sha_bytes(launch_stdout),
        "systemd_run_stderr_base64": base64.b64encode(launch_stderr).decode("ascii"),
        "systemd_run_stderr_bytes": len(launch_stderr),
        "systemd_run_stderr_sha256": sha_bytes(launch_stderr),
        "systemd_properties": launch_props,
        "invocation_id": launch_props["properties"].get("InvocationID"),
    }
    write_create_only(DEST / "ROOT_LAUNCH.json", launch_record)
    print(json.dumps(launch_record, indent=2, sort_keys=True))
    if launch_returncode != 0 or not launch_record["invocation_id"]:
        raise RuntimeError(
            "systemd-run did not produce a bound successful R10 invocation; do not retry"
        )


def assert_no_unit_before_second_gate():
    unit = systemd_show(UNIT, ["LoadState", "InvocationID", "ActiveState", "SubState"])
    if unit["properties"].get("LoadState") != "not-found":
        raise RuntimeError("exact R10 support unit appeared before the single launch")
    if OUT.exists():
        raise RuntimeError("R10 output path appeared before launch")


main()
