"""Admit one bounded native teacher profile after the verified F0 propensity fit."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"
PAYLOAD = json.loads(base64.b64decode(PAYLOAD_B64, validate=True))
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA = B / "data"
GPU = "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c"
PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
SOURCE = Path(PAYLOAD["source_root"])
PACKET = B / "research/full_gru_tweaks_r1/teacher_profile_v2"
OUTPUT = B / "research_tmp/fg19_teacher_profile_v2"
RUN_NAME = "fg19_t3_h384x2_profile_s3_v2"
RUN_DIR = DATA / "artifacts/runs" / RUN_NAME
UNIT = "computer-vision-homework-fg19-teacher-profile-v2.service"
MIN_GPU_FREE = 2_500_000_000 + 1_000_000_000
FIT_MEMORY = 8_000_000_000


def _run(args: list[str], timeout: int = 20) -> str:
    """Run a read-only preflight command and fail on nonzero status."""
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args!r}: {result.returncode}: {result.stderr}")
    return result.stdout


def _sha(path: Path) -> str:
    """Hash a bounded metadata/model file without loading its contents as tensors."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    """Create a durable JSON receipt without replacing prior evidence."""
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _unit(unit: str, fields: list[str]) -> dict[str, str]:
    """Read one live unit without falling back to a host or user slice."""
    args = ["systemctl", "--user", "show", "--all", unit]
    args.extend(part for field in fields for part in ("-p", field))
    return dict(line.split("=", 1) for line in _run(args).splitlines())


def _check_terminal_propensity() -> dict[str, object]:
    """Bind durable verification and locally captured terminal/stop evidence."""
    expected = PAYLOAD["propensity"]
    verification_path = B / "research_tmp/fg19_propensity_verify_v2/VERIFICATION.json"
    raw = verification_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected[
        "verification_sha256"
    ] or raw != base64.b64decode(expected["verification_capture"]["base64"], validate=True):
        raise RuntimeError("verified propensity receipt hash changed")
    doc = json.loads(raw)
    model_sha = expected["model_sha256"]
    if (
        doc.get("schema") != "fg19.propensity-terminal-verification.v1"
        or doc.get("status") != "VERIFIED"
        or doc.get("native_unit")
        != {"exit_status": 0, "invocation_id": expected["fit_invocation_id"]}
        or doc.get("split", {}).get("sha256") != PAYLOAD["split_sha256"]
        or doc.get("run", {}).get("name") != expected["run_name"]
        or doc.get("run", {}).get("fit_status") != "completed"
        or doc.get("run", {}).get("selected_epoch") != 3
        or doc.get("training_state", {}).get("next_epoch") != 4
        or doc.get("progress", {}).get("aggregate", {}).get("optimizer_completed_updates") != 3520
        or doc.get("progress", {}).get("aggregate", {}).get("nonfinite_updates") != 0
        or doc.get("model", {}).get("epoch") != 3
        or doc.get("model", {}).get("hidden") != 64
        or doc.get("model", {}).get("n_in") != 211
        or doc.get("files", {}).get("model.pt", {}).get("sha256") != model_sha
        or doc.get("files", {}).get("model_ep3.pt", {}).get("sha256") != model_sha
    ):
        raise RuntimeError("propensity terminal verification differs from the bound fit")
    model = DATA / "artifacts/runs" / expected["run_name"] / "model.pt"
    if model.is_symlink() or not model.is_file() or model.stat().st_size != 174545:
        raise RuntimeError("verified propensity checkpoint is missing or has the wrong type/size")
    if _sha(model) != model_sha:
        raise RuntimeError("verified propensity model bytes changed")
    evidence: dict[str, object] = {}
    for name in (
        "acceptance",
        "native_launch",
        "verifier_admission",
        "verifier_launch",
        "stop",
    ):
        item = expected[name]
        evidence_raw = base64.b64decode(item["base64"], validate=True)
        if hashlib.sha256(evidence_raw).hexdigest() != item["sha256"]:
            raise RuntimeError(f"local propensity {name} bytes do not match the reviewed binding")
        evidence[name] = json.loads(evidence_raw)
    acceptance = evidence["acceptance"]
    native_launch = evidence["native_launch"]
    verifier_admission = evidence["verifier_admission"]
    verifier_launch = evidence["verifier_launch"]
    stop = evidence["stop"]
    if (
        acceptance.get("schema") != "fg19.propensity-terminal-root-acceptance.v1"
        or acceptance.get("status") != "ACCEPTED"
        or acceptance.get("verification_sha256") != expected["verification_sha256"]
        or acceptance.get("model", {}).get("sha256") != model_sha
        or acceptance.get("model", {}).get("bytes") != model.stat().st_size
        or acceptance.get("counts", {}).get("successful_updates") != 3520
        or acceptance.get("counts", {}).get("nonfinite_updates") != 0
    ):
        raise RuntimeError("root propensity terminal acceptance does not bind the verified model")
    native_submission = native_launch.get("submission", {})
    native_stdout = native_launch.get("systemctl_after_submission", {}).get("stdout", "")
    verifier_post = verifier_launch.get("post", {})
    verifier_command = verifier_launch.get("command", [])
    if (
        native_launch.get("unit") != "computer-vision-homework-fg19-propensity-f0-e4-v3.service"
        or native_submission.get("returncode") != 0
        or native_submission.get("outcome") != "submitted"
        or expected["fit_invocation_id"] not in native_stdout
        or "scripts/train_propensity.py" not in " ".join(native_launch.get("command", []))
        or verifier_launch.get("returncode") != 0
        or expected["verify_invocation_id"] not in verifier_launch.get("stderr", "")
        or verifier_post.get("InvocationID") != expected["verify_invocation_id"]
        or verifier_post.get("ExecMainStatus") != "0"
        or verifier_admission.get("native", {}).get("InvocationID") != expected["fit_invocation_id"]
        or verifier_admission.get("native", {}).get("ExecMainStatus") != "0"
        or verifier_admission.get("populated") != {}
        or "--output" not in verifier_command
        or not any(
            verifier_command[index + 1].endswith("/fg19_propensity_verify_v2/VERIFICATION.json")
            for index, value in enumerate(verifier_command[:-1])
            if value == "--output"
        )
    ):
        raise RuntimeError(
            "captured propensity launch records do not bind the verified invocations"
        )
    fit_unit = acceptance.get("native_unit", {})
    verify_unit = acceptance.get("verifier_unit", {})
    for observed, invocation in (
        (fit_unit, expected["fit_invocation_id"]),
        (verify_unit, expected["verify_invocation_id"]),
    ):
        if (
            observed.get("InvocationID") != invocation
            or observed.get("ActiveState") != "active"
            or observed.get("SubState") != "exited"
            or observed.get("ExecMainCode") != "1"
            or observed.get("ExecMainStatus") != "0"
        ):
            raise RuntimeError(
                "root terminal acceptance lacks the exact prior zero-exit invocation"
            )
    before = stop.get("before", {})
    after = stop.get("after", {})
    stop_results = stop.get("stop", {})
    unit_ids = {
        "computer-vision-homework-fg19-propensity-f0-e4-v3.service": expected["fit_invocation_id"],
        "computer-vision-homework-fg19-propensity-verify-v2.service": expected[
            "verify_invocation_id"
        ],
    }
    for unit, invocation in unit_ids.items():
        if (
            before.get(unit, {}).get("InvocationID") != invocation
            or before.get(unit, {}).get("ExecMainStatus") != "0"
            or stop_results.get(unit, {}).get("returncode") != 0
            or after.get(unit, {}).get("LoadState") != "not-found"
            or after.get(unit, {}).get("ActiveState") != "inactive"
            or after.get(unit, {}).get("SubState") != "dead"
            or after.get(unit, {}).get("InvocationID") != ""
        ):
            raise RuntimeError("exact propensity stop capture does not show stopped prior units")
    if stop.get("populated") != {}:
        raise RuntimeError(
            "propensity terminal stop capture did not leave the campaign cgroup empty"
        )
    return {
        "verification_sha256": expected["verification_sha256"],
        "model_sha256": model_sha,
        "model_bytes": model.stat().st_size,
        "acceptance_sha256": expected["acceptance"]["sha256"],
        "native_launch_sha256": expected["native_launch"]["sha256"],
        "verifier_launch_sha256": expected["verifier_launch"]["sha256"],
        "stop_capture_sha256": expected["stop"]["sha256"],
        "fit_invocation_id": expected["fit_invocation_id"],
        "verify_invocation_id": expected["verify_invocation_id"],
    }


def _validate_profile_arguments() -> list[str]:
    """Accept only the reviewed fold-0, fit-only six-batch profile command."""
    args = PAYLOAD["profile_argv"]
    if len(args) < 3 or args[:2] != ["python", "scripts/train_rnn.py"]:
        raise RuntimeError("profile template is not the reviewed canonical training argv")
    pairs: dict[str, str] = {}
    index = 2
    while index < len(args):
        key = args[index]
        if not key.startswith("--"):
            raise RuntimeError("profile argv has an unexpected positional argument")
        if key in pairs:
            raise RuntimeError(f"profile argv repeats an option: {key}")
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            pairs[key] = args[index + 1]
            index += 2
        else:
            pairs[key] = ""
            index += 1
    required = {
        "--name": RUN_NAME,
        "--seed": "3",
        "--device": "cuda",
        "--arch": "gru",
        "--cell": "gru",
        "--enc": "linear",
        "--hidden": "384",
        "--layers": "2",
        "--head-layers": "1",
        "--epochs": "3",
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": "2",
        "--batch": "32",
        "--chunk": "1000",
        "--optim": "muon",
        "--muon-scope": "all",
        "--muon-ortho-dtype": "fp32",
        "--lr": "0.001",
        "--wd": "0.01",
        "--beta2": "0.999",
        "--sched": "onecycle",
        "--warmup-frac": "0.05",
        "--clip": "1",
        "--loss": "wmse",
        "--loss-norm": "chunk",
        "--precision": "fp32",
        "--aux-struct": "0.3",
        "--aux-next": "0.3",
        "--since-weight": "32",
        "--since-floor": "0.25",
        "--propensity": "fg19_propensity_clean_f0_seed0_e4",
        "--prop-power": "2",
        "--prop-eps": "0.03",
        "--valid-true-mask": "0.03",
        "--max-batches": "6",
        "--profile-warmup": "2",
        "--profile-steps": "4",
        "--split-manifest": str(B / "splits/fold0.json"),
        "--valid-fold": "0",
        "--train-source": "train+valid",
    }
    if any(pairs.get(key) != value for key, value in required.items()):
        raise RuntimeError("profile template differs from the reviewed seed-3 configuration")
    required_flags = {
        "--strict-muon-fp32",
        "--sync-free",
        "--pin-memory",
        "--grad-stats",
        "--no-eval-during-fit",
    }
    if not required_flags <= pairs.keys():
        raise RuntimeError("profile template is missing a required finite/fit-only flag")
    if set(pairs) != set(required) | required_flags:
        raise RuntimeError("profile template contains an unreviewed option")
    if "--valid-fold" not in pairs:
        args = [*args, "--valid-fold", "0"]
    return args


def _data_identity() -> dict[str, dict[str, object]]:
    """Require exact registered dataset aliases and stable metadata identities."""
    registry = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
    if registry.is_symlink() or _sha(registry) != PAYLOAD["registry_sha256"]:
        raise RuntimeError("portable data registry changed")
    sources = json.loads(registry.read_bytes()).get("sources", {})
    expected = PAYLOAD["data_resolution_map"]
    if set(sources) != set(expected) or len(expected) != 6:
        raise RuntimeError("registered F0 source membership differs from the bound six paths")
    b_root, data_root = B.resolve(strict=True), DATA.resolve(strict=True)
    if not data_root.is_relative_to(b_root):
        raise RuntimeError("data root resolves outside the campaign volume")
    result: dict[str, dict[str, object]] = {}
    for relative, bound in expected.items():
        path = DATA / relative
        entry = sources[relative]
        resolved = path.resolve(strict=True)
        current = resolved.stat()
        observed = {
            "path": str(path),
            "resolved": str(resolved),
            "islink": path.is_symlink(),
            "device": current.st_dev,
            "bytes": current.st_size,
            "inode": current.st_ino,
            "mtime_ns": current.st_mtime_ns,
            "registered_sha256": entry.get("sha256"),
        }
        wanted = {
            key: bound[key]
            for key in (
                "path",
                "resolved",
                "islink",
                "device",
                "bytes",
                "inode",
                "mtime_ns",
                "registered_sha256",
            )
        }
        if (
            not stat.S_ISREG(current.st_mode)
            or observed != wanted
            or observed["device"] != b_root.stat().st_dev
            or observed["bytes"] != entry.get("original_identity", {}).get("bytes")
        ):
            raise RuntimeError(f"registered dataset alias or stat identity changed: {relative}")
        result[relative] = observed
    return result


def _parent() -> tuple[dict[str, str], dict[str, str], Path]:
    """Require current campaign quotas and an empty, explicitly scoped cgroup tree."""
    fields = [
        "ControlGroup",
        "MemoryCurrent",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
        "CPUQuotaPerSecUSec",
    ]
    parent = _unit("wnn-campaign.slice", fields)
    if (
        not parent.get("ControlGroup")
        or parent["ControlGroup"] == "/"
        or (
            parent["CPUQuotaPerSecUSec"],
            parent["MemoryHigh"],
            parent["MemoryMax"],
            parent["MemorySwapMax"],
        )
        != ("2s", "18000000000", "20000000000", "0")
        or int(parent["MemoryCurrent"]) + FIT_MEMORY >= 17_000_000_000
    ):
        raise RuntimeError(f"campaign parent lacks required headroom/limits: {parent}")
    group = Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
    paths = [group / "cgroup.procs", *group.rglob("cgroup.procs")]
    if len(paths) > 128:
        raise RuntimeError("campaign cgroup descendant scan exceeded its fixed bound")
    populated = {str(path): value for path in paths if (value := path.read_text().strip())}
    if populated:
        raise RuntimeError(f"campaign already has populated descendants: {populated}")
    return parent, populated, group


def _gpu() -> tuple[str, list[dict[str, object]]]:
    """Reserve only the selected GPU headroom while preserving foreign processes."""
    prefix = ["nvidia-smi", "-i", GPU]
    status = _run(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    )
    fields = [item.strip() for item in status.strip().split(",")]
    if (
        len(fields) != 4
        or fields[0] != GPU
        or int(fields[2].split()[0]) * 1024 * 1024 < MIN_GPU_FREE
    ):
        raise RuntimeError(f"selected GPU identity/free reservation failed: {status.strip()}")
    output = _run(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"]
    )
    processes: list[dict[str, object]] = []
    for row in output.splitlines():
        cols = [value.strip() for value in row.split(",")]
        if row.strip().startswith("No running processes found"):
            continue
        if len(cols) != 3 or cols[1] != GPU:
            raise RuntimeError("selected GPU process query returned an unexpected row")
        pid = Path("/proc") / cols[0]
        try:
            owner = (pid / "cgroup").read_text().strip()
            uid = pid.stat().st_uid
            command = (pid / "cmdline").read_bytes().decode(errors="replace")
        except FileNotFoundError:
            processes.append({"pid": cols[0], "exited_during_observation": True})
            continue
        if "wnn-campaign.slice" in owner or (uid == os.getuid() and str(B) in command):
            raise RuntimeError(
                f"selected GPU is occupied by another owned campaign process: {cols[0]}"
            )
        processes.append(
            {"pid": cols[0], "uid": uid, "cgroup": owner, "used_memory": cols[2], "preserved": True}
        )
    return status.strip(), processes


def _validate_files() -> dict[str, str]:
    """Verify staged source/packet bytes and require every destination absent."""
    files = PAYLOAD["files"]
    expected_source = {str(SOURCE / item["path"]) for item in PAYLOAD["source_manifest"]["files"]}
    expected_packet = {str(PACKET / name) for name in PAYLOAD["packet_names"]}
    if set(files) != expected_source | expected_packet:
        raise RuntimeError(
            "staged file set does not match the sealed source and packet inventories"
        )
    observed: dict[str, str] = {}
    total = 0
    for filename, item in files.items():
        path = Path(filename)
        if not path.is_relative_to(SOURCE) and not path.is_relative_to(PACKET):
            raise ValueError(f"file escaped the two declared roots: {filename}")
        if ".." in path.parts or os.path.lexists(path):
            raise FileExistsError(f"staging destination exists or is unsafe: {path}")
        for ancestor in path.parents:
            if ancestor == B:
                break
            if os.path.lexists(ancestor) and ancestor.is_symlink():
                raise RuntimeError(f"staging parent is a symlink: {ancestor}")
        raw = base64.b64decode(item["base64"], validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"local file payload hash mismatch: {filename}")
        if len(raw) > 8_000_000:
            raise RuntimeError(
                f"single source file exceeds the bounded code-file limit: {filename}"
            )
        total += len(raw)
        observed[filename] = digest
    if total > 30_000_000:
        raise RuntimeError("teacher source/packet payload exceeds the bounded staging budget")
    return observed


def main() -> None:
    """Run all admission gates before any remote write, then launch one profile unit."""
    profile_args = _validate_profile_arguments()
    if (
        PAYLOAD["split_sha256"]
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
    ):
        raise RuntimeError("profile is not bound to the current canonical fold0 split")
    if any(os.path.lexists(path) for path in (SOURCE, PACKET, OUTPUT, RUN_DIR)):
        raise FileExistsError("teacher source/packet/profile output/run directory already exists")
    if (B / "runtime/STOP_ADMISSION.json").exists():
        raise RuntimeError("campaign admission is stopped")
    if (
        _run(["systemctl", "--user", "show", "--all", UNIT, "-p", "LoadState", "--value"]).strip()
        != "not-found"
    ):
        raise RuntimeError("exact teacher profile unit already exists")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise FileNotFoundError("pinned project interpreter is unavailable")
    if B.is_symlink() or not B.is_dir() or DATA.resolve(strict=True) != B / "data":
        raise RuntimeError("campaign/data root is not the expected physical B tree")
    source_pins = _validate_files()
    propensity = _check_terminal_propensity()
    dataset = _data_identity()
    parent, populated, parent_cgroup = _parent()
    gpu, gpu_processes = _gpu()
    scratch = _run(
        ["findmnt", "-T", str(B / "research_tmp"), "-n", "-o", "FSTYPE,SOURCE,TARGET"]
    ).strip()
    if (
        scratch.split()[0] in {"tmpfs", "ramfs"}
        or os.statvfs(B / "research_tmp").f_bavail * os.statvfs(B / "research_tmp").f_frsize
        < 20_000_000_000
    ):
        raise RuntimeError("physical scratch lacks the required 20 GB free")
    if any(os.path.lexists(path) for path in (SOURCE, PACKET, OUTPUT, RUN_DIR)):
        raise FileExistsError("a profile destination appeared during admission")

    for filename, item in PAYLOAD["files"].items():
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(item["base64"], validate=True)
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if _sha(path) != item["sha256"]:
            raise RuntimeError(f"staged file hash changed after write: {path}")
    OUTPUT.mkdir(mode=0o700)
    admission = {
        "schema": "fg19_teacher_profile_admission.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "unit": UNIT,
        "source_commit": PAYLOAD["source_manifest"]["source_commit"],
        "source_files": source_pins,
        "source_manifest_sha256": PAYLOAD["source_manifest_sha256"],
        "binding_sha256": PAYLOAD["binding_sha256"],
        "cold_import_sha256": PAYLOAD["cold_import_sha256"],
        "review_sha256": PAYLOAD["review_sha256"],
        "propensity": propensity,
        "split_sha256": PAYLOAD["split_sha256"],
        "data_identity_stats_only": dataset,
        "parent": parent,
        "populated_campaign_descendants": populated,
        "parent_cgroup_events": (parent_cgroup / "memory.events").read_text(),
        "parent_cgroup_events_local": (parent_cgroup / "memory.events.local").read_text(),
        "gpu": gpu,
        "gpu_processes_preserved": gpu_processes,
        "scratch_filesystem": scratch,
        "profile_scope": "120 optimizer attempts (2 warmup + 4 measured batches), train+valid only, no heldout validation, checkpoint save, or quality score",
        "worker_reservation": {
            "cpu_quota": "100%",
            "memory_high": 7_500_000_000,
            "memory_max": FIT_MEMORY,
            "swap_max": 0,
            "runtime_max_seconds": 1200,
        },
    }
    _write(OUTPUT / "ADMISSION.json", admission)

    command = [str(PYTHON), "-u", str(SOURCE / "scripts/profile_teacher.py"), *profile_args[2:]]
    env = {
        "PYTHONPATH": str(SOURCE / "src") + ":" + str(SOURCE / "scripts"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "WNN_PROJECT_ROOT": str(DATA),
        "WNN_PORTABLE_DATA_MANIFEST": str(B / "support-source/terminal-v1/PORTABLE_SOURCES.json"),
        "WNN_GPU_MEMORY_BYTES": "2500000000",
        "WNN_LOADER_WORKERS": "0",
        "WNN_LOADER_RAM_GB": "1",
        "WNN_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": GPU,
        "NVIDIA_TF32_OVERRIDE": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TMPDIR": str(B / "research_tmp"),
        "WNN_TMP_ROOT": str(B / "research_tmp"),
    }
    systemd = [
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
    systemd.extend("--setenv=" + key + "=" + value for key, value in env.items())
    systemd.extend(command)
    try:
        result = subprocess.run(systemd, capture_output=True, text=True, timeout=30)
        submission = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "outcome": "submitted" if result.returncode == 0 else "unknown_or_failed",
        }
    except subprocess.TimeoutExpired as error:
        submission = {
            "returncode": None,
            "stdout": error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout,
            "stderr": error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr,
            "outcome": "unknown",
        }
    props = ["InvocationID", "ExecStart", "ActiveState", "SubState", "MainPID", "ControlGroup"]
    show = subprocess.run(
        ["systemctl", "--user", "show", "--all", UNIT, *sum((["-p", key] for key in props), [])],
        capture_output=True,
        text=True,
        timeout=10,
    )
    launch = {
        "schema": "fg19_teacher_profile_launch.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "unit": UNIT,
        "command": systemd,
        "submission": submission,
        "systemctl_after_submission": {
            "returncode": show.returncode,
            "stdout": show.stdout,
            "stderr": show.stderr,
        },
        "unknown_outcome_policy": "never retry; inspect this exact unit and durable profile output",
    }
    _write(OUTPUT / "LAUNCH.json", launch)
    print(
        json.dumps(
            {
                name: base64.b64encode((OUTPUT / name).read_bytes()).decode("ascii")
                for name in ("ADMISSION.json", "LAUNCH.json")
            }
        )
    )


if __name__ == "__main__":
    main()
