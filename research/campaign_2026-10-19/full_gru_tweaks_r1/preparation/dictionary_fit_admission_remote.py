"""Admit the frozen CPU dictionary fit against current B data and resource state."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"
P = json.loads(base64.b64decode(PAYLOAD_B64, validate=True))
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA = B / "data"
SOURCE = B / "source-fg19-gru-51a9b168"
SPLIT = B / "splits/fold0.json"
REGISTRY = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
RUN = DATA / "artifacts/research/fg19_dictionary_f0_v1"
CACHE, RECEIPT = RUN / "input_representation.pt", RUN / "RECEIPT.json"
LOGS = B / "research_tmp/fg19_dictionary_f0_v1"
UNIT = "computer-vision-homework-fg19-dictionary-f0-v1.service"
FIT_UNIT = "computer-vision-homework-fg19-propensity-f0-e4-v3.service"
PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
COMMIT = "51a9b168584e86ca10ebdc0532465d6158ab19d3"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
GPU = "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c"
FIT_MEMORY, DICTIONARY_MEMORY = 8_000_000_000, 1_500_000_000


def _run(args: list[str], timeout: int = 20) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args!r}: {result.returncode}: {result.stderr}")
    return result.stdout


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: str | bytes | None) -> str | None:
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _write(path: Path, value: dict[str, object]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _data() -> dict[str, dict[str, object]]:
    registry = json.loads(REGISTRY.read_bytes()).get("sources", {})
    expected, resolutions = P["profile_data_stats"], P["profile_data_resolved_paths"]
    if set(registry) != set(expected) or set(expected) != set(resolutions):
        raise ValueError("portable registry membership differs from the profiled six-file map")
    b_root, data_root = B.resolve(strict=True), DATA.resolve(strict=True)
    if data_root.stat().st_dev != b_root.stat().st_dev:
        raise RuntimeError("data and B are not on the profiled physical device")
    result: dict[str, dict[str, object]] = {}
    for relative, prior in expected.items():
        path = DATA / relative
        bound, registered = resolutions[relative], registry[relative]
        resolved = path.resolve(strict=True)
        if str(path) != bound["path"] or str(resolved) != bound["resolved"]:
            raise RuntimeError(f"registered data resolution changed: {relative}")
        if path.is_symlink() != bound["islink"]:
            raise RuntimeError(f"registered data link type changed: {relative}")
        info = resolved.stat()
        identity = {
            "device": info.st_dev,
            "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns,
            "bytes": info.st_size,
        }
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_dev != P["profile_data_device"]
            or any(identity[key] != prior[key] for key in identity if key != "device")
            or any(identity[key] != bound[key] for key in identity)
            or registered.get("sha256") != prior["registered_sha256"]
        ):
            raise RuntimeError(f"registered data identity changed: {relative}")
        if resolved.stat() != info or path.resolve(strict=True) != resolved:
            raise RuntimeError(f"registered data changed while statting: {relative}")
        result[relative] = {"path": str(path), "resolved": str(resolved), **identity}
    return result


def _service(name: str, keys: list[str]) -> dict[str, str]:
    args = ["systemctl", "--user", "show", "--all", name]
    args.extend(part for key in keys for part in ("-p", key))
    return dict(line.split("=", 1) for line in _run(args).splitlines())


def _resources() -> dict[str, object]:
    parent = _service(
        "wnn-campaign.slice",
        [
            "ControlGroup",
            "CPUQuotaPerSecUSec",
            "MemoryCurrent",
            "MemoryHigh",
            "MemoryMax",
            "MemorySwapMax",
        ],
    )
    if (
        parent.get("CPUQuotaPerSecUSec") != "2s"
        or parent.get("MemoryHigh") != "18000000000"
        or parent.get("MemoryMax") != "20000000000"
        or parent.get("MemorySwapMax") != "0"
    ):
        raise RuntimeError(f"campaign parent resource limits differ from contract: {parent}")
    if not parent.get("ControlGroup"):
        raise RuntimeError("campaign slice has no cgroup path")
    fit = _service(
        FIT_UNIT,
        [
            "LoadState",
            "ActiveState",
            "SubState",
            "InvocationID",
            "ControlGroup",
            "CPUQuotaPerSecUSec",
            "MemoryCurrent",
            "MemoryHigh",
            "MemoryMax",
            "MemorySwapMax",
            "ExecMainStatus",
        ],
    )
    fit_state = (fit.get("LoadState"), fit.get("ActiveState"), fit.get("SubState"))
    fit_active = fit_state in {
        ("loaded", "active", "running"),
        ("loaded", "active", "start"),
        ("loaded", "activating", "start"),
        ("loaded", "reloading", "stop-sigterm"),
    }
    fit_completed = (
        fit.get("InvocationID") == P["fit_invocation_id"]
        and fit_state == ("loaded", "active", "exited")
        and fit.get("ExecMainStatus") == "0"
    )
    fit_absent = fit_state in {
        ("not-found", "inactive", "dead"),
        ("loaded", "inactive", "dead"),
    } and not fit.get("ControlGroup")
    if not (fit_active or fit_completed or fit_absent):
        raise RuntimeError(f"propensity unit is in an unrecognized state: {fit}")
    if fit_active and (
        fit.get("LoadState") != "loaded"
        or fit.get("InvocationID") != P["fit_invocation_id"]
        or not fit.get("ControlGroup")
        or fit.get("CPUQuotaPerSecUSec") != "1s"
        or fit.get("MemoryHigh") != "7500000000"
        or fit.get("MemoryMax") != str(FIT_MEMORY)
        or fit.get("MemorySwapMax") != "0"
    ):
        raise RuntimeError(f"active fit is not the exact reviewed invocation: {fit}")
    parent_group = Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
    procs = [parent_group / "cgroup.procs", *parent_group.rglob("cgroup.procs")]
    if len(procs) > 128:
        raise RuntimeError("campaign cgroup descendant scan exceeded bound")
    populated = {str(path): path.read_text().strip() for path in procs if path.read_text().strip()}
    fit_group = fit.get("ControlGroup", "") if fit_active else ""
    fit_relative = (
        (Path("/sys/fs/cgroup") / fit_group.lstrip("/")).relative_to(parent_group)
        if fit_active and fit_group
        else None
    )
    for path in populated:
        relative = Path(path).parent.relative_to(
            Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
        )
        allowed = fit_active and (
            relative == fit_relative
            or (fit_relative is not None and relative.is_relative_to(fit_relative))
        )
        if not allowed:
            raise RuntimeError(f"unexpected populated campaign descendant: {path}")
    fit_current = int(fit.get("MemoryCurrent", "0")) if fit_active else 0
    unused_fit = max(0, FIT_MEMORY - fit_current) if fit_active else 0
    projected = int(parent["MemoryCurrent"]) + unused_fit + DICTIONARY_MEMORY
    if projected > 17_000_000_000:
        raise RuntimeError(f"campaign parent lacks conservative memory headroom: {projected}")
    cpu_reserved = (1.0 if fit_active else 0.0) + 0.5
    if cpu_reserved > 2.0:
        raise RuntimeError(f"campaign parent lacks CPU headroom: {cpu_reserved}")
    return {
        "parent": parent,
        "fit_unit": fit,
        "fit_active": fit_active,
        "populated_campaign_descendants": populated,
        "projected_memory_bytes": projected,
        "reserved_cpu_cores": cpu_reserved,
    }


def _source() -> dict[str, object]:
    raw = base64.b64decode(P["archive_b64"], validate=True)
    if len(raw) != P["archive_bytes"] or _sha_bytes(raw) != P["archive_sha256"]:
        raise ValueError("source archive bytes do not match the frozen manifest")
    expected = P["source_files"]
    if len(expected) != 162 or sum(P["source_sizes"].values()) != 1_556_908:
        raise ValueError("frozen source inventory has unexpected size")
    if SOURCE.exists() or SOURCE.is_symlink():
        raise FileExistsError(
            "immutable source destination already exists; inspect, never overwrite"
        )
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        prefix = "source-fg19-gru-51a9b168/"
        if len(members) != len(expected) or any(not member.isfile() for member in members):
            raise ValueError("source archive contains missing, non-file, or extra members")
        names = {member.name.removeprefix(prefix): member for member in members}
        if set(names) != set(expected):
            raise ValueError("source archive paths differ from the frozen manifest")
        SOURCE.mkdir(mode=0o700)
        observed: dict[str, str] = {}
        for relative, digest in expected.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("unsafe path in frozen source inventory")
            member = names[relative]
            target = SOURCE / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"source member cannot be read: {relative}")
            content = stream.read()
            if len(content) != P["source_sizes"][relative] or _sha_bytes(content) != digest:
                raise ValueError(f"source member differs from immutable commit: {relative}")
            with target.open("xb") as output:
                output.write(content)
            target.chmod(0o644)
            observed[relative] = _sha_file(target)
        if observed != expected:
            raise ValueError("extracted source tree did not reproduce all pinned files")
    return {"commit": COMMIT, "file_count": len(observed), "files_sha256": observed}


def main() -> None:
    config_raw = base64.b64decode(P["configuration_raw_b64"], validate=True)
    config = json.loads(config_raw)
    if (
        _sha_bytes(config_raw) != P["configuration_sha256"]
        or config != P["configuration"]
        or P["source_commit"] != COMMIT
    ):
        raise ValueError("frozen preparation contract changed")
    if _sha_file(SPLIT) != SPLIT_SHA or _sha_file(REGISTRY) != REGISTRY_SHA:
        raise RuntimeError("canonical fold split or portable data registry changed")
    if any(path.is_symlink() or not path.is_dir() for path in (B, DATA, B / "research_tmp")):
        raise RuntimeError("required B/data/scratch directories are not physical directories")
    if _sha_file(REGISTRY) != REGISTRY_SHA:
        raise RuntimeError("portable registry changed during preflight")
    data = _data()
    initial_resources = _resources()
    filesystem = _run(
        ["findmnt", "-T", str(B / "research_tmp"), "-n", "-o", "FSTYPE,SOURCE,TARGET"]
    ).strip()
    if (
        filesystem.split()[0] in {"tmpfs", "ramfs"}
        or shutil.disk_usage(B / "research_tmp").free < 20_000_000_000
    ):
        raise RuntimeError("physical research scratch must have at least 20 GB free")
    if any(
        os.path.lexists(path)
        for path in (SOURCE, RUN, CACHE, RECEIPT, LOGS, B / "runtime/STOP_ADMISSION.json")
    ):
        raise FileExistsError(
            "source, output, logs, or campaign-stop path exists; refusing overwrite"
        )
    if _service(UNIT, ["LoadState"]).get("LoadState") != "not-found":
        raise RuntimeError("exact dictionary fit unit already exists")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise FileNotFoundError("pinned Beleriand project interpreter is unavailable")
    source = _source()
    final_data = _data()
    final_resources = _resources()
    if final_data != data:
        raise RuntimeError("registered data stat identity changed while staging the source")
    if any(
        os.path.lexists(path)
        for path in (RUN, CACHE, RECEIPT, LOGS, B / "runtime/STOP_ADMISSION.json")
    ):
        raise FileExistsError("output, logs, or campaign-stop path appeared during staging")
    if _service(UNIT, ["LoadState"]).get("LoadState") != "not-found":
        raise RuntimeError("dictionary fit unit appeared during source staging")
    LOGS.mkdir(mode=0o700, parents=False, exist_ok=False)
    admission = {
        "schema": "fg19_dictionary_cpu_admission.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "source": source,
        "source_archive_sha256": P["archive_sha256"],
        "source_manifest_sha256": P["source_manifest_sha256"],
        "preparation_sha256": P["configuration_sha256"],
        "split_sha256": SPLIT_SHA,
        "registry_sha256": REGISTRY_SHA,
        "profile_result_sha256": P["profile_result_sha256"],
        "cold_import_receipt_sha256": P["cold_import_receipt_sha256"],
        "profile_admission_sha256": P["profile_admission_sha256"],
        "path_diagnostic_sha256": P["path_diagnostic_sha256"],
        "allowed_concurrent_fit_invocation": P["fit_invocation_id"],
        "data_stat_identity_only": data,
        "initial_resource_snapshot": initial_resources,
        "resource_snapshot": final_resources,
        "physical_scratch": filesystem,
        "limits": {
            "cpu_quota": "50%",
            "memory_high": 1_400_000_000,
            "memory_max": DICTIONARY_MEMORY,
            "swap_max": 0,
            "runtime_seconds": 600,
        },
        "scope": "one CPU-only train-fold dictionary preparation; no outer feature/target values, scoring, or downstream model fit; canonical index and graph metadata are read",
    }
    _write(LOGS / "ADMISSION.json", admission)
    env = {
        "PYTHONPATH": f"{SOURCE}/src:{SOURCE}/scripts",
        "PYTHONDONTWRITEBYTECODE": "1",
        "WNN_PROJECT_ROOT": str(DATA),
        "WNN_PORTABLE_DATA_MANIFEST": str(REGISTRY),
        "WNN_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TMPDIR": str(B / "research_tmp"),
        "WNN_TMP_ROOT": str(B / "research_tmp"),
    }
    command = [
        "systemd-run",
        "--user",
        f"--unit={UNIT}",
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=50%",
        "--property=MemoryHigh=1400000000",
        f"--property=MemoryMax={DICTIONARY_MEMORY}",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=30",
        f"--property=StandardOutput=append:{LOGS}/unit.log",
        f"--property=StandardError=append:{LOGS}/unit.log",
        f"--working-directory={SOURCE}",
    ]
    command.extend(f"--setenv={key}={value}" for key, value in env.items())
    command += [
        str(PYTHON),
        "-u",
        str(SOURCE / "scripts/fit_input_representation.py"),
        "--split-manifest",
        str(SPLIT),
        "--cache",
        str(CACHE),
        "--output",
        str(RECEIPT),
    ]
    try:
        submitted = subprocess.run(command, capture_output=True, text=True, timeout=30)
        submission = {
            "returncode": submitted.returncode,
            "stdout": submitted.stdout,
            "stderr": submitted.stderr,
            "outcome": "submitted" if submitted.returncode == 0 else "failed_or_unknown",
        }
    except subprocess.TimeoutExpired as error:
        submission = {
            "returncode": None,
            "stdout": _text(error.stdout),
            "stderr": _text(error.stderr),
            "stdout_base64_if_bytes": base64.b64encode(error.stdout).decode("ascii")
            if isinstance(error.stdout, bytes)
            else None,
            "stderr_base64_if_bytes": base64.b64encode(error.stderr).decode("ascii")
            if isinstance(error.stderr, bytes)
            else None,
            "outcome": "unknown",
        }
    keys = ["InvocationID", "ExecStart", "ActiveState", "SubState", "MainPID", "ControlGroup"]
    observed = subprocess.run(
        ["systemctl", "--user", "show", "--all", UNIT, *sum((["-p", key] for key in keys), [])],
        capture_output=True,
        text=True,
        timeout=10,
    )
    launch = {
        "schema": "fg19_dictionary_cpu_launch.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "command": command,
        "submission": submission,
        "systemctl_after_submission": {
            "returncode": observed.returncode,
            "stdout": observed.stdout,
            "stderr": observed.stderr,
        },
        "unknown_outcome_policy": "inspect exact unit and immutable paths; never blindly retry",
    }
    _write(LOGS / "LAUNCH.json", launch)
    print(
        json.dumps(
            {
                name: base64.b64encode((LOGS / name).read_bytes()).decode("ascii")
                for name in ("ADMISSION.json", "LAUNCH.json")
            }
        )
    )
    if submission["returncode"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
