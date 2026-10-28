from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Legacy trainer receipts and checkpoint provenance are heterogeneous JSON boundaries.
from typing import Any

RECEIPT_REQUIRED = {
    "run",
    "command",
    "environment",
    "pid",
    "start_ticks",
    "source_identity_sha256",
    "source_git",
    "split_sha256",
    "limits",
    "started_utc",
}
EVENT_SCHEMA = "wnn.monitor.v1"
BRIDGE_SCHEMA = "wnn.tl3-dashboard-bridge.v2"
ADOPTION_SCHEMA = "wnn.tl3-fit-adoption.v1"
TRAINING_SOURCE_SCHEMA = "wnn.training-source.v1"
MAX_JSON_BYTES = 16 << 20
MAX_LOG_BYTES = 64 << 20
MAX_VIEW_LOG_BYTES = 8 << 20
MAX_EVENT_BYTES = 131_072
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EVENT_METRIC = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
PROGRESS = re.compile(
    r"^ep (?P<epoch>\d+) batch (?P<batch>\d+)/(?P<total>\d+) "
    r"loss (?P<loss>[-+0-9.eE]+) lr (?P<lr>[-+0-9.eE]+) "
    r"seq/s (?P<rate>[-+0-9.eE]+)$"
)


class BridgeError(RuntimeError):
    """Raised when receipt, artifact, or ownership evidence is incomplete."""


@dataclass(frozen=True)
class RunBinding:
    """Paths and launch identity pinned by one TL3 launch receipt."""

    run: str
    data_root: Path
    source_root: Path
    output: Path
    log: Path
    split_manifest: Path
    gpu_uuid: str
    parent_pid: int
    parent_start_ticks: int
    command: tuple[str, ...]
    receipt_sha256: str
    adoption_sha256: str
    source_identity_file_sha256: str
    actual_training_source_sha256: str
    training_source_identity_record_sha256: str
    source_identity_file: Path
    training_source_identity_file: Path
    split_sha256: str


def _read_json(path: Path, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Read one bounded regular JSON object without following a final symlink."""
    if path.is_symlink() or not path.is_file():
        raise BridgeError(f"expected a regular non-symlink JSON file: {path}")
    size = path.stat().st_size
    if not 0 < size <= maximum:
        raise BridgeError(f"JSON file is outside its {maximum}-byte bound: {path.name}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise BridgeError(f"JSON file changed while being read: {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise BridgeError(f"expected a JSON object: {path.name}")
    return value


def _sha256_file(path: Path) -> str:
    """Hash a stable regular file in bounded chunks."""
    if path.is_symlink() or not path.is_file():
        raise BridgeError(f"expected a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
        after = os.fstat(source.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise BridgeError(f"file changed while hashing: {path.name}")
    return digest.hexdigest()


def _command_value(command: tuple[str, ...], flag: str) -> str:
    """Return one unique flag value from a frozen argv vector."""
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise BridgeError(f"launch command must contain one {flag} value")
    return command[positions[0] + 1]


def load_binding(
    receipt_path: Path,
    adoption_path: Path | None = None,
    training_source_identity_path: Path | None = None,
    source_identity_file_path: Path | None = None,
) -> tuple[RunBinding, dict[str, Any]]:
    """Validate launch, adoption, and both independent source identity scopes."""
    receipt = _read_json(receipt_path, 1 << 20)
    if not RECEIPT_REQUIRED.issubset(receipt):
        raise BridgeError("TL3 launch receipt is missing required ownership fields")
    run = receipt["run"]
    command = receipt["command"]
    environment = receipt["environment"]
    if not isinstance(run, str) or not RUN_ID.fullmatch(run) or ".." in run:
        raise BridgeError("launch receipt contains an unsafe run identifier")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise BridgeError("launch receipt command is malformed")
    if not isinstance(environment, dict):
        raise BridgeError("launch receipt environment is malformed")
    data_root_value = environment.get("WNN_PROJECT_ROOT")
    gpu_uuid = environment.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(data_root_value, str) or not isinstance(gpu_uuid, str):
        raise BridgeError("launch receipt omits data root or visible GPU UUID")
    if not SHA256.fullmatch(str(receipt["source_identity_sha256"])) or not SHA256.fullmatch(
        str(receipt["split_sha256"])
    ):
        raise BridgeError("launch receipt source or split digest is malformed")
    if type(receipt["pid"]) is not int or receipt["pid"] < 2:
        raise BridgeError("launch receipt timeout PID is invalid")
    try:
        parent_ticks = int(receipt["start_ticks"])
    except (TypeError, ValueError) as error:
        raise BridgeError("launch receipt start ticks are invalid") from error
    data_root = Path(data_root_value).resolve(strict=True)
    script_index = next(
        (
            index
            for index, value in enumerate(command)
            if value.replace("\\", "/").endswith("/scripts/train_rnn.py")
        ),
        None,
    )
    if script_index is None:
        raise BridgeError("launch command does not identify scripts/train_rnn.py")
    source_script = Path(command[script_index]).resolve(strict=True)
    source_root = source_script.parent.parent
    if not source_script.is_file():
        raise BridgeError("receipt-owned trainer source is missing")
    if _command_value(tuple(command), "--name") != run:
        raise BridgeError("launch run name disagrees with trainer argv")
    manifest = Path(_command_value(tuple(command), "--split-manifest")).resolve(strict=True)
    node_id = run.removeprefix("tl3c_")
    adoption_path = adoption_path or receipt_path.with_name(f"{node_id}.adoption.json")
    training_source_identity_path = training_source_identity_path or receipt_path.with_name(
        "adopted-source.identity.json"
    )
    source_identity_file_path = source_identity_file_path or (
        source_root.parent / f"{source_root.name}.identity.json"
    )
    adoption = _read_json(adoption_path, 1 << 20)
    training_source_identity = _read_json(training_source_identity_path, 1 << 20)
    source_identity_file = _read_json(source_identity_file_path, 8 << 20)
    adoption_fields = {
        "schema",
        "node_id",
        "run",
        "source_sha256",
        "source_identity_file_sha256",
        "source_root",
        "argv",
        "training_args",
        "pid",
        "start_ticks",
    }
    if not adoption_fields.issubset(adoption):
        raise BridgeError("TL3 adoption receipt is missing required source-binding fields")
    if (
        adoption.get("schema") != ADOPTION_SCHEMA
        or adoption.get("node_id") != node_id
        or adoption.get("run") != run
        or adoption.get("argv") != command
        or adoption.get("pid") != receipt["pid"]
        or str(adoption.get("start_ticks")) != str(receipt["start_ticks"])
        or adoption.get("source_identity_file_sha256") != receipt["source_identity_sha256"]
        or not SHA256.fullmatch(str(adoption.get("source_sha256")))
    ):
        raise BridgeError("TL3 adoption receipt does not bind the exact launch and two source hashes")
    if Path(str(adoption.get("source_root"))).resolve(strict=True) != source_root:
        raise BridgeError("TL3 adoption source root differs from the launch command")
    if adoption.get("training_args") != list(command[script_index + 1 :]):
        raise BridgeError("TL3 adoption trainer argv differs from the launch command")
    source_identity_file_sha256 = _sha256_file(source_identity_file_path)
    if source_identity_file_sha256 != receipt["source_identity_sha256"]:
        raise BridgeError("full source identity file bytes differ from the launch receipt")
    if source_identity_file.get("git") != receipt.get("source_git"):
        raise BridgeError("full source identity git field differs from the launch receipt")
    if not isinstance(source_identity_file.get("files"), dict):
        raise BridgeError("full source identity file omits its file hash map")
    if training_source_identity.get("schema") != TRAINING_SOURCE_SCHEMA:
        raise BridgeError("adopted trainer source identity schema is invalid")
    training_files = training_source_identity.get("files")
    if (
        not isinstance(training_files, dict)
        or training_source_identity.get("sha256") != adoption["source_sha256"]
        or _canonical_sha256(training_files) != adoption["source_sha256"]
    ):
        raise BridgeError("adopted trainer source identity hash does not match its file map")
    full_source_files = source_identity_file["files"]
    for relative_name, expected_sha256 in training_files.items():
        if (
            not isinstance(relative_name, str)
            or not isinstance(expected_sha256, str)
            or not SHA256.fullmatch(expected_sha256)
        ):
            raise BridgeError("adopted trainer source identity has a malformed file entry")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise BridgeError("adopted trainer source identity has an unsafe path")
        source_file = source_root.joinpath(*relative.parts)
        resolved_source_file = source_file.resolve(strict=True)
        if resolved_source_file != source_file or not resolved_source_file.is_relative_to(source_root):
            raise BridgeError(f"trainer source path escapes or aliases its root: {relative_name}")
        if _sha256_file(resolved_source_file) != expected_sha256:
            raise BridgeError(f"trainer source bytes differ from adoption: {relative_name}")
        if relative.suffix == ".py" and full_source_files.get(relative_name) != expected_sha256:
            raise BridgeError(f"trainer source file differs from the full source map: {relative_name}")
    output = data_root / "artifacts" / "runs" / run
    log = data_root / "logs" / f"{run}.log"
    if output.resolve(strict=False) != output or log.resolve(strict=False) != log:
        raise BridgeError("launch output paths contain unresolved components")
    if output.is_symlink() or log.is_symlink():
        raise BridgeError("source run output or log cannot be a symlink")
    binding = RunBinding(
        run=run,
        data_root=data_root,
        source_root=source_root,
        output=output,
        log=log,
        split_manifest=manifest,
        gpu_uuid=gpu_uuid,
        parent_pid=receipt["pid"],
        parent_start_ticks=parent_ticks,
        command=tuple(command),
        receipt_sha256=_sha256_file(receipt_path),
        adoption_sha256=_sha256_file(adoption_path),
        source_identity_file_sha256=source_identity_file_sha256,
        actual_training_source_sha256=str(adoption["source_sha256"]),
        training_source_identity_record_sha256=_canonical_sha256(training_source_identity),
        source_identity_file=source_identity_file_path.resolve(strict=True),
        training_source_identity_file=training_source_identity_path.resolve(strict=True),
        split_sha256=str(receipt["split_sha256"]),
    )
    return binding, receipt


def _read_proc_stat(pid: int, proc_root: Path = Path("/proc")) -> dict[str, int | str] | None:
    """Read process identity and parent PID without signaling the process."""
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    try:
        return {
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "process_group": int(fields[2]),
            "start_ticks": int(fields[19]),
        }
    except (IndexError, ValueError):
        return None


def _trainer_argv(command: tuple[str, ...]) -> tuple[str, ...]:
    """Get the Python trainer argv suffix wrapped by timeout/taskset."""
    for index, value in enumerate(command):
        name = Path(value).name.casefold()
        if name in {"python", "python3", "python3.11"} or name.startswith("python3."):
            return command[index:]
    raise BridgeError("launch receipt does not contain a Python trainer command")


def find_owned_trainer(
    binding: RunBinding, proc_root: Path = Path("/proc")
) -> tuple[int, dict[str, int | str]] | None:
    """Find the exact trainer argv among descendants of the receipt-owned timeout PID."""
    parent = _read_proc_stat(binding.parent_pid, proc_root)
    if parent is None:
        return None
    if parent["start_ticks"] != binding.parent_start_ticks:
        raise BridgeError("timeout PID was reused; refusing to trust its descendants")
    descendants: dict[int, dict[str, int | str]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        identity = _read_proc_stat(int(entry.name), proc_root)
        if identity is not None:
            descendants[int(entry.name)] = identity
    owned: set[int] = {binding.parent_pid}
    changed = True
    while changed:
        changed = False
        for pid, identity in descendants.items():
            if pid not in owned and identity["parent_pid"] in owned:
                owned.add(pid)
                changed = True
    expected = list(_trainer_argv(binding.command))
    matches: list[tuple[int, dict[str, int | str]]] = []
    for pid in sorted(owned - {binding.parent_pid}):
        try:
            argv = (
                (proc_root / str(pid) / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            )
        except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
            continue
        if argv == expected:
            identity = descendants[pid]
            if identity["state"] not in {"Z", "X"}:
                matches.append((pid, identity))
    if len(matches) > 1:
        raise BridgeError("multiple receipt-descended processes match the trainer argv")
    return matches[0] if matches else None


def is_timeout_parent_live(binding: RunBinding, proc_root: Path = Path("/proc")) -> bool:
    """Check the receipt-owned timeout process without signaling it."""
    parent = _read_proc_stat(binding.parent_pid, proc_root)
    if parent is None:
        return False
    if parent["start_ticks"] != binding.parent_start_ticks:
        raise BridgeError("timeout PID was reused; refusing to infer run completion")
    return parent["state"] not in {"Z", "X"}


def has_exact_trainer_process(binding: RunBinding, proc_root: Path = Path("/proc")) -> bool:
    """Detect a receipt-exact trainer that outlived or was reparented from timeout."""
    expected = list(_trainer_argv(binding.command))
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == binding.parent_pid:
            continue
        pid = int(entry.name)
        identity = _read_proc_stat(pid, proc_root)
        if identity is None or identity["state"] in {"Z", "X"}:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
            continue
        if argv == expected:
            return True
    return False


def _run_nvidia_smi(arguments: list[str]) -> str:
    """Run one bounded read-only NVIDIA inventory query."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BridgeError("nvidia-smi resource query failed") from error
    if len(completed.stdout) > 1 << 20:
        raise BridgeError("nvidia-smi response exceeded its bounded size")
    return completed.stdout


def query_verified_resources(
    trainer_pid: int,
    trainer_identity: dict[str, int | str],
    gpu_uuid: str,
    proc_root: Path = Path("/proc"),
) -> dict[str, int | float] | None:
    """Return resources only when PID, launch UUID, and active GPU process agree."""
    current = _read_proc_stat(trainer_pid, proc_root)
    if current is None or current != trainer_identity:
        raise BridgeError("trainer process identity changed before resource sampling")
    gpu_rows = list(
        csv.reader(
            _run_nvidia_smi(
                [
                    "--query-gpu=uuid,memory.used,memory.free,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            ).splitlines()
        )
    )
    device = [
        tuple(field.strip() for field in row)
        for row in gpu_rows
        if row and row[0].strip() == gpu_uuid
    ]
    if len(device) != 1:
        raise BridgeError("receipt GPU UUID is not uniquely present in nvidia-smi")
    process_rows = list(
        csv.reader(
            _run_nvidia_smi(
                [
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            ).splitlines()
        )
    )
    process = [
        tuple(field.strip() for field in row)
        for row in process_rows
        if len(row) == 3 and row[0].strip() == str(trainer_pid) and row[1].strip() == gpu_uuid
    ]
    if len(process) != 1:
        return None
    status_path = proc_root / str(trainer_pid) / "status"
    rss_kib = None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass
    process_memory = _numeric(process[0][2])
    device_memory = [_numeric(value) for value in device[0][1:]]
    metrics: dict[str, int | float] = {}
    if process_memory is not None:
        metrics["gpu/process_memory_mib"] = process_memory
    for key, value in zip(
        (
            "gpu/device_memory_used_mib",
            "gpu/device_memory_free_mib",
            "gpu/utilization_pct",
            "gpu/power_watts",
        ),
        device_memory,
        strict=True,
    ):
        if value is not None:
            metrics[key] = value
    if rss_kib is not None:
        metrics["cpu/process_rss_mib"] = rss_kib / 1024.0
    return metrics


def _numeric(value: str) -> int | float | None:
    """Parse one finite nvidia-smi numeric field; N/A remains absent."""
    try:
        number = float(value.strip().removesuffix(" MiB").removesuffix(" W"))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _read_log(path: Path) -> list[str]:
    """Read complete bounded log lines while ignoring a concurrent partial tail."""
    if path.is_symlink() or not path.exists():
        return []
    if not stat.S_ISREG(path.lstat().st_mode):
        raise BridgeError("legacy trainer log is not a regular file")
    size = path.stat().st_size
    if size > MAX_LOG_BYTES:
        raise BridgeError("legacy trainer log exceeded the 64 MiB bridge bound")
    raw = path.read_bytes()
    if len(raw) > size or len(raw) > MAX_LOG_BYTES:
        raise BridgeError("legacy trainer log changed outside its read bound")
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    return raw.decode("utf-8", errors="replace").splitlines()


def parse_legacy_log(lines: list[str]) -> dict[str, Any]:
    """Extract machine-readable epoch records and explicit batch progress."""
    valid: dict[int, dict[str, Any]] = {}
    gradients: dict[int, dict[str, Any]] = {}
    progress: list[dict[str, Any]] = []
    selected_fixed: int | None = None
    for line in lines:
        for prefix, destination in (("VALID ", valid), ("GRAD_STATS ", gradients)):
            if line.startswith(prefix):
                try:
                    record = json.loads(line[len(prefix) :])
                except ValueError:
                    continue
                if isinstance(record, dict) and type(record.get("epoch")) is int:
                    destination[record["epoch"]] = record
                break
        else:
            match = PROGRESS.fullmatch(line)
            if match is not None:
                record = {
                    key: int(value)
                    for key, value in match.groupdict().items()
                    if key in {"epoch", "batch", "total"}
                }
                record.update({key: float(match.group(key)) for key in ("loss", "lr", "rate")})
                progress.append(record)
                continue
            if line.startswith("SELECTED_FIXED "):
                with contextlib.suppress(ValueError):
                    selected_fixed = int(line.partition(" ")[2])
    return {
        "valid": valid,
        "gradients": gradients,
        "progress": progress,
        "selected_fixed": selected_fixed,
    }


def _canonical_sha256(value: object) -> str:
    """Hash one canonical JSON-safe value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_run_provenance(
    binding: RunBinding, provenance: dict[str, Any]
) -> tuple[dict[str, Any], list[int], list[int], list[int]]:
    """Bind run code, official split, and actual fit/evaluation memberships."""
    identity = provenance.get("trainer_implementation")
    if (
        not isinstance(identity, dict)
        or identity.get("sha256") != binding.actual_training_source_sha256
        or _canonical_sha256(identity) != binding.training_source_identity_record_sha256
    ):
        raise BridgeError("run trainer source identity differs from the adopted training source")
    if provenance.get("split_manifest_sha256") != binding.split_sha256:
        raise BridgeError("run provenance split hash differs from its launch receipt")
    manifest = _read_json(binding.split_manifest)
    if _sha256_file(binding.split_manifest) != binding.split_sha256:
        raise BridgeError("official split manifest bytes differ from the launch receipt")
    if manifest.get("schema") != "wnn.chronological-primary.v1":
        raise BridgeError("run split is not the official chronological primary manifest")
    fit = manifest.get("fit")
    evaluation = manifest.get("evaluation")
    if not isinstance(fit, dict) or not isinstance(evaluation, dict):
        raise BridgeError("chronological split manifest has no fit/evaluation membership")
    expected_train = [int(value) for value in fit.get("train", [])]
    expected_fit_valid = [int(value) for value in fit.get("valid", [])]
    expected_valid = [int(value) for value in evaluation.get("valid", [])]
    groups = provenance.get("fit_row_groups")
    train_groups = groups.get("train") if isinstance(groups, dict) else None
    fit_valid_groups = groups.get("valid") if isinstance(groups, dict) else None
    evaluation_groups = provenance.get("evaluation_row_groups")
    if not all(
        isinstance(value, list) for value in (train_groups, fit_valid_groups, evaluation_groups)
    ):
        raise BridgeError("run provenance omits actual row-group membership")
    train = [int(value) for value in train_groups]
    fit_valid = [int(value) for value in fit_valid_groups]
    evaluated = [int(value) for value in evaluation_groups]
    if (
        len(set(train)) != len(train)
        or len(set(fit_valid)) != len(fit_valid)
        or len(set(evaluated)) != len(evaluated)
        or set(train) != set(expected_train)
        or fit_valid
        or set(fit_valid) != set(expected_fit_valid)
    ):
        raise BridgeError("actual train fit membership differs from official train-only manifest")
    if set(evaluated) != set(expected_valid):
        raise BridgeError("actual evaluation membership differs from full official validation")
    args = provenance.get("args")
    if not isinstance(args, dict):
        raise BridgeError("run provenance omits effective trainer arguments")
    eval_source = args.get("eval_source")
    if eval_source not in {"dev", "vfold"}:
        raise BridgeError("effective trainer arguments do not select official validation")
    if eval_source == "vfold" and (
        type(args.get("valid_fold")) is not int
        or args["valid_fold"] != 0
        or type(args.get("n_valid_folds")) is not int
        or args["n_valid_folds"] != 1
    ):
        raise BridgeError("manifest-managed vfold is not the single full official-valid fold")
    if (
        args.get("name") != binding.run
        or args.get("train_source") != "train"
        or args.get("full_valid") is not True
        or args.get("train_seqs") != len(expected_train)
        or args.get("no_eval_during_fit") is not False
        or args.get("checkpoint_policy") != "fixed"
        or type(args.get("epochs")) is not int
        or args["epochs"] <= 0
        or type(args.get("checkpoint_epoch")) is not int
        or not 0 <= args["checkpoint_epoch"] < args["epochs"]
        or args.get("precision") != "fp32"
        or args.get("resume_state")
        or args.get("propensity")
    ):
        raise BridgeError("effective trainer recipe violates the fresh train-only TL3 contract")
    return args, train, fit_valid, evaluated


def _safe_mkdir(path: Path, runs_root: Path, binding: RunBinding) -> None:
    """Create one dashboard direct-child run folder or verify bridge ownership."""
    runs_root = runs_root.expanduser().absolute()
    if (
        runs_root.is_symlink()
        or not runs_root.is_dir()
        or runs_root.resolve(strict=True) != runs_root
    ):
        raise BridgeError("dashboard runs root must be an existing resolved directory")
    if path.parent != runs_root or path.name != binding.run or path.is_symlink():
        raise BridgeError("dashboard view must be a direct run-ID child without symlinks")
    marker = path / "bridge_receipt.json"
    if not path.exists():
        path.mkdir(mode=0o700)
        _atomic_json(
            marker,
            {
                "schema": BRIDGE_SCHEMA,
                "run": binding.run,
                "launch_receipt_sha256": binding.receipt_sha256,
                "adoption_receipt_sha256": binding.adoption_sha256,
                "source_identity_file_sha256": binding.source_identity_file_sha256,
                "actual_training_source_sha256": binding.actual_training_source_sha256,
                "training_source_identity_record_sha256": binding.training_source_identity_record_sha256,
                "source_identity_file": str(binding.source_identity_file),
                "training_source_identity_file": str(binding.training_source_identity_file),
                "source_output": str(binding.output),
            },
            exclusive=True,
        )
        for filename in ("monitor_events.jsonl", "monitor_evaluations.jsonl"):
            descriptor = os.open(path / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
    elif not path.is_dir() or marker.is_symlink():
        raise BridgeError("dashboard view path exists without bridge ownership")
    else:
        existing = _read_json(marker, 64_000)
        if (
            existing.get("schema") != BRIDGE_SCHEMA
            or existing.get("run") != binding.run
            or existing.get("launch_receipt_sha256") != binding.receipt_sha256
            or existing.get("adoption_receipt_sha256") != binding.adoption_sha256
            or existing.get("source_identity_file_sha256") != binding.source_identity_file_sha256
            or existing.get("actual_training_source_sha256")
            != binding.actual_training_source_sha256
            or existing.get("training_source_identity_record_sha256")
            != binding.training_source_identity_record_sha256
            or existing.get("source_identity_file") != str(binding.source_identity_file)
            or existing.get("training_source_identity_file")
            != str(binding.training_source_identity_file)
            or existing.get("source_output") != str(binding.output)
        ):
            raise BridgeError("dashboard view is owned by a different bridge receipt")


def _atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    """Durably write bounded bridge metadata without replacing a foreign link."""
    if path.is_symlink():
        raise BridgeError(f"bridge output cannot be a symlink: {path.name}")
    if exclusive and path.exists():
        raise BridgeError(f"bridge output already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise BridgeError("bridge temporary output path already exists")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def _event(
    kind: str,
    step: int,
    epoch: int | float,
    metrics: dict[str, int | float],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Create one validated, bounded dashboard monitor event."""
    if kind not in {"status", "train_step", "evaluation", "diagnostics", "resources"}:
        raise BridgeError("invalid monitor event kind")
    if (
        type(step) is not int
        or step < 0
        or isinstance(epoch, bool)
        or not isinstance(epoch, (int, float))
    ):
        raise BridgeError("monitor event position is invalid")
    if not math.isfinite(float(epoch)) or epoch < 0:
        raise BridgeError("monitor event epoch is nonfinite")
    if kind == "evaluation" and type(epoch) is not int:
        raise BridgeError("evaluation event epoch must be an integer")
    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or EVENT_METRIC.fullmatch(key) is None
            or len(key) > 128
            or isinstance(value, bool)
        ):
            raise BridgeError("monitor metrics require slash keys and numeric values")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise BridgeError(f"monitor metric is nonfinite: {key}")
    event = {
        "schema": EVENT_SCHEMA,
        "kind": kind,
        "step": step,
        "epoch": epoch,
        "wall_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
        "context": context,
    }
    encoded = (
        json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > MAX_EVENT_BYTES:
        raise BridgeError("monitor event exceeds the bounded dashboard line size")
    return event


def _read_events(path: Path) -> dict[tuple[str, int | float, int], dict[str, Any]]:
    """Read the small bridge event stream and index stable event identities."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_VIEW_LOG_BYTES:
        raise BridgeError("dashboard bridge event file is unsafe or exceeds its bound")
    result: dict[tuple[str, int | float, int], dict[str, Any]] = {}
    for line in path.read_bytes().splitlines():
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA:
            raise BridgeError("dashboard bridge event log contains an invalid record")
        key = (str(event.get("kind")), event.get("epoch"), event.get("step"))
        if key in result:
            raise BridgeError("dashboard bridge event log contains duplicate identities")
        result[key] = event
    return result


def _append_event(
    path: Path, event: dict[str, Any], known: dict[tuple[str, int | float, int], dict[str, Any]]
) -> bool:
    """Append a new event once, treating wall time as observation-only metadata."""
    key = (event["kind"], event["epoch"], event["step"])
    old = known.get(key)
    if old is not None:
        stable_old = {name: value for name, value in old.items() if name != "wall_time"}
        stable_new = {name: value for name, value in event.items() if name != "wall_time"}
        if stable_old != stable_new:
            if event["kind"] == "resources":
                return False
            raise BridgeError(f"replayed event conflicts with existing {event['kind']} record")
        return False
    encoded = (
        json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    if path.stat().st_size + len(encoded) > MAX_VIEW_LOG_BYTES:
        raise BridgeError("dashboard bridge log reached its 8 MiB cap")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "ab", buffering=0) as stream:
            stream.write(encoded)
            os.fsync(stream.fileno())
    except BaseException:
        raise
    known[key] = event
    return True


def _append_evaluation(
    view: Path, event: dict[str, Any], known: dict[tuple[str, int | float, int], dict[str, Any]]
) -> None:
    """Append one completed pooled evaluation to both dashboard streams."""
    archive = view / "monitor_evaluations.jsonl"
    journal = _read_events(archive)
    prior = [
        record
        for (kind, epoch, _), record in journal.items()
        if kind == "evaluation" and epoch == event["epoch"]
    ]
    if prior:
        if len(prior) != 1:
            raise BridgeError("evaluation archive contains duplicate epochs")
        stable_old = {name: value for name, value in prior[0].items() if name != "wall_time"}
        stable_new = {name: value for name, value in event.items() if name != "wall_time"}
        if stable_old != stable_new:
            raise BridgeError("completed evaluation conflicts with the archived epoch")
    else:
        _append_event(archive, event, journal)
    _append_event(view / "monitor_events.jsonl", event, known)


def _flatten_gradients(record: dict[str, Any]) -> dict[str, int | float]:
    """Map legacy pre-clip GradStats scalars to dashboard slash keys."""
    metrics: dict[str, int | float] = {}
    for source, target in (
        ("steps", "optimizer/gradient_steps"),
        ("nonfinite_steps", "optimizer/gradient_nonfinite_steps"),
        ("clip", "optimizer/gradient_clip_threshold"),
        ("clip_rate", "optimizer/gradient_clip_rate"),
        ("mean_clip_factor", "optimizer/gradient_mean_clip_factor"),
    ):
        value = record.get(source)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            metrics[target] = value
    quantiles = record.get("total_q05_q50_q95")
    if isinstance(quantiles, list) and len(quantiles) == 3:
        for suffix, value in zip(("q05", "q50", "q95"), quantiles, strict=True):
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"optimizer/gradient_norm_preclip/{suffix}"] = value
    for field, prefix in (
        ("family_median", "optimizer/gradient_family_median"),
        ("family_sq_share_mean", "optimizer/gradient_family_sq_share"),
    ):
        values = record.get(field)
        if isinstance(values, dict):
            for name, value in values.items():
                if (
                    isinstance(name, str)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                ):
                    metrics[f"{prefix}/{name}"] = value
    return metrics


def _run_evaluator(
    binding: RunBinding, run_identity: str, epoch_index: int, batch_size: int, timeout: int
) -> dict[str, Any]:
    """Run the bounded frozen-checkpoint evaluation in a separate process."""
    worker = Path(__file__).with_name("tl3_checkpoint_eval.py").resolve(strict=True)
    command = [
        sys.executable,
        str(worker),
        "--run-dir",
        str(binding.output),
        "--data-root",
        str(binding.data_root),
        "--source-root",
        str(binding.source_root),
        "--source-identity-file",
        str(binding.source_identity_file),
        "--source-identity-file-sha256",
        binding.source_identity_file_sha256,
        "--split-manifest",
        str(binding.split_manifest),
        "--epoch-index",
        str(epoch_index),
        "--run-identity-sha256",
        run_identity,
        "--training-source-sha256",
        binding.actual_training_source_sha256,
        "--batch-size",
        str(batch_size),
    ]
    env_keys = ("PATH", "HOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CUDA_HOME", "TORCH_HOME")
    environment = {key: os.environ[key] for key in env_keys if key in os.environ}
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": binding.gpu_uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "WNN_PROJECT_ROOT": str(binding.data_root),
            "PYTHONPATH": str(binding.source_root / "src"),
            "WNN_THREADS": "2",
            "WNN_LOADER_WORKERS": "0",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise BridgeError("frozen checkpoint evaluation exceeded its timeout") from error
    except (OSError, subprocess.CalledProcessError) as error:
        message = getattr(error, "stderr", "")
        raise BridgeError(f"frozen checkpoint evaluation failed: {str(message)[:2000]}") from error
    if len(completed.stdout) > 1 << 20:
        raise BridgeError("frozen checkpoint evaluation receipt exceeded its size bound")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise BridgeError("frozen checkpoint worker did not return an object")
    return result


def _run_identity(binding: RunBinding) -> str:
    """Bind chart events to this exact launch receipt, source and split."""
    return _canonical_sha256(
        {
            "run": binding.run,
            "receipt_sha256": binding.receipt_sha256,
            "adoption_receipt_sha256": binding.adoption_sha256,
            "source_identity_file_sha256": binding.source_identity_file_sha256,
            "actual_training_source_sha256": binding.actual_training_source_sha256,
            "split_sha256": binding.split_sha256,
        }
    )


def _metadata(
    binding: RunBinding,
    receipt: dict[str, Any],
    provenance: dict[str, Any] | None,
    run_identity: str,
) -> dict[str, Any]:
    """Build the dashboard's existing run metadata envelope."""
    return {
        "source": "TL3 chronological primary fit (legacy trainer bridge)",
        "config": {
            "run": binding.run,
            "command": receipt["command"],
            "environment": {
                "CUDA_VISIBLE_DEVICES": binding.gpu_uuid,
                "WNN_PROJECT_ROOT": str(binding.data_root),
                "threads": receipt["environment"].get("WNN_THREADS"),
            },
            "effective_args": provenance.get("args", {}) if provenance else {},
            "status": "ready" if provenance else "awaiting_run_provenance",
        },
        "provenance": {
            "bridge_schema": BRIDGE_SCHEMA,
            "run_identity_sha256": run_identity,
            "launch_receipt_sha256": binding.receipt_sha256,
            "adoption_receipt_sha256": binding.adoption_sha256,
            "source_identity_file_sha256": binding.source_identity_file_sha256,
            "actual_training_source_sha256": binding.actual_training_source_sha256,
            "training_source_identity_record_sha256": binding.training_source_identity_record_sha256,
            "source_identity_file": str(binding.source_identity_file),
            "training_source_identity_file": str(binding.training_source_identity_file),
            "source_git": receipt["source_git"],
            "launch_profile_evidence": receipt.get("profile_evidence", {}),
            "launch_checkpoint_policy": receipt.get("checkpoint_policy"),
            "split_manifest": str(binding.split_manifest),
            "split_manifest_sha256": binding.split_sha256,
            "trainer": provenance.get("trainer_implementation", {}) if provenance else {},
            "fit_row_groups": provenance.get("fit_row_groups", {}) if provenance else {},
            "evaluation_row_groups": provenance.get("evaluation_row_groups", [])
            if provenance
            else [],
            "input_feature_schema": provenance.get("input_feature_schema", {})
            if provenance
            else {},
            "normalization_identity": provenance.get("normalization_identity", {})
            if provenance
            else {},
            "source_output_read_only": str(binding.output),
        },
        "synthetic": False,
    }


def _event_context(
    binding: RunBinding, run_identity: str, checkpoint_sha: str | None = None
) -> dict[str, Any]:
    """Shared run identity fields for every bridge event."""
    context: dict[str, Any] = {
        "run": binding.run,
        "run_identity_sha256": run_identity,
        "source_identity_file_sha256": binding.source_identity_file_sha256,
        "actual_training_source_sha256": binding.actual_training_source_sha256,
        "split_manifest_sha256": binding.split_sha256,
        "source_timestamp_available": False,
    }
    if checkpoint_sha is not None:
        context["checkpoint_sha256"] = checkpoint_sha
    return context


def _set_evaluation_status(
    view: Path,
    binding: RunBinding,
    run_identity: str,
    pending: bool,
    known: dict[tuple[str, int | float, int], dict[str, Any]],
) -> bool:
    """Record only transitions into or out of pending evaluation state."""
    step = 0 if pending else 1
    event = _event(
        "status",
        step,
        0,
        {"evaluation/pending": int(pending)},
        {
            **_event_context(binding, run_identity),
            "evaluation_status": "pending" if pending else "complete",
        },
    )
    return _append_event(view / "monitor_events.jsonl", event, known)


def bridge_once(
    receipt_path: Path,
    dashboard_runs_root: Path,
    *,
    adoption_path: Path | None = None,
    training_source_identity_path: Path | None = None,
    source_identity_file_path: Path | None = None,
    evaluate_missing: bool = True,
    eval_batch_size: int = 16,
    eval_timeout: int = 1800,
) -> dict[str, Any]:
    """Import one bounded snapshot from a receipt-owned legacy fit."""
    binding, receipt = load_binding(
        receipt_path,
        adoption_path,
        training_source_identity_path,
        source_identity_file_path,
    )
    view = dashboard_runs_root / binding.run
    _safe_mkdir(view, dashboard_runs_root, binding)
    run_identity = _run_identity(binding)
    provenance_path = binding.output / "provenance.json"
    provenance = (
        _read_json(provenance_path)
        if provenance_path.is_file() and not provenance_path.is_symlink()
        else None
    )
    metadata = _metadata(binding, receipt, provenance, run_identity)
    _atomic_json(view / "run_metadata.json", metadata)
    known = _read_events(view / "monitor_events.jsonl")
    if provenance is None:
        added = int(_set_evaluation_status(view, binding, run_identity, True, known))
        return {"run": binding.run, "status": "awaiting_provenance", "events_added": added}
    args, train_groups, _, valid_groups = validate_run_provenance(binding, provenance)
    log_records = parse_legacy_log(_read_log(binding.log))
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict):
        raise BridgeError("run provenance has no checkpoint hash map")
    added = 0
    pending = False
    progress_epochs: dict[int, int] = {}
    for point in log_records["progress"]:
        epoch_index, batch_index, total_batches = point["epoch"], point["batch"], point["total"]
        if total_batches < 1 or batch_index >= total_batches:
            continue
        progress_epochs[epoch_index] = total_batches
        global_batch = epoch_index * total_batches + batch_index + 1
        event = _event(
            "train_step",
            global_batch,
            epoch_index + (batch_index + 1) / total_batches,
            {
                "train/loss": point["loss"],
                "train/objective_batch": point["loss"],
                "optimizer/lr": point["lr"],
                "throughput/sequences_per_second": point["rate"],
                "train/epoch_fraction": (batch_index + 1) / total_batches,
            },
            {
                **_event_context(binding, run_identity),
                "objective_definition": "legacy combined training scalar; not validation WMSE",
                "step_unit": "logged training batch ordinal",
                "source_epoch_index": epoch_index,
            },
        )
        added += int(_append_event(view / "monitor_events.jsonl", event, known))
    for epoch_index, valid in sorted(log_records["valid"].items()):
        checkpoint_name = f"model_ep{epoch_index}.pt"
        checkpoint = binding.output / checkpoint_name
        expected_sha = checkpoint_hashes.get(checkpoint_name)
        if not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha):
            pending = True
            continue
        checkpoint_sha = _sha256_file(checkpoint)
        if checkpoint_sha != expected_sha:
            raise BridgeError(f"epoch checkpoint hash mismatch: {checkpoint_name}")
        update_diagnostics = valid.get("update_diagnostics", {})
        cumulative = (
            update_diagnostics.get("cumulative", {}) if isinstance(update_diagnostics, dict) else {}
        )
        step = cumulative.get("optimizer_attempts") if isinstance(cumulative, dict) else None
        if type(step) is not int:
            raise BridgeError("VALID log record lacks cumulative optimizer attempts")
        grad = log_records["gradients"].get(epoch_index)
        diagnostic_metrics = _flatten_gradients(grad) if isinstance(grad, dict) else {}
        for source, target in (
            ("weighted_pearson", "valid/score"),
            ("global_t0", "valid/score/t0"),
            ("global_t1", "valid/score/t1"),
        ):
            value = valid.get(source)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                diagnostic_metrics[target] = value
        if diagnostic_metrics:
            diagnostic_context = {
                **_event_context(binding, run_identity, checkpoint_sha),
                "valid_score_source": "trainer VALID log; pooled full official validation",
                "source_epoch_index": epoch_index,
            }
            if isinstance(grad, dict):
                diagnostic_context["gradient_timing"] = "after unscale and before clipping"
            event = _event(
                "diagnostics",
                step,
                epoch_index + 1,
                diagnostic_metrics,
                diagnostic_context,
            )
            added += int(_append_event(view / "monitor_events.jsonl", event, known))
        trainer = find_owned_trainer(binding)
        if trainer is not None:
            trainer_pid, trainer_identity = trainer
            try:
                resource_metrics = query_verified_resources(
                    trainer_pid, trainer_identity, binding.gpu_uuid
                )
            except BridgeError:
                resource_metrics = None
            if resource_metrics:
                resource_context = {
                    **_event_context(binding, run_identity, checkpoint_sha),
                    "pid": trainer_pid,
                    "pid_start_ticks": trainer_identity["start_ticks"],
                    "timeout_parent_pid": binding.parent_pid,
                    "gpu_uuid": binding.gpu_uuid,
                    "process_identity_verified": True,
                    "gpu_process_pid_uuid_verified": True,
                }
                event = _event(
                    "resources", step, epoch_index + 1, resource_metrics, resource_context
                )
                added += int(_append_event(view / "monitor_events.jsonl", event, known))
        evaluation_path = view / "eval_receipts" / f"epoch_{epoch_index + 1}.json"
        evaluation = None
        if evaluation_path.exists():
            evaluation = _read_json(evaluation_path, 1 << 20)
            if (
                evaluation.get("checkpoint_sha256") != checkpoint_sha
                or evaluation.get("run_identity_sha256") != run_identity
            ):
                raise BridgeError(
                    "saved frozen-evaluation receipt is bound to different checkpoint bytes"
                )
        elif evaluate_missing:
            evaluation = _run_evaluator(
                binding, run_identity, epoch_index, eval_batch_size, eval_timeout
            )
            if (
                evaluation.get("checkpoint_sha256") != checkpoint_sha
                or evaluation.get("run_identity_sha256") != run_identity
                or evaluation.get("completed_epoch") != epoch_index + 1
            ):
                raise BridgeError("frozen evaluator returned a mismatched checkpoint/run identity")
            if evaluation_path.parent.is_symlink():
                raise BridgeError("evaluation receipt directory cannot be a symlink")
            evaluation_path.parent.mkdir(mode=0o700, exist_ok=True)
            _atomic_json(evaluation_path, evaluation, exclusive=True)
        if evaluation is None:
            pending = True
            continue
        if (
            evaluation.get("schema") != "wnn.tl3-frozen-checkpoint-evaluation.v1"
            or evaluation.get("source_identity_file_sha256")
            != binding.source_identity_file_sha256
            or evaluation.get("actual_training_source_sha256")
            != binding.actual_training_source_sha256
            or evaluation.get("completed_epoch") != epoch_index + 1
            or evaluation.get("source_epoch_index") != epoch_index
        ):
            raise BridgeError("frozen evaluator receipt has a mismatched schema, source, or epoch")
        metrics: dict[str, int | float] = {}
        for source, target in (
            ("weighted_pearson", "valid/wp"),
            ("global_t0", "valid/t0"),
            ("global_t1", "valid/t1"),
        ):
            value = valid.get(source, valid.get(source.removeprefix("global_")))
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[target] = value
        valid_eval = evaluation.get("valid")
        train_eval = evaluation.get("train_eval")
        if not isinstance(valid_eval, dict) or not isinstance(train_eval, dict):
            raise BridgeError("frozen evaluator receipt omits valid/train_eval cohorts")
        expected_train_order = [int(group) for group in train_groups[:32]]
        if (
            train_eval.get("row_mask") != "all_after_warmup"
            or train_eval.get("cohort_count") != 32
            or train_eval.get("cohort_group_order_sha256")
            != _canonical_sha256(expected_train_order)
            or valid_eval.get("row_mask") != "is_scored"
            or valid_eval.get("cohort_count") != len(valid_groups)
            or valid_eval.get("cohort_group_order_sha256")
            != _canonical_sha256([int(group) for group in valid_groups])
        ):
            raise BridgeError("frozen evaluator cohort order or explicit row-mask policy differs")
        valid_metrics = valid_eval.get("metrics")
        if not isinstance(valid_metrics, dict):
            raise BridgeError("frozen evaluator valid metrics are malformed")
        train_metrics = train_eval.get("metrics")
        if not isinstance(train_metrics, dict):
            raise BridgeError("frozen evaluator train-eval metrics are malformed")
        for cohort_name, cohort_metrics in (
            ("valid", valid_metrics),
            ("train-eval", train_metrics),
        ):
            for metric_name in ("wp", "t0", "t1", "wmse"):
                metric = cohort_metrics.get(metric_name)
                if (
                    not isinstance(metric, (int, float))
                    or isinstance(metric, bool)
                    or not math.isfinite(float(metric))
                ):
                    raise BridgeError(
                        f"frozen evaluator omitted finite {cohort_name} {metric_name}"
                    )
        logged_wp = valid.get("weighted_pearson")
        evaluated_wp = valid_metrics.get("wp")
        if (
            not isinstance(logged_wp, (int, float))
            or not isinstance(evaluated_wp, (int, float))
            or not math.isfinite(float(logged_wp))
            or not math.isfinite(float(evaluated_wp))
            or not math.isclose(float(logged_wp), float(evaluated_wp), rel_tol=0.0, abs_tol=1e-5)
        ):
            raise BridgeError(
                "frozen evaluator pooled WP does not match the trainer's full-valid result"
            )
        for prefix, cohort in (("valid", valid_eval), ("train_eval", train_eval)):
            cohort_metrics = cohort.get("metrics")
            if not isinstance(cohort_metrics, dict):
                raise BridgeError(f"frozen evaluator {prefix} metrics are malformed")
            for name in ("wp", "t0", "t1", "wmse"):
                value = cohort_metrics.get(name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metrics[f"{prefix}/{name}"] = value
            metrics[f"{prefix}/score"] = cohort_metrics["wp"]
            metrics[f"{prefix}/score/t0"] = cohort_metrics["t0"]
            metrics[f"{prefix}/score/t1"] = cohort_metrics["t1"]
            metrics[f"{prefix}/loss"] = cohort_metrics["wmse"]
        fit_summary = valid.get("fit_summary", {})
        if isinstance(fit_summary, dict):
            objective = fit_summary.get("objective_mean")
            if isinstance(objective, (int, float)) and math.isfinite(float(objective)):
                metrics["train/objective_window_mean"] = objective
            rates = fit_summary.get("learning_rate_by_group", [])
            if isinstance(rates, list):
                for group in rates:
                    if isinstance(group, dict) and isinstance(group.get("group"), int):
                        value = group.get("mean")
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            metrics[f"optimizer/lr/group_{group['group']}/mean"] = value
        context = {
            **_event_context(binding, run_identity, checkpoint_sha),
            "source_epoch_index": epoch_index,
            "completed_epoch": epoch_index + 1,
            "train_eval": {
                "row_mask": "all_after_warmup",
                "cohort_count": len(train_groups[:32]),
                "cohort_group_order_sha256": _canonical_sha256(train_groups[:32]),
            },
            "valid": {
                "row_mask": "is_scored",
                "cohort_count": len(valid_groups),
                "official_split": "valid",
            },
            "training_objective": "mean full mixed per-window scalar; not WMSE",
        }
        event = _event("evaluation", step, epoch_index + 1, metrics, context)
        before = known.get(("evaluation", epoch_index + 1, step))
        if before is None:
            _append_evaluation(view, event, known)
            added += 1
        elif before.get("metrics") != metrics or before.get("context") != context:
            raise BridgeError("evaluation event conflicts with existing completed epoch")
        progress_total = progress_epochs.get(epoch_index)
        status_metrics = {"run/completed_epoch": epoch_index + 1}
        if progress_total:
            status_metrics["run/epoch_progress"] = 1.0
        status_event = _event("status", step, epoch_index + 1, status_metrics, context)
        added += int(_append_event(view / "monitor_events.jsonl", status_event, known))
    selected = log_records["selected_fixed"]
    requested_epochs = args.get("epochs")
    expected_epochs = set(range(requested_epochs)) if isinstance(requested_epochs, int) else set()
    complete_epochs = set(log_records["valid"])
    terminal = (
        not is_timeout_parent_live(binding)
        and not has_exact_trainer_process(binding)
        and bool(expected_epochs)
        and complete_epochs == expected_epochs
        and selected == args.get("checkpoint_epoch")
    )
    if (
        not terminal
        and isinstance(requested_epochs, int)
        and len(complete_epochs) < requested_epochs
    ):
        pending = True
    _set_evaluation_status(view, binding, run_identity, pending, known)
    if terminal:
        status_event = _event(
            "status",
            requested_epochs + 1,
            requested_epochs,
            {"run/completed": 1},
            {
                **_event_context(binding, run_identity),
                "terminal": "selected_fixed_checkpoint",
                "selected_epoch": selected,
            },
        )
        added += int(_append_event(view / "monitor_events.jsonl", status_event, known))
    status = (
        "completed"
        if terminal and not pending
        else "fit_complete_pending_evaluation"
        if terminal
        else "running"
    )
    return {
        "run": binding.run,
        "status": status,
        "events_added": added,
        "completed_epochs": len(log_records["valid"]),
    }


def _parser() -> argparse.ArgumentParser:
    """Build bounded bridge CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Bridge one receipt-owned legacy TL3 run to the local dashboard."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--training-source-identity", type=Path, required=True)
    parser.add_argument("--source-identity-file", type=Path, required=True)
    parser.add_argument("--dashboard-runs-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=22_200)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--once", action="store_true", help="perform one import pass and exit")
    return parser


def main() -> None:
    """Poll one frozen legacy run without touching its output or process."""
    args = _parser().parse_args()
    if not 15 <= args.poll_seconds <= 300 or not 60 <= args.timeout_seconds <= 7 * 24 * 3600:
        raise SystemExit("poll and timeout bounds are invalid")
    if not 1 <= args.eval_batch_size <= 16 or not 60 <= args.eval_timeout_seconds <= 3600:
        raise SystemExit("evaluation worker bounds are invalid")
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        result = bridge_once(
            args.receipt,
            args.dashboard_runs_root,
            adoption_path=args.adoption,
            training_source_identity_path=args.training_source_identity,
            source_identity_file_path=args.source_identity_file,
            evaluate_missing=not args.skip_evaluation,
            eval_batch_size=args.eval_batch_size,
            eval_timeout=args.eval_timeout_seconds,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        if args.once or result["status"] == "completed":
            return
        if time.monotonic() >= deadline:
            raise SystemExit("TL3 dashboard bridge reached its bounded timeout")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
