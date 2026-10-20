"""One bounded, read-only startup snapshot for the exact seed-5 teacher unit."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
UNIT = "computer-vision-homework-fg19-teacher-s5-v1.service"
INVOCATION = "4317408cdc69442c8438c74aaf5809eb"
GPU = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
SOURCE = B / "source-fg19-teacher-af21d6d8"
RUN = B / "data/artifacts/runs/fg19_t3_h384x2_e3_s5_v1"
LOG = B / "research_tmp/fg19_teacher_s5_v1/unit.log"
MAX_CGROUPS = 256
MAX_LOG_BYTES = 262_144


def command(args: list[str], timeout: int = 10) -> dict[str, object]:
    """Capture one bounded read-only command without hiding failures."""
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {
            "argv": args,
            "timeout_seconds": timeout,
            "timed_out": True,
            "stdout": _decode(error.stdout),
            "stderr": _decode(error.stderr),
        }
    return {
        "argv": args,
        "returncode": result.returncode,
        "stdout": result.stdout.decode(errors="replace"),
        "stderr": result.stderr.decode(errors="replace"),
    }


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def show_unit(unit: str, fields: list[str]) -> dict[str, str]:
    result = command(
        ["systemctl", "--user", "show", "--all", unit, *sum((["-p", key] for key in fields), [])]
    )
    if result.get("returncode") != 0:
        return {"_error": json.dumps(result, sort_keys=True)}
    return dict(line.split("=", 1) for line in str(result["stdout"]).splitlines() if "=" in line)


def cgroup_files(root: Path) -> tuple[list[Path], bool]:
    """Enumerate cgroup.procs files with a hard directory-count bound."""
    found: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        if not directory.is_dir():
            continue
        found.append(directory / "cgroup.procs")
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
        except OSError:
            continue
        if len(found) > MAX_CGROUPS:
            return found[:MAX_CGROUPS], True
    return found, False


def process_identity(pid: str) -> dict[str, object]:
    directory = Path("/proc") / pid
    try:
        owner = (directory / "cgroup").read_text().strip()
        uid = directory.stat().st_uid
        commandline = (
            (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        )
    except FileNotFoundError:
        return {"pid": pid, "exited_during_observation": True}
    return {"pid": pid, "uid": uid, "cgroup": owner, "cmdline": commandline[:1000]}


def read_events(group: Path) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in ("memory.events", "memory.events.local"):
        try:
            values[name] = (group / name).read_text()
        except OSError as error:
            values[name] = f"unavailable: {type(error).__name__}: {error}"
    return values


def file_stat(path: Path) -> dict[str, object]:
    try:
        value = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": path.is_file() or path.is_dir(),
        "is_symlink": path.is_symlink(),
        "mode": oct(value.st_mode),
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "device": value.st_dev,
        "inode": value.st_ino,
    }


def bounded_provenance(path: Path) -> dict[str, object]:
    """Read only the small JSON provenance, summarize memberships, never load tensors."""
    metadata = file_stat(path)
    if not metadata.get("exists") or metadata.get("bytes", MAX_LOG_BYTES + 1) > MAX_LOG_BYTES:
        return {"stat": metadata, "summary": None, "reason": "missing or over bounded JSON size"}
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        return {"stat": metadata, "summary": None, "reason": f"read/parse failure: {error}"}
    args = document.get("args", {})
    names = (
        "name",
        "seed",
        "device",
        "arch",
        "cell",
        "hidden",
        "layers",
        "batch",
        "chunk",
        "epochs",
        "optim",
        "muon_scope",
        "precision",
        "checkpoint_policy",
        "checkpoint_epoch",
        "valid_fold",
        "train_source",
        "include_heldout_valid",
        "no_eval_during_fit",
    )
    fit_groups = document.get("fit_row_groups", {})
    diagnostics = document.get("update_diagnostics", [])
    latest = diagnostics[-1] if diagnostics else None
    return {
        "stat": metadata,
        "summary": {
            "seed": document.get("seed"),
            "args": {name: args.get(name) for name in names if name in args},
            "split_manifest_sha256": document.get("split_manifest_sha256"),
            "fit_group_counts": {
                name: len(ids) for name, ids in fit_groups.items() if isinstance(ids, list)
            },
            "evaluation_group_count": len(document.get("evaluation_row_groups", [])),
            "propensity_checkpoint_sha256": document.get("propensity_checkpoint_sha256"),
            "completed_epoch_diagnostics": [
                {
                    "epoch_index": entry.get("epoch_index"),
                    "epoch": entry.get("epoch"),
                    "cumulative": entry.get("cumulative"),
                }
                for entry in diagnostics
                if isinstance(entry, dict)
            ],
            "latest_diagnostic": latest,
        },
        "tensor_load_or_hash": False,
    }


def log_tail(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_BYTES))
            data = stream.read(MAX_LOG_BYTES)
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "lines": [], "fit_summaries": []}
    lines = data.decode(errors="replace").splitlines()
    fit_summaries: list[dict[str, object]] = []
    for line in lines:
        if not line.startswith("FIT "):
            continue
        try:
            summary = json.loads(line[4:])
        except json.JSONDecodeError:
            continue
        update = summary.get("update_diagnostics", {})
        fit_summaries.append(
            {
                "epoch": summary.get("epoch"),
                "evaluation": summary.get("evaluation"),
                "train_seconds": summary.get("train_seconds"),
                "epoch_updates": update.get("epoch"),
                "cumulative_updates": update.get("cumulative"),
            }
        )
    return {
        "path": str(path),
        "exists": True,
        "bytes": size,
        "tail_byte_cap": MAX_LOG_BYTES,
        "truncated_at_start": size > len(data),
        "line_count": len(lines),
        "fit_summaries": fit_summaries[-6:],
        "tail_lines": lines[-80:],
    }


UNIT_FIELDS = [
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "InvocationID",
    "MainPID",
    "ControlGroup",
    "ExecMainCode",
    "ExecMainStatus",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ExecStart",
    "CPUQuotaPerSecUSec",
    "MemoryCurrent",
    "MemoryPeak",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
]
PARENT_FIELDS = [
    "LoadState",
    "ActiveState",
    "SubState",
    "ControlGroup",
    "CPUQuotaPerSecUSec",
    "MemoryCurrent",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
]


def main() -> None:
    unit = show_unit(UNIT, UNIT_FIELDS)
    parent = show_unit("wnn-campaign.slice", PARENT_FIELDS)
    parent_group = Path("/sys/fs/cgroup") / parent.get("ControlGroup", "").lstrip("/")
    if not parent.get("ControlGroup") or parent.get("ControlGroup") == "/":
        raise RuntimeError("campaign slice returned no explicit cgroup path")
    paths, overflow = cgroup_files(parent_group)
    populated: dict[str, list[str]] = {}
    for proc_file in paths:
        try:
            pids = proc_file.read_text().split()
        except OSError:
            continue
        if pids:
            populated[str(proc_file.parent)] = pids
    all_pids = {pid for values in populated.values() for pid in values}
    own_group_text = unit.get("ControlGroup", "")
    own_group = Path("/sys/fs/cgroup") / own_group_text.lstrip("/") if own_group_text else None

    gpu_status = command(
        [
            "nvidia-smi",
            "-i",
            GPU,
            "--query-gpu=uuid,name,memory.used,memory.free,memory.total",
            "--format=csv,noheader",
        ]
    )
    gpu_apps = command(
        [
            "nvidia-smi",
            "-i",
            GPU,
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ]
    )
    gpu_processes: list[dict[str, object]] = []
    for row in str(gpu_apps.get("stdout", "")).splitlines():
        if row.strip().startswith("No running processes found") or not row.strip():
            continue
        columns = [value.strip() for value in row.split(",")]
        if len(columns) != 3:
            gpu_processes.append({"raw": row, "parse_error": True})
            continue
        identity = process_identity(columns[0])
        identity.update(
            {
                "gpu_uuid": columns[1],
                "used_memory": columns[2],
                "campaign_descendant": columns[0] in all_pids,
            }
        )
        gpu_processes.append(identity)

    run_files = {
        name: file_stat(RUN / name)
        for name in (
            "provenance.json",
            "training_state.pt",
            "model.pt",
            "model_ep2.pt",
            "history.json",
            "progress.jsonl",
        )
    }
    source_stats = {
        "source_root": file_stat(SOURCE),
        "train_rnn.py": file_stat(SOURCE / "scripts/train_rnn.py"),
        "provenance.py": file_stat(SOURCE / "src/wnn/training/provenance.py"),
        "source_commit": "af21d6d8409c46b36830f0bdddd50e67ee17025f",
    }
    snapshot = {
        "schema": "full_gru_tweaks.teacher_s3_startup_health.v1",
        "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
        "expected": {
            "unit": UNIT,
            "invocation_id": INVOCATION,
            "gpu_uuid": GPU,
            "source_commit": source_stats["source_commit"],
        },
        "unit": unit,
        "exact_invocation_match": unit.get("InvocationID") == INVOCATION,
        "parent": parent,
        "parent_cgroup_scan": {
            "root": str(parent_group),
            "cgroup_count": len(paths),
            "overflow": overflow,
            "populated": populated,
        },
        "parent_memory_events": read_events(parent_group),
        "own_cgroup": None
        if own_group is None
        else {
            "path": own_group_text,
            "populated_pids": (own_group / "cgroup.procs").read_text().split()
            if (own_group / "cgroup.procs").exists()
            else [],
            "memory_current": (own_group / "memory.current").read_text().strip()
            if (own_group / "memory.current").exists()
            else None,
            "memory_peak": (own_group / "memory.peak").read_text().strip()
            if (own_group / "memory.peak").exists()
            else None,
            "events": read_events(own_group),
        },
        "campaign_processes": [process_identity(pid) for pid in sorted(all_pids, key=int)],
        "gpu": {"status": gpu_status, "compute_apps": gpu_apps, "processes": gpu_processes},
        "source": source_stats,
        "run": {
            "path": str(RUN),
            "directory_stat": file_stat(RUN),
            "files": run_files,
            "provenance": bounded_provenance(RUN / "provenance.json"),
            "log_tail": log_tail(LOG),
            "training_state_payload_read_or_hashed": False,
            "model_payload_read_or_hashed": False,
        },
        "scope": "one read-only startup snapshot; no model/checkpoint load or hash, no data array read, no unit mutation",
    }
    print(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
