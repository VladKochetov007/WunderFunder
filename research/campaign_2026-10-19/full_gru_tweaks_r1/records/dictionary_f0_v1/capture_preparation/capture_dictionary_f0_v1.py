"""Capture bounded, read-only evidence for the exact dictionary-fit unit."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[3]
PREP = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records/dictionary_f0_v1"
CAPTURES = RECORDS / "captures"
ADMISSION = RECORDS / "ADMISSION.json"
LAUNCH = RECORDS / "LAUNCH.json"
SOURCE_MANIFEST = PREP / "DICTIONARY_SOURCE_MANIFEST.json"
UNIT = "computer-vision-homework-fg19-dictionary-f0-v1.service"
INVOCATION = "4e6d90ff7d4344df92d61d0d1c3f01fb"
SOURCE_COMMIT = "51a9b168584e86ca10ebdc0532465d6158ab19d3"
SOURCE_MANIFEST_SHA = "0d12d3b24839d6a31cdc957dbd20464335b2ad0a7a87b8396c6eb5be7e281a9f"
CONFIG_SHA = "b6d21725a4609ede93a75c5be8420afc51833be2be23e2748ee1a80dc5b94b6c"

REMOTE = r"""
import base64, hashlib, json, os, stat, subprocess
from pathlib import Path

P = json.loads(base64.b64decode("__PAYLOAD_B64__", validate=True))
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
UNIT = "computer-vision-homework-fg19-dictionary-f0-v1.service"
SOURCE = B / "source-fg19-gru-51a9b168"
LOGS = B / "research_tmp/fg19_dictionary_f0_v1"
RUN = B / "data/artifacts/research/fg19_dictionary_f0_v1"
CACHE = RUN / "input_representation.pt"
RECEIPT = RUN / "RECEIPT.json"

def raw_file(path, limit):
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            return {"exists": True, "error": "not_regular_file", "stat": {"bytes": info.st_size, "inode": info.st_ino}}
        with path.open("rb") as stream:
            stream.seek(max(0, info.st_size - limit))
            tail = stream.read(limit)
        return {"exists": True, "bytes": info.st_size, "captured_bytes": len(tail), "truncated_prefix_bytes": max(0, info.st_size-len(tail)), "sha256_of_captured_bytes": hashlib.sha256(tail).hexdigest(), "data_b64": base64.b64encode(tail).decode("ascii")}
    except FileNotFoundError:
        return {"exists": False}
    except OSError as error:
        return {"exists": None, "error": repr(error)}

def command(argv, limit=2_000_000):
    try:
        result = subprocess.run(argv, capture_output=True, timeout=20)
        stdout = result.stdout[-limit:]
        stderr = result.stderr[-limit:]
        return {"argv": argv, "returncode": result.returncode, "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr), "stdout_truncated_prefix_bytes": max(0,len(result.stdout)-len(stdout)), "stderr_truncated_prefix_bytes": max(0,len(result.stderr)-len(stderr)), "stdout_b64": base64.b64encode(stdout).decode("ascii"), "stderr_b64": base64.b64encode(stderr).decode("ascii")}
    except subprocess.TimeoutExpired as error:
        out = error.stdout or b""
        err = error.stderr or b""
        return {"argv": argv, "returncode": None, "timed_out": True, "stdout_b64": base64.b64encode(out[-limit:]).decode("ascii"), "stderr_b64": base64.b64encode(err[-limit:]).decode("ascii")}

def source_check():
    expected = P["source_files"]
    errors = []
    observed = {}
    actual = set()
    if SOURCE.is_symlink() or not SOURCE.is_dir():
        return {"status": "FAIL", "error": "source_root_missing_or_symlink", "expected_count": len(expected)}
    for path in SOURCE.rglob("*"):
        if path.is_symlink():
            errors.append({"path": str(path.relative_to(SOURCE)), "error": "symlink"})
        elif path.is_file():
            actual.add(path.relative_to(SOURCE).as_posix())
    if actual != set(expected):
        errors.append({"missing": sorted(set(expected)-actual), "extra": sorted(actual-set(expected))})
    for relative, digest in expected.items():
        path = SOURCE / relative
        try:
            if path.is_symlink() or not path.is_file():
                errors.append({"path": relative, "error": "not_regular_file"})
                continue
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            observed[relative] = h.hexdigest()
            if observed[relative] != digest:
                errors.append({"path": relative, "error": "sha256_mismatch", "observed": observed[relative]})
        except OSError as error:
            errors.append({"path": relative, "error": repr(error)})
    return {"status": "PASS" if not errors else "FAIL", "commit": P["source_commit"], "expected_count": len(expected), "actual_count": len(actual), "observed_sha256": observed, "errors": errors}

def parent_snapshot(unit_props):
    keys = ["ControlGroup", "CPUQuotaPerSecUSec", "MemoryCurrent", "MemoryHigh", "MemoryMax", "MemorySwapMax"]
    argv = ["systemctl", "--user", "show", "--all", "wnn-campaign.slice"]
    argv.extend(value for key in keys for value in ("-p", key))
    result = command(argv)
    text = base64.b64decode(result.get("stdout_b64", "")).decode(errors="replace")
    parent = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    root = Path("/sys/fs/cgroup").resolve()
    control = parent.get("ControlGroup", "")
    if not control or control == "/":
        return {"status": "FAIL", "error": "empty_or_root_controlgroup", "parent": parent, "show": result}
    group = root / control.lstrip("/")
    if not group.resolve().is_relative_to(root) or group.resolve() == root or not group.is_dir():
        return {"status": "FAIL", "error": "invalid_parent_cgroup_path", "parent": parent, "show": result}
    paths = [group / "cgroup.procs", *group.rglob("cgroup.procs")]
    if len(paths) > 128:
        return {"status": "FAIL", "error": "descendant_scan_bound_exceeded", "parent": parent, "show": result}
    populated = {}
    for path in paths:
        pids = path.read_text().split()
        if pids:
            populated[str(path)] = pids
    unit_group_text = unit_props.get("ControlGroup", "")
    unit_group = root / unit_group_text.lstrip("/") if unit_group_text else None
    unit_inside = bool(unit_group and unit_group.resolve().is_relative_to(group.resolve()) and unit_group.resolve() != group.resolve())
    return {"status": "PASS", "parent": parent, "show": result, "parent_cgroup": str(group), "unit_cgroup": unit_group_text, "unit_cgroup_inside_parent": unit_inside, "populated_descendants": populated}

def main():
    unit_keys = ["LoadState", "ActiveState", "SubState", "InvocationID", "ControlGroup", "MainPID", "ExecStart", "ExecMainCode", "ExecMainStatus", "Result", "MemoryCurrent", "MemoryPeak", "CPUUsageNSec", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax", "RuntimeMaxUSec"]
    argv = ["systemctl", "--user", "show", "--all", UNIT]
    argv.extend(value for key in unit_keys for value in ("-p", key))
    unit_show = command(argv)
    show_text = base64.b64decode(unit_show.get("stdout_b64", "")).decode(errors="replace")
    props = dict(line.split("=", 1) for line in show_text.splitlines() if "=" in line)
    if props.get("InvocationID") not in (None, "", P["invocation_id"]):
        unit_match = False
    else:
        unit_match = props.get("InvocationID") == P["invocation_id"]
    parent = parent_snapshot(props)
    source = source_check()
    receipt = raw_file(RECEIPT, 8_000_000)
    cache = {}
    try:
        info = CACHE.stat()
        link = CACHE.lstat()
        cache = {"exists": True, "stat_only": True, "symlink": CACHE.is_symlink(), "bytes": info.st_size, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns, "device": info.st_dev, "mode": stat.S_IMODE(info.st_mode), "lstat_mode": stat.S_IMODE(link.st_mode)}
    except FileNotFoundError:
        cache = {"exists": False, "stat_only": True}
    log = raw_file(LOGS / "unit.log", 2_000_000)
    admission = raw_file(LOGS / "ADMISSION.json", 4_000_000)
    launch = raw_file(LOGS / "LAUNCH.json", 1_000_000)
    status = command(["systemctl", "--user", "status", UNIT, "--no-pager", "--full"], 300_000)
    journal = command(["journalctl", "--user", "-u", UNIT, "--no-pager", "-o", "short-iso", "-n", "200"], 500_000)
    result = {"schema": "fg19_dictionary_readonly_capture.v1", "captured_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "expected_unit": UNIT, "expected_invocation_id": P["invocation_id"], "unit_invocation_matches": unit_match, "unit_show": unit_show, "unit_status": status, "journal_tail_200": journal, "remote_admission": admission, "remote_launch": launch, "native_log_tail": log, "artifact_receipt": receipt, "cache_stat_only": cache, "source_identity": source, "campaign_parent_and_descendants": parent}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))

main()
"""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _payload() -> dict[str, object]:
    admission_raw, launch_raw, manifest_raw = (
        ADMISSION.read_bytes(),
        LAUNCH.read_bytes(),
        SOURCE_MANIFEST.read_bytes(),
    )
    admission, launch, manifest = (
        json.loads(admission_raw),
        json.loads(launch_raw),
        json.loads(manifest_raw),
    )
    source_files = {row["path"]: row["sha256"] for row in manifest["files"]}
    stdout = launch.get("systemctl_after_submission", {}).get("stdout", "")
    if (
        _sha(manifest_raw) != SOURCE_MANIFEST_SHA
        or manifest.get("source_commit") != SOURCE_COMMIT
        or len(source_files) != 162
        or admission.get("unit") != UNIT
        or admission.get("source", {}).get("commit") != SOURCE_COMMIT
        or admission.get("source", {}).get("file_count") != 162
        or admission.get("source", {}).get("files_sha256") != source_files
        or admission.get("source_archive_sha256") != manifest.get("archive_sha256")
        or admission.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA
        or admission.get("preparation_sha256") != CONFIG_SHA
        or launch.get("unit") != UNIT
        or launch.get("submission", {}).get("returncode") != 0
        or f"InvocationID={INVOCATION}" not in stdout
        or f"invocation ID: {INVOCATION}" not in launch.get("submission", {}).get("stderr", "")
    ):
        raise ValueError("local admission, launch, manifest, or exact invocation binding changed")
    return {
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_files": source_files,
        "expected_admission_sha256": _sha(admission_raw),
        "expected_launch_sha256": _sha(launch_raw),
        "expected_profile_result_sha256": admission["profile_result_sha256"],
        "expected_path_diagnostic_sha256": admission["path_diagnostic_sha256"],
    }


def _save_remote_files(result: dict[str, object], destination: Path) -> None:
    fields = {
        "remote_admission": "REMOTE_ADMISSION.raw.json",
        "remote_launch": "REMOTE_LAUNCH.raw.json",
        "artifact_receipt": "ARTIFACT_RECEIPT.raw.json",
        "native_log_tail": "UNIT_LOG.tail.raw",
    }
    for field, filename in fields.items():
        record = result.get(field)
        if not isinstance(record, dict) or not record.get("exists"):
            continue
        encoded = record.get("data_b64")
        if isinstance(encoded, str):
            (destination / filename).write_bytes(base64.b64decode(encoded, validate=True))
    for field in ("unit_show", "unit_status", "journal_tail_200"):
        record = result.get(field)
        if isinstance(record, dict):
            name = {
                "unit_show": "UNIT_SHOW",
                "unit_status": "UNIT_STATUS",
                "journal_tail_200": "JOURNAL_TAIL_200",
            }[field]
            (destination / f"{name}.stdout.raw").write_bytes(
                base64.b64decode(record.get("stdout_b64", ""), validate=True)
            )
            (destination / f"{name}.stderr.raw").write_bytes(
                base64.b64decode(record.get("stderr_b64", ""), validate=True)
            )


def _metadata(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _metadata(item)
            for key, item in value.items()
            if key not in {"data_b64", "stdout_b64", "stderr_b64"}
        }
    if isinstance(value, list):
        return [_metadata(item) for item in value]
    return value


def main() -> None:
    payload = _payload()
    template = REMOTE.replace(
        "__PAYLOAD_B64__",
        base64.b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode(),
    )
    compile(template, "dictionary_capture_remote.py", "exec")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if CAPTURES.exists() or CAPTURES.is_symlink():
        if CAPTURES.is_symlink() or not CAPTURES.is_dir():
            raise FileExistsError(f"capture root is not a plain directory: {CAPTURES}")
    else:
        CAPTURES.mkdir(mode=0o700, parents=False, exist_ok=False)
    destination = CAPTURES / f"capture_{stamp}"
    destination.mkdir(mode=0o700, exist_ok=False)
    (destination / "BINDING.json").write_text(
        json.dumps(
            {
                **payload,
                "remote_code_sha256": _sha(template.encode()),
                "local_helper_sha256": _sha(Path(__file__).read_bytes()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (destination / "REMOTE_CAPTURE.py").write_text(template)
    command = [
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
    ]
    try:
        response = subprocess.run(
            command, input=template.encode(), capture_output=True, timeout=120
        )
        (destination / "SSH_STDOUT.raw").write_bytes(response.stdout)
        (destination / "SSH_STDERR.raw").write_bytes(response.stderr)
        (destination / "TRANSPORT.json").write_text(
            json.dumps({"returncode": response.returncode, "timed_out": False}, indent=2) + "\n"
        )
        try:
            result = json.loads(response.stdout)
        except json.JSONDecodeError:
            result = None
        if isinstance(result, dict):
            _save_remote_files(result, destination)
            (destination / "CAPTURE.json").write_text(
                json.dumps(_metadata(result), indent=2, sort_keys=True) + "\n"
            )
    except subprocess.TimeoutExpired as error:
        (destination / "SSH_STDOUT.partial.raw").write_bytes(error.stdout or b"")
        (destination / "SSH_STDERR.partial.raw").write_bytes(error.stderr or b"")
        (destination / "TRANSPORT.json").write_text(
            json.dumps(
                {
                    "returncode": None,
                    "timed_out": True,
                    "policy": "read-only capture may be rerun into a different create-only capture folder",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    print(
        json.dumps(
            {
                "capture_directory": str(destination),
                "status": "captured_read_only",
                "ssh_returncode": response.returncode,
            }
        )
    )


if __name__ == "__main__":
    main()
