from __future__ import annotations

import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def require(condition: bool, message: str) -> None:
    """Fail closed before the first support-output mutation."""
    if not condition:
        raise RuntimeError(message)


def digest(raw: bytes) -> str:
    """Return a SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse a bounded JSON object while retaining its original bytes."""
    value = json.loads(raw)
    require(isinstance(value, dict), f"JSON object expected: {name}")
    return value


def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
    """Read one bounded regular file through a no-follow descriptor and verify stability."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"unsafe file: {path}")
    require(before.st_size <= cap, f"file exceeds cap: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            f"file changed before read: {path}",
        )
        raw = bytearray()
        while len(raw) <= cap:
            block = os.read(descriptor, min(65_536, cap + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(len(raw) == before.st_size and len(raw) <= cap, f"file changed or exceeded cap: {path}")
    require(
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        f"file changed while read: {path}",
    )
    return bytes(raw), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def file_hash(path: Path, cap: int) -> tuple[str, dict[str, int]]:
    """Stream a source file hash while checking its stable inode and timestamp."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"unsafe source file: {path}")
    require(before.st_size <= cap, f"source file exceeds cap: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    hasher = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            f"source changed before hash: {path}",
        )
        bytes_read = 0
        while True:
            block = os.read(descriptor, min(1 << 20, cap - bytes_read + 1))
            if not block:
                break
            bytes_read += len(block)
            require(bytes_read <= cap, f"source file exceeded cap while hashing: {path}")
            hasher.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(bytes_read == after.st_size, f"source bytes read differ from stable file size: {path}")
    require(
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        f"source changed while hashing: {path}",
    )
    return hasher.hexdigest(), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _bounded_gpu_query(args: list[str]) -> tuple[str, str]:
    """Run one bounded nvidia-smi query without shell interpretation."""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(
        process.stdout is not None and process.stderr is not None, "GPU query pipes unavailable"
    )
    output = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + 15
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                require(remaining > 0, "selected-GPU query timed out")
                events = selector.select(remaining)
                require(events, "selected-GPU query timed out")
                for key, _ in events:
                    label = streams[key.fileobj]
                    chunk = os.read(key.fileobj.fileno(), min(4096, 65537 - len(output[label])))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output[label].extend(chunk)
                    require(
                        len(output[label]) <= 65536, "selected-GPU query output exceeded 64 KiB"
                    )
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        require(
            returncode == 0,
            "selected-GPU query returned nonzero: "
            + output["stderr"].decode("utf-8", errors="replace")[:512],
        )
    except (OSError, subprocess.TimeoutExpired, RuntimeError):
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    return output["stdout"].decode("utf-8", errors="strict"), output["stderr"].decode(
        "utf-8", errors="replace"
    )


def _memory_bytes(value: str) -> int:
    """Parse nvidia-smi's integer MiB field without accepting unknown values."""
    fields = value.strip().split()
    require(
        len(fields) == 2 and fields[0].isdigit() and fields[1] == "MiB",
        "selected-GPU memory field is malformed",
    )
    return int(fields[0]) * 1024 * 1024


def _bounded_proc_read(path: Path, maximum: int) -> bytes:
    """Read a proc metadata file with no-follow and an explicit byte ceiling."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, f"proc metadata exceeded {maximum} bytes: {path.name}")
    return raw


def _proc_starttime(proc: Path, pid: str) -> str:
    """Return Linux /proc stat starttime, distinguishing PID reuse."""
    text = _bounded_proc_read(proc / "stat", 8192).decode("ascii", errors="strict")
    prefix, separator, remainder = text.rpartition(")")
    require(separator == ")" and prefix.startswith(pid + " ("), "proc stat identity is malformed")
    fields = remainder.strip().split()
    require(len(fields) > 19 and fields[19].isdigit(), "proc stat starttime is unavailable")
    return fields[19]


def show(unit: str, fields: tuple[str, ...], timeout: int = 15) -> dict[str, str]:
    """Read named systemd properties for one exact unit."""
    command = ["systemctl", "--user", "show", "--all", unit]
    for field in fields:
        command.extend(("-p", field))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    require(result.returncode == 0, f"systemctl show failed for {unit}")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def write_exclusive(path: Path, raw: bytes) -> None:
    """Create and fsync one private file without replacing existing output."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    """Create one compact JSON receipt with exclusive creation."""
    write_exclusive(
        path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )


import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

# Conditional one-shot admission for the fixed first-cell score consumer.

# The renderer prepends this exact reviewed stdlib helper source. These type-only
# bindings keep the unrendered template checkable and never replace runtime helpers.
if TYPE_CHECKING:
    require = cast(Callable[[bool, str], None], None)
    digest = cast(Callable[[bytes], str], None)
    json_object = cast(Callable[[bytes, str], dict[str, object]], None)
    stable_file = cast(Callable[[Path, int], tuple[bytes, dict[str, int]]], None)
    file_hash = cast(Callable[[Path, int], tuple[str, dict[str, int]]], None)
    _bounded_gpu_query = cast(Callable[[list[str]], tuple[str, str]], None)
    _memory_bytes = cast(Callable[[str], int], None)
    _bounded_proc_read = cast(Callable[[Path, int], bytes], None)
    _proc_starttime = cast(Callable[[Path, str], str], None)
    show = cast(Callable[[str, tuple[str, ...], int], dict[str, str]], None)
    write_exclusive = cast(Callable[[Path, bytes], None], None)
    write_json_exclusive = cast(Callable[[Path, dict[str, object]], None], None)


PROJECT_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA_ROOT = PROJECT_BASE / "data"
SOURCE_ROOT = PROJECT_BASE / "source-fg19-cache-68ff6829"
SPLIT_PATH = PROJECT_BASE / "splits/fold0.json"
SCRATCH_ROOT = PROJECT_BASE / "research_tmp"
FIT_RECEIPTS_ROOT = SCRATCH_ROOT / "fg19_child_allthree_fit_v1"
PROFILE_ROOT = SCRATCH_ROOT / "fg19_child_allthree_profile_v1"
CACHE_ROOT = SCRATCH_ROOT / "fg19_t3_fit_homogeneous_cache_v1"
CACHE_RECEIPTS_ROOT = PROJECT_BASE / "research/full_gru_tweaks_r1/cache_cpu_v2"
EXPORT_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_evaluation_v3"
EXPORT_ROOT = EXPORT_D / "exports"
SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v2"
SCORE_ACCEPTED_ROOT = SCORE_D / "accepted_export"
RESULT_PATH = EXPORT_D / "RESULT.json"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
PORTABLE_REGISTRY = PROJECT_BASE / "support-source/terminal-v1/PORTABLE_SOURCES.json"
UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"
EXPORT_UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_SET_SHA256 = "9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8"
EXPORT_BINDING_BYTES = 15_796
EXPORT_BINDING_SHA256 = "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd"
SOURCE_MANIFEST_BYTES = 27_569
SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SOURCE_REVIEW_BYTES = 8_712
SOURCE_REVIEW_SHA256 = "64c93d8279a2daeb15f2750e6366a0fe94a268c7b8e9a04dbdb28e4a36a56e73"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
PORTABLE_REGISTRY_BYTES = 2_157
PORTABLE_REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
FIT_ADMISSION_BYTES = 12_912
FIT_ADMISSION_SHA256 = "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"
FIT_LAUNCH_BYTES = 9_594
FIT_LAUNCH_SHA256 = "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538"
PARENT_CHECKPOINT_SHA256 = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
DICTIONARY_SHA256 = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
CACHE_METADATA_SHA256 = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA256 = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_STOP_SHA256 = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
PROFILE_PINS = {
    "ADMISSION.json": "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
    "LAUNCH.json": "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
    "unit.log": "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
}
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
    "PARENT_EXACT_STOP.json": "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
    "DICTIONARY_ACCEPTANCE.json": "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
    "DICTIONARY_EXACT_STOP.json": "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
    "PROFILE_TERMINAL_ACCEPTANCE.json": "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
    "PROFILE_EXACT_STOP.json": "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    "CONTROL_TERMINAL_ACCEPTANCE.json": "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    "CONTROL_EXACT_STOP.json": "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
    "EMA_TERMINAL_ACCEPTANCE.json": "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8",
    "EMA_EXACT_STOP.json": "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf",
    "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1",
    "DICTIONARY_CHILD_EXACT_STOP.json": "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3",
    "NLL_CHILD_TERMINAL_ACCEPTANCE.json": "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5",
    "NLL_CHILD_EXACT_STOP.json": "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1",
}
SCORER_FILES = {
    "first_cell_score_contract.py": (
        20_165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
    ),
    "first_cell_score_core.py": (
        26_965,
        "961f566eaeec0cd7e9ab48788666d03da9b0482810a11f16a019b083cfc65005",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}
ARMS = (
    ("control", "base"),
    ("ema", "ema"),
    ("sparse", "sparse"),
    ("nll", "base"),
    ("allthree", "ema_sparse"),
)
PREDICTION_SHAPE = [468, 20_000, 2]
PREDICTION_DATA_BYTES = 74_880_000
MAX_PREDICTION_BYTES = PREDICTION_DATA_BYTES + 4_096
MAX_METADATA_BYTES = 16_000_000
MAX_SOURCE_BYTES = 128_000_000
MAX_SOURCE_AGGREGATE_BYTES = 10_000_000
WORKER_CPU_QUOTA = "50%"
WORKER_MEMORY_HIGH_BYTES = 1_400_000_000
WORKER_MEMORY_MAX_BYTES = 1_500_000_000
WORKER_RUNTIME_SECONDS = 600
PARENT_MEMORY_TARGET_BYTES = 17_000_000_000
PARENT_MEMORY_RESERVE_BYTES = 1_000_000_000
MINIMUM_FILESYSTEM_FREE_BYTES = 25 * 1024**3
EXPECTED_FILESYSTEM_DEVICE = 64_512
UNBOUND = "UNBOUND_ACTUAL_EXPORT_ACCEPTANCE_OR_STOP"
EXPORT_ACCEPTANCE_BYTES = 8833
EXPORT_ACCEPTANCE_SHA256 = '5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d'
EXPORT_EXACT_STOP_BYTES = 1140
EXPORT_EXACT_STOP_SHA256 = '695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f'
PAYLOAD_B64: dict[str, str] = {
    'first_cell_score_contract.py': 'IiIiVmFsaWRhdGUgdGhlIGZpeGVkIGZpcnN0LWNlbGwgZXhwb3J0IGJ1bmRsZSBhbmQgaXRzIHByZWRpY3Rpb24gYXJyYXlzLiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55Cgpmcm9tIGZpcnN0X2NlbGxfc2NvcmVfY29yZSBpbXBvcnQgKAogICAgQVJNX01PREVTLAogICAgRVhQRUNURURfQklORElOR19CWVRFUywKICAgIEVYUEVDVEVEX0JJTkRJTkdfU0hBMjU2LAogICAgRVhQRUNURURfREFUQV9BRE1JU1NJT04sCiAgICBFWFBFQ1RFRF9QQVJFTlRfUkVTT0xVVElPTlMsCiAgICBFWFBFQ1RFRF9QT1JUQUJMRV9NQU5JRkVTVCwKICAgIEVYUEVDVEVEX1BPUlRBQkxFX1NPVVJDRV9OQU1FUywKICAgIEVYUEVDVEVEX1JFR0lTVEVSRURfU09VUkNFUywKICAgIEVYUEVDVEVEX1NPVVJDRV9NQU5JRkVTVF9CWVRFUywKICAgIEVYUEVDVEVEX1NPVVJDRV9NQU5JRkVTVF9TSEEyNTYsCiAgICBFWFBFQ1RFRF9TT1VSQ0VfU0VUX1NIQTI1NiwKICAgIEVYUEVDVEVEX1NQTElUX1NIQTI1NiwKICAgIEVYUEVDVEVEX1ZBTElEX1BBUlFVRVQsCiAgICBNQVhfRVhQT1JUX01BTklGRVNUX0JZVEVTLAogICAgTUFYX1NNQUxMX0pTT05fQllURVMsCiAgICBOX0dST1VQUywKICAgIE5fT1VUUFVUUywKICAgIFBSRURJQ1RJT05fREFUQV9CWVRFUywKICAgIFBSRURJQ1RJT05fU0hBUEUsCiAgICBTT1VSQ0VfQ09NTUlULAogICAgU09VUkNFX1BJTlMsCiAgICBfcmVhZF9qc29uLAogICAgX3NhZmVfY2hpbGQsCikKCgpkZWYgX3ZhbGlkYXRlX2RhdGFfaWRlbnRpdHkoYmluZGluZzogZGljdFtzdHIsIEFueV0sIG1hbmlmZXN0OiBkaWN0W3N0ciwgQW55XSkgLT4gTm9uZToKICAgICIiIkJpbmQgcHJvZHVjZXIgbWV0YWRhdGEgdG8gdGhlIHNpeCBhZG1pdHRlZCBzb3VyY2VzIGFuZCBwb3J0YWJsZSByZWdpc3RyeS4iIiIKICAgIHJlZ2lzdGVyZWQgPSBiaW5kaW5nLmdldCgicmVnaXN0ZXJlZF9kYXRhIikKICAgIGlmIG5vdCBpc2luc3RhbmNlKHJlZ2lzdGVyZWQsIGRpY3QpIG9yIHNldChyZWdpc3RlcmVkKSAhPSB7CiAgICAgICAgImFkbWlzc2lvbiIsCiAgICAgICAgInBvcnRhYmxlX21hbmlmZXN0IiwKICAgICAgICAic291cmNlX2lkZW50aXRpZXMiLAogICAgICAgICJ2YWxpZF9wYXJxdWV0IiwKICAgIH06CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IGJpbmRpbmcgcmVnaXN0ZXJlZC1kYXRhIHNlY3Rpb24gaXMgbWFsZm9ybWVkIikKICAgIGFkbWlzc2lvbiA9IHJlZ2lzdGVyZWQuZ2V0KCJhZG1pc3Npb24iKQogICAgdmFsaWQgPSByZWdpc3RlcmVkLmdldCgidmFsaWRfcGFycXVldCIpCiAgICBzb3VyY2VfaWRlbnRpdGllcyA9IHJlZ2lzdGVyZWQuZ2V0KCJzb3VyY2VfaWRlbnRpdGllcyIpCiAgICBwb3J0YWJsZSA9IHJlZ2lzdGVyZWQuZ2V0KCJwb3J0YWJsZV9tYW5pZmVzdCIpCiAgICBpZiAoCiAgICAgICAgYWRtaXNzaW9uICE9IEVYUEVDVEVEX0RBVEFfQURNSVNTSU9OCiAgICAgICAgb3IgdmFsaWQgIT0gRVhQRUNURURfVkFMSURfUEFSUVVFVAogICAgICAgIG9yIHNvdXJjZV9pZGVudGl0aWVzICE9IEVYUEVDVEVEX1JFR0lTVEVSRURfU09VUkNFUwogICAgICAgIG9yIHBvcnRhYmxlICE9IEVYUEVDVEVEX1BPUlRBQkxFX01BTklGRVNUCiAgICApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImV4cG9ydCBiaW5kaW5nIHJlZ2lzdGVyZWQtZGF0YSBpZGVudGl0eSBkaWZmZXJzIGZyb20gYWRtaXNzaW9uIikKICAgIGlmIGFueSh0eXBlKHZhbGlkW2tleV0pIGlzIG5vdCB0eXBlKHZhbHVlKSBmb3Iga2V5LCB2YWx1ZSBpbiBFWFBFQ1RFRF9WQUxJRF9QQVJRVUVULml0ZW1zKCkpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoInJlZ2lzdGVyZWQgdmFsaWRhdGlvbi1kYXRhIGlkZW50aXR5IGhhcyB1bmV4cGVjdGVkIGZpZWxkIHR5cGVzIikKICAgIGlmIGFueSgKICAgICAgICB0eXBlKGFkbWlzc2lvbltrZXldKSBpcyBub3QgdHlwZSh2YWx1ZSkgZm9yIGtleSwgdmFsdWUgaW4gRVhQRUNURURfREFUQV9BRE1JU1NJT04uaXRlbXMoKQogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJyZWdpc3RlcmVkLWRhdGEgYWRtaXNzaW9uIHBpbiBoYXMgdW5leHBlY3RlZCBmaWVsZCB0eXBlcyIpCgogICAgaWYgc2V0KHNvdXJjZV9pZGVudGl0aWVzKSAhPSBzZXQoRVhQRUNURURfUE9SVEFCTEVfU09VUkNFX05BTUVTKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJyZWdpc3RlcmVkLWRhdGEgc291cmNlIG1lbWJlcnNoaXAgZGlmZmVycyBmcm9tIHRoZSBwb3J0YWJsZSByZWdpc3RyeSIpCiAgICBmb3IgcmVsYXRpdmUsIGVudHJ5IGluIHNvdXJjZV9pZGVudGl0aWVzLml0ZW1zKCk6CiAgICAgICAgZXhwZWN0ZWRfZW50cnkgPSBFWFBFQ1RFRF9SRUdJU1RFUkVEX1NPVVJDRVNbcmVsYXRpdmVdCiAgICAgICAgaWYgKAogICAgICAgICAgICBzZXQoZW50cnkpCiAgICAgICAgICAgICE9IHsKICAgICAgICAgICAgICAgICJieXRlcyIsCiAgICAgICAgICAgICAgICAiZGV2aWNlIiwKICAgICAgICAgICAgICAgICJpbm9kZSIsCiAgICAgICAgICAgICAgICAiaXNsaW5rIiwKICAgICAgICAgICAgICAgICJtdGltZV9ucyIsCiAgICAgICAgICAgICAgICAicGF0aCIsCiAgICAgICAgICAgICAgICAicmVnaXN0ZXJlZF9zaGEyNTYiLAogICAgICAgICAgICAgICAgInJlc29sdmVkIiwKICAgICAgICAgICAgfQogICAgICAgICAgICBvciBlbnRyeSAhPSBleHBlY3RlZF9lbnRyeQogICAgICAgICAgICBvciBhbnkodHlwZShlbnRyeVtrZXldKSBpcyBub3QgdHlwZSh2YWx1ZSkgZm9yIGtleSwgdmFsdWUgaW4gZXhwZWN0ZWRfZW50cnkuaXRlbXMoKSkKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKAogICAgICAgICAgICAgICAgZiJyZWdpc3RlcmVkIHNvdXJjZSBpZGVudGl0eSBoYXMgdW5leHBlY3RlZCBmaWVsZHMgb3IgdHlwZXM6IHtyZWxhdGl2ZX0iCiAgICAgICAgICAgICkKICAgICAgICBpZiBlbnRyeVsicGF0aCJdICE9IHN0cihQYXRoKGJpbmRpbmdbImRhdGFfcm9vdCJdKSAvIHJlbGF0aXZlKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmInJlZ2lzdGVyZWQgc291cmNlIHBhdGggZGlmZmVycyBmcm9tIHRoZSBmaXhlZCBkYXRhIHJvb3Q6IHtyZWxhdGl2ZX0iKQogICAgaWYgc291cmNlX2lkZW50aXRpZXNbImRhdGFzZXRzL3ZhbGlkLnBhcnF1ZXQiXSAhPSB2YWxpZDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJkdXBsaWNhdGVkIHZhbGlkLnBhcnF1ZXQgcGluIGRpZmZlcnMgZnJvbSB0aGUgc2l4LXNvdXJjZSBpZGVudGl0eSBtYXAiKQoKICAgIGlkZW50aXR5ID0gbWFuaWZlc3QuZ2V0KCJkYXRhX2lkZW50aXR5IikKICAgIGlmIG5vdCBpc2luc3RhbmNlKGlkZW50aXR5LCBkaWN0KSBvciBzZXQoaWRlbnRpdHkpICE9IHsKICAgICAgICAic2NoZW1hIiwKICAgICAgICAiYWRtaXNzaW9uIiwKICAgICAgICAicG9ydGFibGVfbWFuaWZlc3QiLAogICAgICAgICJyZWdpc3RlcmVkX3NvdXJjZV9pZGVudGl0aWVzIiwKICAgICAgICAicmVnaXN0ZXJlZF92YWxpZF9wYXJxdWV0IiwKICAgICAgICAib2JzZXJ2ZWRfc291cmNlc19iZWZvcmVfc3BsaXQiLAogICAgICAgICJvYnNlcnZlZF9zb3VyY2VzX2FmdGVyX3NwbGl0IiwKICAgICAgICAib2JzZXJ2ZWRfYmVmb3JlX3NlcXVlbmNlX29wZW4iLAogICAgICAgICJvYnNlcnZlZF9hZnRlcl9leHBvcnQiLAogICAgfToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgbWFuaWZlc3QgcmVnaXN0ZXJlZC1kYXRhIG9ic2VydmF0aW9ucyBhcmUgbWFsZm9ybWVkIikKICAgIGlmICgKICAgICAgICBpZGVudGl0eS5nZXQoInNjaGVtYSIpICE9ICJmZzE5LnJlZ2lzdGVyZWQtdmFsaWQtZGF0YS1pZGVudGl0eS52MSIKICAgICAgICBvciBpZGVudGl0eS5nZXQoImFkbWlzc2lvbiIpICE9IGFkbWlzc2lvbgogICAgICAgIG9yIGlkZW50aXR5LmdldCgicmVnaXN0ZXJlZF9zb3VyY2VfaWRlbnRpdGllcyIpICE9IHNvdXJjZV9pZGVudGl0aWVzCiAgICAgICAgb3IgaWRlbnRpdHkuZ2V0KCJyZWdpc3RlcmVkX3ZhbGlkX3BhcnF1ZXQiKSAhPSB2YWxpZAogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgbWFuaWZlc3QgZGF0YSBpZGVudGl0eSBkaWZmZXJzIGZyb20gaXRzIGZpeGVkIGJpbmRpbmciKQogICAgZXhwZWN0ZWRfcG9ydGFibGVfb2JzZXJ2YXRpb24gPSB7CiAgICAgICAgKipFWFBFQ1RFRF9QT1JUQUJMRV9NQU5JRkVTVCwKICAgICAgICAic291cmNlX25hbWVzIjogRVhQRUNURURfUE9SVEFCTEVfU09VUkNFX05BTUVTLAogICAgICAgICJzcGxpdF9tYW5pZmVzdF9zaGEyNTYiOiBFWFBFQ1RFRF9TUExJVF9TSEEyNTYsCiAgICAgICAgInZlcnNpb24iOiAxLAogICAgfQogICAgbWFuaWZlc3RfcG9ydGFibGUgPSBpZGVudGl0eS5nZXQoInBvcnRhYmxlX21hbmlmZXN0IikKICAgIGlmIG1hbmlmZXN0X3BvcnRhYmxlICE9IGV4cGVjdGVkX3BvcnRhYmxlX29ic2VydmF0aW9uIG9yIGFueSgKICAgICAgICB0eXBlKG1hbmlmZXN0X3BvcnRhYmxlW2tleV0pIGlzIG5vdCB0eXBlKHZhbHVlKQogICAgICAgIGZvciBrZXksIHZhbHVlIGluIGV4cGVjdGVkX3BvcnRhYmxlX29ic2VydmF0aW9uLml0ZW1zKCkKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IG1hbmlmZXN0IHBvcnRhYmxlIHJlZ2lzdHJ5IGlkZW50aXR5IGRpZmZlcnMiKQoKICAgIHNvdXJjZV9vYnNlcnZhdGlvbl9maWVsZHMgPSB7CiAgICAgICAgInBhdGgiLAogICAgICAgICJyZXNvbHZlZCIsCiAgICAgICAgImJ5dGVzIiwKICAgICAgICAiZGV2aWNlIiwKICAgICAgICAiaW5vZGUiLAogICAgICAgICJpc2xpbmsiLAogICAgICAgICJtdGltZV9ucyIsCiAgICB9CiAgICBmb3IgbmFtZSBpbiAoIm9ic2VydmVkX3NvdXJjZXNfYmVmb3JlX3NwbGl0IiwgIm9ic2VydmVkX3NvdXJjZXNfYWZ0ZXJfc3BsaXQiKToKICAgICAgICBvYnNlcnZhdGlvbnMgPSBpZGVudGl0eVtuYW1lXQogICAgICAgIGlmIG5vdCBpc2luc3RhbmNlKG9ic2VydmF0aW9ucywgZGljdCkgb3Igc2V0KG9ic2VydmF0aW9ucykgIT0gc2V0KAogICAgICAgICAgICBFWFBFQ1RFRF9QT1JUQUJMRV9TT1VSQ0VfTkFNRVMKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZXhwb3J0IG1hbmlmZXN0IHtuYW1lfSBtZW1iZXJzaGlwIGRpZmZlcnMiKQogICAgICAgIGZvciByZWxhdGl2ZSBpbiBFWFBFQ1RFRF9QT1JUQUJMRV9TT1VSQ0VfTkFNRVM6CiAgICAgICAgICAgIG9ic2VydmF0aW9uID0gb2JzZXJ2YXRpb25zW3JlbGF0aXZlXQogICAgICAgICAgICByZWdpc3RlcmVkX3NvdXJjZSA9IHNvdXJjZV9pZGVudGl0aWVzW3JlbGF0aXZlXQogICAgICAgICAgICBleHBlY3RlZF9zb3VyY2Vfb2JzZXJ2YXRpb24gPSB7CiAgICAgICAgICAgICAgICBrZXk6IHJlZ2lzdGVyZWRfc291cmNlW2tleV0gZm9yIGtleSBpbiBzb3VyY2Vfb2JzZXJ2YXRpb25fZmllbGRzCiAgICAgICAgICAgIH0KICAgICAgICAgICAgaWYgKAogICAgICAgICAgICAgICAgbm90IGlzaW5zdGFuY2Uob2JzZXJ2YXRpb24sIGRpY3QpCiAgICAgICAgICAgICAgICBvciBzZXQob2JzZXJ2YXRpb24pICE9IHNvdXJjZV9vYnNlcnZhdGlvbl9maWVsZHMKICAgICAgICAgICAgICAgIG9yIG9ic2VydmF0aW9uICE9IGV4cGVjdGVkX3NvdXJjZV9vYnNlcnZhdGlvbgogICAgICAgICAgICAgICAgb3IgYW55KAogICAgICAgICAgICAgICAgICAgIHR5cGUob2JzZXJ2YXRpb25ba2V5XSkgaXMgbm90IHR5cGUodmFsdWUpCiAgICAgICAgICAgICAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gZXhwZWN0ZWRfc291cmNlX29ic2VydmF0aW9uLml0ZW1zKCkKICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgKToKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJleHBvcnQgbWFuaWZlc3Qge25hbWV9IGRpZmZlcnMgZnJvbSBhZG1pc3Npb246IHtyZWxhdGl2ZX0iKQogICAgaWYgaWRlbnRpdHlbIm9ic2VydmVkX3NvdXJjZXNfYmVmb3JlX3NwbGl0Il0gIT0gaWRlbnRpdHlbIm9ic2VydmVkX3NvdXJjZXNfYWZ0ZXJfc3BsaXQiXToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJyZWdpc3RlcmVkIHNvdXJjZSBpZGVudGl0aWVzIGNoYW5nZWQgZHVyaW5nIHNwbGl0IHZhbGlkYXRpb24iKQoKICAgIG9ic2VydmF0aW9uX2tleXMgPSB7CiAgICAgICAgInBhdGgiLAogICAgICAgICJyZXNvbHZlZCIsCiAgICAgICAgImJ5dGVzIiwKICAgICAgICAiZGV2aWNlIiwKICAgICAgICAiaW5vZGUiLAogICAgICAgICJpc2xpbmsiLAogICAgICAgICJtdGltZV9ucyIsCiAgICAgICAgInBhcmVudCIsCiAgICB9CiAgICBwYXJlbnRfa2V5cyA9IHsicGF0aCIsICJyZXNvbHZlZCIsICJkZXZpY2UiLCAiaW5vZGUifQogICAgZm9yIG5hbWUgaW4gKCJvYnNlcnZlZF9iZWZvcmVfc2VxdWVuY2Vfb3BlbiIsICJvYnNlcnZlZF9hZnRlcl9leHBvcnQiKToKICAgICAgICBvYnNlcnZhdGlvbiA9IGlkZW50aXR5W25hbWVdCiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2Uob2JzZXJ2YXRpb24sIGRpY3QpIG9yIHNldChvYnNlcnZhdGlvbikgIT0gb2JzZXJ2YXRpb25fa2V5czoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImV4cG9ydCBtYW5pZmVzdCB7bmFtZX0gZmllbGRzIGRpZmZlciIpCiAgICAgICAgZXhwZWN0ZWRfb2JzZXJ2YXRpb24gPSB7CiAgICAgICAgICAgIGtleTogdmFsaWRba2V5XSBmb3Iga2V5IGluICgicGF0aCIsICJyZXNvbHZlZCIsICJieXRlcyIsICJkZXZpY2UiLCAiaW5vZGUiLCAibXRpbWVfbnMiKQogICAgICAgIH0KICAgICAgICBpZiAoCiAgICAgICAgICAgIGFueShvYnNlcnZhdGlvbi5nZXQoa2V5KSAhPSB2YWx1ZSBmb3Iga2V5LCB2YWx1ZSBpbiBleHBlY3RlZF9vYnNlcnZhdGlvbi5pdGVtcygpKQogICAgICAgICAgICBvciBvYnNlcnZhdGlvbi5nZXQoImlzbGluayIpIGlzIG5vdCBGYWxzZQogICAgICAgICk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJleHBvcnQgbWFuaWZlc3Qge25hbWV9IGRpZmZlcnMgZnJvbSBhZG1pdHRlZCBsZWFmIHN0YXRzIikKICAgICAgICBpZiBhbnkoCiAgICAgICAgICAgIHR5cGUob2JzZXJ2YXRpb25ba2V5XSkgaXMgbm90IHR5cGUodmFsdWUpIGZvciBrZXksIHZhbHVlIGluIGV4cGVjdGVkX29ic2VydmF0aW9uLml0ZW1zKCkKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZXhwb3J0IG1hbmlmZXN0IHtuYW1lfSBoYXMgdW5leHBlY3RlZCBsZWFmIHN0YXQgdHlwZXMiKQogICAgICAgIHBhcmVudCA9IG9ic2VydmF0aW9uLmdldCgicGFyZW50IikKICAgICAgICBpZiAoCiAgICAgICAgICAgIG5vdCBpc2luc3RhbmNlKHBhcmVudCwgZGljdCkKICAgICAgICAgICAgb3Igc2V0KHBhcmVudCkgIT0gcGFyZW50X2tleXMKICAgICAgICAgICAgb3IgcGFyZW50LmdldCgicGF0aCIpICE9IHN0cihQYXRoKHZhbGlkWyJwYXRoIl0pLnBhcmVudCkKICAgICAgICAgICAgb3IgcGFyZW50LmdldCgicmVzb2x2ZWQiKSAhPSBFWFBFQ1RFRF9QQVJFTlRfUkVTT0xVVElPTlNbImRhdGFzZXRzIl0KICAgICAgICAgICAgb3IgdHlwZShwYXJlbnQuZ2V0KCJkZXZpY2UiKSkgaXMgbm90IGludAogICAgICAgICAgICBvciB0eXBlKHBhcmVudC5nZXQoImlub2RlIikpIGlzIG5vdCBpbnQKICAgICAgICAgICAgb3IgcGFyZW50WyJkZXZpY2UiXSA8IDAKICAgICAgICAgICAgb3IgcGFyZW50WyJpbm9kZSJdIDw9IDAKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZXhwb3J0IG1hbmlmZXN0IHtuYW1lfSBwYXJlbnQgYWxpYXMgb2JzZXJ2YXRpb24gZGlmZmVycyIpCiAgICBpZiBpZGVudGl0eVsib2JzZXJ2ZWRfYmVmb3JlX3NlcXVlbmNlX29wZW4iXSAhPSBpZGVudGl0eVsib2JzZXJ2ZWRfYWZ0ZXJfZXhwb3J0Il06CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigicmVnaXN0ZXJlZCBkYXRhIGlkZW50aXR5IGNoYW5nZWQgZHVyaW5nIGZlYXR1cmUtb25seSBleHBvcnQiKQoKCmRlZiBfcmVhZF9leHBvcnRfYnVuZGxlKAogICAgZXZpZGVuY2Vfcm9vdDogUGF0aCwKICAgIGV4cG9ydF9yb290OiBQYXRoLAogICAgYmluZGluZ19wYXRoOiBQYXRoLAogICAgc291cmNlX21hbmlmZXN0X3BhdGg6IFBhdGgsCikgLT4gdHVwbGVbCiAgICBkaWN0W3N0ciwgQW55XSwKICAgIGRpY3Rbc3RyLCBBbnldLAogICAgZGljdFtzdHIsIEFueV0sCiAgICBkaWN0W3N0ciwgQW55XSwKICAgIHR1cGxlW2ludCwgaW50LCBpbnQsIGludF0sCl06CiAgICAiIiJSZWFkIGFuZCB2ZXJpZnkgdGhlIGNvbXBsZXRlZCBleHBvcnRlciBtYW5pZmVzdCwgYmluZGluZywgYW5kIHNvdXJjZSBwaW5zLiIiIgogICAgaWYgZXZpZGVuY2Vfcm9vdC5pc19zeW1saW5rKCkgb3Igbm90IGV2aWRlbmNlX3Jvb3QuaXNfZGlyKCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigicHJpdmF0ZSBldmlkZW5jZSByb290IG11c3QgYmUgYSByZWFsIGRpcmVjdG9yeSIpCiAgICBpZiBldmlkZW5jZV9yb290LnJlc29sdmUoc3RyaWN0PVRydWUpICE9IGV2aWRlbmNlX3Jvb3Q6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigicHJpdmF0ZSBldmlkZW5jZSByb290IGNhbm5vdCB0cmF2ZXJzZSBhIHN5bWxpbmsiKQogICAgaWYgKAogICAgICAgIGV4cG9ydF9yb290ICE9IGV2aWRlbmNlX3Jvb3QgLyAiZXhwb3J0cyIKICAgICAgICBvciBiaW5kaW5nX3BhdGggIT0gZXZpZGVuY2Vfcm9vdCAvICJFWFBPUlRfQklORElOR19WNC5qc29uIgogICAgICAgIG9yIHNvdXJjZV9tYW5pZmVzdF9wYXRoICE9IGV2aWRlbmNlX3Jvb3QgLyAiRVhQT1JUX1NPVVJDRV9NQU5JRkVTVC5qc29uIgogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKAogICAgICAgICAgICAiZXhwb3J0IG1ldGFkYXRhIGFuZCBwcmVkaWN0aW9ucyBtdXN0IHVzZSB0aGUgZml4ZWQgcHJpdmF0ZSBldmlkZW5jZSBsYXlvdXQiCiAgICAgICAgKQogICAgaWYgZXhwb3J0X3Jvb3QuaXNfc3ltbGluaygpIG9yIG5vdCBleHBvcnRfcm9vdC5pc19kaXIoKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgcm9vdCBtdXN0IGJlIGEgcmVhbCBkaXJlY3RvcnkiKQogICAgaWYgZXhwb3J0X3Jvb3QucmVzb2x2ZShzdHJpY3Q9VHJ1ZSkgIT0gZXhwb3J0X3Jvb3Q6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IHJvb3QgY2Fubm90IHRyYXZlcnNlIGEgc3ltbGluayIpCiAgICBpZiBiaW5kaW5nX3BhdGguaXNfc3ltbGluaygpIG9yIHNvdXJjZV9tYW5pZmVzdF9wYXRoLmlzX3N5bWxpbmsoKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJiaW5kaW5nIGFuZCBzb3VyY2UtbWFuaWZlc3QgZmlsZXMgY2Fubm90IGJlIHN5bWxpbmtzIikKICAgIGJpbmRpbmcsIGJpbmRpbmdfcGluLCBfID0gX3JlYWRfanNvbihiaW5kaW5nX3BhdGgsIE1BWF9TTUFMTF9KU09OX0JZVEVTKQogICAgaWYgKGJpbmRpbmdfcGluWyJieXRlcyJdLCBiaW5kaW5nX3Bpblsic2hhMjU2Il0pICE9ICgKICAgICAgICBFWFBFQ1RFRF9CSU5ESU5HX0JZVEVTLAogICAgICAgIEVYUEVDVEVEX0JJTkRJTkdfU0hBMjU2LAogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJmaXJzdC1jZWxsIGV4cG9ydCBiaW5kaW5nIGJ5dGVzIGRpZmZlciBmcm9tIHRoZSByZXZpZXdlZCBjb250cmFjdCIpCiAgICBpZiBiaW5kaW5nLmdldCgic2NoZW1hIikgIT0gImZnMTkuZmlyc3QtY2VsbC1ldmFsdWF0aW9uLWV4cG9ydC1iaW5kaW5nLnYxIjoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJ1bnN1cHBvcnRlZCBmaXJzdC1jZWxsIGV4cG9ydCBiaW5kaW5nIHNjaGVtYSIpCiAgICBzb3VyY2VfbWFuaWZlc3QsIHNvdXJjZV9waW4sIF8gPSBfcmVhZF9qc29uKHNvdXJjZV9tYW5pZmVzdF9wYXRoLCBNQVhfRVhQT1JUX01BTklGRVNUX0JZVEVTKQogICAgaWYgKHNvdXJjZV9waW5bImJ5dGVzIl0sIHNvdXJjZV9waW5bInNoYTI1NiJdKSAhPSAoCiAgICAgICAgRVhQRUNURURfU09VUkNFX01BTklGRVNUX0JZVEVTLAogICAgICAgIEVYUEVDVEVEX1NPVVJDRV9NQU5JRkVTVF9TSEEyNTYsCiAgICApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoInNvdXJjZSBtYW5pZmVzdCBieXRlcyBkaWZmZXIgZnJvbSB0aGUgcmV2aWV3ZWQgY29udHJhY3QiKQogICAgZm9yIGtleSwgYWN0dWFsIGluICgKICAgICAgICAoImV2aWRlbmNlX3Jvb3QiLCBldmlkZW5jZV9yb290KSwKICAgICAgICAoIm91dHB1dF9kaXIiLCBleHBvcnRfcm9vdCksCiAgICApOgogICAgICAgIGV4cGVjdGVkID0gYmluZGluZy5nZXQoa2V5KQogICAgICAgIGlmIG5vdCBpc2luc3RhbmNlKGV4cGVjdGVkLCBzdHIpIG9yIGV4cGVjdGVkICE9IHN0cihhY3R1YWwpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZXhwb3J0IGJpbmRpbmcge2tleX0gZGlmZmVycyBmcm9tIHRoZSBleHBsaWNpdCBwcml2YXRlIHJvb3QiKQogICAgZm9yIGtleSBpbiAoImRhdGFfcm9vdCIsICJzb3VyY2Vfcm9vdCIpOgogICAgICAgIHZhbHVlID0gYmluZGluZy5nZXQoa2V5KQogICAgICAgIGlmIG5vdCBpc2luc3RhbmNlKHZhbHVlLCBzdHIpIG9yIG5vdCBQYXRoKHZhbHVlKS5pc19hYnNvbHV0ZSgpIG9yICIuLiIgaW4gUGF0aCh2YWx1ZSkucGFydHM6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJleHBvcnQgYmluZGluZyB7a2V5fSBpcyBub3QgYW4gZXhwbGljaXQgYWJzb2x1dGUgQiBwYXRoIikKICAgIHNvdXJjZSA9IGJpbmRpbmcuZ2V0KCJzb3VyY2UiKQogICAgaWYgbm90IGlzaW5zdGFuY2Uoc291cmNlLCBkaWN0KToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgYmluZGluZyBzb3VyY2Ugc2VjdGlvbiBpcyBtYWxmb3JtZWQiKQogICAgc291cmNlX21hbmlmZXN0X2ZpbGVzID0gc291cmNlX21hbmlmZXN0LmdldCgiZmlsZXMiKQogICAgaWYgKAogICAgICAgIHNvdXJjZS5nZXQoImNvbW1pdCIpICE9IFNPVVJDRV9DT01NSVQKICAgICAgICBvciBzb3VyY2UuZ2V0KCJtYW5pZmVzdF9zaGEyNTYiKSAhPSBFWFBFQ1RFRF9TT1VSQ0VfTUFOSUZFU1RfU0hBMjU2CiAgICAgICAgb3Igc291cmNlLmdldCgic291cmNlX3NldF9zaGEyNTYiKSAhPSBFWFBFQ1RFRF9TT1VSQ0VfU0VUX1NIQTI1NgogICAgICAgIG9yIHNvdXJjZS5nZXQoIm1hbmlmZXN0X3JlbGF0aXZlX3BhdGgiKSAhPSAiRVhQT1JUX1NPVVJDRV9NQU5JRkVTVC5qc29uIgogICAgICAgIG9yIHNvdXJjZS5nZXQoIm1hbmlmZXN0X2J5dGVzIikgIT0gRVhQRUNURURfU09VUkNFX01BTklGRVNUX0JZVEVTCiAgICAgICAgb3IgdHlwZShzb3VyY2UuZ2V0KCJmaWxlX2NvdW50IikpIGlzIG5vdCBpbnQKICAgICAgICBvciBzb3VyY2UuZ2V0KCJmaWxlX2NvdW50IikgIT0gMTY3CiAgICAgICAgb3Igc291cmNlX21hbmlmZXN0LmdldCgic291cmNlX2NvbW1pdCIpICE9IFNPVVJDRV9DT01NSVQKICAgICAgICBvciBzb3VyY2VfbWFuaWZlc3QuZ2V0KCJzY2hlbWEiKSAhPSAiZmcxOS50ZWFjaGVyLWNhY2hlLXNvdXJjZS1tYW5pZmVzdC52MSIKICAgICAgICBvciBzb3VyY2VfbWFuaWZlc3QuZ2V0KCJzb3VyY2Vfcm9vdCIpICE9IGJpbmRpbmcuZ2V0KCJzb3VyY2Vfcm9vdCIpCiAgICAgICAgb3Igc2V0KHNvdXJjZV9tYW5pZmVzdCkgIT0geyJmaWxlcyIsICJzY2hlbWEiLCAic291cmNlX2NvbW1pdCIsICJzb3VyY2Vfcm9vdCJ9CiAgICAgICAgb3Igbm90IGlzaW5zdGFuY2Uoc291cmNlX21hbmlmZXN0X2ZpbGVzLCBsaXN0KQogICAgICAgIG9yIGxlbihzb3VyY2VfbWFuaWZlc3RfZmlsZXMpICE9IHNvdXJjZS5nZXQoImZpbGVfY291bnQiKQogICAgICAgIG9yIHNvdXJjZS5nZXQoInNvdXJjZV9yb290IikgIT0gYmluZGluZy5nZXQoInNvdXJjZV9yb290IikKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IGJpbmRpbmcgYW5kIHNvdXJjZSBtYW5pZmVzdCBkbyBub3QgaWRlbnRpZnkgc291cmNlNjgiKQogICAgc291cmNlX3RyZWVfZGlnZXN0ID0gaGFzaGxpYi5zaGEyNTYoKQogICAgc291cmNlX3BhdGhzOiBzZXRbc3RyXSA9IHNldCgpCiAgICBmb3IgaXRlbSBpbiBzb3VyY2VfbWFuaWZlc3RfZmlsZXM6CiAgICAgICAgaWYgKAogICAgICAgICAgICBub3QgaXNpbnN0YW5jZShpdGVtLCBkaWN0KQogICAgICAgICAgICBvciBzZXQoaXRlbSkgIT0geyJieXRlcyIsICJwYXRoIiwgInNoYTI1NiJ9CiAgICAgICAgICAgIG9yIHR5cGUoaXRlbS5nZXQoImJ5dGVzIikpIGlzIG5vdCBpbnQKICAgICAgICAgICAgb3IgaXRlbVsiYnl0ZXMiXSA8PSAwCiAgICAgICAgICAgIG9yIG5vdCBpc2luc3RhbmNlKGl0ZW0uZ2V0KCJwYXRoIiksIHN0cikKICAgICAgICAgICAgb3Igbm90IGlzaW5zdGFuY2UoaXRlbS5nZXQoInNoYTI1NiIpLCBzdHIpCiAgICAgICAgICAgIG9yIGxlbihpdGVtWyJzaGEyNTYiXSkgIT0gNjQKICAgICAgICAgICAgb3IgYW55KGNoYXJhY3RlciBub3QgaW4gIjAxMjM0NTY3ODlhYmNkZWYiIGZvciBjaGFyYWN0ZXIgaW4gaXRlbVsic2hhMjU2Il0pCiAgICAgICAgICAgIG9yIGl0ZW1bInBhdGgiXSBpbiBzb3VyY2VfcGF0aHMKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJzb3VyY2UgbWFuaWZlc3QgY29udGFpbnMgYSBtYWxmb3JtZWQgb3IgZHVwbGljYXRlIHNvdXJjZSBwaW4iKQogICAgICAgIHNvdXJjZV9wYXRocy5hZGQoaXRlbVsicGF0aCJdKQogICAgZm9yIGl0ZW0gaW4gc29ydGVkKHNvdXJjZV9tYW5pZmVzdF9maWxlcywga2V5PWxhbWJkYSByb3c6IHJvd1sicGF0aCJdKToKICAgICAgICBzb3VyY2VfdHJlZV9kaWdlc3QudXBkYXRlKAogICAgICAgICAgICBpdGVtWyJwYXRoIl0uZW5jb2RlKCJ1dGYtOCIpICsgYiJcMCIgKyBpdGVtWyJzaGEyNTYiXS5lbmNvZGUoImFzY2lpIikgKyBiIlxuIgogICAgICAgICkKICAgIGlmIHNvdXJjZV90cmVlX2RpZ2VzdC5oZXhkaWdlc3QoKSAhPSBFWFBFQ1RFRF9TT1VSQ0VfU0VUX1NIQTI1NjoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJzb3VyY2UgbWFuaWZlc3QgYWdncmVnYXRlIHBpbiBkaWZmZXJzIGZyb20gc291cmNlNjgiKQogICAgZm9yIHJlbGF0aXZlLCAoYnl0ZV9jb3VudCwgZGlnZXN0KSBpbiBTT1VSQ0VfUElOUy5pdGVtcygpOgogICAgICAgIGV4cGVjdGVkX3BpbiA9IHsiYnl0ZXMiOiBieXRlX2NvdW50LCAic2hhMjU2IjogZGlnZXN0fQogICAgICAgIGlmIHNvdXJjZV9tYW5pZmVzdC5nZXQoImZpbGVzIikgaXMgTm9uZSBvciBub3QgYW55KAogICAgICAgICAgICBpdGVtLmdldCgicGF0aCIpID09IHJlbGF0aXZlIGFuZCBpdGVtID09IHsicGF0aCI6IHJlbGF0aXZlLCAqKmV4cGVjdGVkX3Bpbn0KICAgICAgICAgICAgZm9yIGl0ZW0gaW4gc291cmNlX21hbmlmZXN0WyJmaWxlcyJdCiAgICAgICAgICAgIGlmIGlzaW5zdGFuY2UoaXRlbSwgZGljdCkKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKAogICAgICAgICAgICAgICAgZiJzb3VyY2UgbWFuaWZlc3QgZG9lcyBub3QgY29udGFpbiB0aGUgZXhhY3QgY2Fub25pY2FsIHBpbjoge3JlbGF0aXZlfSIKICAgICAgICAgICAgKQogICAgc291cmNlX2ZpbGVfcGlucyA9IHsKICAgICAgICBpdGVtLmdldCgicGF0aCIpOiBpdGVtIGZvciBpdGVtIGluIHNvdXJjZV9tYW5pZmVzdFsiZmlsZXMiXSBpZiBpc2luc3RhbmNlKGl0ZW0sIGRpY3QpCiAgICB9CiAgICBhcGlfZmlsZXMgPSBzb3VyY2UuZ2V0KCJhcGlfZmlsZXMiKQogICAgaWYgbm90IGlzaW5zdGFuY2UoYXBpX2ZpbGVzLCBkaWN0KToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgYmluZGluZyBjYW5vbmljYWwgQVBJIHBpbnMgZGlmZmVyIGZyb20gdGhlIHNvdXJjZSBtYW5pZmVzdCIpCiAgICBmb3IgcmVsYXRpdmUsIGl0ZW0gaW4gYXBpX2ZpbGVzLml0ZW1zKCk6CiAgICAgICAgc291cmNlX3BpbiA9IHNvdXJjZV9maWxlX3BpbnMuZ2V0KHJlbGF0aXZlKQogICAgICAgIGlmICgKICAgICAgICAgICAgbm90IGlzaW5zdGFuY2UocmVsYXRpdmUsIHN0cikKICAgICAgICAgICAgb3Igbm90IGlzaW5zdGFuY2Uoc291cmNlX3BpbiwgZGljdCkKICAgICAgICAgICAgb3Igbm90IGlzaW5zdGFuY2UoaXRlbSwgZGljdCkKICAgICAgICAgICAgb3Igc2V0KGl0ZW0pICE9IHsiYnl0ZXMiLCAic2hhMjU2In0KICAgICAgICAgICAgb3IgdHlwZShpdGVtLmdldCgiYnl0ZXMiKSkgaXMgbm90IGludAogICAgICAgICAgICBvciBpdGVtWyJieXRlcyJdIDw9IDAKICAgICAgICAgICAgb3Igbm90IGlzaW5zdGFuY2UoaXRlbS5nZXQoInNoYTI1NiIpLCBzdHIpCiAgICAgICAgICAgIG9yIGxlbihpdGVtWyJzaGEyNTYiXSkgIT0gNjQKICAgICAgICAgICAgb3IgYW55KGNoYXJhY3RlciBub3QgaW4gIjAxMjM0NTY3ODlhYmNkZWYiIGZvciBjaGFyYWN0ZXIgaW4gaXRlbVsic2hhMjU2Il0pCiAgICAgICAgICAgIG9yIGl0ZW1bImJ5dGVzIl0gIT0gc291cmNlX3Bpbi5nZXQoImJ5dGVzIikKICAgICAgICAgICAgb3IgaXRlbVsic2hhMjU2Il0gIT0gc291cmNlX3Bpbi5nZXQoInNoYTI1NiIpCiAgICAgICAgKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IGJpbmRpbmcgY2Fub25pY2FsIEFQSSBwaW5zIGRpZmZlciBmcm9tIHRoZSBzb3VyY2UgbWFuaWZlc3QiKQogICAgc3BsaXQgPSBiaW5kaW5nLmdldCgic3BsaXQiKQogICAgZXhwZWN0ZWRfc3BsaXQgPSB7CiAgICAgICAgImFic29sdXRlX3BhdGgiOiBzdHIoUGF0aChiaW5kaW5nWyJwcm9qZWN0X2Jhc2UiXSkgLyAic3BsaXRzIiAvICJmb2xkMC5qc29uIiksCiAgICAgICAgImNodW5rX3Jvd3MiOiAyMDAwLAogICAgICAgICJkdHlwZSI6ICI8ZjQiLAogICAgICAgICJldmFsdWF0aW9uX2dyb3VwX2NvdW50IjogTl9HUk9VUFMsCiAgICAgICAgImZlYXR1cmVfY291bnQiOiAxMTIsCiAgICAgICAgImZpdF92YWxpZF9ncm91cF9jb3VudCI6IDE0MDUsCiAgICAgICAgImZvbGQiOiAwLAogICAgICAgICJuX2ZvbGRzIjogNCwKICAgICAgICAib3V0cHV0X2NvdW50IjogTl9PVVRQVVRTLAogICAgICAgICJzZXF1ZW5jZV9yb3dzIjogMjBfMDAwLAogICAgICAgICJzaGEyNTYiOiBFWFBFQ1RFRF9TUExJVF9TSEEyNTYsCiAgICAgICAgInZhbGlkX3Jvd19ncm91cF9jb3VudCI6IDE4NzMsCiAgICB9CiAgICBpZiBub3QgaXNpbnN0YW5jZShzcGxpdCwgZGljdCkgb3Igc3BsaXQgIT0gZXhwZWN0ZWRfc3BsaXQ6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IGJpbmRpbmcgc3BsaXQgY29udHJhY3QgZGlmZmVycyBmcm9tIHRoZSBmaXhlZCBjMDlmIHBhbmVsIikKICAgIG91dHB1dHMgPSBiaW5kaW5nLmdldCgib3V0cHV0cyIpCiAgICBpZiBvdXRwdXRzICE9IHsKICAgICAgICAiY29tcGxldGlvbl9tYXJrZXIiOiAiRVhQT1JUX01BTklGRVNUX1YxLmpzb24iLAogICAgICAgICJkYXRhX2J5dGVzX2VhY2giOiBQUkVESUNUSU9OX0RBVEFfQllURVMsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgImZpbGVzX3Blcl9hcm0iOiBbInByZWRpY3Rpb25zLm5weSIsICJQUkVESUNUSU9OX1NJREVDQVIuanNvbiJdLAogICAgICAgICJzY29yZV9vcl9tZXRyaWMiOiBGYWxzZSwKICAgICAgICAic2NvcmVkX21hc2tfcmVhZCI6IEZhbHNlLAogICAgICAgICJzaGFwZSI6IGxpc3QoUFJFRElDVElPTl9TSEFQRSksCiAgICAgICAgInRhcmdldHNfcmVhZCI6IEZhbHNlLAogICAgfToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgYmluZGluZyBvdXRwdXQgY29udHJhY3QgZGlmZmVycyBmcm9tIHRoZSBmaXhlZCBmZWF0dXJlLW9ubHkgcGFuZWwiKQogICAgYXJtcyA9IGJpbmRpbmcuZ2V0KCJhcm1zIikKICAgIGlmIG5vdCBpc2luc3RhbmNlKGFybXMsIGxpc3QpIG9yIFthcm0uZ2V0KCJhcm0iKSBmb3IgYXJtIGluIGFybXNdICE9IGxpc3QoQVJNX01PREVTKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgYmluZGluZyBkb2VzIG5vdCBsaXN0IHRoZSBmaXZlIGZpeGVkIGFybXMgaW4gY2Fub25pY2FsIG9yZGVyIikKICAgIGNoZWNrcG9pbnQgPSBiaW5kaW5nLmdldCgiZXhwZWN0ZWRfY2hlY2twb2ludCIpCiAgICBpZiAoCiAgICAgICAgbm90IGlzaW5zdGFuY2UoY2hlY2twb2ludCwgZGljdCkKICAgICAgICBvciBjaGVja3BvaW50LmdldCgiZXBvY2giKSAhPSA1CiAgICAgICAgb3IgY2hlY2twb2ludC5nZXQoInVwZGF0ZXMiKSAhPSA0NDc2MAogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgYmluZGluZyBkb2VzIG5vdCBmaXggdGhlIHNlbGVjdGVkIGVwb2NoLTUvNDQsNzYwLXVwZGF0ZSBlbmRwb2ludCIpCiAgICBmb3IgYXJtIGluIGFybXM6CiAgICAgICAgbmFtZSA9IGFybVsiYXJtIl0KICAgICAgICBtb2RlbCA9IGFybS5nZXQoIm1vZGVsIikKICAgICAgICBpZiAoCiAgICAgICAgICAgIGFybS5nZXQoInJlcHJlc2VudGF0aW9uX21vZGUiKSAhPSBBUk1fTU9ERVNbbmFtZV0KICAgICAgICAgICAgb3IgYXJtLmdldCgic2VsZWN0ZWRfZXBvY2giLCA1KSAhPSA1CiAgICAgICAgICAgIG9yIG5vdCBpc2luc3RhbmNlKG1vZGVsLCBkaWN0KQogICAgICAgICAgICBvciBtb2RlbC5nZXQoImJ5dGVzIikgIT0gNF8yNzlfOTcwCiAgICAgICAgICAgIG9yIG5vdCBpc2luc3RhbmNlKG1vZGVsLmdldCgic2hhMjU2IiksIHN0cikKICAgICAgICAgICAgb3IgbGVuKG1vZGVsWyJzaGEyNTYiXSkgIT0gNjQKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie25hbWV9IGJpbmRpbmcgaWRlbnRpdHkgZGlmZmVycyBmcm9tIHRoZSBmaXhlZCBlcG9jaC01IHBhbmVsIikKICAgIG1hbmlmZXN0LCBtYW5pZmVzdF9waW4sIG1hbmlmZXN0X2lkZW50aXR5ID0gX3JlYWRfanNvbigKICAgICAgICBfc2FmZV9jaGlsZChleHBvcnRfcm9vdCwgIkVYUE9SVF9NQU5JRkVTVF9WMS5qc29uIiksIE1BWF9FWFBPUlRfTUFOSUZFU1RfQllURVMKICAgICkKICAgIGlmICgKICAgICAgICBzZXQobWFuaWZlc3QpCiAgICAgICAgIT0gewogICAgICAgICAgICAic2NoZW1hIiwKICAgICAgICAgICAgInN0YXR1cyIsCiAgICAgICAgICAgICJjcmVhdGVkX3V0YyIsCiAgICAgICAgICAgICJiaW5kaW5nX3NoYTI1NiIsCiAgICAgICAgICAgICJzb3VyY2UiLAogICAgICAgICAgICAic3BsaXQiLAogICAgICAgICAgICAiZGF0YV9pZGVudGl0eSIsCiAgICAgICAgICAgICJvcmRlcmVkX2dyb3VwcyIsCiAgICAgICAgICAgICJtZW1iZXJzaGlwX3NoYTI1NiIsCiAgICAgICAgICAgICJhcm1zIiwKICAgICAgICAgICAgInF1YWxpdHlfb3JfbWV0cmljX2NsYWltIiwKICAgICAgICB9CiAgICAgICAgb3IgbWFuaWZlc3QuZ2V0KCJzY2hlbWEiKSAhPSAiZmcxOS5maXJzdC1jZWxsLWV2YWx1YXRpb24tZXhwb3J0LW1hbmlmZXN0LnYxIgogICAgICAgIG9yIG1hbmlmZXN0LmdldCgic3RhdHVzIikgIT0gIkNPTVBMRVRFX1BSRURJQ1RJT05fRVhQT1JUU19OT1RfU0NPUkVEIgogICAgICAgIG9yIG1hbmlmZXN0LmdldCgiYmluZGluZ19zaGEyNTYiKSAhPSBiaW5kaW5nX3Bpblsic2hhMjU2Il0KICAgICAgICBvciBtYW5pZmVzdC5nZXQoInNvdXJjZSIpICE9IHNvdXJjZQogICAgICAgIG9yIG1hbmlmZXN0LmdldCgicXVhbGl0eV9vcl9tZXRyaWNfY2xhaW0iKSBpcyBub3QgRmFsc2UKICAgICAgICBvciBtYW5pZmVzdC5nZXQoInNwbGl0IikKICAgICAgICAhPSBiaW5kaW5nLmdldCgic3BsaXQiKSB8IHsicm9sZSI6ICJmaXQtZXhjbHVkZWQtZXhwb3NlZC1kZXZlbG9wbWVudC1ldmFsdWF0aW9uIn0KICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IGNvbXBsZXRpb24gbWFuaWZlc3QgZGlmZmVycyBmcm9tIHRoZSBmaXhlZCBiaW5kaW5nIikKICAgIF92YWxpZGF0ZV9kYXRhX2lkZW50aXR5KGJpbmRpbmcsIG1hbmlmZXN0KQogICAgcmV0dXJuIGJpbmRpbmcsIHNvdXJjZV9tYW5pZmVzdCwgbWFuaWZlc3QsIG1hbmlmZXN0X3BpbiwgbWFuaWZlc3RfaWRlbnRpdHkKCgpkZWYgX3ZhbGlkYXRlX2JvdW5kX3BhdGhzKGFyZ3M6IGFyZ3BhcnNlLk5hbWVzcGFjZSwgYmluZGluZzogZGljdFtzdHIsIEFueV0pIC0+IE5vbmU6CiAgICAiIiJSZXF1aXJlIGV2ZXJ5IHJ1bnRpbWUgcm9vdCB0byBtYXRjaCB0aGUgZXhhY3QgQiBsYXlvdXQgYmVmb3JlIHNvdXJjZSBpbXBvcnRzLiIiIgogICAgcGF0aHMgPSB7CiAgICAgICAgInByb2plY3RfYmFzZSI6IGFyZ3MuZXZpZGVuY2Vfcm9vdC5wYXJlbnQucGFyZW50LAogICAgICAgICJkYXRhX3Jvb3QiOiBhcmdzLmRhdGFfcm9vdCwKICAgICAgICAic291cmNlX3Jvb3QiOiBhcmdzLnNvdXJjZV9yb290LAogICAgICAgICJldmlkZW5jZV9yb290IjogYXJncy5ldmlkZW5jZV9yb290LAogICAgfQogICAgZm9yIGtleSwgcGF0aCBpbiBwYXRocy5pdGVtcygpOgogICAgICAgIGlmIHBhdGguaXNfc3ltbGluaygpIG9yIG5vdCBwYXRoLmlzX2RpcigpIG9yIHBhdGgucmVzb2x2ZShzdHJpY3Q9VHJ1ZSkgIT0gcGF0aDoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntrZXl9IG11c3QgYmUgYW4gZXhwbGljaXQgcmVhbCBkaXJlY3Rvcnkgd2l0aG91dCBzeW1saW5rIHRyYXZlcnNhbCIpCiAgICAgICAgaWYgYmluZGluZy5nZXQoa2V5KSAhPSBzdHIocGF0aCk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7a2V5fSBkaWZmZXJzIGZyb20gdGhlIGV4YWN0IGV4cG9ydCBiaW5kaW5nIikKICAgIGJhc2UgPSBwYXRoc1sicHJvamVjdF9iYXNlIl0KICAgIGlmICgKICAgICAgICBwYXRoc1siZGF0YV9yb290Il0gIT0gYmFzZSAvICJkYXRhIgogICAgICAgIG9yIHBhdGhzWyJzb3VyY2Vfcm9vdCJdICE9IGJhc2UgLyAic291cmNlLWZnMTktY2FjaGUtNjhmZjY4MjkiCiAgICAgICAgb3IgcGF0aHNbImV2aWRlbmNlX3Jvb3QiXQogICAgICAgICE9IGJhc2UgLyAicmVzZWFyY2hfdG1wIiAvICJmZzE5X2ZpcnN0X2NlbGxfZjBfczEwMDFfZXZhbHVhdGlvbl92MyIKICAgICAgICBvciBhcmdzLmV4cG9ydF9yb290ICE9IHBhdGhzWyJldmlkZW5jZV9yb290Il0gLyAiZXhwb3J0cyIKICAgICAgICBvciBhcmdzLmJpbmRpbmcgIT0gcGF0aHNbImV2aWRlbmNlX3Jvb3QiXSAvICJFWFBPUlRfQklORElOR19WNC5qc29uIgogICAgICAgIG9yIGFyZ3Muc291cmNlX21hbmlmZXN0ICE9IHBhdGhzWyJldmlkZW5jZV9yb290Il0gLyAiRVhQT1JUX1NPVVJDRV9NQU5JRkVTVC5qc29uIgogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJydW50aW1lIHBhdGhzIGRvIG5vdCBtYXRjaCB0aGUgZml4ZWQgcHJvamVjdC9kYXRhL3NvdXJjZS9ldmlkZW5jZSBsYXlvdXQiKQogICAgc3BsaXRfcGF0aCA9IFBhdGgoc3RyKGJpbmRpbmcuZ2V0KCJzcGxpdCIsIHt9KS5nZXQoImFic29sdXRlX3BhdGgiLCAiIikpKQogICAgaWYgc3BsaXRfcGF0aCAhPSBiYXNlIC8gInNwbGl0cyIgLyAiZm9sZDAuanNvbiIgb3Igc3BsaXRfcGF0aC5pc19zeW1saW5rKCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiY2FtcGFpZ24gc3BsaXQgcGF0aCBkaWZmZXJzIGZyb20gdGhlIGZpeGVkIGN1cnJlbnQgRjAgbWFuaWZlc3QiKQogICAgaWYgYXJncy5vdXQuaXNfc3ltbGluaygpIG9yIG5vdCBhcmdzLm91dC5pc19hYnNvbHV0ZSgpIG9yICIuLiIgaW4gYXJncy5vdXQucGFydHM6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic2NvcmUgcmVwb3J0IG11c3QgYmUgYSBkaXJlY3QgYWJzb2x1dGUgY3JlYXRlLW9ubHkgZXZpZGVuY2UgZmlsZSIpCiAgICBpZiBhcmdzLm91dC5wYXJlbnQgIT0gcGF0aHNbImV2aWRlbmNlX3Jvb3QiXSBvciBhcmdzLm91dC5zdWZmaXggIT0gIi5qc29uIjoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJzY29yZSByZXBvcnQgbXVzdCBiZSBhIEpTT04gY2hpbGQgb2YgdGhlIHByaXZhdGUgZXZpZGVuY2Ugcm9vdCIpCiAgICBpZiBhcmdzLm91dC5leGlzdHMoKToKICAgICAgICByYWlzZSBGaWxlRXhpc3RzRXJyb3IoZiJzY29yZSByZXBvcnQgYWxyZWFkeSBleGlzdHM6IHthcmdzLm91dH0iKQo=',
    'first_cell_score_core.py': 'IiIiU3RyZWFtIG9mZmljaWFsIHNjb3JlcyBhbmQgcGFpcmVkIGNvbXBvbmVudCBpbnRlcnZhbHMgZm9yIHRoZSBmaXhlZCBGMCBwYW5lbC4KClRoaXMgY29uc3VtZXIgYWNjZXB0cyBvbmx5IHRoZSByZXZpZXdlZCBmaXZlLWFybSwgZXBvY2gtNSwgZmVhdHVyZS1vbmx5IGV4cG9ydC4KSXQgcmVhZHMgb25lIHZhbGlkYXRpb24gZ3JvdXAgYXQgYSB0aW1lIGFuZCB3cml0ZXMgYSBjcmVhdGUtb25seSBsb2NhbCByZXBvcnQuCkl0IGRvZXMgbm90IHJ1biBpbmZlcmVuY2UsIGZpdCwgc2VsZWN0IGVwb2Nocywgb3IgY3JlYXRlIGEgcHVibGljLXRyYW5zZmVyIGNsYWltLgoiIiIKCmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnMKCmltcG9ydCBoYXNobGliCmltcG9ydCBpbXBvcnRsaWIKaW1wb3J0IGpzb24KaW1wb3J0IG9zCmltcG9ydCBzdGF0CmltcG9ydCBzeXMKaW1wb3J0IHppcGZpbGUKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aCwgUHVyZVBvc2l4UGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55LCBQcm90b2NvbAoKaW1wb3J0IG51bXB5IGFzIG5wCgpTT1VSQ0VfQ09NTUlUID0gIjY4ZmY2ODI5NzdiODIyZjM4MzUxYWFjMGY2MjExNDZmNWYxM2I0NDAiClNPVVJDRV9QSU5TID0gewogICAgInNyYy93bm4vX19pbml0X18ucHkiOiAoNjMsICIyZGNhMTBlZTdlMmU4YzExMzliZWQyMTY1YjMyMmQyMDc3ZmRiZWFjODA5NTQyOGU4Yzc5NmVmZTQ3ZWE2NmM2IiksCiAgICAic3JjL3dubi9kYXRhLnB5IjogKDQwMDAsICJhOTNkYTg0MDEzN2FkYmQ3YWZkYjc5YTIyNjg3ZmI3MzE1MGU3NzEwN2Y5MWNhMmFkYzUyMTFiODU0YmM3ZTAwIiksCiAgICAic3JjL3dubi9nbWV0cmljLnB5IjogKAogICAgICAgIDg5MDIsCiAgICAgICAgIjJjZjEwZTc0MGE4MzEzNzlmNjJiN2QxMjY5NDU2MDUzZjVjMWY5ZWZhMmRiYWIwMzE1NmUzNjY5NzJiZWQ1YWEiLAogICAgKSwKICAgICJzcmMvd25uL2V2YWx1YXRpb24ucHkiOiAoCiAgICAgICAgODA4MCwKICAgICAgICAiOTZjZWJmMmM4ZDU2NDQ5M2Y1YTNhODY2YjhkODc5OTY0MjQ3Zjg4NTU4N2E0NTVlYmIxNDQ5ZTkzNzIzMGQwZSIsCiAgICApLAogICAgInNyYy93bm4vcG9ydGFiaWxpdHkucHkiOiAoCiAgICAgICAgMjIzOSwKICAgICAgICAiYTM1Zjg0ZWU1Yzk1ZmQ0NWNkNzZhYTVmMzRmNzAxM2M4ZTFjNTk2M2E2MzA2YjBlNDA0OTM4NjUxNTU4OWE0NCIsCiAgICApLAogICAgInNyYy93bm4vc3BsaXRzLnB5IjogKDkwMDAsICIyNDAyNGZiYjgyZDMzYWZkYzJjNzA5OTEyNDA0ZjdjMWMyNjUwNGEwODE2YTA1MDQyOGM5ZjY2ZTA0ZTk1MWZlIiksCn0KRVhQRUNURURfQklORElOR19CWVRFUyA9IDE1Nzk2CkVYUEVDVEVEX0JJTkRJTkdfU0hBMjU2ID0gIjRhOGFlMjIyMDQ5NzFhYzA3ZDZlNWNkZmQ5ZjY1YTYyZTlmODQ0MDgzMTAzNWRiNzI3MWU0ZTBkYTNkODNhYmQiCkVYUEVDVEVEX1NPVVJDRV9NQU5JRkVTVF9CWVRFUyA9IDI3NTY5CkVYUEVDVEVEX1NPVVJDRV9NQU5JRkVTVF9TSEEyNTYgPSAiYTQ0ZjhhM2ZkM2ZkYTFlZmRkYTU3OGM4Njk0MDM2ZGZmYWY3MmNlNjk5OWU0ZTkwOTVmNjI5ODM2NWIzNTYyMSIKRVhQRUNURURfU09VUkNFX1NFVF9TSEEyNTYgPSAiOWM0NjFiY2M1ODgzNTcyYTA3MWQ0MWE4NmUxOGI5YTkxMTM2MDk5Y2Y4ZmE4NzM3MWNlN2MzZWU0MTQ0NWJkOCIKRVhQRUNURURfU1BMSVRfU0hBMjU2ID0gImMwOWY3MjQwMzk2YTMzY2E4Mjk3NzExNWFmOGQ2MDY3ZTU3MWMwZWMyNWM1OGEzYzk5YjQzOGNhMmQyMjk0YjciCkVYUEVDVEVEX1ZBTElEX1BBUlFVRVQgPSB7CiAgICAiYnl0ZXMiOiA1XzYxNV81MDlfMTI4LAogICAgImRldmljZSI6IDY0NTEyLAogICAgImlub2RlIjogNzQzODM2MjU3LAogICAgImlzbGluayI6IEZhbHNlLAogICAgIm10aW1lX25zIjogMV81NzdfODMzXzIwMF8wMDBfMDAwXzAwMCwKICAgICJwYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvZGF0YS9kYXRhc2V0cy92YWxpZC5wYXJxdWV0IiwKICAgICJyZWdpc3RlcmVkX3NoYTI1NiI6ICIyODYxZWUzMDkxNDA2NTcwMjYyNmEzYWMwMzNkODBkMDU2M2NkYWNhZWE1MGM2NjdmYzgyOGMyZWViNDE4NzgyIiwKICAgICJyZXNvbHZlZCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvV3VuZGVyRnVuZGVyL2RhdGFzZXRzL3ZhbGlkLnBhcnF1ZXQiLAp9CkVYUEVDVEVEX0RBVEFfQURNSVNTSU9OID0gewogICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9jaGlsZF9hbGx0aHJlZV9maXRfdjEvQURNSVNTSU9OLmpzb24iLAogICAgImJ5dGVzIjogMTI5MTIsCiAgICAicmVsYXRpdmVfcGF0aCI6ICJBRE1JU1NJT04uanNvbiIsCiAgICAicm9vdF9rZXkiOiAiZml0X3JlY2VpcHRzX3Jvb3QiLAogICAgInNoYTI1NiI6ICI1MzEyYWU1ZmVmZmM2N2Q3YTMzMDUxMzc5YWZhYmI5YzEzNGRkOTQ2ZjFmOTczMTVkODM5YzdkNmYxMGQ1NTQwIiwKfQpFWFBFQ1RFRF9QT1JUQUJMRV9SRUdJU1RSWV9TSEEyNTYgPSAoCiAgICAiMWYzOWNiZjhiMjQyNDk1ZTE4YmFhZjVkMzZlZWFjOTA3YzBhZDY1NWU5NTFiMGVmYTZhNDg1ODBiNzY5OTM3OCIKKQpFWFBFQ1RFRF9QT1JUQUJMRV9SRUdJU1RSWV9CWVRFUyA9IDIxNTcKRVhQRUNURURfUE9SVEFCTEVfUkVHSVNUUllfUkVMQVRJVkVfUEFUSCA9ICJzdXBwb3J0LXNvdXJjZS90ZXJtaW5hbC12MS9QT1JUQUJMRV9TT1VSQ0VTLmpzb24iCkVYUEVDVEVEX1BPUlRBQkxFX01BTklGRVNUID0gewogICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9zdXBwb3J0LXNvdXJjZS90ZXJtaW5hbC12MS9QT1JUQUJMRV9TT1VSQ0VTLmpzb24iLAogICAgImJ5dGVzIjogMjE1NywKICAgICJzaGEyNTYiOiBFWFBFQ1RFRF9QT1JUQUJMRV9SRUdJU1RSWV9TSEEyNTYsCn0KRVhQRUNURURfUE9SVEFCTEVfU09VUkNFX05BTUVTID0gWwogICAgImNhY2hlL2NoYWluX2ZvbGRzLm5weiIsCiAgICAiY2FjaGUvY2hhaW5zLm5weiIsCiAgICAiY2FjaGUvdHJhaW5fdGFyZ2V0cy5ucHoiLAogICAgImNhY2hlL3ZhbGlkX3RhcmdldHMubnB6IiwKICAgICJkYXRhc2V0cy90cmFpbi5wYXJxdWV0IiwKICAgICJkYXRhc2V0cy92YWxpZC5wYXJxdWV0IiwKXQpFWFBFQ1RFRF9QQVJFTlRfUkVTT0xVVElPTlMgPSB7CiAgICAiY2FjaGUiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL1d1bmRlckZ1bmRlci9jYWNoZSIsCiAgICAiZGF0YXNldHMiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL1d1bmRlckZ1bmRlci9kYXRhc2V0cyIsCn0KRVhQRUNURURfUkVHSVNURVJFRF9TT1VSQ0VTID0gewogICAgImNhY2hlL2NoYWluX2ZvbGRzLm5weiI6IHsKICAgICAgICAiYnl0ZXMiOiA0NjI3MTksCiAgICAgICAgImRldmljZSI6IDY0NTEyLAogICAgICAgICJpbm9kZSI6IDczNTQ4MzU0OSwKICAgICAgICAiaXNsaW5rIjogVHJ1ZSwKICAgICAgICAibXRpbWVfbnMiOiAxNzkyMjcyNDc2MDM0OTI4NTI0LAogICAgICAgICJwYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvZGF0YS9jYWNoZS9jaGFpbl9mb2xkcy5ucHoiLAogICAgICAgICJyZWdpc3RlcmVkX3NoYTI1NiI6ICJhMTEyZjgwOTI1OWY2MzFjMmVlN2MxMjQ4MGU0ZWMxZDRlZWYxOTdhOTgwZTdjZTIzYzcwN2Q4NDJkMDkyNTY3IiwKICAgICAgICAicmVzb2x2ZWQiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZWNlaXB0cy9jaGFpbl9mb2xkc19kZXJpdmVkLm5weiIsCiAgICB9LAogICAgImNhY2hlL2NoYWlucy5ucHoiOiB7CiAgICAgICAgImJ5dGVzIjogNjUwMTQyLAogICAgICAgICJkZXZpY2UiOiA2NDUxMiwKICAgICAgICAiaW5vZGUiOiA3NDI0MjgzNjAsCiAgICAgICAgImlzbGluayI6IFRydWUsCiAgICAgICAgIm10aW1lX25zIjogMTc5MDYzOTAwNDAzMjkwNTIzOSwKICAgICAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L2RhdGEvY2FjaGUvY2hhaW5zLm5weiIsCiAgICAgICAgInJlZ2lzdGVyZWRfc2hhMjU2IjogIjM5YWJhNDZmYWQ3YjcxNzgzMGU5MDY4N2UxYzRhNjk5MTc5NDg2NGNkMmNmNjIxYjBiYmYzNDFmMmQ5YTc3NjYiLAogICAgICAgICJyZXNvbHZlZCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvV3VuZGVyRnVuZGVyL2NhY2hlL2NoYWlucy5ucHoiLAogICAgfSwKICAgICJjYWNoZS90cmFpbl90YXJnZXRzLm5weiI6IHsKICAgICAgICAiYnl0ZXMiOiAxOTA5MzM1NzkwLAogICAgICAgICJkZXZpY2UiOiA2NDUxMiwKICAgICAgICAiaW5vZGUiOiA3NDI0Mjg1OTksCiAgICAgICAgImlzbGluayI6IFRydWUsCiAgICAgICAgIm10aW1lX25zIjogMTc5MDcwMjcyOTU2NTI1NjAzNiwKICAgICAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L2RhdGEvY2FjaGUvdHJhaW5fdGFyZ2V0cy5ucHoiLAogICAgICAgICJyZWdpc3RlcmVkX3NoYTI1NiI6ICJlYWQxYTA1ZTdhYTVhYzNkMWMxMGNkZTY1OTg2YmU5ZjQ4ZmQzNDJjYWRiOTUzMzA2MTFlNThjM2E4ZjgxN2U5IiwKICAgICAgICAicmVzb2x2ZWQiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL1d1bmRlckZ1bmRlci9jYWNoZS90cmFpbl90YXJnZXRzLm5weiIsCiAgICB9LAogICAgImNhY2hlL3ZhbGlkX3RhcmdldHMubnB6IjogewogICAgICAgICJieXRlcyI6IDMzNzE4NjE5OCwKICAgICAgICAiZGV2aWNlIjogNjQ1MTIsCiAgICAgICAgImlub2RlIjogNzQyNDI4MDc4LAogICAgICAgICJpc2xpbmsiOiBUcnVlLAogICAgICAgICJtdGltZV9ucyI6IDE3OTA2Mzg5NDQ0NTI1OTY5OTksCiAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9kYXRhL2NhY2hlL3ZhbGlkX3RhcmdldHMubnB6IiwKICAgICAgICAicmVnaXN0ZXJlZF9zaGEyNTYiOiAiZGJiN2ExYTQ4ZGQwYWY5ZmE1MjRhODBjMjY5NDAwOGE1NmNiOTFkNWRhZTgwZThkMGQ0OTg0YzBjNTYyMjk1NSIsCiAgICAgICAgInJlc29sdmVkIjogIi9ob21lL2dyaW5jaGVua29lZy9XdW5kZXJGdW5kZXIvY2FjaGUvdmFsaWRfdGFyZ2V0cy5ucHoiLAogICAgfSwKICAgICJkYXRhc2V0cy90cmFpbi5wYXJxdWV0IjogewogICAgICAgICJieXRlcyI6IDI5MjQwMjE5MDExLAogICAgICAgICJkZXZpY2UiOiA2NDUxMiwKICAgICAgICAiaW5vZGUiOiA3NDM4MzYyNDksCiAgICAgICAgImlzbGluayI6IEZhbHNlLAogICAgICAgICJtdGltZV9ucyI6IDE1Nzc4MzMyMDAwMDAwMDAwMDAsCiAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9kYXRhL2RhdGFzZXRzL3RyYWluLnBhcnF1ZXQiLAogICAgICAgICJyZWdpc3RlcmVkX3NoYTI1NiI6ICI0YTc3NTY0NDlhNGJjZjI3MmNmNzE1YmI0MDVhMjE1Mjk4NzA0MTYwNWY2YzFjY2U2NTAyNDNhYjJhYmY0ZTgxIiwKICAgICAgICAicmVzb2x2ZWQiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL1d1bmRlckZ1bmRlci9kYXRhc2V0cy90cmFpbi5wYXJxdWV0IiwKICAgIH0sCiAgICAiZGF0YXNldHMvdmFsaWQucGFycXVldCI6IEVYUEVDVEVEX1ZBTElEX1BBUlFVRVQsCn0KQVJNX01PREVTID0gewogICAgImNvbnRyb2wiOiAiYmFzZSIsCiAgICAiZW1hIjogImVtYSIsCiAgICAic3BhcnNlIjogInNwYXJzZSIsCiAgICAibmxsIjogImJhc2UiLAogICAgImFsbHRocmVlIjogImVtYV9zcGFyc2UiLAp9Ck5fR1JPVVBTID0gNDY4Ck5fUk9XUyA9IDIwXzAwMApOX09VVFBVVFMgPSAyClBSRURJQ1RJT05fU0hBUEUgPSAoTl9HUk9VUFMsIE5fUk9XUywgTl9PVVRQVVRTKQpQUkVESUNUSU9OX0RBVEFfQllURVMgPSBpbnQobnAucHJvZChQUkVESUNUSU9OX1NIQVBFKSkgKiBucC5kdHlwZSgiPGY0IikuaXRlbXNpemUKTUFYX1NNQUxMX0pTT05fQllURVMgPSAyXzAwMF8wMDAKTUFYX0VYUE9SVF9NQU5JRkVTVF9CWVRFUyA9IDRfMDAwXzAwMApNQVhfU1BMSVRfQllURVMgPSAyXzAwMF8wMDAKTUFYX01FTUJFUlNISVBfQVJDSElWRV9CWVRFUyA9IDY0XzAwMF8wMDAKTUFYX01FTUJFUlNISVBfQVJSQVlfQllURVMgPSA2NF8wMDBfMDAwCk1BWF9QUkVESUNUSU9OX0JZVEVTID0gUFJFRElDVElPTl9EQVRBX0JZVEVTICsgNDA5NgpCT09UU1RSQVBfUkVQTElDQVRFUyA9IDIwMDAKQk9PVFNUUkFQX1NFRUQgPSAyNzE4MjgKCgpjbGFzcyBTZXF1ZW5jZVJlYWRlcihQcm90b2NvbCk6CiAgICAiIiJQcm92aWRlIG9uZSBzb3VyY2UtcXVhbGlmaWVkIHNlcXVlbmNlIHdpdGggdGFyZ2V0cyBhbmQgc2NvcmVkIG1hc2suIiIiCgogICAgZGVmIHJlYWQoc2VsZiwgaW5kZXg6IGludCwgKiwgdGFyZ2V0czogYm9vbCA9IFRydWUsIG1hc2s6IGJvb2wgPSBUcnVlKSAtPiBBbnk6IC4uLgoKCkBkYXRhY2xhc3MoZnJvemVuPVRydWUpCmNsYXNzIE1ldHJpY0hlbHBlcnM6CiAgICAiIiJIb2xkIGNhbm9uaWNhbCBtZXRyaWMgYW5kIHNvdXJjZS1kYXRhIEFQSXMgYWZ0ZXIgZXhhY3Qgc291cmNlIHZlcmlmaWNhdGlvbi4iIiIKCiAgICBzZXF1ZW5jZV9maWxlOiB0eXBlCiAgICBnbG9iYWxfYWNjdW11bGF0b3I6IHR5cGUKICAgIGdsb2JhbF93cDogQW55CiAgICBwYWlyZWRfY29tcG9uZW50X21vbWVudHM6IEFueQogICAgcGFpcmVkX2NvbXBvbmVudF9ib290c3RyYXA6IEFueQogICAgbG9hZF9jYW1wYWlnbl9zcGxpdDogQW55CiAgICB3YXJtdXA6IGludAogICAgbWV0cmljX2lkOiBzdHIKCgpkZWYgX3N0YWJsZV9pZGVudGl0eShiZWZvcmU6IG9zLnN0YXRfcmVzdWx0LCBhZnRlcjogb3Muc3RhdF9yZXN1bHQpIC0+IHR1cGxlW2ludCwgaW50LCBpbnQsIGludF06CiAgICAiIiJSZXR1cm4gZmllbGRzIHVzZWQgdG8gZGV0ZWN0IGEgZmlsZSBjaGFuZ2luZyBkdXJpbmcgYSBib3VuZGVkIHJlYWQuIiIiCiAgICByZXR1cm4gYmVmb3JlLnN0X2RldiwgYmVmb3JlLnN0X2lubywgYmVmb3JlLnN0X3NpemUsIGJlZm9yZS5zdF9tdGltZV9ucwoKCmRlZiBfcmVhZF9ib3VuZGVkKHBhdGg6IFBhdGgsIGNhcDogaW50KSAtPiB0dXBsZVtieXRlcywgZGljdFtzdHIsIEFueV0sIHR1cGxlW2ludCwgaW50LCBpbnQsIGludF1dOgogICAgIiIiUmVhZCBhIGJvdW5kZWQgcmVndWxhciBmaWxlIHRocm91Z2ggYSBuby1mb2xsb3cgZGVzY3JpcHRvciBhbmQgcGluIGl0cyBieXRlcy4iIiIKICAgIGRlc2NyaXB0b3IgPSBvcy5vcGVuKHBhdGgsIG9zLk9fUkRPTkxZIHwgZ2V0YXR0cihvcywgIk9fTk9GT0xMT1ciLCAwKSkKICAgIHRyeToKICAgICAgICBiZWZvcmUgPSBvcy5mc3RhdChkZXNjcmlwdG9yKQogICAgICAgIGlmIG5vdCBzdGF0LlNfSVNSRUcoYmVmb3JlLnN0X21vZGUpIG9yIGJlZm9yZS5zdF9zaXplID4gY2FwOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYibWV0YWRhdGEgaXMgbm90IGEgcmVndWxhciBmaWxlIG9yIGV4Y2VlZHMge2NhcH0gYnl0ZXM6IHtwYXRofSIpCiAgICAgICAgY2h1bmtzOiBsaXN0W2J5dGVzXSA9IFtdCiAgICAgICAgZGlnZXN0ID0gaGFzaGxpYi5zaGEyNTYoKQogICAgICAgIHRvdGFsID0gMAogICAgICAgIHdoaWxlIGNodW5rIDo9IG9zLnJlYWQoZGVzY3JpcHRvciwgMSA8PCAyMCk6CiAgICAgICAgICAgIGlmIHRvdGFsICsgbGVuKGNodW5rKSA+IGNhcDoKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJtZXRhZGF0YSBncmV3IGJleW9uZCBpdHMge2NhcH0tYnl0ZSBib3VuZDoge3BhdGh9IikKICAgICAgICAgICAgY2h1bmtzLmFwcGVuZChjaHVuaykKICAgICAgICAgICAgZGlnZXN0LnVwZGF0ZShjaHVuaykKICAgICAgICAgICAgdG90YWwgKz0gbGVuKGNodW5rKQogICAgICAgIGFmdGVyID0gb3MuZnN0YXQoZGVzY3JpcHRvcikKICAgICAgICBsaW5rZWQgPSBwYXRoLmxzdGF0KCkKICAgICAgICBpZGVudGl0eSA9IF9zdGFibGVfaWRlbnRpdHkoYmVmb3JlLCBhZnRlcikKICAgICAgICBpZiAoCiAgICAgICAgICAgIGlkZW50aXR5ICE9IF9zdGFibGVfaWRlbnRpdHkoYWZ0ZXIsIGFmdGVyKQogICAgICAgICAgICBvciAobGlua2VkLnN0X2RldiwgbGlua2VkLnN0X2lubykgIT0gaWRlbnRpdHlbOjJdCiAgICAgICAgKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIm1ldGFkYXRhIGNoYW5nZWQgd2hpbGUgYmVpbmcgcmVhZDoge3BhdGh9IikKICAgICAgICBpZiB0b3RhbCAhPSBiZWZvcmUuc3Rfc2l6ZToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIm1ldGFkYXRhIGxlbmd0aCBjaGFuZ2VkIHdoaWxlIGJlaW5nIHJlYWQ6IHtwYXRofSIpCiAgICAgICAgcmF3ID0gYiIiLmpvaW4oY2h1bmtzKQogICAgICAgIHJldHVybiByYXcsIHsicGF0aCI6IHN0cihwYXRoKSwgImJ5dGVzIjogdG90YWwsICJzaGEyNTYiOiBkaWdlc3QuaGV4ZGlnZXN0KCl9LCBpZGVudGl0eQogICAgZmluYWxseToKICAgICAgICBvcy5jbG9zZShkZXNjcmlwdG9yKQoKCmRlZiBfcGluX2ZpbGUocGF0aDogUGF0aCwgY2FwOiBpbnQpIC0+IHR1cGxlW2RpY3Rbc3RyLCBBbnldLCB0dXBsZVtpbnQsIGludCwgaW50LCBpbnRdXToKICAgICIiIkhhc2ggb25lIHJlZ3VsYXIgbm8tZm9sbG93IGZpbGUgaW4gYm91bmRlZCBjaHVua3MuIiIiCiAgICBkZXNjcmlwdG9yID0gb3Mub3BlbihwYXRoLCBvcy5PX1JET05MWSB8IGdldGF0dHIob3MsICJPX05PRk9MTE9XIiwgMCkpCiAgICB0cnk6CiAgICAgICAgYmVmb3JlID0gb3MuZnN0YXQoZGVzY3JpcHRvcikKICAgICAgICBpZiBub3Qgc3RhdC5TX0lTUkVHKGJlZm9yZS5zdF9tb2RlKSBvciBiZWZvcmUuc3Rfc2l6ZSA+IGNhcDoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImZpbGUgaXMgbm90IHJlZ3VsYXIgb3IgZXhjZWVkcyB7Y2FwfSBieXRlczoge3BhdGh9IikKICAgICAgICBkaWdlc3QgPSBoYXNobGliLnNoYTI1NigpCiAgICAgICAgdG90YWwgPSAwCiAgICAgICAgd2hpbGUgY2h1bmsgOj0gb3MucmVhZChkZXNjcmlwdG9yLCAxIDw8IDIwKToKICAgICAgICAgICAgaWYgdG90YWwgKyBsZW4oY2h1bmspID4gY2FwOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImZpbGUgZ3JldyBiZXlvbmQgaXRzIHtjYXB9LWJ5dGUgYm91bmQ6IHtwYXRofSIpCiAgICAgICAgICAgIGRpZ2VzdC51cGRhdGUoY2h1bmspCiAgICAgICAgICAgIHRvdGFsICs9IGxlbihjaHVuaykKICAgICAgICBhZnRlciA9IG9zLmZzdGF0KGRlc2NyaXB0b3IpCiAgICAgICAgbGlua2VkID0gcGF0aC5sc3RhdCgpCiAgICAgICAgaWRlbnRpdHkgPSBfc3RhYmxlX2lkZW50aXR5KGJlZm9yZSwgYWZ0ZXIpCiAgICAgICAgaWYgKAogICAgICAgICAgICBpZGVudGl0eSAhPSBfc3RhYmxlX2lkZW50aXR5KGFmdGVyLCBhZnRlcikKICAgICAgICAgICAgb3IgKGxpbmtlZC5zdF9kZXYsIGxpbmtlZC5zdF9pbm8pICE9IGlkZW50aXR5WzoyXQogICAgICAgICk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJmaWxlIGNoYW5nZWQgd2hpbGUgYmVpbmcgaGFzaGVkOiB7cGF0aH0iKQogICAgICAgIGlmIHRvdGFsICE9IGJlZm9yZS5zdF9zaXplOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZmlsZSBsZW5ndGggY2hhbmdlZCB3aGlsZSBiZWluZyBoYXNoZWQ6IHtwYXRofSIpCiAgICAgICAgcmV0dXJuIHsicGF0aCI6IHN0cihwYXRoKSwgImJ5dGVzIjogdG90YWwsICJzaGEyNTYiOiBkaWdlc3QuaGV4ZGlnZXN0KCl9LCBpZGVudGl0eQogICAgZmluYWxseToKICAgICAgICBvcy5jbG9zZShkZXNjcmlwdG9yKQoKCmRlZiBfc2FmZV9jaGlsZChyb290OiBQYXRoLCByZWxhdGl2ZTogc3RyKSAtPiBQYXRoOgogICAgIiIiUmVzb2x2ZSBhIGJ1bmRsZSBjaGlsZCB3aGlsZSByZWZ1c2luZyB0cmF2ZXJzYWwgYW5kIHN5bWxpbmsgY29tcG9uZW50cy4iIiIKICAgIGNoaWxkID0gUHVyZVBvc2l4UGF0aChyZWxhdGl2ZSkKICAgIGlmIGNoaWxkLmlzX2Fic29sdXRlKCkgb3Igbm90IGNoaWxkLnBhcnRzIG9yIGFueShwYXJ0IGluICgiLiIsICIuLiIpIGZvciBwYXJ0IGluIGNoaWxkLnBhcnRzKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYidW5zYWZlIHJlbGF0aXZlIHBhdGg6IHtyZWxhdGl2ZX0iKQogICAgY2FuZGlkYXRlID0gcm9vdAogICAgZm9yIHBhcnQgaW4gY2hpbGQucGFydHNbOi0xXToKICAgICAgICBjYW5kaWRhdGUgPSBjYW5kaWRhdGUgLyBwYXJ0CiAgICAgICAgaWYgY2FuZGlkYXRlLmlzX3N5bWxpbmsoKSBvciBub3QgY2FuZGlkYXRlLmlzX2RpcigpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYidW5zYWZlIGJ1bmRsZSBkaXJlY3Rvcnk6IHtjYW5kaWRhdGV9IikKICAgIHJldHVybiBjYW5kaWRhdGUgLyBjaGlsZC5wYXJ0c1stMV0KCgpkZWYgX3ByZWZsaWdodF9zcGxpdF9pbnB1dHMoCiAgICBzcGxpdF9wYXRoOiBQYXRoLAogICAgZGF0YV9yb290OiBQYXRoLAogICAgcmVnaXN0ZXJlZF9zb3VyY2VzOiBkaWN0W3N0ciwgZGljdFtzdHIsIEFueV1dLAopIC0+IE5vbmU6CiAgICAiIiJCb3VuZCBtZW1iZXJzaGlwIGFuZCBjYWNoZSBpbnB1dHMgYmVmb3JlIHRoZSBjYW5vbmljYWwgc3BsaXQgdmFsaWRhdG9yIG9wZW5zIHRoZW0uIiIiCiAgICByYXcsIF8sIF8gPSBfcmVhZF9ib3VuZGVkKHNwbGl0X3BhdGgsIE1BWF9TUExJVF9CWVRFUykKICAgIGlmIGhhc2hsaWIuc2hhMjU2KHJhdykuaGV4ZGlnZXN0KCkgIT0gRVhQRUNURURfU1BMSVRfU0hBMjU2OgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImNhbXBhaWduIHNwbGl0IGJ5dGVzIGRpZmZlciBmcm9tIHRoZSBmaXhlZCBjMDlmIEYwIG1hbmlmZXN0IikKICAgIG1hbmlmZXN0ID0ganNvbi5sb2FkcyhyYXcpCiAgICBzb3VyY2VzID0gbWFuaWZlc3QuZ2V0KCJzb3VyY2VzIikKICAgIGlmIG5vdCBpc2luc3RhbmNlKHNvdXJjZXMsIGRpY3QpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImNhbXBhaWduIHNwbGl0IHNvdXJjZSBpZGVudGl0aWVzIGFyZSBtYWxmb3JtZWQiKQogICAgZm9yIHJlbGF0aXZlLCBpZGVudGl0eSBpbiBzb3VyY2VzLml0ZW1zKCk6CiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2UocmVsYXRpdmUsIHN0cikgb3Igbm90IGlzaW5zdGFuY2UoaWRlbnRpdHksIGRpY3QpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJjYW1wYWlnbiBzcGxpdCBzb3VyY2UgaWRlbnRpdHkgaXMgbWFsZm9ybWVkIikKICAgICAgICBpc19yZWdpc3RlcmVkID0gcmVsYXRpdmUgaW4gcmVnaXN0ZXJlZF9zb3VyY2VzCiAgICAgICAgaWYgaXNfcmVnaXN0ZXJlZDoKICAgICAgICAgICAgcmVnaXN0ZXJlZCA9IHJlZ2lzdGVyZWRfc291cmNlc1tyZWxhdGl2ZV0KICAgICAgICAgICAgb2JzZXJ2YXRpb24gPSBvYnNlcnZlX3JlZ2lzdGVyZWRfc291cmNlKGRhdGFfcm9vdCwgcmVsYXRpdmUsIHJlZ2lzdGVyZWQpCiAgICAgICAgICAgIHNvdXJjZV9wYXRoID0gUGF0aChzdHIocmVnaXN0ZXJlZFsicGF0aCJdKSkKICAgICAgICAgICAgaWYgKAogICAgICAgICAgICAgICAgaWRlbnRpdHkuZ2V0KCJieXRlcyIpICE9IG9ic2VydmF0aW9uWyJieXRlcyJdCiAgICAgICAgICAgICAgICBvciBpZGVudGl0eS5nZXQoIm10aW1lX25zIikgIT0gb2JzZXJ2YXRpb25bIm10aW1lX25zIl0KICAgICAgICAgICAgKToKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJzcGxpdCBzb3VyY2Ugc3RhdCBkaWZmZXJzIGZyb20gYWRtaXNzaW9uOiB7cmVsYXRpdmV9IikKICAgICAgICBlbHNlOgogICAgICAgICAgICBzb3VyY2VfcGF0aCA9IF9zYWZlX2NoaWxkKGRhdGFfcm9vdCwgcmVsYXRpdmUpCiAgICAgICAgbWV0YWRhdGEgPSBzb3VyY2VfcGF0aC5zdGF0KCkgaWYgaXNfcmVnaXN0ZXJlZCBlbHNlIHNvdXJjZV9wYXRoLmxzdGF0KCkKICAgICAgICBpZiBub3Qgc3RhdC5TX0lTUkVHKG1ldGFkYXRhLnN0X21vZGUpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiY2FtcGFpZ24gc291cmNlIGlzIG5vdCBhIHJlZ3VsYXIgZmlsZToge3JlbGF0aXZlfSIpCiAgICAgICAgaWYgbm90IGlzX3JlZ2lzdGVyZWQgYW5kICJpbmRleF9zaGEyNTYiIGluIGlkZW50aXR5IGFuZCBtZXRhZGF0YS5zdF9zaXplID4gNjRfMDAwXzAwMDoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImluZGV4ZWQgY2FtcGFpZ24gc291cmNlIGV4Y2VlZHMgdGhlIDY0IE1CIHByZWZsaWdodCBjYXA6IHtyZWxhdGl2ZX0iKQogICAgICAgIGlmIG5vdCBpc19yZWdpc3RlcmVkIGFuZCAic2hhMjU2IiBpbiBpZGVudGl0eSBhbmQgbWV0YWRhdGEuc3Rfc2l6ZSA+IDY0XzAwMF8wMDA6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJoYXNoZWQgY2FtcGFpZ24gc291cmNlIGV4Y2VlZHMgdGhlIDY0IE1CIHByZWZsaWdodCBjYXA6IHtyZWxhdGl2ZX0iKQogICAgICAgIGlmICJpbmRleF9zaGEyNTYiIGluIGlkZW50aXR5OgogICAgICAgICAgICBfcHJlZmxpZ2h0X25wel9tZW1iZXJzKHNvdXJjZV9wYXRoLCB7InJvd19ncm91cCIsICJzZXFfaXgifSwgMikKICAgIGZvciByZWxhdGl2ZSBpbiAoImNhY2hlL2NoYWlucy5ucHoiLCAiY2FjaGUvY2hhaW5fZm9sZHMubnB6Iik6CiAgICAgICAgaWYgcmVsYXRpdmUgaW4gcmVnaXN0ZXJlZF9zb3VyY2VzOgogICAgICAgICAgICBvYnNlcnZlX3JlZ2lzdGVyZWRfc291cmNlKGRhdGFfcm9vdCwgcmVsYXRpdmUsIHJlZ2lzdGVyZWRfc291cmNlc1tyZWxhdGl2ZV0pCiAgICAgICAgICAgIGNhY2hlX3BhdGggPSBQYXRoKHN0cihyZWdpc3RlcmVkX3NvdXJjZXNbcmVsYXRpdmVdWyJwYXRoIl0pKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGNhY2hlX3BhdGggPSBfc2FmZV9jaGlsZChkYXRhX3Jvb3QsIHJlbGF0aXZlKQogICAgICAgIG1ldGFkYXRhID0gY2FjaGVfcGF0aC5zdGF0KCkKICAgICAgICBpZiBub3Qgc3RhdC5TX0lTUkVHKG1ldGFkYXRhLnN0X21vZGUpIG9yIG1ldGFkYXRhLnN0X3NpemUgPiBNQVhfTUVNQkVSU0hJUF9BUkNISVZFX0JZVEVTOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiY2FtcGFpZ24gbWVtYmVyc2hpcCBjYWNoZSBpcyBub3QgYSBib3VuZGVkIHJlZ3VsYXIgZmlsZToge3JlbGF0aXZlfSIpCiAgICAgICAgX3ByZWZsaWdodF9ucHpfbWVtYmVycyhjYWNoZV9wYXRoLCBzZXQoKSwgTUFYX01FTUJFUlNISVBfQVJSQVlfQllURVMpCgoKZGVmIF9wcmVmbGlnaHRfbnB6X21lbWJlcnMocGF0aDogUGF0aCwgcmVxdWlyZWQ6IHNldFtzdHJdLCB0b3RhbF9jYXA6IGludCkgLT4gTm9uZToKICAgICIiIkJvdW5kIFpJUCBtZW1iZXIgbWV0YWRhdGEgYmVmb3JlIE51bVB5IGV4cGFuZHMgdGhlIGNhbm9uaWNhbCBtZW1iZXJzaGlwIGFycmF5cy4iIiIKICAgIHdpdGggemlwZmlsZS5aaXBGaWxlKHBhdGgpIGFzIGFyY2hpdmU6CiAgICAgICAgbWVtYmVycyA9IFtpdGVtIGZvciBpdGVtIGluIGFyY2hpdmUuaW5mb2xpc3QoKSBpZiBub3QgaXRlbS5pc19kaXIoKV0KICAgIGlmIGxlbihtZW1iZXJzKSA+IDMyOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJtZW1iZXJzaGlwIGFyY2hpdmUgY29udGFpbnMgdG9vIG1hbnkgYXJyYXlzOiB7cGF0aH0iKQogICAgbmFtZXMgPSB7UGF0aChpdGVtLmZpbGVuYW1lKS5uYW1lLnJlbW92ZXN1ZmZpeCgiLm5weSIpIGZvciBpdGVtIGluIG1lbWJlcnN9CiAgICBpZiBub3QgcmVxdWlyZWQuaXNzdWJzZXQobmFtZXMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJtZW1iZXJzaGlwIGFyY2hpdmUgaXMgbWlzc2luZyByZXF1aXJlZCBpbmRleCBhcnJheXM6IHtwYXRofSIpCiAgICBleHBhbmRlZF90b3RhbCA9IDAKICAgIGZvciBpdGVtIGluIG1lbWJlcnM6CiAgICAgICAgaWYgaXRlbS5mbGFnX2JpdHMgJiAweDEgb3IgaXRlbS5maWxlX3NpemUgPiB0b3RhbF9jYXA6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJtZW1iZXJzaGlwIGFyY2hpdmUgY29udGFpbnMgYW4gdW5zYWZlIGV4cGFuZGVkIG1lbWJlcjoge3BhdGh9IikKICAgICAgICBleHBhbmRlZF90b3RhbCArPSBpdGVtLmZpbGVfc2l6ZQogICAgaWYgZXhwYW5kZWRfdG90YWwgPiB0b3RhbF9jYXA6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIm1lbWJlcnNoaXAgYXJjaGl2ZSBleGNlZWRzIHRoZSBleHBhbmRlZC1zaXplIGJvdW5kOiB7cGF0aH0iKQoKCmRlZiBvYnNlcnZlX3JlZ2lzdGVyZWRfc291cmNlKAogICAgZGF0YV9yb290OiBQYXRoLCByZWxhdGl2ZTogc3RyLCBleHBlY3RlZF9pZGVudGl0eTogZGljdFtzdHIsIEFueV0KKSAtPiBkaWN0W3N0ciwgQW55XToKICAgICIiIlN0YXQgb25lIGV4YWN0IGFkbWl0dGVkIGxlYWYsIGFsbG93aW5nIG9ubHkgaXRzIHJlZ2lzdGVyZWQgcmVzb2x2ZWQgdGFyZ2V0LiIiIgogICAgcGF0aCA9IFBhdGgoc3RyKGV4cGVjdGVkX2lkZW50aXR5LmdldCgicGF0aCIsICIiKSkpCiAgICBleHBlY3RlZF9yZXNvbHZlZCA9IFBhdGgoc3RyKGV4cGVjdGVkX2lkZW50aXR5LmdldCgicmVzb2x2ZWQiLCAiIikpKQogICAgaWYgcGF0aCAhPSBkYXRhX3Jvb3QgLyBQdXJlUG9zaXhQYXRoKHJlbGF0aXZlKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYicmVnaXN0ZXJlZCBzb3VyY2UgcGF0aCBkaWZmZXJzIGZyb20gdGhlIGZpeGVkIGRhdGEgcm9vdDoge3JlbGF0aXZlfSIpCiAgICBwYXJlbnQgPSBwYXRoLnBhcmVudAogICAgcGFyZW50X2JlZm9yZSA9IHBhcmVudC5zdGF0KCkKICAgIHBhcmVudF9yZXNvbHZlZF9iZWZvcmUgPSBwYXJlbnQucmVzb2x2ZShzdHJpY3Q9VHJ1ZSkKICAgIGlmIG5vdCBzdGF0LlNfSVNESVIocGFyZW50X2JlZm9yZS5zdF9tb2RlKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYicmVnaXN0ZXJlZCBzb3VyY2UgcGFyZW50IGlzIG5vdCBhIGRpcmVjdG9yeToge3JlbGF0aXZlfSIpCiAgICBpZiBwYXRoLnJlc29sdmUoc3RyaWN0PVRydWUpICE9IGV4cGVjdGVkX3Jlc29sdmVkOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJyZWdpc3RlcmVkIHNvdXJjZSBkaWZmZXJzIGZyb20gaXRzIGV4YWN0IGFsaWFzIHRhcmdldDoge3JlbGF0aXZlfSIpCiAgICBsaW5rZWRfYmVmb3JlID0gcGF0aC5sc3RhdCgpCiAgICBpZiBzdGF0LlNfSVNMTksobGlua2VkX2JlZm9yZS5zdF9tb2RlKSBpcyBub3QgZXhwZWN0ZWRfaWRlbnRpdHlbImlzbGluayJdOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJyZWdpc3RlcmVkIHNvdXJjZSBzeW1saW5rIHN0YXR1cyBkaWZmZXJzIGZyb20gYWRtaXNzaW9uOiB7cmVsYXRpdmV9IikKICAgIHNvdXJjZV9zdGF0ID0gcGF0aC5zdGF0KCkKICAgIGxpbmtlZF9hZnRlciA9IHBhdGgubHN0YXQoKQogICAgc291cmNlX2FmdGVyID0gcGF0aC5zdGF0KCkKICAgIHBhcmVudF9hZnRlciA9IHBhcmVudC5zdGF0KCkKICAgIHJlc29sdmVkX2FmdGVyID0gcGF0aC5yZXNvbHZlKHN0cmljdD1UcnVlKQogICAgcGFyZW50X3Jlc29sdmVkX2FmdGVyID0gcGFyZW50LnJlc29sdmUoc3RyaWN0PVRydWUpCiAgICBpZiAoCiAgICAgICAgbm90IHN0YXQuU19JU1JFRyhzb3VyY2Vfc3RhdC5zdF9tb2RlKQogICAgICAgIG9yIHN0YXQuU19JU0xOSyhsaW5rZWRfYWZ0ZXIuc3RfbW9kZSkgaXMgbm90IGV4cGVjdGVkX2lkZW50aXR5WyJpc2xpbmsiXQogICAgICAgIG9yIChzb3VyY2Vfc3RhdC5zdF9kZXYsIHNvdXJjZV9zdGF0LnN0X2lubywgc291cmNlX3N0YXQuc3Rfc2l6ZSwgc291cmNlX3N0YXQuc3RfbXRpbWVfbnMpCiAgICAgICAgIT0gKAogICAgICAgICAgICBleHBlY3RlZF9pZGVudGl0eVsiZGV2aWNlIl0sCiAgICAgICAgICAgIGV4cGVjdGVkX2lkZW50aXR5WyJpbm9kZSJdLAogICAgICAgICAgICBleHBlY3RlZF9pZGVudGl0eVsiYnl0ZXMiXSwKICAgICAgICAgICAgZXhwZWN0ZWRfaWRlbnRpdHlbIm10aW1lX25zIl0sCiAgICAgICAgKQogICAgICAgIG9yIChzb3VyY2Vfc3RhdC5zdF9kZXYsIHNvdXJjZV9zdGF0LnN0X2lubywgc291cmNlX3N0YXQuc3Rfc2l6ZSwgc291cmNlX3N0YXQuc3RfbXRpbWVfbnMpCiAgICAgICAgIT0gKAogICAgICAgICAgICBzb3VyY2VfYWZ0ZXIuc3RfZGV2LAogICAgICAgICAgICBzb3VyY2VfYWZ0ZXIuc3RfaW5vLAogICAgICAgICAgICBzb3VyY2VfYWZ0ZXIuc3Rfc2l6ZSwKICAgICAgICAgICAgc291cmNlX2FmdGVyLnN0X210aW1lX25zLAogICAgICAgICkKICAgICAgICBvciAobGlua2VkX2JlZm9yZS5zdF9kZXYsIGxpbmtlZF9iZWZvcmUuc3RfaW5vLCBsaW5rZWRfYmVmb3JlLnN0X210aW1lX25zKQogICAgICAgICE9IChsaW5rZWRfYWZ0ZXIuc3RfZGV2LCBsaW5rZWRfYWZ0ZXIuc3RfaW5vLCBsaW5rZWRfYWZ0ZXIuc3RfbXRpbWVfbnMpCiAgICAgICAgb3IgKHBhcmVudF9iZWZvcmUuc3RfZGV2LCBwYXJlbnRfYmVmb3JlLnN0X2lubykKICAgICAgICAhPSAocGFyZW50X2FmdGVyLnN0X2RldiwgcGFyZW50X2FmdGVyLnN0X2lubykKICAgICAgICBvciByZXNvbHZlZF9hZnRlciAhPSBleHBlY3RlZF9yZXNvbHZlZAogICAgICAgIG9yIHBhcmVudF9yZXNvbHZlZF9hZnRlciAhPSBwYXJlbnRfcmVzb2x2ZWRfYmVmb3JlCiAgICApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoCiAgICAgICAgICAgIGYicmVnaXN0ZXJlZCBzb3VyY2UgaWRlbnRpdHkgY2hhbmdlZCBvciBkaWZmZXJzIGZyb20gYWRtaXNzaW9uOiB7cmVsYXRpdmV9IgogICAgICAgICkKICAgIHJldHVybiB7CiAgICAgICAgInBhdGgiOiBzdHIocGF0aCksCiAgICAgICAgInJlc29sdmVkIjogc3RyKGV4cGVjdGVkX3Jlc29sdmVkKSwKICAgICAgICAiYnl0ZXMiOiBzb3VyY2Vfc3RhdC5zdF9zaXplLAogICAgICAgICJkZXZpY2UiOiBzb3VyY2Vfc3RhdC5zdF9kZXYsCiAgICAgICAgImlub2RlIjogc291cmNlX3N0YXQuc3RfaW5vLAogICAgICAgICJpc2xpbmsiOiBib29sKGV4cGVjdGVkX2lkZW50aXR5WyJpc2xpbmsiXSksCiAgICAgICAgIm10aW1lX25zIjogc291cmNlX3N0YXQuc3RfbXRpbWVfbnMsCiAgICAgICAgInBhcmVudCI6IHsKICAgICAgICAgICAgInBhdGgiOiBzdHIocGFyZW50KSwKICAgICAgICAgICAgInJlc29sdmVkIjogc3RyKHBhcmVudF9yZXNvbHZlZF9hZnRlciksCiAgICAgICAgICAgICJkZXZpY2UiOiBwYXJlbnRfYWZ0ZXIuc3RfZGV2LAogICAgICAgICAgICAiaW5vZGUiOiBwYXJlbnRfYWZ0ZXIuc3RfaW5vLAogICAgICAgIH0sCiAgICB9CgoKZGVmIF9yZWFkX2pzb24oCiAgICBwYXRoOiBQYXRoLCBjYXA6IGludAopIC0+IHR1cGxlW2RpY3Rbc3RyLCBBbnldLCBkaWN0W3N0ciwgQW55XSwgdHVwbGVbaW50LCBpbnQsIGludCwgaW50XV06CiAgICAiIiJQYXJzZSBhIGJvdW5kZWQgSlNPTiBvYmplY3QgYW5kIHJldGFpbiBpdHMgZXhhY3QgZmlsZSBwaW4uIiIiCiAgICByYXcsIHBpbiwgaWRlbnRpdHkgPSBfcmVhZF9ib3VuZGVkKHBhdGgsIGNhcCkKICAgIHZhbHVlID0ganNvbi5sb2FkcyhyYXcpCiAgICBpZiBub3QgaXNpbnN0YW5jZSh2YWx1ZSwgZGljdCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImV4cGVjdGVkIGEgSlNPTiBvYmplY3Q6IHtwYXRofSIpCiAgICByZXR1cm4gdmFsdWUsIHBpbiwgaWRlbnRpdHkKCgpkZWYgX3ZlcmlmeV9zb3VyY2VfcGlucyhzb3VyY2Vfcm9vdDogUGF0aCkgLT4gTm9uZToKICAgICIiIlZlcmlmeSB0aGUgZGlyZWN0IHNvdXJjZSBtb2R1bGVzIHVzZWQgYnkgdGhpcyBzY29yZXIgYmVmb3JlIGltcG9ydGluZyB0aGVtLiIiIgogICAgaWYgc291cmNlX3Jvb3QuaXNfc3ltbGluaygpIG9yIG5vdCBzb3VyY2Vfcm9vdC5pc19kaXIoKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJzb3VyY2Ugcm9vdCBtdXN0IGJlIGEgcmVhbCBkaXJlY3RvcnkiKQogICAgaWYgbm90IHN5cy5kb250X3dyaXRlX2J5dGVjb2RlOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibGF1bmNoIHRoaXMgc2NvcmVyIHdpdGggUHl0aG9uIC1CIHRvIGF2b2lkIHNvdXJjZS10cmVlIGJ5dGVjb2RlIikKICAgIGZvciByZWxhdGl2ZSwgKGV4cGVjdGVkX2J5dGVzLCBleHBlY3RlZF9zaGEyNTYpIGluIFNPVVJDRV9QSU5TLml0ZW1zKCk6CiAgICAgICAgcGF0aCA9IF9zYWZlX2NoaWxkKHNvdXJjZV9yb290LCByZWxhdGl2ZSkKICAgICAgICBwaW4sIF8gPSBfcGluX2ZpbGUocGF0aCwgNTBfMDAwXzAwMCkKICAgICAgICBpZiAocGluWyJieXRlcyJdLCBwaW5bInNoYTI1NiJdKSAhPSAoZXhwZWN0ZWRfYnl0ZXMsIGV4cGVjdGVkX3NoYTI1Nik6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJzb3VyY2UgcGluIGRpZmZlcnM6IHtyZWxhdGl2ZX0iKQoKCmRlZiBfbG9hZF9oZWxwZXJzKHNvdXJjZV9yb290OiBQYXRoLCAqLCBwb3J0YWJsZV9yZWdpc3RyeTogUGF0aCB8IE5vbmUgPSBOb25lKSAtPiBNZXRyaWNIZWxwZXJzOgogICAgIiIiSW1wb3J0IG9ubHkgZXhhY3Qgc291cmNlLXBpbm5lZCBkYXRhLCBzY29yaW5nLCBhbmQgcG9ydGFibGUtaW5wdXQgaGVscGVycy4iIiIKICAgIHJlZ2lzdHJ5X2VudiA9IG9zLmVudmlyb24uZ2V0KCJXTk5fUE9SVEFCTEVfREFUQV9NQU5JRkVTVCIpCiAgICBpZiBwb3J0YWJsZV9yZWdpc3RyeSBpcyBOb25lOgogICAgICAgIGlmIHJlZ2lzdHJ5X2VudjoKICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJhIHBvcnRhYmxlIHJlZ2lzdHJ5IHBhdGggbXVzdCBiZSBleHBsaWNpdGx5IGJvdW5kIGJ5IHRoZSBzY29yZXIiKQogICAgZWxzZToKICAgICAgICBleHBlY3RlZF9yZWdpc3RyeSA9IF9zYWZlX2NoaWxkKAogICAgICAgICAgICBzb3VyY2Vfcm9vdC5wYXJlbnQsIEVYUEVDVEVEX1BPUlRBQkxFX1JFR0lTVFJZX1JFTEFUSVZFX1BBVEgKICAgICAgICApCiAgICAgICAgaWYgcG9ydGFibGVfcmVnaXN0cnkgIT0gZXhwZWN0ZWRfcmVnaXN0cnkgb3IgcmVnaXN0cnlfZW52ICE9IHN0cihleHBlY3RlZF9yZWdpc3RyeSk6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigicG9ydGFibGUgcmVnaXN0cnkgZW52aXJvbm1lbnQgZGlmZmVycyBmcm9tIHRoZSBmaXhlZCB3b3JrZXIgcGF0aCIpCiAgICAgICAgcmVnaXN0cnlfcGluLCBfID0gX3Bpbl9maWxlKGV4cGVjdGVkX3JlZ2lzdHJ5LCAyXzAwMF8wMDApCiAgICAgICAgaWYgKHJlZ2lzdHJ5X3BpblsiYnl0ZXMiXSwgcmVnaXN0cnlfcGluWyJzaGEyNTYiXSkgIT0gKAogICAgICAgICAgICBFWFBFQ1RFRF9QT1JUQUJMRV9SRUdJU1RSWV9CWVRFUywKICAgICAgICAgICAgRVhQRUNURURfUE9SVEFCTEVfUkVHSVNUUllfU0hBMjU2LAogICAgICAgICk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoInBvcnRhYmxlIHNvdXJjZSByZWdpc3RyeSBkaWZmZXJzIGZyb20gdGhlIGZpeGVkIGFkbWl0dGVkIHJlZ2lzdHJ5IikKICAgIF92ZXJpZnlfc291cmNlX3BpbnMoc291cmNlX3Jvb3QpCiAgICBpZiBhbnkobmFtZSA9PSAid25uIiBvciBuYW1lLnN0YXJ0c3dpdGgoIndubi4iKSBmb3IgbmFtZSBpbiBzeXMubW9kdWxlcyk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJXTk4gbW9kdWxlcyB3ZXJlIHByZWxvYWRlZCBiZWZvcmUgc291cmNlIHBpbiB2ZXJpZmljYXRpb24iKQogICAgc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihzb3VyY2Vfcm9vdCAvICJzcmMiKSkKICAgIHByZXZpb3VzX2NhY2hlX3ByZWZpeCA9IHN5cy5weWNhY2hlX3ByZWZpeAogICAgY2FjaGVfcHJlZml4ID0gb3MucGF0aC5qb2luKG9zLmRldm51bGwsICJ3dW5kZXItZmlyc3QtY2VsbC1uby1ieXRlY29kZSIpCiAgICBpZiBQYXRoKGNhY2hlX3ByZWZpeCkuZXhpc3RzKCk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJpc29sYXRlZCBieXRlY29kZSBjYWNoZSBwYXRoIHVuZXhwZWN0ZWRseSBleGlzdHMiKQogICAgc3lzLnB5Y2FjaGVfcHJlZml4ID0gY2FjaGVfcHJlZml4CiAgICB0cnk6CiAgICAgICAgZGF0YSA9IGltcG9ydGxpYi5pbXBvcnRfbW9kdWxlKCJ3bm4uZGF0YSIpCiAgICAgICAgZ21ldHJpYyA9IGltcG9ydGxpYi5pbXBvcnRfbW9kdWxlKCJ3bm4uZ21ldHJpYyIpCiAgICAgICAgZXZhbHVhdGlvbiA9IGltcG9ydGxpYi5pbXBvcnRfbW9kdWxlKCJ3bm4uZXZhbHVhdGlvbiIpCiAgICAgICAgcG9ydGFiaWxpdHkgPSBpbXBvcnRsaWIuaW1wb3J0X21vZHVsZSgid25uLnBvcnRhYmlsaXR5IikKICAgICAgICBzcGxpdHMgPSBpbXBvcnRsaWIuaW1wb3J0X21vZHVsZSgid25uLnNwbGl0cyIpCiAgICAgICAgcGFja2FnZSA9IGltcG9ydGxpYi5pbXBvcnRfbW9kdWxlKCJ3bm4iKQogICAgZmluYWxseToKICAgICAgICBzeXMucHljYWNoZV9wcmVmaXggPSBwcmV2aW91c19jYWNoZV9wcmVmaXgKICAgIG1vZHVsZXMgPSB7CiAgICAgICAgInNyYy93bm4vX19pbml0X18ucHkiOiBwYWNrYWdlLAogICAgICAgICJzcmMvd25uL2RhdGEucHkiOiBkYXRhLAogICAgICAgICJzcmMvd25uL2dtZXRyaWMucHkiOiBnbWV0cmljLAogICAgICAgICJzcmMvd25uL2V2YWx1YXRpb24ucHkiOiBldmFsdWF0aW9uLAogICAgICAgICJzcmMvd25uL3BvcnRhYmlsaXR5LnB5IjogcG9ydGFiaWxpdHksCiAgICAgICAgInNyYy93bm4vc3BsaXRzLnB5Ijogc3BsaXRzLAogICAgfQogICAgZm9yIHJlbGF0aXZlLCBtb2R1bGUgaW4gbW9kdWxlcy5pdGVtcygpOgogICAgICAgIG9yaWdpbiA9IFBhdGgobW9kdWxlLl9fZmlsZV9fKS5yZXNvbHZlKHN0cmljdD1UcnVlKQogICAgICAgIGV4cGVjdGVkID0gKHNvdXJjZV9yb290IC8gcmVsYXRpdmUpLnJlc29sdmUoc3RyaWN0PVRydWUpCiAgICAgICAgaWYgb3JpZ2luICE9IGV4cGVjdGVkOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiaW1wb3J0IG9yaWdpbiBkaWZmZXJzIGZvciB7cmVsYXRpdmV9OiB7b3JpZ2lufSIpCiAgICAgICAgcGluLCBfID0gX3Bpbl9maWxlKG9yaWdpbiwgNTBfMDAwXzAwMCkKICAgICAgICBpZiAocGluWyJieXRlcyJdLCBwaW5bInNoYTI1NiJdKSAhPSBTT1VSQ0VfUElOU1tyZWxhdGl2ZV06CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJpbXBvcnRlZCBzb3VyY2UgY2hhbmdlZCBmb3Ige3JlbGF0aXZlfSIpCiAgICBpZiBnbWV0cmljLldBUk1VUCAhPSA5OSBvciBnbWV0cmljLk1FVFJJQ19JRCAhPSAiZ2xvYmFsX3dwX2NsaXAyX3YxIjoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJvZmZpY2lhbCBtZXRyaWMgaWRlbnRpdHkgZGlmZmVycyBmcm9tIHRoZSBmaXhlZCBwYW5lbCIpCiAgICByZXR1cm4gTWV0cmljSGVscGVycygKICAgICAgICBzZXF1ZW5jZV9maWxlPWRhdGEuU2VxdWVuY2VGaWxlLAogICAgICAgIGdsb2JhbF9hY2N1bXVsYXRvcj1nbWV0cmljLkdsb2JhbEFjY3VtdWxhdG9yLAogICAgICAgIGdsb2JhbF93cD1nbWV0cmljLmdsb2JhbF93cCwKICAgICAgICBwYWlyZWRfY29tcG9uZW50X21vbWVudHM9ZXZhbHVhdGlvbi5wYWlyZWRfY29tcG9uZW50X21vbWVudHMsCiAgICAgICAgcGFpcmVkX2NvbXBvbmVudF9ib290c3RyYXA9ZXZhbHVhdGlvbi5wYWlyZWRfY29tcG9uZW50X2Jvb3RzdHJhcCwKICAgICAgICBsb2FkX2NhbXBhaWduX3NwbGl0PXNwbGl0cy5sb2FkX2NhbXBhaWduX3NwbGl0LAogICAgICAgIHdhcm11cD1nbWV0cmljLldBUk1VUCwKICAgICAgICBtZXRyaWNfaWQ9Z21ldHJpYy5NRVRSSUNfSUQsCiAgICApCgoKZGVmIHJlcXVpcmVfb3JkZXJlZF9ncm91cHMoZXhwZWN0ZWQ6IGxpc3RbaW50XSwgYWN0dWFsOiBBbnkpIC0+IE5vbmU6CiAgICAiIiJSZXF1aXJlIGV4YWN0IGdyb3VwIG1lbWJlcnNoaXAgYW5kIG9yZGVyIGZyb20gdGhlIGZyb3plbiBleHBvcnQgY29udHJhY3QuIiIiCiAgICBpZiBub3QgaXNpbnN0YW5jZShhY3R1YWwsIGxpc3QpIG9yIGFueSh0eXBlKGdyb3VwKSBpcyBub3QgaW50IGZvciBncm91cCBpbiBhY3R1YWwpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIm9yZGVyZWQgZ3JvdXAgbGlzdCBtdXN0IGNvbnRhaW4gc3RyaWN0IGludGVnZXIgSURzIikKICAgIGlmIGFjdHVhbCAhPSBleHBlY3RlZDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgZ3JvdXAgb3JkZXIgZGlmZmVycyBmcm9tIHRoZSBjYW5vbmljYWwgRjAgZXZhbHVhdGlvbiBvcmRlciIpCgoKZGVmIG1lbWJlcnNoaXBfc2hhMjU2KGdyb3VwczogbGlzdFtpbnRdKSAtPiBzdHI6CiAgICAiIiJIYXNoIHRoZSBwcm9kdWNlcidzIEFTQ0lJIGRlY2ltYWwgcm93LWdyb3VwIGxpc3QsIG9uZSBJRCBwZXIgbGluZS4iIiIKICAgIGVuY29kZWQgPSAiIi5qb2luKGYie2dyb3VwfVxuIiBmb3IgZ3JvdXAgaW4gZ3JvdXBzKS5lbmNvZGUoImFzY2lpIikKICAgIHJldHVybiBoYXNobGliLnNoYTI1NihlbmNvZGVkKS5oZXhkaWdlc3QoKQoKCmRlZiBzY29yZV9zdHJlYW0oCiAgICBncm91cHM6IGxpc3RbaW50XSwKICAgIGNvbXBvbmVudF9ieV9ncm91cDogZGljdFtpbnQsIGludF0sCiAgICBwcmVkaWN0aW9uczogZGljdFtzdHIsIG5wLm5kYXJyYXldLAogICAgc2VxdWVuY2VfcmVhZGVyOiBTZXF1ZW5jZVJlYWRlciwKICAgIGhlbHBlcnM6IE1ldHJpY0hlbHBlcnMsCikgLT4gZGljdFtzdHIsIEFueV06CiAgICAiIiJTdHJlYW0gcGVyLWFybSBvZmZpY2lhbCBzY29yZXMgYW5kIGZvdXIgcGFpcmVkIGNvbXBvbmVudCBib290c3RyYXBzLiIiIgogICAgaWYgc2V0KHByZWRpY3Rpb25zKSAhPSBzZXQoQVJNX01PREVTKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleGFjdGx5IHRoZSBmaXZlIGZpeGVkIGNvbXBhcmlzb24gYXJtcyBhcmUgcmVxdWlyZWQiKQogICAgaWYgKAogICAgICAgIGxlbihncm91cHMpID09IDAKICAgICAgICBvciBhbnkodHlwZShncm91cCkgaXMgbm90IGludCBmb3IgZ3JvdXAgaW4gZ3JvdXBzKQogICAgICAgIG9yIGxlbihzZXQoZ3JvdXBzKSkgIT0gbGVuKGdyb3VwcykKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic2NvcmluZyBncm91cCBsaXN0IG11c3QgYmUgbm9uZW1wdHkgYW5kIHVuaXF1ZSIpCiAgICBpZiBhbnkoZ3JvdXAgbm90IGluIGNvbXBvbmVudF9ieV9ncm91cCBmb3IgZ3JvdXAgaW4gZ3JvdXBzKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJhIHNlbGVjdGVkIGdyb3VwIGxhY2tzIGl0cyBjYW5vbmljYWwgZmVhdHVyZS1vbmx5IGNvbXBvbmVudCIpCiAgICBpZiBhbnkodHlwZShjb21wb25lbnRfYnlfZ3JvdXBbZ3JvdXBdKSBpcyBub3QgaW50IGZvciBncm91cCBpbiBncm91cHMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImNvbXBvbmVudCBJRHMgbXVzdCBiZSBjYW5vbmljYWwgc3RyaWN0IGludGVnZXJzIikKICAgIGZvciBhcm0sIGFycmF5IGluIHByZWRpY3Rpb25zLml0ZW1zKCk6CiAgICAgICAgaWYgYXJyYXkuc2hhcGUgIT0gKGxlbihncm91cHMpLCBOX1JPV1MsIE5fT1VUUFVUUykgb3IgYXJyYXkuZHR5cGUuc3RyICE9ICI8ZjQiOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie2FybX0gcHJlZGljdGlvbiBhcnJheSBpcyBub3QgYWxpZ25lZCBGUDMyIChOLCAyMDAwMCwgMikiKQoKICAgIGFjY3VtdWxhdG9ycyA9IHthcm06IGhlbHBlcnMuZ2xvYmFsX2FjY3VtdWxhdG9yKCkgZm9yIGFybSBpbiBBUk1fTU9ERVN9CiAgICBjb250cmFzdF9hcm1zID0gdHVwbGUoYXJtIGZvciBhcm0gaW4gQVJNX01PREVTIGlmIGFybSAhPSAiY29udHJvbCIpCiAgICBwYWlyZWRfY2FuZGlkYXRlID0gewogICAgICAgIGFybTogbnAuemVyb3MoKGxlbihncm91cHMpLCAyLCA2KSwgZHR5cGU9bnAuZmxvYXQ2NCkgZm9yIGFybSBpbiBjb250cmFzdF9hcm1zCiAgICB9CiAgICBwYWlyZWRfY29udHJvbCA9IHthcm06IG5wLnplcm9zKChsZW4oZ3JvdXBzKSwgMiwgNiksIGR0eXBlPW5wLmZsb2F0NjQpIGZvciBhcm0gaW4gY29udHJhc3RfYXJtc30KICAgIHN0ZXBzID0gbnAuYXJhbmdlKE5fUk9XUykgPj0gaGVscGVycy53YXJtdXAKCiAgICBmb3IgcG9zaXRpb24sIGdyb3VwIGluIGVudW1lcmF0ZShncm91cHMpOgogICAgICAgIHNlcXVlbmNlID0gc2VxdWVuY2VfcmVhZGVyLnJlYWQoZ3JvdXAsIHRhcmdldHM9VHJ1ZSwgbWFzaz1UcnVlKQogICAgICAgIHRhcmdldHMgPSBzZXF1ZW5jZS50YXJnZXRzCiAgICAgICAgc2NvcmVkID0gc2VxdWVuY2Uuc2NvcmVkCiAgICAgICAgaWYgdGFyZ2V0cyBpcyBOb25lIG9yIHNjb3JlZCBpcyBOb25lOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZ3JvdXAge2dyb3VwfSBsYWNrcyB2YWxpZGF0aW9uIHRhcmdldHMgb3Igb2ZmaWNpYWwgc2NvcmVkIG1hc2siKQogICAgICAgIGlmIHRhcmdldHMuc2hhcGUgIT0gKE5fUk9XUywgMikgb3IgdGFyZ2V0cy5kdHlwZSAhPSBucC5mbG9hdDMyOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZ3JvdXAge2dyb3VwfSB0YXJnZXQgc2hhcGUgb3IgZHR5cGUgZGlmZmVycyIpCiAgICAgICAgaWYgc2NvcmVkLnNoYXBlICE9IChOX1JPV1MsKSBvciBzY29yZWQuZHR5cGUua2luZCAhPSAiYiI6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJncm91cCB7Z3JvdXB9IHNjb3JlZCBtYXNrIHNoYXBlIG9yIGR0eXBlIGRpZmZlcnMiKQogICAgICAgIG9mZmljaWFsX21hc2sgPSBucC5hc2FycmF5KHNjb3JlZCwgZHR5cGU9Ym9vbCkgJiBzdGVwcwogICAgICAgIHRhcmdldHNfYmF0Y2ggPSB0YXJnZXRzW05vbmUsIC4uLl0KICAgICAgICBtYXNrX2JhdGNoID0gb2ZmaWNpYWxfbWFza1tOb25lLCAuLi5dCiAgICAgICAgZ3JvdXBfcHJlZGljdGlvbnMgPSB7CiAgICAgICAgICAgIGFybTogbnAuYXNhcnJheShwcmVkaWN0aW9uc1thcm1dW3Bvc2l0aW9uXSwgZHR5cGU9bnAuZmxvYXQzMilbTm9uZSwgLi4uXQogICAgICAgICAgICBmb3IgYXJtIGluIEFSTV9NT0RFUwogICAgICAgIH0KICAgICAgICBmb3IgYXJtIGluIEFSTV9NT0RFUzoKICAgICAgICAgICAgaWYgbm90IG5wLmlzZmluaXRlKGdyb3VwX3ByZWRpY3Rpb25zW2FybV0pLmFsbCgpOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmInthcm19IGhhcyBhIG5vbmZpbml0ZSBleHBvcnRlZCBwcmVkaWN0aW9uIGluIGdyb3VwIHtncm91cH0iKQogICAgICAgICAgICBhY2N1bXVsYXRvcnNbYXJtXS5hZGQodGFyZ2V0c19iYXRjaCwgZ3JvdXBfcHJlZGljdGlvbnNbYXJtXSwgbWFza19iYXRjaCkKICAgICAgICBpZiBvZmZpY2lhbF9tYXNrLmFueSgpOgogICAgICAgICAgICBmb3IgYXJtIGluIGNvbnRyYXN0X2FybXM6CiAgICAgICAgICAgICAgICBhcm1fbW9tZW50cywgY29udHJvbF9tb21lbnRzID0gaGVscGVycy5wYWlyZWRfY29tcG9uZW50X21vbWVudHMoCiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0c19iYXRjaCwKICAgICAgICAgICAgICAgICAgICBncm91cF9wcmVkaWN0aW9uc1thcm1dLAogICAgICAgICAgICAgICAgICAgIGdyb3VwX3ByZWRpY3Rpb25zWyJjb250cm9sIl0sCiAgICAgICAgICAgICAgICAgICAgbWFza19iYXRjaCwKICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIHBhaXJlZF9jYW5kaWRhdGVbYXJtXVtwb3NpdGlvbl0gPSBhcm1fbW9tZW50c1swXQogICAgICAgICAgICAgICAgcGFpcmVkX2NvbnRyb2xbYXJtXVtwb3NpdGlvbl0gPSBjb250cm9sX21vbWVudHNbMF0KICAgICAgICBkZWwgc2VxdWVuY2UsIHRhcmdldHMsIHNjb3JlZCwgb2ZmaWNpYWxfbWFzaywgdGFyZ2V0c19iYXRjaCwgbWFza19iYXRjaCwgZ3JvdXBfcHJlZGljdGlvbnMKCiAgICBhcm1fc2NvcmVzID0ge2FybTogYWNjdW11bGF0b3JzW2FybV0ucmVzdWx0KCkgZm9yIGFybSBpbiBBUk1fTU9ERVN9CiAgICBjb21wb25lbnRzID0gbnAuYXNhcnJheShbY29tcG9uZW50X2J5X2dyb3VwW2dyb3VwXSBmb3IgZ3JvdXAgaW4gZ3JvdXBzXSwgZHR5cGU9bnAuaW50NjQpCiAgICBjb250cmFzdHM6IGRpY3Rbc3RyLCBBbnldID0ge30KICAgIGZvciBhcm0gaW4gY29udHJhc3RfYXJtczoKICAgICAgICBwYWlyZWQgPSBoZWxwZXJzLnBhaXJlZF9jb21wb25lbnRfYm9vdHN0cmFwKAogICAgICAgICAgICBwYWlyZWRfY2FuZGlkYXRlW2FybV0sCiAgICAgICAgICAgIHBhaXJlZF9jb250cm9sW2FybV0sCiAgICAgICAgICAgIGNvbXBvbmVudHMsCiAgICAgICAgICAgIG5fYm9vdD1CT09UU1RSQVBfUkVQTElDQVRFUywKICAgICAgICAgICAgc2VlZD1CT09UU1RSQVBfU0VFRCwKICAgICAgICAgICAgY2VudGVyZWQ9VHJ1ZSwKICAgICAgICApCiAgICAgICAgY29udHJhc3RzW2FybV0gPSB7ImNhbmRpZGF0ZV9taW51c19jb250cm9sIjogcGFpcmVkfQogICAgcmV0dXJuIHsKICAgICAgICAibWV0cmljX2lkIjogaGVscGVycy5tZXRyaWNfaWQsCiAgICAgICAgIm9yZGVyZWRfZ3JvdXBfY291bnQiOiBsZW4oZ3JvdXBzKSwKICAgICAgICAiZGlzdGluY3RfY29tcG9uZW50X2NvdW50IjogaW50KG5wLnVuaXF1ZShjb21wb25lbnRzKS5zaXplKSwKICAgICAgICAib2ZmaWNpYWxfbWFzayI6ICJpc19zY29yZWQgQU5EICh0ID49IDk5KSIsCiAgICAgICAgImFybXMiOiBhcm1fc2NvcmVzLAogICAgICAgICJwYWlyZWRfY29udHJhc3RzX3ZzX2NvbnRyb2wiOiBjb250cmFzdHMsCiAgICAgICAgInVuY2VydGFpbnR5X3Njb3BlIjogInBhaXJlZCBmZWF0dXJlLW9ubHkgc3VjY2Vzc29yLWNvbXBvbmVudCBib290c3RyYXAgY29uZGl0aW9uYWwgb24gdGhlc2UgZml2ZSBmaXhlZCBlcG9jaC01IG1vZGVsczsgbm90IGZpdC1zZWVkIG9yIHB1YmxpYy10cmFuc2ZlciB1bmNlcnRhaW50eSIsCiAgICAgICAgImJvb3RzdHJhcCI6IHsKICAgICAgICAgICAgInJlcGxpY2F0ZXMiOiBCT09UU1RSQVBfUkVQTElDQVRFUywKICAgICAgICAgICAgInNlZWQiOiBCT09UU1RSQVBfU0VFRCwKICAgICAgICAgICAgImNlbnRlcmVkX21vbWVudHMiOiBUcnVlLAogICAgICAgIH0sCiAgICB9Cg==',
    'first_cell_score_predictions.py': 'IiIiVmVyaWZ5IGV4cG9ydGVkIHNpZGVjYXJzIGFuZCBvcGVuIGZpeGVkIHJlYWQtb25seSBwcmVkaWN0aW9uIG1hcHBpbmdzLiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCmZyb20gdHlwaW5nIGltcG9ydCBBbnkKCmltcG9ydCBudW1weSBhcyBucApmcm9tIGZpcnN0X2NlbGxfc2NvcmVfY29yZSBpbXBvcnQgKAogICAgQVJNX01PREVTLAogICAgRVhQRUNURURfQklORElOR19TSEEyNTYsCiAgICBFWFBFQ1RFRF9TUExJVF9TSEEyNTYsCiAgICBNQVhfUFJFRElDVElPTl9CWVRFUywKICAgIE1BWF9TTUFMTF9KU09OX0JZVEVTLAogICAgTl9HUk9VUFMsCiAgICBQUkVESUNUSU9OX0RBVEFfQllURVMsCiAgICBQUkVESUNUSU9OX1NIQVBFLAogICAgX3Bpbl9maWxlLAogICAgX3JlYWRfanNvbiwKICAgIF9zYWZlX2NoaWxkLAogICAgX3N0YWJsZV9pZGVudGl0eSwKICAgIG1lbWJlcnNoaXBfc2hhMjU2LAogICAgcmVxdWlyZV9vcmRlcmVkX2dyb3VwcywKKQoKCmRlZiBfb3Blbl9wcmVkaWN0aW9ucygKICAgIGV4cG9ydF9yb290OiBQYXRoLCBiaW5kaW5nOiBkaWN0W3N0ciwgQW55XSwgbWFuaWZlc3Q6IGRpY3Rbc3RyLCBBbnldCikgLT4gdHVwbGVbZGljdFtzdHIsIG5wLm5kYXJyYXldLCBkaWN0W3N0ciwgZGljdFtzdHIsIEFueV1dLCBsaXN0W2RpY3Rbc3RyLCBBbnldXV06CiAgICAiIiJWYWxpZGF0ZSBhbGwgZml2ZSBmaWxlcywgdGhlbiBvcGVuIHRoZWlyIHJlYWQtb25seSBOUFkgbWFwcGluZ3MuIiIiCiAgICBtYW5pZmVzdF9ncm91cHMgPSBtYW5pZmVzdC5nZXQoIm9yZGVyZWRfZ3JvdXBzIikKICAgIGlmIG5vdCBpc2luc3RhbmNlKG1hbmlmZXN0X2dyb3VwcywgbGlzdCkgb3IgbGVuKG1hbmlmZXN0X2dyb3VwcykgIT0gTl9HUk9VUFM6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IG1hbmlmZXN0IGRvZXMgbm90IGNvbnRhaW4gdGhlIGZpeGVkIDQ2OCBncm91cCBvcmRlciIpCiAgICBpZiAoCiAgICAgICAgYW55KHR5cGUoZ3JvdXApIGlzIG5vdCBpbnQgZm9yIGdyb3VwIGluIG1hbmlmZXN0X2dyb3VwcykKICAgICAgICBvciBsZW4oc2V0KG1hbmlmZXN0X2dyb3VwcykpICE9IE5fR1JPVVBTCiAgICApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImV4cG9ydCBtYW5pZmVzdCBncm91cCBJRHMgYXJlIG5vdCBzdHJpY3QgdW5pcXVlIGludGVnZXJzIikKICAgIGlmIGFueShncm91cCA8IDAgb3IgZ3JvdXAgPj0gMjBfMDAwIGZvciBncm91cCBpbiBtYW5pZmVzdF9ncm91cHMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImV4cG9ydCBtYW5pZmVzdCBjb250YWlucyBhbiBvdXQtb2YtcmFuZ2UgdmFsaWRhdGlvbiByb3ctZ3JvdXAgSUQiKQogICAgZXhwZWN0ZWRfbWVtYmVyc2hpcF9zaGEyNTYgPSBtZW1iZXJzaGlwX3NoYTI1NihtYW5pZmVzdF9ncm91cHMpCiAgICBpZiBtYW5pZmVzdC5nZXQoIm1lbWJlcnNoaXBfc2hhMjU2IikgIT0gZXhwZWN0ZWRfbWVtYmVyc2hpcF9zaGEyNTY6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IG1hbmlmZXN0IG9yZGVyZWQtbWVtYmVyc2hpcCBkaWdlc3QgZGlmZmVycyIpCiAgICBleHBlY3RlZF9tb2RlcyA9IHthcm1bImFybSJdOiBhcm0gZm9yIGFybSBpbiBiaW5kaW5nLmdldCgiYXJtcyIsIFtdKX0KICAgIGlmIHNldChleHBlY3RlZF9tb2RlcykgIT0gc2V0KEFSTV9NT0RFUykgb3IgYW55KAogICAgICAgIGV4cGVjdGVkX21vZGVzW25hbWVdLmdldCgicmVwcmVzZW50YXRpb25fbW9kZSIpICE9IG1vZGUgZm9yIG5hbWUsIG1vZGUgaW4gQVJNX01PREVTLml0ZW1zKCkKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IGJpbmRpbmcgZG9lcyBub3QgcmVnaXN0ZXIgdGhlIGZpdmUgZml4ZWQgYXJtcy9tb2RlcyIpCiAgICBtYW5pZmVzdF9hcm1zID0gbWFuaWZlc3QuZ2V0KCJhcm1zIikKICAgIGlmICgKICAgICAgICBub3QgaXNpbnN0YW5jZShtYW5pZmVzdF9hcm1zLCBsaXN0KQogICAgICAgIG9yIFtyb3cuZ2V0KCJhcm0iKSBmb3Igcm93IGluIG1hbmlmZXN0X2FybXMgaWYgaXNpbnN0YW5jZShyb3csIGRpY3QpXSAhPSBsaXN0KEFSTV9NT0RFUykKICAgICAgICBvciBsZW4obWFuaWZlc3RfYXJtcykgIT0gbGVuKEFSTV9NT0RFUykKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiZXhwb3J0IG1hbmlmZXN0IGFybSBpbnZlbnRvcnkgZGlmZmVycyIpCiAgICByb3dzID0ge3Jvdy5nZXQoImFybSIpOiByb3cgZm9yIHJvdyBpbiBtYW5pZmVzdF9hcm1zIGlmIGlzaW5zdGFuY2Uocm93LCBkaWN0KX0KICAgIGlmIHNldChyb3dzKSAhPSBzZXQoQVJNX01PREVTKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJleHBvcnQgbWFuaWZlc3QgYXJtcyBhcmUgbWlzc2luZywgZHVwbGljYXRlZCwgb3IgdW5leHBlY3RlZCIpCiAgICBhcnJheXM6IGRpY3Rbc3RyLCBucC5uZGFycmF5XSA9IHt9CiAgICBwaW5zOiBkaWN0W3N0ciwgZGljdFtzdHIsIEFueV1dID0ge30KICAgIGlkZW50aXRpZXM6IGxpc3RbZGljdFtzdHIsIEFueV1dID0gW10KICAgIG1hcHBpbmdfaW5wdXRzOiBkaWN0W3N0ciwgdHVwbGVbUGF0aCwgaW50LCB0dXBsZVtpbnQsIGludCwgaW50LCBpbnRdXV0gPSB7fQogICAgZm9yIG5hbWUsIG1vZGUgaW4gQVJNX01PREVTLml0ZW1zKCk6CiAgICAgICAgYXJtID0gZXhwZWN0ZWRfbW9kZXNbbmFtZV0KICAgICAgICByb3cgPSByb3dzW25hbWVdCiAgICAgICAgc2lkZWNhcl9wYXRoID0gX3NhZmVfY2hpbGQoZXhwb3J0X3Jvb3QsIGYie25hbWV9L1BSRURJQ1RJT05fU0lERUNBUi5qc29uIikKICAgICAgICBzaWRlY2FyLCBzaWRlY2FyX3Bpbiwgc2lkZWNhcl9pZGVudGl0eSA9IF9yZWFkX2pzb24oc2lkZWNhcl9wYXRoLCBNQVhfU01BTExfSlNPTl9CWVRFUykKICAgICAgICBwcmVkaWN0aW9uX3BhdGggPSBfc2FmZV9jaGlsZChleHBvcnRfcm9vdCwgZiJ7bmFtZX0vcHJlZGljdGlvbnMubnB5IikKICAgICAgICBwcmVkaWN0aW9uX3BpbiwgcHJlZGljdGlvbl9pZGVudGl0eSA9IF9waW5fZmlsZShwcmVkaWN0aW9uX3BhdGgsIE1BWF9QUkVESUNUSU9OX0JZVEVTKQogICAgICAgIGlmIChyb3cuZ2V0KCJkaXJlY3RvcnkiKSwgcm93LmdldCgic2lkZWNhcl9ieXRlcyIpLCByb3cuZ2V0KCJzaWRlY2FyX3NoYTI1NiIpKSAhPSAoCiAgICAgICAgICAgIG5hbWUsCiAgICAgICAgICAgIHNpZGVjYXJfcGluWyJieXRlcyJdLAogICAgICAgICAgICBzaWRlY2FyX3Bpblsic2hhMjU2Il0sCiAgICAgICAgKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBzaWRlY2FyIGRvZXMgbm90IGpvaW4gdGhlIGNvbXBsZXRpb24gbWFuaWZlc3QiKQogICAgICAgIGlmIChyb3cuZ2V0KCJwcmVkaWN0aW9uX2J5dGVzIiksIHJvdy5nZXQoInByZWRpY3Rpb25fc2hhMjU2IikpICE9ICgKICAgICAgICAgICAgcHJlZGljdGlvbl9waW5bImJ5dGVzIl0sCiAgICAgICAgICAgIHByZWRpY3Rpb25fcGluWyJzaGEyNTYiXSwKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie25hbWV9IHByZWRpY3Rpb24gYnl0ZXMgZG8gbm90IGpvaW4gdGhlIGNvbXBsZXRpb24gbWFuaWZlc3QiKQogICAgICAgIGV4cGVjdGVkX2NoZWNrcG9pbnQgPSB7CiAgICAgICAgICAgICoqYXJtWyJtb2RlbCJdLAogICAgICAgICAgICAibG9hZGVkX3NoYTI1NiI6IGFybVsibW9kZWwiXVsic2hhMjU2Il0sCiAgICAgICAgfQogICAgICAgIGV4cGVjdGVkX3JvdyA9IHsKICAgICAgICAgICAgImFybSI6IG5hbWUsCiAgICAgICAgICAgICJkaXJlY3RvcnkiOiBuYW1lLAogICAgICAgICAgICAicnVuX25hbWUiOiBhcm1bInJ1bl9uYW1lIl0sCiAgICAgICAgICAgICJ1bml0IjogYXJtWyJ1bml0Il0sCiAgICAgICAgICAgICJpbnZvY2F0aW9uX2lkIjogYXJtWyJpbnZvY2F0aW9uX2lkIl0sCiAgICAgICAgICAgICJyZXByZXNlbnRhdGlvbl9tb2RlIjogbW9kZSwKICAgICAgICAgICAgInNlbGVjdGVkX2Vwb2NoIjogNSwKICAgICAgICAgICAgInN1Y2Nlc3NmdWxfdXBkYXRlcyI6IDQ0NzYwLAogICAgICAgICAgICAiY2hlY2twb2ludCI6IGV4cGVjdGVkX2NoZWNrcG9pbnQsCiAgICAgICAgICAgICJtZW1iZXJzaGlwX3NoYTI1NiI6IGV4cGVjdGVkX21lbWJlcnNoaXBfc2hhMjU2LAogICAgICAgICAgICAicHJlZGljdGlvbl9wYXRoIjogZiJ7bmFtZX0vcHJlZGljdGlvbnMubnB5IiwKICAgICAgICAgICAgInNpZGVjYXJfcGF0aCI6IGYie25hbWV9L1BSRURJQ1RJT05fU0lERUNBUi5qc29uIiwKICAgICAgICAgICAgInByZWRpY3Rpb25fYnl0ZXMiOiBwcmVkaWN0aW9uX3BpblsiYnl0ZXMiXSwKICAgICAgICAgICAgInByZWRpY3Rpb25fc2hhMjU2IjogcHJlZGljdGlvbl9waW5bInNoYTI1NiJdLAogICAgICAgICAgICAic2lkZWNhcl9ieXRlcyI6IHNpZGVjYXJfcGluWyJieXRlcyJdLAogICAgICAgICAgICAic2lkZWNhcl9zaGEyNTYiOiBzaWRlY2FyX3Bpblsic2hhMjU2Il0sCiAgICAgICAgfQogICAgICAgIGlmIGFueShyb3cuZ2V0KGtleSkgIT0gdmFsdWUgZm9yIGtleSwgdmFsdWUgaW4gZXhwZWN0ZWRfcm93Lml0ZW1zKCkpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie25hbWV9IHJlcGVhdGVkIGFybSBpZGVudGl0eSBvciBmaWxlIHBpbnMgZGlmZmVyIGluIHRoZSBtYW5pZmVzdCIpCiAgICAgICAgaWYgc2V0KHJvdykgIT0gc2V0KGV4cGVjdGVkX3Jvdyk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gbWFuaWZlc3Qgcm93IGhhcyB1bmV4cGVjdGVkIG9yIG1pc3NpbmcgZmllbGRzIikKICAgICAgICBleHBlY3RlZF9zaWRlY2FyX2ZpZWxkcyA9IHsKICAgICAgICAgICAgInNjaGVtYSIsCiAgICAgICAgICAgICJzdGF0dXMiLAogICAgICAgICAgICAiYmluZGluZ19zaGEyNTYiLAogICAgICAgICAgICAiYXJtIiwKICAgICAgICAgICAgInJ1bl9uYW1lIiwKICAgICAgICAgICAgInVuaXQiLAogICAgICAgICAgICAiaW52b2NhdGlvbl9pZCIsCiAgICAgICAgICAgICJzZWxlY3RlZF9lcG9jaCIsCiAgICAgICAgICAgICJzdWNjZXNzZnVsX3VwZGF0ZXMiLAogICAgICAgICAgICAidGVybWluYWxfYWNjZXB0YW5jZSIsCiAgICAgICAgICAgICJleGFjdF9zdG9wIiwKICAgICAgICAgICAgImNoZWNrcG9pbnQiLAogICAgICAgICAgICAicmVwcmVzZW50YXRpb25fbW9kZSIsCiAgICAgICAgICAgICJzb3VyY2UiLAogICAgICAgICAgICAic3BsaXQiLAogICAgICAgICAgICAib3JkZXJlZF9ncm91cHMiLAogICAgICAgICAgICAibWVtYmVyc2hpcF9lbmNvZGluZyIsCiAgICAgICAgICAgICJtZW1iZXJzaGlwX3NoYTI1NiIsCiAgICAgICAgICAgICJmZWF0dXJlX29ubHkiLAogICAgICAgICAgICAicmVwcmVzZW50YXRpb24iLAogICAgICAgICAgICAiaW5mZXJlbmNlIiwKICAgICAgICAgICAgInByZWRpY3Rpb24iLAogICAgICAgICAgICAic2NvcmluZ19wZXJmb3JtZWQiLAogICAgICAgIH0KICAgICAgICBpZiBzZXQoc2lkZWNhcikgIT0gZXhwZWN0ZWRfc2lkZWNhcl9maWVsZHM6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gc2lkZWNhciBoYXMgdW5leHBlY3RlZCBvciBtaXNzaW5nIGZpZWxkcyIpCiAgICAgICAgaWYgKAogICAgICAgICAgICBzaWRlY2FyLmdldCgic2NoZW1hIikgIT0gImZnMTkuZmlyc3QtY2VsbC1mZWF0dXJlLW9ubHktcHJlZGljdGlvbi52MSIKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoInN0YXR1cyIpICE9ICJQUkVESUNUSU9OX0VYUE9SVF9PTkxZX05PVF9TQ09SRUQiCiAgICAgICAgICAgIG9yIHNpZGVjYXIuZ2V0KCJiaW5kaW5nX3NoYTI1NiIpICE9IEVYUEVDVEVEX0JJTkRJTkdfU0hBMjU2CiAgICAgICAgICAgIG9yIHNpZGVjYXIuZ2V0KCJhcm0iKSAhPSBuYW1lCiAgICAgICAgICAgIG9yIHNpZGVjYXIuZ2V0KCJydW5fbmFtZSIpICE9IGFybS5nZXQoInJ1bl9uYW1lIikKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoInVuaXQiKSAhPSBhcm0uZ2V0KCJ1bml0IikKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoImludm9jYXRpb25faWQiKSAhPSBhcm0uZ2V0KCJpbnZvY2F0aW9uX2lkIikKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoInNlbGVjdGVkX2Vwb2NoIikgIT0gNQogICAgICAgICAgICBvciBzaWRlY2FyLmdldCgic3VjY2Vzc2Z1bF91cGRhdGVzIikgIT0gNDQ3NjAKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoInRlcm1pbmFsX2FjY2VwdGFuY2UiKSAhPSBhcm0uZ2V0KCJ0ZXJtaW5hbF9hY2NlcHRhbmNlIikKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoImV4YWN0X3N0b3AiKSAhPSBhcm0uZ2V0KCJleGFjdF9zdG9wIikKICAgICAgICAgICAgb3Igc2lkZWNhci5nZXQoInNvdXJjZSIpICE9IGJpbmRpbmcuZ2V0KCJzb3VyY2UiKQogICAgICAgICk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gc2lkZWNhciBpZGVudGl0eSBkaWZmZXJzIGZyb20gdGhlIHJlZ2lzdGVyZWQgYXJtIikKICAgICAgICBjaGVja3BvaW50ID0gc2lkZWNhci5nZXQoImNoZWNrcG9pbnQiKQogICAgICAgIGlmICgKICAgICAgICAgICAgbm90IGlzaW5zdGFuY2UoY2hlY2twb2ludCwgZGljdCkKICAgICAgICAgICAgb3IgYW55KAogICAgICAgICAgICAgICAgY2hlY2twb2ludC5nZXQoa2V5KSAhPSBhcm0uZ2V0KCJtb2RlbCIsIHt9KS5nZXQoa2V5KQogICAgICAgICAgICAgICAgZm9yIGtleSBpbiAoImJ5dGVzIiwgInNoYTI1NiIsICJyZWxhdGl2ZV9wYXRoIikKICAgICAgICAgICAgKQogICAgICAgICAgICBvciBjaGVja3BvaW50LmdldCgibG9hZGVkX3NoYTI1NiIpICE9IGFybS5nZXQoIm1vZGVsIiwge30pLmdldCgic2hhMjU2IikKICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie25hbWV9IGNoZWNrcG9pbnQgcGluIGRpZmZlcnMiKQogICAgICAgIGlmIHNpZGVjYXIuZ2V0KCJzcGxpdCIsIHt9KS5nZXQoInNoYTI1NiIpICE9IEVYUEVDVEVEX1NQTElUX1NIQTI1NjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBzcGxpdCBwaW4gZGlmZmVycyIpCiAgICAgICAgaWYgc2lkZWNhci5nZXQoInNwbGl0IikgIT0gYmluZGluZy5nZXQoInNwbGl0IikgfCB7CiAgICAgICAgICAgICJyb2xlIjogImZpdC1leGNsdWRlZC1leHBvc2VkLWRldmVsb3BtZW50LWV2YWx1YXRpb24iCiAgICAgICAgfToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBzcGxpdCBjb250cmFjdCBkaWZmZXJzIikKICAgICAgICBpZiBzaWRlY2FyLmdldCgicmVwcmVzZW50YXRpb24iLCB7fSkuZ2V0KCJtb2RlIikgIT0gbW9kZToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSByZXByZXNlbnRhdGlvbiBtb2RlIGRpZmZlcnMiKQogICAgICAgIHJlcHJlc2VudGF0aW9uID0gYmluZGluZy5nZXQoInJlcHJlc2VudGF0aW9uIikKICAgICAgICBpZiBub3QgaXNpbnN0YW5jZShyZXByZXNlbnRhdGlvbiwgZGljdCkgb3Igc2lkZWNhci5nZXQoInJlcHJlc2VudGF0aW9uIikgIT0gewogICAgICAgICAgICAibW9kZSI6IG1vZGUsCiAgICAgICAgICAgICJhcnRpZmFjdF9zaGEyNTYiOiByZXByZXNlbnRhdGlvbi5nZXQoImFydGlmYWN0X3NoYTI1NiIpLAogICAgICAgICAgICAic3RhdGVfc2hhMjU2IjogcmVwcmVzZW50YXRpb24uZ2V0KCJzdGF0ZV9zaGEyNTYiKSwKICAgICAgICAgICAgIm5vcm1hbGl6YXRpb24iOiByZXByZXNlbnRhdGlvbi5nZXQoIm5vcm1hbGl6YXRpb24iKSwKICAgICAgICB9OgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie25hbWV9IHJlcHJlc2VudGF0aW9uIGFydGlmYWN0IG9yIG5vcm1hbGl6YXRpb24gZGlmZmVycyIpCiAgICAgICAgcmVxdWlyZV9vcmRlcmVkX2dyb3VwcyhtYW5pZmVzdF9ncm91cHMsIHNpZGVjYXIuZ2V0KCJvcmRlcmVkX2dyb3VwcyIpKQogICAgICAgIGlmICgKICAgICAgICAgICAgc2lkZWNhci5nZXQoIm1lbWJlcnNoaXBfZW5jb2RpbmciKQogICAgICAgICAgICAhPSAiQVNDSUkgZGVjaW1hbCByb3ctZ3JvdXAgSURzLCBvbmUgcGVyIGxpbmUsIGZpbmFsIExGIgogICAgICAgICAgICBvciBzaWRlY2FyLmdldCgibWVtYmVyc2hpcF9zaGEyNTYiKSAhPSBleHBlY3RlZF9tZW1iZXJzaGlwX3NoYTI1NgogICAgICAgICk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gb3JkZXJlZC1tZW1iZXJzaGlwIGVuY29kaW5nIG9yIGRpZ2VzdCBkaWZmZXJzIikKICAgICAgICBpZiBzaWRlY2FyLmdldCgiZmVhdHVyZV9vbmx5IikgIT0geyJ0YXJnZXRzX3JlYWQiOiBGYWxzZSwgInNjb3JlZF9tYXNrX3JlYWQiOiBGYWxzZX06CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gcHJlZGljdGlvbiBwYXNzIHdhcyBub3QgZmVhdHVyZS1vbmx5IikKICAgICAgICBpZiBzaWRlY2FyLmdldCgiaW5mZXJlbmNlIikgIT0gewogICAgICAgICAgICAicHJlY2lzaW9uIjogInN0cmljdF9mcDMyIiwKICAgICAgICAgICAgImF1dG9jYXN0IjogRmFsc2UsCiAgICAgICAgICAgICJ0ZjMyIjogRmFsc2UsCiAgICAgICAgICAgICJjaHVua19yb3dzIjogMjAwMCwKICAgICAgICAgICAgInNlcXVlbmNlX2JhdGNoX3NpemUiOiAxLAogICAgICAgIH06CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gaW5mZXJlbmNlIHByZWNpc2lvbiBvciBiYXRjaCBjb250cmFjdCBkaWZmZXJzIikKICAgICAgICBwcmVkaWN0aW9uX21ldGEgPSBzaWRlY2FyLmdldCgicHJlZGljdGlvbiIpCiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2UocHJlZGljdGlvbl9tZXRhLCBkaWN0KToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBwcmVkaWN0aW9uIG1ldGFkYXRhIGlzIG1hbGZvcm1lZCIpCiAgICAgICAgaGVhZGVyX2J5dGVzID0gcHJlZGljdGlvbl9tZXRhLmdldCgibnB5X2hlYWRlcl9ieXRlcyIpCiAgICAgICAgaWYgdHlwZShoZWFkZXJfYnl0ZXMpIGlzIG5vdCBpbnQgb3Igbm90IDAgPCBoZWFkZXJfYnl0ZXMgPD0gNDA5NjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBOUFkgaGVhZGVyIGxlbmd0aCBpcyBvdXRzaWRlIHRoZSBmaXhlZCBib3VuZCIpCiAgICAgICAgaWYgKAogICAgICAgICAgICBwcmVkaWN0aW9uX21ldGEKICAgICAgICAgICAgIT0gewogICAgICAgICAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAgICAgICAgICJieXRlcyI6IHByZWRpY3Rpb25fcGluWyJieXRlcyJdLAogICAgICAgICAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiBoZWFkZXJfYnl0ZXMsCiAgICAgICAgICAgICAgICAiZGF0YV9ieXRlcyI6IFBSRURJQ1RJT05fREFUQV9CWVRFUywKICAgICAgICAgICAgICAgICJzaGEyNTYiOiBwcmVkaWN0aW9uX3Bpblsic2hhMjU2Il0sCiAgICAgICAgICAgICAgICAiZHR5cGUiOiAiZmxvYXQzMi1saXR0bGUtZW5kaWFuIiwKICAgICAgICAgICAgICAgICJzaGFwZSI6IGxpc3QoUFJFRElDVElPTl9TSEFQRSksCiAgICAgICAgICAgIH0KICAgICAgICAgICAgb3IgcHJlZGljdGlvbl9waW5bImJ5dGVzIl0gIT0gUFJFRElDVElPTl9EQVRBX0JZVEVTICsgaGVhZGVyX2J5dGVzCiAgICAgICAgKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBOUFkgbWV0YWRhdGEgZGlmZmVycyIpCiAgICAgICAgaWYgc2lkZWNhci5nZXQoInNjb3JpbmdfcGVyZm9ybWVkIikgaXMgbm90IEZhbHNlOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie25hbWV9IGV4cG9ydCBwYWNrZXQgYWxyZWFkeSBjbGFpbXMgc2NvcmluZyIpCiAgICAgICAgbWFwcGluZ19pbnB1dHNbbmFtZV0gPSAocHJlZGljdGlvbl9wYXRoLCBoZWFkZXJfYnl0ZXMsIHByZWRpY3Rpb25faWRlbnRpdHkpCiAgICAgICAgcGluc1tuYW1lXSA9IHsicHJlZGljdGlvbiI6IHByZWRpY3Rpb25fcGluLCAic2lkZWNhciI6IHNpZGVjYXJfcGlufQogICAgICAgIGlkZW50aXRpZXMuZXh0ZW5kKAogICAgICAgICAgICBbCiAgICAgICAgICAgICAgICB7ImFybSI6IG5hbWUsICJraW5kIjogInByZWRpY3Rpb24iLCAiaWRlbnRpdHkiOiBwcmVkaWN0aW9uX2lkZW50aXR5fSwKICAgICAgICAgICAgICAgIHsiYXJtIjogbmFtZSwgImtpbmQiOiAic2lkZWNhciIsICJpZGVudGl0eSI6IHNpZGVjYXJfaWRlbnRpdHl9LAogICAgICAgICAgICBdCiAgICAgICAgKQogICAgZm9yIG5hbWUsIChwcmVkaWN0aW9uX3BhdGgsIGhlYWRlcl9ieXRlcywgcHJlZGljdGlvbl9pZGVudGl0eSkgaW4gbWFwcGluZ19pbnB1dHMuaXRlbXMoKToKICAgICAgICBhcnJheSA9IG5wLmxvYWQocHJlZGljdGlvbl9wYXRoLCBtbWFwX21vZGU9InIiLCBhbGxvd19waWNrbGU9RmFsc2UpCiAgICAgICAgaWYgKAogICAgICAgICAgICBub3QgaXNpbnN0YW5jZShhcnJheSwgbnAubWVtbWFwKQogICAgICAgICAgICBvciBhcnJheS5zaGFwZSAhPSBQUkVESUNUSU9OX1NIQVBFCiAgICAgICAgICAgIG9yIGFycmF5LmR0eXBlLnN0ciAhPSAiPGY0IgogICAgICAgICAgICBvciBhcnJheS5mbGFncy53cml0ZWFibGUKICAgICAgICAgICAgb3Igbm90IGFycmF5LmZsYWdzLmNfY29udGlndW91cwogICAgICAgICAgICBvciBhcnJheS5vZmZzZXQgIT0gaGVhZGVyX2J5dGVzCiAgICAgICAgKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntuYW1lfSBOUFkgYXJyYXkgc2hhcGUsIGR0eXBlLCBvciByZWFkLW9ubHkgbW9kZSBkaWZmZXJzIikKICAgICAgICBsaW5rZWQgPSBwcmVkaWN0aW9uX3BhdGgubHN0YXQoKQogICAgICAgIGlmIF9zdGFibGVfaWRlbnRpdHkobGlua2VkLCBsaW5rZWQpICE9IHByZWRpY3Rpb25faWRlbnRpdHk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ7bmFtZX0gcHJlZGljdGlvbiBwYXRoIGNoYW5nZWQgYmVmb3JlIG1hcHBpbmciKQogICAgICAgIGFycmF5c1tuYW1lXSA9IGFycmF5CiAgICByZXR1cm4gYXJyYXlzLCBwaW5zLCBpZGVudGl0aWVzCg==',
    'score_first_cell.py': 'IiIiU3RyZWFtIG9mZmljaWFsIHNjb3JlcyBmb3IgdGhlIGZpeGVkIGZpdmUtYXJtIEYwL3NlZWQtMTAwMSBwYW5lbC4iIiIKCmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnMKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgZGF0ZXRpbWUgYXMgZHQKaW1wb3J0IGpzb24KaW1wb3J0IG9zCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55Cgpmcm9tIGZpcnN0X2NlbGxfc2NvcmVfY29udHJhY3QgaW1wb3J0IF9yZWFkX2V4cG9ydF9idW5kbGUsIF92YWxpZGF0ZV9ib3VuZF9wYXRocwpmcm9tIGZpcnN0X2NlbGxfc2NvcmVfY29yZSBpbXBvcnQgKAogICAgRVhQRUNURURfUE9SVEFCTEVfUkVHSVNUUllfQllURVMsCiAgICBFWFBFQ1RFRF9QT1JUQUJMRV9SRUdJU1RSWV9TSEEyNTYsCiAgICBFWFBFQ1RFRF9QT1JUQUJMRV9TT1VSQ0VfTkFNRVMsCiAgICBFWFBFQ1RFRF9TT1VSQ0VfTUFOSUZFU1RfU0hBMjU2LAogICAgRVhQRUNURURfU09VUkNFX1NFVF9TSEEyNTYsCiAgICBFWFBFQ1RFRF9TUExJVF9TSEEyNTYsCiAgICBNQVhfRVhQT1JUX01BTklGRVNUX0JZVEVTLAogICAgTl9HUk9VUFMsCiAgICBTT1VSQ0VfQ09NTUlULAogICAgX2xvYWRfaGVscGVycywKICAgIF9waW5fZmlsZSwKICAgIF9wcmVmbGlnaHRfc3BsaXRfaW5wdXRzLAogICAgX3N0YWJsZV9pZGVudGl0eSwKICAgIG9ic2VydmVfcmVnaXN0ZXJlZF9zb3VyY2UsCiAgICByZXF1aXJlX29yZGVyZWRfZ3JvdXBzLAogICAgc2NvcmVfc3RyZWFtLAopCmZyb20gZmlyc3RfY2VsbF9zY29yZV9wcmVkaWN0aW9ucyBpbXBvcnQgX29wZW5fcHJlZGljdGlvbnMKCgpkZWYgX3Njb3JlX2J1bmRsZSgKICAgIGFyZ3M6IGFyZ3BhcnNlLk5hbWVzcGFjZSwKKSAtPiBkaWN0W3N0ciwgQW55XToKICAgICIiIlZhbGlkYXRlIGEgY29tcGxldGUgZXhwb3J0IGFuZCBzY29yZSBpdHMgZml4ZWQgcGFuZWwgaW4gc291cmNlIG9yZGVyLiIiIgogICAgYmluZGluZywgX3NvdXJjZV9tYW5pZmVzdCwgbWFuaWZlc3QsIG1hbmlmZXN0X3BpbiwgbWFuaWZlc3RfaWRlbnRpdHkgPSBfcmVhZF9leHBvcnRfYnVuZGxlKAogICAgICAgIGFyZ3MuZXZpZGVuY2Vfcm9vdCwKICAgICAgICBhcmdzLmV4cG9ydF9yb290LAogICAgICAgIGFyZ3MuYmluZGluZywKICAgICAgICBhcmdzLnNvdXJjZV9tYW5pZmVzdCwKICAgICkKICAgIF92YWxpZGF0ZV9ib3VuZF9wYXRocyhhcmdzLCBiaW5kaW5nKQogICAgcmVnaXN0ZXJlZF9kYXRhID0gYmluZGluZ1sicmVnaXN0ZXJlZF9kYXRhIl0KICAgIHJlZ2lzdGVyZWRfc291cmNlcyA9IHJlZ2lzdGVyZWRfZGF0YVsic291cmNlX2lkZW50aXRpZXMiXQogICAgcG9ydGFibGVfcmVnaXN0cnkgPSBQYXRoKHJlZ2lzdGVyZWRfZGF0YVsicG9ydGFibGVfbWFuaWZlc3QiXVsiYWJzb2x1dGVfcGF0aCJdKQogICAgaGVscGVycyA9IF9sb2FkX2hlbHBlcnMoYXJncy5zb3VyY2Vfcm9vdCwgcG9ydGFibGVfcmVnaXN0cnk9cG9ydGFibGVfcmVnaXN0cnkpCiAgICBwb3J0YWJsZV9yZWdpc3RyeV9waW4sIHBvcnRhYmxlX3JlZ2lzdHJ5X2lkZW50aXR5ID0gX3Bpbl9maWxlKHBvcnRhYmxlX3JlZ2lzdHJ5LCAyXzAwMF8wMDApCiAgICBpZiAocG9ydGFibGVfcmVnaXN0cnlfcGluWyJieXRlcyJdLCBwb3J0YWJsZV9yZWdpc3RyeV9waW5bInNoYTI1NiJdKSAhPSAoCiAgICAgICAgRVhQRUNURURfUE9SVEFCTEVfUkVHSVNUUllfQllURVMsCiAgICAgICAgRVhQRUNURURfUE9SVEFCTEVfUkVHSVNUUllfU0hBMjU2LAogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJwb3J0YWJsZSBzb3VyY2UgcmVnaXN0cnkgY2hhbmdlZCBhZnRlciBoZWxwZXIgaW1wb3J0IikKICAgIGRhdGFfcm9vdCA9IGFyZ3MuZGF0YV9yb290CiAgICBzcGxpdF9wYXRoID0gUGF0aChzdHIoYmluZGluZ1sic3BsaXQiXVsiYWJzb2x1dGVfcGF0aCJdKSkKICAgIGlmIHNwbGl0X3BhdGgucmVzb2x2ZShzdHJpY3Q9VHJ1ZSkgIT0gc3BsaXRfcGF0aDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJjYW1wYWlnbiBzcGxpdCBwYXRoIGNhbm5vdCB0cmF2ZXJzZSBhIHN5bWxpbmsiKQogICAgcmVnaXN0ZXJlZF9iZWZvcmVfc3BsaXQgPSB7CiAgICAgICAgcmVsYXRpdmU6IG9ic2VydmVfcmVnaXN0ZXJlZF9zb3VyY2UoZGF0YV9yb290LCByZWxhdGl2ZSwgcmVnaXN0ZXJlZF9zb3VyY2VzW3JlbGF0aXZlXSkKICAgICAgICBmb3IgcmVsYXRpdmUgaW4gRVhQRUNURURfUE9SVEFCTEVfU09VUkNFX05BTUVTCiAgICB9CiAgICBfcHJlZmxpZ2h0X3NwbGl0X2lucHV0cyhzcGxpdF9wYXRoLCBkYXRhX3Jvb3QsIHJlZ2lzdGVyZWRfc291cmNlcykKICAgIHNwbGl0ID0gaGVscGVycy5sb2FkX2NhbXBhaWduX3NwbGl0KHNwbGl0X3BhdGgsIGRhdGFfcm9vdCkKICAgIHJlZ2lzdGVyZWRfYWZ0ZXJfc3BsaXQgPSB7CiAgICAgICAgcmVsYXRpdmU6IG9ic2VydmVfcmVnaXN0ZXJlZF9zb3VyY2UoZGF0YV9yb290LCByZWxhdGl2ZSwgcmVnaXN0ZXJlZF9zb3VyY2VzW3JlbGF0aXZlXSkKICAgICAgICBmb3IgcmVsYXRpdmUgaW4gRVhQRUNURURfUE9SVEFCTEVfU09VUkNFX05BTUVTCiAgICB9CiAgICBpZiByZWdpc3RlcmVkX2JlZm9yZV9zcGxpdCAhPSByZWdpc3RlcmVkX2FmdGVyX3NwbGl0OgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIm9uZSBvciBtb3JlIHJlZ2lzdGVyZWQgZGF0YSBpZGVudGl0aWVzIGNoYW5nZWQgZHVyaW5nIHNwbGl0IHZhbGlkYXRpb24iKQogICAgY3VycmVudF9yZWdpc3RyeV9waW4sIGN1cnJlbnRfcmVnaXN0cnlfaWRlbnRpdHkgPSBfcGluX2ZpbGUocG9ydGFibGVfcmVnaXN0cnksIDJfMDAwXzAwMCkKICAgIGlmIChjdXJyZW50X3JlZ2lzdHJ5X3BpbiwgY3VycmVudF9yZWdpc3RyeV9pZGVudGl0eSkgIT0gKAogICAgICAgIHBvcnRhYmxlX3JlZ2lzdHJ5X3BpbiwKICAgICAgICBwb3J0YWJsZV9yZWdpc3RyeV9pZGVudGl0eSwKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigicG9ydGFibGUgc291cmNlIHJlZ2lzdHJ5IGNoYW5nZWQgZHVyaW5nIGNhbm9uaWNhbCBzcGxpdCB2YWxpZGF0aW9uIikKICAgIGdyb3VwcyA9IFtpbnQoZ3JvdXApIGZvciBncm91cCBpbiBzcGxpdC5ldmFsdWF0aW9uX2dyb3Vwc10KICAgIGlmIHNwbGl0LnNoYTI1NiAhPSBFWFBFQ1RFRF9TUExJVF9TSEEyNTYgb3Igc3BsaXQuZm9sZCAhPSAwIG9yIHNwbGl0Lm5fZm9sZHMgIT0gNDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJjYW5vbmljYWwgc3BsaXQgbG9hZGVyIHJldHVybmVkIGEgZGlmZmVyZW50IEYwIGlkZW50aXR5IikKICAgIHJlcXVpcmVfb3JkZXJlZF9ncm91cHMoZ3JvdXBzLCBtYW5pZmVzdC5nZXQoIm9yZGVyZWRfZ3JvdXBzIikpCiAgICBpZiBsZW4oZ3JvdXBzKSAhPSBOX0dST1VQUzoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJjYW5vbmljYWwgRjAgZXZhbHVhdGlvbiBtZW1iZXJzaGlwIGlzIG5vdCBleGFjdGx5IDQ2OCBncm91cHMiKQogICAgdmFsaWRfY29tcG9uZW50X2dyb3VwcyA9IFsKICAgICAgICBpbnQoZ3JvdXApIGZvciBzb3VyY2UsIGdyb3VwLCBfY29tcG9uZW50IGluIHNwbGl0LmNvbXBvbmVudHMgaWYgc291cmNlID09ICJ2YWxpZCIKICAgIF0KICAgIHZhbGlkX3Jvd19ncm91cF9jb3VudCA9IGJpbmRpbmdbInNwbGl0Il1bInZhbGlkX3Jvd19ncm91cF9jb3VudCJdCiAgICBpZiAoCiAgICAgICAgbGVuKHZhbGlkX2NvbXBvbmVudF9ncm91cHMpICE9IHZhbGlkX3Jvd19ncm91cF9jb3VudAogICAgICAgIG9yIGxlbihzZXQodmFsaWRfY29tcG9uZW50X2dyb3VwcykpICE9IHZhbGlkX3Jvd19ncm91cF9jb3VudAogICAgICAgIG9yIHNldCh2YWxpZF9jb21wb25lbnRfZ3JvdXBzKSAhPSBzZXQocmFuZ2UodmFsaWRfcm93X2dyb3VwX2NvdW50KSkKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigKICAgICAgICAgICAgImNhbm9uaWNhbCB2YWxpZCBjb21wb25lbnQgaW52ZW50b3J5IGRpZmZlcnMgZnJvbSB0aGUgZml4ZWQgcm93LWdyb3VwIGNvdW50IgogICAgICAgICkKICAgIGNvbXBvbmVudF9ieV9ncm91cCA9IHsKICAgICAgICBpbnQoZ3JvdXApOiBpbnQoY29tcG9uZW50KQogICAgICAgIGZvciBzb3VyY2UsIGdyb3VwLCBjb21wb25lbnQgaW4gc3BsaXQuY29tcG9uZW50cwogICAgICAgIGlmIHNvdXJjZSA9PSAidmFsaWQiCiAgICB9CiAgICBwcmVkaWN0aW9ucywgcHJlZGljdGlvbl9waW5zLCBwcmVkaWN0aW9uX2lkZW50aXRpZXMgPSBfb3Blbl9wcmVkaWN0aW9ucygKICAgICAgICBhcmdzLmV4cG9ydF9yb290LCBiaW5kaW5nLCBtYW5pZmVzdAogICAgKQogICAgcmVnaXN0ZXJlZF9iZWZvcmVfc2NvcmluZyA9IHsKICAgICAgICByZWxhdGl2ZTogb2JzZXJ2ZV9yZWdpc3RlcmVkX3NvdXJjZShkYXRhX3Jvb3QsIHJlbGF0aXZlLCByZWdpc3RlcmVkX3NvdXJjZXNbcmVsYXRpdmVdKQogICAgICAgIGZvciByZWxhdGl2ZSBpbiBFWFBFQ1RFRF9QT1JUQUJMRV9TT1VSQ0VfTkFNRVMKICAgIH0KICAgIGlmIHJlZ2lzdGVyZWRfYWZ0ZXJfc3BsaXQgIT0gcmVnaXN0ZXJlZF9iZWZvcmVfc2NvcmluZzoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJyZWdpc3RlcmVkIGRhdGEgaWRlbnRpdHkgY2hhbmdlZCBiZWZvcmUgc2NvcmluZyBiZWdhbiIpCiAgICB2YWxpZF9wYXRoID0gUGF0aChyZWdpc3RlcmVkX3NvdXJjZXNbImRhdGFzZXRzL3ZhbGlkLnBhcnF1ZXQiXVsicGF0aCJdKQogICAgc2VxdWVuY2VfZmlsZSA9IGhlbHBlcnMuc2VxdWVuY2VfZmlsZSh2YWxpZF9wYXRoKQogICAgdHJ5OgogICAgICAgIGlmICgKICAgICAgICAgICAgbGVuKHNlcXVlbmNlX2ZpbGUpICE9IHZhbGlkX3Jvd19ncm91cF9jb3VudAogICAgICAgICAgICBvciBub3Qgc2VxdWVuY2VfZmlsZS5oYXNfdGFyZ2V0cwogICAgICAgICAgICBvciBub3Qgc2VxdWVuY2VfZmlsZS5oYXNfbWFzawogICAgICAgICk6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoCiAgICAgICAgICAgICAgICAicmVnaXN0ZXJlZCB2YWxpZGF0aW9uIGZpbGUgbGFja3MgY29tcGxldGUgZ3JvdXBzLCB0YXJnZXRzLCBvciBzY29yZWQgbWFzayIKICAgICAgICAgICAgKQogICAgICAgIHNjb3JlZCA9IHNjb3JlX3N0cmVhbShncm91cHMsIGNvbXBvbmVudF9ieV9ncm91cCwgcHJlZGljdGlvbnMsIHNlcXVlbmNlX2ZpbGUsIGhlbHBlcnMpCiAgICBmaW5hbGx5OgogICAgICAgIHNlcXVlbmNlX2ZpbGUuX3BmLmNsb3NlKCkKICAgIHJlZ2lzdGVyZWRfYWZ0ZXIgPSB7CiAgICAgICAgcmVsYXRpdmU6IG9ic2VydmVfcmVnaXN0ZXJlZF9zb3VyY2UoZGF0YV9yb290LCByZWxhdGl2ZSwgcmVnaXN0ZXJlZF9zb3VyY2VzW3JlbGF0aXZlXSkKICAgICAgICBmb3IgcmVsYXRpdmUgaW4gRVhQRUNURURfUE9SVEFCTEVfU09VUkNFX05BTUVTCiAgICB9CiAgICBpZiByZWdpc3RlcmVkX2JlZm9yZV9zY29yaW5nICE9IHJlZ2lzdGVyZWRfYWZ0ZXI6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigib25lIG9yIG1vcmUgcmVnaXN0ZXJlZCBkYXRhIGlkZW50aXRpZXMgY2hhbmdlZCB3aGlsZSBzY29yaW5nIikKICAgIGZpbmFsX3JlZ2lzdHJ5X3BpbiwgZmluYWxfcmVnaXN0cnlfaWRlbnRpdHkgPSBfcGluX2ZpbGUocG9ydGFibGVfcmVnaXN0cnksIDJfMDAwXzAwMCkKICAgIGlmIChmaW5hbF9yZWdpc3RyeV9waW4sIGZpbmFsX3JlZ2lzdHJ5X2lkZW50aXR5KSAhPSAoCiAgICAgICAgcG9ydGFibGVfcmVnaXN0cnlfcGluLAogICAgICAgIHBvcnRhYmxlX3JlZ2lzdHJ5X2lkZW50aXR5LAogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJwb3J0YWJsZSBzb3VyY2UgcmVnaXN0cnkgY2hhbmdlZCB3aGlsZSBzY29yaW5nIikKICAgIG1hbmlmZXN0X3BhdGggPSBhcmdzLmV4cG9ydF9yb290IC8gIkVYUE9SVF9NQU5JRkVTVF9WMS5qc29uIgogICAgZmluYWxfbWFuaWZlc3RfcGluLCBmaW5hbF9tYW5pZmVzdF9pZGVudGl0eSA9IF9waW5fZmlsZSgKICAgICAgICBtYW5pZmVzdF9wYXRoLCBNQVhfRVhQT1JUX01BTklGRVNUX0JZVEVTCiAgICApCiAgICBpZiBmaW5hbF9tYW5pZmVzdF9waW4gIT0gbWFuaWZlc3RfcGluIG9yIGZpbmFsX21hbmlmZXN0X2lkZW50aXR5ICE9IG1hbmlmZXN0X2lkZW50aXR5OgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImV4cG9ydCBjb21wbGV0aW9uIG1hbmlmZXN0IGNoYW5nZWQgd2hpbGUgYmVpbmcgc2NvcmVkIikKICAgIGZvciBpdGVtIGluIHByZWRpY3Rpb25faWRlbnRpdGllczoKICAgICAgICBpdGVtX3BhdGggPSAoCiAgICAgICAgICAgIGFyZ3MuZXhwb3J0X3Jvb3QgLyBpdGVtWyJhcm0iXSAvICJQUkVESUNUSU9OX1NJREVDQVIuanNvbiIKICAgICAgICAgICAgaWYgaXRlbVsia2luZCJdID09ICJzaWRlY2FyIgogICAgICAgICAgICBlbHNlIGFyZ3MuZXhwb3J0X3Jvb3QgLyBpdGVtWyJhcm0iXSAvICJwcmVkaWN0aW9ucy5ucHkiCiAgICAgICAgKQogICAgICAgIGxpbmtlZCA9IGl0ZW1fcGF0aC5sc3RhdCgpCiAgICAgICAgaWYgX3N0YWJsZV9pZGVudGl0eShsaW5rZWQsIGxpbmtlZCkgIT0gaXRlbVsiaWRlbnRpdHkiXToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntpdGVtWydhcm0nXX0ge2l0ZW1bJ2tpbmQnXX0gY2hhbmdlZCBkdXJpbmcgc2NvcmluZyIpCiAgICBwcmVkaWN0aW9ucy5jbGVhcigpCiAgICBmaXhlZF9tb2RlbHMgPSBbCiAgICAgICAgewogICAgICAgICAgICAiYXJtIjogYXJtWyJhcm0iXSwKICAgICAgICAgICAgInJ1bl9uYW1lIjogYXJtWyJydW5fbmFtZSJdLAogICAgICAgICAgICAidW5pdCI6IGFybVsidW5pdCJdLAogICAgICAgICAgICAiaW52b2NhdGlvbl9pZCI6IGFybVsiaW52b2NhdGlvbl9pZCJdLAogICAgICAgICAgICAicmVwcmVzZW50YXRpb25fbW9kZSI6IGFybVsicmVwcmVzZW50YXRpb25fbW9kZSJdLAogICAgICAgICAgICAic2VsZWN0ZWRfZXBvY2giOiA1LAogICAgICAgICAgICAic3VjY2Vzc2Z1bF91cGRhdGVzIjogNDQ3NjAsCiAgICAgICAgICAgICJjaGVja3BvaW50IjogZGljdChhcm1bIm1vZGVsIl0pLAogICAgICAgICAgICAidGVybWluYWxfYWNjZXB0YW5jZSI6IGRpY3QoYXJtWyJ0ZXJtaW5hbF9hY2NlcHRhbmNlIl0pLAogICAgICAgICAgICAiZXhhY3Rfc3RvcCI6IGRpY3QoYXJtWyJleGFjdF9zdG9wIl0pLAogICAgICAgIH0KICAgICAgICBmb3IgYXJtIGluIGJpbmRpbmdbImFybXMiXQogICAgXQogICAgcmV0dXJuIHsKICAgICAgICAic2NoZW1hIjogImZnMTkuZmlyc3QtY2VsbC1maXZlLWFybS1zdHJlYW1pbmctc2NvcmUudjEiLAogICAgICAgICJzdGF0dXMiOiAiRklYRURfRklSU1RfQ0VMTF9ERVZFTE9QTUVOVF9FVklERU5DRV9TQ09SRUQiLAogICAgICAgICJjcmVhdGVkX3V0YyI6IGR0LmRhdGV0aW1lLm5vdyhkdC5VVEMpLmlzb2Zvcm1hdCgpLAogICAgICAgICJzY29wZSI6ICJPbmx5IHRoZSByZWdpc3RlcmVkIEYwL3NlZWQtMTAwMSBmaXZlLWFybSBwYW5lbCBhdCBmaXhlZCBlcG9jaCA1OyBhbGwgNDY4IGZpdC1leGNsdWRlZCBleHBvc2VkIGRldmVsb3BtZW50IGdyb3Vwcy4iLAogICAgICAgICJzb3VyY2VfY29tbWl0IjogU09VUkNFX0NPTU1JVCwKICAgICAgICAic291cmNlX21hbmlmZXN0X3NoYTI1NiI6IEVYUEVDVEVEX1NPVVJDRV9NQU5JRkVTVF9TSEEyNTYsCiAgICAgICAgInNvdXJjZV9zZXRfc2hhMjU2IjogRVhQRUNURURfU09VUkNFX1NFVF9TSEEyNTYsCiAgICAgICAgInNwbGl0X21hbmlmZXN0X3NoYTI1NiI6IHNwbGl0LnNoYTI1NiwKICAgICAgICAib3JkZXJlZF9tZW1iZXJzaGlwX3NoYTI1NiI6IG1hbmlmZXN0LmdldCgibWVtYmVyc2hpcF9zaGEyNTYiKSwKICAgICAgICAiZml4ZWRfbW9kZWxzIjogZml4ZWRfbW9kZWxzLAogICAgICAgICJleHBvcnRfbWFuaWZlc3QiOiBtYW5pZmVzdF9waW4sCiAgICAgICAgImV4cG9ydGVkX2ZpbGVfcGlucyI6IHByZWRpY3Rpb25fcGlucywKICAgICAgICAicmVnaXN0ZXJlZF9kYXRhIjogewogICAgICAgICAgICAiYWRtaXNzaW9uIjogcmVnaXN0ZXJlZF9kYXRhWyJhZG1pc3Npb24iXSwKICAgICAgICAgICAgInBvcnRhYmxlX21hbmlmZXN0IjogcmVnaXN0ZXJlZF9kYXRhWyJwb3J0YWJsZV9tYW5pZmVzdCJdLAogICAgICAgICAgICAicmVnaXN0ZXJlZF9zb3VyY2VfaWRlbnRpdGllcyI6IHJlZ2lzdGVyZWRfc291cmNlcywKICAgICAgICAgICAgIm9ic2VydmVkX2JlZm9yZV9zcGxpdCI6IHJlZ2lzdGVyZWRfYmVmb3JlX3NwbGl0LAogICAgICAgICAgICAib2JzZXJ2ZWRfYWZ0ZXJfc3BsaXQiOiByZWdpc3RlcmVkX2FmdGVyX3NwbGl0LAogICAgICAgICAgICAib2JzZXJ2ZWRfYmVmb3JlX3Njb3JpbmciOiByZWdpc3RlcmVkX2JlZm9yZV9zY29yaW5nLAogICAgICAgICAgICAib2JzZXJ2ZWRfYWZ0ZXJfc2NvcmluZyI6IHJlZ2lzdGVyZWRfYWZ0ZXIsCiAgICAgICAgfSwKICAgICAgICAic2NvcmluZyI6IHNjb3JlZCwKICAgICAgICAibGltaXRhdGlvbnMiOiBbCiAgICAgICAgICAgICJUaGUgNDY4IEYwIGdyb3VwcyBhcmUgZml0LWV4Y2x1ZGVkIGJ1dCByZXBlYXRlZGx5IGV4cG9zZWQgZGV2ZWxvcG1lbnQgZXZpZGVuY2UsIG5vdCBhIGZyZXNoIG9yIHB1YmxpYyBob2xkb3V0LiIsCiAgICAgICAgICAgICJDb21wb25lbnQgYm9vdHN0cmFwIGludGVydmFscyBkZXNjcmliZSBvbmx5IGdyb3VwL2NvbXBvbmVudCByZXNhbXBsaW5nIGNvbmRpdGlvbmFsIG9uIHRoZXNlIGZpeGVkIG1vZGVsczsgdGhleSBkbyBub3QgZXN0aW1hdGUgZml0LXNlZWQgb3IgcHVibGljLXRyYW5zZmVyIHVuY2VydGFpbnR5LiIsCiAgICAgICAgICAgICJObyBwcmVkaWN0aW9ucyB3ZXJlIGdlbmVyYXRlZCBieSB0aGlzIHNjb3JlcjsgaXQgY29uc3VtZXMgb25seSB0aGUgcmV2aWV3ZWQgZml4ZWQgZXBvY2gtNSBleHBvcnQuIiwKICAgICAgICBdLAogICAgfQoKCmRlZiBtYWluKGFyZ3Y6IGxpc3Rbc3RyXSB8IE5vbmUgPSBOb25lKSAtPiBpbnQ6CiAgICAiIiJWYWxpZGF0ZSBhbmQgc2NvcmUgdGhlIGZpeGVkIGZpcnN0LWNlbGwgZXhwb3J0IGludG8gYSBjcmVhdGUtb25seSByZXBvcnQuIiIiCiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcihkZXNjcmlwdGlvbj1fX2RvY19fKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1ldmlkZW5jZS1yb290IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1leHBvcnQtcm9vdCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYmluZGluZyIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tc291cmNlLW1hbmlmZXN0IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zb3VyY2Utcm9vdCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tZGF0YS1yb290IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1vdXQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBhcmdzID0gcGFyc2VyLnBhcnNlX2FyZ3MoYXJndikKICAgIHJlc3VsdCA9IF9zY29yZV9idW5kbGUoYXJncykKICAgIGVuY29kZWQgPSBqc29uLmR1bXBzKHJlc3VsdCwgZW5zdXJlX2FzY2lpPUZhbHNlLCBpbmRlbnQ9MikgKyAiXG4iCiAgICBkZXNjcmlwdG9yID0gb3Mub3BlbigKICAgICAgICBhcmdzLm91dCwKICAgICAgICBvcy5PX1dST05MWSB8IG9zLk9fQ1JFQVQgfCBvcy5PX0VYQ0wgfCBnZXRhdHRyKG9zLCAiT19OT0ZPTExPVyIsIDApLAogICAgICAgIDBvNjAwLAogICAgKQogICAgd2l0aCBvcy5mZG9wZW4oZGVzY3JpcHRvciwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBzdHJlYW06CiAgICAgICAgc3RyZWFtLndyaXRlKGVuY29kZWQpCiAgICAgICAgc3RyZWFtLmZsdXNoKCkKICAgICAgICBvcy5mc3luYyhzdHJlYW0uZmlsZW5vKCkpCiAgICBkaXJlY3RvcnlfZGVzY3JpcHRvciA9IG9zLm9wZW4oCiAgICAgICAgYXJncy5vdXQucGFyZW50LAogICAgICAgIG9zLk9fUkRPTkxZIHwgZ2V0YXR0cihvcywgIk9fRElSRUNUT1JZIiwgMCkgfCBnZXRhdHRyKG9zLCAiT19OT0ZPTExPVyIsIDApLAogICAgKQogICAgdHJ5OgogICAgICAgIG9zLmZzeW5jKGRpcmVjdG9yeV9kZXNjcmlwdG9yKQogICAgZmluYWxseToKICAgICAgICBvcy5jbG9zZShkaXJlY3RvcnlfZGVzY3JpcHRvcikKICAgIHByaW50KAogICAgICAgIGpzb24uZHVtcHMoCiAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICJzdGF0dXMiOiByZXN1bHRbInN0YXR1cyJdLAogICAgICAgICAgICAgICAgImFybXMiOiBsaXN0KHJlc3VsdFsic2NvcmluZyJdWyJhcm1zIl0pLAogICAgICAgICAgICAgICAgImNvbnRyYXN0cyI6IGxpc3QocmVzdWx0WyJzY29yaW5nIl1bInBhaXJlZF9jb250cmFzdHNfdnNfY29udHJvbCJdKSwKICAgICAgICAgICAgICAgICJvdXRwdXQiOiBzdHIoYXJncy5vdXQpLAogICAgICAgICAgICB9CiAgICAgICAgKQogICAgKQogICAgcmV0dXJuIDAKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgcmFpc2UgU3lzdGVtRXhpdChtYWluKCkpCg==',
    'EXPORT_ACCEPTANCE.json': 'ewogICJzY2hlbWEiOiAiZmcxOS5maXJzdC1jZWxsLWV4cG9ydC1hY2NlcHRhbmNlLnYxIiwKICAic3RhdHVzIjogIkFDQ0VQVEVEX0ZJWEVEX0ZJVkVfUFJFRElDVElPTl9FWFBPUlRTX05PVF9TQ09SRUQiLAogICJjcmVhdGVkX3V0YyI6ICIyMDI2LTEwLTIxVDExOjA3OjQwLjkyNDcyNiswMDowMCIsCiAgInVuaXQiOiAiY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLWZnMTktZmlyc3QtY2VsbC1leHBvcnQtdjMuc2VydmljZSIsCiAgImludm9jYXRpb25faWQiOiAiNzhmYTI1ZDBhYmM0NGQxOTllMzY5NmQxNzJkYmM0ZjkiLAogICJuYXRpdmVfdGVybWluYWwiOiB7CiAgICAiYWN0aXZlX3N0YXRlIjogImFjdGl2ZSIsCiAgICAic3ViX3N0YXRlIjogImV4aXRlZCIsCiAgICAibWFpbl9waWQiOiAwLAogICAgImV4ZWNfbWFpbl9jb2RlIjogMSwKICAgICJleGVjX21haW5fc3RhdHVzIjogMAogIH0sCiAgInNvdXJjZV9yZXZpZXciOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcHJlcGFyYXRpb24vZmlyc3RfY2VsbF9ldmFsdWF0aW9uX2V4cG9ydF92NC9FWFBPUlRfU09VUkNFX1JFVklFV19WNC5qc29uIiwKICAgICJieXRlcyI6IDg3MTIsCiAgICAic2hhMjU2IjogIjY0YzkzZDgyNzlhMmRhZWIxNWYyNzUwZTYzNjZhMGZlOTRhMjY4YzdiOGU5YTA0ZGJkYjI4ZTRhMzZhNTZlNzMiCiAgfSwKICAiYmluZGluZyI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9wcmVwYXJhdGlvbi9maXJzdF9jZWxsX2V2YWx1YXRpb25fZXhwb3J0X3Y0L0VYUE9SVF9CSU5ESU5HX1Y0Lmpzb24iLAogICAgImJ5dGVzIjogMTU3OTYsCiAgICAic2hhMjU2IjogIjRhOGFlMjIyMDQ5NzFhYzA3ZDZlNWNkZmQ5ZjY1YTYyZTlmODQ0MDgzMTAzNWRiNzI3MWU0ZTBkYTNkODNhYmQiCiAgfSwKICAibGF1bmNoX2FjY2VwdGFuY2UiOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcmVjb3Jkcy9maXJzdF9jZWxsX2V2YWx1YXRpb25fdjEvZXhwb3J0X3Y1L0xBVU5DSF9BQ0NFUFRBTkNFLmpzb24iLAogICAgImJ5dGVzIjogMjIzOCwKICAgICJzaGEyNTYiOiAiYzQzZmY0YWU0Yzg4ZWIyMGM1YzQwOGU4MzA3ZDhmZTNiZGEzODJkOWVjNzY2NDhhYTUyZWYwZjE3ODFjNTY1MCIKICB9LAogICJsYXVuY2hfcmF3X2NhcHR1cmUiOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcmVjb3Jkcy9maXJzdF9jZWxsX2V2YWx1YXRpb25fdjEvZXhwb3J0X3Y1L1NURE9VVC5yYXcuanNvbiIsCiAgICAiYnl0ZXMiOiAzODUyOCwKICAgICJzaGEyNTYiOiAiYTEzZGNmN2EwMjBmMDgyZDIzNjdmMDA5YWRjM2NhYTI4ZDUzOTZlYTRjNWU4NTY2MzM4YTk2YWIzNDcwYTg5OCIKICB9LAogICJsYXVuY2hfcmF3X3RyYW5zcG9ydCI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvVFJBTlNQT1JULmpzb24iLAogICAgImJ5dGVzIjogNzMyLAogICAgInNoYTI1NiI6ICJjMmNjMzNiYTQwNjJmNDg0ZmQ5MTQ2Zjg2MzMyYzk4MGNiODE5YWZmZTYyY2I2NTdjZDMxNjE0NzUyMzhjMTA1IgogIH0sCiAgImxhdW5jaF9yZWNlaXB0cyI6IHsKICAgICJBRE1JU1NJT04uanNvbiI6IHsKICAgICAgImJ5dGVzIjogMjQ2NjAsCiAgICAgICJzaGEyNTYiOiAiMjZhYzljMDhjOGJhNjMyYTgxNjM0MDc1ODRiOWZjNmVmOWRmZTAzNDViYzUzNTlkZDZmNWRkZDc0NTEwYWJhMSIKICAgIH0sCiAgICAiTEFVTkNILmpzb24iOiB7CiAgICAgICJieXRlcyI6IDQxMzAsCiAgICAgICJzaGEyNTYiOiAiM2U5ODQ3NjEwMDVjNmM1NmVmYjA1NThmNDMyOTdiNWQzMGMwYWUwZmIwN2ZhODU5NTg5NTFjNGViZjNlMmIzNyIKICAgIH0KICB9LAogICJzdGF0dXNfY2FwdHVyZV9yb290IjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvc3RhdHVzXzIwMjYxMDIxVDExMDdaIiwKICAibmF0aXZlX3Jhd19jYXB0dXJlIjogewogICAgInBhdGgiOiAiL1VzZXJzL25iL0RvY3VtZW50cy93dW5kZXIvV3VuZGVyRnVuZGVyLW1haW50ZW5hbmNlL3Jlc2VhcmNoL2NhbXBhaWduXzIwMjYtMTAtMTkvZnVsbF9ncnVfdHdlYWtzX3IxL3JlY29yZHMvZmlyc3RfY2VsbF9ldmFsdWF0aW9uX3YxL2V4cG9ydF92NS9zdGF0dXNfMjAyNjEwMjFUMTEwN1ovU1RET1VULnJhdy5qc29uIiwKICAgICJieXRlcyI6IDE0MzE4OCwKICAgICJzaGEyNTYiOiAiMDRlMTJiNjI0ZDMyNWFlOTNjMTFjMjQxYzljZDEwMTFjZWIyZWY3YWNkMGFlZDdkMDkyNTJhNzZhODM0YWIxYSIKICB9LAogICJuYXRpdmVfcmF3X3RyYW5zcG9ydCI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvc3RhdHVzXzIwMjYxMDIxVDExMDdaL1RSQU5TUE9SVC5qc29uIiwKICAgICJieXRlcyI6IDYwOSwKICAgICJzaGEyNTYiOiAiMmVjYTI5YTNmNmRiZTA3NjFlNWJjOTc0NDExMGIwMTFkOTc4YmZmZjVmODI3MTMwYzRjZjE2ZDlhZWM4OGQ2MyIKICB9LAogICJzdGF0dXNfc3RkZXJyX2NhcHR1cmUiOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcmVjb3Jkcy9maXJzdF9jZWxsX2V2YWx1YXRpb25fdjEvZXhwb3J0X3Y1L3N0YXR1c18yMDI2MTAyMVQxMTA3Wi9TVERFUlIucmF3LnR4dCIsCiAgICAiYnl0ZXMiOiAwLAogICAgInNoYTI1NiI6ICJlM2IwYzQ0Mjk4ZmMxYzE0OWFmYmY0Yzg5OTZmYjkyNDI3YWU0MWU0NjQ5YjkzNGNhNDk1OTkxYjc4NTJiODU1IgogIH0sCiAgInN0YXR1c19yZWFkZXJfYm9keSI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvc3RhdHVzXzIwMjYxMDIxVDExMDdaL1JFTU9URS5weSIsCiAgICAiYnl0ZXMiOiAxNDg4MywKICAgICJzaGEyNTYiOiAiNGUzNWU2MjExZjY2OGFkNmFlNGY5NDY2MTE5Zjk5NTQ0ZjQyZTNmY2E0NWJhOTEzMzdjNjA5OGRhNTRmZjMyOSIKICB9LAogICJzdGF0dXNfcmVhZGVyX2FzdF9iaW5kaW5nIjogewogICAgInBhdGgiOiAiL1VzZXJzL25iL0RvY3VtZW50cy93dW5kZXIvV3VuZGVyRnVuZGVyLW1haW50ZW5hbmNlL3Jlc2VhcmNoL2NhbXBhaWduXzIwMjYtMTAtMTkvZnVsbF9ncnVfdHdlYWtzX3IxL3JlY29yZHMvZmlyc3RfY2VsbF9ldmFsdWF0aW9uX3YxL2V4cG9ydF92NS9zdGF0dXNfMjAyNjEwMjFUMTEwN1ovQVNUX0JJTkRJTkcuanNvbiIsCiAgICAiYnl0ZXMiOiA4NTgsCiAgICAic2hhMjU2IjogIjFjMjIwOGYxOTllZDdkOTg3NWJkZjg4ZmU1ZDgyM2Y5YTkxY2MyZjc2NGU1MTBmOGEwNzJjZGNmMzJlNTU2NjYiCiAgfSwKICAic3RhdHVzX2RyaXZlcl9yZXZpZXciOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcHJlcGFyYXRpb24vZmlyc3RfY2VsbF9ldmFsdWF0aW9uX2FkbWlzc2lvbl92MS9ST09UX0NBUFRVUkVfRVhQT1JUX1NUQVRVU19TT1VSQ0VfUkVWSUVXX1Y1Lmpzb24iLAogICAgImJ5dGVzIjogMzU2NCwKICAgICJzaGEyNTYiOiAiZDIxNGY3NzVhYTcxZGU5NjJiOWFmNWIxYjU3MDdjY2IzMzliNTEzMmFlNjdlMDA0ZTA4MmY1ZmMyMjA0MGVjZCIKICB9LAogICJzdGF0dXNfZHJpdmVyX3Jldmlld19waW5fY29ycmVjdGlvbiI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9wcmVwYXJhdGlvbi9maXJzdF9jZWxsX2V2YWx1YXRpb25fYWRtaXNzaW9uX3YxL1JPT1RfQ0FQVFVSRV9FWFBPUlRfU1RBVFVTX1Y1X1JFVklFV19QSU5fQ09SUkVDVElPTl9WMS5qc29uIiwKICAgICJieXRlcyI6IDIxNTcsCiAgICAic2hhMjU2IjogIjlkMTczN2U5YTEzZmU5YTVkN2JhODFhNjc1MzE5OGMwNDczMDFjYmQ4MGYyNTliYjJhMDFiYzIyNTY0ZWMyYWIiCiAgfSwKICAibWFuaWZlc3RfY2FwdHVyZSI6IHsKICAgICJieXRlcyI6IDIzMTMwLAogICAgInNoYTI1NiI6ICI2YjkzY2FiZjBlMGI0NDUwYTExODIzNjgyZGZlYmE5Mzk2MDc2ZWZiYzdiNGEyMDMzNjczZmM0NmI3ZjA0YzIzIgogIH0sCiAgIm1lbWJlcnNoaXBfc2hhMjU2IjogIjE4ZjBjOTUxNGQyNWU5NGNiZTVlYjI0NjA3ODc5MzI0YmI4M2QxMTdkOTYyNjcwZDRkZGY5NzMzNjQ1ZjRmMGMiLAogICJldmFsdWF0aW9uX2dyb3VwcyI6IDQ2OCwKICAib3V0cHV0cyI6IHsKICAgICJjb250cm9sIjogewogICAgICAicHJlZGljdGlvbiI6IHsKICAgICAgICAiYnl0ZXMiOiA3NDg4MDEyOCwKICAgICAgICAiZGF0YV9ieXRlcyI6IDc0ODgwMDAwLAogICAgICAgICJkdHlwZSI6ICJmbG9hdDMyLWxpdHRsZS1lbmRpYW4iLAogICAgICAgICJucHlfaGVhZGVyX2J5dGVzIjogMTI4LAogICAgICAgICJwYXRoIjogInByZWRpY3Rpb25zLm5weSIsCiAgICAgICAgInNoYTI1NiI6ICI2YWYyYjAxODM3ZTc4MmY0YThkMzM0ZWE5MDc4OWU2OTQzMWI2ZDA1OTAzMGFhNWNkYWI5N2FmYjBmZmJmOGM2IiwKICAgICAgICAic2hhcGUiOiBbCiAgICAgICAgICA0NjgsCiAgICAgICAgICAyMDAwMCwKICAgICAgICAgIDIKICAgICAgICBdCiAgICAgIH0sCiAgICAgICJjaGVja3BvaW50IjogewogICAgICAgICJieXRlcyI6IDQyNzk5NzAsCiAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9jb250cm9sX2YwX3MxMDAxX2ZpdF92MS9tb2RlbF9lcDUucHQiLAogICAgICAgICJzaGEyNTYiOiAiZDNmMjVlNTJhYTU5MDAyOTU5M2U1YWY2ZWEwNmM2ZDRmNDYwYmM2MTMxZmZhODVkNGNmMWNiMDljMzZiM2IzZiIsCiAgICAgICAgImxvYWRlZF9zaGEyNTYiOiAiZDNmMjVlNTJhYTU5MDAyOTU5M2U1YWY2ZWEwNmM2ZDRmNDYwYmM2MTMxZmZhODVkNGNmMWNiMDljMzZiM2IzZiIKICAgICAgfQogICAgfSwKICAgICJlbWEiOiB7CiAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICJkYXRhX2J5dGVzIjogNzQ4ODAwMDAsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiAxMjgsCiAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAic2hhMjU2IjogImVkY2QyMjAzODIyNzBlNzRhY2E5OTA0NWE5OGE0MzU3YzFhOGExMTJhOTliNjBhZDBhMjRhYzc4NmE1NmZjZTciLAogICAgICAgICJzaGFwZSI6IFsKICAgICAgICAgIDQ2OCwKICAgICAgICAgIDIwMDAwLAogICAgICAgICAgMgogICAgICAgIF0KICAgICAgfSwKICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX2VtYV9mMF9zMTAwMV9maXRfdjEvbW9kZWxfZXA1LnB0IiwKICAgICAgICAic2hhMjU2IjogIjZlOTBiYzRkYmE4YWRkMGNhZjY3MDA5NzUxOWYwOGYzY2UyMzlkMGI2NGMzN2M0N2IzNzYwYWEyMmFmNDE4NjIiLAogICAgICAgICJsb2FkZWRfc2hhMjU2IjogIjZlOTBiYzRkYmE4YWRkMGNhZjY3MDA5NzUxOWYwOGYzY2UyMzlkMGI2NGMzN2M0N2IzNzYwYWEyMmFmNDE4NjIiCiAgICAgIH0KICAgIH0sCiAgICAic3BhcnNlIjogewogICAgICAicHJlZGljdGlvbiI6IHsKICAgICAgICAiYnl0ZXMiOiA3NDg4MDEyOCwKICAgICAgICAiZGF0YV9ieXRlcyI6IDc0ODgwMDAwLAogICAgICAgICJkdHlwZSI6ICJmbG9hdDMyLWxpdHRsZS1lbmRpYW4iLAogICAgICAgICJucHlfaGVhZGVyX2J5dGVzIjogMTI4LAogICAgICAgICJwYXRoIjogInByZWRpY3Rpb25zLm5weSIsCiAgICAgICAgInNoYTI1NiI6ICJjYjU2ZmNhZWEyNWU2OTUzZDhjZTg0NmM0ZWFkZWVmMWM4N2FiYTJhNjQzZTRkYTI1YzA0ODc3MGRlM2IyMWFiIiwKICAgICAgICAic2hhcGUiOiBbCiAgICAgICAgICA0NjgsCiAgICAgICAgICAyMDAwMCwKICAgICAgICAgIDIKICAgICAgICBdCiAgICAgIH0sCiAgICAgICJjaGVja3BvaW50IjogewogICAgICAgICJieXRlcyI6IDQyNzk5NzAsCiAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9kaWN0aW9uYXJ5X2YwX3MxMDAxX2ZpdF92MS9tb2RlbF9lcDUucHQiLAogICAgICAgICJzaGEyNTYiOiAiYjg0MGQ5YzgxZDQ2NDA1ZjJkMjQzOWI5NGFkNmVmMTVmZjg1MDFiYjE0NzE1ZWM3OTc5MmQ5ODBlZmJmNzE1MSIsCiAgICAgICAgImxvYWRlZF9zaGEyNTYiOiAiYjg0MGQ5YzgxZDQ2NDA1ZjJkMjQzOWI5NGFkNmVmMTVmZjg1MDFiYjE0NzE1ZWM3OTc5MmQ5ODBlZmJmNzE1MSIKICAgICAgfQogICAgfSwKICAgICJubGwiOiB7CiAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICJkYXRhX2J5dGVzIjogNzQ4ODAwMDAsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiAxMjgsCiAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAic2hhMjU2IjogImNhYjZlNWIyZDIyNWM5MjM2ZjcyYjBiYjc0NzAzNGJlODJmYjVlM2Y0YjMxOWM4NjdkMjBmOTEzZTczNTcwZDgiLAogICAgICAgICJzaGFwZSI6IFsKICAgICAgICAgIDQ2OCwKICAgICAgICAgIDIwMDAwLAogICAgICAgICAgMgogICAgICAgIF0KICAgICAgfSwKICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX25sbF9mMF9zMTAwMV9maXRfdjEvbW9kZWxfZXA1LnB0IiwKICAgICAgICAic2hhMjU2IjogImIxOWE2MTM2M2I0YjQ5Y2Q5OGQwMjc3ZjJmNzRmZDNhYTNkYjUwYmI2YzFmY2IzMTRiNWZhYjU3MWUyN2ZkNTgiLAogICAgICAgICJsb2FkZWRfc2hhMjU2IjogImIxOWE2MTM2M2I0YjQ5Y2Q5OGQwMjc3ZjJmNzRmZDNhYTNkYjUwYmI2YzFmY2IzMTRiNWZhYjU3MWUyN2ZkNTgiCiAgICAgIH0KICAgIH0sCiAgICAiYWxsdGhyZWUiOiB7CiAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICJkYXRhX2J5dGVzIjogNzQ4ODAwMDAsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiAxMjgsCiAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAic2hhMjU2IjogImFmYWVjOGRhOGI3NWI1Y2JlYzc2MDVhYjlkOGFlZjFlMzVkYWMzMmUxMmQ0NDRhM2NiMjAzMmI1NmY2Njg1NzUiLAogICAgICAgICJzaGFwZSI6IFsKICAgICAgICAgIDQ2OCwKICAgICAgICAgIDIwMDAwLAogICAgICAgICAgMgogICAgICAgIF0KICAgICAgfSwKICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX2FsbHRocmVlX2YwX3MxMDAxX2ZpdF92MS9tb2RlbF9lcDUucHQiLAogICAgICAgICJzaGEyNTYiOiAiNzA3MjA3NzA5OTc0NGU2OGZmMjNkZTNlMzkwYjE0NjkyN2MyM2RhMGUxYzU0ZjViYjE4ZWIyZmIwNDY3NmMwNyIsCiAgICAgICAgImxvYWRlZF9zaGEyNTYiOiAiNzA3MjA3NzA5OTc0NGU2OGZmMjNkZTNlMzkwYjE0NjkyN2MyM2RhMGUxYzU0ZjViYjE4ZWIyZmIwNDY3NmMwNyIKICAgICAgfQogICAgfQogIH0sCiAgImV4cG9zdXJlIjogImZpdC1leGNsdWRlZCBFWFBPU0VEIGRldmVsb3BtZW50IGdyb3Vwcywgbm90IGEgcHJpc3RpbmUvcHVibGljIGhvbGRvdXQiLAogICJxdWFsaXR5X3Jlc3VsdCI6IG51bGwsCiAgImFycmF5X3BheWxvYWRzX3JlYWRfb3JfaGFzaGVkX2J5X1JPT1QiOiBmYWxzZSwKICAic3RvcF9zdGF0dXMiOiAiUEVORElOR19FWEFDVF9TVE9QIgp9Cg==',
    'EXPORT_EXACT_STOP.json': 'eyJhY2NlcHRhbmNlX3NoYTI1NiI6ICI1YTlhOTkxMjFmODBlYjdmYTA3ZTQ1OTEzMThiZWE4MTA3OGNhOGIyZGIxMDMyZjE1MGNjMzNhZmNjNzI4ZDBkIiwgImFmdGVyIjogeyJBY3RpdmVTdGF0ZSI6ICJpbmFjdGl2ZSIsICJDb250cm9sR3JvdXAiOiAiIiwgIkV4ZWNNYWluQ29kZSI6ICIwIiwgIkV4ZWNNYWluU3RhdHVzIjogIjAiLCAiSWQiOiAiY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLWZnMTktZmlyc3QtY2VsbC1leHBvcnQtdjMuc2VydmljZSIsICJJbnZvY2F0aW9uSUQiOiAiIiwgIkxvYWRTdGF0ZSI6ICJub3QtZm91bmQiLCAiTWFpblBJRCI6ICIwIiwgIlJlc3VsdCI6ICJzdWNjZXNzIiwgIlN1YlN0YXRlIjogImRlYWQifSwgImJlZm9yZSI6IHsiQWN0aXZlU3RhdGUiOiAiYWN0aXZlIiwgIkNvbnRyb2xHcm91cCI6ICIiLCAiRXhlY01haW5Db2RlIjogIjEiLCAiRXhlY01haW5TdGF0dXMiOiAiMCIsICJJZCI6ICJjb21wdXRlci12aXNpb24taG9tZXdvcmstZmcxOS1maXJzdC1jZWxsLWV4cG9ydC12My5zZXJ2aWNlIiwgIkludm9jYXRpb25JRCI6ICI3OGZhMjVkMGFiYzQ0ZDE5OWUzNjk2ZDE3MmRiYzRmOSIsICJMb2FkU3RhdGUiOiAibG9hZGVkIiwgIk1haW5QSUQiOiAiMCIsICJSZXN1bHQiOiAic3VjY2VzcyIsICJTdWJTdGF0ZSI6ICJleGl0ZWQifSwgImNhcHR1cmVkX3V0YyI6ICIyMDI2LTEwLTIxVDExOjE2OjA4LjI4NjMzNCswMDowMCIsICJpbnZvY2F0aW9uX2lkIjogIjc4ZmEyNWQwYWJjNDRkMTk5ZTM2OTZkMTcyZGJjNGY5IiwgInBvcHVsYXRlZF9hZnRlciI6IHt9LCAicG9wdWxhdGVkX2JlZm9yZSI6IHt9LCAic2NoZW1hIjogImZnMTktZmlyc3QtY2VsbC1leHBvcnQtZXhhY3Qtb3duZWQtc3RvcC52MSIsICJzY29wZSI6ICJPbmx5IHRoZSBvbmUgc3VjY2Vzc2Z1bCBleGl0ZWQgZXhwb3J0IHVuaXQuIEFsbCBvbGQgbmF0aXZlL3N1cHBvcnQgYW5kIGZvcmVpZ24gdW5pdHMgZXhjbHVkZWQ7IG5vIGZpbGUgb3IgcGF5bG9hZCBtdXRhdGlvbi4iLCAic3RvcF9yZXR1cm5jb2RlIjogMCwgInN0b3Bfc3RkZXJyIjogIiIsICJzdG9wX3N0ZG91dCI6ICIiLCAidW5pdCI6ICJjb21wdXRlci12aXNpb24taG9tZXdvcmstZmcxOS1maXJzdC1jZWxsLWV4cG9ydC12My5zZXJ2aWNlIn0K',
}


def validate_export_authority(
    acceptance: dict[str, object],
    exact_stop: dict[str, object],
    *,
    acceptance_sha256: str,
    binding_sha256: str,
    review_sha256: str,
    expected_unit: str,
) -> str:
    """Validate a root-owned fixed-export acceptance and its exact stop receipt."""
    if acceptance.get("schema") != "fg19.first-cell-export-acceptance.v1":
        raise ValueError("export acceptance schema differs")
    if acceptance.get("status") != "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED":
        raise ValueError("fixed prediction export is not root-accepted")
    unit, invocation = acceptance.get("unit"), acceptance.get("invocation_id")
    if unit != expected_unit or not is_invocation_id(invocation):
        raise ValueError("export acceptance unit/invocation differs")
    native = acceptance.get("native_terminal")
    if not isinstance(native, dict) or set(native) != {
        "active_state",
        "sub_state",
        "main_pid",
        "exec_main_code",
        "exec_main_status",
    }:
        raise ValueError("export native terminal fields differ")
    if (
        (native.get("active_state"), native.get("sub_state")) != ("active", "exited")
        or any(
            type(native.get(key)) is not int
            for key in ("main_pid", "exec_main_code", "exec_main_status")
        )
        or (native.get("main_pid"), native.get("exec_main_code"), native.get("exec_main_status"))
        != (0, 1, 0)
    ):
        raise ValueError("export native terminal outcome is not strict 0/1/0")
    binding = acceptance.get("binding")
    review = acceptance.get("source_review")
    if not isinstance(binding, dict) or (binding.get("bytes"), binding.get("sha256")) != (
        EXPORT_BINDING_BYTES,
        binding_sha256,
    ):
        raise ValueError("accepted export binding pin differs")
    if not isinstance(review, dict) or (review.get("bytes"), review.get("sha256")) != (
        SOURCE_REVIEW_BYTES,
        review_sha256,
    ):
        raise ValueError("accepted producer/consumer review pin differs")
    if (
        acceptance.get("evaluation_groups") != 468
        or type(acceptance.get("evaluation_groups")) is not int
    ):
        raise ValueError("accepted exposed membership count differs")
    membership = acceptance.get("membership_sha256")
    if not is_sha256(membership):
        raise ValueError("accepted exposed membership digest is malformed")
    if (
        acceptance.get("quality_result") is not None
        or acceptance.get("stop_status") != "PENDING_EXACT_STOP"
    ):
        raise ValueError("export acceptance includes a score or unexpected stop state")
    if acceptance.get("array_payloads_read_or_hashed_by_ROOT") is not False:
        raise ValueError("ROOT acceptance exceeds metadata-only export review")
    outputs = acceptance.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {arm for arm, _mode in ARMS}:
        raise ValueError("accepted output membership differs")
    for arm, _mode in ARMS:
        row = outputs[arm]
        if not isinstance(row, dict) or set(row) != {"prediction", "checkpoint"}:
            raise ValueError(f"accepted output row is malformed: {arm}")
        prediction, checkpoint = row["prediction"], row["checkpoint"]
        if not isinstance(prediction, dict) or not isinstance(checkpoint, dict):
            raise ValueError(f"accepted output pins are malformed: {arm}")
        if (
            prediction.get("shape") != PREDICTION_SHAPE
            or prediction.get("dtype") != "float32-little-endian"
            or prediction.get("data_bytes") != PREDICTION_DATA_BYTES
            or type(prediction.get("npy_header_bytes")) is not int
            or not 0 < prediction["npy_header_bytes"] <= 4_096
            or prediction.get("bytes") != PREDICTION_DATA_BYTES + prediction["npy_header_bytes"]
            or prediction.get("path") != "predictions.npy"
            or not is_sha256(prediction.get("sha256"))
        ):
            raise ValueError(f"accepted prediction metadata differs: {arm}")
        if (
            checkpoint.get("relative_path") is None
            or type(checkpoint.get("bytes")) is not int
            or checkpoint["bytes"] <= 0
            or not is_sha256(checkpoint.get("sha256"))
            or checkpoint.get("loaded_sha256") != checkpoint.get("sha256")
        ):
            raise ValueError(f"accepted checkpoint metadata differs: {arm}")
    exact_stop_schema = "fg19-first-cell-export-exact-owned-stop.v1"
    if exact_stop.get("schema") != exact_stop_schema:
        raise ValueError("export exact-stop schema differs")
    if set(exact_stop) != {
        "schema",
        "captured_utc",
        "unit",
        "invocation_id",
        "acceptance_sha256",
        "before",
        "after",
        "populated_before",
        "populated_after",
        "stop_returncode",
        "stop_stdout",
        "stop_stderr",
        "scope",
    }:
        raise ValueError("export exact-stop fields differ")
    if (
        exact_stop.get("unit"),
        exact_stop.get("invocation_id"),
        exact_stop.get("acceptance_sha256"),
    ) != (unit, invocation, acceptance_sha256):
        raise ValueError("export exact-stop identity or acceptance digest differs")
    before, after = exact_stop.get("before"), exact_stop.get("after")
    if (
        type(exact_stop.get("stop_returncode")),
        exact_stop.get("stop_returncode"),
        exact_stop.get("stop_stdout"),
        exact_stop.get("stop_stderr"),
    ) != (int, 0, "", ""):
        raise ValueError("exact stop command did not return strict zero with empty streams")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("exact-stop observations are malformed")
    if (
        before.get("Id"),
        before.get("ActiveState"),
        before.get("SubState"),
        before.get("MainPID"),
        before.get("InvocationID"),
        before.get("ExecMainCode"),
        before.get("ExecMainStatus"),
        before.get("Result"),
    ) != (unit, "active", "exited", "0", invocation, "1", "0", "success"):
        raise ValueError("export unit was not the accepted invocation before exact stop")
    if (
        after.get("ActiveState"),
        after.get("SubState"),
        after.get("MainPID"),
        after.get("LoadState"),
    ) != ("inactive", "dead", "0", "not-found"):
        raise ValueError("export unit is not absent and stopped after exact stop")
    if after.get("InvocationID") != "":
        raise ValueError("export invocation remains loaded after exact stop")
    if exact_stop.get("populated_before") != {} or exact_stop.get("populated_after") != {}:
        raise ValueError("exact-stop cgroup was populated")
    if exact_stop.get("scope") != (
        "Only the one successful exited export unit. All old native/support and foreign units "
        "excluded; no file or payload mutation."
    ):
        raise ValueError("exact-stop scope differs")
    return str(invocation)


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def is_invocation_id(value: object) -> bool:
    """Return whether a value is a lowercase 32-hex systemd invocation ID."""
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(char in "0123456789abcdef" for char in value)
    )


def payload_bytes() -> dict[str, bytes]:
    """Decode and verify exactly four scorer modules and two accepted receipts."""
    require(
        UNBOUND not in (EXPORT_ACCEPTANCE_SHA256, EXPORT_EXACT_STOP_SHA256),
        "actual export acceptance and exact-stop pins are not bound",
    )
    names = set(SCORER_FILES) | {"EXPORT_ACCEPTANCE.json", "EXPORT_EXACT_STOP.json"}
    require(set(PAYLOAD_B64) == names, "scorer and export-authority payload membership differs")
    result: dict[str, bytes] = {}
    for name, encoded in PAYLOAD_B64.items():
        raw = base64.b64decode(encoded, validate=True)
        if name in SCORER_FILES:
            expected_bytes, expected_sha = SCORER_FILES[name]
            require(
                (len(raw), digest(raw)) == (expected_bytes, expected_sha),
                f"scorer code pin differs: {name}",
            )
        elif name == "EXPORT_ACCEPTANCE.json":
            require(
                (len(raw), digest(raw)) == (EXPORT_ACCEPTANCE_BYTES, EXPORT_ACCEPTANCE_SHA256),
                "accepted export receipt pin differs",
            )
        elif name == "EXPORT_EXACT_STOP.json":
            require(
                (len(raw), digest(raw)) == (EXPORT_EXACT_STOP_BYTES, EXPORT_EXACT_STOP_SHA256),
                "export stop receipt pin differs",
            )
        else:
            raise RuntimeError(f"unapproved staged payload name: {name}")
        result[name] = raw
    require(
        sum(map(len, result.values())) <= 2_000_000, "bounded scorer staging payload exceeds cap"
    )
    return result


def path_stat(path: Path, *, expected_symlink: bool | None = False) -> dict[str, object]:
    """Observe one registered path without opening or hashing its payload."""
    link_stat = path.lstat()
    is_link = path.is_symlink()
    require(
        expected_symlink is None or is_link is expected_symlink,
        f"registered symlink status differs: {path}",
    )
    target_stat = path.stat()
    require(stat.S_ISREG(target_stat.st_mode), f"registered source is not a regular file: {path}")
    return {
        "path": str(path),
        "resolved": str(path.resolve(strict=True)),
        "islink": is_link,
        "device": target_stat.st_dev,
        "bytes": target_stat.st_size,
        "inode": target_stat.st_ino,
        "mtime_ns": target_stat.st_mtime_ns,
        "link_device": link_stat.st_dev,
        "link_inode": link_stat.st_ino,
    }


def check_registered_data(
    binding: dict[str, object], fit_admission: dict[str, object]
) -> dict[str, object]:
    """Recheck the exact six fit-registered paths and portable registry by metadata only."""
    registered = binding["registered_data"]
    sources = registered["source_identities"]
    expected_sources = fit_admission.get("data_identity_stats_only")
    require(
        isinstance(sources, dict) and isinstance(expected_sources, dict),
        "six-source admission map is missing",
    )
    require(
        sources == expected_sources and len(sources) == 6,
        "export binding differs from original six-source fit admission",
    )
    registry_raw, registry_info = stable_file(PORTABLE_REGISTRY, 2_000_000)
    require(
        (len(registry_raw), digest(registry_raw))
        == (PORTABLE_REGISTRY_BYTES, PORTABLE_REGISTRY_SHA256),
        "portable registry pin differs",
    )
    registry = json_object(registry_raw, "portable source registry")
    registry_sources = registry.get("sources")
    require(
        isinstance(registry_sources, dict) and set(registry_sources) == set(sources),
        "portable registry source membership differs",
    )
    require(
        type(registry.get("version")) is int
        and registry.get("version") == 1
        and registry.get("split_manifest_sha256") == SPLIT_SHA256,
        "portable registry version or split differs",
    )
    observed: dict[str, object] = {}
    for relative, expected in sources.items():
        require(isinstance(expected, dict), f"registered path identity malformed: {relative}")
        entry = registry_sources[relative]
        require(
            isinstance(entry, dict) and entry.get("sha256") == expected.get("registered_sha256"),
            f"portable source digest differs: {relative}",
        )
        identity = entry.get("original_identity")
        require(
            isinstance(identity, dict)
            and identity.get("bytes") == expected.get("bytes")
            and identity.get("mtime_ns") == expected.get("mtime_ns"),
            f"portable source stat differs: {relative}",
        )
        path = DATA_ROOT / relative
        current = path_stat(path, expected_symlink=expected["islink"])
        comparable = {
            key: current[key]
            for key in ("path", "resolved", "islink", "device", "bytes", "inode", "mtime_ns")
        }
        require(
            comparable == {key: expected[key] for key in comparable},
            f"registered source identity changed: {relative}",
        )
        observed[relative] = comparable
    return {
        "registry": {**registry_info, "sha256": PORTABLE_REGISTRY_SHA256},
        "sources_stat_only": observed,
        "payload_bytes_read_or_hashed": False,
    }


def inventory_exact_source_files(source_root: Path, expected_paths: set[str]) -> dict[str, int]:
    """Close the live source tree over the manifest file set before any source hash."""
    require(
        source_root.is_dir()
        and not source_root.is_symlink()
        and source_root.resolve(strict=True) == source_root,
        "canonical source root is redirected",
    )
    expected_directories: set[str] = set()
    for relative in expected_paths:
        path = Path(relative)
        require(
            not path.is_absolute() and ".." not in path.parts and path.parts,
            f"unsafe source manifest path: {relative}",
        )
        for parent in path.parents:
            if parent == Path("."):
                break
            expected_directories.add(parent.as_posix())

    actual_files: set[str] = set()
    entry_count = 0
    directory_count = 0

    def walk_error(error: OSError) -> None:
        require(False, f"canonical source inventory failed: {error}")

    for root, directories, files in os.walk(
        source_root, topdown=True, onerror=walk_error, followlinks=False
    ):
        directories.sort()
        files.sort()
        root_path = Path(root)
        for name in directories:
            path = root_path / name
            relative = path.relative_to(source_root).as_posix()
            entry_count += 1
            directory_count += 1
            require(entry_count <= 4096, "canonical source inventory exceeds entry cap")
            require(directory_count <= 512, "canonical source inventory exceeds directory cap")
            require(not path.is_symlink(), f"source inventory contains a symbolic link: {relative}")
            require(
                relative in expected_directories and path.is_dir(),
                f"source inventory contains an unlisted directory: {relative}",
            )
        for name in files:
            path = root_path / name
            relative = path.relative_to(source_root).as_posix()
            entry_count += 1
            require(entry_count <= 4096, "canonical source inventory exceeds entry cap")
            require(not path.is_symlink(), f"source inventory contains a symbolic link: {relative}")
            require(path.is_file(), f"source inventory contains a non-regular entry: {relative}")
            require(
                relative in expected_paths,
                f"source inventory contains an unlisted file: {relative}",
            )
            actual_files.add(relative)
    require(
        actual_files == expected_paths, "source inventory differs from the exact manifest file set"
    )
    return {
        "file_count": len(actual_files),
        "directory_count": directory_count,
        "entry_count": entry_count,
    }


def check_source_tree(binding: dict[str, object], source_manifest_raw: bytes) -> dict[str, object]:
    """Inventory the exact admitted source set, then hash its 167 files under byte limits."""
    manifest = json_object(source_manifest_raw, "source manifest")
    source = binding["source"]
    require(
        manifest.get("schema") == "fg19.teacher-cache-source-manifest.v1",
        "canonical source manifest schema differs",
    )
    require(
        manifest.get("source_commit") == SOURCE_COMMIT
        and manifest.get("source_root") == str(SOURCE_ROOT),
        "canonical source revision or root differs",
    )
    files = manifest.get("files")
    require(
        isinstance(files, list) and len(files) == 167,
        "canonical source manifest does not contain 167 files",
    )
    require(
        (source.get("commit"), source.get("file_count"), source.get("source_set_sha256"))
        == (SOURCE_COMMIT, 167, SOURCE_SET_SHA256),
        "binding source identity differs",
    )
    require(
        source.get("manifest_bytes") == SOURCE_MANIFEST_BYTES
        and source.get("manifest_sha256") == SOURCE_MANIFEST_SHA256,
        "source manifest binding pin differs",
    )
    require(
        SOURCE_ROOT.is_dir()
        and not SOURCE_ROOT.is_symlink()
        and SOURCE_ROOT.resolve(strict=True) == SOURCE_ROOT,
        "canonical source root is redirected",
    )
    ordered_rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for row in files:
        require(isinstance(row, dict), "source manifest row is malformed")
        relative, expected_bytes, expected_sha = (
            row.get("path"),
            row.get("bytes"),
            row.get("sha256"),
        )
        require(
            isinstance(relative, str) and relative not in seen,
            "source manifest path duplicate or malformed",
        )
        require(
            type(expected_bytes) is int
            and 0 <= expected_bytes <= MAX_SOURCE_BYTES
            and is_sha256(expected_sha),
            f"source manifest row malformed: {relative}",
        )
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            f"unsafe source manifest path: {relative}",
        )
        seen.add(relative)
        ordered_rows.append((relative, expected_bytes, expected_sha))
    expected_total_bytes = sum(size for _path, size, _sha in ordered_rows)
    require(
        expected_total_bytes <= MAX_SOURCE_AGGREGATE_BYTES,
        "canonical source manifest aggregate exceeds cap before hashing",
    )
    inventory = inventory_exact_source_files(SOURCE_ROOT, seen)
    aggregate = hashlib.sha256()
    total_bytes = 0
    results: dict[str, object] = {}
    for relative, expected_bytes, expected_sha in sorted(ordered_rows):
        path = SOURCE_ROOT / relative
        ancestor = path
        while ancestor != SOURCE_ROOT:
            require(not ancestor.is_symlink(), f"source path traverses a symbolic link: {relative}")
            ancestor = ancestor.parent
        require(
            path.resolve(strict=True).is_relative_to(SOURCE_ROOT),
            f"source path escaped pinned root: {relative}",
        )
        live_size = path.stat().st_size
        require(live_size == expected_bytes, f"source size differs from pinned manifest: {relative}")
        remaining_source_bytes = MAX_SOURCE_AGGREGATE_BYTES - total_bytes
        observed_sha, info = file_hash(
            path, min(expected_bytes, remaining_source_bytes)
        )
        require(
            (info["bytes"], observed_sha) == (expected_bytes, expected_sha),
            f"canonical source file differs: {relative}",
        )
        total_bytes += info["bytes"]
        require(
            total_bytes <= MAX_SOURCE_AGGREGATE_BYTES, "canonical source aggregate exceeded cap"
        )
        aggregate.update(relative.encode("utf-8") + b"\0" + observed_sha.encode("ascii") + b"\n")
        results[relative] = {**info, "sha256": observed_sha}
    require(aggregate.hexdigest() == SOURCE_SET_SHA256, "canonical source aggregate digest differs")
    return {
        "source_commit": SOURCE_COMMIT,
        "file_count": len(results),
        "aggregate_bytes": total_bytes,
        "source_set_sha256": aggregate.hexdigest(),
        "inventory": inventory,
        "files": results,
    }


def validate_acceptance_outputs(
    acceptance: dict[str, object],
    binding: dict[str, object],
    manifest: dict[str, object],
    manifest_pin: dict[str, object],
    sidecars: dict[str, tuple[dict[str, object], dict[str, object]]],
) -> dict[str, object]:
    """Join root-accepted fixed outputs to the existing manifest and sidecars without opening arrays."""
    accepted_manifest = acceptance.get("manifest_capture")
    require(
        isinstance(accepted_manifest, dict)
        and (accepted_manifest.get("bytes"), accepted_manifest.get("sha256"))
        == (manifest_pin["bytes"], manifest_pin["sha256"]),
        "accepted export manifest pin differs from current bytes",
    )
    require(
        manifest.get("schema") == "fg19.first-cell-evaluation-export-manifest.v1"
        and manifest.get("status") == "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED"
        and manifest.get("quality_or_metric_claim") is False,
        "export result is not a complete prediction-only bundle",
    )
    require(
        manifest.get("binding_sha256") == EXPORT_BINDING_SHA256
        and manifest.get("source") == binding.get("source"),
        "export manifest binding/source differs",
    )
    groups = manifest.get("ordered_groups")
    require(
        isinstance(groups, list)
        and len(groups) == 468
        and len(set(groups)) == 468
        and all(type(group) is int for group in groups),
        "export membership is not 468 strict unique groups",
    )
    membership_sha = digest("".join(f"{group}\n" for group in groups).encode("ascii"))
    require(
        membership_sha == acceptance.get("membership_sha256") == manifest.get("membership_sha256"),
        "accepted and exported membership digest differs",
    )
    require(
        manifest.get("split")
        == {**binding["split"], "role": "fit-excluded-exposed-development-evaluation"},
        "export split is not the fixed source-qualified F0 split",
    )
    arms = manifest.get("arms")
    require(
        isinstance(arms, list)
        and [row.get("arm") for row in arms if isinstance(row, dict)]
        == [name for name, _mode in ARMS],
        "export manifest arm order differs",
    )
    output_rows = acceptance.get("outputs")
    binding_arms = binding.get("arms")
    require(
        isinstance(output_rows, dict) and isinstance(binding_arms, list) and len(binding_arms) == 5,
        "accepted or bound arm rows are malformed",
    )
    joined: dict[str, object] = {}
    for manifest_row, bound_arm, (arm_name, mode) in zip(arms, binding_arms, ARMS, strict=True):
        require(
            manifest_row.get("arm") == arm_name
            and manifest_row.get("representation_mode") == mode
            and bound_arm.get("arm") == arm_name
            and bound_arm.get("representation_mode") == mode,
            f"fixed arm/mode differs: {arm_name}",
        )
        sidecar, sidecar_pin = sidecars[arm_name]
        accepted_row = output_rows[arm_name]
        prediction = sidecar.get("prediction")
        require(
            isinstance(prediction, dict) and isinstance(accepted_row, dict),
            f"prediction/acceptance row malformed: {arm_name}",
        )
        accepted_prediction = accepted_row.get("prediction")
        accepted_checkpoint = accepted_row.get("checkpoint")
        require(
            isinstance(accepted_prediction, dict) and isinstance(accepted_checkpoint, dict),
            f"accepted output row malformed: {arm_name}",
        )
        checkpoint = {**bound_arm["model"], "loaded_sha256": bound_arm["model"]["sha256"]}
        require(
            sidecar.get("schema") == "fg19.first-cell-feature-only-prediction.v1"
            and sidecar.get("status") == "PREDICTION_EXPORT_ONLY_NOT_SCORED",
            f"prediction sidecar status differs: {arm_name}",
        )
        require(
            sidecar.get("checkpoint") == checkpoint
            and accepted_checkpoint == checkpoint
            and manifest_row.get("checkpoint") == checkpoint,
            f"fixed model checkpoint pin differs: {arm_name}",
        )
        require(
            sidecar.get("selected_epoch") == manifest_row.get("selected_epoch") == 5
            and sidecar.get("successful_updates")
            == manifest_row.get("successful_updates")
            == 44_760,
            f"fixed model endpoint differs: {arm_name}",
        )
        require(
            sidecar.get("terminal_acceptance") == bound_arm.get("terminal_acceptance")
            and sidecar.get("exact_stop") == bound_arm.get("exact_stop"),
            f"fit acceptance/stop lineage differs: {arm_name}",
        )
        require(
            sidecar.get("binding_sha256") == EXPORT_BINDING_SHA256
            and sidecar.get("representation_mode") == mode
            and sidecar.get("ordered_groups") == groups
            and sidecar.get("membership_sha256") == membership_sha,
            f"sidecar binding, mode, or membership differs: {arm_name}",
        )
        require(
            sidecar.get("feature_only") == {"targets_read": False, "scored_mask_read": False}
            and sidecar.get("scoring_performed") is False,
            f"export is not feature-only: {arm_name}",
        )
        require(
            prediction.get("shape") == PREDICTION_SHAPE
            and prediction.get("dtype") == "float32-little-endian"
            and prediction.get("data_bytes") == PREDICTION_DATA_BYTES
            and type(prediction.get("npy_header_bytes")) is int
            and 0 < prediction["npy_header_bytes"] <= 4_096
            and prediction.get("bytes") == PREDICTION_DATA_BYTES + prediction["npy_header_bytes"]
            and prediction.get("path") == "predictions.npy"
            and is_sha256(prediction.get("sha256")),
            f"prediction array contract differs: {arm_name}",
        )
        require(
            accepted_prediction == prediction
            and manifest_row.get("prediction_bytes") == prediction["bytes"]
            and manifest_row.get("prediction_sha256") == prediction["sha256"]
            and manifest_row.get("sidecar_bytes") == sidecar_pin["bytes"]
            and manifest_row.get("sidecar_sha256") == sidecar_pin["sha256"],
            f"accepted output metadata differs: {arm_name}",
        )
        arm_dir = EXPORT_ROOT / arm_name
        require(
            arm_dir.is_dir() and not arm_dir.is_symlink(),
            f"prediction arm directory is unsafe: {arm_name}",
        )
        array_path = arm_dir / "predictions.npy"
        array_info = path_stat(array_path, expected_symlink=False)
        require(
            array_info["bytes"] == prediction["bytes"],
            f"prediction array size differs before admission: {arm_name}",
        )
        joined[arm_name] = {
            "sidecar": sidecar_pin,
            "prediction_stat_only": array_info,
            "prediction_sha256_from_root_acceptance_not_rehashed": prediction["sha256"],
            "checkpoint_sha256_from_root_acceptance_not_rehashed": checkpoint["sha256"],
        }
    return joined


def check_export_bundle(payload: dict[str, bytes]) -> dict[str, object]:
    """Revalidate export metadata and stat arrays; leave array hashing/mapping to the worker."""
    binding_path = EXPORT_D / "EXPORT_BINDING_V4.json"
    manifest_path = EXPORT_ROOT / "EXPORT_MANIFEST_V1.json"
    binding_raw, binding_info = stable_file(binding_path, 2_000_000)
    require(
        (len(binding_raw), digest(binding_raw)) == (EXPORT_BINDING_BYTES, EXPORT_BINDING_SHA256),
        "read-only export binding changed",
    )
    binding = json_object(binding_raw, "export binding")
    require(
        binding.get("schema") == "fg19.first-cell-evaluation-export-binding.v1"
        and (
            binding.get("project_base"),
            binding.get("data_root"),
            binding.get("source_root"),
            binding.get("evidence_root"),
            binding.get("output_dir"),
        )
        == (str(PROJECT_BASE), str(DATA_ROOT), str(SOURCE_ROOT), str(EXPORT_D), str(EXPORT_ROOT)),
        "export binding paths differ",
    )
    manifest_raw, manifest_info = stable_file(manifest_path, 4_000_000)
    manifest_pin = {**manifest_info, "bytes": len(manifest_raw), "sha256": digest(manifest_raw)}
    manifest = json_object(manifest_raw, "export completion manifest")
    acceptance = json_object(payload["EXPORT_ACCEPTANCE.json"], "root export acceptance")
    exact_stop = json_object(payload["EXPORT_EXACT_STOP.json"], "root export exact stop")
    invocation = validate_export_authority(
        acceptance,
        exact_stop,
        acceptance_sha256=EXPORT_ACCEPTANCE_SHA256,
        binding_sha256=EXPORT_BINDING_SHA256,
        review_sha256=SOURCE_REVIEW_SHA256,
        expected_unit=EXPORT_UNIT,
    )
    expected_sidecars = {arm: f"{arm}/PREDICTION_SIDECAR.json" for arm, _mode in ARMS}
    sidecars: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for arm, relative in expected_sidecars.items():
        raw, info = stable_file(EXPORT_ROOT / relative, 2_000_000)
        row = next(
            item
            for item in manifest.get("arms", [])
            if isinstance(item, dict) and item.get("arm") == arm
        )
        require(
            (len(raw), digest(raw)) == (row.get("sidecar_bytes"), row.get("sidecar_sha256")),
            f"export sidecar pin differs: {arm}",
        )
        sidecars[arm] = (
            json_object(raw, f"{arm} sidecar"),
            {"path": relative, "bytes": len(raw), "sha256": digest(raw), **info},
        )
    joined = validate_acceptance_outputs(acceptance, binding, manifest, manifest_pin, sidecars)
    return {
        "export_binding": {**binding_info, "sha256": EXPORT_BINDING_SHA256, "read_only": True},
        "source_manifest": {
            "path": str(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json"),
            "bytes": SOURCE_MANIFEST_BYTES,
            "sha256": SOURCE_MANIFEST_SHA256,
            "read_only": True,
        },
        "export_manifest": {**manifest_info, "sha256": digest(manifest_raw), "read_only": True},
        "export_invocation_id": invocation,
        "five_accepted_outputs_stat_only": joined,
        "array_payloads_read_or_hashed_before_worker": False,
    }


def check_predecessors(fit_admission: dict[str, object]) -> dict[str, object]:
    """Recheck the fourteen readonly receipts, profile metadata, and two ancestor stats."""
    staged = fit_admission.get("staged_receipt_sha256")
    require(
        isinstance(staged, dict) and staged == READONLY_RECEIPTS,
        "AllThree predecessor receipt map differs",
    )
    receipts: dict[str, object] = {}
    for name, expected_sha in READONLY_RECEIPTS.items():
        raw, info = stable_file(FIT_RECEIPTS_ROOT / name, 1_000_000)
        require(digest(raw) == expected_sha, f"accepted predecessor receipt differs: {name}")
        receipts[name] = {**info, "sha256": expected_sha, "read_only": True}
    profile: dict[str, object] = {}
    for name, expected_sha in PROFILE_PINS.items():
        raw, info = stable_file(PROFILE_ROOT / name, 1_000_000)
        require(digest(raw) == expected_sha, f"profile metadata differs: {name}")
        profile[name] = {**info, "sha256": expected_sha, "read_only": True}
    ancestry = (
        (
            PROJECT_BASE / "data/artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt",
            fit_admission.get("parent_checkpoint_identity"),
            PARENT_CHECKPOINT_SHA256,
            3_988_711,
            779_788_689,
        ),
        (
            PROJECT_BASE / "data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt",
            fit_admission.get("dictionary"),
            DICTIONARY_SHA256,
            22_557,
            733_618_561,
        ),
    )
    observed: dict[str, object] = {}
    for path, recorded, expected_sha, expected_bytes, expected_inode in ancestry:
        require(
            isinstance(recorded, dict)
            and recorded.get("path") == str(path)
            and recorded.get("sha256") == expected_sha,
            "parent or dictionary accepted identity differs",
        )
        require(not path.is_symlink(), f"ancestor checkpoint path is a symlink: {path}")
        current = path.stat()
        require(
            stat.S_ISREG(current.st_mode)
            and (current.st_size, current.st_dev, current.st_ino)
            == (expected_bytes, EXPECTED_FILESYSTEM_DEVICE, expected_inode),
            f"ancestor checkpoint stat differs: {path}",
        )
        require(
            (recorded.get("bytes"), recorded.get("device"), recorded.get("inode"))
            == (current.st_size, current.st_dev, current.st_ino),
            f"ancestor checkpoint admission stat differs: {path}",
        )
        observed[path.name] = {
            "path": str(path),
            "bytes": current.st_size,
            "device": current.st_dev,
            "inode": current.st_ino,
            "accepted_sha256_not_rehashed": expected_sha,
        }
    return {
        "readonly_receipts": receipts,
        "profile_metadata": profile,
        "ancestor_artifact_stats_only": observed,
        "model_payloads_read_or_hashed": False,
    }


def check_cache(fit_admission: dict[str, object]) -> dict[str, object]:
    """Recheck cache receipts, metadata, and training-array stat identities only."""
    cache = fit_admission.get("cache")
    require(isinstance(cache, dict), "accepted cache metadata is absent")
    metadata_raw, metadata_info = stable_file(CACHE_ROOT / "metadata.json", MAX_METADATA_BYTES)
    acceptance_raw, acceptance_info = stable_file(
        CACHE_RECEIPTS_ROOT / "TERMINAL_ACCEPTANCE.json", 10_000
    )
    stop_raw, stop_info = stable_file(CACHE_RECEIPTS_ROOT / "EXACT_STOP.json", 10_000)
    require(
        digest(metadata_raw) == CACHE_METADATA_SHA256
        and digest(acceptance_raw) == CACHE_ACCEPTANCE_SHA256
        and digest(stop_raw) == CACHE_STOP_SHA256,
        "accepted cache metadata or receipts changed",
    )
    metadata = json_object(metadata_raw, "homogeneous cache metadata")
    acceptance = json_object(acceptance_raw, "homogeneous cache acceptance")
    require(
        metadata.get("split_manifest_sha256") == SPLIT_SHA256
        and acceptance.get("metadata_sha256") == CACHE_METADATA_SHA256,
        "homogeneous cache metadata-to-split identity differs",
    )
    stats, outputs = cache.get("array_stats_only"), acceptance.get("output")
    require(isinstance(stats, dict) and isinstance(outputs, dict), "cache array metadata missing")
    arrays: dict[str, object] = {}
    for name, size, count, expected_sha, expected_inode in (
        (
            "train",
            1_685_600_128,
            10_535,
            "4abbb9be31ec1135a5e0ff5bb87c93a2db6e44943e95d23ff70835b8044c5194",
            739_677_840,
        ),
        (
            "valid",
            224_800_128,
            1_405,
            "6963d3c565caf469b023a61ba6d2dd376d929b2bcf724af88492dc8adb76f591",
            739_677_841,
        ),
    ):
        path = CACHE_ROOT / f"main_homogeneous_{name}.npy"
        observed = path_stat(path, expected_symlink=False)
        recorded = stats.get(name)
        output = outputs.get(name)
        require(
            isinstance(recorded, dict) and isinstance(output, dict),
            f"cache {name} accepted stats are missing",
        )
        compared = {key: observed[key] for key in ("bytes", "device", "inode", "mtime_ns")}
        require(
            compared == recorded
            and (observed["bytes"], observed["device"], observed["inode"])
            == (size, EXPECTED_FILESYSTEM_DEVICE, expected_inode),
            f"cache {name} stat differs",
        )
        require(
            (output.get("bytes"), output.get("sha256"), output.get("shape"), output.get("dtype"))
            == (size, expected_sha, [count, 20_000, 2], "float32"),
            f"cache {name} acceptance differs",
        )
        arrays[name] = {**compared, "sha256_from_acceptance_not_rehashed": expected_sha}
    return {
        "metadata": {**metadata_info, "sha256": CACHE_METADATA_SHA256},
        "acceptance": {**acceptance_info, "sha256": CACHE_ACCEPTANCE_SHA256},
        "exact_stop": {**stop_info, "sha256": CACHE_STOP_SHA256},
        "array_stat_only": arrays,
        "cache_payloads_read_or_hashed": False,
    }


def check_parent_resources() -> dict[str, object]:
    """Require the two-core parent envelope, empty scoped cgroup, and physical scratch."""
    parent = show(
        "wnn-campaign.slice",
        ("ControlGroup", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax"),
    )
    require(
        (
            parent.get("CPUQuotaPerSecUSec"),
            parent.get("MemoryHigh"),
            parent.get("MemoryMax"),
            parent.get("MemorySwapMax"),
        )
        == ("2s", "18000000000", "20000000000", "0"),
        "aggregate parent resource envelope differs",
    )
    group = parent.get("ControlGroup", "")
    require(
        group not in ("", "/") and group.endswith("/wnn-campaign.slice"),
        "parent cgroup is not narrowly scoped",
    )
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    require(
        root.is_dir() and root.resolve(strict=True) == root,
        "parent cgroup is unavailable or redirected",
    )
    inventory: list[Path] = []
    stack = [root]
    total_dirs = 0
    aggregate = 0
    populated: dict[str, str] = {}
    while stack:
        current = stack.pop()
        total_dirs += 1
        require(total_dirs <= 128, "parent cgroup directory cap exceeded")
        count = 0
        with os.scandir(current) as entries:
            for entry in entries:
                count += 1
                require(count <= 256, "parent cgroup entry cap exceeded")
                require(not entry.is_symlink(), "parent cgroup traversal encountered symlink")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    require(total_dirs + len(stack) <= 128, "parent cgroup directory cap exceeded")
        inventory.append(current / "cgroup.procs")
    require(0 < len(inventory) <= 128, "parent cgroup process-file inventory differs")
    for path in inventory:
        raw = _bounded_proc_read(path, 32_768)
        aggregate += len(raw)
        require(aggregate <= 262_144, "parent cgroup process metadata aggregate exceeds cap")
        if raw.strip():
            populated[str(path)] = raw.decode("ascii", errors="strict")
    require(not populated, "campaign parent cgroup is populated")
    current_raw = _bounded_proc_read(root / "memory.current", 4_096).decode("ascii").strip()
    require(current_raw.isdigit(), "parent memory.current is unavailable")
    current = int(current_raw)
    require(
        current + WORKER_MEMORY_MAX_BYTES + PARENT_MEMORY_RESERVE_BYTES
        < PARENT_MEMORY_TARGET_BYTES,
        "aggregate memory headroom is insufficient",
    )
    events = {
        name: _bounded_proc_read(root / name, 4_096).decode("ascii")
        for name in ("memory.events", "memory.events.local")
    }
    local = dict(
        line.split()
        for line in events["memory.events.local"].splitlines()
        if len(line.split()) == 2
    )
    require(
        all(local.get(key) == "0" for key in ("oom", "oom_kill", "oom_group_kill")),
        "parent local OOM counters are nonzero",
    )
    require(
        SCRATCH_ROOT.is_dir()
        and not SCRATCH_ROOT.is_symlink()
        and SCRATCH_ROOT.resolve(strict=True) == SCRATCH_ROOT,
        "physical scratch root is redirected",
    )
    device = PROJECT_BASE.stat().st_dev
    require(
        device == SCRATCH_ROOT.stat().st_dev == EXPECTED_FILESYSTEM_DEVICE,
        "project scratch filesystem device differs",
    )
    filesystem = subprocess.run(
        ["findmnt", "-T", str(SCRATCH_ROOT), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        filesystem.returncode == 0 and filesystem.stdout.strip() == "ext4",
        "scratch filesystem is not ext4",
    )
    free = os.statvfs(SCRATCH_ROOT).f_bavail * os.statvfs(SCRATCH_ROOT).f_frsize
    require(
        free >= MINIMUM_FILESYSTEM_FREE_BYTES,
        "physical scratch free space is below the fixed reserve",
    )
    return {
        "parent_properties": parent,
        "cgroup": str(root),
        "populated": populated,
        "memory_current_bytes": current,
        "memory_events": events,
        "filesystem_device": device,
        "scratch_filesystem": filesystem.stdout.strip(),
        "scratch_free_bytes": free,
        "minimum_scratch_free_bytes": MINIMUM_FILESYSTEM_FREE_BYTES,
    }


def read_gpu_owner_details(
    proc: Path,
    owner_uid: int,
    current_uid: int,
    read_proc: Callable[[Path, int], bytes],
) -> tuple[str, str | None]:
    """Read each GPU owner's cgroup and same-UID command metadata only."""
    cgroup = read_proc(proc / "cgroup", 32_768).decode("ascii", errors="strict").strip()
    if owner_uid != current_uid:
        return cgroup, None
    command = read_proc(proc / "cmdline", 8_192).decode("utf-8", errors="replace")
    return cgroup, command


def check_selected_gpu() -> dict[str, object]:
    """Observe all selected-GPU owners and preserve every process without action."""
    prefix = ["nvidia-smi", "-i", GPU_UUID]
    identity, _ = _bounded_gpu_query(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    )
    lines = [line.strip() for line in identity.splitlines() if line.strip()]
    require(len(lines) == 1, "selected GPU inventory did not return one device")
    fields = [part.strip() for part in lines[0].split(",")]
    require(
        len(fields) == 4 and fields[0] == GPU_UUID and fields[1], "selected GPU identity differs"
    )
    free, total = _memory_bytes(fields[2]), _memory_bytes(fields[3])
    require(total > 0 and free <= total, "selected GPU memory metadata is inconsistent")
    listing, _ = _bounded_gpu_query(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"]
    )
    process_rows = [line.strip() for line in listing.splitlines() if line.strip()]
    if len(process_rows) == 1 and process_rows[0].startswith("No running processes found"):
        process_rows = []
    require(len(process_rows) <= 256, "selected GPU process inventory exceeds cap")
    owners: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in process_rows:
        parts = [part.strip() for part in row.split(",")]
        require(
            len(parts) == 3
            and parts[0].isascii()
            and parts[0].isdigit()
            and int(parts[0]) > 0
            and str(int(parts[0])) == parts[0]
            and parts[1] == GPU_UUID
            and parts[0] not in seen,
            "selected GPU process row is malformed or duplicated",
        )
        seen.add(parts[0])
        proc = Path("/proc") / parts[0]
        before, start_before = proc.stat(follow_symlinks=False), _proc_starttime(proc, parts[0])
        cgroup, command = read_gpu_owner_details(
            proc, before.st_uid, os.getuid(), _bounded_proc_read
        )
        after, start_after = proc.stat(follow_symlinks=False), _proc_starttime(proc, parts[0])
        require(
            (before.st_ino, before.st_uid, start_before)
            == (after.st_ino, after.st_uid, start_after),
            "GPU owner PID changed during observation",
        )
        campaign_owned = "wnn-campaign.slice" in cgroup or (
            command is not None and str(PROJECT_BASE) in command
        )
        require(not campaign_owned, "campaign project process currently owns the selected GPU")
        owners.append(
            {
                "pid": int(parts[0]),
                "gpu_uuid": parts[1],
                "used_memory": parts[2],
                "uid": after.st_uid,
                "cgroup": cgroup,
                "cgroup_inspected": True,
                "cmdline_inspected": command is not None,
                "proc_starttime": start_before,
                "preserved": True,
                "action": "none",
            }
        )
    return {
        "uuid": GPU_UUID,
        "name": fields[1],
        "free_bytes": free,
        "total_bytes": total,
        "process_count": len(owners),
        "owners": owners,
        "foreign_process_action": "none",
    }


def check_original_fit_receipts() -> tuple[
    bytes, dict[str, int], bytes, dict[str, int], dict[str, object], dict[str, object]
]:
    """Join pinned AllThree fit receipts through the nested systemctl capture."""
    fit_admission_raw, fit_admission_info = stable_file(
        FIT_RECEIPTS_ROOT / "ADMISSION.json", 20_000
    )
    fit_launch_raw, fit_launch_info = stable_file(FIT_RECEIPTS_ROOT / "LAUNCH.json", 20_000)
    require(
        (len(fit_admission_raw), digest(fit_admission_raw))
        == (FIT_ADMISSION_BYTES, FIT_ADMISSION_SHA256),
        "original AllThree fit admission differs",
    )
    require(
        (len(fit_launch_raw), digest(fit_launch_raw)) == (FIT_LAUNCH_BYTES, FIT_LAUNCH_SHA256),
        "original AllThree fit launch differs",
    )
    fit_admission = json_object(fit_admission_raw, "original AllThree admission")
    fit_launch = json_object(fit_launch_raw, "original AllThree launch")
    native_unit = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
    native_invocation = "94d3a7132a434d448345b0206bd3daa3"
    require(
        fit_admission.get("schema") == "fg19-child-allthree-fit-admission.v1"
        and fit_admission.get("run_name") == "fg19_child_allthree_f0_s1001_fit_v1"
        and fit_admission.get("unit") == native_unit,
        "original AllThree fit admission identity differs",
    )
    submission = fit_launch.get("submission")
    after_submission = fit_launch.get("systemctl_after_submission")
    require(
        fit_launch.get("schema") == "fg19-child-allthree-fit-launch.v1"
        and fit_launch.get("unit") == native_unit
        and isinstance(submission, dict)
        and set(submission) == {"outcome", "returncode", "stderr_b64", "stdout_b64"}
        and submission.get("outcome") == "submitted"
        and type(submission.get("returncode")) is int
        and submission.get("returncode") == 0,
        "original AllThree submission receipt differs",
    )
    require(
        isinstance(after_submission, dict)
        and set(after_submission) == {"outcome", "returncode", "stderr_b64", "stdout_b64"}
        and after_submission.get("outcome") == "captured"
        and type(after_submission.get("returncode")) is int
        and after_submission.get("returncode") == 0,
        "original AllThree systemctl-after-submission receipt differs",
    )
    submission_stdout = base64.b64decode(submission["stdout_b64"], validate=True)
    submission_stderr = base64.b64decode(submission["stderr_b64"], validate=True)
    require(
        submission_stdout == b""
        and submission_stderr.decode("utf-8", errors="strict")
        == f"Running as unit: {native_unit}; invocation ID: {native_invocation}\n",
        "original AllThree submit output does not join the native unit and invocation",
    )
    captured_stdout = base64.b64decode(after_submission["stdout_b64"], validate=True)
    captured_stderr = base64.b64decode(after_submission["stderr_b64"], validate=True)
    require(
        captured_stderr == b"" and len(captured_stdout) <= 8_192,
        "original AllThree systemctl capture output differs",
    )
    capture_properties: dict[str, str] = {}
    for line in captured_stdout.decode("utf-8", errors="strict").splitlines():
        key, separator, value = line.partition("=")
        require(
            bool(separator) and bool(key) and key not in capture_properties,
            "original AllThree systemctl capture has duplicate or malformed properties",
        )
        capture_properties[key] = value
    require(
        set(capture_properties)
        == {"MainPID", "ExecStart", "ControlGroup", "ActiveState", "SubState", "InvocationID"}
        and is_invocation_id(capture_properties.get("InvocationID"))
        and capture_properties.get("InvocationID") == native_invocation
        and capture_properties.get("MainPID", "").isdecimal()
        and int(capture_properties["MainPID"]) > 0
        and capture_properties.get("ActiveState") == "active"
        and capture_properties.get("SubState") == "running"
        and capture_properties.get("ControlGroup", "").endswith(f"/{native_unit}")
        and f"--name {fit_admission.get('run_name')}" in capture_properties.get("ExecStart", "")
        and "--representation-mode ema_sparse" in capture_properties.get("ExecStart", ""),
        "original AllThree systemctl properties do not join the native run",
    )
    return (
        fit_admission_raw,
        fit_admission_info,
        fit_launch_raw,
        fit_launch_info,
        fit_admission,
        fit_launch,
    )


def check_paths_and_inputs(payload: dict[str, bytes]) -> dict[str, object]:
    """Perform every read-only path, receipt, source, data, and result-absence gate."""
    require(
        PROJECT_BASE.is_dir()
        and not PROJECT_BASE.is_symlink()
        and PROJECT_BASE.resolve(strict=True) == PROJECT_BASE,
        "project base is redirected",
    )
    require(
        DATA_ROOT.is_dir()
        and not DATA_ROOT.is_symlink()
        and DATA_ROOT.resolve(strict=True) == DATA_ROOT,
        "data root is redirected",
    )
    require(
        CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
        "canonical project Python is unavailable",
    )
    require(
        EXPORT_D.is_dir()
        and not EXPORT_D.is_symlink()
        and EXPORT_D.resolve(strict=True) == EXPORT_D,
        "accepted export evidence root is missing or redirected",
    )
    require(
        EXPORT_ROOT.is_dir()
        and not EXPORT_ROOT.is_symlink()
        and EXPORT_ROOT.resolve(strict=True) == EXPORT_ROOT,
        "accepted prediction export root is missing or redirected",
    )
    require(
        not os.path.lexists(SCORE_D), "private score directory already exists; never reuse or retry"
    )
    require(not os.path.lexists(RESULT_PATH), "create-only score result path already exists")
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found",
        "score support unit already exists; never reuse",
    )
    binding_raw, _ = stable_file(EXPORT_D / "EXPORT_BINDING_V4.json", 2_000_000)
    binding = json_object(binding_raw, "export binding")
    require(
        (len(binding_raw), digest(binding_raw)) == (EXPORT_BINDING_BYTES, EXPORT_BINDING_SHA256),
        "fixed export binding pin differs",
    )
    manifest_raw, source_info = stable_file(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json", 2_000_000)
    require(
        (len(manifest_raw), digest(manifest_raw))
        == (SOURCE_MANIFEST_BYTES, SOURCE_MANIFEST_SHA256),
        "source manifest pin differs",
    )
    (
        fit_admission_raw,
        fit_admission_info,
        fit_launch_raw,
        fit_launch_info,
        fit_admission,
        fit_launch,
    ) = check_original_fit_receipts()
    require(
        not SPLIT_PATH.is_symlink() and SPLIT_PATH.resolve(strict=True) == SPLIT_PATH,
        "canonical split path is redirected",
    )
    split_raw, split_info = stable_file(SPLIT_PATH, 2_000_000)
    require(
        digest(split_raw) == SPLIT_SHA256 and json_object(split_raw, "canonical split"),
        "canonical F0 split bytes differ",
    )
    require(
        binding.get("split", {}).get("sha256") == SPLIT_SHA256
        and binding.get("source", {}).get("commit") == SOURCE_COMMIT,
        "export binding source or split differs",
    )
    registered = check_registered_data(binding, fit_admission)
    source_tree = check_source_tree(binding, manifest_raw)
    predecessors = check_predecessors(fit_admission)
    cache = check_cache(fit_admission)
    export_bundle = check_export_bundle(payload)
    parent = check_parent_resources()
    gpu = check_selected_gpu()
    model_stats: dict[str, object] = {}
    for arm in binding.get("arms", []):
        model = arm.get("model")
        require(isinstance(model, dict), "accepted model checkpoint metadata is malformed")
        relative = Path(str(model.get("relative_path", "")))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe accepted checkpoint path",
        )
        path = DATA_ROOT / relative
        ancestor = path
        while ancestor != DATA_ROOT:
            require(
                not ancestor.is_symlink(), f"checkpoint path traverses a symbolic link: {ancestor}"
            )
            ancestor = ancestor.parent
        observed = path_stat(path, expected_symlink=False)
        require(
            observed["bytes"] == model.get("bytes"),
            f"accepted checkpoint size differs: {arm.get('arm')}",
        )
        model_stats[str(arm.get("arm"))] = {
            key: observed[key] for key in ("path", "bytes", "device", "inode", "mtime_ns")
        }
    sources_unchanged = stable_file(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json", 2_000_000)
    require(sources_unchanged[0] == manifest_raw, "source manifest changed during preflight")
    return {
        "binding": {"bytes": len(binding_raw), "sha256": digest(binding_raw), "read_only": True},
        "source_manifest": {
            **source_info,
            "sha256": digest(manifest_raw),
            "file_count": 167,
            "read_only": True,
        },
        "source_tree": source_tree,
        "split": {**split_info, "sha256": SPLIT_SHA256},
        "registered_six_sources_stat_only": registered,
        "fit_admission": {
            **fit_admission_info,
            "sha256": digest(fit_admission_raw),
            "read_only": True,
        },
        "fit_launch": {**fit_launch_info, "sha256": digest(fit_launch_raw), "read_only": True},
        "predecessors_profile_parent_dictionary": predecessors,
        "cache_stat_only": cache,
        "accepted_export": export_bundle,
        "fit_checkpoints_stat_only": model_stats,
        "parent_disk_cgroup_headroom": parent,
        "selected_gpu_owners": gpu,
        "result_absent_before_mutation": True,
        "score_unit_absent_before_mutation": True,
    }


def stage_packet(payload: dict[str, bytes]) -> dict[str, object]:
    """Create the private scorer directory and stage only four modules plus root receipts."""
    SCORE_D.mkdir(mode=0o700, exist_ok=False)
    accepted_dir = SCORE_D / "accepted_export"
    accepted_dir.mkdir(mode=0o700, exist_ok=False)
    temp_dir = SCORE_D / "tmp"
    temp_dir.mkdir(mode=0o700, exist_ok=False)
    for name in SCORER_FILES:
        write_exclusive(SCORE_D / name, payload[name])
    write_exclusive(accepted_dir / "EXPORT_ACCEPTANCE.json", payload["EXPORT_ACCEPTANCE.json"])
    write_exclusive(accepted_dir / "EXPORT_EXACT_STOP.json", payload["EXPORT_EXACT_STOP.json"])
    for directory in (accepted_dir, temp_dir, SCORE_D):
        fd = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    staged: dict[str, object] = {}
    for relative, raw in (
        *((name, payload[name]) for name in SCORER_FILES),
        ("accepted_export/EXPORT_ACCEPTANCE.json", payload["EXPORT_ACCEPTANCE.json"]),
        ("accepted_export/EXPORT_EXACT_STOP.json", payload["EXPORT_EXACT_STOP.json"]),
    ):
        disk, info = stable_file(SCORE_D / relative, len(raw))
        require(disk == raw, f"staged source/receipt changed: {relative}")
        staged[relative] = {**info, "sha256": digest(raw)}
    return staged


def bounded_submit(command: list[str]) -> tuple[int, bytes, bytes]:
    """Capture one systemd-run response within a fixed timeout and stream cap."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        start_new_session=True,
    )
    require(
        process.stdout is not None and process.stderr is not None, "systemd-run pipes unavailable"
    )
    output = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + 30
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                require(
                    remaining > 0,
                    "submission response timed out; outcome is unknown and must not be retried",
                )
                events = selector.select(remaining)
                require(
                    events,
                    "submission response timed out; outcome is unknown and must not be retried",
                )
                for key, _ in events:
                    label = streams[key.fileobj]
                    chunk = os.read(key.fileobj.fileno(), min(4096, 65_537 - len(output[label])))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output[label].extend(chunk)
                    require(
                        len(output[label]) <= 65_536,
                        "submission output exceeded stream cap; outcome unknown",
                    )
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, RuntimeError, subprocess.SubprocessError):
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    return returncode, bytes(output["stdout"]), bytes(output["stderr"])


def submit_once(admission: dict[str, object]) -> dict[str, object]:
    """Submit exactly one CPU score unit and durably capture its invocation ID."""
    command = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=" + WORKER_CPU_QUOTA,
        "--property=MemoryHigh=" + str(WORKER_MEMORY_HIGH_BYTES),
        "--property=MemoryMax=" + str(WORKER_MEMORY_MAX_BYTES),
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=" + str(WORKER_RUNTIME_SECONDS),
        "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(SCORE_D / "unit.log"),
        "--property=StandardError=append:" + str(SCORE_D / "unit.log"),
        "--working-directory=" + str(SCORE_D),
        "--setenv=CUDA_VISIBLE_DEVICES=",
        "--setenv=WNN_PROJECT_ROOT=" + str(DATA_ROOT),
        "--setenv=WNN_PORTABLE_DATA_MANIFEST=" + str(PORTABLE_REGISTRY),
        "--setenv=WNN_THREADS=1",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--setenv=VECLIB_MAXIMUM_THREADS=1",
        "--setenv=TMPDIR=" + str(SCORE_D / "tmp"),
        str(CANONICAL_PYTHON),
        "-E",
        "-B",
        "-s",
        str(SCORE_D / "score_first_cell.py"),
        "--evidence-root",
        str(EXPORT_D),
        "--export-root",
        str(EXPORT_ROOT),
        "--binding",
        str(EXPORT_D / "EXPORT_BINDING_V4.json"),
        "--source-manifest",
        str(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json"),
        "--source-root",
        str(SOURCE_ROOT),
        "--data-root",
        str(DATA_ROOT),
        "--out",
        str(RESULT_PATH),
    ]
    try:
        returncode, stdout, stderr = bounded_submit(command)
        submission: dict[str, object] = {
            "outcome": "submitted" if returncode == 0 else "submit_failed_or_unknown",
            "returncode": returncode,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "argv": command,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        submission = {
            "outcome": "submit_error_or_timeout_unknown_never_retry",
            "returncode": None,
            "error": repr(error),
            "stdout_b64": "",
            "stderr_b64": "",
            "argv": command,
        }
    fields = (
        "Id",
        "LoadState",
        "InvocationID",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
        "ControlGroup",
    )
    post: dict[str, object] = {"outcome": "captured", "properties": {}}
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            properties = show(UNIT, fields)
            post["properties"] = properties
            if properties.get("Id") == UNIT and is_invocation_id(properties.get("InvocationID")):
                break
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            post = {"outcome": "query_failed_unknown", "error": repr(error), "properties": {}}
        time.sleep(0.25)
    properties = post.get("properties", {})
    invocation = properties.get("InvocationID") if isinstance(properties, dict) else None
    if not is_invocation_id(invocation):
        write_json_exclusive(
            SCORE_D / "LAUNCH_UNRESOLVED.json",
            {
                "schema": "fg19.first-cell-score-unresolved-launch.v1",
                "unit": UNIT,
                "invocation_id": None,
                "submission": submission,
                "systemctl_after_submission": post,
                "unknown_outcome_policy": "Inspect only this exact unit and score directory; never retry.",
            },
        )
        directory_fd = os.open(
            SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        raise RuntimeError(
            "score invocation ID unresolved; durable unknown outcome retained, never retry"
        )
    launch = {
        "schema": "fg19.first-cell-score-launch.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "invocation_id": invocation,
        "export_unit": EXPORT_UNIT,
        "export_invocation_id": admission["export_invocation_id"],
        "admission_sha256": digest(
            (json.dumps(admission, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        ),
        "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Inspect only this exact unit and score directory; never retry.",
    }
    write_json_exclusive(SCORE_D / "LAUNCH.json", launch)
    directory_fd = os.open(
        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if submission.get("outcome") != "submitted" or submission.get("returncode") != 0:
        raise RuntimeError(
            "submission outcome is not one confirmed launch; preserve records and never retry"
        )
    return launch


def main() -> None:
    """Run all read-only gates before staging the one private CPU score packet."""
    payload = payload_bytes()
    acceptance = json_object(payload["EXPORT_ACCEPTANCE.json"], "root export acceptance")
    exact_stop = json_object(payload["EXPORT_EXACT_STOP.json"], "root export exact stop")
    checks = check_paths_and_inputs(payload)
    export_invocation = validate_export_authority(
        acceptance,
        exact_stop,
        acceptance_sha256=EXPORT_ACCEPTANCE_SHA256,
        binding_sha256=EXPORT_BINDING_SHA256,
        review_sha256=SOURCE_REVIEW_SHA256,
        expected_unit=EXPORT_UNIT,
    )
    staged = stage_packet(payload)
    admission = {
        "schema": "fg19.first-cell-score-admission.v1",
        "status": "FRESH_GATES_PASSED_SUBMISSION_PENDING",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "export_unit": EXPORT_UNIT,
        "export_invocation_id": export_invocation,
        "export_acceptance": {
            "bytes": EXPORT_ACCEPTANCE_BYTES,
            "sha256": EXPORT_ACCEPTANCE_SHA256,
            "staged_path": "accepted_export/EXPORT_ACCEPTANCE.json",
        },
        "export_exact_stop": {
            "bytes": EXPORT_EXACT_STOP_BYTES,
            "sha256": EXPORT_EXACT_STOP_SHA256,
            "staged_path": "accepted_export/EXPORT_EXACT_STOP.json",
        },
        "checks": checks,
        "staged_files": staged,
        "result_path": str(RESULT_PATH),
        "result_create_only_and_absent_before_mutation": True,
        "resource_limits": {
            "cpu_quota": WORKER_CPU_QUOTA,
            "memory_high_bytes": WORKER_MEMORY_HIGH_BYTES,
            "memory_max_bytes": WORKER_MEMORY_MAX_BYTES,
            "memory_swap_max_bytes": 0,
            "runtime_seconds": WORKER_RUNTIME_SECONDS,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "scope": "One source-qualified streaming score of the fixed five epoch-5 F0/seed-1001 arms over 468 exposed development groups; no inference, fit, retuning, model load, new endpoint selection, or public-transfer claim.",
        "unknown_outcome_policy": "Never retry or recreate this unit, score directory, or create-only result path; inspect only the exact durable receipt and unit.",
    }
    write_json_exclusive(SCORE_D / "ADMISSION.json", admission)
    write_exclusive(SCORE_D / "unit.log", b"")
    directory_fd = os.open(
        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    launch = submit_once(admission)
    admission_raw, _ = stable_file(SCORE_D / "ADMISSION.json", 1_000_000)
    launch_raw, _ = stable_file(SCORE_D / "LAUNCH.json", 1_000_000)
    require(
        len(admission_raw) + len(launch_raw) <= 4_000_000,
        "bounded score launch response exceeds 4 MB",
    )
    durable_admission = json_object(admission_raw, "durable score admission")
    durable_launch = json_object(launch_raw, "durable score launch")
    require(
        durable_admission.get("unit") == UNIT
        and durable_launch.get("unit") == UNIT
        and durable_launch.get("invocation_id") == launch.get("invocation_id")
        and is_invocation_id(durable_launch.get("invocation_id"))
        and durable_launch.get("submission", {}).get("outcome") == "submitted"
        and durable_launch.get("submission", {}).get("returncode") == 0,
        "durable score receipts do not bind one successful launch",
    )
    print(
        json.dumps(
            {
                "unit": UNIT,
                "outcome": "submitted",
                "admission_base64": base64.b64encode(admission_raw).decode("ascii"),
                "launch_base64": base64.b64encode(launch_raw).decode("ascii"),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
