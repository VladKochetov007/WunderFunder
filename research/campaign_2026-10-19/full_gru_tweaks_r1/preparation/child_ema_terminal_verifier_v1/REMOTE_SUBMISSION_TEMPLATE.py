"""One-shot, create-only CPU admission for the completed EMA integrity check."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path

PAYLOAD = {}
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA = B / "data"
RUN = DATA / "artifacts/runs/fg19_child_ema_f0_s1001_fit_v1"
SOURCE = B / "source-fg19-cache-68ff6829"
SPLIT = B / "splits/fold0.json"
CACHE = B / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"
DICT = DATA / "artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
PARENT_CHECKPOINT = DATA / "artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt"
REGISTRY = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
EVIDENCE = B / "research_tmp/fg19_child_ema_fit_v1"
PROFILE_EVIDENCE = B / "research_tmp/fg19_child_allthree_profile_v1"
CACHE_RECEIPTS = B / "research/full_gru_tweaks_r1/cache_cpu_v2"
D = B / "research_tmp/fg19_child_ema_verify_v1"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
UNIT = "computer-vision-homework-fg19-child-ema-verify-v1.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
PATH_EXCEPTION_ENV = "FG19_APPROVED_PATH_COMPAT_EXCEPTION"
PATH_EXCEPTION_VALUE = "pathlib._local.PosixPath:exact-dictionary-path-only"
INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
RUN_NAME = "fg19_child_ema_f0_s1001_fit_v1"
CONTROL_RUN = "fg19_child_control_f0_s1001_fit_v1"
CONTROL_UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
CONTROL_INVOCATION = "e66e0d0703314e4fb5d47dcdbfcaf202"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
TERMINAL_SHA = "7cb75057155f3582e0493d1f6ca8c88eff475b022215d755f53751495f5c7d0a"
NATIVE_RAW_SHA = "f8534769efdfd1949d5570927a0b5804f2819570c6154bac60c38387410a7cad"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_EXACT_STOP_SHA = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
PARENT_ACCEPTANCE_SHA = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
PARENT_STOP_SHA = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
PARENT_CHECKPOINT_SHA = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
DICTIONARY_ACCEPTANCE_SHA = "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
DICTIONARY_STOP_SHA = "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158"
DICTIONARY_SHA = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
PROFILE_ACCEPTANCE_SHA = "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045"
PROFILE_STOP_SHA = "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce"
PROFILE_ADMISSION_SHA = "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5"
PROFILE_LAUNCH_SHA = "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44"
PROFILE_LOG_SHA = "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289"
ADMISSION_SHA = "db13e5433cb0a414c1a235806220c53db4ded6c128fc5ffbda5e465dde9fde51"
LAUNCH_SHA = "8c43f6fb26d3f6a046304617094e266a3e1940c83a7ea56787781fa176ef133e"
LAUNCH_ACCEPTANCE_SHA = "4fc34f4769da2f08aaf1fcca97d87731dc38b1709a93afa9376be003c28ffb72"
CHECKER_SHA = "a63ec6c4f6b0a3c4ba7e0cb8800e4787c5d89f1a652e0deec15ad84ffc95d30a"
CONTROL_CHECKER_SHA = "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167"
PARENT_CHECKER_SHA = "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41"
SUPPORT_SHA = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
TEACHER_SHA = "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5"
MAX_SOURCE_BYTES = 128_000_000
MAX_JSON_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_ARTIFACT_BYTES = 128_000_000
SUPPORT_MAX_BYTES = 1_500_000_000
PARENT_TARGET_BYTES = 17_000_000_000
PARENT_RESERVE_BYTES = 1_000_000_000
EXPECTED_HASHES = {
    "child_ema_terminal_check_v1.py": CHECKER_SHA,
    "child_control_terminal_check_v3.py": CONTROL_CHECKER_SHA,
    "parent_fit_terminal_check_v3.py": PARENT_CHECKER_SHA,
    "propensity_terminal_support.py": SUPPORT_SHA,
    "teacher_s5_terminal_check_v1.py": TEACHER_SHA,
    "SOURCE_MANIFEST.json": MANIFEST_SHA,
    "LAUNCH_ACCEPTANCE.json": LAUNCH_ACCEPTANCE_SHA,
    "TERMINAL_EVIDENCE.json": TERMINAL_SHA,
    "NATIVE_STDOUT.raw.json": NATIVE_RAW_SHA,
}
MAX_INPUT_BYTES = {
    "child_ema_terminal_check_v1.py": 40_000,
    "child_control_terminal_check_v3.py": 80_000,
    "parent_fit_terminal_check_v3.py": 70_000,
    "propensity_terminal_support.py": 15_000,
    "teacher_s5_terminal_check_v1.py": 40_000,
    "SOURCE_MANIFEST.json": 30_000,
    "LAUNCH_ACCEPTANCE.json": 6_000,
    "TERMINAL_EVIDENCE.json": 2_000,
    "NATIVE_STDOUT.raw.json": 80_000,
}
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE_SHA,
    "PARENT_EXACT_STOP.json": PARENT_STOP_SHA,
    "DICTIONARY_ACCEPTANCE.json": DICTIONARY_ACCEPTANCE_SHA,
    "DICTIONARY_EXACT_STOP.json": DICTIONARY_STOP_SHA,
    "PROFILE_TERMINAL_ACCEPTANCE.json": PROFILE_ACCEPTANCE_SHA,
    "PROFILE_EXACT_STOP.json": PROFILE_STOP_SHA,
    "CONTROL_TERMINAL_ACCEPTANCE.json": "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    "CONTROL_EXACT_STOP.json": "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
}


def require(condition: bool, message: str) -> None:
    """Fail closed before any output creation or support submission."""
    if not condition:
        raise RuntimeError(message)


def require_path_exception() -> str:
    """Require the explicit one-name, exact-path compatibility exception."""
    accepted = os.environ.get(PATH_EXCEPTION_ENV)
    require(accepted == PATH_EXCEPTION_VALUE, f"{PATH_EXCEPTION_ENV} must be explicitly approved")
    return accepted


def digest(raw: bytes) -> str:
    """Return a SHA-256 digest for bounded bytes."""
    return hashlib.sha256(raw).hexdigest()


def utc_now() -> str:
    """Return a UTC timestamp for durable admission and launch records."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse a small JSON object without changing its original staged bytes."""
    value = json.loads(raw)
    require(isinstance(value, dict), f"JSON object expected: {name}")
    return value


def payload_bytes() -> dict[str, bytes]:
    """Decode only fixed, locally pinned checker and metadata inputs."""
    require(set(PAYLOAD) == set(EXPECTED_HASHES), "fixed payload names differ")
    require(set(PAYLOAD) == set(MAX_INPUT_BYTES), "fixed payload caps differ")
    payload: dict[str, bytes] = {}
    for name, item in PAYLOAD.items():
        raw = base64.b64decode(item["base64"], validate=True)
        require(type(item.get("bytes")) is int, f"invalid payload byte count: {name}")
        require(len(raw) == item["bytes"] <= MAX_INPUT_BYTES[name], f"payload size differs: {name}")
        require(
            digest(raw) == EXPECTED_HASHES[name] == item.get("sha256"),
            f"payload SHA differs: {name}",
        )
        payload[name] = raw
    return payload


def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
    """Read a bounded regular file and verify that its identity stayed stable."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, f"unsafe/large file: {path}")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    after = path.stat()
    require(len(raw) == before.st_size, f"file changed or exceeded cap: {path}")
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while read: {path}",
    )
    return raw, {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def file_hash(path: Path, cap: int) -> tuple[str, dict[str, int]]:
    """Stream a bounded code or metadata file without loading its contents."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, f"unsafe/large file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return hasher.hexdigest(), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def physical_code_root(path: Path) -> None:
    """Require a fixed code root with no symlink redirection."""
    require(path.is_dir() and not path.is_symlink(), f"code root unavailable or linked: {path}")
    require(path.resolve(strict=True) == path, f"code root is redirected: {path}")


def show(unit: str, fields: tuple[str, ...], timeout: int = 15) -> dict[str, str]:
    """Read explicitly named systemd properties for the exact current unit."""
    command = ["systemctl", "--user", "show", "--all", unit]
    for field in fields:
        command.extend(("-p", field))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    require(result.returncode == 0, f"systemctl show failed for {unit}: {result.stderr[:2000]}")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def write_exclusive(path: Path, raw: bytes) -> None:
    """Create and sync one file without replacement."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    """Create one compact metadata receipt and sync its parent directory."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    write_exclusive(path, raw)


def captured_receipt(capture: dict[str, object], name: str) -> bytes:
    """Decode one exact base64 receipt from the native capture metadata."""
    receipts = capture.get("receipts")
    require(
        isinstance(receipts, dict) and isinstance(receipts.get(name), dict), f"capture omits {name}"
    )
    record = receipts[name]
    require(
        record.get("present") is True and record.get("regular_file") is True,
        f"capture marks {name} unsafe",
    )
    raw = base64.b64decode(record.get("base64", ""), validate=True)
    require(
        type(record.get("bytes")) is int and len(raw) == record["bytes"],
        f"capture byte count differs: {name}",
    )
    require(digest(raw) == record.get("sha256"), f"capture SHA differs: {name}")
    return raw


def check_native(
    payload: dict[str, bytes],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Join the EMA receipts, strict typed exit evidence, and current native unit."""
    admission_raw, _ = stable_file(EVIDENCE / "ADMISSION.json", 10_000)
    launch_raw, _ = stable_file(EVIDENCE / "LAUNCH.json", 10_000)
    require(digest(admission_raw) == ADMISSION_SHA, "current EMA admission receipt digest differs")
    require(digest(launch_raw) == LAUNCH_SHA, "current EMA launch receipt digest differs")
    admission = json_object(admission_raw, "EMA admission")
    launch = json_object(launch_raw, "EMA launch")
    accepted = json_object(payload["LAUNCH_ACCEPTANCE.json"], "EMA root launch acceptance")
    terminal = json_object(payload["TERMINAL_EVIDENCE.json"], "EMA typed terminal evidence")
    capture = json_object(payload["NATIVE_STDOUT.raw.json"], "EMA native raw capture")
    control_predecessor = {
        "acceptance_sha256": READONLY_RECEIPTS["CONTROL_TERMINAL_ACCEPTANCE.json"],
        "exact_stop_sha256": READONLY_RECEIPTS["CONTROL_EXACT_STOP.json"],
        "invocation_id": CONTROL_INVOCATION,
        "run_name": CONTROL_RUN,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": CONTROL_UNIT,
    }
    require(
        (
            admission.get("schema"),
            admission.get("run_name"),
            admission.get("unit"),
            admission.get("source_commit"),
            admission.get("source_files_verified"),
            admission.get("source_manifest_sha256"),
            admission.get("split_sha256"),
            admission.get("control_predecessor"),
        )
        == (
            "fg19-child-ema-fit-admission.v1",
            RUN_NAME,
            NATIVE_UNIT,
            COMMIT,
            167,
            MANIFEST_SHA,
            SPLIT_SHA,
            control_predecessor,
        ),
        "EMA admission identity or source differs",
    )
    require(admission.get("populated") == {}, "EMA admission recorded populated descendants")
    require(
        (launch.get("schema"), launch.get("unit"), launch.get("submission", {}).get("returncode"))
        == ("fg19-child-ema-fit-launch.v1", NATIVE_UNIT, 0),
        "EMA original launch identity differs",
    )
    require(
        launch.get("control_predecessor") == control_predecessor, "EMA launch predecessor differs"
    )
    expected_root_receipts = {
        "admission": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/ADMISSION.json",
            "bytes": 9_880,
            "sha256": ADMISSION_SHA,
        },
        "launch": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/LAUNCH.json",
            "bytes": 7_404,
            "sha256": LAUNCH_SHA,
        },
        "body": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/REMOTE_SUBMISSION.py",
            "bytes": 163_040,
            "sha256": "f83ac3120a2e445f1752fcd026123a0e8a21ba53594d27b398462a5236ffebd6",
        },
        "source_review": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/CHILD_EMA_FIT_SOURCE_REVIEW_V1.json",
            "bytes": 8_000,
            "sha256": "3d2f78b0c8902c1a43748a237310ae6a31277e60fa9c18c13a5d162c13722138",
        },
    }
    require(
        (
            accepted.get("schema"),
            accepted.get("status"),
            accepted.get("run_name"),
            accepted.get("unit"),
            accepted.get("invocation_id"),
            accepted.get("control_predecessor"),
            {key: accepted.get(key) for key in expected_root_receipts},
        )
        == (
            "fg19-child-ema-fit-launch-acceptance.v1",
            "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            control_predecessor,
            expected_root_receipts,
        ),
        "EMA root launch acceptance differs",
    )
    require(
        (
            terminal.get("schema"),
            terminal.get("run_name"),
            terminal.get("unit"),
            terminal.get("invocation_id"),
            terminal.get("active_state"),
            terminal.get("sub_state"),
            terminal.get("raw_bytes"),
            terminal.get("raw_sha256"),
        )
        == (
            "fg19.child-ema-fit-native-terminal-evidence.v1",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            "active",
            "exited",
            75_226,
            NATIVE_RAW_SHA,
        ),
        "EMA typed native terminal evidence differs",
    )
    require(
        type(terminal.get("main_pid")) is int
        and terminal["main_pid"] == 0
        and type(terminal.get("exec_main_code")) is int
        and terminal["exec_main_code"] == 1
        and type(terminal.get("exec_main_status")) is int
        and terminal["exec_main_status"] == 0,
        "EMA terminal process/exit fields must be exact integers",
    )
    require(
        terminal.get("raw_path")
        == "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/terminal_native_capture_v1/STDOUT.raw.json",
        "EMA native raw capture path differs",
    )
    require(type(terminal.get("raw_bytes")) is int, "EMA raw capture byte count must be an integer")
    require(
        (
            capture.get("schema"),
            capture.get("expected_run"),
            capture.get("expected_unit"),
            capture.get("expected_invocation_id"),
            capture.get("captured_utc"),
        )
        == (
            "fg19-child-control-startup-capture.v2",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            terminal.get("captured_utc"),
        ),
        "EMA raw native capture identity differs",
    )
    captured_unit = capture.get("unit_properties")
    require(isinstance(captured_unit, dict), "EMA native unit properties are missing")
    require(
        tuple(
            captured_unit.get(key)
            for key in (
                "InvocationID",
                "ActiveState",
                "SubState",
                "MainPID",
                "ExecMainCode",
                "ExecMainStatus",
                "Result",
            )
        )
        == (INVOCATION, "active", "exited", "0", "1", "0", "success"),
        "EMA captured native unit status differs",
    )
    require(
        captured_receipt(capture, "ADMISSION.json") == admission_raw,
        "captured EMA admission bytes differ",
    )
    require(
        captured_receipt(capture, "LAUNCH.json") == launch_raw,
        "captured EMA launch bytes differ",
    )
    current = show(
        NATIVE_UNIT,
        (
            "LoadState",
            "InvocationID",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
            "CPUUsageNSec",
        ),
    )
    require(
        tuple(
            current.get(key)
            for key in (
                "LoadState",
                "InvocationID",
                "ActiveState",
                "SubState",
                "MainPID",
                "ExecMainCode",
                "ExecMainStatus",
                "Result",
            )
        )
        == ("loaded", INVOCATION, "active", "exited", "0", "1", "0", "success"),
        "fresh EMA invocation is no longer the exact successful exited unit",
    )
    require(
        current.get("CPUUsageNSec", "").isdigit() and int(current["CPUUsageNSec"]) > 0,
        "native CPU usage evidence is missing",
    )
    raw_log = captured_receipt(capture, "unit.log")
    log, log_stat = stable_file(EVIDENCE / "unit.log", MAX_JSON_BYTES)
    require(log == raw_log, "EMA native unit log differs from the exact captured bytes")
    artifacts = check_run_stats(capture)
    return (
        current,
        {
            "typed": terminal,
            "raw_capture_sha256": NATIVE_RAW_SHA,
            "raw_capture_bytes": len(payload["NATIVE_STDOUT.raw.json"]),
            "unit_log": {**log_stat, "sha256": digest(log)},
        },
        artifacts,
    )


def check_run_stats(capture: dict[str, object]) -> dict[str, object]:
    """Recheck terminal run artifacts by stat only before the checker can read them."""
    run_directory = RUN.lstat()
    require(
        stat.S_ISDIR(run_directory.st_mode) and not RUN.is_symlink(),
        "terminal run directory is not a direct directory",
    )
    inventory = capture.get("run_directory_stat_only")
    require(
        isinstance(inventory, dict)
        and inventory.get("exists") is True
        and inventory.get("symlink") is False,
        "captured run directory identity is unsafe",
    )
    entries = inventory.get("entries")
    require(isinstance(entries, list), "captured run inventory is missing")
    expected_names = {
        "model.pt",
        *(f"model_ep{epoch}.pt" for epoch in range(6)),
        "training_state.pt",
        "provenance.json",
        "history.json",
    }
    by_name = {row.get("name"): row for row in entries if isinstance(row, dict)}
    require(
        len(by_name) == len(entries) and set(by_name) == expected_names,
        "captured run file inventory differs",
    )
    rows: dict[str, object] = {}
    total = 0
    for name in sorted(expected_names):
        limit = (
            MAX_STATE_BYTES
            if name == "training_state.pt"
            else MAX_MODEL_BYTES
            if name.startswith("model")
            else MAX_JSON_BYTES
        )
        path = RUN / name
        before = path.lstat()
        require(
            stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size <= limit,
            f"unsafe or oversized run artifact: {name}",
        )
        actual = {
            "name": name,
            "bytes": before.st_size,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "mode": before.st_mode,
            "is_symlink": False,
        }
        require(actual == by_name[name], f"terminal stat differs from native capture: {name}")
        total += before.st_size
        rows[name] = {key: actual[key] for key in ("bytes", "device", "inode", "mtime_ns")}
    require(total <= MAX_ARTIFACT_BYTES, "terminal artifact aggregate exceeds the checker cap")
    return {
        "directory": {
            "path": str(RUN),
            "resolved": str(RUN.resolve(strict=True)),
            "device": run_directory.st_dev,
            "inode": run_directory.st_ino,
            "mtime_ns": run_directory.st_mtime_ns,
        },
        "files": rows,
        "aggregate_bytes": total,
        "payload_bytes_read": 0,
    }


def check_source(payload: dict[str, bytes], admission: dict[str, object]) -> dict[str, object]:
    """Verify the already staged 167-file source tree without copying or modifying it."""
    manifest = json_object(payload["SOURCE_MANIFEST.json"], "source manifest")
    entries = manifest.get("files")
    require(
        (manifest.get("source_commit"), len(entries) if isinstance(entries, list) else None)
        == (COMMIT, 167),
        "source manifest identity differs",
    )
    require(
        (admission.get("source_files_verified"), admission.get("source_manifest_sha256"))
        == (167, MANIFEST_SHA),
        "admitted source pin differs",
    )
    physical_code_root(SOURCE)
    source_map: dict[str, tuple[str, int]] = {}
    total = 0
    for row in entries:
        require(isinstance(row, dict), "malformed source manifest row")
        relative = Path(row.get("path", ""))
        size, expected_sha = row.get("bytes"), row.get("sha256")
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and type(size) is int
            and size >= 0,
            "unsafe source manifest path",
        )
        require(
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and str(relative) not in source_map,
            "invalid source manifest digest",
        )
        path = SOURCE / relative
        ancestor = path
        while ancestor != SOURCE:
            require(not ancestor.is_symlink(), f"source code path contains a symlink: {ancestor}")
            ancestor = ancestor.parent
        actual_sha, info = file_hash(path, size)
        require(
            (actual_sha, info["bytes"]) == (expected_sha, size), f"source pin differs: {relative}"
        )
        source_map[str(relative)] = (expected_sha, size)
        total += size
        require(total <= MAX_SOURCE_BYTES, "source tree exceeds bounded code budget")
    inventory: set[str] = set()
    for root, dirs, files in os.walk(SOURCE, followlinks=False):
        for name in dirs:
            require(not (Path(root) / name).is_symlink(), "source contains a linked directory")
        for name in files:
            path = Path(root) / name
            require(not path.is_symlink(), "source contains a linked file")
            inventory.add(str(path.relative_to(SOURCE)))
    require(inventory == set(source_map), "source tree inventory differs from its frozen manifest")
    return {
        "commit": COMMIT,
        "file_count": len(source_map),
        "bytes": total,
        "manifest_sha256": MANIFEST_SHA,
    }


def check_split(payload: dict[str, bytes], admission: dict[str, object]) -> dict[str, object]:
    """Bind the canonical fold split by its exact small metadata bytes."""
    require(
        digest(payload["SOURCE_MANIFEST.json"]) == MANIFEST_SHA, "source manifest digest differs"
    )
    raw, info = stable_file(SPLIT, 2_000_000)
    require(digest(raw) == SPLIT_SHA == admission.get("split_sha256"), "canonical split differs")
    json_object(raw, "canonical split")
    return {**info, "sha256": SPLIT_SHA}


def check_registered_data(admission: dict[str, object]) -> dict[str, object]:
    """Allow only the six registered paths at their exact original resolved stats."""
    raw, registry_stat = stable_file(REGISTRY, 2_000_000)
    require(digest(raw) == REGISTRY_SHA, "portable data registry digest differs")
    registry = json_object(raw, "portable data registry")
    sources = registry.get("sources")
    expected = admission.get("data_identity_stats_only")
    require(
        isinstance(sources, dict)
        and isinstance(expected, dict)
        and len(expected) == 6
        and set(sources) == set(expected),
        "registered data membership differs",
    )
    observed: dict[str, object] = {}
    for relative, pin in expected.items():
        require(
            isinstance(pin, dict) and isinstance(sources.get(relative), dict),
            f"malformed registered path: {relative}",
        )
        path = DATA / relative
        entry = sources[relative]
        resolved = path.resolve(strict=True)
        target = path.stat()
        actual = {
            "path": str(path),
            "resolved": str(resolved),
            "islink": path.is_symlink(),
            "device": target.st_dev,
            "bytes": target.st_size,
            "inode": target.st_ino,
            "mtime_ns": target.st_mtime_ns,
            "registered_sha256": entry.get("sha256"),
        }
        require(
            actual == {key: pin.get(key) for key in actual},
            f"registered data path/stat differs: {relative}",
        )
        require(
            entry.get("sha256") == pin.get("registered_sha256")
            and target.st_size == entry.get("original_identity", {}).get("bytes"),
            f"portable registry source differs: {relative}",
        )
        observed[relative] = actual
    return {"registry": {**registry_stat, "sha256": REGISTRY_SHA}, "paths_stat_only": observed}


def check_receipt_file(path: Path, expected_sha: str, cap: int = 1_000_000) -> dict[str, object]:
    """Verify one existing acceptance or stop receipt read-only."""
    raw, info = stable_file(path, cap)
    require(digest(raw) == expected_sha, f"read-only receipt digest differs: {path.name}")
    return {**info, "sha256": expected_sha}


def check_predecessors(admission: dict[str, object]) -> dict[str, object]:
    """Verify parent, profile and dictionary receipts without recreating any of them."""
    staged = admission.get("staged_receipt_sha256")
    require(
        isinstance(staged, dict)
        and {name: staged.get(name) for name in READONLY_RECEIPTS} == READONLY_RECEIPTS,
        "admitted predecessor receipt map differs",
    )
    native_receipts = {
        name: check_receipt_file(EVIDENCE / name, sha) for name, sha in READONLY_RECEIPTS.items()
    }
    profile_admission, profile_admission_stat = stable_file(
        PROFILE_EVIDENCE / "ADMISSION.json", 20_000
    )
    profile_launch, profile_launch_stat = stable_file(PROFILE_EVIDENCE / "LAUNCH.json", 20_000)
    profile_log_sha, profile_log_stat = file_hash(PROFILE_EVIDENCE / "unit.log", 1_000_000)
    require(
        digest(profile_admission) == PROFILE_ADMISSION_SHA
        and digest(profile_launch) == PROFILE_LAUNCH_SHA
        and profile_log_sha == PROFILE_LOG_SHA,
        "profile evidence file pins differ",
    )
    profile_raw_stat = {
        "ADMISSION.json": profile_admission_stat,
        "LAUNCH.json": profile_launch_stat,
        "unit.log": profile_log_stat,
    }
    require(
        (
            native_receipts["PROFILE_TERMINAL_ACCEPTANCE.json"]["sha256"],
            native_receipts["PROFILE_EXACT_STOP.json"]["sha256"],
        )
        == (PROFILE_ACCEPTANCE_SHA, PROFILE_STOP_SHA),
        "profile acceptance/stop pins differ",
    )
    parent_identity = admission.get("parent_checkpoint_identity")
    dictionary_identity = admission.get("dictionary")
    require(
        isinstance(parent_identity, dict) and isinstance(dictionary_identity, dict),
        "accepted parent/dictionary stats are missing",
    )
    parent_path = Path(parent_identity.get("path", ""))
    dictionary_path = Path(dictionary_identity.get("path", ""))
    require(
        parent_path == PARENT_CHECKPOINT and dictionary_path == DICT,
        "accepted ancestor paths differ",
    )
    require(
        admission.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA
        and parent_identity.get("sha256") == PARENT_CHECKPOINT_SHA,
        "accepted parent checkpoint digest metadata differs",
    )
    require(
        dictionary_identity.get("sha256") == DICTIONARY_SHA,
        "accepted dictionary digest metadata differs",
    )
    parent_stat = parent_path.stat()
    parent_observed = {
        "path": str(parent_path),
        "bytes": parent_stat.st_size,
        "device": parent_stat.st_dev,
        "inode": parent_stat.st_ino,
        "mtime_ns": parent_stat.st_mtime_ns,
    }
    require(
        parent_observed == {key: parent_identity.get(key) for key in parent_observed},
        "accepted parent checkpoint stat differs",
    )
    dictionary_stat = dictionary_path.stat()
    dictionary_observed = {
        "path": str(dictionary_path),
        "bytes": dictionary_stat.st_size,
        "device": dictionary_stat.st_dev,
        "inode": dictionary_stat.st_ino,
    }
    require(
        dictionary_observed == {key: dictionary_identity.get(key) for key in dictionary_observed},
        "accepted dictionary artifact stat differs",
    )
    return {
        "receipts_read_only": native_receipts,
        "profile_files_read_only": profile_raw_stat,
        "parent_checkpoint_stat_only": parent_observed,
        "dictionary_artifact_stat_only": dictionary_observed,
        "payload_bytes_hashed": False,
    }


def check_cache(
    payload: dict[str, bytes], admission: dict[str, object], split: dict[str, object]
) -> dict[str, object]:
    """Bind accepted cache metadata and array stats without opening array payloads."""
    cache = admission.get("cache")
    require(isinstance(cache, dict), "admitted cache identity is missing")
    acceptance_raw, _ = stable_file(CACHE_RECEIPTS / "TERMINAL_ACCEPTANCE.json", 10_000)
    stop_raw, _ = stable_file(CACHE_RECEIPTS / "EXACT_STOP.json", 10_000)
    require(
        digest(acceptance_raw) == CACHE_ACCEPTANCE_SHA == cache.get("acceptance_sha256")
        and digest(stop_raw) == CACHE_EXACT_STOP_SHA == cache.get("exact_stop_sha256"),
        "cache receipt pins differ",
    )
    receipt_paths = (
        (
            CACHE_RECEIPTS / "TERMINAL_ACCEPTANCE.json",
            acceptance_raw,
            cache.get("acceptance_sha256"),
        ),
        (CACHE_RECEIPTS / "EXACT_STOP.json", stop_raw, cache.get("exact_stop_sha256")),
    )
    for path, expected_raw, expected_sha in receipt_paths:
        actual, _ = stable_file(path, 10_000)
        require(
            actual == expected_raw and digest(actual) == expected_sha,
            f"read-only cache receipt differs: {path.name}",
        )
    acceptance = json_object(acceptance_raw, "cache acceptance")
    stop = json_object(stop_raw, "cache exact stop")
    require(
        (
            acceptance.get("status"),
            acceptance.get("metadata_sha256"),
            acceptance.get("source_commit"),
            acceptance.get("source_files_verified"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
        )
        == (
            "ACCEPTED",
            CACHE_METADATA_SHA,
            COMMIT,
            167,
            cache.get("unit"),
            cache.get("invocation_id"),
        ),
        "cache acceptance identity differs",
    )
    before = stop.get("before", {}).get(cache.get("unit"), {})
    after = stop.get("after", {}).get(cache.get("unit"), {})
    stopped = stop.get("stop", {}).get(cache.get("unit"), {})
    require(
        (
            before.get("InvocationID"),
            before.get("ExecMainStatus"),
            stopped.get("returncode"),
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("InvocationID"),
            stop.get("populated"),
        )
        == (cache.get("invocation_id"), "0", 0, "not-found", "inactive", "dead", "", {}),
        "cache exact-stop evidence differs",
    )
    metadata, metadata_stat = stable_file(CACHE / "metadata.json", MAX_JSON_BYTES)
    require(digest(metadata) == CACHE_METADATA_SHA, "current cache metadata digest differs")
    cache_metadata = json_object(metadata, "cache metadata")
    require(
        cache_metadata.get("split_manifest_sha256") == split.get("sha256"),
        "cache split identity differs",
    )
    stats = cache.get("array_stats_only")
    output = acceptance.get("output")
    require(
        isinstance(stats, dict) and isinstance(output, dict),
        "cache array stat/acceptance maps are missing",
    )
    arrays: dict[str, object] = {}
    for name, count, size, sha in (
        (
            "train",
            10_535,
            1_685_600_128,
            "4abbb9be31ec1135a5e0ff5bb87c93a2db6e44943e95d23ff70835b8044c5194",
        ),
        (
            "valid",
            1_405,
            224_800_128,
            "6963d3c565caf469b023a61ba6d2dd376d929b2bcf724af88492dc8adb76f591",
        ),
    ):
        path = CACHE / f"main_homogeneous_{name}.npy"
        st = path.stat()
        expected = stats.get(name)
        accepted = output.get(name)
        require(
            isinstance(expected, dict) and isinstance(accepted, dict),
            f"cache {name} receipt is missing",
        )
        current_stat = {
            "bytes": st.st_size,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
        }
        require(current_stat == expected, f"cache {name} stat differs")
        require(
            (
                st.st_size,
                accepted.get("sha256"),
                accepted.get("bytes"),
                accepted.get("shape"),
                accepted.get("dtype"),
            )
            == (size, sha, size, [count, 20_000, 2], "float32"),
            f"cache {name} accepted identity differs",
        )
        arrays[name] = {
            **current_stat,
            "resolved": str(path.resolve(strict=True)),
            "payload_read_or_hashed": False,
        }
    return {
        "metadata_sha256": CACHE_METADATA_SHA,
        "metadata": metadata_stat,
        "acceptance_sha256": digest(acceptance_raw),
        "exact_stop_sha256": digest(stop_raw),
        "arrays_stat_only": arrays,
        "array_payloads_read_or_hashed": False,
    }


def check_parent() -> dict[str, object]:
    """Enforce the empty two-core parent, one-time headroom reservation and disk floor."""
    fields = (
        "ControlGroup",
        "MemoryCurrent",
        "CPUQuotaPerSecUSec",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
    )
    parent = show("wnn-campaign.slice", fields)
    require(
        (
            parent.get("CPUQuotaPerSecUSec"),
            parent.get("MemoryHigh"),
            parent.get("MemoryMax"),
            parent.get("MemorySwapMax"),
        )
        == ("2s", "18000000000", "20000000000", "0"),
        "parent resource envelope differs",
    )
    control_group = parent.get("ControlGroup", "")
    require(
        control_group not in ("", "/") and control_group.endswith("/wnn-campaign.slice"),
        "empty/unscoped parent cgroup is NA; refusing host-root fallback",
    )
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    require(
        cgroup.is_dir() and cgroup.resolve(strict=True) == cgroup,
        "parent cgroup path unavailable or redirected",
    )
    proc_files = [cgroup / "cgroup.procs", *cgroup.rglob("cgroup.procs")]
    require(0 < len(proc_files) < 128, "parent cgroup process inventory is NA or exceeds its bound")
    populated: dict[str, str] = {}
    for path in proc_files:
        value = path.read_text(encoding="ascii")
        if value.strip():
            populated[str(path)] = value
    require(not populated, "campaign cgroup descendants are populated")
    current_raw = (cgroup / "memory.current").read_text(encoding="ascii").strip()
    require(current_raw.isdigit(), "parent memory current is unavailable")
    current = int(current_raw)
    require(
        current + SUPPORT_MAX_BYTES + PARENT_RESERVE_BYTES < PARENT_TARGET_BYTES,
        "parent headroom after one support reservation is insufficient",
    )
    events = {
        name: (cgroup / name).read_text(encoding="ascii")
        for name in ("memory.events", "memory.events.local")
    }
    free = os.statvfs(B).f_bavail * os.statvfs(B).f_frsize
    require(free >= 25 * 1024**3, "project filesystem free space is below 25 GiB")
    project_device = os.stat(B).st_dev
    scratch_device = os.stat(D.parent).st_dev
    require(project_device == scratch_device, "research_tmp device differs from the project device")
    result = subprocess.run(
        ["findmnt", "-T", str(D.parent), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        result.returncode == 0 and result.stdout.strip() not in ("tmpfs", "ramfs"),
        "research_tmp is not on physical storage",
    )
    return {
        "systemd": parent,
        "cgroup": str(cgroup),
        "populated": populated,
        "memory_current_bytes": current,
        "headroom_formula": "current + 1500000000 support max + 1000000000 reserve < 17000000000 target",
        "memory_events": events,
        "free_bytes": free,
        "project_device": project_device,
        "research_tmp_device": scratch_device,
        "research_tmp_fstype": result.stdout.strip(),
    }


def text_bytes(value: object) -> bytes:
    """Preserve subprocess output bytes for base64 durable receipts."""
    if isinstance(value, bytes):
        return value
    return str(value or "").encode("utf-8", errors="replace")


def main() -> None:
    """Run fresh local-to-B gates, stage fixed bytes, and submit one CPU verifier."""
    path_exception = require_path_exception()
    require(
        B.is_dir() and not B.is_symlink() and B.resolve(strict=True) == B,
        "project root is unavailable or redirected",
    )
    require(
        (B / "research_tmp").is_dir() and not (B / "research_tmp").is_symlink(),
        "research_tmp is not a physical directory",
    )
    require(
        CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
        "canonical project Python is unavailable",
    )
    require(
        D.parent == B / "research_tmp" and not os.path.lexists(D),
        "create-only verifier output already exists",
    )
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found", "support unit already exists"
    )
    payload = payload_bytes()
    expected_names = set(EXPECTED_HASHES)
    require(set(payload) == expected_names, "fixed verifier input set differs")
    admission_raw, _ = stable_file(EVIDENCE / "ADMISSION.json", 10_000)
    admission = json_object(admission_raw, "EMA admission")
    physical_code_root(SOURCE)
    source = check_source(payload, admission)
    split = check_split(payload, admission)
    registered = check_registered_data(admission)
    native, terminal, run_stats = check_native(payload)
    predecessors = check_predecessors(admission)
    cache = check_cache(payload, admission, split)
    parent = check_parent()
    staged_files = {
        name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in payload.items()
    }
    limits = {
        "cpu_quota": "50%",
        "memory_high_bytes": 1_400_000_000,
        "memory_max_bytes": SUPPORT_MAX_BYTES,
        "swap_max_bytes": 0,
        "runtime_seconds": 600,
        "cuda_visible_devices": "",
        "blas_threads": 1,
    }

    D.mkdir(mode=0o700, exist_ok=False)
    for name, raw in payload.items():
        write_exclusive(D / name, raw)
    verifier_admission = {
        "schema": "fg19.child-ema-terminal-verifier-admission.v1",
        "created_utc": utc_now(),
        "status": "FRESH_GATES_PASSED_SUPPORT_SUBMISSION_PENDING",
        "run_name": RUN_NAME,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "terminal_evidence_sha256": TERMINAL_SHA,
        "terminal_raw_sha256": NATIVE_RAW_SHA,
        "source": source,
        "split": split,
        "registered_data_stat_only": registered,
        "terminal_run_artifacts_stat_only": run_stats,
        "native_unit_snapshot": native,
        "typed_terminal_evidence": terminal,
        "predecessors_read_only": predecessors,
        "cache_stat_only": cache,
        "parent": parent,
        "staged_files": staged_files,
        "resource_limits": limits,
        "approved_path_compatibility_exception": {
            "environment_variable": PATH_EXCEPTION_ENV,
            "value": path_exception,
            "global": "pathlib._local.PosixPath",
            "scope": "training_state.pt only; one exact dictionary-cache path constructor; weights_only=True",
        },
        "scope": "Five pinned code files, exact source manifest and bounded terminal/acceptance metadata only; no source restage, model/state load, cache-array/dictionary payload hash or native stop.",
        "unknown_outcome_policy": "Inspect this exact unit and durable output; never retry or recreate this output directory.",
    }
    write_json_exclusive(D / "VERIFIER_ADMISSION.json", verifier_admission)
    command = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=50%",
        "--property=MemoryHigh=1400000000",
        "--property=MemoryMax=1500000000",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(D / "unit.log"),
        "--property=StandardError=append:" + str(D / "unit.log"),
        "--working-directory=" + str(D),
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "WNN_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TMPDIR": str(B / "research_tmp"),
        "PYTHONPATH": str(D) + ":" + str(SOURCE / "src"),
        "WNN_PROJECT_ROOT": str(DATA),
        PATH_EXCEPTION_ENV: path_exception,
    }
    command.extend("--setenv=" + key + "=" + value for key, value in environment.items())
    command.extend(
        [
            str(CANONICAL_PYTHON),
            "-E",
            "-B",
            "-u",
            str(D / "child_ema_terminal_check_v1.py"),
            "--run-dir",
            str(RUN),
            "--source-root",
            str(SOURCE),
            "--source-manifest",
            str(D / "SOURCE_MANIFEST.json"),
            "--split",
            str(SPLIT),
            "--evidence-root",
            str(EVIDENCE),
            "--profile-evidence-root",
            str(PROFILE_EVIDENCE),
            "--admission",
            str(EVIDENCE / "ADMISSION.json"),
            "--launch",
            str(EVIDENCE / "LAUNCH.json"),
            "--launch-acceptance",
            str(D / "LAUNCH_ACCEPTANCE.json"),
            "--terminal-evidence",
            str(D / "TERMINAL_EVIDENCE.json"),
            "--cache-root",
            str(CACHE),
            "--cache-acceptance",
            str(CACHE_RECEIPTS / "TERMINAL_ACCEPTANCE.json"),
            "--cache-exact-stop",
            str(CACHE_RECEIPTS / "EXACT_STOP.json"),
            "--output",
            str(D / "VERIFICATION.json"),
        ]
    )
    write_json_exclusive(
        D / "VERIFIER_SUBMISSION_INTENT.json",
        {
            "created_utc": utc_now(),
            "unit": UNIT,
            "command": command,
            "unknown_outcome_policy": "Never retry; inspect this exact unit and durable output.",
        },
    )
    submission: dict[str, object] = {"outcome": "unknown"}
    try:
        result = subprocess.run(command, capture_output=True, timeout=25, check=False)
        submission = {
            "outcome": "submitted" if result.returncode == 0 else "unknown_or_failed",
            "returncode": result.returncode,
            "stdout_b64": base64.b64encode(result.stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(result.stderr).decode("ascii"),
        }
    except subprocess.TimeoutExpired as error:
        submission = {
            "outcome": "submit_timeout_unknown_never_retry",
            "returncode": None,
            "stdout_b64": base64.b64encode(text_bytes(error.stdout)).decode("ascii"),
            "stderr_b64": base64.b64encode(text_bytes(error.stderr)).decode("ascii"),
        }
    except OSError as error:
        submission = {
            "outcome": "submit_error_unknown_never_retry",
            "returncode": None,
            "error": repr(error),
            "stdout_b64": "",
            "stderr_b64": "",
        }
    post_fields = (
        "LoadState",
        "InvocationID",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "ControlGroup",
        "Result",
    )
    try:
        post = {"outcome": "captured", "properties": show(UNIT, post_fields)}
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        post = {"outcome": "query_failed_unknown", "error": repr(error)}
    support_properties = post.get("properties", {}) if isinstance(post, dict) else {}
    support_invocation = (
        support_properties.get("InvocationID") if isinstance(support_properties, dict) else None
    )
    launch_record = {
        "schema": "fg19.child-ema-terminal-verifier-launch.v1",
        "created_utc": utc_now(),
        "unit": UNIT,
        "invocation_id": support_invocation,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Durably retain this outcome; inspect exact unit and output; never retry or create another output directory.",
    }
    write_json_exclusive(D / "VERIFIER_LAUNCH.json", launch_record)
    response = {
        "unit": UNIT,
        "outcome": submission["outcome"],
        "admission_base64": base64.b64encode((D / "VERIFIER_ADMISSION.json").read_bytes()).decode(
            "ascii"
        ),
        "launch_base64": base64.b64encode((D / "VERIFIER_LAUNCH.json").read_bytes()).decode(
            "ascii"
        ),
    }
    print(json.dumps(response, sort_keys=True))


main()
