"""One-shot create-only CPU verifier admission for the already-run NLL fit."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import time
from datetime import datetime
from pathlib import Path

PAYLOAD = {}
FINAL_REVIEW_BINDING = {}
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA = B / "data"
RUN = DATA / "artifacts/runs/fg19_child_nll_f0_s1001_fit_v1"
SOURCE = B / "source-fg19-cache-68ff6829"
SPLIT = B / "splits/fold0.json"
CACHE = B / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"
DICT = DATA / "artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
PARENT_CHECKPOINT = DATA / "artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt"
REGISTRY = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
EVIDENCE = B / "research_tmp/fg19_child_nll_fit_v1"
PROFILE_EVIDENCE = B / "research_tmp/fg19_child_allthree_profile_v1"
CACHE_RECEIPTS = B / "research/full_gru_tweaks_r1/cache_cpu_v2"
D = B / "research_tmp/fg19_child_nll_verify_v1"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
UNIT = "computer-vision-homework-fg19-child-nll-verify-v1.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
PATH_EXCEPTION_ENV = "FG19_APPROVED_PATH_COMPAT_EXCEPTION"
PATH_EXCEPTION_VALUE = "pathlib._local.PosixPath:exact-dictionary-path-only"
INVOCATION = "c833c613fb92426981ab255a5faf4452"
RUN_NAME = "fg19_child_nll_f0_s1001_fit_v1"
CONTROL_RUN = "fg19_child_control_f0_s1001_fit_v1"
CONTROL_UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
CONTROL_INVOCATION = "e66e0d0703314e4fb5d47dcdbfcaf202"
EMA_RUN = "fg19_child_ema_f0_s1001_fit_v1"
EMA_UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
EMA_INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
EMA_MODEL_EP5_SHA = "6e90bc4dba8add0caf670097519f08f3ce239d0b64c37c47b3760aa22af41862"
DICTIONARY_CHILD_RUN = "fg19_child_dictionary_f0_s1001_fit_v1"
DICTIONARY_CHILD_UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
DICTIONARY_CHILD_INVOCATION = "73b10b699e034a1b9952b2377b845095"
DICTIONARY_CHILD_ACCEPTANCE_SHA = "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1"
DICTIONARY_CHILD_STOP_SHA = "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3"
DICTIONARY_CHILD_MODEL_EP5_SHA = "b840d9c81d46405f2d2439b94ad6ef15ff8501bb14715ec79792d980efbf7151"
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
ADMISSION_SHA = "54b97021a829d1352e9b48a7a6a64fa97a5d1d69c81883bd6518c514858a356c"
LAUNCH_SHA = "2b313f148559b12379f5c98ffd8e94805c5d1f8c33de18145ee4d3fa6efc7738"
LAUNCH_ACCEPTANCE_SHA = "44a708fc5a32b101c97a8a28c22e3b15e815059582de564e17f528f5369dc7a5"
FIT_BODY_SHA = "893e06d9f17ee75450251bb3f8918312041ca6f147cf1cd30312335889843f15"
FIT_BODY_BYTES = 227_746
FIT_SOURCE_REVIEW_SHA = "7354aeb1c1980a11f0aa67ecac03e1f2ecca8e853052ac778e349fcf84fae081"
FIT_SOURCE_REVIEW_BYTES = 14_660
FIT_ROOT_PREPARATION_SHA = "50eef2910b740d5481a28893dcf3ae5c231c81f106149eab2a643d5345485138"
FIT_ROOT_PREPARATION_BYTES = 2_622
FIT_STDOUT_SHA = "90c7fcd2a916ffd6a1fa63ee9a7192f6cbc439f989bb7ade25435ecd1902efbb"
FIT_STDOUT_BYTES = 27_426
FIT_TRANSPORT_SHA = "5864d2ecacb9d8287002e061598560e71c65a73d49054eadc81c3842e101f7a3"
FIT_TRANSPORT_BYTES = 1_213
ROOT_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_nll_terminal_check_v1/ROOT_PREPARATION.json"
TERMINAL_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/TERMINAL_EVIDENCE.json"
NATIVE_RAW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
TERMINAL_SHA = "__RENDER_TERMINAL_SHA256__"
NATIVE_RAW_SHA = "__RENDER_NATIVE_RAW_SHA256__"
NATIVE_RAW_BYTES = 0
CHECKER_SHA = "__RENDER_CHECKER_SHA256__"
CHECKER_BYTES = 0
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
EMA_ACCEPTANCE_SHA = "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8"
EMA_STOP_SHA = "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_EXACT_STOP_SHA = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
CONTROL_CHECKER_SHA = "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167"
PARENT_CHECKER_SHA = "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41"
SUPPORT_SHA = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
TEACHER_SHA = "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5"
EMA_CHECKER_SHA = "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d"
MAX_SOURCE_BYTES = 128_000_000
MAX_JSON_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_ARTIFACT_BYTES = 128_000_000
SUPPORT_MAX_BYTES = 1_500_000_000
PARENT_TARGET_BYTES = 17_000_000_000
PARENT_RESERVE_BYTES = 1_000_000_000
EXPECTED_HASHES = {
    "child_nll_terminal_check_v1.py": CHECKER_SHA,
    "child_control_terminal_check_v3.py": CONTROL_CHECKER_SHA,
    "parent_fit_terminal_check_v3.py": PARENT_CHECKER_SHA,
    "propensity_terminal_support.py": SUPPORT_SHA,
    "teacher_s5_terminal_check_v1.py": TEACHER_SHA,
    "child_ema_terminal_check_v3.py": EMA_CHECKER_SHA,
    "SOURCE_MANIFEST.json": MANIFEST_SHA,
    "LAUNCH_ACCEPTANCE.json": LAUNCH_ACCEPTANCE_SHA,
    "NLL_LAUNCH_STDOUT.raw.json": FIT_STDOUT_SHA,
    "NLL_TRANSPORT.json": FIT_TRANSPORT_SHA,
    "TERMINAL_EVIDENCE.json": TERMINAL_SHA,
    "NATIVE_STDOUT.raw.json": NATIVE_RAW_SHA,
}
MAX_INPUT_BYTES = {
    "child_nll_terminal_check_v1.py": 80_000,
    "child_control_terminal_check_v3.py": 80_000,
    "parent_fit_terminal_check_v3.py": 70_000,
    "propensity_terminal_support.py": 15_000,
    "teacher_s5_terminal_check_v1.py": 40_000,
    "child_ema_terminal_check_v3.py": 40_000,
    "SOURCE_MANIFEST.json": 30_000,
    "LAUNCH_ACCEPTANCE.json": 6_000,
    "NLL_LAUNCH_STDOUT.raw.json": 40_000,
    "NLL_TRANSPORT.json": 5_000,
    "TERMINAL_EVIDENCE.json": 16_000,
    "NATIVE_STDOUT.raw.json": 2_000_000,
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
    "EMA_TERMINAL_ACCEPTANCE.json": EMA_ACCEPTANCE_SHA,
    "EMA_EXACT_STOP.json": EMA_STOP_SHA,
    "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": DICTIONARY_CHILD_ACCEPTANCE_SHA,
    "DICTIONARY_CHILD_EXACT_STOP.json": DICTIONARY_CHILD_STOP_SHA,
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
    """Recheck all twelve accepted predecessor receipts and stats read-only."""
    staged = admission.get("staged_receipt_sha256")
    require(
        isinstance(staged, dict)
        and {name: staged.get(name) for name in READONLY_RECEIPTS} == READONLY_RECEIPTS,
        "admitted predecessor receipt map differs",
    )
    receipts = {
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
        "profile metadata pins differ",
    )
    parent_identity = admission.get("parent_checkpoint_identity")
    dictionary_identity = admission.get("dictionary")
    require(
        isinstance(parent_identity, dict) and isinstance(dictionary_identity, dict),
        "accepted parent/dictionary stat metadata is missing",
    )
    parent_path = Path(parent_identity.get("path", ""))
    dictionary_path = Path(dictionary_identity.get("path", ""))
    require(
        parent_path == PARENT_CHECKPOINT and dictionary_path == DICT,
        "accepted parent/dictionary paths differ",
    )
    require(
        admission.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA
        and parent_identity.get("sha256") == PARENT_CHECKPOINT_SHA,
        "accepted parent checkpoint digest metadata differs",
    )
    require(
        dictionary_identity.get("sha256") == DICTIONARY_SHA, "accepted dictionary digest differs"
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
        "receipts_read_only": receipts,
        "profile_files_read_only": {
            "ADMISSION.json": profile_admission_stat,
            "LAUNCH.json": profile_launch_stat,
            "unit.log": profile_log_stat,
        },
        "parent_checkpoint_stat_only": parent_observed,
        "dictionary_artifact_stat_only": dictionary_observed,
        "payload_bytes_hashed": False,
        "ancestor_units_polled": False,
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


def text_bytes(value: object) -> bytes:
    """Preserve subprocess output bytes for base64 durable receipts."""
    if isinstance(value, bytes):
        return value
    return str(value or "").encode("utf-8", errors="replace")


def check_fit_launch(payload: dict[str, bytes]) -> tuple[bytes, bytes, dict[str, object]]:
    """Bind the original NLL launch receipts, response, transport, and recipe."""
    admission_raw, _ = stable_file(EVIDENCE / "ADMISSION.json", 20_000)
    launch_raw, _ = stable_file(EVIDENCE / "LAUNCH.json", 20_000)
    require(digest(admission_raw) == ADMISSION_SHA, "original NLL admission digest differs")
    require(digest(launch_raw) == LAUNCH_SHA, "original NLL launch digest differs")
    admission = json_object(admission_raw, "original NLL admission")
    launch = json_object(launch_raw, "original NLL launch")
    acceptance = json_object(payload["LAUNCH_ACCEPTANCE.json"], "original NLL launch acceptance")
    require(
        (
            admission.get("schema"),
            admission.get("run_name"),
            admission.get("unit"),
            admission.get("source_commit"),
            admission.get("source_files_verified"),
            admission.get("source_manifest_sha256"),
            admission.get("split_sha256"),
            admission.get("populated"),
        )
        == (
            "fg19-child-nll-fit-admission.v1",
            RUN_NAME,
            NATIVE_UNIT,
            COMMIT,
            167,
            MANIFEST_SHA,
            SPLIT_SHA,
            {},
        ),
        "original NLL admission identity differs",
    )
    recipe = admission.get("recipe")
    expected_recipe = {
        "batch_size": 16,
        "checkpoint_epoch_zero_based": 5,
        "chunk_rows": 2_000,
        "epochs": 6,
        "evaluation_during_fit": False,
        "expected_optimizer_updates": 44_760,
        "expected_updates_are_arithmetic_not_observed": True,
        "fit_groups": 11_940,
        "held_out_evaluation_groups_excluded": 468,
        "inherited_parent_preprocessing": True,
        "objective": "nll",
        "representation_mode": "base",
        "seed": 1_001,
        "trainer_argv_sha256": "51905d5ecd65117ec0b0c8ab7a4f8aa4f24795015ac29a4af57dd9da781c4b90",
    }
    require(recipe == expected_recipe, "original NLL recipe metadata differs")
    require(
        admission.get("staged_receipt_sha256") == READONLY_RECEIPTS,
        "original NLL admission predecessor receipt map differs",
    )
    require(
        admission.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA
        and admission.get("parent_acceptance_sha256") == PARENT_ACCEPTANCE_SHA
        and admission.get("parent_exact_stop_sha256") == PARENT_STOP_SHA,
        "original parent initializer metadata differs",
    )
    submission = launch.get("submission")
    require(
        (
            launch.get("schema"),
            launch.get("unit"),
            launch.get("control_predecessor"),
            launch.get("ema_predecessor"),
            launch.get("dictionary_child_fit_predecessor"),
            isinstance(submission, dict),
            submission.get("outcome") if isinstance(submission, dict) else None,
            submission.get("returncode") if isinstance(submission, dict) else None,
            launch.get("unknown_outcome_policy"),
        )
        == (
            "fg19-child-nll-fit-launch.v1",
            NATIVE_UNIT,
            admission.get("control_predecessor"),
            admission.get("ema_predecessor"),
            admission.get("dictionary_child_fit_predecessor"),
            True,
            "submitted",
            0,
            "never retry; inspect this exact unit and durable NLL output",
        ),
        "original NLL launch identity/outcome differs",
    )
    raw_stdout = payload["NLL_LAUNCH_STDOUT.raw.json"]
    transport = json_object(payload["NLL_TRANSPORT.json"], "original NLL transport")
    raw_response = json_object(raw_stdout, "original NLL raw response")
    require(
        set(raw_response) == {"ADMISSION.json", "LAUNCH.json"}
        and base64.b64decode(raw_response["ADMISSION.json"], validate=True) == admission_raw
        and base64.b64decode(raw_response["LAUNCH.json"], validate=True) == launch_raw,
        "original NLL response does not contain the exact admission and launch bytes",
    )
    expected_refs = {
        "admission": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/ADMISSION.json",
            "bytes": 11_750,
            "sha256": ADMISSION_SHA,
        },
        "launch": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/LAUNCH.json",
            "bytes": 8_787,
            "sha256": LAUNCH_SHA,
        },
        "body": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/REMOTE_SUBMISSION.py",
            "bytes": FIT_BODY_BYTES,
            "sha256": FIT_BODY_SHA,
        },
        "source_review": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_nll_fit_v1/FINAL_BODY_SOURCE_REVIEW_V2.json",
            "bytes": FIT_SOURCE_REVIEW_BYTES,
            "sha256": FIT_SOURCE_REVIEW_SHA,
        },
        "root_preparation": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/ROOT_ADMISSION_PREPARATION.json",
            "bytes": FIT_ROOT_PREPARATION_BYTES,
            "sha256": FIT_ROOT_PREPARATION_SHA,
        },
        "dictionary_child_acceptance": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/TERMINAL_ACCEPTANCE.json",
            "bytes": 14_914,
            "sha256": DICTIONARY_CHILD_ACCEPTANCE_SHA,
        },
        "dictionary_child_exact_stop": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/exact_stop/EXACT_STOP.json",
            "bytes": 3_113,
            "sha256": DICTIONARY_CHILD_STOP_SHA,
        },
        "raw_stdout": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/STDOUT.raw.json",
            "bytes": FIT_STDOUT_BYTES,
            "sha256": FIT_STDOUT_SHA,
        },
        "transport": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/TRANSPORT.json",
            "bytes": FIT_TRANSPORT_BYTES,
            "sha256": FIT_TRANSPORT_SHA,
        },
    }
    require(
        digest(raw_stdout) == FIT_STDOUT_SHA
        and len(raw_stdout) == FIT_STDOUT_BYTES
        and digest(payload["NLL_TRANSPORT.json"]) == FIT_TRANSPORT_SHA
        and len(payload["NLL_TRANSPORT.json"]) == FIT_TRANSPORT_BYTES,
        "original NLL raw transport pins differ",
    )
    require(
        (
            transport.get("schema"),
            transport.get("ssh_returncode"),
            transport.get("timed_out"),
            transport.get("raw_persisted_before_parse"),
            transport.get("retry_performed"),
            transport.get("unknown_outcome"),
            transport.get("body"),
            transport.get("source_review"),
            transport.get("stdout"),
        )
        == (
            "fg19.raw-ssh-transport.v1",
            0,
            False,
            True,
            False,
            False,
            expected_refs["body"],
            expected_refs["source_review"],
            expected_refs["raw_stdout"],
        ),
        "original NLL transport outcome/authority differs",
    )
    require(
        (
            acceptance.get("schema"),
            acceptance.get("status"),
            acceptance.get("run_name"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            {key: acceptance.get(key) for key in expected_refs},
            acceptance.get("expected_updates_not_observed"),
            acceptance.get("scientific_fits_started"),
            acceptance.get("quality_score"),
            acceptance.get("unknown_outcome_policy"),
        )
        == (
            "fg19.child-nll-fit-launch-acceptance.v1",
            "ACTUAL_ONE_GPU_LAUNCH_BOUND_NO_COMPLETION_OR_QUALITY",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            expected_refs,
            44_760,
            1,
            None,
            "Inspect exact unit/durable receipts; never retry/restart/resubmit.",
        ),
        "original NLL root launch acceptance differs",
    )
    return admission_raw, launch_raw, acceptance


def check_native(
    payload: dict[str, bytes], admission_raw: bytes, launch_raw: bytes
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Join strict native exit proof to its raw-first capture and current exact unit."""
    terminal_raw = payload["TERMINAL_EVIDENCE.json"]
    capture_raw = payload["NATIVE_STDOUT.raw.json"]
    terminal = json_object(terminal_raw, "typed NLL native terminal evidence")
    capture = json_object(capture_raw, "raw-first NLL native capture")
    require(
        (
            terminal.get("schema"),
            terminal.get("run_name"),
            terminal.get("unit"),
            terminal.get("invocation_id"),
            terminal.get("active_state"),
            terminal.get("sub_state"),
            terminal.get("raw_path"),
            terminal.get("raw_bytes"),
            terminal.get("raw_sha256"),
        )
        == (
            "fg19.child-nll-fit-native-terminal-evidence.v1",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            "active",
            "exited",
            NATIVE_RAW_PATH,
            NATIVE_RAW_BYTES,
            NATIVE_RAW_SHA,
        ),
        "typed NLL terminal identity/raw binding differs",
    )
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        require(
            type(terminal.get(key)) is int and terminal[key] == expected,
            f"typed native {key} differs",
        )
    require(
        len(capture_raw) == NATIVE_RAW_BYTES and digest(capture_raw) == NATIVE_RAW_SHA,
        "native raw capture exact bytes differ",
    )
    require(
        (
            capture.get("schema"),
            capture.get("expected_run"),
            capture.get("expected_unit"),
            capture.get("expected_invocation_id"),
            capture.get("captured_utc"),
        )
        == (
            "fg19-child-nll-startup-capture.v1",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            terminal.get("captured_utc"),
        ),
        "raw NLL capture identity differs",
    )
    props = capture.get("unit_properties")
    require(isinstance(props, dict), "raw NLL capture unit properties are missing")
    require(
        tuple(
            props.get(key)
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
        "raw NLL capture exit properties differ",
    )
    stamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        require(isinstance(value, str), f"typed NLL {key} is missing")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise RuntimeError(f"typed NLL {key} is malformed") from error
        require(parsed.tzinfo is not None, f"typed NLL {key} has no timezone")
        stamps[key] = parsed
    require(
        stamps["native_exit_utc"] <= stamps["captured_utc"],
        "typed NLL exit timestamp is after the raw capture",
    )
    require(
        captured_receipt(capture, "ADMISSION.json") == admission_raw
        and captured_receipt(capture, "LAUNCH.json") == launch_raw,
        "raw NLL capture original admission/launch bytes differ",
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
        "fresh NLL invocation is not the exact exited native unit",
    )
    require(
        current.get("CPUUsageNSec", "").isdigit() and int(current["CPUUsageNSec"]) > 0,
        "native NLL CPU usage evidence is missing",
    )
    raw_log = captured_receipt(capture, "unit.log")
    log, log_stat = stable_file(EVIDENCE / "unit.log", MAX_JSON_BYTES)
    require(log == raw_log, "NLL native unit log differs from captured bytes")
    artifacts = check_run_stats(capture)
    return (
        current,
        {
            "typed": terminal,
            "raw_capture_sha256": NATIVE_RAW_SHA,
            "raw_capture_bytes": len(capture_raw),
            "unit_log": {**log_stat, "sha256": digest(log)},
        },
        artifacts,
    )


def check_parent() -> dict[str, object]:
    """Enforce empty campaign descendants, one-time support headroom, and physical scratch."""
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
        "parent cgroup is unavailable or redirected",
    )
    proc_files = [cgroup / "cgroup.procs", *cgroup.rglob("cgroup.procs")]
    require(
        0 < len(proc_files) < 128, "parent populated-descendant inventory is NA or over its bound"
    )
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
    scratch = B / "research_tmp"
    require(
        scratch.is_dir() and not scratch.is_symlink() and scratch.resolve(strict=True) == scratch,
        "research_tmp is not a direct physical directory",
    )
    free = os.statvfs(B).f_bavail * os.statvfs(B).f_frsize
    require(free >= 25 * 1024**3, "project filesystem free space is below 25 GiB")
    project_device = os.stat(B).st_dev
    scratch_device = os.stat(scratch).st_dev
    require(
        project_device == scratch_device == 64_512, "project/research_tmp device identity differs"
    )
    result = subprocess.run(
        ["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        result.returncode == 0 and result.stdout.strip() == "ext4",
        "research_tmp is not on the expected ext4 filesystem",
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


def main() -> None:
    """Run fresh gates, stage fixed metadata create-only, and submit one CPU verifier."""
    require(
        FINAL_REVIEW_BINDING.get("schema") == "fg19.child-nll-terminal-verifier-render-binding.v1"
        and FINAL_REVIEW_BINDING.get("status")
        == "ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND",
        "final NLL checker review/root preparation/native exit binding is missing",
    )
    checker_pin = FINAL_REVIEW_BINDING.get("checker")
    terminal_pin = FINAL_REVIEW_BINDING.get("native_terminal")
    source_review_pin = FINAL_REVIEW_BINDING.get("source_review")
    root_preparation_pin = FINAL_REVIEW_BINDING.get("root_preparation")
    require(
        isinstance(checker_pin, dict)
        and (checker_pin.get("bytes"), checker_pin.get("sha256")) == (CHECKER_BYTES, CHECKER_SHA),
        "final NLL checker pin differs from the exact embedded code guard",
    )
    require(
        isinstance(terminal_pin, dict)
        and (
            terminal_pin.get("typed_sha256"),
            terminal_pin.get("raw_bytes"),
            terminal_pin.get("raw_sha256"),
        )
        == (TERMINAL_SHA, NATIVE_RAW_BYTES, NATIVE_RAW_SHA),
        "final NLL native proof pin differs from exact embedded bytes",
    )
    require(
        isinstance(source_review_pin, dict)
        and source_review_pin.get("status") == "PASS_SOURCE_ONLY"
        and isinstance(source_review_pin.get("sha256"), str)
        and len(source_review_pin["sha256"]) == 64
        and isinstance(root_preparation_pin, dict)
        and root_preparation_pin.get("status")
        == "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED"
        and isinstance(root_preparation_pin.get("sha256"), str)
        and len(root_preparation_pin["sha256"]) == 64,
        "final NLL checker source/root pins are missing",
    )
    path_exception = require_path_exception()
    require(
        B.is_dir() and not B.is_symlink() and B.resolve(strict=True) == B,
        "project root is redirected",
    )
    scratch = B / "research_tmp"
    require(
        scratch.is_dir() and not scratch.is_symlink() and scratch.resolve(strict=True) == scratch,
        "research_tmp is not a direct physical directory",
    )
    require(
        CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
        "canonical project Python is unavailable",
    )
    require(
        D.parent == scratch and not os.path.lexists(D),
        "create-only NLL verifier output already exists",
    )
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found",
        "NLL support unit already exists",
    )
    payload = payload_bytes()
    require(set(payload) == set(EXPECTED_HASHES), "fixed NLL verifier payload names differ")
    admission_raw, launch_raw, acceptance = check_fit_launch(payload)
    admission = json_object(admission_raw, "original NLL admission")
    physical_code_root(SOURCE)
    source = check_source(payload, admission)
    split = check_split(payload, admission)
    registered = check_registered_data(admission)
    native, terminal, run_stats = check_native(payload, admission_raw, launch_raw)
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
        "schema": "fg19.child-nll-terminal-verifier-admission.v1",
        "created_utc": utc_now(),
        "status": "FRESH_GATES_PASSED_SUPPORT_SUBMISSION_PENDING",
        "run_name": RUN_NAME,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "terminal_evidence_sha256": TERMINAL_SHA,
        "terminal_raw_sha256": NATIVE_RAW_SHA,
        "original_fit_launch_acceptance_sha256": LAUNCH_ACCEPTANCE_SHA,
        "source": source,
        "split": split,
        "registered_data_stat_only": registered,
        "terminal_run_artifacts_stat_only": run_stats,
        "native_unit_snapshot": native,
        "typed_terminal_evidence": terminal,
        "predecessors_read_only": predecessors,
        "cache_stat_only": cache,
        "parent": parent,
        "original_fit_launch_acceptance": acceptance,
        "staged_files": staged_files,
        "rendered_body_authority": {
            "schema": "fg19.root-source-review-raw-transport-authority.v1",
            "owner": "root",
            "source_review": source_review_pin,
            "source_review_scope": "checker bytes only; exact rendered V1 stdin body remains a root prelaunch gate",
            "root_preparation": root_preparation_pin,
            "raw_transport": {
                "owner": "root",
                "method": "exact V1 body bytes supplied on stdin",
                "external_binding_required": "root binds raw transport bytes to the reviewed exact V1 body",
                "worker_self_read": False,
                "worker_body_sha256_claimed": False,
            },
        },
        "final_local_review_binding": FINAL_REVIEW_BINDING,
        "resource_limits": limits,
        "approved_path_compatibility_exception": {
            "environment_variable": PATH_EXCEPTION_ENV,
            "value": path_exception,
            "global": "pathlib._local.PosixPath",
            "scope": "training_state.pt only; one exact dictionary-cache path constructor; weights_only=True",
        },
        "scope": "Six prehashed checker/helper files and bounded source/native/launch metadata only; all twelve predecessor receipts and profile files stay read-only; no source restage, native stop, model/state load, or cache-array/dictionary payload hashing.",
        "unknown_outcome_policy": "Inspect this exact support unit and durable output; never retry or recreate the output directory.",
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
        "TMPDIR": str(scratch),
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
            str(D / "child_nll_terminal_check_v1.py"),
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
        "schema": "fg19.child-nll-terminal-verifier-launch.v1",
        "created_utc": utc_now(),
        "unit": UNIT,
        "invocation_id": support_invocation,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Durably retain this result; inspect exact unit/output; never retry or create another output directory.",
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


if __name__ == "__main__":
    main()
