"""One-shot, capped CPU verifier admission for the completed AllThree fit."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from datetime import datetime
from pathlib import Path

PAYLOAD = {}
FIXED_PAYLOAD_PINS = {}
FINAL_REVIEW_BINDING = {}
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA = B / "data"
RUN = DATA / "artifacts/runs/fg19_child_allthree_f0_s1001_fit_v1"
SOURCE = B / "source-fg19-cache-68ff6829"
SPLIT = B / "splits/fold0.json"
CACHE = B / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"
DICT = DATA / "artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
PARENT_CHECKPOINT = DATA / "artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt"
REGISTRY = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
EVIDENCE = B / "research_tmp/fg19_child_allthree_fit_v1"
PROFILE_EVIDENCE = B / "research_tmp/fg19_child_allthree_profile_v1"
CACHE_RECEIPTS = B / "research/full_gru_tweaks_r1/cache_cpu_v2"
D = B / "research_tmp/fg19_child_allthree_verify_v1"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
UNIT = "computer-vision-homework-fg19-child-allthree-verify-v1.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
PATH_EXCEPTION_ENV = "FG19_APPROVED_PATH_COMPAT_EXCEPTION"
PATH_EXCEPTION_VALUE = "pathlib._local.PosixPath:exact-dictionary-path-only"
INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
RUN_NAME = "fg19_child_allthree_f0_s1001_fit_v1"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
ADMISSION_SHA = "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"
ADMISSION_BYTES = 12_912
LAUNCH_SHA = "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538"
LAUNCH_BYTES = 9_594
LAUNCH_ACCEPTANCE_SHA = "2afbba374ea2e332766368087490660198110027784924c69d03a5656c924346"
LAUNCH_ACCEPTANCE_BYTES = 2_933
FIT_STDOUT_SHA = "98345f64e9a8ec41978f8679cd631095b9bbb5936be8895d4bda574927855e73"
FIT_STDOUT_BYTES = 30_050
FIT_TRANSPORT_SHA = "1c30ca323cb5899c8269c24f277e6b73a33305c4855c8a3167490785038e5a57"
FIT_TRANSPORT_BYTES = 546
TERMINAL_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/"
    "TERMINAL_EVIDENCE.json"
)
NATIVE_RAW_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/"
    "terminal_native_capture_v1/STDOUT.raw.json"
)
NATIVE_SCHEMA = "fg19.child-allthree-fit-native-terminal-evidence.v1"
RAW_SCHEMA = "fg19-child-allthree-startup-capture.v1"
ROOT_BINDING_SCHEMA = "fg19.child-allthree-terminal-verifier-render-binding.v1"
ROOT_BINDING_STATUS = "ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND"
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
PARENT_ACCEPTANCE_SHA = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
PARENT_STOP_SHA = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
PARENT_CHECKPOINT_SHA = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
DICTIONARY_SHA = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_STOP_SHA = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
PROFILE_ADMISSION_SHA = "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5"
PROFILE_LAUNCH_SHA = "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44"
PROFILE_LOG_SHA = "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289"
MAX_SOURCE_BYTES = 128_000_000
MAX_METADATA_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_ARTIFACT_BYTES = 128_000_000
SUPPORT_MAX_BYTES = 1_500_000_000
PARENT_TARGET_BYTES = 17_000_000_000
PARENT_RESERVE_BYTES = 1_000_000_000
SELECTED_GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE_SHA,
    "PARENT_EXACT_STOP.json": PARENT_STOP_SHA,
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
MAX_INPUT_BYTES = {
    "child_allthree_terminal_check_v1.py": 100_000,
    "child_nll_terminal_check_v1.py": 80_000,
    "child_control_terminal_check_v3.py": 80_000,
    "child_ema_terminal_check_v3.py": 50_000,
    "parent_fit_terminal_check_v3.py": 70_000,
    "propensity_terminal_support.py": 15_000,
    "teacher_s5_terminal_check_v1.py": 40_000,
    "SOURCE_MANIFEST.json": 30_000,
    "ADMISSION.json": 20_000,
    "LAUNCH.json": 20_000,
    "LAUNCH_ACCEPTANCE.json": 6_000,
    "ALLTHREE_LAUNCH_STDOUT.raw.json": 40_000,
    "TRANSPORT.json": 5_000,
    "TERMINAL_EVIDENCE.json": 16_000,
    "NATIVE_STDOUT.raw.json": 2_000_000,
}


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


def payload_bytes() -> dict[str, bytes]:
    """Prehash every private code dependency and bounded metadata payload."""
    require(set(PAYLOAD) == set(MAX_INPUT_BYTES), "fixed payload names differ")
    require(set(FIXED_PAYLOAD_PINS) == set(PAYLOAD), "fixed payload pin names differ")
    payload: dict[str, bytes] = {}
    for name, item in PAYLOAD.items():
        require(isinstance(item, dict), f"malformed embedded payload record: {name}")
        raw = base64.b64decode(item.get("base64", ""), validate=True)
        pin = FIXED_PAYLOAD_PINS[name]
        require(type(item.get("bytes")) is int, f"invalid payload byte count: {name}")
        require(type(pin.get("bytes")) is int, f"invalid fixed byte count: {name}")
        require(
            len(raw) == item["bytes"] == pin["bytes"] <= MAX_INPUT_BYTES[name],
            f"payload size differs: {name}",
        )
        require(
            digest(raw) == item.get("sha256") == pin.get("sha256"), f"payload SHA differs: {name}"
        )
        payload[name] = raw
    return payload


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
        while block := os.read(descriptor, 1 << 20):
            hasher.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
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


def check_render_binding(payload: dict[str, bytes]) -> dict[str, object]:
    """Require root's exact source review, checker pins, and native-exit binding."""
    require(
        FINAL_REVIEW_BINDING.get("schema") == ROOT_BINDING_SCHEMA
        and FINAL_REVIEW_BINDING.get("status") == ROOT_BINDING_STATUS,
        "final AllThree root/source/native binding is missing",
    )
    checker = FINAL_REVIEW_BINDING.get("checker")
    terminal = FINAL_REVIEW_BINDING.get("native_terminal")
    review = FINAL_REVIEW_BINDING.get("source_review")
    preparation = FINAL_REVIEW_BINDING.get("root_preparation")
    checker_raw = payload["child_allthree_terminal_check_v1.py"]
    typed_raw = payload["TERMINAL_EVIDENCE.json"]
    native_raw = payload["NATIVE_STDOUT.raw.json"]
    require(
        isinstance(checker, dict)
        and (checker.get("bytes"), checker.get("sha256"))
        == (len(checker_raw), digest(checker_raw)),
        "root checker binding differs from its embedded bytes",
    )
    require(
        isinstance(terminal, dict)
        and (
            terminal.get("typed_bytes"),
            terminal.get("typed_sha256"),
            terminal.get("raw_bytes"),
            terminal.get("raw_sha256"),
        )
        == (len(typed_raw), digest(typed_raw), len(native_raw), digest(native_raw)),
        "root native terminal binding differs from embedded bytes",
    )
    require(
        isinstance(review, dict)
        and review.get("status") == "PASS_SOURCE_ONLY"
        and isinstance(review.get("sha256"), str)
        and len(review["sha256"]) == 64,
        "root source review binding is missing",
    )
    require(
        isinstance(preparation, dict)
        and preparation.get("status")
        == "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED"
        and isinstance(preparation.get("sha256"), str)
        and len(preparation["sha256"]) == 64,
        "root preparation binding is missing",
    )
    return {
        "checker": checker,
        "source_review": review,
        "root_preparation": preparation,
        "native_terminal": terminal,
    }


def check_native(payload: dict[str, bytes]) -> tuple[dict[str, object], dict[str, object]]:
    """Join strict typed exit metadata to the exact raw-first capture."""
    terminal = json_object(payload["TERMINAL_EVIDENCE.json"], "typed AllThree native terminal")
    capture_raw = payload["NATIVE_STDOUT.raw.json"]
    capture = json_object(capture_raw, "raw-first AllThree native capture")
    require(
        (
            terminal.get("schema"),
            terminal.get("run_name"),
            terminal.get("unit"),
            terminal.get("invocation_id"),
            terminal.get("active_state"),
            terminal.get("sub_state"),
        )
        == (NATIVE_SCHEMA, RUN_NAME, NATIVE_UNIT, INVOCATION, "active", "exited"),
        "typed AllThree native identity/state differs",
    )
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        require(
            type(terminal.get(key)) is int and terminal[key] == expected,
            f"typed AllThree {key} is not strict integer {expected}",
        )
    raw_sha = digest(capture_raw)
    require(
        type(terminal.get("raw_bytes")) is int and terminal["raw_bytes"] == len(capture_raw) > 0,
        "typed AllThree raw byte count differs",
    )
    require(
        terminal.get("raw_sha256") == raw_sha and terminal.get("raw_path") == NATIVE_RAW_PATH,
        "typed AllThree raw binding differs",
    )
    require(
        (
            capture.get("schema"),
            capture.get("expected_run"),
            capture.get("expected_unit"),
            capture.get("expected_invocation_id"),
            capture.get("captured_utc"),
        )
        == (RAW_SCHEMA, RUN_NAME, NATIVE_UNIT, INVOCATION, terminal.get("captured_utc")),
        "raw AllThree capture identity differs",
    )
    properties = capture.get("unit_properties")
    require(
        isinstance(properties, dict)
        and tuple(
            properties.get(key)
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
        "raw AllThree unit properties differ",
    )
    stamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        require(isinstance(value, str), f"typed AllThree {key} is missing")
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise RuntimeError(f"typed AllThree {key} is malformed") from error
        require(stamp.tzinfo is not None, f"typed AllThree {key} has no timezone")
        stamps[key] = stamp
    require(
        stamps["native_exit_utc"] <= stamps["captured_utc"],
        "AllThree exit is later than its raw capture",
    )
    return terminal, capture


def check_source(payload: dict[str, bytes], admission: dict[str, object]) -> dict[str, object]:
    """Freshly hash the accepted 167-file source tree without modifying it."""
    manifest_raw = payload["SOURCE_MANIFEST.json"]
    require(digest(manifest_raw) == MANIFEST_SHA, "embedded source manifest digest differs")
    manifest = json_object(manifest_raw, "source manifest")
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
    require(
        SOURCE.is_dir() and not SOURCE.is_symlink() and SOURCE.resolve(strict=True) == SOURCE,
        "source root is unavailable or redirected",
    )
    names: set[str] = set()
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
            and str(relative) not in names,
            "invalid source manifest hash",
        )
        path = SOURCE / relative
        ancestor = path
        while ancestor != SOURCE:
            require(not ancestor.is_symlink(), f"source path contains symlink: {ancestor}")
            ancestor = ancestor.parent
        actual_sha, info = file_hash(path, size)
        require(
            (actual_sha, info["bytes"]) == (expected_sha, size),
            f"source file pin differs: {relative}",
        )
        names.add(str(relative))
        total += size
        require(total <= MAX_SOURCE_BYTES, "source tree exceeds bounded code budget")
    inventory: set[str] = set()
    for root, dirs, files in os.walk(SOURCE, followlinks=False):
        for name in dirs + files:
            path = Path(root) / name
            require(not path.is_symlink(), "source tree contains a symlink")
        inventory.update(str((Path(root) / name).relative_to(SOURCE)) for name in files)
    require(inventory == names, "source tree inventory differs from the frozen manifest")
    return {
        "commit": COMMIT,
        "file_count": len(names),
        "bytes": total,
        "manifest_sha256": MANIFEST_SHA,
    }


def check_fit_receipts(
    payload: dict[str, bytes],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Bind the original AllThree admission, launch, response, and recipe metadata."""
    expected = {
        "ADMISSION.json": (ADMISSION_BYTES, ADMISSION_SHA),
        "LAUNCH.json": (LAUNCH_BYTES, LAUNCH_SHA),
        "LAUNCH_ACCEPTANCE.json": (LAUNCH_ACCEPTANCE_BYTES, LAUNCH_ACCEPTANCE_SHA),
        "ALLTHREE_LAUNCH_STDOUT.raw.json": (FIT_STDOUT_BYTES, FIT_STDOUT_SHA),
        "TRANSPORT.json": (FIT_TRANSPORT_BYTES, FIT_TRANSPORT_SHA),
    }
    for name, (size, sha) in expected.items():
        raw, _ = stable_file(EVIDENCE / name, MAX_INPUT_BYTES[name])
        require(
            (len(raw), digest(raw)) == (size, sha) and raw == payload[name],
            f"original AllThree receipt changed: {name}",
        )
    admission = json_object(payload["ADMISSION.json"], "original AllThree admission")
    launch = json_object(payload["LAUNCH.json"], "original AllThree launch")
    acceptance = json_object(
        payload["LAUNCH_ACCEPTANCE.json"], "original AllThree launch acceptance"
    )
    recipe = admission.get("recipe")
    require(
        (admission.get("schema"), admission.get("run_name"), admission.get("unit"))
        == ("fg19-child-allthree-fit-admission.v1", RUN_NAME, NATIVE_UNIT),
        "original AllThree admission identity differs",
    )
    require(
        isinstance(recipe, dict)
        and (
            recipe.get("epochs"),
            recipe.get("checkpoint_epoch_zero_based"),
            recipe.get("expected_optimizer_updates"),
            recipe.get("objective"),
            recipe.get("representation_mode"),
            recipe.get("seed"),
            recipe.get("inherited_parent_preprocessing"),
        )
        == (6, 5, 44_760, "nll", "ema_sparse", 1001, True),
        "original AllThree recipe differs",
    )
    require(
        (
            launch.get("schema"),
            launch.get("unit"),
            launch.get("submission", {}).get("outcome"),
            launch.get("submission", {}).get("returncode"),
            launch.get("unknown_outcome_policy"),
        )
        == (
            "fg19-child-allthree-fit-launch.v1",
            NATIVE_UNIT,
            "submitted",
            0,
            "never retry; inspect this exact unit and durable NLL output",
        ),
        "original AllThree launch differs",
    )
    require(
        (
            acceptance.get("schema"),
            acceptance.get("status"),
            acceptance.get("run_name"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            acceptance.get("expected_updates_not_observed"),
            acceptance.get("scientific_fits_started"),
            acceptance.get("quality_score"),
        )
        == (
            "fg19-child-allthree-fit-launch-acceptance.v1",
            "ACTUAL_ONE_LAUNCH_ACCEPTED_NOT_TERMINAL_OR_QUALITY",
            RUN_NAME,
            NATIVE_UNIT,
            INVOCATION,
            44_760,
            1,
            None,
        ),
        "original AllThree root acceptance differs",
    )
    require(
        admission.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA
        and admission.get("parent_acceptance_sha256") == PARENT_ACCEPTANCE_SHA
        and admission.get("parent_exact_stop_sha256") == PARENT_STOP_SHA,
        "AllThree parent lineage differs",
    )
    return admission, launch, acceptance


def check_predecessors(admission: dict[str, object]) -> dict[str, object]:
    """Verify the exact fourteen accepted fit receipts and profile metadata read-only."""
    staged = admission.get("staged_receipt_sha256")
    require(
        isinstance(staged, dict)
        and {name: staged.get(name) for name in READONLY_RECEIPTS} == READONLY_RECEIPTS
        and set(staged) == set(READONLY_RECEIPTS),
        "AllThree predecessor receipt map differs",
    )
    receipts: dict[str, object] = {}
    for name, sha in READONLY_RECEIPTS.items():
        raw, info = stable_file(EVIDENCE / name, 1_000_000)
        require(digest(raw) == sha, f"accepted predecessor receipt differs: {name}")
        receipts[name] = {**info, "sha256": sha, "read_only": True}
    profile_files = {
        "ADMISSION.json": PROFILE_ADMISSION_SHA,
        "LAUNCH.json": PROFILE_LAUNCH_SHA,
        "unit.log": PROFILE_LOG_SHA,
    }
    profile: dict[str, object] = {}
    for name, sha in profile_files.items():
        raw, info = stable_file(PROFILE_EVIDENCE / name, 1_000_000)
        require(digest(raw) == sha, f"AllThree profile metadata differs: {name}")
        profile[name] = {**info, "sha256": sha, "read_only": True}
    parent_identity = admission.get("parent_checkpoint_identity")
    dictionary_identity = admission.get("dictionary")
    require(
        isinstance(parent_identity, dict) and isinstance(dictionary_identity, dict),
        "parent/dictionary stat metadata missing",
    )
    require(
        Path(parent_identity.get("path", "")) == PARENT_CHECKPOINT
        and Path(dictionary_identity.get("path", "")) == DICT,
        "parent or dictionary path differs",
    )
    for path, identity, sha, label in (
        (PARENT_CHECKPOINT, parent_identity, PARENT_CHECKPOINT_SHA, "parent"),
        (DICT, dictionary_identity, DICTIONARY_SHA, "dictionary"),
    ):
        observed = path.stat()
        require(
            identity.get("sha256") == sha
            and (identity.get("bytes"), identity.get("device"), identity.get("inode"))
            == (observed.st_size, observed.st_dev, observed.st_ino),
            f"{label} artifact stat or metadata differs",
        )
    return {
        "receipts_read_only": receipts,
        "profile_files_read_only": profile,
        "parent_checkpoint_stat_only": True,
        "dictionary_artifact_stat_only": True,
        "payload_bytes_read_or_hashed": False,
    }


def check_split(admission: dict[str, object]) -> dict[str, object]:
    """Recheck the canonical split bytes without touching training payloads."""
    raw, info = stable_file(SPLIT, 2_000_000)
    require(
        digest(raw) == SPLIT_SHA == admission.get("split_sha256"), "canonical fold split differs"
    )
    json_object(raw, "canonical split")
    return {**info, "sha256": SPLIT_SHA}


def check_registered_data(admission: dict[str, object]) -> dict[str, object]:
    """Recheck six registered data paths by identity and stat only."""
    raw, registry_stat = stable_file(REGISTRY, 2_000_000)
    require(digest(raw) == REGISTRY_SHA, "portable source registry digest differs")
    registry = json_object(raw, "portable source registry")
    sources, expected = registry.get("sources"), admission.get("data_identity_stats_only")
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
        target = path.stat()
        actual = {
            "path": str(path),
            "resolved": str(path.resolve(strict=True)),
            "islink": path.is_symlink(),
            "device": target.st_dev,
            "bytes": target.st_size,
            "inode": target.st_ino,
            "mtime_ns": target.st_mtime_ns,
            "registered_sha256": sources[relative].get("sha256"),
        }
        require(
            actual == {key: pin.get(key) for key in actual},
            f"registered path/stat differs: {relative}",
        )
        require(
            sources[relative].get("sha256") == pin.get("registered_sha256")
            and target.st_size == sources[relative].get("original_identity", {}).get("bytes"),
            f"registered source differs: {relative}",
        )
        observed[relative] = actual
    return {"registry": {**registry_stat, "sha256": REGISTRY_SHA}, "paths_stat_only": observed}


def check_cache(admission: dict[str, object], split: dict[str, object]) -> dict[str, object]:
    """Check accepted cache metadata and array identities without reading arrays."""
    cache = admission.get("cache")
    require(isinstance(cache, dict), "cache identity is missing")
    acceptance_raw, _ = stable_file(CACHE_RECEIPTS / "TERMINAL_ACCEPTANCE.json", 10_000)
    stop_raw, _ = stable_file(CACHE_RECEIPTS / "EXACT_STOP.json", 10_000)
    require(
        digest(acceptance_raw) == CACHE_ACCEPTANCE_SHA == cache.get("acceptance_sha256")
        and digest(stop_raw) == CACHE_STOP_SHA == cache.get("exact_stop_sha256"),
        "cache receipts differ",
    )
    metadata, metadata_stat = stable_file(CACHE / "metadata.json", MAX_METADATA_BYTES)
    require(digest(metadata) == CACHE_METADATA_SHA, "cache metadata differs")
    require(
        json_object(metadata, "cache metadata").get("split_manifest_sha256") == split["sha256"],
        "cache split identity differs",
    )
    acceptance = json_object(acceptance_raw, "cache acceptance")
    stats, outputs = cache.get("array_stats_only"), acceptance.get("output")
    require(isinstance(stats, dict) and isinstance(outputs, dict), "cache array receipts missing")
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
        current = {
            "bytes": st.st_size,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
        }
        accepted, expected = outputs.get(name), stats.get(name)
        require(
            isinstance(expected, dict) and isinstance(accepted, dict) and current == expected,
            f"cache {name} stat differs",
        )
        require(
            (
                st.st_size,
                accepted.get("sha256"),
                accepted.get("bytes"),
                accepted.get("shape"),
                accepted.get("dtype"),
            )
            == (size, sha, size, [count, 20_000, 2], "float32"),
            f"cache {name} receipt differs",
        )
        arrays[name] = {**current, "payload_read_or_hashed": False}
    return {
        "metadata_sha256": CACHE_METADATA_SHA,
        "metadata": metadata_stat,
        "acceptance_sha256": CACHE_ACCEPTANCE_SHA,
        "exact_stop_sha256": CACHE_STOP_SHA,
        "arrays_stat_only": arrays,
        "array_payloads_read_or_hashed": False,
    }


def check_run_stats(capture: dict[str, object]) -> dict[str, object]:
    """Compare terminal run outputs by stat only against the raw native capture."""
    directory = RUN.lstat()
    require(
        stat.S_ISDIR(directory.st_mode) and not RUN.is_symlink(),
        "terminal run directory is not direct",
    )
    inventory = capture.get("run_directory_stat_only")
    require(
        isinstance(inventory, dict)
        and inventory.get("exists") is True
        and inventory.get("symlink") is False,
        "captured run directory identity is unsafe",
    )
    entries = inventory.get("entries")
    expected = {
        "model.pt",
        *(f"model_ep{i}.pt" for i in range(6)),
        "training_state.pt",
        "provenance.json",
        "history.json",
    }
    require(isinstance(entries, list), "raw capture omits run artifact inventory")
    by_name = {row.get("name"): row for row in entries if isinstance(row, dict)}
    require(
        len(by_name) == len(entries) and set(by_name) == expected, "run artifact inventory differs"
    )
    rows: dict[str, object] = {}
    total = 0
    for name in sorted(expected):
        path = RUN / name
        current = path.lstat()
        limit = (
            MAX_STATE_BYTES
            if name == "training_state.pt"
            else MAX_MODEL_BYTES
            if name.startswith("model")
            else MAX_METADATA_BYTES
        )
        require(
            stat.S_ISREG(current.st_mode) and not path.is_symlink() and current.st_size <= limit,
            f"unsafe/oversized terminal artifact: {name}",
        )
        actual = {
            "name": name,
            "bytes": current.st_size,
            "device": current.st_dev,
            "inode": current.st_ino,
            "mtime_ns": current.st_mtime_ns,
            "mode": current.st_mode,
            "is_symlink": False,
        }
        require(actual == by_name[name], f"terminal artifact stat differs from raw capture: {name}")
        total += current.st_size
        rows[name] = {key: actual[key] for key in ("bytes", "device", "inode", "mtime_ns")}
    require(total <= MAX_ARTIFACT_BYTES, "terminal artifacts exceed aggregate cap")
    return {"files": rows, "aggregate_bytes": total, "payload_bytes_read": 0}


def check_parent() -> dict[str, object]:
    """Require the two-core aggregate envelope, an empty campaign cgroup, and physical scratch."""
    fields = ("ControlGroup", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax")
    parent = show("wnn-campaign.slice", fields)
    require(
        (
            parent.get("CPUQuotaPerSecUSec"),
            parent.get("MemoryHigh"),
            parent.get("MemoryMax"),
            parent.get("MemorySwapMax"),
        )
        == ("2s", "18000000000", "20000000000", "0"),
        "aggregate resource envelope differs",
    )
    control_group = parent.get("ControlGroup", "")
    require(
        control_group not in ("", "/") and control_group.endswith("/wnn-campaign.slice"),
        "parent cgroup is unscoped",
    )
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    require(
        cgroup.is_dir() and cgroup.resolve(strict=True) == cgroup,
        "campaign cgroup is unavailable or redirected",
    )
    proc_files = [cgroup / "cgroup.procs", *cgroup.rglob("cgroup.procs")]
    require(
        0 < len(proc_files) < 128, "campaign populated-descendant inventory is outside its bound"
    )
    populated = {
        str(path): path.read_text(encoding="ascii")
        for path in proc_files
        if path.read_text(encoding="ascii").strip()
    }
    require(not populated, "campaign cgroup still has populated descendants")
    current_raw = (cgroup / "memory.current").read_text(encoding="ascii").strip()
    require(current_raw.isdigit(), "campaign memory current is unavailable")
    current = int(current_raw)
    require(
        current + SUPPORT_MAX_BYTES + PARENT_RESERVE_BYTES < PARENT_TARGET_BYTES,
        "support memory headroom is insufficient",
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
    project_device, scratch_device = os.stat(B).st_dev, os.stat(scratch).st_dev
    require(
        project_device == scratch_device == 64_512, "project and scratch filesystem devices differ"
    )
    result = subprocess.run(
        ["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        result.returncode == 0 and result.stdout.strip() == "ext4", "research_tmp is not on ext4"
    )
    return {
        "systemd": parent,
        "cgroup": str(cgroup),
        "populated": populated,
        "memory_current_bytes": current,
        "headroom_formula": "current + 1500000000 support max + 1000000000 reserve < 17000000000 target",
        "memory_events": events,
        "free_bytes": free,
        "research_tmp_fstype": "ext4",
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


def _gpu_process_owner(pid: str, gpu_uuid: str, used_memory: str) -> dict[str, object]:
    """Observe one selected-GPU process and preserve unrelated owners."""
    proc = Path("/proc") / pid
    try:
        before = proc.stat(follow_symlinks=False)
        require(stat.S_ISDIR(before.st_mode), "selected-GPU PID is not a proc directory")
        start_before = _proc_starttime(proc, pid)
        cgroup = _bounded_proc_read(proc / "cgroup", 32768).decode("ascii", errors="strict").strip()
        command_has_project_path = False
        if before.st_uid == os.getuid():
            command = _bounded_proc_read(proc / "cmdline", 8192).decode("utf-8", errors="replace")
            command_has_project_path = str(B) in command
        after = proc.stat(follow_symlinks=False)
        start_after = _proc_starttime(proc, pid)
    except (FileNotFoundError, ProcessLookupError) as error:
        raise RuntimeError(f"selected-GPU PID identity became unavailable: {pid}") from error
    identity_before = (before.st_ino, before.st_uid, start_before)
    identity_after = (after.st_ino, after.st_uid, start_after)
    require(
        identity_before == identity_after,
        f"selected-GPU PID identity changed during observation: {pid}",
    )
    if "wnn-campaign.slice" in cgroup or command_has_project_path:
        raise RuntimeError(f"selected GPU is in use by a campaign-owned process: {pid}")
    return {
        "pid": int(pid),
        "gpu_uuid": gpu_uuid,
        "used_memory": used_memory,
        "uid": before.st_uid,
        "cgroup": cgroup,
        "proc_starttime": start_before,
        "preserved": True,
    }


def check_selected_gpu_ownership() -> dict[str, object]:
    """Require the selected physical GPU and snapshot owners before output mutation."""
    prefix = ["nvidia-smi", "-i", SELECTED_GPU_UUID]
    gpu_output, _ = _bounded_gpu_query(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    )
    gpu_rows = [line.strip() for line in gpu_output.splitlines() if line.strip()]
    require(len(gpu_rows) == 1, "selected-GPU identity query did not return one row")
    columns = [value.strip() for value in gpu_rows[0].split(",")]
    require(
        len(columns) == 4 and columns[0] == SELECTED_GPU_UUID and columns[1],
        "selected-GPU UUID/name differs",
    )
    free_bytes = _memory_bytes(columns[2])
    total_bytes = _memory_bytes(columns[3])
    require(
        total_bytes > 0 and free_bytes <= total_bytes,
        "selected-GPU memory observation is inconsistent",
    )

    process_output, _ = _bounded_gpu_query(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"]
    )
    rows = [line.strip() for line in process_output.splitlines() if line.strip()]
    if len(rows) == 1 and rows[0].startswith("No running processes found"):
        rows = []
    require(len(rows) <= 256, "selected-GPU process inventory exceeded 256 rows")
    owners: list[dict[str, object]] = []
    seen_pids: set[str] = set()
    for row in rows:
        fields = [value.strip() for value in row.split(",")]
        require(
            len(fields) == 3
            and fields[0].isascii()
            and fields[0].isdigit()
            and int(fields[0]) > 0
            and str(int(fields[0])) == fields[0]
            and fields[1] == SELECTED_GPU_UUID
            and fields[2],
            "selected-GPU process row is malformed or names another GPU",
        )
        require(
            fields[0] not in seen_pids, "selected-GPU process inventory contains a duplicate PID"
        )
        seen_pids.add(fields[0])
        owners.append(_gpu_process_owner(fields[0], fields[1], fields[2]))
    return {
        "selected_gpu_uuid": SELECTED_GPU_UUID,
        "gpu_name": columns[1],
        "memory_free_bytes": free_bytes,
        "memory_total_bytes": total_bytes,
        "minimum_free_memory_required_bytes": 0,
        "process_rows": len(rows),
        "process_owners": owners,
        "unrelated_owners_preserved": True,
        "foreign_process_action": "none",
    }


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


def make_launch_record(submission: dict[str, object], post: dict[str, object]) -> dict[str, object]:
    """Bind one submission to the exact support-unit invocation observed afterward."""
    properties = post.get("properties", {})
    invocation_id = properties.get("InvocationID") if isinstance(properties, dict) else None
    return {
        "schema": "fg19.child-allthree-terminal-verifier-launch.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unit": UNIT,
        "invocation_id": invocation_id,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Durably retain this result; inspect exact unit/output; never retry or create another output directory.",
    }


def captured_receipt(capture: dict[str, object], name: str) -> bytes:
    """Decode one exact receipt captured in the native raw bundle."""
    receipts = capture.get("receipts")
    require(
        isinstance(receipts, dict) and isinstance(receipts.get(name), dict),
        f"raw capture omits {name}",
    )
    record = receipts[name]
    require(
        record.get("present") is True and record.get("regular_file") is True,
        f"raw capture marks {name} unsafe",
    )
    raw = base64.b64decode(record.get("base64", ""), validate=True)
    require(
        type(record.get("bytes")) is int
        and len(raw) == record["bytes"]
        and digest(raw) == record.get("sha256"),
        f"raw capture bytes differ: {name}",
    )
    return raw


def check_staged_private_closure() -> dict[str, dict[str, object]]:
    """Prehash all seven staged code files before the support interpreter imports them."""
    require(
        not any(path.name == "__pycache__" for path in D.rglob("__pycache__")),
        "fresh verifier output contains bytecode cache",
    )
    observed: dict[str, dict[str, object]] = {}
    for name in MAX_INPUT_BYTES:
        if not name.endswith(".py"):
            continue
        raw, info = stable_file(D / name, MAX_INPUT_BYTES[name])
        pin = FIXED_PAYLOAD_PINS[name]
        require(
            (len(raw), digest(raw)) == (pin["bytes"], pin["sha256"]),
            f"staged private source differs before import: {name}",
        )
        observed[name] = {**info, "sha256": digest(raw), "preimport": True}
    require(len(observed) == 7, "staged private code closure is not exactly seven files")
    return observed


def preflight(payload: dict[str, bytes]) -> dict[str, object]:
    """Run fresh source, receipt, native, path, resource, and GPU gates before D.mkdir."""
    binding = check_render_binding(payload)
    require(
        os.environ.get(PATH_EXCEPTION_ENV) == PATH_EXCEPTION_VALUE,
        f"{PATH_EXCEPTION_ENV} is not the exact approved value",
    )
    require(
        B.is_dir() and not B.is_symlink() and B.resolve(strict=True) == B,
        "project root is redirected",
    )
    scratch = B / "research_tmp"
    require(
        scratch.is_dir() and not scratch.is_symlink() and scratch.resolve(strict=True) == scratch,
        "physical scratch is unavailable or redirected",
    )
    require(
        CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
        "canonical project Python is unavailable",
    )
    require(
        D.parent == scratch and not os.path.lexists(D), "AllThree support output already exists"
    )
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found",
        "AllThree support unit already exists",
    )
    admission, launch, acceptance = check_fit_receipts(payload)
    source = check_source(payload, admission)
    split = check_split(admission)
    registered = check_registered_data(admission)
    terminal, capture = check_native(payload)
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
        "fresh AllThree unit is not the exact exited invocation",
    )
    require(
        current.get("CPUUsageNSec", "").isdigit() and int(current["CPUUsageNSec"]) > 0,
        "native AllThree CPU usage evidence is missing",
    )
    raw_log = captured_receipt(capture, "unit.log")
    live_log, log_stat = stable_file(EVIDENCE / "unit.log", MAX_METADATA_BYTES)
    require(live_log == raw_log, "current AllThree unit log differs from the raw capture")
    run_stats = check_run_stats(capture)
    predecessors = check_predecessors(admission)
    cache = check_cache(admission, split)
    parent = check_parent()
    gpu = check_selected_gpu_ownership()
    return {
        "binding": binding,
        "source": source,
        "split": split,
        "registered_data_stat_only": registered,
        "native_unit_snapshot": current,
        "native_terminal": terminal,
        "unit_log": {**log_stat, "sha256": digest(live_log)},
        "terminal_run_artifacts_stat_only": run_stats,
        "predecessors_read_only": predecessors,
        "cache_stat_only": cache,
        "parent": parent,
        "selected_gpu_ownership": gpu,
        "fit_launch_acceptance": acceptance,
    }


def main() -> None:
    """Stage exact reviewed metadata and submit one bounded CPU checker."""
    payload = payload_bytes()
    gates = preflight(payload)
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
    preimport_closure = check_staged_private_closure()
    staged = {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in payload.items()}
    admission = {
        "schema": "fg19.child-allthree-terminal-verifier-admission.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FRESH_GATES_PASSED_SUPPORT_SUBMISSION_PENDING",
        "run_name": RUN_NAME,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION,
        "terminal_evidence_sha256": digest(payload["TERMINAL_EVIDENCE.json"]),
        "terminal_raw_sha256": digest(payload["NATIVE_STDOUT.raw.json"]),
        "gates": gates,
        "staged_files": staged,
        "private_code_preimport_hashes": preimport_closure,
        "resource_limits": limits,
        "approved_path_compatibility_exception": {
            "environment_variable": PATH_EXCEPTION_ENV,
            "value": PATH_EXCEPTION_VALUE,
            "global": "pathlib._local.PosixPath",
            "scope": "training_state.pt only; exact dictionary-cache path constructor; weights_only=True",
        },
        "scope": "Seven prehashed private code files and bounded source/native/launch metadata only; fourteen predecessor receipts and profile metadata remain read-only; no source restage, native stop, model/state load outside the exact restricted decoder, cache-array/dictionary payload hashing, inference, or quality scoring.",
        "unknown_outcome_policy": "Inspect this exact support unit and durable output; never retry or recreate the output directory.",
    }
    write_json_exclusive(D / "VERIFIER_ADMISSION.json", admission)
    closure_pins = {
        name: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for name, item in preimport_closure.items()
    }
    runner = (
        "import hashlib, json, os, stat, sys\n"
        "from pathlib import Path\n"
        "root = Path.cwd()\n"
        "if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:\n"
        "    raise RuntimeError('verifier working directory is redirected')\n"
        "if any(path.name == '__pycache__' for path in root.rglob('__pycache__')):\n"
        "    raise RuntimeError('unreviewed bytecode cache exists before checker import')\n"
        "if any(path.suffix == '.pyc' for path in root.rglob('*') if path.is_file()):\n"
        "    raise RuntimeError('unreviewed bytecode file exists before checker import')\n"
        "pins = " + repr(closure_pins) + "\n"
        "def verify_source(name, pin):\n"
        "    path = root / name\n"
        "    before = path.lstat()\n"
        "    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size != pin['bytes']:\n"
        "        raise RuntimeError('unsafe private source: ' + name)\n"
        "    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0))\n"
        "    try:\n"
        "        opened = os.fstat(fd)\n"
        "        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):\n"
        "            raise RuntimeError('private source changed before read: ' + name)\n"
        "        raw = bytearray()\n"
        "        while len(raw) <= pin['bytes']:\n"
        "            block = os.read(fd, min(65536, pin['bytes'] + 1 - len(raw)))\n"
        "            if not block:\n"
        "                break\n"
        "            raw.extend(block)\n"
        "        after = os.fstat(fd)\n"
        "    finally:\n"
        "        os.close(fd)\n"
        "    if len(raw) != pin['bytes'] or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) or hashlib.sha256(raw).hexdigest() != pin['sha256']:\n"
        "        raise RuntimeError('private source pin differs: ' + name)\n"
        "    return path.resolve(strict=True)\n"
        "origins = {name: verify_source(name, pin) for name, pin in pins.items()}\n"
        "module_names = [name[:-3] for name in pins]\n"
        "if any(name in sys.modules for name in module_names):\n"
        "    raise RuntimeError('private module was loaded before source prehash')\n"
        "sys.path.insert(0, str(root))\n"
        "sys.dont_write_bytecode = True\n"
        "sys.pycache_prefix = None\n"
        "import child_allthree_terminal_check_v1 as checker\n"
        "if Path(checker.__file__).resolve(strict=True) != origins['child_allthree_terminal_check_v1.py']:\n"
        "    raise RuntimeError('AllThree checker imported from an unexpected source path')\n"
        "closure = checker.assert_helper_closure()\n"
        "expected_helpers = {name: pin for name, pin in pins.items() if name != 'child_allthree_terminal_check_v1.py'}\n"
        "if closure != expected_helpers:\n"
        "    raise RuntimeError('post-import helper closure differs from pre-import pins')\n"
        "for name, source_path in origins.items():\n"
        "    module = sys.modules.get(name[:-3])\n"
        "    if module is None or not getattr(module, '__file__', None) or Path(module.__file__).resolve(strict=True) != source_path:\n"
        "        raise RuntimeError('private import origin differs from prehash: ' + name)\n"
        "if any(path.name == '__pycache__' for path in root.rglob('__pycache__')):\n"
        "    raise RuntimeError('bytecode cache appeared during checker import')\n"
        "if any(path.suffix == '.pyc' for path in root.rglob('*') if path.is_file()):\n"
        "    raise RuntimeError('bytecode file appeared during checker import')\n"
        "for name, pin in pins.items():\n"
        "    verify_source(name, pin)\n"
        "result = checker.verify_terminal(*tuple(Path(value) for value in "
        + repr(
            (
                str(RUN),
                str(SOURCE),
                str(D / "SOURCE_MANIFEST.json"),
                str(SPLIT),
                str(EVIDENCE),
                str(PROFILE_EVIDENCE),
                str(D / "ADMISSION.json"),
                str(D / "LAUNCH.json"),
                str(D / "LAUNCH_ACCEPTANCE.json"),
                str(D / "TERMINAL_EVIDENCE.json"),
                str(CACHE),
                str(CACHE_RECEIPTS / "TERMINAL_ACCEPTANCE.json"),
                str(CACHE_RECEIPTS / "EXACT_STOP.json"),
            )
        )
        + "))\n"
        "print(json.dumps(result, sort_keys=True, allow_nan=False))\n"
    )
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
        "--setenv=CUDA_VISIBLE_DEVICES=",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        str(CANONICAL_PYTHON),
        "-E",
        "-B",
        "-I",
        "-c",
        runner,
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
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
            "stdout_b64": base64.b64encode(error.stdout or b"").decode("ascii"),
            "stderr_b64": base64.b64encode(error.stderr or b"").decode("ascii"),
        }
    except OSError as error:
        submission = {
            "outcome": "submit_error_unknown_never_retry",
            "returncode": None,
            "error": repr(error),
            "stdout_b64": "",
            "stderr_b64": "",
        }
    submission["argv"] = command
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
    launch = make_launch_record(submission, post)
    write_json_exclusive(D / "VERIFIER_LAUNCH.json", launch)
    support_invocation = launch.get("invocation_id")
    if (
        not isinstance(support_invocation, str)
        or len(support_invocation) != 32
        or any(character not in "0123456789abcdef" for character in support_invocation)
    ):
        raise RuntimeError(
            "support unit invocation ID was not captured; inspect durable launch record and never retry"
        )
    if submission.get("returncode") != 0:
        raise RuntimeError(
            "bounded AllThree verifier submission returned nonzero or unknown; inspect its exact durable output, never retry"
        )


if __name__ == "__main__":
    main()
