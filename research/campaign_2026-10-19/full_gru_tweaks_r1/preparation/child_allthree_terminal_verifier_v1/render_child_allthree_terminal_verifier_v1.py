"""Render the AllThree support body only after its strict native exit proof exists."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records"
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY = OUTPUT / "REMOTE_SUBMISSION.py"
ROOT_PREPARATION = PREPARATION / "child_allthree_terminal_check_v1/ROOT_PREPARATION.json"
TERMINAL = RECORDS / "child_allthree_fit_v1/TERMINAL_EVIDENCE.json"
NATIVE_RAW = RECORDS / "child_allthree_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
SOURCE_MANIFEST = PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
CHECKER = PREPARATION / "child_allthree_terminal_check_v1/child_allthree_terminal_check_v1.py"
STATIC_REVIEW = OUTPUT / "CONDITIONAL_SOURCE_REVIEW_V1.json"
IMPORT_BOUNDARY_PROOF = (
    PREPARATION / "child_allthree_terminal_check_v1/PRIVATE_IMPORT_BOUNDARY_PROOF_V1.json"
)
IMPORT_BOUNDARY_HARNESS = (
    PREPARATION
    / "child_allthree_terminal_check_v1/private_import_boundary_proof_allthree_terminal_check_v1.py"
)
IMPORT_BOUNDARY_SCOPE = (
    PREPARATION / "child_allthree_terminal_check_v1/PRIVATE_IMPORT_BOUNDARY_SCOPE_V1.md"
)
NATIVE_SCHEMA = "fg19.child-allthree-fit-native-terminal-evidence.v1"
RAW_SCHEMA = "fg19-child-allthree-startup-capture.v1"
ROOT_SCHEMA = "fg19.child-allthree-terminal-root-preparation.v1"
ROOT_STATUS = "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED"
RUN = "fg19_child_allthree_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
MAX_METADATA_BYTES = 1_000_000
MAX_NATIVE_RAW_BYTES = 2_000_000
TEMPLATE_BASE_SHA = "119e5a7ed17701c7b8b41d4d7436268d6879856a58dcfe81bc23ae84e1f93f32"

CODE_INPUTS = {
    "child_allthree_terminal_check_v1.py": CHECKER,
    "child_nll_terminal_check_v1.py": PREPARATION
    / "child_nll_terminal_check_v1/child_nll_terminal_check_v1.py",
    "child_control_terminal_check_v3.py": PREPARATION
    / "child_control_terminal_verifier_v3/child_control_terminal_check_v3.py",
    "child_ema_terminal_check_v3.py": PREPARATION
    / "child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py",
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
}
CODE_PINS = {
    "child_allthree_terminal_check_v1.py": (
        40_442,
        "f4e9b7a2ed6798edf7ac0fbd30517687606494f6aacc44dddc96a0bba1f544dd",
    ),
    "child_nll_terminal_check_v1.py": (
        40_474,
        "a3cf3b10515bb16275561a0ab9338227c00661347afbdf081888ec258dd1f7a2",
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
    ),
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
    ),
    "parent_fit_terminal_check_v3.py": (
        64_550,
        "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    ),
    "propensity_terminal_support.py": (
        13_107,
        "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    ),
    "teacher_s5_terminal_check_v1.py": (
        37_448,
        "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    ),
}
METADATA_INPUTS = {
    "SOURCE_MANIFEST.json": SOURCE_MANIFEST,
    "ADMISSION.json": RECORDS / "child_allthree_fit_v1/ADMISSION.json",
    "LAUNCH.json": RECORDS / "child_allthree_fit_v1/LAUNCH.json",
    "LAUNCH_ACCEPTANCE.json": RECORDS / "child_allthree_fit_v1/LAUNCH_ACCEPTANCE.json",
    "ALLTHREE_LAUNCH_STDOUT.raw.json": RECORDS / "child_allthree_fit_v1/STDOUT.raw.json",
    "TRANSPORT.json": RECORDS / "child_allthree_fit_v1/TRANSPORT.json",
    "TERMINAL_EVIDENCE.json": TERMINAL,
    "NATIVE_STDOUT.raw.json": NATIVE_RAW,
}
EXPECTED_CODE_PATHS = {
    name: str(path.relative_to(CAMPAIGN)).replace("\\", "/") for name, path in CODE_INPUTS.items()
}
FIXED_METADATA = {
    "SOURCE_MANIFEST.json": (27_569, MANIFEST_SHA),
    "ADMISSION.json": (12_912, "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"),
    "LAUNCH.json": (9_594, "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538"),
    "LAUNCH_ACCEPTANCE.json": (
        2_933,
        "2afbba374ea2e332766368087490660198110027784924c69d03a5656c924346",
    ),
    "ALLTHREE_LAUNCH_STDOUT.raw.json": (
        30_050,
        "98345f64e9a8ec41978f8679cd631095b9bbb5936be8895d4bda574927855e73",
    ),
    "TRANSPORT.json": (546, "1c30ca323cb5899c8269c24f277e6b73a33305c4855c8a3167490785038e5a57"),
}
PREDECESSOR_SHAS = {
    "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
    "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
    "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
    "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
    "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
    "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
    "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8",
    "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf",
    "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1",
    "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3",
    "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5",
    "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1",
}
READONLY_RECEIPT_SHAS = {
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
FIXED_METADATA_PATHS = {
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json": (
        27_569,
        MANIFEST_SHA,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/ADMISSION.json": (
        12_912,
        "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/LAUNCH.json": (
        9_594,
        "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/REMOTE_SUBMISSION.py": (
        263_545,
        "c2961ef594953c5885a948bd3433e9deeadd376010b60064d9a556b842ab655a",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/LAUNCH_ACCEPTANCE.json": (
        2_933,
        "2afbba374ea2e332766368087490660198110027784924c69d03a5656c924346",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/STDOUT.raw.json": (
        30_050,
        "98345f64e9a8ec41978f8679cd631095b9bbb5936be8895d4bda574927855e73",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/TRANSPORT.json": (
        546,
        "1c30ca323cb5899c8269c24f277e6b73a33305c4855c8a3167490785038e5a57",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_fit_v1/FINAL_BODY_SOURCE_REVIEW_V1.json": (
        7_046,
        "1ab84547255d6c9b9c950569b09e6410eb640ce7e1deaed5950a718c36e1bada",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/ROOT_ADMISSION_PREPARATION.json": (
        2_518,
        "67b25be1265091d7b29dd08612a146a4d44223129efa8a5c104f25a2e1bfaa44",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/LAUNCH_ACCEPTANCE.json": (
        2_931,
        "44a708fc5a32b101c97a8a28c22e3b15e815059582de564e17f528f5369dc7a5",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_control_terminal_verifier_v3/HUMAN_PATH_COMPAT_EXCEPTION_AUTHORITY_V1.json": (
        1_862,
        "9d0259675d97856b744abe63030575e38551b58d1fc9ac7adaa1281f535abacb",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1/helper_reference_audit.md": (
        2_948,
        "b52ef439825f2fa9caaefbf9b4136356671b8e128bf6a5c62b396765714c833b",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1/dependency_manifest.json": (
        5_480,
        "aa797e8e8f6c7f1cbd10667df64fdc85e7d94ff9bd6ac1319a3737ec7b655721",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1/PRIVATE_IMPORT_BOUNDARY_PROOF_V1.json": (
        12_712,
        "1456ca8e051aa44ca39c836e1d6a531deecd0cac87307e1be2aae982aba7ba93",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1/private_import_boundary_proof_allthree_terminal_check_v1.py": (
        23_177,
        "6f4318aaa3eea9c88acc693b3d7a822a6b0e9ae61450202f6dedf79ef5bceda8",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1/PRIVATE_IMPORT_BOUNDARY_SCOPE_V1.md": (
        3_643,
        "8554ae72ee26d964e54cced97ee4509f75efcef81430389440e8aa9be17ec33f",
    ),
}


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def read_regular(path: Path, cap: int) -> bytes:
    """Read a bounded regular file with no-follow and stable inode checks."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > cap:
        raise ValueError(f"unsafe or oversized input: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"input changed before read: {path}")
        raw = bytearray()
        while len(raw) <= cap:
            part = os.read(descriptor, min(65_536, cap + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > cap:
        raise ValueError(f"input size changed: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"input changed while read: {path}")
    return bytes(raw)


def json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse a JSON object while retaining its separately pinned bytes."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {name}")
    return value


def safe_repo_path(value: object) -> Path:
    """Resolve a repository-relative metadata path without traversal or redirects."""
    if not isinstance(value, str):
        raise ValueError("metadata pin path is not text")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe metadata pin path: {value!r}")
    path = REPOSITORY / relative
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError(f"metadata input is redirected: {path}")
    return path


def check_native_proof(terminal_raw: bytes, capture_raw: bytes) -> dict[str, object]:
    """Join strict native exit fields to the exact raw-first native capture."""
    terminal = json_object(terminal_raw, "typed AllThree native terminal")
    if (
        terminal.get("schema"),
        terminal.get("run_name"),
        terminal.get("unit"),
        terminal.get("invocation_id"),
        terminal.get("active_state"),
        terminal.get("sub_state"),
    ) != (NATIVE_SCHEMA, RUN, UNIT, INVOCATION, "active", "exited"):
        raise ValueError("typed native identity or state differs")
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        if type(terminal.get(key)) is not int or terminal[key] != expected:
            raise ValueError(f"typed exit field {key} is not exact strict integer {expected}")
    if (
        type(terminal.get("raw_bytes")) is not int
        or terminal["raw_bytes"] != len(capture_raw)
        or not 0 < len(capture_raw) <= MAX_NATIVE_RAW_BYTES
    ):
        raise ValueError("native raw capture byte count differs or is outside bounds")
    raw_sha = sha256(capture_raw)
    if terminal.get("raw_sha256") != raw_sha or terminal.get("raw_path") != str(
        NATIVE_RAW.relative_to(REPOSITORY)
    ):
        raise ValueError("typed native raw capture pin differs")
    capture = json_object(capture_raw, "raw-first AllThree native capture")
    if (
        capture.get("schema"),
        capture.get("expected_run"),
        capture.get("expected_unit"),
        capture.get("expected_invocation_id"),
        capture.get("captured_utc"),
    ) != (RAW_SCHEMA, RUN, UNIT, INVOCATION, terminal.get("captured_utc")):
        raise ValueError("raw-first capture identity differs")
    props = capture.get("unit_properties")
    if not isinstance(props, dict) or tuple(
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
    ) != (INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("raw-first capture does not prove exact successful native exit")
    stamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        if not isinstance(value, str):
            raise ValueError(f"typed terminal omits {key}")
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"typed terminal {key} is malformed") from error
        if stamp.tzinfo is None:
            raise ValueError(f"typed terminal {key} lacks a timezone")
        stamps[key] = stamp
    if stamps["native_exit_utc"] > stamps["captured_utc"]:
        raise ValueError("native exit follows its capture")
    return {
        "typed_sha256": sha256(terminal_raw),
        "typed_bytes": len(terminal_raw),
        "raw_sha256": raw_sha,
        "raw_bytes": len(capture_raw),
        "captured_utc": terminal["captured_utc"],
        "native_exit_utc": terminal["native_exit_utc"],
    }


def check_code_dependencies(root: dict[str, object]) -> dict[str, dict[str, object]]:
    """Prehash the exact seven-file private closure before the worker imports it."""
    entries = root.get("code_dependencies")
    if not isinstance(entries, list) or len(entries) != len(CODE_INPUTS):
        raise ValueError("root preparation must bind the exact seven-file code closure")
    by_name: dict[str, dict[str, object]] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("malformed code dependency entry")
        path = safe_repo_path(item["path"])
        name = path.name
        if name in by_name or name not in CODE_INPUTS or path != CODE_INPUTS[name]:
            raise ValueError(f"unexpected or redirected private dependency: {name}")
        raw = read_regular(path, 100_000)
        expected = CODE_PINS[name]
        if (
            type(item.get("bytes")) is not int
            or (len(raw), sha256(raw)) != expected
            or (item["bytes"], item.get("sha256")) != expected
        ):
            raise ValueError(f"private code dependency pin differs: {name}")
        by_name[name] = {"path": item["path"], "bytes": len(raw), "sha256": sha256(raw), "raw": raw}
    if set(by_name) != set(CODE_INPUTS):
        raise ValueError("private code closure names differ")
    return by_name


def check_source_review(root: dict[str, object], checker: dict[str, object]) -> dict[str, object]:
    """Require a final source-only PASS that binds the exact AllThree checker."""
    item = root.get("source_review")
    if not isinstance(item, dict) or item.get("status") != "PASS_SOURCE_ONLY":
        raise ValueError("root preparation has no final source-only review")
    raw = read_regular(safe_repo_path(item.get("path")), MAX_METADATA_BYTES)
    if (
        type(item.get("bytes")) is not int
        or len(raw) != item["bytes"]
        or sha256(raw) != item.get("sha256")
    ):
        raise ValueError("source-review pin differs")
    review = json_object(raw, "AllThree checker source review")
    if (
        not isinstance(review.get("status"), str)
        or not review["status"].startswith("PASS")
        or review.get("blocking_findings") not in (None, [])
    ):
        raise ValueError("source review has a failure or blocking finding")
    wanted_path = checker["path"]
    stack: list[object] = [review]
    found = False
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if value.get("path") == wanted_path and (value.get("bytes"), value.get("sha256")) == (
                checker.get("bytes"),
                checker.get("sha256"),
            ):
                found = True
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    if not found:
        raise ValueError("source review does not bind the exact checker bytes")
    return {
        "path": item["path"],
        "bytes": len(raw),
        "sha256": sha256(raw),
        "status": item["status"],
    }


def check_import_boundary_proof(boundary_doc: dict[str, object]) -> None:
    """Require the frozen seven-file import proof and its exact execution boundary."""
    expected_code_pins = {
        name: {"bytes": size, "sha256": sha} for name, (size, sha) in CODE_PINS.items()
    }
    if (boundary_doc.get("schema"), boundary_doc.get("status")) != (
        "fg19.child-allthree-terminal-import-boundary-proof.v1",
        "PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V1",
    ):
        raise ValueError("accepted import-boundary proof schema or status differs")
    private_closure = boundary_doc.get("private_closure")
    if not isinstance(private_closure, dict):
        raise ValueError("accepted import-boundary proof omits private closure evidence")
    for key in (
        "pre_copy_source_pins",
        "temporary_copy_pins_before_child_import",
        "child_pre_import_pins",
        "child_post_import_pins",
    ):
        if private_closure.get(key) != expected_code_pins:
            raise ValueError(
                f"accepted import-boundary proof {key} differs from the exact seven-code closure"
            )
    if private_closure.get("child_preloaded_module_names") != []:
        raise ValueError(
            "accepted import-boundary proof does not reject all preloaded private module names"
        )
    helper_pins = {
        name: pin
        for name, pin in expected_code_pins.items()
        if name != "child_allthree_terminal_check_v1.py"
    }
    if private_closure.get("helper_closure_recheck") != helper_pins:
        raise ValueError("accepted import-boundary proof helper closure recheck differs")
    origins = private_closure.get("child_module_origins")
    if (
        not isinstance(origins, dict)
        or set(origins) != set(CODE_PINS)
        or any(
            not isinstance(value, str) or not value.startswith("/") for value in origins.values()
        )
    ):
        raise ValueError("accepted import-boundary proof module origins are incomplete")
    if private_closure.get("exact_file_count") != 7 or set(
        private_closure.get("exact_file_set_before_and_after_import", [])
    ) != set(CODE_PINS):
        raise ValueError("accepted import-boundary proof exact file set differs")
    entrypoint = boundary_doc.get("entrypoint_scope")
    if not isinstance(entrypoint, dict) or (
        entrypoint.get("checker_functions_called"),
        entrypoint.get("verify_terminal_called"),
        entrypoint.get("main_or_cli_called"),
        entrypoint.get("fit_or_evaluation_called"),
        entrypoint.get("source_cli_run_identity_representation_or_model_imports_exercised"),
        entrypoint.get("wnn_source_modules_loaded"),
    ) != (["assert_helper_closure"], False, False, False, False, 0):
        raise ValueError(
            "accepted import-boundary proof exercised a forbidden entrypoint or source module"
        )
    interpreter = boundary_doc.get("interpreter_binding")
    if not isinstance(interpreter, dict) or (
        interpreter.get("child_python_flags"),
        interpreter.get("child_dont_write_bytecode"),
        interpreter.get("child_pycache_prefix"),
        interpreter.get("private_bytecode_absent_before_import"),
        interpreter.get("private_bytecode_absent_after_import"),
    ) != (["-E", "-B", "-I"], True, None, True, True):
        raise ValueError(
            "accepted import-boundary proof lacks isolated no-bytecode import controls"
        )
    harness_pin = boundary_doc.get("harness")
    harness_path = str(IMPORT_BOUNDARY_HARNESS.resolve(strict=True))
    if not isinstance(harness_pin, dict) or (
        harness_pin.get("path"),
        harness_pin.get("bytes"),
        harness_pin.get("sha256"),
    ) != (
        harness_path,
        23_177,
        "6f4318aaa3eea9c88acc693b3d7a822a6b0e9ae61450202f6dedf79ef5bceda8",
    ):
        raise ValueError("accepted import-boundary proof harness pin differs")


def check_metadata_pins(root: dict[str, object]) -> list[dict[str, object]]:
    """Recheck historical fit anchors and the acyclic conditional review graph."""
    entries = root.get("metadata_and_proof_pins")
    if not isinstance(entries, list) or not entries:
        raise ValueError("root preparation omits metadata/proof pins")
    observed: list[dict[str, object]] = []
    seen: set[str] = set()
    hashes: set[str] = set()
    by_path: dict[str, dict[str, object]] = {}
    for item in entries:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or item["path"] in seen
        ):
            raise ValueError("malformed or duplicate metadata/proof pin")
        seen.add(item["path"])
        if Path(item["path"]).suffix.lower() in {".pt", ".pth", ".npy", ".npz", ".parquet"}:
            raise ValueError("root metadata pin names a model or data payload")
        raw = read_regular(safe_repo_path(item["path"]), MAX_METADATA_BYTES)
        if (
            type(item.get("bytes")) is not int
            or len(raw) != item["bytes"]
            or sha256(raw) != item.get("sha256")
        ):
            raise ValueError(f"metadata/proof pin differs: {item['path']}")
        hashes.add(sha256(raw))
        current = {"path": item["path"], "bytes": len(raw), "sha256": sha256(raw)}
        observed.append(current)
        by_path[item["path"]] = current
    if not PREDECESSOR_SHAS.issubset(hashes) or len(PREDECESSOR_SHAS) != 14:
        raise ValueError("root preparation does not pin all fourteen accepted predecessor receipts")
    if set(READONLY_RECEIPT_SHAS.values()) != PREDECESSOR_SHAS:
        raise ValueError("local fourteen-receipt map differs from the checker contract")
    if root.get("readonly_ancestry_receipt_sha256") != READONLY_RECEIPT_SHAS:
        raise ValueError(
            "root preparation does not bind the exact fourteen accepted receipt names and hashes"
        )
    for path, (size, sha) in FIXED_METADATA_PATHS.items():
        if (by_path.get(path, {}).get("bytes"), by_path.get(path, {}).get("sha256")) != (size, sha):
            raise ValueError(f"root preparation omits exact historical fit metadata: {path}")
    required_history_hashes = {
        "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
        "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
        "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
        "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7",
        "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4",
    }
    if not required_history_hashes.issubset(hashes):
        raise ValueError("root preparation omits fixed profile or accepted cache receipts")
    source_review = root.get("source_review")
    if not isinstance(source_review, dict) or (
        by_path.get(source_review.get("path"), {}).get("bytes"),
        by_path.get(source_review.get("path"), {}).get("sha256"),
    ) != (source_review.get("bytes"), source_review.get("sha256")):
        raise ValueError(
            "root preparation source review is not identically present in metadata pins"
        )
    template_path = str(TEMPLATE.relative_to(REPOSITORY))
    renderer_path = str(Path(__file__).resolve().relative_to(REPOSITORY))
    review_path = str(STATIC_REVIEW.relative_to(REPOSITORY))
    for path, expected in ((template_path, None), (renderer_path, None), (review_path, None)):
        if path not in by_path:
            raise ValueError(f"root preparation omits finalized conditional source pin: {path}")
        if expected is not None and (by_path[path]["bytes"], by_path[path]["sha256"]) != expected:
            raise ValueError(f"conditional source pin differs: {path}")
    review_pin = by_path[review_path]
    static_review_raw = read_regular(STATIC_REVIEW, MAX_METADATA_BYTES)
    if (len(static_review_raw), sha256(static_review_raw)) != (
        review_pin["bytes"],
        review_pin["sha256"],
    ):
        raise ValueError("conditional source review changed after its root-preparation pin")
    static_review = json_object(static_review_raw, "conditional source review")
    if static_review.get("status") != "PASS_CONDITIONAL_SOURCE_ONLY" or static_review.get(
        "blocking_findings"
    ) not in (None, []):
        raise ValueError("conditional source review is not a nonblocking static-only PASS")
    for path in (template_path, renderer_path):
        pin = by_path[path]
        stack: list[object] = [static_review]
        found = False
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if value.get("path") == path and (value.get("bytes"), value.get("sha256")) == (
                    pin["bytes"],
                    pin["sha256"],
                ):
                    found = True
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        if not found:
            raise ValueError(f"conditional source review does not bind exact file bytes: {path}")
    import_boundary = root.get("accepted_import_boundary_proof")
    if not isinstance(import_boundary, dict) or not isinstance(import_boundary.get("path"), str):
        raise ValueError("accepted import-boundary proof pin is missing")
    boundary_path = import_boundary["path"]
    if boundary_path != str(IMPORT_BOUNDARY_PROOF.relative_to(REPOSITORY)):
        raise ValueError("accepted import-boundary proof path differs from the fixed private proof")
    boundary = by_path.get(boundary_path)
    if boundary is None or (boundary.get("bytes"), boundary.get("sha256")) != (
        import_boundary.get("bytes"),
        import_boundary.get("sha256"),
    ):
        raise ValueError("accepted import-boundary proof is not identically pinned in metadata")
    boundary_doc = json_object(
        read_regular(safe_repo_path(boundary_path), MAX_METADATA_BYTES),
        "accepted import-boundary proof",
    )
    check_import_boundary_proof(boundary_doc)
    return observed


def check_root_preparation(
    raw: bytes,
) -> tuple[
    dict[str, object], dict[str, dict[str, object]], dict[str, object], list[dict[str, object]]
]:
    """Require the final local preparation, source review, code closure, and proof pins."""
    root = json_object(raw, "AllThree root preparation")
    identity = root.get("identity")
    expected_identity = {
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "source_commit": COMMIT,
        "source_files": 167,
        "source_manifest_sha256": MANIFEST_SHA,
    }
    if (
        (root.get("schema"), root.get("status"), identity)
        != (ROOT_SCHEMA, ROOT_STATUS, expected_identity)
        or type(root.get("dependency_binding_condition_satisfied")) is not bool
        or root["dependency_binding_condition_satisfied"] is not True
        or type(root.get("native_terminal_evidence_present")) is not bool
        or root["native_terminal_evidence_present"] is not False
        or type(root.get("actual_B_staging_admission_or_execution")) is not bool
        or root["actual_B_staging_admission_or_execution"] is not False
    ):
        raise ValueError("AllThree root preparation identity or local-only status differs")
    code = check_code_dependencies(root)
    review = check_source_review(root, code["child_allthree_terminal_check_v1.py"])
    pins = check_metadata_pins(root)
    return root, code, review, pins


def load_inputs(
    root: dict[str, object], code: dict[str, dict[str, object]], proof: dict[str, object]
) -> dict[str, bytes]:
    """Load only the seven code files and fixed bounded metadata; never fit payloads."""
    inputs: dict[str, bytes] = {
        name: item["raw"] for name, item in code.items() if isinstance(item.get("raw"), bytes)
    }
    for name, path in METADATA_INPUTS.items():
        cap = MAX_NATIVE_RAW_BYTES if name == "NATIVE_STDOUT.raw.json" else MAX_METADATA_BYTES
        raw = read_regular(path, cap)
        if name in FIXED_METADATA and (len(raw), sha256(raw)) != FIXED_METADATA[name]:
            raise ValueError(f"fixed AllThree metadata pin differs: {name}")
        if name == "TERMINAL_EVIDENCE.json" and sha256(raw) != proof["typed_sha256"]:
            raise ValueError("typed terminal changed after proof validation")
        if name == "NATIVE_STDOUT.raw.json" and sha256(raw) != proof["raw_sha256"]:
            raise ValueError("raw native capture changed after proof validation")
        inputs[name] = raw
    return inputs


def check_rendered_body(body: str, payloads: dict[str, bytes]) -> list[str]:
    """AST-check payload identity and fail closed on worker self-read or missing gates."""
    tree = ast.parse(body)
    if not tree.body or not isinstance(tree.body[-1], ast.If):
        raise ValueError("support body must have an inert main guard")
    if (
        "Path(__file__)" in body
        or "stable_file(__file__" in body
        or 'body_sha256_claimed": True' in body
    ):
        raise ValueError("stdin body must not self-read or claim a worker-side body hash")
    if "checker.verify_terminal" not in body or "Path(value) for value in" not in body:
        raise ValueError("checker API inputs must be constructed as pathlib.Path values")
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    payload_node = assignments.get("PAYLOAD")
    pin_node = assignments.get("FIXED_PAYLOAD_PINS")
    if not isinstance(payload_node, ast.Dict) or not isinstance(pin_node, ast.Dict):
        raise ValueError("rendered body lacks literal exact payload and pin maps")
    packaged = ast.literal_eval(payload_node)
    fixed = ast.literal_eval(pin_node)
    if set(packaged) != set(payloads) or set(fixed) != set(payloads):
        raise ValueError("rendered body payload names differ")
    for name, raw in payloads.items():
        item = packaged[name]
        decoded = base64.b64decode(item["base64"], validate=True)
        if decoded != raw or (item["bytes"], item["sha256"]) != (len(raw), sha256(raw)):
            raise ValueError(f"rendered payload differs from exact local bytes: {name}")
        if fixed[name] != {"bytes": len(raw), "sha256": sha256(raw)}:
            raise ValueError(f"rendered fixed pin differs from exact local bytes: {name}")
    main = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    if main is None:
        raise ValueError("main function is absent")
    gate = next(
        (
            i
            for i, node in enumerate(main.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "D"
            and node.value.func.attr == "mkdir"
        ),
        -1,
    )
    payload_call = next(
        (
            i
            for i, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "payload" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "payload_bytes"
        ),
        -1,
    )
    preflight_call = next(
        (
            i
            for i, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "gates" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "preflight"
        ),
        -1,
    )
    if min(payload_call, preflight_call, gate) < 0 or not payload_call < preflight_call < gate:
        raise ValueError("payload prehash and fresh gates must precede output mutation")
    preflight = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preflight"
        ),
        None,
    )
    if preflight is None:
        raise ValueError("fresh preflight function is absent")
    gpu_call = next(
        (
            i
            for i, node in enumerate(preflight.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "gpu" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "check_selected_gpu_ownership"
        ),
        -1,
    )
    preflight_return = next(
        (i for i, node in enumerate(preflight.body) if isinstance(node, ast.Return)), -1
    )
    if gpu_call < 0 or preflight_return < 0 or gpu_call >= preflight_return:
        raise ValueError("fresh selected-GPU ownership gate is missing")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "D"
        and node.func.attr == "mkdir"
        for node in ast.walk(preflight)
    ):
        raise ValueError("fresh preflight performs support-output mutation")
    return [
        "body AST parses without execution",
        "strict native proof is a renderer prerequisite",
        "all seven private code files are prehashed before imports",
        "fresh source, predecessor, resource, and GPU gates precede D.mkdir",
        "checker API paths are pathlib.Path objects",
        "exact terminal/raw bytes are embedded",
        "no stdin body self-read or body-hash claim",
    ]


def render_body() -> tuple[bytes, dict[str, object]]:
    """Create one body only after root review and strict successful native exit proof."""
    if BODY.exists() or BODY.is_symlink():
        raise FileExistsError("AllThree body already exists; renderer is create-only")
    template_raw = read_regular(TEMPLATE, 100_000)
    if sha256(template_raw) != TEMPLATE_BASE_SHA:
        raise ValueError("AllThree template differs from the pinned candidate")
    root_raw = read_regular(ROOT_PREPARATION, MAX_METADATA_BYTES)
    root, code, source_review, metadata_pins = check_root_preparation(root_raw)
    terminal_raw = read_regular(TERMINAL, 16_000)
    capture_raw = read_regular(NATIVE_RAW, MAX_NATIVE_RAW_BYTES)
    proof = check_native_proof(terminal_raw, capture_raw)
    payloads = load_inputs(root, code, proof)
    template = template_raw.decode("utf-8")
    placeholders = ("PAYLOAD = {}", "FIXED_PAYLOAD_PINS = {}", "FINAL_REVIEW_BINDING = {}")
    if any(template.count(value) != 1 for value in placeholders):
        raise ValueError("template must have one empty payload, pin, and final binding slot")
    packaged = {
        name: {
            "bytes": len(raw),
            "sha256": sha256(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in sorted(payloads.items())
    }
    fixed = {
        name: {"bytes": len(raw), "sha256": sha256(raw)} for name, raw in sorted(payloads.items())
    }
    checker_pin = {
        "path": str(CHECKER.relative_to(REPOSITORY)),
        "bytes": len(payloads["child_allthree_terminal_check_v1.py"]),
        "sha256": sha256(payloads["child_allthree_terminal_check_v1.py"]),
    }
    binding = {
        "schema": "fg19.child-allthree-terminal-verifier-render-binding.v1",
        "status": "ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND",
        "checker": checker_pin,
        "source_review": source_review,
        "root_preparation": {
            "path": str(ROOT_PREPARATION.relative_to(REPOSITORY)),
            "bytes": len(root_raw),
            "sha256": sha256(root_raw),
            "status": root.get("status"),
        },
        "native_terminal": proof,
        "body_template": {
            "path": str(TEMPLATE.relative_to(REPOSITORY)),
            "bytes": len(template_raw),
            "sha256": sha256(template_raw),
        },
        "metadata_and_proof_pin_count": len(metadata_pins),
    }
    body = template.replace("PAYLOAD = {}", "PAYLOAD = " + repr(packaged), 1)
    body = body.replace("FIXED_PAYLOAD_PINS = {}", "FIXED_PAYLOAD_PINS = " + repr(fixed), 1)
    body = body.replace("FINAL_REVIEW_BINDING = {}", "FINAL_REVIEW_BINDING = " + repr(binding), 1)
    checks = check_rendered_body(body, payloads)
    body_raw = body.encode("utf-8")
    ast.parse(body_raw.decode("utf-8"))
    report = {
        "schema": "fg19.child-allthree-terminal-verifier-render-binding.v1",
        "status": "RENDERED_CREATE_ONLY_LOCAL_BODY_NOT_STAGED_OR_ADMITTED",
        "body_bytes": len(body_raw),
        "body_sha256": sha256(body_raw),
        "payload_count": len(payloads),
        "code_count": len(code),
        "native_proof": proof,
        "root_preparation_sha256": sha256(root_raw),
        "source_review": source_review,
        "metadata_and_proof_pin_count": len(metadata_pins),
        "checks": checks,
        "body_executed_or_imported": False,
        "remote_or_B_action": False,
    }
    descriptor = os.open(BODY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(body_raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(OUTPUT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return body_raw, report


def readiness() -> dict[str, object]:
    """Report which strict proof inputs are still absent without rendering."""
    required = (ROOT_PREPARATION, TERMINAL, NATIVE_RAW)
    missing = [str(path.relative_to(REPOSITORY)) for path in required if not path.exists()]
    return {
        "status": "WAITING_FOR_ROOT_PREPARATION_AND_NATIVE_EXIT0_PROOF"
        if missing
        else "READY_FOR_EXPLICIT_RENDER_CHECK",
        "missing_inputs": missing,
        "body_rendered": BODY.exists(),
        "remote_or_B_action": False,
    }


def main() -> int:
    """Report readiness or explicitly render after every proof and review gate passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="render only after exact root/source/native bindings exist",
    )
    args = parser.parse_args()
    report = render_body()[1] if args.render else readiness()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
