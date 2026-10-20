"""Render and locally bind V2's one-shot EMA terminal-verifier admission body."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[5]
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records"
OUTPUT = Path(__file__).resolve().parent
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY = OUTPUT / "REMOTE_SUBMISSION.py"
FOCUSED_FIXTURE = OUTPUT / "test_child_ema_terminal_verifier_v2.py"
EMA_PREP = OUTPUT
ROOT_PREPARATION = OUTPUT / "ROOT_PREPARATION.json"
SOURCE_REVIEW = PREPARATION / "child_ema_terminal_check_v1/CHILD_EMA_TERMINAL_SOURCE_REVIEW_V1.json"
V1_BODY = PREPARATION / "child_ema_terminal_verifier_v1/REMOTE_SUBMISSION.py"
NATIVE_METADATA_REVIEW = (
    RECORDS / "child_ema_fit_v1/terminal_native_capture_v1/METADATA_REVIEW.json"
)

PAYLOAD_INPUTS = {
    "child_ema_terminal_check_v2.py": EMA_PREP / "child_ema_terminal_check_v2.py",
    "child_control_terminal_check_v3.py": (
        PREPARATION / "child_control_terminal_verifier_v3/child_control_terminal_check_v3.py"
    ),
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
    "SOURCE_MANIFEST.json": PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json",
    "LAUNCH_ACCEPTANCE.json": RECORDS / "child_ema_fit_v1/LAUNCH_ACCEPTANCE.json",
    "TERMINAL_EVIDENCE.json": RECORDS / "child_ema_fit_v1/TERMINAL_EVIDENCE.json",
    "NATIVE_STDOUT.raw.json": (
        RECORDS / "child_ema_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
    ),
    "V1_FAILURE_STDOUT.raw.json": (
        RECORDS / "child_ema_fit_v1/terminal_verification_v1/capture/STDOUT.raw.json"
    ),
    "V1_FAILURE_UNIT_LOG.txt": (
        RECORDS / "child_ema_fit_v1/terminal_verification_v1/capture/unit.log"
    ),
}

LOCAL_INPUTS = {
    "ADMISSION.json": RECORDS / "child_ema_fit_v1/ADMISSION.json",
    "LAUNCH.json": RECORDS / "child_ema_fit_v1/LAUNCH.json",
    "LAUNCH_ACCEPTANCE.json": RECORDS / "child_ema_fit_v1/LAUNCH_ACCEPTANCE.json",
    "TERMINAL_EVIDENCE.json": RECORDS / "child_ema_fit_v1/TERMINAL_EVIDENCE.json",
    "NATIVE_STDOUT.raw.json": (
        RECORDS / "child_ema_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
    ),
    "V1_FAILURE_STDOUT.raw.json": (
        RECORDS / "child_ema_fit_v1/terminal_verification_v1/capture/STDOUT.raw.json"
    ),
    "V1_FAILURE_UNIT_LOG.txt": (
        RECORDS / "child_ema_fit_v1/terminal_verification_v1/capture/unit.log"
    ),
}

READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "parent_fit_r2_v1/TERMINAL_ACCEPTANCE.json",
        "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
    ),
    "PARENT_EXACT_STOP.json": (
        RECORDS / "parent_fit_r2_v1/exact_stop/STDOUT.raw.json",
        "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
    ),
    "DICTIONARY_ACCEPTANCE.json": (
        RECORDS / "dictionary_f0_v1/ACCEPTANCE.json",
        "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
    ),
    "DICTIONARY_EXACT_STOP.json": (
        RECORDS / "dictionary_f0_v1/exact_stop/STDOUT.raw.json",
        "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
    ),
    "PROFILE_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_allthree_profile_v1/TERMINAL_ACCEPTANCE.json",
        "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
    ),
    "PROFILE_EXACT_STOP.json": (
        RECORDS / "child_allthree_profile_v1/exact_stop/STDOUT.raw.json",
        "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    ),
    "CONTROL_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_control_fit_v1/TERMINAL_ACCEPTANCE.json",
        "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    ),
    "CONTROL_EXACT_STOP.json": (
        RECORDS / "child_control_fit_v1/exact_stop/STDOUT.raw.json",
        "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
    ),
}

PROFILE_FILES = {
    "ADMISSION.json": (
        RECORDS / "child_allthree_profile_v1/ADMISSION.json",
        7_817,
        "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
    ),
    "LAUNCH.json": (
        RECORDS / "child_allthree_profile_v1/LAUNCH.json",
        9_691,
        "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
    ),
    "unit.log": (
        RECORDS / "child_allthree_profile_v1/capture/unit.log",
        7_624,
        "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
    ),
}

CACHE_RECEIPTS = {
    "TERMINAL_ACCEPTANCE.json": (
        RECORDS / "teacher_cache_cpu_build_v2/TERMINAL_ACCEPTANCE.json",
        "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7",
    ),
    "EXACT_STOP.json": (
        RECORDS / "teacher_cache_cpu_build_v2/exact_stop/STDOUT.raw.json",
        "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4",
    ),
}

EXPECTED = {
    "child_ema_terminal_check_v2.py": (
        38_266,
        "c6bf44fe1c2702b5a9b415992136e56e99c811149e8ca3c5893e8e77289bd909",
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
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
    "SOURCE_MANIFEST.json": (
        27_569,
        "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
    ),
    "LAUNCH_ACCEPTANCE.json": (
        4_839,
        "4fc34f4769da2f08aaf1fcca97d87731dc38b1709a93afa9376be003c28ffb72",
    ),
    "TERMINAL_EVIDENCE.json": (
        697,
        "7cb75057155f3582e0493d1f6ca8c88eff475b022215d755f53751495f5c7d0a",
    ),
    "NATIVE_STDOUT.raw.json": (
        75_226,
        "f8534769efdfd1949d5570927a0b5804f2819570c6154bac60c38387410a7cad",
    ),
    "V1_FAILURE_STDOUT.raw.json": (
        26_784,
        "ecb7f92562964f3b29dd4faec8faf81975f95722c892e8323fd72b40b9c4c063",
    ),
    "V1_FAILURE_UNIT_LOG.txt": (
        1_788,
        "61ddbbd97666c595cd404e00c3c143ff24882901aa1fe34e01ed4819831ff192",
    ),
}

RUN = "fg19_child_ema_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = EXPECTED["SOURCE_MANIFEST.json"][1]
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
BODY_SHA = "f83ac3120a2e445f1752fcd026123a0e8a21ba53594d27b398462a5236ffebd6"
BODY_BYTES = 163_040
SOURCE_REVIEW_SHA = "9662dd9437235abfc4bedd0f7136eb1ed49b370bb62bc7bd746c90cc9569a769"
PATH_AUTHORITY_SHA = "9d0259675d97856b744abe63030575e38551b58d1fc9ac7adaa1281f535abacb"
NATIVE_REVIEW_SHA = "8b752246f0f872027ab2038b747e33ef680cb950a00655d7a83a54974d798eee"
CONTROL_ACCEPTANCE_SHA = "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
CONTROL_STOP_SHA = "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
SOURCE_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/CHILD_EMA_FIT_SOURCE_REVIEW_V1.json"
BODY_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/REMOTE_SUBMISSION.py"
)
V1_BODY_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_ema_terminal_verifier_v1/REMOTE_SUBMISSION.py"
V1_BODY_SHA = "03879ea709e48a93a7c11fcfd204d2dabb5c2039a48dde872177860bb157b35f"
V1_CHECKER_SHA = "a63ec6c4f6b0a3c4ba7e0cb8800e4787c5d89f1a652e0deec15ad84ffc95d30a"
V1_FAILURE_RAW_SHA = EXPECTED["V1_FAILURE_STDOUT.raw.json"][1]
V1_FAILURE_LOG_SHA = EXPECTED["V1_FAILURE_UNIT_LOG.txt"][1]
V1_FAILURE_ADMISSION_SHA = "d88a537ba0154c779031c440e44681f8efb7c9b064bd8e3decc3d495e79b9ee8"
V1_FAILURE_LAUNCH_SHA = "386f80ca3e96248d183069907e7ceaa3466da171801bda6ac33a8a938e7fc230"
V1_SUPPORT_INVOCATION = "f0caca9f0ba04efc976a45b4a0b2a651"


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def stable_read(path: Path, cap: int) -> bytes:
    """Read one bounded regular file without following a symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > cap:
        raise ValueError(f"unsafe or oversized local input: {path}")
    raw = path.read_bytes()
    after = path.stat()
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"local input changed while reading: {path}")
    return raw


def json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse bounded JSON metadata and require an object root."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {name}")
    return value


def control_predecessor() -> dict[str, object]:
    """Return the exact accepted-and-stopped CONTROL identity carried by EMA."""
    return {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA,
        "exact_stop_sha256": CONTROL_STOP_SHA,
        "invocation_id": "e66e0d0703314e4fb5d47dcdbfcaf202",
        "run_name": "fg19_child_control_f0_s1001_fit_v1",
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": "computer-vision-homework-fg19-child-control-fit-v1.service",
    }


def expected_root_receipts() -> dict[str, dict[str, object]]:
    """Return the original root acceptance's four nested receipt pins."""
    base = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/"
    return {
        "admission": {
            "path": base + "ADMISSION.json",
            "bytes": 9_880,
            "sha256": "db13e5433cb0a414c1a235806220c53db4ded6c128fc5ffbda5e465dde9fde51",
        },
        "launch": {
            "path": base + "LAUNCH.json",
            "bytes": 7_404,
            "sha256": "8c43f6fb26d3f6a046304617094e266a3e1940c83a7ea56787781fa176ef133e",
        },
        "body": {"path": BODY_PATH, "bytes": BODY_BYTES, "sha256": BODY_SHA},
        "source_review": {
            "path": SOURCE_REVIEW_PATH,
            "bytes": 8_000,
            "sha256": "3d2f78b0c8902c1a43748a237310ae6a31277e60fa9c18c13a5d162c13722138",
        },
    }


def validate_terminal_object(terminal: Mapping[str, object]) -> None:
    """Require EMA identity and exact typed native exit fields."""
    identity = (
        terminal.get("schema"),
        terminal.get("run_name"),
        terminal.get("unit"),
        terminal.get("invocation_id"),
        terminal.get("active_state"),
        terminal.get("sub_state"),
    )
    if identity != (
        "fg19.child-ema-fit-native-terminal-evidence.v1",
        RUN,
        UNIT,
        INVOCATION,
        "active",
        "exited",
    ):
        raise ValueError("EMA typed native identity differs")
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        if type(terminal.get(key)) is not int or terminal[key] != expected:
            raise ValueError(f"EMA terminal {key} must be the exact integer {expected}")
    if (
        type(terminal.get("raw_bytes")) is not int
        or terminal["raw_bytes"] != EXPECTED["NATIVE_STDOUT.raw.json"][0]
    ):
        raise ValueError("EMA native raw byte count differs")
    if terminal.get("raw_sha256") != EXPECTED["NATIVE_STDOUT.raw.json"][1]:
        raise ValueError("EMA native raw digest differs")
    if (
        terminal.get("raw_path")
        != "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
    ):
        raise ValueError("EMA native raw path differs")


def validate_metadata_bindings(
    admission_raw: bytes,
    launch_raw: bytes,
    acceptance_raw: bytes,
    terminal_raw: bytes,
    capture_raw: bytes,
) -> dict[str, object]:
    """Bind the actual launch, original root acceptance, and native exit bytes."""
    admission = json_object(admission_raw, "EMA admission")
    launch = json_object(launch_raw, "EMA launch")
    acceptance = json_object(acceptance_raw, "EMA root launch acceptance")
    terminal = json_object(terminal_raw, "EMA native terminal evidence")
    capture = json_object(capture_raw, "EMA native capture")
    if sha256(admission_raw) != "db13e5433cb0a414c1a235806220c53db4ded6c128fc5ffbda5e465dde9fde51":
        raise ValueError("EMA admission digest differs")
    if sha256(launch_raw) != "8c43f6fb26d3f6a046304617094e266a3e1940c83a7ea56787781fa176ef133e":
        raise ValueError("EMA launch digest differs")
    if sha256(acceptance_raw) != "4fc34f4769da2f08aaf1fcca97d87731dc38b1709a93afa9376be003c28ffb72":
        raise ValueError("EMA root launch acceptance digest differs")
    if sha256(terminal_raw) != EXPECTED["TERMINAL_EVIDENCE.json"][1]:
        raise ValueError("EMA terminal evidence digest differs")
    if sha256(capture_raw) != EXPECTED["NATIVE_STDOUT.raw.json"][1]:
        raise ValueError("EMA native capture digest differs")
    if (
        admission.get("schema"),
        admission.get("run_name"),
        admission.get("unit"),
        admission.get("source_commit"),
        admission.get("source_files_verified"),
        admission.get("source_manifest_sha256"),
        admission.get("split_sha256"),
        admission.get("control_predecessor"),
    ) != (
        "fg19-child-ema-fit-admission.v1",
        RUN,
        UNIT,
        COMMIT,
        167,
        MANIFEST_SHA,
        SPLIT_SHA,
        control_predecessor(),
    ):
        raise ValueError("EMA admission identity differs")
    if admission.get("populated") != {}:
        raise ValueError("EMA original admission contains populated descendants")
    if (
        launch.get("schema"),
        launch.get("unit"),
        launch.get("submission", {}).get("returncode"),
    ) != (
        "fg19-child-ema-fit-launch.v1",
        UNIT,
        0,
    ):
        raise ValueError("EMA original launch identity differs")
    if launch.get("control_predecessor") != control_predecessor():
        raise ValueError("EMA original launch CONTROL predecessor differs")
    if (
        acceptance.get("schema"),
        acceptance.get("status"),
        acceptance.get("run_name"),
        acceptance.get("unit"),
        acceptance.get("invocation_id"),
        acceptance.get("control_predecessor"),
        {key: acceptance.get(key) for key in expected_root_receipts()},
    ) != (
        "fg19-child-ema-fit-launch-acceptance.v1",
        "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
        RUN,
        UNIT,
        INVOCATION,
        control_predecessor(),
        expected_root_receipts(),
    ):
        raise ValueError("EMA actual root launch acceptance differs")
    validate_terminal_object(terminal)
    capture_identity = (
        capture.get("schema"),
        capture.get("expected_run"),
        capture.get("expected_unit"),
        capture.get("expected_invocation_id"),
        capture.get("captured_utc"),
    )
    if capture_identity != (
        "fg19-child-control-startup-capture.v2",
        RUN,
        UNIT,
        INVOCATION,
        terminal.get("captured_utc"),
    ):
        raise ValueError("EMA raw capture identity differs")
    native = capture.get("unit_properties")
    if not isinstance(native, dict) or tuple(
        native.get(key)
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
        raise ValueError("EMA raw capture unit properties differ")
    receipts = capture.get("receipts")
    if not isinstance(receipts, dict):
        raise ValueError("EMA raw capture receipt map is missing")
    for name, raw in (("ADMISSION.json", admission_raw), ("LAUNCH.json", launch_raw)):
        record = receipts.get(name)
        if (
            not isinstance(record, dict)
            or record.get("present") is not True
            or record.get("regular_file") is not True
        ):
            raise ValueError(f"EMA raw capture omits safe {name}")
        decoded = base64.b64decode(record.get("base64", ""), validate=True)
        if (
            type(record.get("bytes")) is not int
            or record["bytes"] != len(decoded)
            or sha256(decoded) != record.get("sha256")
            or decoded != raw
        ):
            raise ValueError(f"EMA raw capture {name} bytes differ")
    return {
        "admission_sha256": sha256(admission_raw),
        "launch_sha256": sha256(launch_raw),
        "launch_acceptance_sha256": sha256(acceptance_raw),
        "terminal_evidence_sha256": sha256(terminal_raw),
        "native_capture_sha256": sha256(capture_raw),
        "native_exit_fields": {
            key: terminal[key] for key in ("main_pid", "exec_main_code", "exec_main_status")
        },
        "existing_receipt_count": len(READONLY_RECEIPTS),
    }


def verify_local_inputs(
    root_manifest: Mapping[str, object],
    payload: Mapping[str, bytes],
    local_metadata: Mapping[str, bytes],
) -> dict[str, object]:
    """Check current root binding, frozen dependency pins, and all read-only evidence."""
    identity = root_manifest.get("identity")
    if not isinstance(identity, dict) or (
        identity.get("run_name"),
        identity.get("unit"),
        identity.get("invocation_id"),
        identity.get("source_commit"),
        identity.get("source_files"),
        identity.get("source_manifest_sha256"),
    ) != (RUN, UNIT, INVOCATION, COMMIT, 167, MANIFEST_SHA):
        raise ValueError("frozen EMA checker preparation identity differs")
    if set(payload) != set(PAYLOAD_INPUTS):
        raise ValueError("fixed EMA payload input set differs")
    if set(local_metadata) != set(LOCAL_INPUTS):
        raise ValueError("fixed local EMA metadata input set differs")
    pin_rows = root_manifest.get("metadata_and_proof_pins")
    if not isinstance(pin_rows, list):
        raise ValueError("frozen checker preparation has no metadata pins")
    pins = {row.get("path"): row.get("sha256") for row in pin_rows if isinstance(row, dict)}
    dependency_rows = root_manifest.get("code_dependencies")
    if not isinstance(dependency_rows, list):
        raise ValueError("frozen EMA checker preparation has no code dependency map")
    dependencies = {row.get("path"): row for row in dependency_rows if isinstance(row, dict)}
    expected_code_paths = {
        str(path.relative_to(REPOSITORY))
        for name, path in PAYLOAD_INPUTS.items()
        if name.endswith(".py")
    }
    if set(dependencies) != expected_code_paths:
        raise ValueError("frozen EMA checker dependency closure differs from the exact five files")
    for name, path in PAYLOAD_INPUTS.items():
        raw = payload[name]
        size, sha = EXPECTED[name]
        if (len(raw), sha256(raw)) != (size, sha):
            raise ValueError(f"frozen payload input differs: {name}")
        if name.endswith(".py") and path.is_symlink():
            raise ValueError(f"code dependency is a symlink: {name}")
        if name.endswith(".py"):
            relative = str(path.relative_to(REPOSITORY))
            row = dependencies.get(relative)
            if not isinstance(row, dict) or (row.get("bytes"), row.get("sha256")) != (size, sha):
                raise ValueError(f"frozen dependency manifest pin differs: {name}")
        elif name == "SOURCE_MANIFEST.json":
            relative = str(path.relative_to(REPOSITORY))
            if pins.get(relative) != sha:
                raise ValueError("frozen source-manifest pin differs")
    old_body = stable_read(V1_BODY, 600_000)
    if sha256(old_body) != V1_BODY_SHA or pins.get(V1_BODY_PATH) != V1_BODY_SHA:
        raise ValueError("exact V1 remote body binding differs")
    for name, path in (
        ("V1_FAILURE_STDOUT.raw.json", PAYLOAD_INPUTS["V1_FAILURE_STDOUT.raw.json"]),
        ("V1_FAILURE_UNIT_LOG.txt", PAYLOAD_INPUTS["V1_FAILURE_UNIT_LOG.txt"]),
    ):
        if pins.get(str(path.relative_to(REPOSITORY))) != EXPECTED[name][1]:
            raise ValueError(f"root preparation V1 failure pin differs: {name}")
    for path, expected_sha in (
        (
            PREPARATION / "child_ema_terminal_check_v1/DEPENDENCY_MANIFEST.json",
            "6a6e7843c278568becfbc3dcc2cc3d0175b9606a549a86ae1ce67a98a902cd7e",
        ),
        (
            PREPARATION / "child_ema_terminal_check_v1/IMPORT_READ_CUDA_PROOF.json",
            "84fea38fe326d1998a76fffa9c1407daad7fe7aaf545b753fdc43b9fc57c128d",
        ),
        (
            PREPARATION / "child_ema_terminal_check_v1/LOCAL_CHECKS.json",
            "d5da5fdedb092f43ef15ff7b63f679d442c4eb524cf675377f78f27128ef9816",
        ),
        (
            PREPARATION / "child_ema_terminal_check_v1/README.md",
            "155ee4baa5503bd0e0cb6b8e5530b7901d7c3dd627c3aaf83e2a242e1b6d1ba2",
        ),
        (
            PREPARATION / "child_ema_terminal_check_v1/CHILD_EMA_TERMINAL_SOURCE_REVIEW_V1.json",
            SOURCE_REVIEW_SHA,
        ),
        (
            PREPARATION
            / "child_control_terminal_verifier_v3/HUMAN_PATH_COMPAT_EXCEPTION_AUTHORITY_V1.json",
            PATH_AUTHORITY_SHA,
        ),
        (NATIVE_METADATA_REVIEW, NATIVE_REVIEW_SHA),
    ):
        raw = stable_read(path, 100_000)
        if sha256(raw) != expected_sha:
            raise ValueError(f"local binding receipt differs: {path.name}")
        relative = str(path.relative_to(REPOSITORY))
        if relative in pins and pins[relative] != expected_sha:
            raise ValueError(f"root preparation pin differs: {path.name}")
    local_rows: dict[str, object] = {}
    for name, (path, expected_sha) in READONLY_RECEIPTS.items():
        raw = stable_read(path, 20_000)
        if sha256(raw) != expected_sha:
            raise ValueError(f"existing read-only predecessor differs: {name}")
        relative = str(path.relative_to(REPOSITORY))
        if pins.get(relative) != expected_sha:
            raise ValueError(f"frozen root receipt pin differs: {name}")
        local_rows[name] = {"sha256": expected_sha, "bytes": len(raw), "read_only": True}
    profile_rows: dict[str, object] = {}
    for name, (path, size, expected_sha) in PROFILE_FILES.items():
        raw = stable_read(path, 1_000_000)
        if (len(raw), sha256(raw)) != (size, expected_sha):
            raise ValueError(f"profile evidence differs: {name}")
        relative = str(path.relative_to(REPOSITORY))
        if pins.get(relative) != expected_sha:
            raise ValueError(f"frozen profile evidence pin differs: {name}")
        profile_rows[name] = {"bytes": size, "sha256": expected_sha, "read_only": True}
    for name, (path, expected_sha) in CACHE_RECEIPTS.items():
        raw = stable_read(path, 20_000)
        if sha256(raw) != expected_sha:
            raise ValueError(f"cache receipt differs: {name}")
        relative = str(path.relative_to(REPOSITORY))
        if pins.get(relative) != expected_sha:
            raise ValueError(f"frozen cache receipt pin differs: {name}")
    metadata = {name: local_metadata[name] for name in LOCAL_INPUTS}
    native_check = validate_metadata_bindings(
        metadata["ADMISSION.json"],
        metadata["LAUNCH.json"],
        metadata["LAUNCH_ACCEPTANCE.json"],
        metadata["TERMINAL_EVIDENCE.json"],
        metadata["NATIVE_STDOUT.raw.json"],
    )
    native_review = json_object(stable_read(NATIVE_METADATA_REVIEW, 20_000), "native review")
    if native_review.get("status") != "PASS_NATIVE_METADATA_ONLY":
        raise ValueError("independent native metadata review is not a pass")
    failure_binding = validate_v1_failure_binding(
        {name: payload[name] for name in ("V1_FAILURE_STDOUT.raw.json", "V1_FAILURE_UNIT_LOG.txt")}
    )
    return {
        "native_metadata_binding": native_check,
        "v1_failure_binding": failure_binding,
        "read_only_predecessor_receipts": local_rows,
        "read_only_profile_files": profile_rows,
        "read_only_cache_receipts": {
            name: {"sha256": sha} for name, (_, sha) in CACHE_RECEIPTS.items()
        },
        "native_metadata_review_sha256": NATIVE_REVIEW_SHA,
        "source_review_sha256": SOURCE_REVIEW_SHA,
        "path_exception_authority_sha256": PATH_AUTHORITY_SHA,
    }


def validate_rendered_body(body: str, expected_payload: Mapping[str, bytes]) -> list[str]:
    """Compile and inspect the one-shot body without executing its main function."""
    compile(body, "REMOTE_SUBMISSION.py", "exec", flags=0, dont_inherit=True)
    module = ast.parse(body)
    main = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    statements = main.body[1:] if ast.get_docstring(main) else main.body
    first = statements[0]
    if not (
        isinstance(first, ast.Assign)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "require_path_exception"
    ):
        raise ValueError("exception marker must be the first admission-body action")
    missing = [
        token
        for token in (
            RUN,
            UNIT,
            "computer-vision-homework-fg19-child-ema-fit-v1.service",
            "computer-vision-homework-fg19-child-ema-verify-v2.service",
            "fg19.child-ema-fit-native-terminal-evidence.v1",
            "MemoryHigh=1400000000",
            "MemoryMax=1500000000",
            "MemorySwapMax=0",
            "RuntimeMaxSec=600",
            "--run-dir",
            "--terminal-evidence",
            "--launch-acceptance",
            "--cache-exact-stop",
            "--profile-evidence-root",
            "--source-root",
            "--source-manifest",
            "--split",
            "--evidence-root",
            "--cache-root",
            "CUDA_VISIBLE_DEVICES",
            "WNN_PROJECT_ROOT",
            "pathlib._local.PosixPath:exact-dictionary-path-only",
            "child_ema_terminal_check_v2.py",
            "V1_FAILURE_STDOUT.raw.json",
            "V1_FAILURE_UNIT_LOG.txt",
            "EMA dictionary stat identity differs",
            '"-E"',
            '"-B"',
            "one exact dictionary-cache path constructor",
            "never retry",
        )
        if token not in body
    ]
    if missing:
        raise ValueError(f"rendered body omits identity or fresh-gate tokens: {missing}")
    if '"torch.load("' in body or '"np.load("' in body or '["systemctl", "stop"' in body:
        raise ValueError("admission body contains a payload loader or stop command")
    if 'CACHE_ACCEPTANCE.json":' in body or 'CACHE_EXACT_STOP.json":' in body:
        raise ValueError("existing cache receipts must not be staged into verifier output")
    return [
        "body_compile",
        "first_action_requires_granted_exact_path_exception",
        "five_code_file_closure_and_bounded_failure_and_terminal_inputs",
        "EMA_native_integer_exit_and_original_launch_binding",
        "existing_predecessor_and_cache_receipts_read_only",
        "fresh_source_split_registry_cache_parent_and_disk_gates",
        "one_shot_cpu_caps_cuda_hidden_and_no_retry",
        "no_stop_or_payload_load_in_admission_body",
    ]


def validate_v1_failure_binding(payload: Mapping[str, bytes]) -> dict[str, object]:
    """Run only the body’s V1 failure metadata predicate on captured bytes."""
    template = stable_read(TEMPLATE, 200_000).decode("utf-8")
    module = ast.parse(template)
    if not module.body or not (
        isinstance(module.body[-1], ast.Expr)
        and isinstance(module.body[-1].value, ast.Call)
        and isinstance(module.body[-1].value.func, ast.Name)
        and module.body[-1].value.func.id == "main"
    ):
        raise ValueError("template must end in the single production main invocation")
    module.body.pop()
    namespace: dict[str, object] = {}
    exec(compile(module, "REMOTE_SUBMISSION_TEMPLATE.py", "exec"), namespace)
    checker = namespace.get("check_v1_failure")
    if not callable(checker):
        raise ValueError("template has no V1 failure metadata predicate")
    result = checker(dict(payload))
    if not isinstance(result, dict):
        raise ValueError("V1 failure predicate returned an invalid receipt")
    return result


def render() -> tuple[str, dict[str, bytes], dict[str, object]]:
    """Validate local metadata and render the final payload-embedded body."""
    root_raw = stable_read(ROOT_PREPARATION, 50_000)
    root_manifest = json_object(root_raw, "frozen EMA checker root preparation")
    payload_raw = {
        name: stable_read(path, 100_000 if name.endswith(".json") else 100_000)
        for name, path in PAYLOAD_INPUTS.items()
    }
    local_metadata = {name: stable_read(path, 100_000) for name, path in LOCAL_INPUTS.items()}
    local_checks = verify_local_inputs(root_manifest, payload_raw, local_metadata)
    template_raw = stable_read(TEMPLATE, 200_000)
    template = template_raw.decode("utf-8")
    packaged = {
        name: {
            "bytes": len(raw),
            "sha256": EXPECTED[name][1],
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in payload_raw.items()
    }
    if template.count("PAYLOAD = {}") != 1:
        raise ValueError("template needs exactly one empty payload slot")
    body = template.replace("PAYLOAD = {}", "PAYLOAD = " + repr(packaged), 1)
    checks = validate_rendered_body(body, payload_raw)
    failure_binding = validate_v1_failure_binding(payload_raw)
    return (
        body,
        payload_raw,
        {
            "local_checks": local_checks,
            "check_names": checks,
            "v1_failure_binding": failure_binding,
            "template_sha256": sha256(template_raw),
        },
    )


def main() -> None:
    """Create the immutable rendered body and its local binding receipts once."""
    outputs = (
        BODY,
        OUTPUT / "ROOT_BINDING.json",
        OUTPUT / "DEPENDENCY_MANIFEST.json",
        OUTPUT / "LOCAL_CHECKS.json",
        OUTPUT / "README.md",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("EMA verifier packet already frozen; inspect before replacement")
    body, payload_raw, checks = render()
    body_raw = body.encode("utf-8")
    BODY.write_bytes(body_raw)
    dependency_entries = {
        name: {"bytes": len(raw), "sha256": sha256(raw), "remote_staged": True}
        for name, raw in sorted(payload_raw.items())
    }
    readonly_entries = {
        name: {
            "path": str(path.relative_to(REPOSITORY)),
            "sha256": expected,
            "remote_staged": False,
        }
        for name, (path, expected) in READONLY_RECEIPTS.items()
    }
    dependency_manifest = {
        "schema": "fg19.child-ema-terminal-verifier-dependencies.v2",
        "status": "FROZEN_LOCAL_PACKET_NOT_STAGED_OR_LAUNCHED",
        "code_closure": {
            name: dependency_entries[name]
            for name in (
                "child_ema_terminal_check_v2.py",
                "child_control_terminal_check_v3.py",
                "parent_fit_terminal_check_v3.py",
                "propensity_terminal_support.py",
                "teacher_s5_terminal_check_v1.py",
            )
        },
        "additional_bounded_inputs": {
            name: dependency_entries[name]
            for name in (
                "SOURCE_MANIFEST.json",
                "LAUNCH_ACCEPTANCE.json",
                "TERMINAL_EVIDENCE.json",
                "NATIVE_STDOUT.raw.json",
                "V1_FAILURE_STDOUT.raw.json",
                "V1_FAILURE_UNIT_LOG.txt",
            )
        },
        "superseded_v1_attempt": {
            "checker_sha256": V1_CHECKER_SHA,
            "body": {"path": V1_BODY_PATH, "sha256": V1_BODY_SHA},
            "failure_capture": {
                "path": "records/child_ema_fit_v1/terminal_verification_v1/capture/STDOUT.raw.json",
                "bytes": EXPECTED["V1_FAILURE_STDOUT.raw.json"][0],
                "sha256": V1_FAILURE_RAW_SHA,
            },
            "failure_log": {
                "path": "records/child_ema_fit_v1/terminal_verification_v1/capture/unit.log",
                "bytes": EXPECTED["V1_FAILURE_UNIT_LOG.txt"][0],
                "sha256": V1_FAILURE_LOG_SHA,
            },
        },
        "original_launch_receipts_read_only_on_worker": {
            "ADMISSION.json": {
                "sha256": "db13e5433cb0a414c1a235806220c53db4ded6c128fc5ffbda5e465dde9fde51",
                "path": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_ema_fit_v1/ADMISSION.json",
            },
            "LAUNCH.json": {
                "sha256": "8c43f6fb26d3f6a046304617094e266a3e1940c83a7ea56787781fa176ef133e",
                "path": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_ema_fit_v1/LAUNCH.json",
            },
        },
        "read_only_predecessor_receipts": readonly_entries,
        "read_only_profile_files": {
            name: {"bytes": size, "sha256": expected}
            for name, (_, size, expected) in PROFILE_FILES.items()
        },
        "read_only_cache_receipts": {
            name: {
                "path": "/home/grinchenkoeg/computer-vision-homework-20261018/research/full_gru_tweaks_r1/cache_cpu_v2/"
                + name,
                "sha256": expected,
            }
            for name, (_, expected) in CACHE_RECEIPTS.items()
        },
        "root_launch_acceptance": {
            "sha256": EXPECTED["LAUNCH_ACCEPTANCE.json"][1],
            "path": "records/child_ema_fit_v1/LAUNCH_ACCEPTANCE.json",
            "status": "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
        },
        "actual_native_terminal": {
            "typed_sha256": EXPECTED["TERMINAL_EVIDENCE.json"][1],
            "raw_sha256": EXPECTED["NATIVE_STDOUT.raw.json"][1],
            "raw_bytes": EXPECTED["NATIVE_STDOUT.raw.json"][0],
            "metadata_review_sha256": NATIVE_REVIEW_SHA,
        },
        "focused_fixture": {
            "path": "test_child_ema_terminal_verifier_v2.py",
            "bytes": len(stable_read(FOCUSED_FIXTURE, 50_000)),
            "sha256": sha256(stable_read(FOCUSED_FIXTURE, 50_000)),
            "tests_passed_after_last_fixture_edit": 6,
        },
    }
    dep_raw = (json.dumps(dependency_manifest, sort_keys=True, indent=2) + "\n").encode()
    (OUTPUT / "DEPENDENCY_MANIFEST.json").write_bytes(dep_raw)
    local_checks = {
        "schema": "fg19.child-ema-terminal-verifier-local-checks.v2",
        "status": "PASS_LOCAL_BINDING_AND_BODY_STRUCTURE_ONLY",
        "body_sha256": sha256(body_raw),
        "body_bytes": len(body_raw),
        "payload_input_count": len(payload_raw),
        "staged_code_file_count": 5,
        "existing_receipts_restaged": False,
        "native_exit_typed_fields": {"main_pid": 0, "exec_main_code": 1, "exec_main_status": 0},
        "checks": checks["check_names"],
        "local_metadata_binding": checks["local_checks"],
        "v1_failure_binding": checks["v1_failure_binding"],
        "model_state_cache_array_dictionary_payloads_read_or_hashed": False,
        "B_or_SSH_actions": False,
        "checker_or_model_suite_executed": False,
        "support_submission": False,
        "focused_fixture": {
            "path": "test_child_ema_terminal_verifier_v2.py",
            "bytes": len(stable_read(FOCUSED_FIXTURE, 50_000)),
            "sha256": sha256(stable_read(FOCUSED_FIXTURE, 50_000)),
            "tests_passed": 6,
        },
    }
    checks_raw = (json.dumps(local_checks, sort_keys=True, indent=2) + "\n").encode()
    (OUTPUT / "LOCAL_CHECKS.json").write_bytes(checks_raw)
    readme = f"""# EMA terminal verifier V2 admission packet

This folder contains a local one-shot CPU admission body for the completed EMA fit. V1's support job failed before any model, state, cache-array, or dictionary payload load: its frozen checker required a dictionary `mtime_ns` field that the exact original accepted dictionary record does not contain. V2 applies separate original schemas: the parent identity has mtime, while the dictionary identity has path, bytes, device, inode, and SHA. The later frozen restricted dictionary load still checks the exact content SHA and lineage. This folder does not stage files, contact B, run the verifier, accept tensor integrity, stop the native unit, or add a fit.

The body is bound to EMA run `{RUN}`, native unit `{UNIT}`, invocation `{INVOCATION}`, and the exact original root launch acceptance. It requires the native capture's strict integer `MainPID=0`, `ExecMainCode=1`, and `ExecMainStatus=0`, then asks the unchanged EMA-specific checker to perform the full restricted-load integrity check. It does not require the native unit to be stopped before verification.

The private closure contains exactly five code files: the narrow EMA V2 checker and its four unchanged helper dependencies. The body creates only `B/research_tmp/fg19_child_ema_verify_v2`, containing those code files, the source manifest, root launch acceptance, typed terminal evidence, native raw capture, and the exact small V1 failure capture/log. It leaves V1's output and unit untouched. Original EMA admission/launch, parent/dictionary/profile/CONTROL receipts, profile files, and cache acceptance/stop are verified read-only at their existing paths.

Fresh source167 content, split, cache metadata and array stats, registered data aliases/stats, ancestor checkpoint and dictionary stats, empty population and ownership, unused one-time headroom reservation, project scratch free space and device, support-unit absence, and create-only output absence are gated before the first write. The one support unit is capped at 50% CPU, 1.4 GB MemoryHigh, 1.5 GB MemoryMax, zero swap, 600 seconds, hidden CUDA, and one BLAS thread. The runner uses the canonical project Python with `-E -B` and `WNN_PROJECT_ROOT=B/data`.

The direct-human exact dictionary-path decoder exception is already recorded in the linked authority receipt and is required before mutation. The body preserves `weights_only=True`, all existing model/state/metadata/aggregate caps, and the exact one-field Path restriction. A submit timeout or unknown result is durably retained and must never be retried.

The local binding receipts pin the original EMA root acceptance, native terminal capture, exact V1 failed support capture/log/body/checker, V2 checker and four frozen helpers, eight existing ancestry receipts, three profile metadata files, and cache receipts. Local work read only small metadata and code; model, training-state, cache-array, dictionary, and dataset payloads were not read or hashed. The V2 checker was not run; no unchanged model/source/profile suite or remote operation ran here.

The focused fixture `test_child_ema_terminal_verifier_v2.py` passes six tests, including positive validation of the exact original dictionary stat schema and rejection of stat/hash/schema drift, exact V1 failure binding, and rejection of altered V1 capture/log bytes. It runs only pure metadata predicates and does not execute the terminal checker or load model/state payloads.

`REMOTE_SUBMISSION.py` SHA-256: `{sha256(body_raw)}` ({len(body_raw)} bytes). See `ROOT_BINDING.json`, `DEPENDENCY_MANIFEST.json`, and `LOCAL_CHECKS.json` for exact pins and scope.
"""
    readme_raw = readme.encode("utf-8")
    (OUTPUT / "README.md").write_bytes(readme_raw)
    binding = {
        "schema": "fg19.child-ema-terminal-verifier-root-binding.v2",
        "status": "FROZEN_FOR_INDEPENDENT_REVIEW_NOT_ADMITTED",
        "utc": datetime.now(UTC).isoformat(),
        "run": {"name": RUN, "unit": UNIT, "invocation_id": INVOCATION},
        "native_terminal": {
            "typed_evidence_sha256": EXPECTED["TERMINAL_EVIDENCE.json"][1],
            "raw_capture_sha256": EXPECTED["NATIVE_STDOUT.raw.json"][1],
            "raw_capture_bytes": EXPECTED["NATIVE_STDOUT.raw.json"][0],
            "metadata_review_sha256": NATIVE_REVIEW_SHA,
            "strict_terminal_fields": {"main_pid": 0, "exec_main_code": 1, "exec_main_status": 0},
        },
        "actual_launch": {
            name: {"bytes": size, "sha256": sha}
            for name, size, sha in (
                (
                    "ADMISSION.json",
                    9_880,
                    "db13e5433cb0a414c1a235806220c53db4ded6c128fc5ffbda5e465dde9fde51",
                ),
                (
                    "LAUNCH.json",
                    7_404,
                    "8c43f6fb26d3f6a046304617094e266a3e1940c83a7ea56787781fa176ef133e",
                ),
                (
                    "LAUNCH_ACCEPTANCE.json",
                    EXPECTED["LAUNCH_ACCEPTANCE.json"][0],
                    EXPECTED["LAUNCH_ACCEPTANCE.json"][1],
                ),
            )
        },
        "source": {"commit": COMMIT, "file_count": 167, "manifest_sha256": MANIFEST_SHA},
        "control_predecessor": control_predecessor(),
        "path_exception_authority_sha256": PATH_AUTHORITY_SHA,
        "base_v1_checker_source_review_sha256": SOURCE_REVIEW_SHA,
        "v2_checker_review": "PENDING_INDEPENDENT_SOURCE_REVIEW",
        "superseded_v1_failure": checks["v1_failure_binding"],
        "v1_body": {"path": V1_BODY_PATH, "sha256": V1_BODY_SHA},
        "v1_checker_sha256": V1_CHECKER_SHA,
        "v1_failure_raw": {
            "bytes": EXPECTED["V1_FAILURE_STDOUT.raw.json"][0],
            "sha256": V1_FAILURE_RAW_SHA,
        },
        "v1_failure_log": {
            "bytes": EXPECTED["V1_FAILURE_UNIT_LOG.txt"][0],
            "sha256": V1_FAILURE_LOG_SHA,
        },
        "body": {
            "path": "REMOTE_SUBMISSION.py",
            "bytes": len(body_raw),
            "sha256": sha256(body_raw),
        },
        "template": {
            "path": "REMOTE_SUBMISSION_TEMPLATE.py",
            "bytes": len(stable_read(TEMPLATE, 200_000)),
            "sha256": sha256(stable_read(TEMPLATE, 200_000)),
        },
        "generator": {
            "path": "prepare_child_ema_terminal_verifier_v2.py",
            "bytes": len(stable_read(Path(__file__), 300_000)),
            "sha256": sha256(stable_read(Path(__file__), 300_000)),
        },
        "dependency_manifest": {"bytes": len(dep_raw), "sha256": sha256(dep_raw)},
        "local_checks": {"bytes": len(checks_raw), "sha256": sha256(checks_raw)},
        "readme": {"bytes": len(readme_raw), "sha256": sha256(readme_raw)},
        "focused_fixture": {
            "path": "test_child_ema_terminal_verifier_v2.py",
            "bytes": len(stable_read(FOCUSED_FIXTURE, 50_000)),
            "sha256": sha256(stable_read(FOCUSED_FIXTURE, 50_000)),
            "tests_passed": 6,
        },
        "resource_caps": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1_400_000_000,
            "memory_max_bytes": 1_500_000_000,
            "swap_bytes": 0,
            "runtime_seconds": 600,
            "cuda_hidden": True,
            "blas_threads": 1,
        },
        "submission_policy": "One root-owned admission only; this local packet does not launch. Persist raw response before parsing and never retry an unknown submission outcome.",
        "existing_predecessor_receipts_staged": False,
        "native_unit_stop_requested": False,
        "model_state_or_dataset_payloads_read_or_hashed": False,
    }
    binding_raw = (json.dumps(binding, sort_keys=True, indent=2) + "\n").encode()
    (OUTPUT / "ROOT_BINDING.json").write_bytes(binding_raw)
    print(
        json.dumps(
            {
                "body_bytes": len(body_raw),
                "body_sha256": sha256(body_raw),
                "root_binding_sha256": sha256(binding_raw),
                "dependency_manifest_sha256": sha256(dep_raw),
                "local_checks_sha256": sha256(checks_raw),
                "readme_sha256": sha256(readme_raw),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
