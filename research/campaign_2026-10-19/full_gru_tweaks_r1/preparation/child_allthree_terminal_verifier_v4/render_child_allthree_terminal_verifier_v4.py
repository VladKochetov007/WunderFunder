"""Render the V4 AllThree wrapper after exact local review and native-exit gates."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records"
OUTPUT = Path(__file__).resolve().parent
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_V4_TEMPLATE.py"
BODY = OUTPUT / "REMOTE_SUBMISSION_V4.py"
REPORT = OUTPUT / "V4_RENDER_BINDING.json"
V3_BODY = PREPARATION / "child_allthree_terminal_verifier_v3/REMOTE_SUBMISSION_V3.py"
CHECK_DIR = PREPARATION / "child_allthree_terminal_check_v2"
CHECKER = CHECK_DIR / "child_allthree_terminal_check_v2.py"
SOURCE_REVIEW = CHECK_DIR / "CHILD_ALLTHREE_TERMINAL_SOURCE_REVIEW_V2.json"
ROOT_PREPARATION = CHECK_DIR / "ROOT_PREPARATION.json"
IMPORT_PROOF = CHECK_DIR / "PRIVATE_IMPORT_BOUNDARY_PROOF_V2.json"
TERMINAL = RECORDS / "child_allthree_fit_v1/TERMINAL_EVIDENCE.json"
NATIVE_RAW = RECORDS / "child_allthree_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"

V3_BODY_BYTES = 707_678
V3_BODY_SHA = "12f357f2127abded88b0e89d5ba08323310bcc5337a980f89a2ce90b4ec57151"
CHECKER_BYTES = 40_442
CHECKER_SHA = "6ff4ba93a4020a4b34ade46a2512ac77b871328e95c47f2ca4561b8dcbd435b5"
PROOF_BYTES = 12_712
PROOF_SHA = "d590fccb9c001791cdd3656a423e424a8d0042cf3a33e2415c381b5a1d306a11"
NATIVE_SCHEMA = "fg19.child-allthree-fit-native-terminal-evidence.v1"
RAW_SCHEMA = "fg19-child-allthree-startup-capture.v1"
RUN = "fg19_child_allthree_f0_s1001_fit_v1"
NATIVE_UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
NATIVE_INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
NATIVE_RAW_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/"
    "terminal_native_capture_v1/STDOUT.raw.json"
)
MAX_METADATA_BYTES = 1_000_000
MAX_PINNED_METADATA_TOTAL = 16_000_000
MAX_NATIVE_RAW_BYTES = 2_000_000

CODE_PINS = {
    "child_allthree_terminal_check_v2.py": (CHECKER_BYTES, CHECKER_SHA),
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


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def read_regular(path: Path, cap: int) -> bytes:
    """Read one bounded regular file with no-follow and stable inode checks."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > cap:
        raise ValueError(f"unsafe, redirected, or oversized input: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"input changed before read: {path}")
        chunks: list[bytes] = []
        size = 0
        while block := os.read(descriptor, min(65_536, cap + 1 - size)):
            chunks.append(block)
            size += len(block)
            if size > cap:
                raise ValueError(f"input exceeded read cap: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) or size != before.st_size:
        raise ValueError(f"input changed during read: {path}")
    return b"".join(chunks)


def json_object(raw: bytes, label: str) -> dict[str, object]:
    """Parse bounded JSON and require a top-level object."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {label}")
    return value


def _literal_assignment(tree: ast.Module, name: str) -> object:
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError(f"frozen V3 source has no unique {name} assignment")
    return ast.literal_eval(matches[0])


def _find_pin(value: object, path: str, size: int, digest: str) -> bool:
    if isinstance(value, dict):
        if value.get("path") == path and (value.get("bytes"), value.get("sha256")) == (
            size,
            digest,
        ):
            return True
        return any(_find_pin(item, path, size, digest) for item in value.values())
    if isinstance(value, list):
        return any(_find_pin(item, path, size, digest) for item in value)
    return False


def _check_native(terminal_raw: bytes, capture_raw: bytes) -> dict[str, object]:
    """Require exact typed exit identity, raw-first capture, and strict unit status."""
    if len(terminal_raw) != 717 or sha256(terminal_raw) != (
        "da91b782652e98843fdadeadcb41ab2c3f67492ead0e95c108ef18c85916f27a"
    ):
        raise ValueError("actual typed native terminal evidence pin differs")
    if len(capture_raw) != 82_591 or sha256(capture_raw) != (
        "0d9ba24a1e241ee22090247711f5e875def72faf5b052cf149d227cdca2f262a"
    ):
        raise ValueError("actual raw-first native capture pin differs")
    terminal = json_object(terminal_raw, "actual typed native terminal")
    if set(terminal) != {
        "schema",
        "run_name",
        "unit",
        "invocation_id",
        "active_state",
        "sub_state",
        "main_pid",
        "exec_main_code",
        "exec_main_status",
        "raw_path",
        "raw_bytes",
        "raw_sha256",
        "captured_utc",
        "native_exit_utc",
    }:
        raise ValueError("typed native terminal evidence keys differ")
    if tuple(terminal.get(key) for key in ("schema", "run_name", "unit", "invocation_id")) != (
        NATIVE_SCHEMA,
        RUN,
        NATIVE_UNIT,
        NATIVE_INVOCATION,
    ):
        raise ValueError("typed native terminal identity differs")
    if tuple(terminal.get(key) for key in ("active_state", "sub_state")) != (
        "active",
        "exited",
    ):
        raise ValueError("native fit is not active/exited")
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        if type(terminal.get(key)) is not int or terminal[key] != expected:
            raise ValueError(f"native {key} is not strict integer {expected}")
    if (
        type(terminal.get("raw_bytes")) is not int
        or terminal.get("raw_bytes") != len(capture_raw)
        or terminal.get("raw_sha256") != sha256(capture_raw)
        or terminal.get("raw_path") != NATIVE_RAW_PATH
    ):
        raise ValueError("typed-to-raw native capture join differs")
    capture = json_object(capture_raw, "actual raw-first native capture")
    if tuple(
        capture.get(key)
        for key in (
            "schema",
            "expected_run",
            "expected_unit",
            "expected_invocation_id",
            "captured_utc",
        )
    ) != (
        RAW_SCHEMA,
        RUN,
        NATIVE_UNIT,
        NATIVE_INVOCATION,
        terminal.get("captured_utc"),
    ):
        raise ValueError("raw-first native capture identity differs")
    properties = capture.get("unit_properties")
    if not isinstance(properties, dict) or tuple(
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
    ) != (NATIVE_INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("raw native systemd properties differ")
    stamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        stamp = terminal.get(key)
        if not isinstance(stamp, str):
            raise ValueError(f"typed native evidence omits {key}")
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"typed native {key} has no timezone")
        stamps[key] = parsed
    if stamps["native_exit_utc"] > stamps["captured_utc"]:
        raise ValueError("native exit follows its capture")
    return {
        "typed_bytes": len(terminal_raw),
        "typed_sha256": sha256(terminal_raw),
        "raw_bytes": len(capture_raw),
        "raw_sha256": sha256(capture_raw),
        "raw_path": NATIVE_RAW_PATH,
        "captured_utc": terminal["captured_utc"],
        "native_exit_utc": terminal["native_exit_utc"],
        "unit": NATIVE_UNIT,
        "invocation_id": NATIVE_INVOCATION,
        "active_state": "active",
        "sub_state": "exited",
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
    }


def _check_review_and_proof(root: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Verify the V2 checker source review and pure private-import proof pins."""
    checker = read_regular(CHECKER, 100_000)
    if (len(checker), sha256(checker)) != (CHECKER_BYTES, CHECKER_SHA):
        raise ValueError("V2 checker source pin differs")
    review_pin = root.get("source_review")
    if not isinstance(review_pin, dict) or review_pin.get("path") != str(
        SOURCE_REVIEW.relative_to(REPOSITORY)
    ):
        raise ValueError("V2 root preparation source-review identity differs")
    review_raw = read_regular(SOURCE_REVIEW, MAX_METADATA_BYTES)
    if (len(review_raw), sha256(review_raw)) != (
        review_pin.get("bytes"),
        review_pin.get("sha256"),
    ) or review_pin.get("status") != "PASS_SOURCE_ONLY":
        raise ValueError("V2 checker source-review pin or status differs")
    review = json_object(review_raw, "V2 checker source review")
    if (
        review.get("status") != "PASS_SOURCE_ONLY"
        or review.get("blocking_findings")
        not in (
            None,
            [],
        )
        or not _find_pin(
            review,
            str(CHECKER.relative_to(REPOSITORY)),
            CHECKER_BYTES,
            CHECKER_SHA,
        )
    ):
        raise ValueError("V2 source review does not approve exact checker bytes")
    proof_pin = root.get("accepted_import_boundary_proof")
    if not isinstance(proof_pin, dict) or proof_pin.get("path") != str(
        IMPORT_PROOF.relative_to(REPOSITORY)
    ):
        raise ValueError("V2 root preparation proof path differs")
    proof_raw = read_regular(IMPORT_PROOF, MAX_METADATA_BYTES)
    if (len(proof_raw), sha256(proof_raw)) != (PROOF_BYTES, PROOF_SHA) or (
        proof_pin.get("bytes"),
        proof_pin.get("sha256"),
    ) != (PROOF_BYTES, PROOF_SHA):
        raise ValueError("V2 private-import proof pin differs")
    proof = json_object(proof_raw, "V2 private-import proof")
    if (proof.get("schema"), proof.get("status")) != (
        "fg19.child-allthree-terminal-import-boundary-proof.v2",
        "PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V2",
    ):
        raise ValueError("V2 private-import proof status differs")
    expected_pins = {
        name: {"bytes": size, "sha256": digest} for name, (size, digest) in CODE_PINS.items()
    }
    private = proof.get("private_closure")
    if not isinstance(private, dict) or any(
        private.get(key) != expected_pins
        for key in (
            "pre_copy_source_pins",
            "temporary_copy_pins_before_child_import",
            "child_pre_import_pins",
            "child_post_import_pins",
        )
    ):
        raise ValueError("V2 private-import proof closure differs")
    if (
        private.get("child_preloaded_module_names") != []
        or private.get("exact_file_count") != 7
        or set(private.get("exact_file_set_before_and_after_import", [])) != set(CODE_PINS)
        or private.get("helper_closure_recheck")
        != {
            name: pin
            for name, pin in expected_pins.items()
            if name != "child_allthree_terminal_check_v2.py"
        }
    ):
        raise ValueError("V2 private-import proof preloaded-module or helper scope differs")
    origins = private.get("child_module_origins")
    if not isinstance(origins, dict) or set(origins) != set(CODE_PINS):
        raise ValueError("V2 private-import proof module origins are incomplete")
    entrypoint = proof.get("entrypoint_scope")
    if not isinstance(entrypoint, dict) or (
        entrypoint.get("checker_functions_called"),
        entrypoint.get("verify_terminal_called"),
        entrypoint.get("main_or_cli_called"),
        entrypoint.get("fit_or_evaluation_called"),
        entrypoint.get("source_cli_run_identity_representation_or_model_imports_exercised"),
        entrypoint.get("wnn_source_modules_loaded"),
    ) != (["assert_helper_closure"], False, False, False, False, 0):
        raise ValueError("V2 private-import proof crossed its pure-helper boundary")
    guard = proof.get("protected_open_guard")
    cuda = proof.get("cuda_guard")
    if (
        not isinstance(guard, dict)
        or guard.get("attempts") != 0
        or guard.get("model_or_state_dictionary_cache_data_payload_read_or_hashed") is not False
        or any(value != 0 for value in guard.get("attempts_by_category", {}).values())
    ):
        raise ValueError("V2 private-import proof observed a protected-payload open")
    if (
        not isinstance(cuda, dict)
        or cuda.get("guarded_action_attempts") != 0
        or cuda.get("initialized_before_and_after") is not False
        or cuda.get("observer_is_separate_from_action_counter") is not True
    ):
        raise ValueError("V2 private-import proof observed a guarded CUDA action")
    return (
        {
            "path": str(SOURCE_REVIEW.relative_to(REPOSITORY)),
            "bytes": len(review_raw),
            "sha256": sha256(review_raw),
            "status": "PASS_SOURCE_ONLY",
        },
        {
            "path": str(IMPORT_PROOF.relative_to(REPOSITORY)),
            "bytes": len(proof_raw),
            "sha256": sha256(proof_raw),
            "schema": proof["schema"],
            "status": proof["status"],
        },
    )


def _check_root_preparation() -> tuple[dict[str, object], dict[str, object]]:
    """Require root-owned V2 preparation and recheck its bounded metadata pins."""
    root_raw = read_regular(ROOT_PREPARATION, MAX_METADATA_BYTES)
    root = json_object(root_raw, "V2 root preparation")
    identity = root.get("identity")
    expected_identity = {
        "run_name": RUN,
        "unit": NATIVE_UNIT,
        "invocation_id": NATIVE_INVOCATION,
        "source_commit": "68ff682977b822f38351aac0f621146f5f13b440",
        "source_files": 167,
        "source_manifest_sha256": "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
    }
    if (root.get("schema"), root.get("status"), identity) != (
        "fg19.child-allthree-terminal-root-preparation.v2",
        "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED",
        expected_identity,
    ):
        raise ValueError("V2 root preparation status or identity differs")
    if (
        root.get("dependency_binding_condition_satisfied") is not True
        or (root.get("native_terminal_evidence_present") is not True)
        or root.get("actual_B_staging_admission_or_execution") is not False
    ):
        raise ValueError("V2 root preparation is incomplete or claims B execution")
    entries = root.get("code_dependencies")
    if not isinstance(entries, list) or len(entries) != len(CODE_PINS):
        raise ValueError("V2 root preparation code closure is not exactly seven files")
    observed: dict[str, tuple[int, str]] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("malformed V2 code dependency entry")
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("V2 code dependency path escapes the repository")
        absolute = REPOSITORY / path
        name = absolute.name
        raw = read_regular(absolute, 100_000)
        pin = (len(raw), sha256(raw))
        if (
            name in observed
            or pin != CODE_PINS.get(name)
            or (
                item.get("bytes"),
                item.get("sha256"),
            )
            != pin
        ):
            raise ValueError(f"V2 code dependency pin differs: {name}")
        observed[name] = pin
    if set(observed) != set(CODE_PINS):
        raise ValueError("V2 code dependency names differ")
    metadata = root.get("metadata_and_proof_pins")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("V2 root preparation omits metadata/proof pins")
    total = 0
    hashes: set[str] = set()
    metadata_by_path: dict[str, tuple[int, str]] = {}
    for item in metadata:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("malformed V2 metadata/proof pin")
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower()
            in {
                ".pt",
                ".pth",
                ".npy",
                ".npz",
                ".parquet",
            }
        ):
            raise ValueError("V2 metadata pin names an unsafe path or payload")
        raw = read_regular(REPOSITORY / relative, MAX_METADATA_BYTES)
        total += len(raw)
        if item["path"] in metadata_by_path:
            raise ValueError(f"duplicate V2 metadata/proof pin: {relative}")
        if total > MAX_PINNED_METADATA_TOTAL or (item.get("bytes"), item.get("sha256")) != (
            len(raw),
            sha256(raw),
        ):
            raise ValueError(f"V2 metadata/proof pin differs: {relative}")
        hashes.add(sha256(raw))
        metadata_by_path[item["path"]] = (len(raw), sha256(raw))
    ancestry = root.get("readonly_ancestry_receipt_sha256")
    if (
        not isinstance(ancestry, dict)
        or len(ancestry) != 14
        or not set(ancestry.values()) <= hashes
    ):
        raise ValueError("V2 root preparation does not preserve the fourteen read-only receipts")
    review_for_pin = root.get("source_review")
    if not isinstance(review_for_pin, dict):
        raise ValueError("V2 source-review pin is malformed")
    required_metadata = {
        str(TERMINAL.relative_to(REPOSITORY)): (
            717,
            "da91b782652e98843fdadeadcb41ab2c3f67492ead0e95c108ef18c85916f27a",
        ),
        str(NATIVE_RAW.relative_to(REPOSITORY)): (
            82_591,
            "0d9ba24a1e241ee22090247711f5e875def72faf5b052cf149d227cdca2f262a",
        ),
        str(SOURCE_REVIEW.relative_to(REPOSITORY)): (
            review_for_pin.get("bytes"),
            review_for_pin.get("sha256"),
        ),
        str(IMPORT_PROOF.relative_to(REPOSITORY)): (PROOF_BYTES, PROOF_SHA),
    }
    if any(metadata_by_path.get(path) != pin for path, pin in required_metadata.items()):
        raise ValueError("V2 root-preparation pins omit exact review, proof, or native bytes")
    if root.get("support_unit") != "computer-vision-homework-fg19-child-allthree-verify-v2.service":
        raise ValueError("V2 root preparation binds another support unit")
    if root.get("support_private_directory") != (
        "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_allthree_verify_v2"
    ):
        raise ValueError("V2 root preparation binds another private directory")
    if root.get("native_terminal_evidence") != {
        "bytes": 717,
        "path": str(TERMINAL.relative_to(REPOSITORY)),
        "sha256": "da91b782652e98843fdadeadcb41ab2c3f67492ead0e95c108ef18c85916f27a",
    } or root.get("native_raw_capture") != {
        "bytes": 82_591,
        "path": str(NATIVE_RAW.relative_to(REPOSITORY)),
        "sha256": "0d9ba24a1e241ee22090247711f5e875def72faf5b052cf149d227cdca2f262a",
    }:
        raise ValueError("V2 root preparation native-evidence pins differ")
    failed = root.get("prior_failed_supports")
    if not isinstance(failed, list) or not any(
        isinstance(item, dict)
        and item.get("unit") == "computer-vision-homework-fg19-child-allthree-verify-v1.service"
        and item.get("invocation_id") == "f0c857aa1a68498cb4be237d423281cf"
        and item.get("action") == "PRESERVE_FAILED_NEVER_STOP_RETRY_RESTART_POLL"
        for item in failed
    ):
        raise ValueError("V2 root preparation does not preserve the failed V1 support invocation")
    return root, {
        "path": str(ROOT_PREPARATION.relative_to(REPOSITORY)),
        "bytes": len(root_raw),
        "sha256": sha256(root_raw),
        "status": root.get("status"),
    }


def _build_payload(
    native_terminal: dict[str, object], root: dict[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, int | str]]]:
    """Reuse exact V3 metadata, replace only the checker with reviewed V2 bytes."""
    v3_raw = read_regular(V3_BODY, V3_BODY_BYTES)
    if (len(v3_raw), sha256(v3_raw)) != (V3_BODY_BYTES, V3_BODY_SHA):
        raise ValueError("frozen V3 wrapper source pin differs")
    tree = ast.parse(v3_raw.decode("utf-8"))
    old_payload = _literal_assignment(tree, "PAYLOAD")
    old_fixed = _literal_assignment(tree, "FIXED_PAYLOAD_PINS")
    if not isinstance(old_payload, dict) or not isinstance(old_fixed, dict):
        raise ValueError("frozen V3 payload literals are malformed")
    if set(old_payload) != set(old_fixed) or len(old_payload) != 15:
        raise ValueError("frozen V3 payload keys differ")
    payload: dict[str, bytes] = {}
    for name, item in old_payload.items():
        if not isinstance(item, dict) or not isinstance(item.get("base64"), str):
            raise ValueError(f"frozen V3 payload record is malformed: {name}")
        raw = base64.b64decode(item["base64"], validate=True)
        if (len(raw), sha256(raw)) != (item.get("bytes"), item.get("sha256")):
            raise ValueError(f"frozen V3 embedded payload pin differs: {name}")
        fixed_item = old_fixed.get(name)
        if not isinstance(fixed_item, dict) or (
            fixed_item.get("bytes"),
            fixed_item.get("sha256"),
        ) != (len(raw), sha256(raw)):
            raise ValueError(f"frozen V3 fixed payload pin differs: {name}")
        payload[name] = raw
    payload.pop("child_allthree_terminal_check_v1.py", None)
    admission = json_object(payload["ADMISSION.json"], "frozen AllThree admission")
    receipts = admission.get("staged_receipt_sha256")
    if (
        not isinstance(receipts, dict)
        or len(receipts) != 14
        or root.get("readonly_ancestry_receipt_sha256") != receipts
    ):
        raise ValueError("frozen V3 and V2 root-preparation receipt maps differ")
    checker_raw = read_regular(CHECKER, 100_000)
    if (len(checker_raw), sha256(checker_raw)) != (CHECKER_BYTES, CHECKER_SHA):
        raise ValueError("V2 checker differs from its reviewed code pin")
    terminal_raw = read_regular(TERMINAL, 16_000)
    capture_raw = read_regular(NATIVE_RAW, MAX_NATIVE_RAW_BYTES)
    if (len(terminal_raw), sha256(terminal_raw), len(capture_raw), sha256(capture_raw)) != (
        native_terminal["typed_bytes"],
        native_terminal["typed_sha256"],
        native_terminal["raw_bytes"],
        native_terminal["raw_sha256"],
    ):
        raise ValueError("native evidence changed after strict proof validation")
    payload["child_allthree_terminal_check_v2.py"] = checker_raw
    payload["TERMINAL_EVIDENCE.json"] = terminal_raw
    payload["NATIVE_STDOUT.raw.json"] = capture_raw
    code_names = set(CODE_PINS)
    expected_names = code_names | {
        "ADMISSION.json",
        "ALLTHREE_LAUNCH_STDOUT.raw.json",
        "LAUNCH.json",
        "LAUNCH_ACCEPTANCE.json",
        "NATIVE_STDOUT.raw.json",
        "SOURCE_MANIFEST.json",
        "TERMINAL_EVIDENCE.json",
        "TRANSPORT.json",
    }
    if set(payload) != expected_names:
        raise ValueError("V4 packaged files differ from the fixed fifteen-file set")
    packaged = {
        name: {
            "bytes": len(raw),
            "sha256": sha256(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in sorted(payload.items())
    }
    fixed = {
        name: {"bytes": len(raw), "sha256": sha256(raw)} for name, raw in sorted(payload.items())
    }
    return packaged, fixed


def _write_exclusive(path: Path, raw: bytes) -> None:
    """Create and sync one reviewed output without replacing existing files."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def render() -> dict[str, object]:
    """Create the actual wrapper only after root, source, proof, and native gates pass."""
    if BODY.exists() or BODY.is_symlink() or REPORT.exists() or REPORT.is_symlink():
        raise FileExistsError("V4 output already exists; renderer is create-only")
    template_raw = read_regular(TEMPLATE, 100_000)
    template = template_raw.decode("utf-8")
    placeholders = ("PAYLOAD = {}", "FIXED_PAYLOAD_PINS = {}", "FINAL_REVIEW_BINDING = {}")
    if any(template.count(value) != 1 for value in placeholders):
        raise ValueError("V4 template must contain exactly three empty binding slots")
    root, root_pin = _check_root_preparation()
    review_pin, proof_pin = _check_review_and_proof(root)
    terminal_raw = read_regular(TERMINAL, 16_000)
    capture_raw = read_regular(NATIVE_RAW, MAX_NATIVE_RAW_BYTES)
    native_terminal = _check_native(terminal_raw, capture_raw)
    packaged, fixed = _build_payload(native_terminal, root)
    checker_pin = {
        "path": str(CHECKER.relative_to(REPOSITORY)),
        "bytes": CHECKER_BYTES,
        "sha256": CHECKER_SHA,
    }
    binding = {
        "schema": "fg19.child-allthree-terminal-verifier-render-binding.v4",
        "status": "ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND",
        "checker": checker_pin,
        "source_review": review_pin,
        "root_preparation": root_pin,
        "private_import_boundary_proof": proof_pin,
        "native_terminal": native_terminal,
        "body_template": {
            "path": str(TEMPLATE.relative_to(REPOSITORY)),
            "bytes": len(template_raw),
            "sha256": sha256(template_raw),
        },
        "frozen_v3_source": {
            "path": str(V3_BODY.relative_to(REPOSITORY)),
            "bytes": V3_BODY_BYTES,
            "sha256": V3_BODY_SHA,
        },
        "metadata_and_proof_pin_count": len(root["metadata_and_proof_pins"]),
    }
    body = template.replace("PAYLOAD = {}", "PAYLOAD = " + repr(packaged), 1)
    body = body.replace("FIXED_PAYLOAD_PINS = {}", "FIXED_PAYLOAD_PINS = " + repr(fixed), 1)
    body = body.replace("FINAL_REVIEW_BINDING = {}", "FINAL_REVIEW_BINDING = " + repr(binding), 1)
    ast.parse(body)
    body_raw = body.encode("utf-8")
    report = {
        "schema": "fg19.child-allthree-terminal-verifier-v4-local-render.v1",
        "status": "RENDERED_CREATE_ONLY_LOCAL_BODY_NOT_STAGED_ADMITTED_OR_EXECUTED",
        "body_path": str(BODY.relative_to(REPOSITORY)),
        "body_bytes": len(body_raw),
        "body_sha256": sha256(body_raw),
        "support_unit": "computer-vision-homework-fg19-child-allthree-verify-v2.service",
        "private_output_directory": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_allthree_verify_v2",
        "checker": checker_pin,
        "source_review": review_pin,
        "root_preparation": root_pin,
        "private_import_boundary_proof": proof_pin,
        "native_terminal": native_terminal,
        "payload_count": len(packaged),
        "code_count": len(CODE_PINS),
        "body_executed_or_imported": False,
        "remote_or_B_action": False,
    }
    _write_exclusive(BODY, body_raw)
    _write_exclusive(REPORT, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    return report


if __name__ == "__main__":
    print(json.dumps(render(), indent=2, sort_keys=True, allow_nan=False))
