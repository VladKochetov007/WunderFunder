"""Bind local dictionary-fit receipts and report terminal-verifier readiness."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent
RECORDS = CAMPAIGN / "records"
FIT_RECORDS = RECORDS / "child_dictionary_fit_v1"
TEMPLATE_PATH = OUTPUT / "ADMISSION_TEMPLATE_V1.json"
DEPENDENCY_MANIFEST_PATH = OUTPUT / "DEPENDENCY_MANIFEST.json"
LOCAL_CHECKS_PATH = OUTPUT / "LOCAL_CHECKS.json"
BODY_TEMPLATE_PATH = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY_RENDERER_PATH = OUTPUT / "render_child_dictionary_terminal_verifier_v1.py"
BODY_BUILDER_PATH = OUTPUT / "build_dictionary_terminal_body_template_v1.py"
BODY_OUTPUT_PATH = OUTPUT / "REMOTE_SUBMISSION.py"
EMA_V3_TEMPLATE_PATH = (
    CAMPAIGN / "preparation/child_ema_terminal_verifier_v3/REMOTE_SUBMISSION_TEMPLATE.py"
)
EMA_V3_TEMPLATE_BYTES = 59_676
EMA_V3_TEMPLATE_SHA256 = "31bc9986f3d6e0728d36a723cc954899727150afc9d965cf91993c7c0d3f1a12"

RUN = "fg19_child_dictionary_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
INVOCATION = "73b10b699e034a1b9952b2377b845095"
CHECKER_BYTES = 34_164
CHECKER_SHA256 = "d06818648362d571e3b06637bc026b97190476a6b59474f4a57ec84c349e4d82"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
ADMISSION_SHA256 = "b13ccf5834fd210ce02ca6c3aea7251d8dcae3f63823341da15a77abcbc75063"
LAUNCH_SHA256 = "9cbcc8b4df3886620429cae42d678d2e281aff980402f5b19d63d9f515b6ad3c"
LAUNCH_ACCEPTANCE_SHA256 = "33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95"
BODY_SHA256 = "9f545489dd80c13831b1a02c75dddf4059f3cbfb84311b2400db4830b4fdd77e"
BODY_BYTES = 196_111
SOURCE_REVIEW_SHA256 = "67528bf25430a7bd422e233f3ce84059b14d372f59b9b033cdd70e674634528a"
SOURCE_REVIEW_BYTES = 7_213
ROOT_PREPARATION_SHA256 = "2c0ee4b298a744f3c3204c46f5a269f3ea03c73d1a521c6db05dfa7a4144cf66"
ROOT_PREPARATION_BYTES = 2_343

PARENT_ACCEPTANCE_SHA256 = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
PARENT_STOP_SHA256 = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
DICTIONARY_ACCEPTANCE_SHA256 = "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
DICTIONARY_STOP_SHA256 = "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158"
PROFILE_ACCEPTANCE_SHA256 = "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045"
PROFILE_STOP_SHA256 = "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce"
CONTROL_ACCEPTANCE_SHA256 = "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
CONTROL_STOP_SHA256 = "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
EMA_ACCEPTANCE_SHA256 = "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8"
EMA_STOP_SHA256 = "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf"
CACHE_ACCEPTANCE_SHA256 = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_STOP_SHA256 = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"

EXPECTED_FILES = {
    "ADMISSION.json": (10_937, ADMISSION_SHA256),
    "LAUNCH.json": (8_072, LAUNCH_SHA256),
    "LAUNCH_ACCEPTANCE.json": (2_674, LAUNCH_ACCEPTANCE_SHA256),
}
ACCEPTANCE_SUPPORT_FILES = {
    "source_review": (
        CAMPAIGN
        / "preparation/child_dictionary_fit_v1/DICTIONARY_FINAL_BODY_SOURCE_REVIEW_V2.json",
        SOURCE_REVIEW_BYTES,
        SOURCE_REVIEW_SHA256,
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/DICTIONARY_FINAL_BODY_SOURCE_REVIEW_V2.json",
    ),
    "root_preparation": (
        FIT_RECORDS / "ROOT_ADMISSION_PREPARATION.json",
        ROOT_PREPARATION_BYTES,
        ROOT_PREPARATION_SHA256,
        "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/ROOT_ADMISSION_PREPARATION.json",
    ),
}
NATIVE_TERMINAL_PATH = FIT_RECORDS / "TERMINAL_EVIDENCE.json"
NATIVE_CAPTURE_PATH = FIT_RECORDS / "terminal_native_capture_v1/STDOUT.raw.json"
NATIVE_TERMINAL_SCHEMA = "fg19.child-dictionary-fit-native-terminal-evidence.v1"
NATIVE_CAPTURE_SCHEMA = "fg19-child-control-startup-capture.v2"
DICTIONARY_CHECKER_PATH = (
    CAMPAIGN
    / "preparation/child_dictionary_terminal_check_v1/child_dictionary_terminal_check_v1.py"
)
HELPER_INPUTS = {
    "child_control_terminal_check_v3.py": (
        CAMPAIGN
        / "preparation/child_control_terminal_verifier_v3/child_control_terminal_check_v3.py",
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
    ),
    "parent_fit_terminal_check_v3.py": (
        CAMPAIGN / "preparation/parent_fit_terminal_check_v3.py",
        64_550,
        "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    ),
    "propensity_terminal_support.py": (
        CAMPAIGN / "preparation/propensity_terminal_support.py",
        13_107,
        "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    ),
    "teacher_s5_terminal_check_v1.py": (
        CAMPAIGN / "preparation/teacher_s5_terminal_check_v1.py",
        37_448,
        "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    ),
    "child_ema_terminal_check_v3.py": (
        CAMPAIGN / "preparation/child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py",
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
    ),
}


def _sha256(raw: bytes) -> str:
    """Return the lowercase digest of exact input bytes."""
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, cap: int) -> bytes:
    """Read one bounded regular file without following a final symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > cap:
        raise ValueError(f"unsafe or oversized local metadata file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"local metadata changed before read: {path}")
        chunks: list[bytes] = []
        remaining = cap + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > cap:
        raise ValueError(f"local metadata size changed during read: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"local metadata changed during read: {path}")
    return raw


def _object(raw: bytes, label: str) -> dict[str, object]:
    """Parse a small JSON object without rewriting its source bytes."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def _receipt(
    path: Path, expected_bytes: int, expected_sha256: str
) -> tuple[bytes, dict[str, object]]:
    raw = _read_regular(path, expected_bytes)
    if len(raw) != expected_bytes or _sha256(raw) != expected_sha256:
        raise ValueError(f"pinned dictionary receipt differs: {path.name}")
    return raw, _object(raw, path.name)


def _control_predecessor() -> dict[str, object]:
    return {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA256,
        "exact_stop_sha256": CONTROL_STOP_SHA256,
        "invocation_id": "e66e0d0703314e4fb5d47dcdbfcaf202",
        "run_name": "fg19_child_control_f0_s1001_fit_v1",
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": "computer-vision-homework-fg19-child-control-fit-v1.service",
    }


def _ema_predecessor() -> dict[str, object]:
    return {
        "acceptance_sha256": EMA_ACCEPTANCE_SHA256,
        "exact_stop_sha256": EMA_STOP_SHA256,
        "invocation_id": "da5baf82d22b4f02890e184193b8caf7",
        "model_ep5_sha256": "6e90bc4dba8add0caf670097519f08f3ce239d0b64c37c47b3760aa22af41862",
        "run_name": "fg19_child_ema_f0_s1001_fit_v1",
        "selected_epoch": 5,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "successful_updates": 44_760,
        "unit": "computer-vision-homework-fg19-child-ema-fit-v1.service",
    }


def _check_original_receipts() -> dict[str, object]:
    """Verify the actual dictionary launch and its separate root acceptance."""
    loaded: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for name, (byte_count, digest) in EXPECTED_FILES.items():
        raw, value = _receipt(FIT_RECORDS / name, byte_count, digest)
        loaded[name] = value
        hashes[name] = _sha256(raw)

    admission = loaded["ADMISSION.json"]
    launch = loaded["LAUNCH.json"]
    acceptance = loaded["LAUNCH_ACCEPTANCE.json"]
    if (
        admission.get("schema"),
        admission.get("run_name"),
        admission.get("unit"),
        admission.get("source_commit"),
        admission.get("source_files_verified"),
        admission.get("source_manifest_sha256"),
        admission.get("split_sha256"),
    ) != (
        "fg19-child-dictionary-fit-admission.v1",
        RUN,
        UNIT,
        SOURCE_COMMIT,
        167,
        SOURCE_MANIFEST_SHA256,
        SPLIT_SHA256,
    ):
        raise ValueError("original dictionary admission identity differs")
    recipe = admission.get("recipe")
    if not isinstance(recipe, dict) or (
        recipe.get("representation_mode"),
        recipe.get("objective"),
        recipe.get("seed"),
        recipe.get("epochs"),
        recipe.get("checkpoint_epoch_zero_based"),
        recipe.get("expected_optimizer_updates"),
    ) != ("sparse", "wmse", 1001, 6, 5, 44_760):
        raise ValueError("original dictionary sparse/WMSE recipe differs")
    if admission.get("staged_receipt_sha256") != {
        "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE_SHA256,
        "PARENT_EXACT_STOP.json": PARENT_STOP_SHA256,
        "DICTIONARY_ACCEPTANCE.json": DICTIONARY_ACCEPTANCE_SHA256,
        "DICTIONARY_EXACT_STOP.json": DICTIONARY_STOP_SHA256,
        "PROFILE_TERMINAL_ACCEPTANCE.json": PROFILE_ACCEPTANCE_SHA256,
        "PROFILE_EXACT_STOP.json": PROFILE_STOP_SHA256,
        "CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256,
        "CONTROL_EXACT_STOP.json": CONTROL_STOP_SHA256,
        "EMA_TERMINAL_ACCEPTANCE.json": EMA_ACCEPTANCE_SHA256,
        "EMA_EXACT_STOP.json": EMA_STOP_SHA256,
    }:
        raise ValueError("original staged predecessor receipt map differs")
    if (
        launch.get("schema"),
        launch.get("unit"),
        launch.get("control_predecessor"),
        launch.get("ema_predecessor"),
    ) != (
        "fg19-child-dictionary-fit-launch.v1",
        UNIT,
        _control_predecessor(),
        _ema_predecessor(),
    ):
        raise ValueError("original dictionary launch or predecessor identity differs")
    submission = launch.get("submission")
    if not isinstance(submission, dict) or (
        submission.get("outcome"),
        submission.get("returncode"),
    ) != ("submitted", 0):
        raise ValueError("original dictionary launch was not one successful submission")

    expected_acceptance_refs = {
        "admission": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/ADMISSION.json",
            "bytes": EXPECTED_FILES["ADMISSION.json"][0],
            "sha256": ADMISSION_SHA256,
        },
        "launch": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/LAUNCH.json",
            "bytes": EXPECTED_FILES["LAUNCH.json"][0],
            "sha256": LAUNCH_SHA256,
        },
        "body": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/REMOTE_SUBMISSION.py",
            "bytes": BODY_BYTES,
            "sha256": BODY_SHA256,
        },
    }
    if (
        acceptance.get("schema"),
        acceptance.get("status"),
        acceptance.get("run_name"),
        acceptance.get("unit"),
        acceptance.get("invocation_id"),
    ) != (
        "fg19-child-dictionary-fit-launch-acceptance.v1",
        "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
        RUN,
        UNIT,
        INVOCATION,
    ):
        raise ValueError("original dictionary root acceptance identity differs")
    for key, expected in expected_acceptance_refs.items():
        if acceptance.get(key) != expected:
            raise ValueError(f"root acceptance {key} nested pin differs")
    for key, (path, byte_count, digest, relative_path) in ACCEPTANCE_SUPPORT_FILES.items():
        raw = _read_regular(path, byte_count)
        expected = {"path": relative_path, "bytes": byte_count, "sha256": digest}
        if len(raw) != byte_count or _sha256(raw) != digest or acceptance.get(key) != expected:
            raise ValueError(f"root acceptance {key} support pin differs")
    if (
        acceptance.get("native_exit_observed"),
        acceptance.get("expected_terminal_updates_not_observed"),
        acceptance.get("representation_mode"),
        acceptance.get("mean_variance_objective"),
        acceptance.get("quality_score"),
        acceptance.get("source_commit"),
        acceptance.get("source_file_count"),
        acceptance.get("source_manifest_sha256"),
        acceptance.get("parent_initializer"),
    ) != (
        False,
        44_760,
        "sparse",
        "wmse",
        None,
        SOURCE_COMMIT,
        167,
        SOURCE_MANIFEST_SHA256,
        "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
    ):
        raise ValueError("root launch acceptance was misread as terminal evidence")
    return {
        "status": "PASS_ORIGINAL_RECEIPTS",
        "hashes": hashes,
        "body_pin_source": "root LAUNCH_ACCEPTANCE nested body object",
        "launch_body_sha256_field_assumed": False,
        "native_terminal_inferred": False,
    }


def _check_code_inputs() -> dict[str, object]:
    """Verify the five frozen helper bytes before any future subprocess use."""
    observed: dict[str, dict[str, object]] = {}
    for name, (path, expected_bytes, expected_sha256) in HELPER_INPUTS.items():
        raw = _read_regular(path, expected_bytes)
        digest = _sha256(raw)
        if len(raw) != expected_bytes or digest != expected_sha256:
            raise ValueError(f"frozen helper identity differs: {name}")
        observed[name] = {"bytes": len(raw), "sha256": digest, "status": "PINNED"}

    if DICTIONARY_CHECKER_PATH.exists():
        checker_raw = _read_regular(DICTIONARY_CHECKER_PATH, 60_000)
        checker_sha256 = _sha256(checker_raw)
        if (len(checker_raw), checker_sha256) != (CHECKER_BYTES, CHECKER_SHA256):
            raise ValueError("final dictionary checker differs from the root-prepared pin")
        checker = {
            "bytes": len(checker_raw),
            "sha256": checker_sha256,
            "status": "PASS_SOURCE_ONLY",
        }
    else:
        checker = {"status": "MISSING_WAITING_FOR_CHECKER"}
    return {
        "frozen_helpers": observed,
        "dictionary_checker": checker,
        "outer_boundary": "Verify exact six-file private closure lengths and hashes before launching Python; checker-side post-import checks do not replace this pre-launch gate.",
        "files_to_stage": ["child_dictionary_terminal_check_v1.py", *HELPER_INPUTS],
        "stage_only_after_native_exit0_and_final_source_review": True,
    }


def _check_dependency_manifest() -> dict[str, object]:
    """Require the manifest's exact six-name closure and frozen helper pins."""
    raw = _read_regular(DEPENDENCY_MANIFEST_PATH, 16_000)
    manifest = _object(raw, "dependency manifest")
    closure = manifest.get("private_code_closure")
    if (
        manifest.get("schema"),
        manifest.get("private_code_closure_count"),
        isinstance(closure, dict),
    ) != ("fg19.child-dictionary-terminal-verifier-dependencies.v1", 6, True):
        raise ValueError("dependency manifest identity or closure size differs")
    expected_names = {"child_dictionary_terminal_check_v1.py", *HELPER_INPUTS}
    if set(closure) != expected_names:
        raise ValueError("dependency manifest module membership differs")
    for name, (path, byte_count, digest) in HELPER_INPUTS.items():
        relative = path.relative_to(CAMPAIGN)
        expected = {
            "path": relative.as_posix(),
            "bytes": byte_count,
            "sha256": digest,
            "status": "frozen",
        }
        if closure.get(name) != expected:
            raise ValueError(f"dependency manifest helper pin differs: {name}")
    checker = closure.get("child_dictionary_terminal_check_v1.py")
    if not isinstance(checker, dict) or (
        checker.get("path"),
        checker.get("bytes"),
        checker.get("sha256"),
        checker.get("status"),
    ) != (
        "preparation/child_dictionary_terminal_check_v1/child_dictionary_terminal_check_v1.py",
        CHECKER_BYTES,
        CHECKER_SHA256,
        "source_reviewed",
    ):
        raise ValueError(
            "dictionary checker pin differs from the final reviewed dependency manifest"
        )
    boundary = manifest.get("prelaunch_integrity_boundary")
    expected_order = [
        "Read the six code inputs with bounded no-follow file reads.",
        "Require exact byte length and SHA-256 for every file before any Python checker subprocess is started.",
        "Stage only the six verified code inputs and bounded source/receipt/native metadata into the create-only private directory.",
        "Start the fixed canonical Python command with -E -B and the declared CPU/cgroup caps.",
        "Retain the checker sibling-origin and helper-hash checks as a second post-import gate.",
    ]
    if not isinstance(boundary, dict) or (
        boundary.get("order"),
        boundary.get("model_or_state_payload_staged"),
        boundary.get("cache_array_or_dictionary_payload_staged"),
        boundary.get("trainer_or_framework_copied"),
    ) != (expected_order, False, False, False):
        raise ValueError("outer prelaunch integrity boundary differs")
    return {"bytes": len(raw), "sha256": _sha256(raw), "status": manifest.get("status")}


def _terminal_readiness() -> dict[str, object]:
    """Accept only actual, strict-integer native exit evidence already on disk."""
    if not NATIVE_TERMINAL_PATH.exists():
        return {
            "status": "WAITING_FOR_NATIVE_EXIT0_EVIDENCE",
            "terminal_evidence_path": str(NATIVE_TERMINAL_PATH),
            "raw_capture_path": str(NATIVE_CAPTURE_PATH),
            "render_permitted": False,
        }
    raw = _read_regular(NATIVE_TERMINAL_PATH, 16_000)
    terminal = _object(raw, "dictionary typed native terminal evidence")
    expected_identity = (NATIVE_TERMINAL_SCHEMA, RUN, UNIT, INVOCATION)
    observed_identity = tuple(
        terminal.get(key) for key in ("schema", "run_name", "unit", "invocation_id")
    )
    if observed_identity != expected_identity:
        raise ValueError("typed dictionary terminal identity differs")
    if (terminal.get("active_state"), terminal.get("sub_state")) != ("active", "exited"):
        raise ValueError("dictionary native terminal is not exact active/exited")
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        value = terminal.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"dictionary native terminal {key} is not the strict expected integer")
    raw_byte_count = terminal.get("raw_bytes")
    raw_sha256 = terminal.get("raw_sha256")
    if type(raw_byte_count) is not int or not 0 < raw_byte_count <= 2_000_000:
        raise ValueError("dictionary native raw capture byte count is invalid")
    if (
        not isinstance(raw_sha256, str)
        or len(raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha256)
    ):
        raise ValueError("dictionary native raw capture digest is malformed")
    if terminal.get("raw_path") != (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/"
        "terminal_native_capture_v1/STDOUT.raw.json"
    ):
        raise ValueError("dictionary native raw capture path differs")
    capture_raw = _read_regular(NATIVE_CAPTURE_PATH, 2_000_000)
    if len(capture_raw) != raw_byte_count or _sha256(capture_raw) != raw_sha256:
        raise ValueError("dictionary raw-first capture does not match typed terminal evidence")
    capture = _object(capture_raw, "dictionary raw native capture")
    if (
        capture.get("schema"),
        capture.get("expected_run"),
        capture.get("expected_unit"),
        capture.get("expected_invocation_id"),
        capture.get("captured_utc"),
    ) != (
        NATIVE_CAPTURE_SCHEMA,
        RUN,
        UNIT,
        INVOCATION,
        terminal.get("captured_utc"),
    ):
        raise ValueError("dictionary raw native capture identity/time differs")
    captured_unit = capture.get("unit_properties")
    if not isinstance(captured_unit, dict) or tuple(
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
    ) != (INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("dictionary raw native unit properties differ")
    timestamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        if not isinstance(value, str):
            raise ValueError(f"dictionary typed native {key} is missing")
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"dictionary typed native {key} is malformed") from error
        if stamp.tzinfo is None:
            raise ValueError(f"dictionary typed native {key} has no timezone")
        timestamps[key] = stamp
    if timestamps["native_exit_utc"] > timestamps["captured_utc"]:
        raise ValueError("dictionary native exit timestamp is after its capture")
    return {
        "status": "PASS_STRICT_NATIVE_EXIT0_METADATA",
        "typed_terminal_bytes": len(raw),
        "typed_terminal_sha256": _sha256(raw),
        "raw_capture_bytes": len(capture_raw),
        "raw_capture_sha256": raw_sha256,
        "payloads_loaded": False,
        "render_permitted": False,
        "reason": "final checker source review and root-approved body binding are still required",
    }


def build_local_report() -> dict[str, object]:
    """Bind actual original launch metadata and report conditional readiness."""
    if CAMPAIGN.resolve(strict=True) != CAMPAIGN or REPOSITORY.resolve(strict=True) != REPOSITORY:
        raise ValueError("campaign or repository root is redirected")
    template_raw = _read_regular(TEMPLATE_PATH, 50_000)
    template = _object(template_raw, "conditional admission template")
    if template.get("schema") != "fg19.child-dictionary-terminal-verifier-admission-template.v1":
        raise ValueError("conditional admission template schema differs")
    ema_template_raw = _read_regular(EMA_V3_TEMPLATE_PATH, EMA_V3_TEMPLATE_BYTES)
    ema_template_pin = {
        "bytes": len(ema_template_raw),
        "sha256": _sha256(ema_template_raw),
    }
    if ema_template_pin != {
        "bytes": EMA_V3_TEMPLATE_BYTES,
        "sha256": EMA_V3_TEMPLATE_SHA256,
    }:
        raise ValueError("reviewed EMA V3 fresh-gate template changed")
    if template.get("fresh_gate_baseline") != {
        "path": "preparation/child_ema_terminal_verifier_v3/REMOTE_SUBMISSION_TEMPLATE.py",
        **ema_template_pin,
        "use": "Reviewed fresh-gate mechanics reference only; the dictionary packet must bind its own identity, native evidence, checker closure, and predecessor set.",
        "copied_trainer_or_fit_logic": False,
    }:
        raise ValueError("fresh-gate baseline reference differs")
    if BODY_OUTPUT_PATH.exists() or BODY_OUTPUT_PATH.is_symlink():
        raise ValueError(
            "a rendered body already exists; local conditional checks must not inspect it"
        )
    body_template_raw = _read_regular(BODY_TEMPLATE_PATH, 100_000)
    body_template = body_template_raw.decode("utf-8")
    ast.parse(body_template)
    if (
        body_template.count("PAYLOAD = {}") != 1
        or body_template.count("FINAL_REVIEW_BINDING = {}") != 1
    ):
        raise ValueError(
            "body template must retain exactly one empty payload and review-binding slot"
        )
    body_renderer_raw = _read_regular(BODY_RENDERER_PATH, 40_000)
    body_builder_raw = _read_regular(BODY_BUILDER_PATH, 40_000)
    original = _check_original_receipts()
    code_inputs = _check_code_inputs()
    dependencies = _check_dependency_manifest()
    terminal = _terminal_readiness()
    return {
        "schema": "fg19.child-dictionary-terminal-verifier-local-checks.v1",
        "status": "CONDITIONAL_NOT_RENDERED",
        "campaign": str(CAMPAIGN.relative_to(REPOSITORY)),
        "template_sha256": _sha256(template_raw),
        "conditional_body_template": {
            "path": str(BODY_TEMPLATE_PATH.relative_to(REPOSITORY)),
            "bytes": len(body_template_raw),
            "sha256": _sha256(body_template_raw),
            "production_body_emitted": False,
        },
        "conditional_renderer": {
            "path": str(BODY_RENDERER_PATH.relative_to(REPOSITORY)),
            "bytes": len(body_renderer_raw),
            "sha256": _sha256(body_renderer_raw),
            "requires_actual_typed_and_raw_exit0_proof": True,
            "remote_or_B_action": False,
        },
        "template_builder": {
            "path": str(BODY_BUILDER_PATH.relative_to(REPOSITORY)),
            "bytes": len(body_builder_raw),
            "sha256": _sha256(body_builder_raw),
            "uses_reviewed_ema_v3_fresh_gates_only": True,
        },
        "ema_v3_fresh_gate_template": ema_template_pin,
        "original_launch_receipts": original,
        "code_input_pins": code_inputs,
        "dependency_manifest": dependencies,
        "native_terminal": terminal,
        "source_tree_content_hashed": False,
        "model_or_state_payload_read_or_hashed": False,
        "cache_array_or_dictionary_payload_read_or_hashed": False,
        "remote_or_b_action": False,
        "body_rendered": False,
        "remote_submission_attempted": False,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    """Write a deterministic local metadata receipt under this new folder."""
    if path.resolve(strict=False).parent != OUTPUT.resolve(strict=True):
        raise ValueError("local report must stay in the new verifier folder")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        raise ValueError(f"create-only local report already exists: {path.name}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    """Verify only local bounded metadata; never stage or submit a body."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-local-checks", action="store_true")
    args = parser.parse_args()
    report = build_local_report()
    if args.write_local_checks:
        _write_report(LOCAL_CHECKS_PATH, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
