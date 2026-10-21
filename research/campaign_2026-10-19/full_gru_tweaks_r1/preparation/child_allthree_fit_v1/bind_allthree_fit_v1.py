"""Prepare a fail-closed conditional AllThree child from the frozen NLL packet."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
CAMPAIGN = REPO / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PACKET = CAMPAIGN / "preparation/child_allthree_fit_v1"
NLL_PACKET = CAMPAIGN / "preparation/child_nll_fit_v1"
NLL_BODY = NLL_PACKET / "REMOTE_SUBMISSION_V2.py"
NLL_LAUNCH_ACCEPTANCE = CAMPAIGN / "records/child_nll_fit_v1/LAUNCH_ACCEPTANCE.json"
NLL_TERMINAL_ACCEPTANCE = CAMPAIGN / "records/child_nll_fit_v1/TERMINAL_ACCEPTANCE.json"
NLL_EXACT_STOP = CAMPAIGN / "records/child_nll_fit_v1/exact_stop/EXACT_STOP.json"
TEMPLATE = PACKET / "ALLTHREE_LAUNCH_TEMPLATE_V1.py"
FINAL_BODY = PACKET / "REMOTE_SUBMISSION.py"
FINAL_BINDING = PACKET / "ALLTHREE_FINAL_BINDING_RECEIPT_V1.json"

NLL_BODY_BYTES = 227_746
NLL_BODY_SHA256 = "893e06d9f17ee75450251bb3f8918312041ca6f147cf1cd30312335889843f15"
NLL_LAUNCH_ACCEPTANCE_BYTES = 2_931
NLL_LAUNCH_ACCEPTANCE_SHA256 = "44a708fc5a32b101c97a8a28c22e3b15e815059582de564e17f528f5369dc7a5"
NLL_LAUNCH_ACCEPTANCE_BODY_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/REMOTE_SUBMISSION.py"
)
NLL_RUN = "fg19_child_nll_f0_s1001_fit_v1"
NLL_UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
NLL_INVOCATION = "c833c613fb92426981ab255a5faf4452"
ALLTHREE_RUN = "fg19_child_allthree_f0_s1001_fit_v1"
ALLTHREE_UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
ALLTHREE_OUTPUT_RELATIVE = "research_tmp/fg19_child_allthree_fit_v1"
ALLTHREE_MODE = "ema_sparse"
TEMPLATE_BYTES = 238_232
TEMPLATE_SHA256 = "a35fff5323849edea130db297daf549e968f33ad8b4fdf6d3688daac00bdaa6f"
NLL_ACCEPTANCE_UNBOUND = "UNBOUND_PENDING_ACTUAL_NLL_CHILD_TERMINAL_ACCEPTANCE"
NLL_STOP_UNBOUND = "UNBOUND_PENDING_ACTUAL_NLL_CHILD_EXACT_STOP"
NLL_ACCEPTANCE_B64_UNBOUND = "UNBOUND_NLL_ACCEPTANCE_BYTES"
NLL_STOP_B64_UNBOUND = "UNBOUND_NLL_EXACT_STOP_BYTES"
MAX_NLL_RECEIPT_BYTES = 64_000

REMOTE_NLL_CONSTANTS = f'''\
ALLTHREE_NLL_CHILD_RUN = "{NLL_RUN}"
ALLTHREE_NLL_CHILD_UNIT = "{NLL_UNIT}"
ALLTHREE_NLL_CHILD_INVOCATION = "{NLL_INVOCATION}"
ALLTHREE_NLL_LAUNCH_ACCEPTANCE_SHA256 = "{NLL_LAUNCH_ACCEPTANCE_SHA256}"
ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256 = "{NLL_ACCEPTANCE_UNBOUND}"
ALLTHREE_NLL_CHILD_EXACT_STOP_SHA256 = "{NLL_STOP_UNBOUND}"
ALLTHREE_NLL_CHILD_ACCEPTANCE_B64 = "{NLL_ACCEPTANCE_B64_UNBOUND}"
ALLTHREE_NLL_CHILD_EXACT_STOP_B64 = "{NLL_STOP_B64_UNBOUND}"
'''

REMOTE_NLL_PREDECESSOR_GATE = '''\
def _allthree_nll_child_predecessor() -> tuple[bytes, bytes, dict[str, object]]:
    """Require the actual accepted and exactly stopped NLL fit as a sequential gate."""
    unbound = (
        ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256.startswith("UNBOUND_")
        or ALLTHREE_NLL_CHILD_EXACT_STOP_SHA256.startswith("UNBOUND_")
        or ALLTHREE_NLL_CHILD_ACCEPTANCE_B64.startswith("UNBOUND_")
        or ALLTHREE_NLL_CHILD_EXACT_STOP_B64.startswith("UNBOUND_")
    )
    if unbound:
        raise RuntimeError("actual NLL child terminal acceptance and exact stop are unbound")
    acceptance_raw = base64.b64decode(ALLTHREE_NLL_CHILD_ACCEPTANCE_B64, validate=True)
    stop_raw = base64.b64decode(ALLTHREE_NLL_CHILD_EXACT_STOP_B64, validate=True)
    if len(acceptance_raw) > 64_000 or len(stop_raw) > 64_000:
        raise RuntimeError("NLL child predecessor receipt exceeds the fixed bound")
    acceptance_sha = hashlib.sha256(acceptance_raw).hexdigest()
    stop_sha = hashlib.sha256(stop_raw).hexdigest()
    if (
        acceptance_sha != ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256
        or stop_sha != ALLTHREE_NLL_CHILD_EXACT_STOP_SHA256
    ):
        raise RuntimeError("NLL child predecessor receipt bytes differ from their pins")
    acceptance = json.loads(acceptance_raw)
    if not isinstance(acceptance, dict):
        raise RuntimeError("NLL child terminal acceptance must be a JSON object")
    native = acceptance.get("native_unit")
    if not isinstance(native, dict):
        raise RuntimeError("NLL child terminal acceptance lacks native unit evidence")
    expected = {
        "schema": "fg19-child-nll-fit-terminal-acceptance.v1",
        "status": "ACCEPTED",
        "run_name": ALLTHREE_NLL_CHILD_RUN,
        "unit": ALLTHREE_NLL_CHILD_UNIT,
        "invocation_id": ALLTHREE_NLL_CHILD_INVOCATION,
        "source_commit": P["source_commit"],
        "source_file_count": 167,
        "source_manifest_sha256": P["source_manifest_sha256"],
        "split_sha256": P["split_sha256"],
        "representation_mode": "base",
        "mean_variance_objective": "nll",
        "stop_status": "PENDING_EXACT_STOP",
        "launch_acceptance_sha256": ALLTHREE_NLL_LAUNCH_ACCEPTANCE_SHA256,
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "epochs": 6,
        "next_epoch": 6,
        "selected_epoch": 5,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "strict_fp32_finite_model_optimizer_verified": True,
        "selected_model_epoch5_and_training_state_tensor_identity_verified": True,
    }
    for key, value in expected.items():
        observed = acceptance.get(key)
        if observed != value or (type(value) is int and type(observed) is not int):
            raise RuntimeError(f"NLL child acceptance field differs: {key}")
    for key in ("source_file_count", "successful_updates", "completed_batches", "epochs", "next_epoch", "selected_epoch", "amp_skipped_updates", "nonfinite_updates", "main_pid", "exec_main_code", "exec_main_status"):
        if type(acceptance.get(key)) is not int:
            raise RuntimeError(f"NLL child acceptance integer is not strictly typed: {key}")
    for key, value in (
        ("InvocationID", ALLTHREE_NLL_CHILD_INVOCATION),
        ("MainPID", "0"),
        ("ExecMainCode", "1"),
        ("ExecMainStatus", "0"),
        ("ActiveState", "active"),
        ("SubState", "exited"),
    ):
        if native.get(key) != value:
            raise RuntimeError(f"NLL native acceptance state differs: {key}")
    files = acceptance.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("model_ep5.pt"), dict):
        raise RuntimeError("NLL child selected checkpoint identity is missing")
    selected = files["model_ep5.pt"]
    if (
        type(selected.get("bytes")) is not int
        or selected["bytes"] <= 0
        or not isinstance(selected.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", selected["sha256"]) is None
    ):
        raise RuntimeError("NLL child selected checkpoint pin is malformed")
    ancestry = acceptance.get("predecessor_ancestry")
    if not isinstance(ancestry, dict):
        raise RuntimeError("NLL child predecessor ancestry is missing")
    expected_ancestry = {
        "parent_acceptance_sha256": PARENT_ACCEPTANCE_SHA256,
        "parent_exact_stop_sha256": PARENT_EXACT_STOP_SHA256,
        "parent_checkpoint_sha256": PARENT_EP3_SHA256,
        "dictionary_acceptance_sha256": DICTIONARY_ACCEPTANCE_SHA256,
        "dictionary_exact_stop_sha256": DICTIONARY_EXACT_STOP_SHA256,
        "dictionary_sha256": DICTIONARY_SHA256,
        "profile_acceptance_sha256": PROFILE_ACCEPTANCE_SHA256,
        "profile_exact_stop_sha256": PROFILE_EXACT_STOP_SHA256,
    }
    for key, value in expected_ancestry.items():
        if ancestry.get(key) != value:
            raise RuntimeError(f"NLL child ancestor identity differs: {key}")
    nll_control = ancestry.get("control")
    nll_ema = ancestry.get("ema_sequential_predecessor")
    nll_dictionary_child = ancestry.get("dictionary_child_sequential_predecessor")
    if (
        not isinstance(nll_control, dict)
        or nll_control.get("acceptance_sha256") != CONTROL_ACCEPTANCE_SHA256
        or nll_control.get("exact_stop_sha256") != CONTROL_EXACT_STOP_SHA256
        or not isinstance(nll_ema, dict)
        or nll_ema.get("acceptance_sha256") != EMA_PREDECESSOR_ACCEPTANCE_SHA256
        or nll_ema.get("exact_stop_sha256") != EMA_PREDECESSOR_EXACT_STOP_SHA256
        or not isinstance(nll_dictionary_child, dict)
        or nll_dictionary_child.get("acceptance_sha256") != NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256
        or nll_dictionary_child.get("exact_stop_sha256") != NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256
        or nll_dictionary_child.get("launch_acceptance_sha256")
        != NLL_DICTIONARY_CHILD_LAUNCH_ACCEPTANCE_SHA256
    ):
        raise RuntimeError("NLL child accepted predecessor ancestry differs")
    stop = json.loads(stop_raw)
    if not isinstance(stop, dict):
        raise RuntimeError("NLL exact-stop receipt must be a JSON object")
    before, after = stop.get("before"), stop.get("after")
    verifier = stop.get("verifier")
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(verifier, dict):
        raise RuntimeError("NLL exact-stop native/verifier state is incomplete")
    if (
        stop.get("schema") != "fg19-child-nll-exact-owned-stop.v1"
        or stop.get("unit") != ALLTHREE_NLL_CHILD_UNIT
        or stop.get("invocation_id") != ALLTHREE_NLL_CHILD_INVOCATION
        or stop.get("acceptance_sha256") != ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256
        or stop.get("populated_before") != {}
        or stop.get("populated_after") != {}
        or stop.get("stop", {}).get("returncode") != 0
        or stop.get("stop", {}).get("stdout") != ""
        or stop.get("stop", {}).get("stderr") != ""
        or before.get("InvocationID") != ALLTHREE_NLL_CHILD_INVOCATION
        or before.get("MainPID") != "0"
        or before.get("ExecMainCode") != "1"
        or before.get("ExecMainStatus") != "0"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") != "0"
        or after.get("InvocationID") != ""
        or verifier.get("unit") != acceptance.get("verification_unit")
        or verifier.get("invocation_id") != acceptance.get("verification_invocation_id")
    ):
        raise RuntimeError("NLL native/verification exact-stop receipt differs")
    verify_before, verify_after = verifier.get("before"), verifier.get("after")
    if not isinstance(verify_before, dict) or not isinstance(verify_after, dict):
        raise RuntimeError("NLL verifier exact-stop states are missing")
    if (
        verify_before.get("InvocationID") != acceptance.get("verification_invocation_id")
        or verify_before.get("MainPID") != "0"
        or verify_before.get("ExecMainCode") != "1"
        or verify_before.get("ExecMainStatus") != "0"
        or verify_before.get("ActiveState") != "active"
        or verify_before.get("SubState") != "exited"
        or verify_after.get("LoadState") != "not-found"
        or verify_after.get("ActiveState") != "inactive"
        or verify_after.get("SubState") != "dead"
        or verify_after.get("MainPID") != "0"
        or verify_after.get("InvocationID") != ""
    ):
        raise RuntimeError("NLL verifier is not exactly stopped")
    return acceptance_raw, stop_raw, {
        "acceptance_sha256": acceptance_sha,
        "exact_stop_sha256": stop_sha,
        "launch_acceptance_sha256": ALLTHREE_NLL_LAUNCH_ACCEPTANCE_SHA256,
        "run_name": ALLTHREE_NLL_CHILD_RUN,
        "unit": ALLTHREE_NLL_CHILD_UNIT,
        "invocation_id": ALLTHREE_NLL_CHILD_INVOCATION,
        "selected_model_ep5_sha256": selected["sha256"],
        "selected_epoch": 5,
        "successful_updates": 44_760,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "initializer_remains_parent_ep3": True,
    }
'''


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
        raise ValueError(f"pinned source type/size differs: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
        ):
            raise ValueError(f"pinned source changed before read: {path}")
        raw = bytearray()
        while len(raw) <= expected_bytes:
            chunk = os.read(descriptor, min(65_536, expected_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != expected_bytes or _sha256(raw) != expected_sha256:
        raise ValueError(f"pinned source SHA/length differs: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    ):
        raise ValueError(f"pinned source changed while reading: {path}")
    return bytes(raw)


def _read_regular(path: Path, maximum: int, label: str) -> bytes:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError(f"{label} is not a bounded regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
        ):
            raise ValueError(f"{label} changed before read: {path}")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != info.st_size or len(raw) > maximum:
        raise ValueError(f"{label} changed size while reading: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed while reading: {path}")
    return bytes(raw)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def _assignment(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"source has no literal assignment for {name}")


def _argv_sha(argv: list[str]) -> str:
    raw = json.dumps(argv, separators=(",", ":")).encode()
    return _sha256(raw)


def _check_nll_launch_acceptance() -> dict[str, object]:
    raw = _read_pinned(
        NLL_LAUNCH_ACCEPTANCE,
        NLL_LAUNCH_ACCEPTANCE_BYTES,
        NLL_LAUNCH_ACCEPTANCE_SHA256,
    )
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise ValueError("NLL launch acceptance is not a JSON object")
    body = receipt.get("body")
    if (
        receipt.get("schema") != "fg19.child-nll-fit-launch-acceptance.v1"
        or receipt.get("run_name") != NLL_RUN
        or receipt.get("unit") != NLL_UNIT
        or receipt.get("invocation_id") != NLL_INVOCATION
        or receipt.get("status") != "ACTUAL_ONE_GPU_LAUNCH_BOUND_NO_COMPLETION_OR_QUALITY"
        or not isinstance(body, dict)
        or body.get("path") != NLL_LAUNCH_ACCEPTANCE_BODY_PATH
        or body.get("bytes") != NLL_BODY_BYTES
        or body.get("sha256") != NLL_BODY_SHA256
    ):
        raise ValueError("NLL launch acceptance does not bind the pinned actual NLL body")
    return receipt


def _source_argv(source: str) -> list[str]:
    argv = _assignment(source, "TRAINER_ARGV")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("NLL trainer argv is not a string list")
    if _argv_sha(argv) != "51905d5ecd65117ec0b0c8ab7a4f8aa4f24795015ac29a4af57dd9da781c4b90":
        raise ValueError("pinned NLL trainer argv digest differs")
    if (argv[1], argv[48], argv[54]) != (NLL_RUN, "base", "nll"):
        raise ValueError("pinned NLL argv identity/objective differs")
    return argv


def _derive_template() -> tuple[bytes, list[str], list[str]]:
    _check_nll_launch_acceptance()
    source = _read_pinned(NLL_BODY, NLL_BODY_BYTES, NLL_BODY_SHA256).decode("utf-8")
    nll_argv = _source_argv(source)
    allthree_argv = list(nll_argv)
    allthree_argv[1] = ALLTHREE_RUN
    allthree_argv[48] = ALLTHREE_MODE
    if [
        i for i, pair in enumerate(zip(nll_argv, allthree_argv, strict=True)) if pair[0] != pair[1]
    ] != [1, 48]:
        raise ValueError("AllThree must change exactly the run name and representation mode")
    derived_argv_sha = _argv_sha(allthree_argv)

    source = _replace_once(
        source,
        '"""UNBOUND local NLL launch template; no final submission body is admitted."""',
        '"""UNBOUND local AllThree launch template; no final body is admitted."""',
        "template docstring",
    )
    source = _replace_once(
        source,
        'OUTPUT = B / "research_tmp/fg19_child_nll_fit_v1"',
        f'OUTPUT = B / "{ALLTHREE_OUTPUT_RELATIVE}"',
        "private output identity",
    )
    source = _replace_once(source, f'RUN = "{NLL_RUN}"', f'RUN = "{ALLTHREE_RUN}"', "run identity")
    source = _replace_once(
        source, f'UNIT = "{NLL_UNIT}"', f'UNIT = "{ALLTHREE_UNIT}"', "unit identity"
    )
    source = _replace_once(
        source,
        f" '{NLL_RUN}',\n '--device',",
        f" '{ALLTHREE_RUN}',\n '--device',",
        "trainer run-name argv",
    )
    source = _replace_once(
        source,
        " '--representation-mode',\n 'base',\n '--representation-cache',",
        " '--representation-mode',\n 'ema_sparse',\n '--representation-cache',",
        "trainer representation argv",
    )
    source = _replace_once(
        source,
        "TRAINER_ARGV_SHA256 = '51905d5ecd65117ec0b0c8ab7a4f8aa4f24795015ac29a4af57dd9da781c4b90'",
        f"TRAINER_ARGV_SHA256 = '{derived_argv_sha}'",
        "trainer argv digest",
    )
    source = _replace_once(
        source,
        '"schema": "fg19-child-nll-fit-admission.v1"',
        '"schema": "fg19-child-allthree-fit-admission.v1"',
        "admission schema",
    )
    source = _replace_once(
        source,
        '"schema": "fg19-child-nll-fit-launch.v1"',
        '"schema": "fg19-child-allthree-fit-launch.v1"',
        "launch schema",
    )
    source = _replace_once(
        source,
        '"representation_mode": "base"',
        f'"representation_mode": "{ALLTHREE_MODE}"',
        "admission representation mode",
    )
    source = _replace_once(
        source,
        "EMA_PREDECESSOR_RUN = 'fg19_child_ema_f0_s1001_fit_v1'",
        REMOTE_NLL_CONSTANTS + "\nEMA_PREDECESSOR_RUN = 'fg19_child_ema_f0_s1001_fit_v1'",
        "NLL predecessor constants insertion",
    )
    source = _replace_once(
        source,
        "def _trainer_argv() -> list[str]:",
        REMOTE_NLL_PREDECESSOR_GATE + "\n\ndef _trainer_argv() -> list[str]:",
        "NLL predecessor validator insertion",
    )
    main_anchor = (
        "def main() -> None:\n"
        '    """Fail closed unless the dictionary child is accepted and exactly stopped."""\n'
        "    dictionary_child_acceptance_raw, dictionary_child_stop_raw, dictionary_child_predecessor = _dictionary_child_terminal_predecessor()\n"
    )
    main_patch = (
        "def main() -> None:\n"
        '    """Require all accepted predecessors, including the exact NLL terminal and stop."""\n'
        "    nll_child_acceptance_raw, nll_child_stop_raw, nll_child_predecessor = _allthree_nll_child_predecessor()\n"
        "    dictionary_child_acceptance_raw, dictionary_child_stop_raw, dictionary_child_predecessor = _dictionary_child_terminal_predecessor()\n"
    )
    source = _replace_once(source, main_anchor, main_patch, "NLL predecessor main gate")
    receipt_update = (
        'receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, '
        '"CONTROL_EXACT_STOP.json": control_stop_raw, "EMA_TERMINAL_ACCEPTANCE.json": ema_acceptance_raw, '
        '"EMA_EXACT_STOP.json": ema_stop_raw, "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": dictionary_child_acceptance_raw, '
        '"DICTIONARY_CHILD_EXACT_STOP.json": dictionary_child_stop_raw})'
    )
    source = _replace_once(
        source,
        receipt_update,
        receipt_update
        + '\n    receipt_bytes.update({"NLL_CHILD_TERMINAL_ACCEPTANCE.json": nll_child_acceptance_raw, '
        '"NLL_CHILD_EXACT_STOP.json": nll_child_stop_raw})',
        "NLL receipt staging",
    )
    staged_update = (
        'staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, '
        '"CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256, "EMA_TERMINAL_ACCEPTANCE.json": EMA_PREDECESSOR_ACCEPTANCE_SHA256, '
        '"EMA_EXACT_STOP.json": EMA_PREDECESSOR_EXACT_STOP_SHA256, "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256, '
        '"DICTIONARY_CHILD_EXACT_STOP.json": NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256})'
    )
    source = _replace_once(
        source,
        staged_update,
        staged_update
        + '\n    staged_pins.update({"NLL_CHILD_TERMINAL_ACCEPTANCE.json": ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256, '
        '"NLL_CHILD_EXACT_STOP.json": ALLTHREE_NLL_CHILD_EXACT_STOP_SHA256})',
        "NLL staged receipt digest map",
    )
    source = _replace_once(
        source,
        '        "ema_predecessor": ema_predecessor,\n'
        '        "dictionary_child_fit_predecessor": dictionary_child_predecessor,\n'
        '        "staged_receipt_sha256": {\n',
        '        "ema_predecessor": ema_predecessor,\n'
        '        "dictionary_child_fit_predecessor": dictionary_child_predecessor,\n'
        '        "nll_child_fit_predecessor": nll_child_predecessor,\n'
        '        "staged_receipt_sha256": {\n',
        "NLL admission predecessor record",
    )
    source = _replace_once(
        source,
        '        "ema_predecessor": ema_predecessor,\n'
        '        "dictionary_child_fit_predecessor": dictionary_child_predecessor,\n'
        '        "systemctl_after_submission": post_submit,\n',
        '        "ema_predecessor": ema_predecessor,\n'
        '        "dictionary_child_fit_predecessor": dictionary_child_predecessor,\n'
        '        "nll_child_fit_predecessor": nll_child_predecessor,\n'
        '        "systemctl_after_submission": post_submit,\n',
        "NLL launch predecessor record",
    )
    source = source.replace("NLL output/run path exists", "AllThree output/run path exists")
    ast.parse(source)
    observed_argv = _assignment(source, "TRAINER_ARGV")
    if observed_argv != allthree_argv or _argv_sha(observed_argv) != derived_argv_sha:
        raise ValueError("derived AllThree trainer argv differs from the exact two-change recipe")
    return source.encode("utf-8"), nll_argv, allthree_argv


def _require_launch_acceptance_field(acceptance: object) -> str:
    if not isinstance(acceptance, dict):
        raise ValueError("NLL child terminal acceptance must be a JSON object")
    value = acceptance.get("launch_acceptance_sha256")
    if not isinstance(value, str) or value != NLL_LAUNCH_ACCEPTANCE_SHA256:
        raise ValueError(
            "NLL child terminal acceptance does not bind its exact original launch acceptance"
        )
    return value


def _validate_nll_receipts(acceptance_raw: bytes, stop_raw: bytes) -> dict[str, object]:
    if not acceptance_raw or len(acceptance_raw) > MAX_NLL_RECEIPT_BYTES:
        raise ValueError("actual NLL child acceptance is absent or exceeds its fixed bound")
    if not stop_raw or len(stop_raw) > MAX_NLL_RECEIPT_BYTES:
        raise ValueError("actual NLL child exact stop is absent or exceeds its fixed bound")
    acceptance = json.loads(acceptance_raw)
    launch_sha = _require_launch_acceptance_field(acceptance)
    if not isinstance(acceptance, dict):
        raise ValueError("actual NLL child acceptance is not a JSON object")
    native = acceptance.get("native_unit")
    if not isinstance(native, dict):
        raise ValueError("actual NLL child acceptance lacks native unit evidence")
    expected = {
        "schema": "fg19-child-nll-fit-terminal-acceptance.v1",
        "status": "ACCEPTED",
        "run_name": NLL_RUN,
        "unit": NLL_UNIT,
        "invocation_id": NLL_INVOCATION,
        "source_commit": "68ff682977b822f38351aac0f621146f5f13b440",
        "source_file_count": 167,
        "source_manifest_sha256": "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
        "split_sha256": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
        "representation_mode": "base",
        "mean_variance_objective": "nll",
        "stop_status": "PENDING_EXACT_STOP",
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "epochs": 6,
        "next_epoch": 6,
        "selected_epoch": 5,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "strict_fp32_finite_model_optimizer_verified": True,
        "selected_model_epoch5_and_training_state_tensor_identity_verified": True,
    }
    for key, value in expected.items():
        observed = acceptance.get(key)
        if observed != value or (type(value) is int and type(observed) is not int):
            raise ValueError(f"NLL child acceptance field differs: {key}")
    for key in (
        "source_file_count",
        "successful_updates",
        "completed_batches",
        "epochs",
        "next_epoch",
        "selected_epoch",
        "amp_skipped_updates",
        "nonfinite_updates",
        "main_pid",
        "exec_main_code",
        "exec_main_status",
    ):
        if type(acceptance.get(key)) is not int:
            raise ValueError(f"NLL child integer is not strictly typed: {key}")
    native_expected = {
        "InvocationID": NLL_INVOCATION,
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "ActiveState": "active",
        "SubState": "exited",
    }
    if any(native.get(key) != value for key, value in native_expected.items()):
        raise ValueError("NLL child native unit is not an accepted exit-0 completion")
    files = acceptance.get("files")
    selected = files.get("model_ep5.pt") if isinstance(files, dict) else None
    if not isinstance(selected, dict):
        raise ValueError("NLL selected model_ep5 file map is missing")
    if (
        type(selected.get("bytes")) is not int
        or selected["bytes"] <= 0
        or not isinstance(selected.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", selected["sha256"]) is None
    ):
        raise ValueError("NLL selected model_ep5 file pin is malformed")
    ancestry = acceptance.get("predecessor_ancestry")
    if not isinstance(ancestry, dict):
        raise ValueError("NLL predecessor ancestry is missing")
    expected_ancestry = {
        "parent_acceptance_sha256": "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
        "parent_exact_stop_sha256": "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
        "parent_checkpoint_sha256": "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45",
        "dictionary_acceptance_sha256": "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
        "dictionary_exact_stop_sha256": "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
        "dictionary_sha256": "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659",
        "profile_acceptance_sha256": "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
        "profile_exact_stop_sha256": "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    }
    if any(ancestry.get(key) != value for key, value in expected_ancestry.items()):
        raise ValueError("NLL parent/dictionary/profile ancestry differs")
    nll_control = ancestry.get("control")
    nll_ema = ancestry.get("ema_sequential_predecessor")
    nll_dictionary_child = ancestry.get("dictionary_child_sequential_predecessor")
    if (
        not isinstance(nll_control, dict)
        or nll_control.get("acceptance_sha256")
        != "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
        or nll_control.get("exact_stop_sha256")
        != "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
        or not isinstance(nll_ema, dict)
        or nll_ema.get("acceptance_sha256")
        != "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8"
        or nll_ema.get("exact_stop_sha256")
        != "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf"
        or not isinstance(nll_dictionary_child, dict)
        or nll_dictionary_child.get("acceptance_sha256")
        != "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1"
        or nll_dictionary_child.get("exact_stop_sha256")
        != "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3"
        or nll_dictionary_child.get("launch_acceptance_sha256")
        != "33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95"
    ):
        raise ValueError("NLL CONTROL/EMA/dictionary-child ancestry differs")
    stop = json.loads(stop_raw)
    if not isinstance(stop, dict):
        raise ValueError("NLL exact stop is not a JSON object")
    before, after, verifier = stop.get("before"), stop.get("after"), stop.get("verifier")
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or not isinstance(verifier, dict)
    ):
        raise ValueError("NLL exact-stop native or verifier details are absent")
    if (
        stop.get("schema") != "fg19-child-nll-exact-owned-stop.v1"
        or stop.get("unit") != NLL_UNIT
        or stop.get("invocation_id") != NLL_INVOCATION
        or stop.get("acceptance_sha256") != _sha256(acceptance_raw)
        or stop.get("populated_before") != {}
        or stop.get("populated_after") != {}
        or stop.get("stop", {}).get("returncode") != 0
        or stop.get("stop", {}).get("stdout") != ""
        or stop.get("stop", {}).get("stderr") != ""
        or before.get("InvocationID") != NLL_INVOCATION
        or before.get("MainPID") != "0"
        or before.get("ExecMainCode") != "1"
        or before.get("ExecMainStatus") != "0"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") != "0"
        or after.get("InvocationID") != ""
    ):
        raise ValueError("NLL native exact-stop state does not match acceptance")
    verify_before, verify_after = verifier.get("before"), verifier.get("after")
    if not isinstance(verify_before, dict) or not isinstance(verify_after, dict):
        raise ValueError("NLL verifier exact-stop state is missing")
    if (
        not isinstance(acceptance.get("verification_unit"), str)
        or not acceptance["verification_unit"]
        or not isinstance(acceptance.get("verification_invocation_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", acceptance["verification_invocation_id"]) is None
        or verifier.get("unit") != acceptance["verification_unit"]
        or verifier.get("invocation_id") != acceptance["verification_invocation_id"]
        or verify_before.get("InvocationID") != acceptance["verification_invocation_id"]
        or verify_before.get("MainPID") != "0"
        or verify_before.get("ExecMainCode") != "1"
        or verify_before.get("ExecMainStatus") != "0"
        or verify_before.get("ActiveState") != "active"
        or verify_before.get("SubState") != "exited"
        or verify_after.get("LoadState") != "not-found"
        or verify_after.get("ActiveState") != "inactive"
        or verify_after.get("SubState") != "dead"
        or verify_after.get("MainPID") != "0"
        or verify_after.get("InvocationID") != ""
    ):
        raise ValueError("NLL verifier is not exactly stopped")
    return {
        "acceptance_sha256": _sha256(acceptance_raw),
        "exact_stop_sha256": _sha256(stop_raw),
        "launch_acceptance_sha256": launch_sha,
        "run_name": NLL_RUN,
        "unit": NLL_UNIT,
        "invocation_id": NLL_INVOCATION,
        "selected_model_ep5_sha256": selected["sha256"],
        "selected_epoch": 5,
        "successful_updates": 44_760,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "initializer_remains_parent_ep3": True,
    }


def _check_template() -> dict[str, object]:
    expected, nll_argv, allthree_argv = _derive_template()
    raw = _read_pinned(TEMPLATE, TEMPLATE_BYTES, TEMPLATE_SHA256)
    if raw != expected:
        raise ValueError(
            "AllThree template is not the exact pinned NLL-plus-predecessor derivation"
        )
    if FINAL_BODY.exists() or FINAL_BINDING.exists():
        raise ValueError(
            "final AllThree body/binding must remain absent before actual NLL acceptance and stop"
        )
    text = raw.decode("utf-8")
    if any(
        marker not in text
        for marker in (
            NLL_ACCEPTANCE_UNBOUND,
            NLL_STOP_UNBOUND,
            NLL_ACCEPTANCE_B64_UNBOUND,
            NLL_STOP_B64_UNBOUND,
        )
    ):
        raise ValueError("AllThree template lost an explicit unbound NLL receipt slot")
    return {
        "status": "PASS_LOCAL_UNBOUND_ALLTHREE_TEMPLATE",
        "template": {"bytes": len(raw), "sha256": _sha256(raw)},
        "nll_trainer_argv_sha256": _argv_sha(nll_argv),
        "allthree_trainer_argv_sha256": _argv_sha(allthree_argv),
        "changed_argv_indices_zero_based": [1, 48],
        "actual_nll_terminal_receipts_read": False,
        "final_body_absent": True,
    }


def _render_body(acceptance_raw: bytes, stop_raw: bytes) -> tuple[bytes, dict[str, object]]:
    checked = _check_template()
    predecessor = _validate_nll_receipts(acceptance_raw, stop_raw)
    source = _read_pinned(TEMPLATE, TEMPLATE_BYTES, TEMPLATE_SHA256).decode("utf-8")
    substitutions = {
        f'ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256 = "{NLL_ACCEPTANCE_UNBOUND}"': (
            f'ALLTHREE_NLL_CHILD_ACCEPTANCE_SHA256 = "{predecessor["acceptance_sha256"]}"'
        ),
        f'ALLTHREE_NLL_CHILD_EXACT_STOP_SHA256 = "{NLL_STOP_UNBOUND}"': (
            f'ALLTHREE_NLL_CHILD_EXACT_STOP_SHA256 = "{predecessor["exact_stop_sha256"]}"'
        ),
        f'ALLTHREE_NLL_CHILD_ACCEPTANCE_B64 = "{NLL_ACCEPTANCE_B64_UNBOUND}"': (
            "ALLTHREE_NLL_CHILD_ACCEPTANCE_B64 = "
            f'"{base64.b64encode(acceptance_raw).decode("ascii")}"'
        ),
        f'ALLTHREE_NLL_CHILD_EXACT_STOP_B64 = "{NLL_STOP_B64_UNBOUND}"': (
            f'ALLTHREE_NLL_CHILD_EXACT_STOP_B64 = "{base64.b64encode(stop_raw).decode("ascii")}"'
        ),
    }
    for old, new in substitutions.items():
        source = _replace_once(source, old, new, "NLL predecessor receipt slot")
    if any(marker in source for marker in (NLL_ACCEPTANCE_UNBOUND, NLL_STOP_UNBOUND)):
        raise ValueError("rendered AllThree body retains an unbound NLL predecessor")
    ast.parse(source)
    binding = {
        "schema": "fg19-child-allthree-fit-final-binding.v1",
        "status": "PASS_LOCAL_RENDERED_BODY_AWAITING_INDEPENDENT_SOURCE_REVIEW",
        "body": {
            "path": FINAL_BODY.name,
            "bytes": len(source.encode()),
            "sha256": _sha256(source.encode()),
        },
        "nll_predecessor": predecessor,
        "initializer_remains": "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
        "dictionary_and_nll_models_are_sequential_gates_only": True,
        "no_b_action": True,
        "template_check": checked["status"],
    }
    return source.encode(), binding


def _write_create_only(path: Path, raw: bytes, mode: int | None = None) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if mode is not None:
        path.chmod(mode)


def _prepare_template() -> dict[str, object]:
    if TEMPLATE.exists() or FINAL_BODY.exists() or FINAL_BINDING.exists():
        raise FileExistsError("AllThree template/body already exists; refusing overwrite")
    raw, nll_argv, allthree_argv = _derive_template()
    _write_create_only(TEMPLATE, raw, 0o600)
    return {
        "status": "CREATED_LOCAL_UNBOUND_ALLTHREE_TEMPLATE_ONLY",
        "template": {"path": TEMPLATE.name, "bytes": len(raw), "sha256": _sha256(raw)},
        "trainer_argv_changes_zero_based": {
            str(index): [nll_argv[index], allthree_argv[index]] for index in (1, 48)
        },
        "actual_nll_terminal_receipts_read": False,
        "final_body_created": False,
    }


def _render() -> dict[str, object]:
    if FINAL_BODY.exists() or FINAL_BINDING.exists():
        raise FileExistsError("AllThree final output exists; refusing overwrite")
    acceptance_raw = _read_regular(
        NLL_TERMINAL_ACCEPTANCE, MAX_NLL_RECEIPT_BYTES, "actual NLL terminal acceptance"
    )
    stop_raw = _read_regular(NLL_EXACT_STOP, MAX_NLL_RECEIPT_BYTES, "actual NLL exact stop")
    body, binding = _render_body(acceptance_raw, stop_raw)
    record_raw = (json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _write_create_only(FINAL_BODY, body, 0o700)
    _write_create_only(FINAL_BINDING, record_raw)
    return {
        "status": binding["status"],
        "body": binding["body"],
        "binding_sha256": _sha256(record_raw),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-template", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.prepare_template:
        result = _prepare_template()
    elif args.check:
        result = _check_template()
    else:
        result = _render()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
