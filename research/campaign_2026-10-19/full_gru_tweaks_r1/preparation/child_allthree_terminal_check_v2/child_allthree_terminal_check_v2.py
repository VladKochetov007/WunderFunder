"""Verify the one exited AllThree F0 seed-1001 fit using pinned pure leaves."""

from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_HELPER_DIR = Path(__file__).absolute().parent
_PREPARATION_DIR = _HELPER_DIR.parent
if not (_HELPER_DIR / "child_nll_terminal_check_v1.py").is_file():
    for _directory in (
        _PREPARATION_DIR,
        _PREPARATION_DIR / "child_control_terminal_verifier_v3",
        _PREPARATION_DIR / "child_ema_terminal_verifier_v3",
        _PREPARATION_DIR / "child_nll_terminal_check_v1",
    ):
        if str(_directory) not in sys.path:
            sys.path.insert(0, str(_directory))

control_v3 = importlib.import_module("child_control_terminal_check_v3")
parent_v3 = importlib.import_module("parent_fit_terminal_check_v3")
support = importlib.import_module("propensity_terminal_support")
teacher_s5 = importlib.import_module("teacher_s5_terminal_check_v1")
ema_v3 = importlib.import_module("child_ema_terminal_check_v3")
nll_check = importlib.import_module("child_nll_terminal_check_v1")

RUN = "fg19_child_allthree_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
NLL_RUN = "fg19_child_nll_f0_s1001_fit_v1"
NLL_UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
NLL_INVOCATION = "c833c613fb92426981ab255a5faf4452"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
TRAINER_ARGV_SHA = "7bc0f3e81095a7a08b06758f655467d26a06d96d84cebdb2c321e31c0dec7ee5"
ADMISSION_SHA = "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"
LAUNCH_SHA = "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538"
LAUNCH_ACCEPTANCE_SHA = "2afbba374ea2e332766368087490660198110027784924c69d03a5656c924346"
BODY_SHA = "c2961ef594953c5885a948bd3433e9deeadd376010b60064d9a556b842ab655a"
BODY_BYTES = 263_545
ROOT_PREPARATION_SHA = "67b25be1265091d7b29dd08612a146a4d44223129efa8a5c104f25a2e1bfaa44"
SOURCE_REVIEW_SHA = "1ab84547255d6c9b9c950569b09e6410eb640ce7e1deaed5950a718c36e1bada"
SOURCE_REVIEW_BYTES = 7_046
ADMISSION_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/ADMISSION.json"
)
LAUNCH_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/LAUNCH.json"
)
BODY_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/REMOTE_SUBMISSION.py"
LAUNCH_ACCEPTANCE_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/LAUNCH_ACCEPTANCE.json"
SOURCE_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_fit_v1/FINAL_BODY_SOURCE_REVIEW_V1.json"
ROOT_PREPARATION_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/ROOT_ADMISSION_PREPARATION.json"
NATIVE_RAW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
RAW_STDOUT_SHA = "98345f64e9a8ec41978f8679cd631095b9bbb5936be8895d4bda574927855e73"
RAW_STDOUT_BYTES = 30_050
TRANSPORT_SHA = "1c30ca323cb5899c8269c24f277e6b73a33305c4855c8a3167490785038e5a57"
TRANSPORT_BYTES = 546
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
PARENT_MODEL_EP3_SHA = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
NLL_LAUNCH_ACCEPTANCE_SHA = "44a708fc5a32b101c97a8a28c22e3b15e815059582de564e17f528f5369dc7a5"
NLL_ACCEPTANCE_SHA = "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5"
NLL_STOP_SHA = "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1"
DICTIONARY_PATH = control_v3.DICTIONARY_CACHE_PATH
EVIDENCE_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": control_v3.PARENT_ACCEPTANCE_SHA,
    "PARENT_EXACT_STOP.json": control_v3.PARENT_STOP_SHA,
    "DICTIONARY_ACCEPTANCE.json": control_v3.DICTIONARY_ACCEPTANCE_SHA,
    "DICTIONARY_EXACT_STOP.json": control_v3.DICTIONARY_STOP_SHA,
    "PROFILE_TERMINAL_ACCEPTANCE.json": control_v3.PROFILE_ACCEPTANCE_SHA,
    "PROFILE_EXACT_STOP.json": control_v3.PROFILE_STOP_SHA,
    "CONTROL_TERMINAL_ACCEPTANCE.json": nll_check.CONTROL_ACCEPTANCE_SHA,
    "CONTROL_EXACT_STOP.json": nll_check.CONTROL_STOP_SHA,
    "EMA_TERMINAL_ACCEPTANCE.json": nll_check.EMA_ACCEPTANCE_SHA,
    "EMA_EXACT_STOP.json": nll_check.EMA_STOP_SHA,
    "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": nll_check.DICTIONARY_CHILD_ACCEPTANCE_SHA,
    "DICTIONARY_CHILD_EXACT_STOP.json": nll_check.DICTIONARY_CHILD_STOP_SHA,
    "NLL_CHILD_TERMINAL_ACCEPTANCE.json": NLL_ACCEPTANCE_SHA,
    "NLL_CHILD_EXACT_STOP.json": NLL_STOP_SHA,
}
EXPECTED_RECIPE = {
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
    "representation_mode": "ema_sparse",
    "seed": 1001,
    "trainer_argv_sha256": TRAINER_ARGV_SHA,
}

HELPER_PINS = {
    "child_nll_terminal_check_v1.py": (
        40_474,
        "a3cf3b10515bb16275561a0ab9338227c00661347afbdf081888ec258dd1f7a2",
        nll_check,
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
        control_v3,
    ),
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
        ema_v3,
    ),
    "parent_fit_terminal_check_v3.py": (
        64_550,
        "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
        parent_v3,
    ),
    "propensity_terminal_support.py": (
        13_107,
        "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
        support,
    ),
    "teacher_s5_terminal_check_v1.py": (
        37_448,
        "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
        teacher_s5,
    ),
}


def _eq(actual: object, expected: object, label: str) -> None:
    """Use the reviewed non-coercing equality leaf."""
    nll_check._eq(actual, expected, label)


def _check_native_terminal(terminal: Mapping[str, object]) -> dict[str, object]:
    """Require exact actual native exit evidence before any changing payload read."""
    expected_keys = {
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
    }
    _eq(set(terminal), expected_keys, "exact AllThree native terminal evidence keys")
    expected_identity = (
        "fg19.child-allthree-fit-native-terminal-evidence.v1",
        RUN,
        UNIT,
        INVOCATION,
    )
    _eq(
        tuple(terminal.get(key) for key in ("schema", "run_name", "unit", "invocation_id")),
        expected_identity,
        "AllThree native terminal identity",
    )
    _eq(
        (terminal.get("active_state"), terminal.get("sub_state")),
        ("active", "exited"),
        "AllThree native state",
    )
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        if type(terminal.get(key)) is not int or terminal[key] != expected:
            raise ValueError(f"native {key} must be strict integer {expected}")
    raw_bytes = terminal.get("raw_bytes")
    raw_sha = terminal.get("raw_sha256")
    raw_path = terminal.get("raw_path")
    if type(raw_bytes) is not int or raw_bytes <= 0:
        raise ValueError("native raw byte count must be a positive strict integer")
    if not isinstance(raw_sha, str) or re.fullmatch(r"[0-9a-f]{64}", raw_sha) is None:
        raise ValueError("native raw SHA-256 must be 64 lowercase hexadecimal characters")
    _eq(raw_path, NATIVE_RAW_PATH, "canonical AllThree native raw path")
    stamps: dict[str, datetime] = {}
    for key in ("captured_utc", "native_exit_utc"):
        stamp = terminal.get(key)
        if not isinstance(stamp, str):
            raise ValueError(f"native {key} is missing")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"native {key} is not ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError(f"native {key} must include a timezone")
        stamps[key] = parsed
    if stamps["native_exit_utc"] > stamps["captured_utc"]:
        raise ValueError("native exit timestamp follows its evidence capture")
    return {
        "captured_utc": terminal["captured_utc"],
        "exec_main_code": 1,
        "exec_main_status": 0,
        "invocation_id": INVOCATION,
        "main_pid": 0,
        "native_exit_utc": terminal["native_exit_utc"],
        "raw_bytes": raw_bytes,
        "raw_path": raw_path,
        "raw_sha256": raw_sha,
        "state": "active/exited",
    }


def assert_helper_closure() -> dict[str, dict[str, object]]:
    """Check the private pure-code closure without entering terminal verification."""
    return _check_helper_closure()


def _check_helper_closure() -> dict[str, dict[str, object]]:
    """Hash each reviewed pure module and require imports from the pinned file."""
    preparation = _PREPARATION_DIR
    local_paths = {
        "child_nll_terminal_check_v1.py": preparation
        / "child_nll_terminal_check_v1"
        / "child_nll_terminal_check_v1.py",
        "child_control_terminal_check_v3.py": preparation
        / "child_control_terminal_verifier_v3"
        / "child_control_terminal_check_v3.py",
        "child_ema_terminal_check_v3.py": preparation
        / "child_ema_terminal_verifier_v3"
        / "child_ema_terminal_check_v3.py",
        "parent_fit_terminal_check_v3.py": preparation / "parent_fit_terminal_check_v3.py",
        "propensity_terminal_support.py": preparation / "propensity_terminal_support.py",
        "teacher_s5_terminal_check_v1.py": preparation / "teacher_s5_terminal_check_v1.py",
    }
    observed: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha, module) in HELPER_PINS.items():
        flat_path = _HELPER_DIR / name
        path = flat_path if flat_path.is_file() else local_paths[name]
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        if path.is_symlink() or path.resolve(strict=True) != module_path:
            raise ValueError(f"helper import escaped the pinned closure: {name}")
        digest, info = support._hash_file(path, expected_bytes)
        _eq((info["bytes"], digest), (expected_bytes, expected_sha), f"pinned helper {name}")
        observed[name] = {"bytes": info["bytes"], "sha256": digest}
    return observed


def _check_native_predecessor(
    evidence_root: Path, admission: Mapping[str, object]
) -> dict[str, object]:
    """Bind the separately accepted NLL run as a stopped gate, never an initializer."""
    acceptance_path = evidence_root / "NLL_CHILD_TERMINAL_ACCEPTANCE.json"
    stop_path = evidence_root / "NLL_CHILD_EXACT_STOP.json"
    acceptance, acceptance_sha = support._json_file(acceptance_path)
    stop, stop_sha = support._json_file(stop_path)
    _eq((acceptance_sha, stop_sha), (NLL_ACCEPTANCE_SHA, NLL_STOP_SHA), "NLL predecessor pins")
    _eq(
        tuple(
            acceptance.get(key)
            for key in (
                "schema",
                "status",
                "run_name",
                "unit",
                "invocation_id",
                "source_commit",
                "source_manifest_sha256",
                "split_sha256",
                "representation_mode",
                "successful_updates",
                "completed_batches",
                "amp_skipped_updates",
                "nonfinite_updates",
                "next_epoch",
                "selected_epoch",
                "main_pid",
                "exec_main_code",
                "exec_main_status",
                "launch_acceptance_sha256",
                "quality_score",
            )
        ),
        (
            "fg19-child-nll-fit-terminal-acceptance.v1",
            "ACCEPTED",
            NLL_RUN,
            NLL_UNIT,
            NLL_INVOCATION,
            COMMIT,
            SOURCE_MANIFEST_SHA,
            SPLIT_SHA,
            "base",
            44_760,
            4_476,
            0,
            0,
            6,
            5,
            0,
            1,
            0,
            NLL_LAUNCH_ACCEPTANCE_SHA,
            None,
        ),
        "accepted NLL predecessor integrity and scope",
    )
    _eq(
        (
            acceptance.get("strict_fp32_finite_model_optimizer_verified"),
            acceptance.get("selected_model_epoch5_and_training_state_tensor_identity_verified"),
            acceptance.get("files", {}).get("model_ep5.pt", {}).get("sha256"),
        ),
        (True, True, "b19a61363b4b49cd98d0277f2f74fd3aa3db50bb6c1fcb314b5fab571e27fd58"),
        "NLL accepted finite state and selected model identity",
    )
    control_v3._check_stop_receipt(
        stop,
        NLL_UNIT,
        NLL_INVOCATION,
        NLL_ACCEPTANCE_SHA,
        strict_schema="fg19-child-nll-exact-owned-stop.v1",
    )
    expected = {
        "acceptance_sha256": NLL_ACCEPTANCE_SHA,
        "exact_stop_sha256": NLL_STOP_SHA,
        "initializer_remains_parent_ep3": True,
        "invocation_id": NLL_INVOCATION,
        "launch_acceptance_sha256": NLL_LAUNCH_ACCEPTANCE_SHA,
        "run_name": NLL_RUN,
        "selected_epoch": 5,
        "selected_model_ep5_sha256": "b19a61363b4b49cd98d0277f2f74fd3aa3db50bb6c1fcb314b5fab571e27fd58",
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "successful_updates": 44_760,
        "unit": NLL_UNIT,
    }
    _eq(admission.get("nll_child_fit_predecessor"), expected, "NLL sequential predecessor")
    return expected


def _check_predecessor_receipts(evidence_root: Path, admission: Mapping[str, object]) -> None:
    """Verify all 14 staged predecessor receipt bytes without changing any receipt."""
    _eq(admission.get("staged_receipt_sha256"), EVIDENCE_RECEIPTS, "14 receipt SHA map")
    for name, expected_sha in EVIDENCE_RECEIPTS.items():
        actual_sha, _ = support._hash_file(evidence_root / name, control_v3.MAX_JSON_BYTES)
        _eq(actual_sha, expected_sha, f"predecessor receipt {name}")


def _check_paths(source_root: Path, run_dir: Path, split_path: Path, cache_root: Path) -> None:
    """Restrict verification to the admitted source, run, split, and cache paths."""
    source = source_root.resolve(strict=True)
    base = source.parent
    _eq(source.name, "source-fg19-cache-68ff6829", "source directory")
    expected_paths = (
        ("source root", source_root, base / "source-fg19-cache-68ff6829"),
        ("AllThree run", run_dir, base / "data/artifacts/runs" / RUN),
        ("F0 split", split_path, base / "splits/fold0.json"),
        (
            "homogeneous T3 cache",
            cache_root,
            base / "research_tmp/fg19_t3_fit_homogeneous_cache_v1",
        ),
    )
    for label, path, expected in expected_paths:
        if path.is_symlink():
            raise ValueError(f"{label} cannot be a symlink")
        _eq(path.resolve(strict=True), expected.resolve(strict=True), label)


def _check_admission(admission: Mapping[str, object]) -> None:
    """Require the exact source, recipe, ancestry, and no-mtime dictionary receipt."""
    _eq(
        tuple(
            admission.get(key)
            for key in (
                "schema",
                "run_name",
                "unit",
                "source_commit",
                "source_manifest_sha256",
                "source_files_verified",
                "split_sha256",
            )
        ),
        (
            "fg19-child-allthree-fit-admission.v1",
            RUN,
            UNIT,
            COMMIT,
            SOURCE_MANIFEST_SHA,
            167,
            SPLIT_SHA,
        ),
        "AllThree admission identity/source",
    )
    _eq(admission.get("recipe"), EXPECTED_RECIPE, "fixed AllThree recipe")
    _eq(admission.get("populated"), {}, "empty successor descendants at launch")
    dictionary = admission.get("dictionary")
    if not isinstance(dictionary, dict) or set(dictionary) != {
        "bytes",
        "device",
        "inode",
        "path",
        "sha256",
    }:
        raise ValueError("shared dictionary receipt must retain its original five keys")
    _eq(
        dictionary,
        {
            "bytes": control_v3.DICTIONARY_BYTES,
            "device": 64_512,
            "inode": 733_618_561,
            "path": str(DICTIONARY_PATH),
            "sha256": control_v3.DICTIONARY_SHA,
        },
        "immutable shared dictionary receipt",
    )
    _eq(
        tuple(
            admission.get(key)
            for key in (
                "parent_acceptance_sha256",
                "parent_exact_stop_sha256",
                "parent_checkpoint_sha256",
            )
        ),
        (
            control_v3.PARENT_ACCEPTANCE_SHA,
            control_v3.PARENT_STOP_SHA,
            PARENT_MODEL_EP3_SHA,
        ),
        "parent-only initializer ancestry",
    )
    _eq(
        admission.get("child_profile"),
        {
            "acceptance_sha256": control_v3.PROFILE_ACCEPTANCE_SHA,
            "exact_stop_sha256": control_v3.PROFILE_STOP_SHA,
            "files": {
                "ADMISSION.json": {
                    "bytes": 7_817,
                    "sha256": "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
                },
                "LAUNCH.json": {
                    "bytes": 9_691,
                    "sha256": "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
                },
                "unit.log": {
                    "bytes": 7_624,
                    "sha256": "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
                },
            },
            "initial_state_sha256": control_v3.PROFILE_INITIAL_STATE_SHA,
            "invocation_id": control_v3.PROFILE_INVOCATION,
            "optimizer_attempts": 60,
            "optimizer_completed_updates": 60,
            "quality_score": None,
            "scientific_fits": 0,
            "unit": control_v3.PROFILE_UNIT,
        },
        "accepted profile identity",
    )
    _eq(
        admission.get("nll_child_fit_predecessor"),
        {
            "acceptance_sha256": NLL_ACCEPTANCE_SHA,
            "exact_stop_sha256": NLL_STOP_SHA,
            "initializer_remains_parent_ep3": True,
            "invocation_id": NLL_INVOCATION,
            "launch_acceptance_sha256": NLL_LAUNCH_ACCEPTANCE_SHA,
            "run_name": NLL_RUN,
            "selected_epoch": 5,
            "selected_model_ep5_sha256": "b19a61363b4b49cd98d0277f2f74fd3aa3db50bb6c1fcb314b5fab571e27fd58",
            "status": "ACCEPTED_AND_EXACT_STOPPED",
            "successful_updates": 44_760,
            "unit": NLL_UNIT,
        },
        "NLL accepted predecessor, parent remains initializer",
    )


def _check_launch_records(
    admission_sha: str,
    launch: Mapping[str, object],
    launch_sha: str,
    acceptance: Mapping[str, object],
    acceptance_sha: str,
) -> None:
    """Bind one actual AllThree launch to its exact reviewed body and receipts."""
    _eq(
        (admission_sha, launch_sha, acceptance_sha),
        (ADMISSION_SHA, LAUNCH_SHA, LAUNCH_ACCEPTANCE_SHA),
        "AllThree launch receipt pins",
    )
    submission = launch.get("submission")
    _eq(
        (
            launch.get("schema"),
            launch.get("unit"),
            isinstance(submission, dict),
            submission.get("outcome") if isinstance(submission, dict) else None,
            submission.get("returncode") if isinstance(submission, dict) else None,
        ),
        ("fg19-child-allthree-fit-launch.v1", UNIT, True, "submitted", 0),
        "single original AllThree launch",
    )
    _eq(
        launch.get("nll_child_fit_predecessor"),
        {
            "acceptance_sha256": NLL_ACCEPTANCE_SHA,
            "exact_stop_sha256": NLL_STOP_SHA,
            "initializer_remains_parent_ep3": True,
            "invocation_id": NLL_INVOCATION,
            "launch_acceptance_sha256": NLL_LAUNCH_ACCEPTANCE_SHA,
            "run_name": NLL_RUN,
            "selected_epoch": 5,
            "selected_model_ep5_sha256": "b19a61363b4b49cd98d0277f2f74fd3aa3db50bb6c1fcb314b5fab571e27fd58",
            "status": "ACCEPTED_AND_EXACT_STOPPED",
            "successful_updates": 44_760,
            "unit": NLL_UNIT,
        },
        "launch NLL predecessor",
    )
    expected_launch_acceptance = {
        "body": {"path": BODY_PATH, "bytes": BODY_BYTES, "sha256": BODY_SHA},
        "admission": {"path": ADMISSION_PATH, "bytes": 12_912, "sha256": ADMISSION_SHA},
        "launch": {"path": LAUNCH_PATH, "bytes": 9_594, "sha256": LAUNCH_SHA},
        "source_review": {
            "path": SOURCE_REVIEW_PATH,
            "bytes": SOURCE_REVIEW_BYTES,
            "sha256": SOURCE_REVIEW_SHA,
        },
        "root_preparation": {
            "path": ROOT_PREPARATION_PATH,
            "bytes": 2_518,
            "sha256": ROOT_PREPARATION_SHA,
        },
        "raw_stdout": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/STDOUT.raw.json",
            "bytes": RAW_STDOUT_BYTES,
            "sha256": RAW_STDOUT_SHA,
        },
        "transport": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1/TRANSPORT.json",
            "bytes": TRANSPORT_BYTES,
            "sha256": TRANSPORT_SHA,
        },
    }
    _eq(
        (
            acceptance.get("schema"),
            acceptance.get("status"),
            acceptance.get("run_name"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            *(acceptance.get(key) for key in expected_launch_acceptance),
            acceptance.get("nll_child_acceptance"),
            acceptance.get("nll_child_exact_stop"),
            acceptance.get("expected_updates_not_observed"),
            acceptance.get("scientific_fits_started"),
            acceptance.get("quality_score"),
        ),
        (
            "fg19-child-allthree-fit-launch-acceptance.v1",
            "ACTUAL_ONE_LAUNCH_ACCEPTED_NOT_TERMINAL_OR_QUALITY",
            RUN,
            UNIT,
            INVOCATION,
            *expected_launch_acceptance.values(),
            {
                "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/TERMINAL_ACCEPTANCE.json",
                "bytes": 15_860,
                "sha256": NLL_ACCEPTANCE_SHA,
            },
            {
                "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_nll_fit_v1/exact_stop/EXACT_STOP.json",
                "bytes": 3_140,
                "sha256": NLL_STOP_SHA,
            },
            44_760,
            1,
            None,
        ),
        "root actual AllThree launch acceptance",
    )


def _check_recipe_args(
    launch_args: Mapping[str, object], split_path: Path, cache_root: Path
) -> None:
    """Require the reviewed sparse/NLL variant and unchanged coupled recipe."""
    _eq(
        {
            key: launch_args.get(key)
            for key in (
                "name",
                "seed",
                "batch_order",
                "batch",
                "chunk",
                "epochs",
                "checkpoint_epoch",
                "init",
                "representation_mode",
                "representation_cache",
                "mean_variance_head",
                "mean_variance_objective",
                "loss",
                "loss_norm",
                "distill_cache_root",
                "optim",
                "precision",
                "no_eval_during_fit",
                "save_training_state",
                "grad_stats",
            )
        },
        {
            "name": RUN,
            "seed": 1001,
            "batch_order": "seeded",
            "batch": 16,
            "chunk": 2_000,
            "epochs": 6,
            "checkpoint_epoch": 5,
            "init": f"{PARENT_RUN}@ep3",
            "representation_mode": "ema_sparse",
            "representation_cache": str(DICTIONARY_PATH),
            "mean_variance_head": 1,
            "mean_variance_objective": "nll",
            "loss": "wmse",
            "loss_norm": "chunk",
            "distill_cache_root": str(cache_root),
            "optim": "adamw",
            "precision": "fp32",
            "no_eval_during_fit": True,
            "save_training_state": True,
            "grad_stats": True,
        },
        "AllThree fixed recipe",
    )
    _eq(
        (launch_args.get("split_manifest"), launch_args.get("device")),
        (str(split_path), "cuda"),
        "AllThree split and device",
    )


def _check_ema_sparse_auxiliary(
    auxiliary: object,
    provenance: Mapping[str, object],
    dictionary_payload: Mapping[str, object],
    dictionary_path: Path,
) -> dict[str, object]:
    """Bind frozen sparse dictionary state and the source-reviewed causal EMA mode."""
    if not isinstance(auxiliary, dict) or set(auxiliary) != {
        "norm_ema",
        "dynamics",
        "pool_memory",
        "readout_moments",
        "input_representation",
    }:
        raise ValueError("AllThree auxiliary-state schema differs")
    expected_dynamics = {
        "switch": None,
        "caps": {},
        "eta": 0.0,
        "clip": 1.0,
        "no_clip": False,
        "seed": 1001 + 104729,
        "is_switched": False,
        "events": [],
        "generators": {},
    }
    _eq(auxiliary.get("dynamics"), expected_dynamics, "AllThree fit-only optimizer dynamics")
    _eq(
        {key: auxiliary.get(key) for key in ("norm_ema", "pool_memory", "readout_moments")},
        {"norm_ema": None, "pool_memory": None, "readout_moments": None},
        "disabled parameter EMA/SWA auxiliary state",
    )
    dictionary_representation = dictionary_payload.get("representation")
    if not isinstance(dictionary_representation, dict):
        raise ValueError("accepted sparse dictionary representation is missing")
    control_v3._check_representation_payload(
        auxiliary.get("input_representation"), dictionary_representation, "AllThree training state"
    )
    identity = provenance.get("input_representation")
    if not isinstance(identity, dict) or set(identity) != {
        "mode",
        "state_sha256",
        "artifact_path",
        "artifact_file_sha256",
        "artifact_binding",
        "fit_metadata",
    }:
        raise ValueError("AllThree fitted-representation provenance is malformed")
    _eq(
        identity,
        {
            "mode": "ema_sparse",
            "state_sha256": dictionary_representation.get("state_sha256"),
            "artifact_path": str(dictionary_path),
            "artifact_file_sha256": control_v3.DICTIONARY_SHA,
            "artifact_binding": "shared_fit_cache",
            "fit_metadata": dictionary_representation.get("fit_metadata"),
        },
        "AllThree sparse dictionary and causal EMA ancestry",
    )
    return {
        "artifact_binding": identity["artifact_binding"],
        "artifact_sha256": identity["artifact_file_sha256"],
        "mode": "ema_sparse",
        "semantics": "signed top-4-of-16 codes plus 112-wide residual; causal innovation EMA half-life 32",
        "state_sha256": identity["state_sha256"],
    }


def _check_sample_log(
    log_path: Path, history: list[object], order_records: list[dict[str, object]]
) -> dict[str, object]:
    """Reconcile six native FIT and GRAD_STATS records with saved histories."""
    raw, digest, info = support._stable_file(log_path, control_v3.MAX_LOG_BYTES)
    fit_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    selections: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("FIT "):
            value = json.loads(line[4:])
            if isinstance(value, dict):
                fit_rows.append(value)
        elif line.startswith("GRAD_STATS "):
            value = json.loads(line[len("GRAD_STATS ") :])
            if isinstance(value, dict):
                gradient_rows.append(value)
        elif line.startswith("SELECTED_FIXED "):
            selections.append(line.removeprefix("SELECTED_FIXED "))
    if len(fit_rows) != control_v3.EPOCHS or len(gradient_rows) != control_v3.EPOCHS:
        raise ValueError("native log must contain all six FIT and GRAD_STATS records")
    for epoch, (fit, gradients, saved, order) in enumerate(
        zip(fit_rows, gradient_rows, history, order_records, strict=True)
    ):
        _eq(fit, saved, f"native FIT history epoch {epoch}")
        _eq(fit.get("epoch"), epoch, f"native FIT epoch {epoch}")
        _eq(
            gradients,
            saved.get("update_diagnostics", {}).get("grad_stats"),
            f"native GRAD_STATS epoch {epoch}",
        )
        _eq(order.get("epoch"), epoch, f"sample-order epoch {epoch}")
    _eq(selections, ["5"], "fixed epoch-five selection log")
    return {
        "bytes": info["bytes"],
        "fit_records": len(fit_rows),
        "fixed_selection": selections[0],
        "gradient_records": len(gradient_rows),
        "sha256": digest,
    }


def verify_terminal(
    run_dir: Path,
    source_root: Path,
    source_manifest: Path,
    split_path: Path,
    evidence_root: Path,
    profile_evidence_root: Path,
    admission_path: Path,
    launch_path: Path,
    launch_acceptance_path: Path,
    terminal_path: Path,
    cache_root: Path,
    cache_acceptance_path: Path,
    cache_stop_path: Path,
) -> dict[str, object]:
    """Verify the exact exited AllThree run, its six epochs, and fixed ancestry."""
    terminal, terminal_sha = support._json_file(terminal_path)
    native = _check_native_terminal(terminal)
    helper_closure = assert_helper_closure()
    path_exception = control_v3._require_path_compatibility_exception()
    control_v3._reset_safe_globals()
    _check_paths(source_root, run_dir, split_path, cache_root)
    admission, admission_sha = support._json_file(admission_path)
    launch, launch_sha = support._json_file(launch_path)
    launch_acceptance, launch_acceptance_sha = support._json_file(launch_acceptance_path)
    _check_launch_records(
        admission_sha,
        launch,
        launch_sha,
        launch_acceptance,
        launch_acceptance_sha,
    )
    _check_admission(admission)
    _check_predecessor_receipts(evidence_root, admission)
    source = parent_v3._source_identity(source_root, source_manifest, admission)
    data_stats = ema_v3._check_registered_data_stats(admission, source_root.parent / "data")
    split, split_sha = support._json_file(split_path)
    _eq((split_sha, admission.get("split_sha256")), (SPLIT_SHA, SPLIT_SHA), "F0 split manifest")
    parent_v3._check_worker(launch, admission, source_root)
    cache = parent_v3._check_cache(
        admission, cache_root, cache_acceptance_path, cache_stop_path, split
    )
    lineage = control_v3._check_predecessor_receipts(
        evidence_root, profile_evidence_root, admission
    )
    control = ema_v3._check_control_predecessor(evidence_root, admission)
    ema = nll_check._check_ema_predecessor(evidence_root, admission)
    dictionary_child = nll_check._check_dictionary_child_predecessor(evidence_root, admission)
    nll = _check_native_predecessor(evidence_root, admission)
    dictionary_path = (
        source_root.parent / "data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
    )
    ema_v3._check_parent_and_dictionary_stats(admission, source_root, dictionary_path)
    parent_acceptance, _ = control_v3._read_pinned_json(
        evidence_root / "PARENT_TERMINAL_ACCEPTANCE.json", control_v3.PARENT_ACCEPTANCE_SHA
    )
    parent_model, _ = control_v3._check_parent_checkpoint(source_root, parent_acceptance)
    _eq(
        control_v3.POSIXPATH_PICKLE_ARGS,
        (str(DICTIONARY_PATH),),
        "restricted exact Path decoder binding",
    )
    launch_args = teacher_s5._launch_cli_args(launch, source_root)
    launch_args.update(
        {
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "split_manifest_sha256": split_sha,
        }
    )
    _eq(ema_v3._trainer_argv_sha(launch), TRAINER_ARGV_SHA, "full AllThree argv digest")
    _check_recipe_args(launch_args, split_path, cache_root)
    representation_module = importlib.import_module("wnn.training.input_representation")
    representation_fit_module = importlib.import_module("wnn.training.input_representation_fit")
    dictionary_module = importlib.import_module("wnn.training.input_representation_dictionary")
    dictionary_payload, dictionary_identity = control_v3._check_dictionary_artifact(
        dictionary_path,
        split_sha,
        source_root,
        representation_module,
        representation_fit_module,
        dictionary_module,
    )
    provenance, provenance_sha = support._json_file(run_dir / "provenance.json")
    history_raw, history_sha, _ = support._stable_file(
        run_dir / "history.json", control_v3.MAX_JSON_BYTES
    )
    history = json.loads(history_raw)
    if not isinstance(history, list):
        raise ValueError("fit history must be a JSON list")
    caps = {
        "model.pt": control_v3.MAX_MODEL_BYTES,
        **{f"model_ep{epoch}.pt": control_v3.MAX_MODEL_BYTES for epoch in range(control_v3.EPOCHS)},
        "training_state.pt": control_v3.MAX_STATE_BYTES,
        "provenance.json": control_v3.MAX_JSON_BYTES,
        "history.json": control_v3.MAX_JSON_BYTES,
    }
    paths = {name: run_dir / name for name in caps}
    files = {name: support._hash_file(paths[name], cap) for name, cap in caps.items()}
    if sum(item[1]["bytes"] for item in files.values()) > control_v3.MAX_AGGREGATE_BYTES:
        raise ValueError("AllThree terminal artifacts exceed the aggregate byte cap")
    _eq(
        (provenance_sha, history_sha),
        (files["provenance.json"][0], files["history.json"][0]),
        "stable fit metadata",
    )
    _eq(
        (
            provenance.get("selected_epoch"),
            provenance.get("split_manifest_sha256"),
            provenance.get("seed"),
            provenance.get("initial_checkpoint_sha256"),
            provenance.get("propensity_checkpoint_sha256"),
        ),
        (5, SPLIT_SHA, 1001, control_v3.PARENT_CHECKPOINT_SHA, None),
        "selected AllThree endpoint and parent initialization",
    )
    expected_args = dict(launch_args)
    _eq(provenance.get("args"), expected_args, "AllThree provenance arguments")
    membership = parent_v3._check_membership(provenance, split)
    parent_v3._check_cache_provenance(provenance, cache_root)
    tensor_identity = teacher_s5._canonical_tensor_identity(source_root)
    models: dict[str, object] = {}
    fingerprints: dict[str, tuple[str, int]] = {}
    for name in ("model.pt", "model_ep5.pt"):
        payload, digest, stat_info = parent_v3._load_checkpoint(
            paths[name], control_v3.MAX_MODEL_BYTES
        )
        _eq((digest, stat_info), files[name], f"stable {name}")
        _eq(
            (payload.get("epoch"), payload.get("n_in"), payload.get("args")),
            (5, 467, expected_args),
            f"{name} selected endpoint",
        )
        models[name] = payload.get("model")
        fingerprints[name] = control_v3._model_fingerprint(models[name])
        _eq(
            {
                key: payload.get("provenance", {}).get(key)
                for key in control_v3.CHECKPOINT_PROVENANCE_KEYS
            },
            {key: provenance.get(key) for key in control_v3.CHECKPOINT_PROVENANCE_KEYS},
            f"{name} provenance",
        )
        control_v3._check_representation_payload(
            payload.get("input_representation"), dictionary_payload["representation"], name
        )
    _eq(fingerprints["model.pt"], fingerprints["model_ep5.pt"], "selected ep5 tensor identity")
    normalizer = control_v3._check_child_normalizer(
        models["model.pt"], parent_model, provenance, tensor_identity
    )
    state, state_sha, state_stat, state_path_review = control_v3._load_control_training_state(
        paths["training_state.pt"], control_v3.MAX_STATE_BYTES
    )
    _eq((state_sha, state_stat), files["training_state.pt"], "stable training state")
    _eq(
        (state.get("format"), state.get("version"), state.get("boundary"), state.get("next_epoch")),
        ("wnn.epoch-training-state", 1, "epoch", 6),
        "six-epoch saved boundary",
    )
    _eq(set(state), control_v3.TRAINING_STATE_KEYS, "training-state schema")
    _eq(
        (state.get("ema"), state.get("swa"), state.get("scaler")),
        (None, None, {}),
        "parameter EMA/SWA/scaler disabled",
    )
    representation = _check_ema_sparse_auxiliary(
        state.get("auxiliary"), provenance, dictionary_payload, dictionary_path
    )
    _eq(
        control_v3._model_fingerprint(state.get("model")),
        fingerprints["model.pt"],
        "training-state selected model identity",
    )
    scheduler_lr, scheduler_beta1 = control_v3._check_scheduler(state.get("scheduler"))
    optimizer = control_v3._check_optimizer(state.get("optimizer"), scheduler_lr, scheduler_beta1)
    sample_order = control_v3._check_counters(history, state, provenance)
    rng = control_v3._check_loader_and_rng(state)
    manifest, _ = support._json_file(source_manifest)
    control_v3._check_args_and_contract(
        state, provenance, admission, expected_args, manifest, split_sha
    )
    _eq(
        provenance.get("checkpoint_sha256"),
        {
            name: files[name][0]
            for name in ("model.pt", *(f"model_ep{epoch}.pt" for epoch in range(control_v3.EPOCHS)))
        },
        "six epoch checkpoint digest map",
    )
    native_log = _check_sample_log(evidence_root / "unit.log", history, sample_order)
    return {
        "schema": "fg19.child-allthree-fit-terminal-check.v1",
        "status": "PASS",
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "quality_score": None,
        "terminal_evidence_sha256": terminal_sha,
        "launch_sha256": LAUNCH_SHA,
        "launch_acceptance_sha256": LAUNCH_ACCEPTANCE_SHA,
        "native": native,
        "source": source,
        "split_sha256": split_sha,
        "cache": cache,
        "registered_data_stats_only": data_stats,
        "membership_counts": membership,
        "helper_closure": helper_closure,
        "observed_optimizer_updates": control_v3.UPDATES,
        "observed_completed_batches": control_v3.BATCHES,
        "next_epoch": 6,
        "selected_epoch": 5,
        "model_tensor_count": fingerprints["model.pt"][1],
        "model_tensor_fingerprint": fingerprints["model.pt"][0],
        "normalization_identity": normalizer,
        "input_representation": representation,
        "approved_path_compatibility_exception": path_exception,
        "training_state_path_compatibility": state_path_review,
        "accepted_dictionary": dictionary_identity,
        "optimizer": optimizer,
        "rng_observed_after_restricted_load": rng,
        "sample_order_observations": sample_order,
        "predecessor_ancestry": {
            **lineage,
            "control": control,
            "ema_sequential_predecessor": ema,
            "dictionary_child_sequential_predecessor": dictionary_child,
            "nll_sequential_predecessor": nll,
        },
        "native_log": native_log,
        "files": {
            name: {"sha256": item[0], "bytes": item[1]["bytes"]} for name, item in files.items()
        },
    }
