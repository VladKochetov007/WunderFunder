"""Verify one exited F0 seed-1001 sparse dictionary child fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_HELPER_DIR = Path(__file__).resolve().parent
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))

control_v3 = importlib.import_module("child_control_terminal_check_v3")
parent_v3 = importlib.import_module("parent_fit_terminal_check_v3")
support = importlib.import_module("propensity_terminal_support")
teacher_s5 = importlib.import_module("teacher_s5_terminal_check_v1")
ema_v3 = importlib.import_module("child_ema_terminal_check_v3")

_json_file = support._json_file
_hash_file = support._hash_file
_stable_file = support._stable_file
_launch_cli_args = teacher_s5._launch_cli_args
_canonical_tensor_identity = teacher_s5._canonical_tensor_identity

RUN = "fg19_child_dictionary_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
INVOCATION = "73b10b699e034a1b9952b2377b845095"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
TRAINER_ARGV_SHA = "096b036c8af6220512e51b2002f3598729274b482d5db4b1df8099b26f91ab6d"
ADMISSION_SHA = "b13ccf5834fd210ce02ca6c3aea7251d8dcae3f63823341da15a77abcbc75063"
LAUNCH_SHA = "9cbcc8b4df3886620429cae42d678d2e281aff980402f5b19d63d9f515b6ad3c"
LAUNCH_ACCEPTANCE_SHA = "33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95"
BODY_SHA = "9f545489dd80c13831b1a02c75dddf4059f3cbfb84311b2400db4830b4fdd77e"
BODY_BYTES = 196_111
ROOT_PREPARATION_SHA = "2c0ee4b298a744f3c3204c46f5a269f3ea03c73d1a521c6db05dfa7a4144cf66"
SOURCE_REVIEW_SHA = "67528bf25430a7bd422e233f3ce84059b14d372f59b9b033cdd70e674634528a"
SOURCE_REVIEW_BYTES = 7_213
LAUNCH_ACCEPTANCE_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/LAUNCH_ACCEPTANCE.json"
ADMISSION_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/ADMISSION.json"
)
LAUNCH_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/LAUNCH.json"
)
BODY_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/REMOTE_SUBMISSION.py"
ROOT_PREPARATION_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/ROOT_ADMISSION_PREPARATION.json"
SOURCE_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/DICTIONARY_FINAL_BODY_SOURCE_REVIEW_V2.json"
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
CONTROL_ACCEPTANCE_SHA = "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
CONTROL_STOP_SHA = "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
CONTROL_RUN = "fg19_child_control_f0_s1001_fit_v1"
CONTROL_UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
CONTROL_INVOCATION = "e66e0d0703314e4fb5d47dcdbfcaf202"
EMA_ACCEPTANCE_SHA = "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8"
EMA_STOP_SHA = "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf"
EMA_RUN = "fg19_child_ema_f0_s1001_fit_v1"
EMA_UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
EMA_INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
EMA_MODEL_EP5_SHA = "6e90bc4dba8add0caf670097519f08f3ce239d0b64c37c47b3760aa22af41862"
DICTIONARY_PATH = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
)
GPU_ID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
REGISTERED_DATA_NAMES = ema_v3.REGISTERED_DATA_NAMES
EVIDENCE_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": control_v3.PARENT_ACCEPTANCE_SHA,
    "PARENT_EXACT_STOP.json": control_v3.PARENT_STOP_SHA,
    "DICTIONARY_ACCEPTANCE.json": control_v3.DICTIONARY_ACCEPTANCE_SHA,
    "DICTIONARY_EXACT_STOP.json": control_v3.DICTIONARY_STOP_SHA,
    "PROFILE_TERMINAL_ACCEPTANCE.json": control_v3.PROFILE_ACCEPTANCE_SHA,
    "PROFILE_EXACT_STOP.json": control_v3.PROFILE_STOP_SHA,
    "CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA,
    "CONTROL_EXACT_STOP.json": CONTROL_STOP_SHA,
    "EMA_TERMINAL_ACCEPTANCE.json": EMA_ACCEPTANCE_SHA,
    "EMA_EXACT_STOP.json": EMA_STOP_SHA,
}
HELPER_PINS = {
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
        ema_v3,
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
        control_v3,
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
    """Require an exact identity match without type coercion."""
    control_v3._eq(actual, expected, label)


def _check_helper_closure() -> dict[str, dict[str, object]]:
    """Require each imported pure helper to be the pinned private sibling file."""
    observed: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha, module) in HELPER_PINS.items():
        path = _HELPER_DIR / name
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        if path.is_symlink() or module_path != path.resolve(strict=True):
            raise ValueError(f"helper was imported from outside the private closure: {name}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        _eq((len(raw), digest), (expected_bytes, expected_sha), f"pinned helper {name}")
        observed[name] = {"bytes": len(raw), "sha256": digest}
    return observed


def _check_native_terminal(terminal: Mapping[str, object]) -> None:
    """Require strict native-exit evidence without requiring a separate stop."""
    _eq(
        (
            terminal.get("schema"),
            terminal.get("run_name"),
            terminal.get("unit"),
            terminal.get("invocation_id"),
        ),
        ("fg19.child-dictionary-fit-native-terminal-evidence.v1", RUN, UNIT, INVOCATION),
        "dictionary native terminal identity",
    )
    _eq(
        (terminal.get("active_state"), terminal.get("sub_state")),
        ("active", "exited"),
        "native state",
    )
    for key, value in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        control_v3._int(terminal.get(key), value, key)
    if type(terminal.get("raw_bytes")) is not int or terminal["raw_bytes"] <= 0:
        raise ValueError("native raw capture byte count is invalid")
    if not control_v3._is_sha256(terminal.get("raw_sha256")):
        raise ValueError("native raw capture digest is malformed")
    if not isinstance(terminal.get("raw_path"), str) or not terminal["raw_path"]:
        raise ValueError("native raw capture path is missing")
    for key in ("captured_utc", "native_exit_utc"):
        stamp = terminal.get(key)
        if not isinstance(stamp, str):
            raise ValueError(f"native {key} is missing")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"native {key} is not ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError(f"native {key} has no timezone offset")


def _control_predecessor() -> dict[str, object]:
    """Return the exact already accepted CONTROL identity."""
    return {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA,
        "exact_stop_sha256": CONTROL_STOP_SHA,
        "invocation_id": CONTROL_INVOCATION,
        "run_name": CONTROL_RUN,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": CONTROL_UNIT,
    }


def _ema_predecessor() -> dict[str, object]:
    """Return the exact accepted EMA identity used as a sequential gate only."""
    return {
        "acceptance_sha256": EMA_ACCEPTANCE_SHA,
        "exact_stop_sha256": EMA_STOP_SHA,
        "invocation_id": EMA_INVOCATION,
        "model_ep5_sha256": EMA_MODEL_EP5_SHA,
        "run_name": EMA_RUN,
        "selected_epoch": 5,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "successful_updates": 44_760,
        "unit": EMA_UNIT,
    }


def _check_launch_records(
    admission: Mapping[str, object],
    admission_sha: str,
    launch: Mapping[str, object],
    launch_sha: str,
    acceptance: Mapping[str, object],
    acceptance_sha: str,
) -> None:
    """Bind the one original actual launch to its immutable body and review."""
    _eq(
        (admission_sha, launch_sha, acceptance_sha),
        (ADMISSION_SHA, LAUNCH_SHA, LAUNCH_ACCEPTANCE_SHA),
        "launch receipt pins",
    )
    sub = launch.get("submission")
    _eq(
        (
            launch.get("schema"),
            launch.get("unit"),
            isinstance(sub, dict),
            sub.get("outcome") if isinstance(sub, dict) else None,
            sub.get("returncode") if isinstance(sub, dict) else None,
        ),
        ("fg19-child-dictionary-fit-launch.v1", UNIT, True, "submitted", 0),
        "dictionary original launch",
    )
    _eq(launch.get("control_predecessor"), _control_predecessor(), "CONTROL launch predecessor")
    _eq(launch.get("ema_predecessor"), _ema_predecessor(), "EMA sequential launch predecessor")
    _eq(
        launch.get("unknown_outcome_policy"),
        "never retry; inspect this exact unit and durable dictionary output",
        "no-retry launch policy",
    )
    root_acceptance = {
        "admission": {"path": ADMISSION_PATH, "bytes": 10_937, "sha256": ADMISSION_SHA},
        "launch": {"path": LAUNCH_PATH, "bytes": 8_072, "sha256": LAUNCH_SHA},
        "body": {"path": BODY_PATH, "bytes": BODY_BYTES, "sha256": BODY_SHA},
        "source_review": {
            "path": SOURCE_REVIEW_PATH,
            "bytes": SOURCE_REVIEW_BYTES,
            "sha256": SOURCE_REVIEW_SHA,
        },
        "root_preparation": {
            "path": ROOT_PREPARATION_PATH,
            "bytes": 2_343,
            "sha256": ROOT_PREPARATION_SHA,
        },
    }
    _eq(
        (
            acceptance.get("schema"),
            acceptance.get("status"),
            acceptance.get("run_name"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            acceptance.get("body"),
            acceptance.get("admission"),
            acceptance.get("launch"),
            acceptance.get("source_review"),
            acceptance.get("root_preparation"),
        ),
        (
            "fg19-child-dictionary-fit-launch-acceptance.v1",
            "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
            RUN,
            UNIT,
            INVOCATION,
            root_acceptance["body"],
            root_acceptance["admission"],
            root_acceptance["launch"],
            root_acceptance["source_review"],
            root_acceptance["root_preparation"],
        ),
        "root actual launch acceptance",
    )
    _eq(
        (
            acceptance.get("parent_initializer"),
            acceptance.get("representation_mode"),
            acceptance.get("mean_variance_objective"),
            acceptance.get("native_exit_observed"),
            acceptance.get("expected_terminal_updates_not_observed"),
            acceptance.get("scientific_fits_newly_started"),
            acceptance.get("scientific_fits_newly_completed"),
            acceptance.get("quality_score"),
        ),
        (f"{PARENT_RUN}@ep3", "sparse", "wmse", False, 44_760, 1, 0, None),
        "launch acceptance scope",
    )
    _eq(
        admission.get("control_predecessor"),
        _control_predecessor(),
        "CONTROL admission predecessor",
    )
    _eq(admission.get("ema_predecessor"), _ema_predecessor(), "EMA admission predecessor")


def _check_admission(admission: Mapping[str, object]) -> None:
    """Require the frozen sparse recipe, shared ancestors, and resource envelope."""
    _eq(
        (
            admission.get("schema"),
            admission.get("run_name"),
            admission.get("unit"),
            admission.get("source_commit"),
            admission.get("source_manifest_sha256"),
            admission.get("source_files_verified"),
            admission.get("split_sha256"),
        ),
        (
            "fg19-child-dictionary-fit-admission.v1",
            RUN,
            UNIT,
            COMMIT,
            SOURCE_MANIFEST_SHA,
            167,
            SPLIT_SHA,
        ),
        "dictionary admission identity/source",
    )
    _eq(
        admission.get("staged_receipt_sha256"),
        EVIDENCE_RECEIPTS,
        "exact staged ancestry receipt map",
    )
    _eq(admission.get("populated"), {}, "empty successor descendants at launch")
    recipe = admission.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("admitted sparse recipe is missing")
    _eq(
        {
            key: recipe.get(key)
            for key in (
                "batch_size",
                "checkpoint_epoch_zero_based",
                "chunk_rows",
                "epochs",
                "evaluation_during_fit",
                "expected_optimizer_updates",
                "objective",
                "representation_mode",
                "seed",
                "trainer_argv_sha256",
            )
        },
        {
            "batch_size": 16,
            "checkpoint_epoch_zero_based": 5,
            "chunk_rows": 2_000,
            "epochs": 6,
            "evaluation_during_fit": False,
            "expected_optimizer_updates": 44_760,
            "objective": "wmse",
            "representation_mode": "sparse",
            "seed": 1001,
            "trainer_argv_sha256": TRAINER_ARGV_SHA,
        },
        "sparse fit recipe",
    )
    _eq(
        admission.get("dictionary"),
        {
            "bytes": 22_557,
            "device": 64_512,
            "inode": 733_618_561,
            "path": str(DICTIONARY_PATH),
            "sha256": control_v3.DICTIONARY_SHA,
        },
        "shared immutable dictionary path/stat/digest",
    )
    _eq(
        (
            admission.get("parent_acceptance_sha256"),
            admission.get("parent_exact_stop_sha256"),
            admission.get("parent_checkpoint_sha256"),
            admission.get("child_profile", {}).get("acceptance_sha256"),
            admission.get("child_profile", {}).get("exact_stop_sha256"),
        ),
        (
            control_v3.PARENT_ACCEPTANCE_SHA,
            control_v3.PARENT_STOP_SHA,
            control_v3.PARENT_CHECKPOINT_SHA,
            control_v3.PROFILE_ACCEPTANCE_SHA,
            control_v3.PROFILE_STOP_SHA,
        ),
        "parent and profile ancestry",
    )


def _check_ema_predecessor(
    evidence_root: Path, admission: Mapping[str, object]
) -> dict[str, object]:
    """Recheck accepted EMA receipts as a sequential gate, never as initializer."""
    acceptance, acceptance_sha = _json_file(evidence_root / "EMA_TERMINAL_ACCEPTANCE.json")
    stop, stop_sha = _json_file(evidence_root / "EMA_EXACT_STOP.json")
    _eq(
        (acceptance_sha, stop_sha),
        (EMA_ACCEPTANCE_SHA, EMA_STOP_SHA),
        "EMA predecessor receipt pins",
    )
    _eq(
        (
            acceptance.get("schema"),
            acceptance.get("status"),
            acceptance.get("run_name"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            acceptance.get("source_commit"),
            acceptance.get("source_file_count"),
            acceptance.get("source_manifest_sha256"),
            acceptance.get("successful_updates"),
            acceptance.get("completed_batches"),
            acceptance.get("amp_skipped_updates"),
            acceptance.get("nonfinite_updates"),
            acceptance.get("next_epoch"),
            acceptance.get("selected_epoch"),
            acceptance.get("epochs"),
            acceptance.get("strict_fp32_finite_model_optimizer_verified"),
            acceptance.get("selected_model_epoch5_and_training_state_tensor_identity_verified"),
            acceptance.get("representation_mode"),
            acceptance.get("input_representation", {}).get("mode"),
        ),
        (
            "fg19-child-ema-fit-terminal-acceptance.v1",
            "ACCEPTED",
            EMA_RUN,
            EMA_UNIT,
            EMA_INVOCATION,
            COMMIT,
            167,
            SOURCE_MANIFEST_SHA,
            44_760,
            4_476,
            0,
            0,
            6,
            5,
            6,
            True,
            True,
            "ema",
            "ema",
        ),
        "EMA accepted fit integrity and representation",
    )
    _eq(
        acceptance.get("files", {}).get("model_ep5.pt", {}).get("sha256"),
        EMA_MODEL_EP5_SHA,
        "accepted EMA ep5 checkpoint digest",
    )
    _eq(
        (
            acceptance.get("main_pid"),
            acceptance.get("exec_main_code"),
            acceptance.get("exec_main_status"),
            acceptance.get("quality_score"),
        ),
        (0, 1, 0, None),
        "EMA predecessor exit and no quality",
    )
    control_v3._check_stop_receipt(stop, EMA_UNIT, EMA_INVOCATION, EMA_ACCEPTANCE_SHA)
    _eq(admission.get("ema_predecessor"), _ema_predecessor(), "EMA sequential predecessor record")
    return {
        "acceptance_sha256": EMA_ACCEPTANCE_SHA,
        "exact_stop_sha256": EMA_STOP_SHA,
        "invocation_id": EMA_INVOCATION,
        "model_ep5_sha256": EMA_MODEL_EP5_SHA,
        "run_name": EMA_RUN,
        "selected_epoch": 5,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "successful_updates": 44_760,
        "unit": EMA_UNIT,
    }


def _check_recipe_args(
    launch_args: Mapping[str, object], split_path: Path, cache_root: Path
) -> None:
    """Require source-parsed sparse-mode arguments and the frozen six-epoch recipe."""
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
            "representation_mode": "sparse",
            "representation_cache": str(DICTIONARY_PATH),
            "mean_variance_head": 1,
            "mean_variance_objective": "wmse",
            "loss": "wmse",
            "distill_cache_root": str(cache_root),
            "optim": "adamw",
            "precision": "fp32",
            "no_eval_during_fit": True,
            "save_training_state": True,
            "grad_stats": True,
        },
        "sparse scientific recipe",
    )
    _eq(
        (launch_args.get("split_manifest"), launch_args.get("device")),
        (str(split_path), "cuda"),
        "split and device",
    )


def _check_sparse_auxiliary(
    auxiliary: object, provenance: Mapping[str, object], dictionary: Mapping[str, object]
) -> dict[str, object]:
    """Require sparse-mode provenance while preserving the accepted dictionary state."""
    if not isinstance(auxiliary, dict) or set(auxiliary) != {
        "norm_ema",
        "dynamics",
        "pool_memory",
        "readout_moments",
        "input_representation",
    }:
        raise ValueError("dictionary auxiliary state schema differs")
    _eq(
        auxiliary.get("dynamics"),
        {
            "switch": None,
            "caps": {},
            "eta": 0.0,
            "clip": 1.0,
            "no_clip": False,
            "seed": 1001 + 104729,
            "is_switched": False,
            "events": [],
            "generators": {},
        },
        "fit-only dynamics",
    )
    _eq(
        {key: auxiliary.get(key) for key in ("norm_ema", "pool_memory", "readout_moments")},
        {"norm_ema": None, "pool_memory": None, "readout_moments": None},
        "disabled auxiliary state",
    )
    representation = dictionary.get("representation")
    if not isinstance(representation, dict):
        raise ValueError("accepted sparse dictionary representation is missing")
    control_v3._check_representation_payload(
        auxiliary.get("input_representation"), representation, "dictionary training state"
    )
    actual = provenance.get("input_representation")
    expected = {
        "mode": "sparse",
        "state_sha256": representation.get("state_sha256"),
        "artifact_path": str(DICTIONARY_PATH),
        "artifact_file_sha256": control_v3.DICTIONARY_SHA,
        "artifact_binding": "shared_fit_cache",
        "fit_metadata": representation.get("fit_metadata"),
    }
    _eq(actual, expected, "sparse dictionary provenance")
    return {
        "mode": "sparse",
        "state_sha256": expected["state_sha256"],
        "artifact_sha256": control_v3.DICTIONARY_SHA,
        "artifact_binding": "shared_fit_cache",
    }


def _check_paths(source_root: Path, run_dir: Path, split_path: Path, cache_root: Path) -> None:
    """Restrict verification to the canonical source-68 run and cache paths."""
    source = source_root.resolve(strict=True)
    base = source.parent
    _eq(source.name, "source-fg19-cache-68ff6829", "source directory")
    for label, path, expected in (
        ("source root", source_root, base / "source-fg19-cache-68ff6829"),
        ("dictionary run", run_dir, base / "data/artifacts/runs" / RUN),
        ("split", split_path, base / "splits/fold0.json"),
        ("T3 cache", cache_root, base / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"),
    ):
        if path.is_symlink():
            raise ValueError(f"{label} cannot be a symlink")
        _eq(path.resolve(strict=True), expected.resolve(strict=True), label)


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
    """Validate the exact exited dictionary fit and all bounded ancestry evidence."""
    path_exception = control_v3._require_path_compatibility_exception()
    control_v3._reset_safe_globals()
    helper_closure = _check_helper_closure()
    _check_paths(source_root, run_dir, split_path, cache_root)
    admission, admission_sha = _json_file(admission_path)
    launch, launch_sha = _json_file(launch_path)
    launch_acceptance, launch_acceptance_sha = _json_file(launch_acceptance_path)
    terminal, terminal_sha = _json_file(terminal_path)
    _check_launch_records(
        admission, admission_sha, launch, launch_sha, launch_acceptance, launch_acceptance_sha
    )
    _check_admission(admission)
    _check_native_terminal(terminal)
    source = parent_v3._source_identity(source_root, source_manifest, admission)
    data_stats = ema_v3._check_registered_data_stats(admission, source_root.parent / "data")
    split, split_sha = _json_file(split_path)
    _eq(
        (split_sha, admission.get("split_sha256")),
        (SPLIT_SHA, SPLIT_SHA),
        "canonical split manifest",
    )
    parent_v3._check_worker(launch, admission, source_root)
    cache = parent_v3._check_cache(
        admission, cache_root, cache_acceptance_path, cache_stop_path, split
    )
    lineage = control_v3._check_predecessor_receipts(
        evidence_root, profile_evidence_root, admission
    )
    control = ema_v3._check_control_predecessor(evidence_root, admission)
    ema = _check_ema_predecessor(evidence_root, admission)
    dictionary_path = (
        source_root.parent / "data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
    )
    ema_v3._check_parent_and_dictionary_stats(admission, source_root, dictionary_path)
    parent_acceptance, _ = control_v3._read_pinned_json(
        evidence_root / "PARENT_TERMINAL_ACCEPTANCE.json", control_v3.PARENT_ACCEPTANCE_SHA
    )
    parent_model, _ = control_v3._check_parent_checkpoint(source_root, parent_acceptance)
    _eq(
        control_v3.POSIXPATH_PICKLE_ARGS, (str(DICTIONARY_PATH),), "restricted Path decoder binding"
    )
    launch_args = _launch_cli_args(launch, source_root)
    launch_args.update(
        {
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "split_manifest_sha256": split_sha,
        }
    )
    _eq(ema_v3._trainer_argv_sha(launch), TRAINER_ARGV_SHA, "full sparse CLI digest")
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
    provenance, provenance_sha = _json_file(run_dir / "provenance.json")
    history_raw, history_sha, _ = _stable_file(run_dir / "history.json", control_v3.MAX_JSON_BYTES)
    history = json.loads(history_raw)
    if not isinstance(history, list):
        raise ValueError("fit history must be a list")
    caps = {
        "model.pt": control_v3.MAX_MODEL_BYTES,
        **{f"model_ep{epoch}.pt": control_v3.MAX_MODEL_BYTES for epoch in range(control_v3.EPOCHS)},
        "training_state.pt": control_v3.MAX_STATE_BYTES,
        "provenance.json": control_v3.MAX_JSON_BYTES,
        "history.json": control_v3.MAX_JSON_BYTES,
    }
    paths = {name: run_dir / name for name in caps}
    files = {name: _hash_file(paths[name], cap) for name, cap in caps.items()}
    if sum(item[1]["bytes"] for item in files.values()) > control_v3.MAX_AGGREGATE_BYTES:
        raise ValueError("dictionary terminal artifacts exceed aggregate byte cap")
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
        "selected sparse endpoint and parent initialization",
    )
    expected_args = dict(launch_args)
    _eq(provenance.get("args"), expected_args, "sparse provenance arguments")
    membership = parent_v3._check_membership(provenance, split)
    parent_v3._check_cache_provenance(provenance, cache_root)
    tensor_identity = _canonical_tensor_identity(source_root)
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
            f"{name} endpoint",
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
    _eq(
        fingerprints["model.pt"],
        fingerprints["model_ep5.pt"],
        "selected checkpoint equals fixed epoch five",
    )
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
        "disabled EMA/SWA/scaler state",
    )
    representation = _check_sparse_auxiliary(state.get("auxiliary"), provenance, dictionary_payload)
    _eq(
        control_v3._model_fingerprint(state.get("model")),
        fingerprints["model.pt"],
        "training-state selected model identity",
    )
    scheduler_lr, scheduler_beta1 = control_v3._check_scheduler(state.get("scheduler"))
    optimizer = control_v3._check_optimizer(state.get("optimizer"), scheduler_lr, scheduler_beta1)
    sample_order = control_v3._check_counters(history, state, provenance)
    rng = control_v3._check_loader_and_rng(state)
    manifest, _ = _json_file(source_manifest)
    control_v3._check_args_and_contract(
        state, provenance, admission, expected_args, manifest, split_sha
    )
    _eq(
        provenance.get("checkpoint_sha256"),
        {
            name: files[name][0]
            for name in ("model.pt", *(f"model_ep{epoch}.pt" for epoch in range(control_v3.EPOCHS)))
        },
        "checkpoint digest map",
    )
    native_log = control_v3._check_sample_log(evidence_root / "unit.log", history, sample_order)
    return {
        "schema": "fg19.child-dictionary-fit-terminal-check.v1",
        "status": "PASS",
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "terminal_evidence_sha256": terminal_sha,
        "launch_sha256": launch_sha,
        "launch_acceptance_sha256": launch_acceptance_sha,
        "source": source,
        "split_sha256": split_sha,
        "cache": cache,
        "registered_data_stats_only": data_stats,
        "membership_counts": membership,
        "helper_closure": helper_closure,
        "observed_optimizer_updates": control_v3.UPDATES,
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
        "predecessor_ancestry": {**lineage, "control": control, "ema_sequential_predecessor": ema},
        "native_log": native_log,
        "files": {
            name: {"sha256": item[0], "bytes": item[1]["bytes"]} for name, item in files.items()
        },
    }


def main() -> int:
    """Run the bounded terminal checker and create a private no-overwrite result."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "run-dir",
        "source-root",
        "source-manifest",
        "split",
        "evidence-root",
        "profile-evidence-root",
        "admission",
        "launch",
        "launch-acceptance",
        "terminal-evidence",
        "cache-root",
        "cache-acceptance",
        "cache-exact-stop",
        "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    result = verify_terminal(
        args.run_dir,
        args.source_root,
        args.source_manifest,
        args.split,
        args.evidence_root,
        args.profile_evidence_root,
        args.admission,
        args.launch,
        args.launch_acceptance,
        args.terminal_evidence,
        args.cache_root,
        args.cache_acceptance,
        args.cache_exact_stop,
    )
    raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        __import__("os").fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
