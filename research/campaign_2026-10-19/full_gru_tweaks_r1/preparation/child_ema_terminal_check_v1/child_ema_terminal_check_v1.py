"""Verify the pinned F0 seed-1001 EMA representation fit after native exit."""

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
_hash_file = support._hash_file
_json_file = support._json_file
_stable_file = support._stable_file
_canonical_tensor_identity = teacher_s5._canonical_tensor_identity
_launch_cli_args = teacher_s5._launch_cli_args

RUN = "fg19_child_ema_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
ADMISSION_SHA = "db13e5433cb0a414c1a235806220c53db4ded6c128fc5ffbda5e465dde9fde51"
LAUNCH_SHA = "8c43f6fb26d3f6a046304617094e266a3e1940c83a7ea56787781fa176ef133e"
LAUNCH_ACCEPTANCE_SHA = "4fc34f4769da2f08aaf1fcca97d87731dc38b1709a93afa9376be003c28ffb72"
BODY_SHA = "f83ac3120a2e445f1752fcd026123a0e8a21ba53594d27b398462a5236ffebd6"
BODY_BYTES = 163_040
TRAINER_ARGV_SHA = "a20b4ede3b501daa6cd39e847c212ca3d6215dde81e130a8ab09a48004beb449"
LAUNCH_ACCEPTANCE_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/LAUNCH_ACCEPTANCE.json"
ADMISSION_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/ADMISSION.json"
)
LAUNCH_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/LAUNCH.json"
BODY_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_ema_fit_v1/REMOTE_SUBMISSION.py"
)
SOURCE_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/CHILD_EMA_FIT_SOURCE_REVIEW_V1.json"
SOURCE_REVIEW_SHA = "3d2f78b0c8902c1a43748a237310ae6a31277e60fa9c18c13a5d162c13722138"
SOURCE_REVIEW_BYTES = 8_000
CONTROL_ACCEPTANCE_SHA = "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
CONTROL_STOP_SHA = "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
CONTROL_RUN = "fg19_child_control_f0_s1001_fit_v1"
CONTROL_UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
CONTROL_INVOCATION = "e66e0d0703314e4fb5d47dcdbfcaf202"
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
DICTIONARY_PATH = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
)
EVIDENCE_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": control_v3.PARENT_ACCEPTANCE_SHA,
    "PARENT_EXACT_STOP.json": control_v3.PARENT_STOP_SHA,
    "DICTIONARY_ACCEPTANCE.json": control_v3.DICTIONARY_ACCEPTANCE_SHA,
    "DICTIONARY_EXACT_STOP.json": control_v3.DICTIONARY_STOP_SHA,
    "PROFILE_TERMINAL_ACCEPTANCE.json": control_v3.PROFILE_ACCEPTANCE_SHA,
    "PROFILE_EXACT_STOP.json": control_v3.PROFILE_STOP_SHA,
    "CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA,
    "CONTROL_EXACT_STOP.json": CONTROL_STOP_SHA,
}
REGISTERED_DATA_NAMES = (
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "datasets/train.parquet",
    "datasets/valid.parquet",
)


def _eq(actual: object, expected: object, label: str) -> None:
    """Require an exact, non-coercing identity match."""
    control_v3._eq(actual, expected, label)


def _check_paths(source_root: Path, run_dir: Path, split_path: Path, cache_root: Path) -> None:
    """Restrict the checker to the immutable source-68 EMA paths."""
    source = source_root.resolve(strict=True)
    base = source.parent
    _eq(source.name, "source-fg19-cache-68ff6829", "staged source directory")
    targets = {
        "source root": (source_root, base / "source-fg19-cache-68ff6829"),
        "EMA run directory": (run_dir, base / "data/artifacts/runs" / RUN),
        "split manifest": (split_path, base / "splits/fold0.json"),
        "cache root": (cache_root, base / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"),
    }
    for label, (path, expected) in targets.items():
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
        _eq(path.resolve(strict=True), expected.resolve(strict=True), label)


def _check_native_terminal(terminal: Mapping[str, object]) -> None:
    """Require EMA-specific typed native exit evidence before any stop."""
    _eq(
        (
            terminal.get("schema"),
            terminal.get("unit"),
            terminal.get("run_name"),
            terminal.get("invocation_id"),
        ),
        ("fg19.child-ema-fit-native-terminal-evidence.v1", UNIT, RUN, INVOCATION),
        "EMA native terminal identity",
    )
    _eq(
        (terminal.get("active_state"), terminal.get("sub_state")),
        ("active", "exited"),
        "EMA native state",
    )
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        control_v3._int(terminal.get(key), expected, key)
    if type(terminal.get("raw_bytes")) is not int or terminal["raw_bytes"] <= 0:
        raise ValueError("EMA native raw capture byte count is invalid")
    if not control_v3._is_sha256(terminal.get("raw_sha256")):
        raise ValueError("EMA native raw capture digest is malformed")
    if not isinstance(terminal.get("raw_path"), str) or not terminal["raw_path"]:
        raise ValueError("EMA native raw capture path is missing")
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        if not isinstance(value, str):
            raise ValueError(f"EMA native terminal {key} is missing")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"EMA native terminal {key} is not ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError(f"EMA native terminal {key} lacks an offset")


def _check_launch_records(
    admission: Mapping[str, object],
    admission_sha: str,
    launch: Mapping[str, object],
    launch_sha: str,
    launch_acceptance: Mapping[str, object],
    launch_acceptance_sha: str,
) -> None:
    """Bind original EMA launch bytes, actual root capture, body and review."""
    _eq(
        (admission_sha, launch_sha, launch_acceptance_sha),
        (ADMISSION_SHA, LAUNCH_SHA, LAUNCH_ACCEPTANCE_SHA),
        "EMA launch receipt pins",
    )
    control_predecessor = {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA,
        "exact_stop_sha256": CONTROL_STOP_SHA,
        "invocation_id": CONTROL_INVOCATION,
        "run_name": CONTROL_RUN,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": CONTROL_UNIT,
    }
    submission = launch.get("submission")
    if not isinstance(submission, dict):
        raise ValueError("EMA launch submission is missing")
    _eq(
        (
            launch.get("schema"),
            launch.get("unit"),
            submission.get("outcome"),
            submission.get("returncode"),
        ),
        ("fg19-child-ema-fit-launch.v1", UNIT, "submitted", 0),
        "EMA original launch",
    )
    _eq(launch.get("control_predecessor"), control_predecessor, "EMA launch predecessor")
    _eq(
        launch.get("unknown_outcome_policy"),
        "never retry; inspect this exact unit and durable EMA output",
        "EMA no-retry policy",
    )
    expected_receipts = {
        "admission": {"path": ADMISSION_PATH, "bytes": 9_880, "sha256": ADMISSION_SHA},
        "launch": {"path": LAUNCH_PATH, "bytes": 7_404, "sha256": LAUNCH_SHA},
        "body": {"path": BODY_PATH, "bytes": BODY_BYTES, "sha256": BODY_SHA},
        "source_review": {
            "path": SOURCE_REVIEW_PATH,
            "bytes": SOURCE_REVIEW_BYTES,
            "sha256": SOURCE_REVIEW_SHA,
        },
    }
    _eq(
        (
            launch_acceptance.get("schema"),
            launch_acceptance.get("status"),
            launch_acceptance.get("run_name"),
            launch_acceptance.get("unit"),
            launch_acceptance.get("invocation_id"),
            launch_acceptance.get("control_predecessor"),
            {key: launch_acceptance.get(key) for key in expected_receipts},
        ),
        (
            "fg19-child-ema-fit-launch-acceptance.v1",
            "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
            RUN,
            UNIT,
            INVOCATION,
            control_predecessor,
            expected_receipts,
        ),
        "root actual EMA launch acceptance",
    )
    _eq(admission.get("control_predecessor"), control_predecessor, "EMA admission predecessor")


def _trainer_argv_sha(launch: Mapping[str, object]) -> str:
    """Digest the complete original trainer argv from the pinned launch record."""
    command = launch.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError("EMA launch command is malformed")
    positions = [
        index for index, item in enumerate(command) if item.endswith("/scripts/train_rnn.py")
    ]
    if len(positions) != 1:
        raise ValueError("EMA launch must identify one canonical trainer entrypoint")
    argv = command[positions[0] + 1 :]
    return hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()


def _check_admission(admission: Mapping[str, object]) -> None:
    """Require the exact source, recipe, predecessor and resource admission."""
    _eq(
        (admission.get("schema"), admission.get("run_name"), admission.get("unit")),
        ("fg19-child-ema-fit-admission.v1", RUN, UNIT),
        "EMA admission identity",
    )
    _eq(
        (
            admission.get("source_commit"),
            admission.get("source_manifest_sha256"),
            admission.get("source_files_verified"),
            admission.get("split_sha256"),
        ),
        (COMMIT, SOURCE_MANIFEST_SHA, 167, SPLIT_SHA),
        "EMA admission source and split",
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
        "EMA parent and profile ancestry",
    )
    recipe = admission.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("EMA admitted recipe is missing")
    _eq(
        recipe,
        {
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
            "objective": "wmse",
            "representation_mode": "ema",
            "seed": 1001,
            "trainer_argv_sha256": TRAINER_ARGV_SHA,
        },
        "EMA admitted recipe",
    )
    _eq(
        admission.get("staged_receipt_sha256"),
        EVIDENCE_RECEIPTS,
        "EMA exact staged ancestry receipts",
    )
    _eq(admission.get("populated"), {}, "EMA admission descendant population")
    parent = admission.get("parent")
    if not isinstance(parent, dict):
        raise ValueError("EMA parent resource contract is missing")
    _eq(
        {
            key: parent.get(key)
            for key in ("CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax")
        },
        {
            "CPUQuotaPerSecUSec": "2s",
            "MemoryHigh": "18000000000",
            "MemoryMax": "20000000000",
            "MemorySwapMax": "0",
        },
        "EMA aggregate parent resource caps",
    )
    dictionary = admission.get("dictionary")
    if not isinstance(dictionary, dict):
        raise ValueError("EMA dictionary identity is missing")
    _eq(
        (dictionary.get("path"), dictionary.get("bytes"), dictionary.get("sha256")),
        (str(DICTIONARY_PATH), control_v3.DICTIONARY_BYTES, control_v3.DICTIONARY_SHA),
        "EMA dictionary identity",
    )


def _check_control_predecessor(
    evidence_root: Path, admission: Mapping[str, object]
) -> dict[str, object]:
    """Revalidate the exact accepted CONTROL receipt and exact stop staged for EMA."""
    acceptance, _ = control_v3._read_pinned_json(
        evidence_root / "CONTROL_TERMINAL_ACCEPTANCE.json", CONTROL_ACCEPTANCE_SHA
    )
    stop, _ = control_v3._read_pinned_json(
        evidence_root / "CONTROL_EXACT_STOP.json", CONTROL_STOP_SHA
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
            acceptance.get("strict_fp32_finite_model_optimizer_verified"),
            acceptance.get("selected_model_epoch5_and_training_state_tensor_identity_verified"),
        ),
        (
            "fg19-child-control-fit-terminal-acceptance.v1",
            "ACCEPTED",
            CONTROL_RUN,
            CONTROL_UNIT,
            CONTROL_INVOCATION,
            COMMIT,
            167,
            SOURCE_MANIFEST_SHA,
            44_760,
            4_476,
            0,
            0,
            6,
            5,
            True,
            True,
        ),
        "accepted CONTROL predecessor integrity",
    )
    _eq(
        (
            acceptance.get("main_pid"),
            acceptance.get("exec_main_code"),
            acceptance.get("exec_main_status"),
            acceptance.get("quality_score"),
        ),
        (0, 1, 0, None),
        "accepted CONTROL predecessor terminal/quality scope",
    )
    control_v3._check_stop_receipt(stop, CONTROL_UNIT, CONTROL_INVOCATION, CONTROL_ACCEPTANCE_SHA)
    predecessor = {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA,
        "exact_stop_sha256": CONTROL_STOP_SHA,
        "invocation_id": CONTROL_INVOCATION,
        "run_name": CONTROL_RUN,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": CONTROL_UNIT,
    }
    _eq(admission.get("control_predecessor"), predecessor, "EMA staged CONTROL predecessor binding")
    return {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA,
        "exact_stop_sha256": CONTROL_STOP_SHA,
        "invocation_id": CONTROL_INVOCATION,
        "selected_epoch": 5,
        "successful_updates": 44_760,
    }


def _check_ema_auxiliary(
    auxiliary: object,
    provenance: Mapping[str, object],
    dictionary_payload: Mapping[str, object],
    dictionary_path: Path,
) -> dict[str, object]:
    """Preserve the shared dictionary state while requiring EMA-mode provenance."""
    if not isinstance(auxiliary, dict) or set(auxiliary) != {
        "norm_ema",
        "dynamics",
        "pool_memory",
        "readout_moments",
        "input_representation",
    }:
        raise ValueError("EMA auxiliary-state schema differs")
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
    control_v3._eq(auxiliary.get("dynamics"), expected_dynamics, "EMA fit-only optimizer dynamics")
    control_v3._eq(
        {key: auxiliary.get(key) for key in ("norm_ema", "pool_memory", "readout_moments")},
        {"norm_ema": None, "pool_memory": None, "readout_moments": None},
        "EMA disabled auxiliary state",
    )
    dictionary_representation = dictionary_payload.get("representation")
    if not isinstance(dictionary_representation, dict):
        raise ValueError("accepted EMA dictionary transform is missing")
    control_v3._check_representation_payload(
        auxiliary.get("input_representation"), dictionary_representation, "EMA training state"
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
        raise ValueError("EMA fitted-representation provenance is malformed")
    _eq(
        identity,
        {
            "mode": "ema",
            "state_sha256": dictionary_representation.get("state_sha256"),
            "artifact_path": str(dictionary_path),
            "artifact_file_sha256": control_v3.DICTIONARY_SHA,
            "artifact_binding": "shared_fit_cache",
            "fit_metadata": dictionary_representation.get("fit_metadata"),
        },
        "EMA fitted-representation ancestry",
    )
    return {
        "mode": identity["mode"],
        "state_sha256": identity["state_sha256"],
        "artifact_sha256": identity["artifact_file_sha256"],
        "artifact_binding": identity["artifact_binding"],
    }


def _check_registered_data_stats(
    admission: Mapping[str, object], data_root: Path
) -> dict[str, object]:
    """Recheck six admitted paths by stat only, retaining exact recorded aliases."""
    records = admission.get("data_identity_stats_only")
    if not isinstance(records, dict) or set(records) != set(REGISTERED_DATA_NAMES):
        raise ValueError("EMA registered data-stat identity map differs")
    observed: dict[str, object] = {}
    for relative in REGISTERED_DATA_NAMES:
        record = records.get(relative)
        if not isinstance(record, dict) or set(record) != {
            "bytes",
            "device",
            "inode",
            "islink",
            "mtime_ns",
            "path",
            "registered_sha256",
            "resolved",
        }:
            raise ValueError(f"EMA data-stat record is malformed: {relative}")
        path = data_root / relative
        if path.is_symlink() is not record["islink"]:
            raise ValueError(f"EMA registered data symlink identity changed: {relative}")
        resolved = path.resolve(strict=True)
        _eq(str(path), record.get("path"), f"EMA registered lexical path {relative}")
        _eq(str(resolved), record.get("resolved"), f"EMA registered resolved path {relative}")
        stat_info = path.stat()
        identity = {
            "bytes": stat_info.st_size,
            "device": stat_info.st_dev,
            "inode": stat_info.st_ino,
            "mtime_ns": stat_info.st_mtime_ns,
        }
        _eq(
            identity,
            {key: record.get(key) for key in ("bytes", "device", "inode", "mtime_ns")},
            f"EMA registered data stat {relative}",
        )
        if not control_v3._is_sha256(record.get("registered_sha256")):
            raise ValueError(f"EMA registered digest is malformed: {relative}")
        observed[relative] = {**identity, "resolved": str(resolved), "islink": path.is_symlink()}
    return observed


def _check_parent_and_dictionary_stats(
    admission: Mapping[str, object], source_root: Path, dictionary_path: Path
) -> None:
    """Check accepted parent and dictionary path/stat identity before restricted reads."""
    parent_identity = admission.get("parent_checkpoint_identity")
    if not isinstance(parent_identity, dict):
        raise ValueError("EMA accepted parent checkpoint stat identity is missing")
    parent_path = source_root.parent / "data/artifacts/runs" / PARENT_RUN / "model_ep3.pt"
    _eq(parent_identity.get("path"), str(parent_path), "EMA parent checkpoint path")
    for label, path, record in (
        ("parent checkpoint", parent_path, parent_identity),
        ("dictionary", dictionary_path, admission.get("dictionary")),
    ):
        if not isinstance(record, dict) or path.is_symlink():
            raise ValueError(f"EMA {label} path identity is malformed")
        stat_info = path.stat(follow_symlinks=False)
        _eq(
            (stat_info.st_size, stat_info.st_dev, stat_info.st_ino, stat_info.st_mtime_ns),
            (
                record.get("bytes"),
                record.get("device"),
                record.get("inode"),
                record.get("mtime_ns"),
            ),
            f"EMA {label} stat identity",
        )


def _check_recipe_args(
    launch_args: Mapping[str, object], split_path: Path, cache_root: Path
) -> None:
    """Require the frozen EMA arm and shared F0 optimizer recipe."""
    _eq(
        {
            key: launch_args.get(key)
            for key in (
                "name",
                "seed",
                "train_source",
                "valid_fold",
                "train_seqs",
                "valid_repeat",
                "valid_train_fraction",
                "batch_order",
                "batch",
                "chunk",
                "epochs",
                "checkpoint_policy",
                "checkpoint_epoch",
                "arch",
                "cell",
                "enc",
                "hidden",
                "layers",
                "head_layers",
                "dropout",
                "init",
                "inherit_init_preprocessing",
                "representation_mode",
                "representation_cache",
                "mean_variance_head",
                "mean_variance_objective",
                "loss",
                "loss_norm",
                "distill_cache_root",
                "distill_cache_variant",
                "distill_cache_scale",
                "distill_alpha",
                "optim",
                "lr",
                "wd",
                "beta2",
                "sched",
                "warmup_frac",
                "clip",
                "precision",
                "no_eval_during_fit",
                "save_training_state",
                "grad_stats",
            )
        },
        {
            "name": RUN,
            "seed": 1001,
            "train_source": "train+valid",
            "valid_fold": 0,
            "train_seqs": 10_535,
            "valid_repeat": 1,
            "valid_train_fraction": 1.0,
            "batch_order": "seeded",
            "batch": 16,
            "chunk": 2_000,
            "epochs": 6,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 5,
            "arch": "gru",
            "cell": "gru",
            "enc": "linear",
            "hidden": 256,
            "layers": 2,
            "head_layers": 1,
            "dropout": 0.0,
            "init": f"{PARENT_RUN}@ep3",
            "inherit_init_preprocessing": True,
            "representation_mode": "ema",
            "representation_cache": str(DICTIONARY_PATH),
            "mean_variance_head": 1,
            "mean_variance_objective": "wmse",
            "loss": "wmse",
            "loss_norm": "chunk",
            "distill_cache_root": str(cache_root),
            "distill_cache_variant": "homogeneous",
            "distill_cache_scale": 1.0,
            "distill_alpha": 0.9,
            "optim": "adamw",
            "lr": 1e-4,
            "wd": 0.01,
            "beta2": 0.999,
            "sched": "onecycle",
            "warmup_frac": 0.05,
            "clip": 1.0,
            "precision": "fp32",
            "no_eval_during_fit": True,
            "save_training_state": True,
            "grad_stats": True,
        },
        "EMA launched scientific recipe",
    )
    _eq(launch_args.get("split_manifest"), str(split_path), "EMA launched split path")
    _eq(launch_args.get("device"), "cuda", "EMA launched device")


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
    """Verify native exit, pinned EMA recipe, ancestry and bounded artifacts."""
    path_exception = control_v3._require_path_compatibility_exception()
    control_v3._reset_safe_globals()
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
    data_stats = _check_registered_data_stats(admission, source_root.parent / "data")
    split, split_sha = _json_file(split_path)
    _eq((split_sha, admission.get("split_sha256")), (SPLIT_SHA, SPLIT_SHA), "canonical EMA split")
    parent_v3._check_worker(launch, admission, source_root)
    cache = parent_v3._check_cache(
        admission, cache_root, cache_acceptance_path, cache_stop_path, split
    )
    lineage = control_v3._check_predecessor_receipts(
        evidence_root, profile_evidence_root, admission
    )
    control_lineage = _check_control_predecessor(evidence_root, admission)
    _eq(
        admission.get("parent_acceptance_sha256"),
        control_v3.PARENT_ACCEPTANCE_SHA,
        "EMA parent acceptance",
    )
    _eq(
        admission.get("parent_exact_stop_sha256"),
        control_v3.PARENT_STOP_SHA,
        "EMA parent exact stop",
    )
    _eq(
        admission.get("parent_checkpoint_sha256"),
        control_v3.PARENT_CHECKPOINT_SHA,
        "EMA parent initializer",
    )
    dictionary_path = (
        source_root.parent / "data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
    )
    _check_parent_and_dictionary_stats(admission, source_root, dictionary_path)
    parent_acceptance, _ = control_v3._read_pinned_json(
        evidence_root / "PARENT_TERMINAL_ACCEPTANCE.json", parent_v3.PARENT_ACCEPTANCE_SHA
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
    argv_sha = _trainer_argv_sha(launch)
    _eq(argv_sha, TRAINER_ARGV_SHA, "full EMA CLI argument digest")
    _check_recipe_args(launch_args, split_path, cache_root)
    expected_args = dict(launch_args)
    representation_module = importlib.import_module("wnn.training.input_representation")
    representation_fit_module = importlib.import_module("wnn.training.input_representation_fit")
    dictionary_module = importlib.import_module("wnn.training.input_representation_dictionary")
    dictionary_artifact, dictionary_identity = control_v3._check_dictionary_artifact(
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
        raise ValueError("EMA fit history is not a list")
    run_files = {
        "model.pt": control_v3.MAX_MODEL_BYTES,
        **{f"model_ep{epoch}.pt": control_v3.MAX_MODEL_BYTES for epoch in range(control_v3.EPOCHS)},
        "training_state.pt": control_v3.MAX_STATE_BYTES,
        "provenance.json": control_v3.MAX_JSON_BYTES,
        "history.json": control_v3.MAX_JSON_BYTES,
    }
    paths = {name: run_dir / name for name in run_files}
    file_identities = {name: _hash_file(paths[name], limit) for name, limit in run_files.items()}
    if (
        sum(entry[1]["bytes"] for entry in file_identities.values())
        > control_v3.MAX_AGGREGATE_BYTES
    ):
        raise ValueError("EMA terminal artifacts exceed the aggregate byte cap")
    _eq(
        (provenance_sha, history_sha),
        (file_identities["provenance.json"][0], file_identities["history.json"][0]),
        "stable EMA metadata",
    )
    _eq(
        (provenance.get("selected_epoch"), provenance.get("split_manifest_sha256")),
        (5, SPLIT_SHA),
        "selected EMA endpoint",
    )
    _eq(provenance.get("args"), expected_args, "EMA provenance arguments")
    _eq(provenance.get("seed"), 1001, "EMA provenance seed")
    _eq(
        provenance.get("initial_checkpoint_sha256"),
        control_v3.PARENT_CHECKPOINT_SHA,
        "EMA provenance initializer",
    )
    _eq(provenance.get("propensity_checkpoint_sha256"), None, "EMA propensity ancestry")
    membership = parent_v3._check_membership(provenance, split)
    parent_v3._check_cache_provenance(provenance, cache_root)
    tensor_identity = _canonical_tensor_identity(source_root)
    models: dict[str, object] = {}
    fingerprints: dict[str, tuple[str, int]] = {}
    for name in ("model.pt", "model_ep5.pt"):
        payload, digest, stat_info = parent_v3._load_checkpoint(
            paths[name], control_v3.MAX_MODEL_BYTES
        )
        _eq((digest, stat_info), file_identities[name], f"stable EMA {name}")
        _eq(
            (payload.get("epoch"), payload.get("n_in"), payload.get("args")),
            (5, 467, expected_args),
            f"EMA {name} endpoint identity",
        )
        models[name] = payload.get("model")
        fingerprints[name] = control_v3._model_fingerprint(models[name])
        child_provenance = payload.get("provenance")
        if not isinstance(child_provenance, dict):
            raise ValueError(f"EMA {name} provenance is missing")
        _eq(
            {key: child_provenance.get(key) for key in control_v3.CHECKPOINT_PROVENANCE_KEYS},
            {key: provenance.get(key) for key in control_v3.CHECKPOINT_PROVENANCE_KEYS},
            f"EMA {name} provenance identity",
        )
        control_v3._check_representation_payload(
            payload.get("input_representation"), dictionary_artifact["representation"], name
        )
    _eq(fingerprints["model.pt"], fingerprints["model_ep5.pt"], "EMA selected model equals epoch 5")
    normalizer = control_v3._check_child_normalizer(
        models["model.pt"], parent_model, provenance, tensor_identity
    )
    state, state_sha, state_stat, state_path_compatibility = (
        control_v3._load_control_training_state(
            paths["training_state.pt"], control_v3.MAX_STATE_BYTES
        )
    )
    _eq((state_sha, state_stat), file_identities["training_state.pt"], "stable EMA training state")
    _eq(
        (state.get("format"), state.get("version"), state.get("boundary"), state.get("next_epoch")),
        ("wnn.epoch-training-state", 1, "epoch", control_v3.EPOCHS),
        "EMA saved epoch boundary",
    )
    _eq(set(state), control_v3.TRAINING_STATE_KEYS, "EMA training-state schema")
    _eq(
        (state.get("ema"), state.get("swa"), state.get("scaler")),
        (None, None, {}),
        "EMA-disabled parameter state",
    )
    representation_identity = _check_ema_auxiliary(
        state.get("auxiliary"), provenance, dictionary_artifact, dictionary_path
    )
    _eq(
        control_v3._model_fingerprint(state.get("model")),
        fingerprints["model.pt"],
        "EMA state/model fingerprint",
    )
    scheduler_lr, scheduler_beta1 = control_v3._check_scheduler(state.get("scheduler"))
    optimizer = control_v3._check_optimizer(state.get("optimizer"), scheduler_lr, scheduler_beta1)
    order_records = control_v3._check_counters(history, state, provenance)
    rng_signature = control_v3._check_loader_and_rng(state)
    manifest, _ = _json_file(source_manifest)
    control_v3._check_args_and_contract(
        state, provenance, admission, expected_args, manifest, split_sha
    )
    _eq(
        provenance.get("checkpoint_sha256"),
        {
            name: file_identities[name][0]
            for name in ("model.pt", *(f"model_ep{epoch}.pt" for epoch in range(control_v3.EPOCHS)))
        },
        "EMA checkpoint digest map",
    )
    log = control_v3._check_sample_log(evidence_root / "unit.log", history, order_records)
    return {
        "schema": "fg19.child-ema-fit-terminal-check.v1",
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
        "observed_optimizer_updates": control_v3.UPDATES,
        "next_epoch": control_v3.EPOCHS,
        "selected_epoch": control_v3.EPOCHS - 1,
        "model_tensor_count": fingerprints["model.pt"][1],
        "model_tensor_fingerprint": fingerprints["model.pt"][0],
        "normalization_identity": normalizer,
        "input_representation": representation_identity,
        "approved_path_compatibility_exception": path_exception,
        "training_state_path_compatibility": state_path_compatibility,
        "accepted_dictionary": dictionary_identity,
        "optimizer": optimizer,
        "rng_observed_after_restricted_load": rng_signature,
        "sample_order_observations": order_records,
        "predecessor_ancestry": {**lineage, "control": control_lineage},
        "native_log": log,
        "cuda_resource_policy_source": {
            "path": "src/wnn/resources.py",
            "sha256": parent_v3.CUDA_POLICY_SOURCE_SHA,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "files": {
            name: {"sha256": value[0], "bytes": value[1]["bytes"]}
            for name, value in file_identities.items()
        },
    }


def main() -> int:
    """Run the post-exit check and create one private no-overwrite result."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
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
        parser.add_argument(f"--{name}", type=Path, required=True)
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
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
