#!/usr/bin/env python3
"""Run only the frozen R3 single-member Q8 controls after current gates pass."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import struct
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R3_PACKET_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r3"
)
R4_PACKET_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4"
)
R5_PACKET_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r5"
)
R6_PACKET_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6"
)
CONTRACT_REL = R6_PACKET_REL / "CONTRACT_SINGLE_R6.json"
R3_CONTRACT_REL = R3_PACKET_REL / "CONTRACT_SINGLE_R3.json"
PREFLIGHT_REL = R3_PACKET_REL / "INPUT_PREFLIGHT_MAIN_R1.json"
ROOT_ADMISSION_REL = R6_PACKET_REL / "ROOT_ADMISSION_R6.json"
FOLDS = ("0", "1")
ROLES = ("student", "hft")
IDENTITY_KEYS = ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
THREAD_ENV = (
    "WNN_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "ARROW_NUM_THREADS",
)


class GateError(RuntimeError):
    """An input, authority, resource, or accounting gate failed."""


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object, rejecting other top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"expected a JSON object: {path}")
    return value


def read_root_admission(path: Path) -> dict[str, Any]:
    """Read an existing root receipt; absence is a closed authorization gate."""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise GateError("R3 root admission receipt is missing") from exc
    return read_json_object(resolved)


def resolve_repo_path(repo_root: Path, repo_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a POSIX repository path without absolute/traversal/symlink escapes."""
    relative = PurePosixPath(repo_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in repo_path or not relative.parts:
        raise GateError(f"unsafe repository-relative path: {repo_path!r}")
    root = repo_root.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GateError(f"repository path escaped root: {repo_path!r}") from exc
    if must_exist and not resolved.is_file():
        raise GateError(f"repository path is not a file: {repo_path!r}")
    return resolved


def verify_source_bindings(contract: dict[str, Any], repo_root: Path) -> None:
    """Rehash all R3/R4 source files before importing any project/native module."""
    bindings = contract.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise GateError("R4 source binding map is empty")
    for name, binding in bindings.items():
        path = resolve_repo_path(repo_root, binding["repo_path"])
        if sha256_file(path) != binding["sha256"]:
            raise GateError(f"source binding changed: {name}")


def verify_r3_receipts(
    contract: dict[str, Any], preflight_path: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Verify the committed R3 contract/preflight and return the exact input path."""
    binding = contract["r3_binding"]
    r3_path = resolve_repo_path(repo_root, binding["contract_repo_path"])
    if r3_path != (repo_root / R3_CONTRACT_REL).resolve(strict=True):
        raise GateError("R3 contract is not at its frozen packet path")
    if sha256_file(r3_path) != binding["contract_sha256"]:
        raise GateError("R3 contract hash differs from R4")
    r3 = read_json_object(r3_path)
    if r3.get("schema") != "wnn.native.single-member-cleanfold-q8-contract.r3":
        raise GateError("unexpected R3 contract schema")
    if r3.get("repository", {}).get("absolute_root") != str(MAIN_ROOT):
        raise GateError("R3 contract names a different project root")
    if contract.get("r3_producer_receipts") != r3.get("producer_receipts"):
        raise GateError("R4 producer receipt bindings differ from the frozen R3 contract")
    if contract.get("r3_source_bindings") != r3.get("source_bindings"):
        raise GateError("R4 source bindings differ from the frozen R3 contract")
    for name, receipt_binding in contract["r3_producer_receipts"].items():
        receipt_path = resolve_repo_path(repo_root, receipt_binding["repo_path"])
        if sha256_file(receipt_path) != receipt_binding["sha256"]:
            raise GateError(f"R3 producer receipt changed: {name}")

    expected_preflight = resolve_repo_path(repo_root, binding["input_preflight_repo_path"])
    if expected_preflight != (repo_root / PREFLIGHT_REL).resolve(strict=True):
        raise GateError("R3 preflight is not at its frozen packet path")
    if preflight_path.resolve(strict=True) != expected_preflight:
        raise GateError("provided preflight path differs from R3")
    if sha256_file(expected_preflight) != binding["input_preflight_sha256"]:
        raise GateError("R3 input-preflight receipt changed")
    preflight = read_json_object(expected_preflight)
    if preflight.get("schema") != "wnn.native.single-member-input-preflight.r3":
        raise GateError("unexpected R3 preflight schema")
    if preflight.get("status") != "PASS_FROZEN_FOOTER_AND_METADATA_ONLY":
        raise GateError("R3 input preflight is not a pass")
    if preflight.get("repository_root") != str(MAIN_ROOT):
        raise GateError("R3 input preflight names a different repository root")
    if preflight.get("valid_dataset_repo_path") != r3["valid_dataset"]["repo_path"]:
        raise GateError("R3 input preflight names a different dataset path")
    if preflight.get("contract_sha256") != binding["contract_sha256"]:
        raise GateError("R3 preflight is bound to different contract bytes")
    if (
        preflight.get("prediction_data_loaded") is not False
        or preflight.get("full_fold_score_performed") is not False
    ):
        raise GateError("R3 preflight claims data-page loading or scoring")
    expected_identity = {key: r3["valid_dataset"][key] for key in IDENTITY_KEYS}
    actual_identity = preflight.get("valid_dataset_identity", {})
    if {key: actual_identity.get(key) for key in IDENTITY_KEYS} != expected_identity:
        raise GateError("R3 preflight footer identity differs from its contract")
    data_path = resolve_repo_path(repo_root, r3["valid_dataset"]["repo_path"])
    if data_path != (repo_root / "datasets/valid.parquet").resolve(strict=True):
        raise GateError("R3 dataset path does not resolve to MAIN datasets/valid.parquet")
    return r3, preflight, data_path


def admission_expected_bindings(
    contract: dict[str, Any],
    r4_sha: str,
    runner_sha: str,
    preflight_sha: str,
    callback_fixture_sha: str,
) -> dict[str, Any]:
    """Build the exact admission binding map consumed by the executable gate."""
    r3 = contract["r3_binding"]
    identity = {key: r3["valid_dataset_identity"][key] for key in IDENTITY_KEYS}
    models = contract["models"]
    return {
        "r3_contract_sha256": r3["contract_sha256"],
        "r6_contract_sha256": r4_sha,
        "r6_runner_sha256": runner_sha,
        "input_preflight_sha256": preflight_sha,
        "valid_dataset_footer_identity": identity,
        "fold_membership_sha256s": {
            fold: models[fold]["evaluation_group_list_sha256"] for fold in FOLDS
        },
        "checkpoint_provenance_sha256s": {
            fold: {
                role: {
                    "checkpoint": models[fold]["models"][role]["checkpoint_sha256"],
                    "provenance": models[fold]["models"][role]["provenance_sha256"],
                }
                for role in ROLES
            }
            for fold in FOLDS
        },
        "weights_archive_sha256": contract["weights"]["archive_sha256"],
        "weights_receipt_sha256": contract["weights"]["receipt_sha256"],
        "native_elf_sha256": contract["runtime_assets"]["native_elf"]["sha256"],
        "model_selection": {
            "folds": list(FOLDS),
            "roles": list(ROLES),
            "member_count_per_prediction": 1,
            "precision": "q8vnni",
            "blend_search": False,
            "parameter_averaging": False,
        },
        "output_root_repo_path": contract["output"]["root_repo_path"],
        "output_root_absent": True,
        "callback_fixture_sha256": callback_fixture_sha,
        "launch_command": contract["launch_command"],
    }


def verify_admission_object(
    admission: dict[str, Any],
    contract: dict[str, Any],
    r4_sha: str,
    runner_sha: str,
    preflight_sha: str,
    callback_fixture_sha: str,
) -> None:
    """Require an actual root admission bound to R3 inputs and the exact R4 consumer."""
    authority = contract["root_admission"]
    if admission.get("schema") != authority["schema"]:
        raise GateError("root admission schema differs")
    if admission.get("status") != "admitted" or admission.get("authorized") is not True:
        raise GateError("R3 root admission is missing or unauthorized")
    if admission.get("decision") != authority["decision"]:
        raise GateError("root admission decision differs")
    expected = admission_expected_bindings(
        contract, r4_sha, runner_sha, preflight_sha, callback_fixture_sha
    )
    bindings = admission.get("bindings")
    if not isinstance(bindings, dict):
        raise GateError("root admission lacks bindings")
    if bindings != expected:
        differing = sorted(
            key for key in set(bindings) | set(expected) if bindings.get(key) != expected.get(key)
        )
        raise GateError(f"root admission binding set/value differs: {differing}")
    if admission.get("budget") != contract["resource_budget"]:
        raise GateError("root admission budget differs from the frozen R6 budget")


def _parse_cpu_max(raw: str) -> tuple[int, int]:
    fields = raw.split()
    if len(fields) != 2 or fields[0] == "max":
        raise GateError("CPU cgroup is unlimited or malformed")
    try:
        quota, period = (int(value) for value in fields)
    except ValueError as exc:
        raise GateError("CPU cgroup quota is malformed") from exc
    if quota <= 0 or period <= 0 or quota > period:
        raise GateError("effective CPU quota exceeds one core or is invalid")
    return quota, period


def read_cgroup_limits(
    proc_cgroup: Path = Path("/proc/self/cgroup"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> dict[str, Any]:
    """Read the effective cgroup-v2 CPU, memory, and swap limits for this process."""
    lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    relative = next((line.split("::", 1)[1] for line in lines if "::" in line), None)
    if relative is None or ".." in Path(relative).parts:
        raise GateError("could not identify a safe unified cgroup path")
    root = cgroup_root.resolve(strict=True)
    directory = (root / relative.lstrip("/")).resolve(strict=True)
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise GateError("process cgroup path escaped cgroup root") from exc
    quota, period = _parse_cpu_max((directory / "cpu.max").read_text(encoding="ascii").strip())
    memory_raw = (directory / "memory.max").read_text(encoding="ascii").strip()
    swap_raw = (directory / "memory.swap.max").read_text(encoding="ascii").strip()
    if memory_raw == "max" or swap_raw == "max":
        raise GateError("memory or swap cgroup is unlimited")
    current = int((directory / "memory.current").read_text(encoding="ascii").strip())
    return {
        "path": relative,
        "cpu_quota": quota,
        "cpu_period": period,
        "memory_max_bytes": int(memory_raw),
        "memory_current_bytes": current,
        "memory_swap_max_bytes": int(swap_raw),
    }


def verify_wrapper_limits(
    budget: dict[str, Any], environ: dict[str, str], cgroup: dict[str, Any]
) -> None:
    """Fail unless scripts/limited and the current cgroup enforce the admitted envelope."""
    required = {
        "cpu_cores": 1,
        "ram_max_bytes": 1_000_000_000,
        "wall_seconds": 1800,
        "gpu": False,
        "new_spend_usd": 0,
        "thread_count": 1,
        "provider_actions": False,
    }
    if budget != required:
        raise GateError("R4 resource budget differs from the fixed CPU-only envelope")
    expected_env = {
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_QUOTA": "100%",
        "WNN_MEM_MAX": "1000000000",
        "WNN_RUNTIME_MAX": "1800s",
        "WNN_SCOPE_UNIT": "wnn-native-q8-single-controls-r6.scope",
    }
    if any(environ.get(key) != value for key, value in expected_env.items()):
        raise GateError("execution wrapper environment does not match the root-admitted caps")
    if "CUDA_VISIBLE_DEVICES" not in environ or environ["CUDA_VISIBLE_DEVICES"] != "":
        raise GateError("R4 must not expose a GPU")
    if any(environ.get(key) != "1" for key in THREAD_ENV):
        raise GateError("thread environment is not pinned to one")
    segments = cgroup.get("path", "").split("/")
    if "wnn-campaign.slice" not in segments:
        raise GateError("process is outside the admitted campaign cgroup")
    if expected_env["WNN_SCOPE_UNIT"] not in segments:
        raise GateError("process is outside the R6 hard-timeout scope")
    if cgroup["cpu_quota"] > cgroup["cpu_period"]:
        raise GateError("actual CPU cgroup exceeds one core")
    if cgroup["memory_max_bytes"] <= 0 or cgroup["memory_max_bytes"] > budget["ram_max_bytes"]:
        raise GateError("actual memory cgroup exceeds the root-admitted cap")
    if cgroup["memory_swap_max_bytes"] != 0:
        raise GateError("actual cgroup permits swap")
    if cgroup["memory_current_bytes"] > cgroup["memory_max_bytes"]:
        raise GateError("current cgroup memory exceeds its limit")


def verify_python311_and_dependencies() -> dict[str, Any]:
    """Enforce CPython 3.11 and required packages; never create or install an environment."""
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.11").resolve() or tuple(
        sys.version_info[:2]
    ) != (3, 11):
        raise GateError("use the frozen CPython 3.11 interpreter")
    from importlib.util import find_spec

    missing = [name for name in ("numpy", "pyarrow") if find_spec(name) is None]
    if missing:
        raise GateError(f"CPython 3.11 dependencies are missing: {', '.join(missing)}")
    return {
        "python": sys.version.split()[0],
        "required_dependencies": ["numpy", "pyarrow"],
        "missing_dependencies": [],
    }


def file_footer_identity(
    stream: BinaryIO, path: Path, expected_footer_bytes: int
) -> dict[str, Any]:
    """Hash the Parquet footer on this open descriptor and detect replacement/mutation."""
    fd = stream.fileno()
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size < expected_footer_bytes + 12:
        raise GateError("valid.parquet is nonregular or shorter than its frozen footer")
    if os.pread(fd, 4, 0) != b"PAR1":
        raise GateError("valid.parquet leading magic differs")
    trailer = os.pread(fd, 8, before.st_size - 8)
    if len(trailer) != 8 or trailer[4:] != b"PAR1":
        raise GateError("valid.parquet trailing magic differs")
    declared = struct.unpack("<I", trailer[:4])[0]
    if declared != expected_footer_bytes:
        raise GateError("valid.parquet footer byte count differs")
    start = before.st_size - declared - 8
    digest = hashlib.sha256()
    offset = start
    while offset < before.st_size:
        block = os.pread(fd, min(1 << 20, before.st_size - offset), offset)
        if not block:
            raise GateError("short read of valid.parquet footer")
        digest.update(block)
        offset += len(block)
    after = os.fstat(fd)
    path_stat = path.stat()
    identity_stat = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity_stat != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise GateError("valid.parquet changed during identity check")
    if identity_stat != (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_size,
        path_stat.st_mtime_ns,
    ):
        raise GateError("valid.parquet path was replaced after opening")
    return {
        "bytes": before.st_size,
        "footer_bytes": declared,
        "footer_sha256_including_8_byte_trailer": digest.hexdigest(),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
    }


def verify_open_descriptor_stat(stream: BinaryIO, path: Path, expected: dict[str, Any]) -> None:
    """Reject descriptor/path replacement between the full footer checks."""
    descriptor = os.fstat(stream.fileno())
    current_path = path.stat()
    observed = (descriptor.st_dev, descriptor.st_ino, descriptor.st_size, descriptor.st_mtime_ns)
    path_identity = (
        current_path.st_dev,
        current_path.st_ino,
        current_path.st_size,
        current_path.st_mtime_ns,
    )
    expected_identity = tuple(expected[key] for key in ("device", "inode", "bytes", "mtime_ns"))
    if observed != expected_identity or path_identity != expected_identity:
        raise GateError("valid.parquet descriptor/path changed before a row-group read")


def validate_live_identity(
    identity: dict[str, Any], r3: dict[str, Any], preflight: dict[str, Any]
) -> None:
    """Require the current descriptor to match R3's bytes and prior stat identity."""
    expected = {key: r3["valid_dataset"][key] for key in IDENTITY_KEYS}
    if {key: identity.get(key) for key in IDENTITY_KEYS} != expected:
        raise GateError("live valid.parquet footer identity differs from R3")
    prior = preflight.get("valid_dataset_identity", {})
    for key in ("inode", "mtime_ns"):
        if identity.get(key) != prior.get(key):
            raise GateError(f"live valid.parquet {key} differs from the R3 preflight")


def check_metadata(
    parquet: Any, r3: dict[str, Any], preflight: dict[str, Any], feature_columns: list[str]
) -> None:
    """Check complete groups/schema on the same ParquetFile before reading row groups."""
    metadata = parquet.metadata
    expected = r3["valid_dataset"]
    if metadata.num_rows != expected["rows"] or parquet.num_row_groups != expected["row_groups"]:
        raise GateError("live Parquet metadata count differs from R3")
    if any(metadata.row_group(i).num_rows != 20_000 for i in range(parquet.num_row_groups)):
        raise GateError("live Parquet includes an incomplete sequence row group")
    names = set(parquet.schema.names)
    required = {
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        "t0",
        "t1",
        *feature_columns,
    }
    if not required <= names:
        raise GateError("live Parquet lacks required columns")
    old = preflight["metadata_only_validation"]
    actual = {
        "rows": metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "schema_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(str(parquet.schema).encode()).hexdigest(),
    }
    for key, value in actual.items():
        if old.get(key) != value:
            raise GateError(f"live Parquet metadata differs from R3 preflight: {key}")


def prediction_plan(r3: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive exactly four independent one-member prediction files from R3."""
    if list(r3["folds"]) != list(FOLDS):
        raise GateError("R3 fold order differs from fixed F0/F1")
    declared = r3["expected_output"]["prediction_files"]
    plan: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        for role in ROLES:
            key = f"fold{fold}_{role}"
            item = declared.get(key)
            if not isinstance(item, dict):
                raise GateError(f"R3 output plan omits {key}")
            shape = [int(value) for value in item["shape"]]
            groups = int(r3["folds"][fold]["evaluation_groups"])
            if shape != [groups, 20_000, 2] or item.get("dtype") != "float32":
                raise GateError(f"R3 shape/dtype differs for {key}")
            payload_bytes = groups * 20_000 * 2 * 4
            file_bytes = int(item["bytes"])
            npy_header_bytes = file_bytes - payload_bytes
            if npy_header_bytes != 128:
                raise GateError(f"R3 NPY header/file byte accounting differs for {key}")
            plan[key] = {
                "filename": f"predictions_fold{fold}_{role}.npy",
                "shape": shape,
                "dtype": "float32",
                "bytes": file_bytes,
                "payload_bytes": payload_bytes,
                "npy_header_bytes": npy_header_bytes,
                "fold": fold,
                "role": role,
            }
            plan[key]["evaluation_group_list_sha256"] = r3["folds"][fold][
                "evaluation_group_list_sha256"
            ]
            plan[key]["selected_rows"] = int(
                r3["folds"][fold]["fixed_half_output_reference"]["selected_rows"]
            )
    if set(plan) != {"fold0_student", "fold0_hft", "fold1_student", "fold1_hft"}:
        raise GateError("R3 output set contains a noncanonical role or blend")
    return plan


def validate_prediction_inventory(
    output_root: Path, plan: dict[str, dict[str, Any]], result_filename: str | None = None
) -> dict[str, dict[str, Any]]:
    """Fail closed on missing, extra, nested, symlink, or wrong-sized output files."""
    if not output_root.is_dir() or output_root.is_symlink():
        raise GateError("output root is not a regular directory")
    expected = {item["filename"]: item for item in plan.values()}
    if result_filename is not None:
        expected[result_filename] = {"bytes": None, "shape": None, "dtype": "json"}
    observed: dict[str, Path] = {}
    for path in output_root.rglob("*"):
        if path.is_symlink() or path.is_dir() or not path.is_file():
            raise GateError(f"unexpected output entry: {path.relative_to(output_root)}")
        observed[path.relative_to(output_root).as_posix()] = path
    if set(observed) != set(expected):
        raise GateError("output inventory differs from the frozen R3 plan")
    result = {}
    for name, spec in expected.items():
        size = observed[name].stat().st_size
        if spec["bytes"] is not None and size != spec["bytes"]:
            raise GateError(f"output byte count differs: {name}")
        result[name] = {"bytes": size, "dtype": spec["dtype"], "shape": spec["shape"]}
    return result


def canonical_group_hash(groups: list[int]) -> str:
    """Hash the canonical JSON list used by the frozen source split receipts."""
    return hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()


def validate_result_accounting(result: dict[str, Any], plan: dict[str, dict[str, Any]]) -> None:
    """Check the result describes exactly the fixed folds, roles, groups and files."""
    if result.get("folds_evaluated") != list(FOLDS) or result.get(
        "roles_evaluated_independently"
    ) != list(ROLES):
        raise GateError("result fold/role accounting differs from the fixed plan")
    if result.get("member_count_per_prediction") != 1 or result.get("precision") != "q8vnni":
        raise GateError("result member count or native precision differs")
    if result.get("blend_search") is not False or result.get("parameter_averaging") is not False:
        raise GateError("result reports an unapproved blend/parameter search")
    if (
        result.get("training_performed") is not False
        or result.get("fresh_holdout_claim") is not False
    ):
        raise GateError("result overstates the scope of this evaluation")
    expected_files = {item["filename"] for item in plan.values()}
    inventory = result.get("prediction_inventory")
    if not isinstance(inventory, dict) or set(inventory) != expected_files:
        raise GateError("result prediction inventory differs from the frozen output plan")
    folds = result.get("folds")
    if not isinstance(folds, dict) or set(folds) != set(FOLDS):
        raise GateError("result fold sections differ from the frozen output plan")
    for fold in FOLDS:
        if not isinstance(folds[fold], dict) or set(folds[fold]) != set(ROLES):
            raise GateError(f"result role accounting differs for fold {fold}")
        for role in ROLES:
            key = f"fold{fold}_{role}"
            expected = plan[key]
            item = folds[fold][role]
            prediction = item.get("prediction_file", {})
            metric = item.get("metric", {})
            if item.get("evaluation_groups") != expected["shape"][0]:
                raise GateError(f"result evaluated-group count differs for {key}")
            if item.get("evaluation_group_list_sha256") != expected["evaluation_group_list_sha256"]:
                raise GateError(f"result membership binding differs for {key}")
            if item.get("callback_rows") != expected["shape"][0] * 20_000:
                raise GateError(f"result callback-row count differs for {key}")
            if item.get("member_count") != 1 or item.get("precision") != "q8vnni":
                raise GateError(f"result model accounting differs for {key}")
            if metric.get("selected_rows") != expected["selected_rows"]:
                raise GateError(f"result metric-mask row count differs for {key}")
            if prediction != inventory[expected["filename"]]:
                raise GateError(f"result prediction metadata differs from inventory for {key}")
            if (
                prediction.get("bytes") != expected["bytes"]
                or prediction.get("shape") != expected["shape"]
                or prediction.get("dtype") != expected["dtype"]
            ):
                raise GateError(f"result prediction geometry differs for {key}")


def load_pinned_module(name: str, path: Path) -> Any:
    """Import an exact source after caller verified its bound hash."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_predictors(
    contract: dict[str, Any], repo_root: Path, native: Any, runtime_guard: Callable[[], None]
) -> tuple[dict[str, dict[str, Any]], Any, Any, Any, Any]:
    """Load source-bound arrays and construct four fixed one-member Q8 predictors."""
    runtime_guard()
    import numpy as np

    for name, binding in contract["runtime_assets"].items():
        if "sha256" not in binding:
            continue
        asset = Path(binding["absolute_path"]).resolve(strict=True)
        if sha256_file(asset) != binding["sha256"]:
            raise GateError(f"external runtime asset changed: {name}")
    for fold in FOLDS:
        for role in ROLES:
            model_binding = contract["models"][fold]["models"][role]
            for kind in ("checkpoint", "provenance"):
                source = resolve_repo_path(repo_root, model_binding[f"{kind}_repo_path"])
                if sha256_file(source) != model_binding[f"{kind}_sha256"]:
                    raise GateError(f"source-qualified {fold}/{role} {kind} changed")

    r1_binding = contract["r3_producer_receipts"]["shared_contract_r1"]
    r1_contract_path = resolve_repo_path(repo_root, r1_binding["repo_path"])
    if sha256_file(r1_contract_path) != r1_binding["sha256"]:
        raise GateError("R1 producer contract changed")
    r1_contract = read_json_object(r1_contract_path)
    r1_consumer = load_pinned_module(
        "r4_r1_weight_consumer",
        resolve_repo_path(
            repo_root, contract["source_bindings"]["r1_weight_consumer"]["repo_path"]
        ),
    )
    weights_path = Path(contract["weights"]["archive_absolute_path"]).resolve(strict=True)
    receipt_path = Path(contract["weights"]["receipt_absolute_path"]).resolve(strict=True)
    runtime_guard()
    weights, receipt = r1_consumer.load_weights(
        r1_contract, weights_path, receipt_path, r1_contract_path
    )
    if receipt.get("contract_sha256") != r1_binding["sha256"] or set(weights) != set(FOLDS):
        raise GateError("weight archive provenance/fold set differs")

    extension = sys.modules.get("wnn_native.wnn_native")
    if extension is None:
        raise GateError("R5 did not load the bound native extension")
    if Path(native.__file__).resolve(strict=True) != Path(
        contract["runtime_assets"]["native_package_init"]["absolute_path"]
    ).resolve(strict=True):
        raise GateError("loaded native package init path differs from the frozen asset")
    if Path(extension.__file__).resolve(strict=True) != Path(
        contract["runtime_assets"]["native_elf"]["absolute_path"]
    ).resolve(strict=True):
        raise GateError("loaded Q8 extension path differs from the frozen asset")
    wrapper = load_pinned_module(
        "r4_wrapped_predictor",
        resolve_repo_path(repo_root, contract["source_bindings"]["wrapped_predictor"]["repo_path"]),
    )
    predictors: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        predictors[fold] = {}
        for index, role in enumerate(ROLES):
            runtime_guard()
            member = weights[fold]["members"][index]
            model = native.EnsemblePredictor(
                [member], [1.0], fast_activations=True, precision="q8vnni", shared_backbone=False
            )
            if str(model.precision) != "q8vnni":
                raise GateError(f"fold {fold}/{role} did not select q8vnni")
            predictors[fold][role] = wrapper.WrappedPredictor(model)
    runner = load_pinned_module(
        "r4_callback_runner",
        resolve_repo_path(repo_root, contract["source_bindings"]["callback_runner"]["repo_path"]),
    )
    metric = load_pinned_module(
        "r4_official_metric",
        resolve_repo_path(repo_root, contract["source_bindings"]["official_metric"]["repo_path"]),
    )
    utils_path = resolve_repo_path(
        repo_root, contract["source_bindings"]["project_utils"]["repo_path"]
    )
    sys.path.insert(0, str(repo_root))
    import utils

    if Path(utils.__file__).resolve(strict=True) != utils_path:
        raise GateError("utils imported from a different project root")
    if contract["metric"]["id"] != metric.METRIC_ID or float(metric.METRIC_CLIP) != 2.0:
        raise GateError("loaded official metric differs from R3")
    return predictors, runner, metric, utils, np


def run_full_controls(
    contract: dict[str, Any],
    r3: dict[str, Any],
    preflight: dict[str, Any],
    data_path: Path,
    output_root: Path,
    predictors: dict[str, dict[str, Any]],
    runner: Any,
    metric: Any,
    utils: Any,
    np: Any,
    r5: Any,
    runtime_guard: Callable[[], None],
    process_started: float,
    metadata_check: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Evaluate each fixed role/fold separately; never combine members or search weights."""
    runtime_guard()
    import pyarrow.parquet as pq

    plan = prediction_plan(r3)
    stable_metadata_check = metadata_check or r5.check_stable_metadata
    if output_root.exists():
        raise FileExistsError("R3 output root already exists; R4 is create-only")
    r1_contract_path = resolve_repo_path(
        Path(contract["repository"]["actual_root"]),
        contract["r3_producer_receipts"]["shared_contract_r1"]["repo_path"],
    )
    r1 = read_json_object(r1_contract_path)
    start = process_started
    runtime_guard()
    with data_path.open("rb") as source:
        identity = file_footer_identity(source, data_path, int(r3["valid_dataset"]["footer_bytes"]))
        validate_live_identity(identity, r3, preflight)
        parquet = pq.ParquetFile(source)
        # PyArrow may read metadata during construction. Recheck the same open
        # descriptor after that step and immediately before any row-group read.
        runtime_guard()
        validate_live_identity(
            file_footer_identity(source, data_path, int(r3["valid_dataset"]["footer_bytes"])),
            r3,
            preflight,
        )
        stable_metadata_check(parquet, r3, preflight, list(utils.FEATURE_COLUMNS))
        # The same verified descriptor remains open through every row-group read.
        runtime_guard()
        output_root.mkdir(parents=False, exist_ok=False)
        maps = {
            f"{fold}_{role}": np.lib.format.open_memmap(
                output_root / plan[f"fold{fold}_{role}"]["filename"],
                mode="w+",
                dtype=np.float32,
                shape=tuple(plan[f"fold{fold}_{role}"]["shape"]),
            )
            for fold in FOLDS
            for role in ROLES
        }
        accumulators = {key: metric.GlobalAccumulator() for key in maps}
        callbacks = {key: 0 for key in maps}
        seen_ids: dict[str, set[int]] = {fold: set() for fold in FOLDS}
        counts = {key: 0 for key in maps}

        for fold in FOLDS:
            binding = r1["folds"][fold]["split_manifest"]
            groups = binding["evaluation_row_groups_sorted"]
            if len(groups) != int(r3["folds"][fold]["evaluation_groups"]) or groups != sorted(
                set(groups)
            ):
                raise GateError(f"fold {fold} membership is not the fixed sorted unique R3 list")
            if canonical_group_hash(groups) != r3["folds"][fold]["evaluation_group_list_sha256"]:
                raise GateError(f"fold {fold} membership hash differs from R3")
            for index, group_value in enumerate(groups):
                if time.monotonic() - start >= int(contract["resource_budget"]["wall_seconds"]):
                    raise TimeoutError("R6 cooperative wall-clock deadline reached")
                runtime_guard()
                verify_open_descriptor_stat(source, data_path, identity)
                if index == 0:
                    identity = file_footer_identity(
                        source, data_path, int(r3["valid_dataset"]["footer_bytes"])
                    )
                    validate_live_identity(identity, r3, preflight)
                    stable_metadata_check(parquet, r3, preflight, list(utils.FEATURE_COLUMNS))
                group = int(group_value)
                columns = [
                    "seq_ix",
                    "step_in_seq",
                    "need_prediction",
                    "is_scored",
                    "t0",
                    "t1",
                    *utils.FEATURE_COLUMNS,
                ]
                table = parquet.read_row_group(group, columns=columns, use_threads=False)
                sequence_id, need = utils.validate_sequence(table)
                if sequence_id in seen_ids[fold]:
                    raise GateError(f"duplicate sequence ID in fold {fold}")
                seen_ids[fold].add(sequence_id)
                features = np.column_stack(
                    [table[name].to_numpy(zero_copy_only=False) for name in utils.FEATURE_COLUMNS]
                ).astype(np.float32, copy=False)
                target = np.column_stack(
                    [table[name].to_numpy(zero_copy_only=False) for name in ("t0", "t1")]
                ).astype(np.float32, copy=False)
                scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
                if (
                    features.shape != (20_000, 112)
                    or target.shape != (20_000, 2)
                    or scored.shape != (20_000,)
                ):
                    raise GateError(f"fold {fold} group {group} has unexpected data geometry")
                if not np.isfinite(features).all() or not np.isfinite(target).all():
                    raise GateError(f"fold {fold} group {group} contains nonfinite input/labels")
                sequence = {"sequence_id": sequence_id, "features": features, "need": need}
                mask = np.asarray(need, dtype=bool) & scored
                for role in ROLES:
                    key = f"{fold}_{role}"
                    elapsed, raw = runner.run_callbacks(predictors[fold][role], sequence)
                    callbacks[key] += int(elapsed)
                    if raw.shape != (20_000, 2) or not np.isfinite(raw[need]).all():
                        raise GateError(f"invalid required prediction for {key}, group {group}")
                    output = np.zeros((20_000, 2), dtype=np.float32)
                    output[need] = raw[need]
                    accumulators[key].add(target, output, mask, group=group)
                    maps[key][index] = output
                    counts[key] += 1
                del table, features, target, scored, sequence
                if (index + 1) % 10 == 0 or index + 1 == len(groups):
                    print(
                        json.dumps(
                            {
                                "phase": "scoring",
                                "utc": datetime.now(UTC).isoformat(),
                                "fold": fold,
                                "complete": index + 1,
                                "total": len(groups),
                            }
                        ),
                        flush=True,
                    )
    for array in maps.values():
        array.flush()
    del maps

    inventory = validate_prediction_inventory(output_root, plan)
    results: dict[str, Any] = {}
    for fold in FOLDS:
        results[fold] = {}
        for role in ROLES:
            key = f"{fold}_{role}"
            score = accumulators[key].result()
            reference = r3["folds"][fold]["fixed_half_output_reference"]
            if score["selected_rows"] != reference["selected_rows"]:
                raise GateError(f"official mask row count differs for {key}")
            item = plan[f"fold{fold}_{role}"]
            path = output_root / item["filename"]
            prediction = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                list(prediction.shape) != item["shape"]
                or str(prediction.dtype) != "float32"
                or not np.isfinite(prediction).all()
            ):
                raise GateError(f"saved prediction header/finiteness differs for {key}")
            inventory[item["filename"]].update({"sha256": sha256_file(path), "all_finite": True})
            results[fold][role] = {
                "metric": score,
                "fixed_half_reference": reference,
                "delta_wp_vs_fixed_half": float(score["wp"] - reference["wp"]),
                "exposure": contract["models"][fold]["exposure"],
                "evaluation_groups": counts[key],
                "evaluation_group_list_sha256": r3["folds"][fold]["evaluation_group_list_sha256"],
                "callback_rows": counts[key] * 20_000,
                "callback_ns": callbacks[key],
                "member_count": 1,
                "precision": "q8vnni",
                "prediction_file": inventory[item["filename"]],
            }
    if set(item["filename"] for item in plan.values()) != set(inventory):
        raise GateError("final output file accounting is incomplete")
    return {
        "schema": "wnn.native.single-member-cleanfold-q8-result.r3",
        "status": "PASS_EXPOSED_CROSSFOLD_SINGLE_MEMBER_Q8_CONTROLS",
        "created_utc": datetime.now(UTC).isoformat(),
        "folds": results,
        "metric": r3["metric"],
        "folds_evaluated": list(FOLDS),
        "roles_evaluated_independently": list(ROLES),
        "member_count_per_prediction": 1,
        "precision": "q8vnni",
        "blend_search": False,
        "parameter_averaging": False,
        "training_performed": False,
        "fresh_holdout_claim": False,
        "public_forecast": False,
        "prediction_inventory": inventory,
        "elapsed_wall_seconds": time.monotonic() - start,
    }


def main() -> None:
    """Validate immutable inputs, R6 authority, and actual wrapper limits before scoring."""
    process_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=MAIN_ROOT / CONTRACT_REL)
    parser.add_argument("--input-preflight", type=Path, default=MAIN_ROOT / PREFLIGHT_REL)
    parser.add_argument("--root-admission", type=Path, default=MAIN_ROOT / ROOT_ADMISSION_REL)
    args = parser.parse_args()
    root = MAIN_ROOT.resolve(strict=True)
    if Path.cwd().resolve(strict=True) != root:
        raise GateError("R6 must execute with exact MAIN as the working directory")
    contract_path = args.contract.resolve(strict=True)
    if contract_path != (root / CONTRACT_REL).resolve(strict=True):
        raise GateError("R6 contract path differs from its fixed project path")
    contract = read_json_object(contract_path)
    if (
        contract.get("schema") != "wnn.native.single-member-cleanfold-q8-contract.r6"
        or contract.get("status") != "prepared_pending_root_r6_release"
    ):
        raise GateError("R6 contract is not the frozen pending-root preparation")
    if contract.get("repository", {}).get("absolute_root") != str(root):
        raise GateError("R6 contract names a different project root")
    if contract.get("repository", {}).get("actual_root") != str(root):
        raise GateError("R6 runtime repository path differs from the fixed MAIN root")
    verify_source_bindings(contract, root)
    launch = contract["launch_command"]
    if sys.argv[1:] or Path(sys.argv[0]).resolve(strict=True) != (
        root / launch["script_repo_path"]
    ).resolve(strict=True):
        raise GateError("R6 executable argv differs from the frozen scripts/limited command")
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.11").resolve():
        raise GateError("R6 executable differs from the frozen Python 3.11 command")
    r3, preflight, data_path = verify_r3_receipts(contract, args.input_preflight, root)
    fixture_path = resolve_repo_path(root, contract["callback_fixture"]["repo_path"])
    fixture_sha = sha256_file(fixture_path)
    fixture = read_json_object(fixture_path)
    if (
        fixture.get("schema") != contract["callback_fixture"]["schema"]
        or fixture.get("status") != "PASS_ACTUAL_SINGLE_MEMBER_Q8_RESET_REPLAY"
    ):
        raise GateError("single-member actual callback fixture did not pass")
    if (
        fixture.get("full_fold_score_performed") is not False
        or fixture.get("full_fold_rows_scored") != 0
    ):
        raise GateError("R5 callback fixture overstates its scope")
    if fixture.get("provider_or_gpu_action") is not False:
        raise GateError("R5 callback fixture records a provider/GPU action")
    for fold in FOLDS:
        for role in ROLES:
            item = fixture.get("folds", {}).get(fold, {}).get(role, {})
            if (
                item.get("member_count") != 1
                or item.get("precision") != "q8vnni"
                or item.get("complete_rows") != 20_000
                or item.get("labels_loaded") is not False
                or item.get("reset_replay_bitwise_equal") is not True
                or item.get("required_outputs_finite") is not True
            ):
                raise GateError(f"single-member callback fixture is invalid for {fold}/{role}")
    admission_path = args.root_admission.resolve(strict=False)
    if admission_path != (root / ROOT_ADMISSION_REL).resolve(strict=False):
        raise GateError("R6 root release path differs from its frozen receipt location")
    admission = read_root_admission(admission_path)
    r5_path = resolve_repo_path(root, contract["source_bindings"]["r5_runtime"]["repo_path"])
    r5 = load_pinned_module("r6_r5_callback_runtime", r5_path)
    r4_sha = sha256_file(contract_path)
    runner_sha = contract["source_bindings"]["r6_consumer"]["sha256"]
    preflight_sha = contract["r3_binding"]["input_preflight_sha256"]
    verify_admission_object(admission, contract, r4_sha, runner_sha, preflight_sha, fixture_sha)
    cgroup = read_cgroup_limits()
    verify_wrapper_limits(contract["resource_budget"], os.environ, cgroup)
    runtime_info = verify_python311_and_dependencies()

    resource_gate_samples: list[dict[str, Any]] = []

    def runtime_guard() -> None:
        if time.monotonic() - process_started >= int(contract["resource_budget"]["wall_seconds"]):
            raise TimeoutError("R6 cooperative wall-clock deadline reached")
        current = read_cgroup_limits()
        verify_wrapper_limits(contract["resource_budget"], os.environ, current)
        resource_gate_samples.append(
            {
                "elapsed_wall_seconds": time.monotonic() - process_started,
                "cgroup_path": current["path"],
                "cpu_quota": current["cpu_quota"],
                "cpu_period": current["cpu_period"],
                "memory_max_bytes": current["memory_max_bytes"],
                "memory_current_bytes": current["memory_current_bytes"],
                "memory_swap_max_bytes": current["memory_swap_max_bytes"],
            }
        )

    runtime_guard()
    if time.monotonic() - process_started >= int(contract["resource_budget"]["wall_seconds"]):
        raise TimeoutError("R6 startup exhausted its cooperative wall-clock deadline")
    plan = prediction_plan(r3)
    output_root = resolve_repo_path(root, contract["output"]["root_repo_path"], must_exist=False)
    if output_root.exists():
        raise FileExistsError("R3 output root already exists; refusing overwrite")
    runtime_guard()
    native = r5.import_bound_native_package(contract, sys.modules[__name__])
    predictors, runner, metric, utils, np = build_predictors(contract, root, native, runtime_guard)
    runtime_guard()
    result = run_full_controls(
        contract,
        r3,
        preflight,
        data_path,
        output_root,
        predictors,
        runner,
        metric,
        utils,
        np,
        r5,
        runtime_guard,
        process_started,
    )
    result["schema"] = "wnn.native.single-member-cleanfold-q8-result.r6"
    result.update(
        {
            "contract_sha256": r4_sha,
            "r3_contract_sha256": contract["r3_binding"]["contract_sha256"],
            "input_preflight_sha256": preflight_sha,
            "root_admission_sha256": sha256_file(admission_path),
            "callback_fixture_sha256": sha256_file(fixture_path),
            "runtime": runtime_info,
            "resource_limits_before_model_load": cgroup,
            "resource_gate_check_count": len(resource_gate_samples),
            "resource_gate_first_check": resource_gate_samples[0],
            "resource_gate_last_check": resource_gate_samples[-1],
        }
    )
    result_path = output_root / contract["output"]["result_filename"]
    validate_result_accounting(result, plan)
    validate_prediction_inventory(output_root, plan)
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    validate_prediction_inventory(output_root, plan, result_path.name)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_root),
                "folds": list(FOLDS),
                "roles": list(ROLES),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
