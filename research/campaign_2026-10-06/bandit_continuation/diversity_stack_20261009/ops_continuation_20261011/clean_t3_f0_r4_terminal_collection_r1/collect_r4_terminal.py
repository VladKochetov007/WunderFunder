from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = (
    MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011"
)
PACKET = OPS / "clean_t3_fold0_h390_r4_teacher_cohort"
ROOT_RELEASE = OPS / "clean_t3_fold0_h390_r4_root_release"
R2_COLLECTOR = OPS / "clean_t3_f0_profile_collection_r2/collect_profile_r2.py"
R4_KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4-runner"
R4_DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4"
R4_ARCHIVE_SHA256 = "ae8796a9c0cca1ed64d45341c14c917fc19f6889a1df20ff1873a87a200570dc"
R4_MANIFEST_SHA256 = "5f3f0aa8f3759a5ff594c9414c04ad6c4c8834a22f156482af18b411388dcaf7"
R4_BOOTSTRAP_SHA256 = "48ddfe18ad60943efbede30923fad9231ce96705299fea2168ef4dbebdd4e78b"
R4_CONTRACT_SHA256 = "5d0b9347b71d9c817fb7870d41ffea0abc5b3a47b9e038decf406e21e65e4b6b"
R4_PLAN_SHA256 = "aee76572ef4d2ba50b5d27b91aa9a8bd9ea7056702aace87bd936d535442ccd9"
R4_FINALIZATION_SHA256 = "1f3d930393694bd5904837395a8762c4ab4122f5a61d9d2d233500e0f05bb59e"
R4_FINAL_DISPATCH_SHA256 = "0079fed7fdc64bcab31fabdea7c3c2046303604765953ba4dcf6633475b66513"
R4_PUSH_SHA256 = "098feb7c182723d42fa917e3561d295f2e8d1cc838d0c8ebd1ce39ac7c7030e5"
R4_READY_SHA256 = "e788a91c512871af14f98ca8fc681584558dda4a41734185abd6e7833824a65b"
R4_VALIDATION_SHA256 = "ffb6fc5b0bef250674067ae778a1d3434e856013ae44531db0c41bd5b3ff7250"
R4_DOWNLOAD_ADMISSION_SHA256 = "3eee6881ec10d8d4b18b5769733982aa9baf47fa28e5a10cbed8698a310e546e"
R4_READONLY_INVENTORY_SHA256 = "1b834f5e77d6ebd493a93b8e9ae0e2791827078d5b9f3fb05a2f12b21114627b"
R2_ADAPTER_SHA256 = "15b56f98d0b8ca336ea794142764a3d5a220ef75589e931c06b9dc87184e1da1"
R2_REVIEW_SHA256 = "995afd07ecfe7d4ca808e5d6a544eb4621eed971f9d03901fc8ee9fc5fb88ddf"
R4_SOURCE_MANIFEST_SHA256 = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
SPLIT_PATH = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PROPENSITY_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
H382_ACTUAL_NPY = (
    OPS / "kaggle_h382_delivery_20261011_r4/provider_output/out/predictions/"
    "pred_kg_h382r3_f0_muonall_prop2_e3_s3_valid.npy"
)
H382_ACTUAL_SIDECAR = H382_ACTUAL_NPY.with_suffix(".json")
H382_ACTUAL_NPY_SHA256 = "d12ee518ce4ba46b2af890552712f916b1bfb9e451805cf7daa671b4b3afc048"
MAX_INVENTORY_SECONDS = 45
MAX_DOWNLOAD_BYTES = 1024**3
MIN_FREE_BYTES = 4 * 1024**3
MAX_DOWNLOAD_SECONDS = 420
N_SEQUENCE_STEPS = 20_000
N_OUTPUT_CHANNELS = 2


def sha256_file(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = 16 * 1024**2) -> dict[str, Any]:
    """Read one bounded JSON receipt and reject non-object payloads."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError(f"receipt is missing, symlinked, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return value


def assert_file_sha(path: Path, expected: str) -> None:
    """Require one frozen input to match its reviewed digest."""
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"frozen input SHA-256 mismatch: {path}")


def validate_frozen_r4_bindings() -> dict[str, Any]:
    """Bind the exact R4 packet, READY asset and one authorized version-one push."""
    assert_file_sha(
        OPS / "KAGGLE_CLEAN_T3_F0_COLLECTOR_R2_INDEPENDENT_REVIEW_R1.json",
        R2_REVIEW_SHA256,
    )
    assert_file_sha(PACKET / "bundle_teacher_cohort_r3/bundle.wnn", R4_ARCHIVE_SHA256)
    assert_file_sha(PACKET / "bundle_teacher_cohort_r3/bundle-manifest.json", R4_MANIFEST_SHA256)
    assert_file_sha(PACKET / "bundle_teacher_cohort_r3/kernel/kernel.py", R4_BOOTSTRAP_SHA256)
    assert_file_sha(PACKET / "source_snapshot/SOURCE_MANIFEST_H390.json", R4_SOURCE_MANIFEST_SHA256)
    assert_file_sha(PACKET / "COLLECTION_CONTRACT_R3.json", R4_CONTRACT_SHA256)
    assert_file_sha(PACKET / "TEACHER_COHORT_PLAN_R3.json", R4_PLAN_SHA256)
    assert_file_sha(PACKET / "R4_FINALIZATION_RECEIPT_R1.json", R4_FINALIZATION_SHA256)
    assert_file_sha(PACKET / "R4_FINALIZATION_VALIDATION_R1.json", R4_VALIDATION_SHA256)
    assert_file_sha(
        ROOT_RELEASE / "ROOT_FINAL_DISPATCH_ADMISSION_R1.json", R4_FINAL_DISPATCH_SHA256
    )
    assert_file_sha(ROOT_RELEASE / "ROOT_KERNEL_PUSH_R1.json", R4_PUSH_SHA256)
    assert_file_sha(ROOT_RELEASE / "ROOT_PRIVATE_DATASET_READY_READBACK_R2.json", R4_READY_SHA256)
    assert_file_sha(R2_COLLECTOR, R2_ADAPTER_SHA256)

    finalization = read_json(PACKET / "R4_FINALIZATION_RECEIPT_R1.json")
    dispatch = read_json(ROOT_RELEASE / "ROOT_FINAL_DISPATCH_ADMISSION_R1.json")
    push = read_json(ROOT_RELEASE / "ROOT_KERNEL_PUSH_R1.json")
    ready = read_json(ROOT_RELEASE / "ROOT_PRIVATE_DATASET_READY_READBACK_R2.json")
    contract = read_json(PACKET / "COLLECTION_CONTRACT_R3.json")
    plan = read_json(PACKET / "TEACHER_COHORT_PLAN_R3.json")
    input_binding = read_json(PACKET / "INPUT_BINDING_R3.json")

    if (
        finalization.get("archive_sha256") != R4_ARCHIVE_SHA256
        or finalization.get("bootstrap_sha256") != R4_BOOTSTRAP_SHA256
        or finalization.get("bundle_manifest_sha256") != R4_MANIFEST_SHA256
        or finalization.get("final_dispatch_authorized") is not False
        or finalization.get("kernel_id") != R4_KERNEL
        or finalization.get("dataset") != R4_DATASET
    ):
        raise ValueError("R4 finalization receipt no longer binds the sealed packet")
    if (
        dispatch.get("dispatch_authorized") is not True
        or dispatch.get("kernel_id") != R4_KERNEL
        or dispatch.get("dataset") != R4_DATASET
        or dispatch.get("dataset_version") != 1
        or dispatch.get("archive_sha256") != R4_ARCHIVE_SHA256
        or dispatch.get("manifest_sha256") != R4_MANIFEST_SHA256
        or dispatch.get("bootstrap_sha256") != R4_BOOTSTRAP_SHA256
    ):
        raise ValueError("root dispatch receipt does not bind exact R4 v1")
    if (
        push.get("returncode") != 0
        or push.get("admission_sha256") != R4_FINAL_DISPATCH_SHA256
        or "successfully pushed" not in str(push.get("stdout", "")).lower()
        or f"/code/{R4_KERNEL}" not in str(push.get("stdout", ""))
    ):
        raise ValueError("provider push receipt does not bind the admitted R4 kernel")
    if (
        ready.get("dataset") != R4_DATASET
        or ready.get("status") != "ready"
        or ready.get("current_version_number") != 1
        or ready.get("kernel_push") is not False
        or ready.get("ready_byte_identical_readback") is not True
    ):
        raise ValueError("private R4 dataset READY receipt is not exact")
    if (
        contract.get("schema") != "wnn.kaggle.teacher_cohort_collection.r3"
        or contract.get("status") != "required_at_terminal_not_collected"
        or len(plan.get("export_stage", [])) != 9
        or input_binding.get("split_manifest_sha256") != SPLIT_SHA256
    ):
        raise ValueError("R4 collection contract or nine-export plan changed")
    return {
        "kernel": R4_KERNEL,
        "dataset": R4_DATASET,
        "dataset_version": 1,
        "archive_sha256": R4_ARCHIVE_SHA256,
        "manifest_sha256": R4_MANIFEST_SHA256,
        "bootstrap_sha256": R4_BOOTSTRAP_SHA256,
        "contract_sha256": R4_CONTRACT_SHA256,
        "plan_sha256": R4_PLAN_SHA256,
        "source_manifest_sha256": R4_SOURCE_MANIFEST_SHA256,
        "split_manifest_sha256": SPLIT_SHA256,
    }


def require_exact_inventory_names(
    expected_names: list[str], inventory_files: list[dict[str, Any]]
) -> list[str]:
    """Reject any fresh provider filename set differing from the pinned preinventory."""
    if (
        not isinstance(expected_names, list)
        or any(not isinstance(name, str) or not name for name in expected_names)
        or len(set(expected_names)) != len(expected_names)
        or not isinstance(inventory_files, list)
    ):
        raise ValueError("inventory filename lists are malformed or duplicated")
    observed_names = []
    for row in inventory_files:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("provider inventory contains an invalid filename descriptor")
        observed_names.append(row["name"])
    if len(set(observed_names)) != len(observed_names):
        raise ValueError("provider inventory contains duplicate filenames")
    if set(observed_names) != set(expected_names):
        missing = sorted(set(expected_names) - set(observed_names))
        added = sorted(set(observed_names) - set(expected_names))
        raise ValueError(
            f"fresh provider filename set differs from pinned preinventory; missing={len(missing)} "
            f"added={len(added)}"
        )
    return sorted(observed_names)


def validate_download_admission(
    admission_path: Path,
    status_path: Path,
    inventory_receipt_path: Path,
    collection_dir: Path,
) -> dict[str, Any]:
    """Require root's exact terminal-error download-only admission before I/O."""
    expected_path = ROOT_RELEASE / "ROOT_TERMINAL_DOWNLOAD_ADMISSION_R1.json"
    if admission_path.resolve() != expected_path.resolve():
        raise ValueError("download admission must be the exact root-issued R1 receipt")
    assert_file_sha(admission_path, R4_DOWNLOAD_ADMISSION_SHA256)
    assert_file_sha(inventory_receipt_path, R4_READONLY_INVENTORY_SHA256)
    admission = read_json(admission_path)
    inventory_receipt = read_json(inventory_receipt_path)
    if (
        admission.get("schema") != "wnn.kaggle-terminal-download-root-admission.r1"
        or admission.get("status") != "admitted"
        or admission.get("authorized_action") != "terminal-download-only"
        or admission.get("kernel") != R4_KERNEL
        or admission.get("terminal_status") != "ERROR"
        or admission.get("archive_sha256") != R4_ARCHIVE_SHA256
        or admission.get("root_fit_admission_sha256") != R4_FINAL_DISPATCH_SHA256
        or admission.get("status_snapshot_sha256") != sha256_file(status_path)
        or Path(admission.get("status_snapshot", "")).resolve() != status_path.resolve()
        or admission.get("maximum_download_bytes") != MAX_DOWNLOAD_BYTES
        or admission.get("minimum_free_bytes") != MIN_FREE_BYTES
        or admission.get("runtime_max_seconds") != 720
        or admission.get("inner_download_max_seconds") != MAX_DOWNLOAD_SECONDS
        or admission.get("cpus") != 0.5
        or admission.get("ram_bytes") != 700_000_000
        or admission.get("collection_dir") != str(collection_dir.resolve())
        or admission.get("independent_wrapper_review_required_before_execution") is not True
        or admission.get("provider_mutations_authorized") is not False
        or admission.get("fits_authorized") is not False
        or admission.get("overwrite_authorized") is not False
    ):
        raise ValueError("root terminal-download admission does not match the bounded R4 action")
    if (
        inventory_receipt.get("schema") != "wnn.kaggle.r4_readonly_terminal_inventory.r1"
        or inventory_receipt.get("kernel") != R4_KERNEL
        or inventory_receipt.get("provider_status") != "KernelWorkerStatus.ERROR"
        or inventory_receipt.get("output_file_count") != 108
        or inventory_receipt.get("download_performed") is not False
        or inventory_receipt.get("endpoint_inventory", {}).get("complete") is not True
    ):
        raise ValueError("the exact read-only ERROR inventory receipt is not complete")
    expected_names = inventory_receipt.get("output_names")
    endpoint_names = require_exact_inventory_names(
        expected_names, inventory_receipt["endpoint_inventory"].get("files", [])
    )
    if len(expected_names) != 108 or expected_names != endpoint_names:
        raise ValueError("pinned read-only inventory has inconsistent names or count")
    snapshot, status_row = parse_status_snapshot(status_path, "2026-10-11T21:30:16.866625+00:00")
    del snapshot
    if status_row.get("status") != "KernelWorkerStatus.ERROR":
        raise ValueError("root authorized ERROR collection but provider status is no longer ERROR")
    return {
        "admission_sha256": R4_DOWNLOAD_ADMISSION_SHA256,
        "inventory_receipt_sha256": R4_READONLY_INVENTORY_SHA256,
        "expected_output_names": sorted(expected_names),
        "provider_status": status_row.get("status"),
        "download_budget_bytes": MAX_DOWNLOAD_BYTES,
        "inner_runtime_seconds": MAX_DOWNLOAD_SECONDS,
    }


def load_expected_membership() -> dict[str, Any]:
    """Validate source-qualified fit/evaluation membership from the pinned split."""
    assert_file_sha(SPLIT_PATH, SPLIT_SHA256)
    manifest = read_json(SPLIT_PATH, maximum_bytes=8 * 1024**2)
    sequences = manifest.get("sequences")
    fit = manifest.get("fit")
    evaluation = manifest.get("evaluation")
    if (
        not isinstance(sequences, dict)
        or not isinstance(fit, dict)
        or not isinstance(evaluation, dict)
    ):
        raise ValueError("fold-zero split manifest membership schema is invalid")

    source_rows: dict[str, list[int]] = {}
    component_by_source_group: dict[tuple[str, int], int] = {}
    for source in ("train", "valid"):
        nodes = sequences.get(source)
        if not isinstance(nodes, list):
            raise ValueError(f"fold-zero sequence list is missing source {source}")
        row_groups: list[int] = []
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("split sequence identity must be an object")
            row_group = node.get("row_group")
            component = node.get("component")
            if (
                isinstance(row_group, bool)
                or not isinstance(row_group, int)
                or isinstance(component, bool)
                or not isinstance(component, int)
            ):
                raise ValueError("split sequence identity needs integer row-group/component")
            key = (source, row_group)
            if key in component_by_source_group:
                raise ValueError(f"duplicate source-qualified group identity: {key}")
            row_groups.append(row_group)
            component_by_source_group[key] = component
        if len(set(row_groups)) != len(row_groups):
            raise ValueError(f"duplicate row group in source {source}")
        source_rows[source] = row_groups

    expected_fit = {
        "train": list(map(int, fit.get("train", []))),
        "valid": list(map(int, fit.get("valid", []))),
    }
    expected_eval = list(map(int, evaluation.get("valid", [])))
    for source, selected in expected_fit.items():
        if not selected or not set(selected).issubset(source_rows[source]):
            raise ValueError(f"fit membership escapes source-qualified {source} rows")
        if len(set(selected)) != len(selected):
            raise ValueError(f"duplicate fitting row group in {source}")
    if not expected_eval or len(set(expected_eval)) != len(expected_eval):
        raise ValueError("valid evaluation membership is empty or duplicated")
    if not set(expected_eval).issubset(source_rows["valid"]):
        raise ValueError("evaluation groups are not valid-source qualified")

    evaluation_components = {component_by_source_group[("valid", group)] for group in expected_eval}
    fit_components = {
        component_by_source_group[(source, group)]
        for source, groups in expected_fit.items()
        for group in groups
    }
    if fit_components & evaluation_components:
        raise ValueError("fold-zero fitted groups overlap evaluated components")
    for source in ("train", "valid"):
        complete_fit = [
            group
            for group in source_rows[source]
            if component_by_source_group[(source, group)] not in evaluation_components
        ]
        if expected_fit[source] != complete_fit:
            raise ValueError(f"{source} fitting list differs from complete clean membership")

    expected_counts = {
        "train": 10_535,
        "valid": 1_405,
        "excluded_valid_eval": 468,
    }
    if {
        "train": len(expected_fit["train"]),
        "valid": len(expected_fit["valid"]),
        "excluded_valid_eval": len(expected_eval),
    } != expected_counts:
        raise ValueError("fold-zero source-qualified membership counts changed")
    return {
        "fold": manifest.get("fold"),
        "n_folds": manifest.get("n_folds"),
        "split_manifest_sha256": SPLIT_SHA256,
        "fit": expected_fit,
        "evaluation_valid": expected_eval,
        "full_valid_order": source_rows["valid"],
        "counts": expected_counts,
        "source_qualified_component_exclusion": True,
    }


def load_adapter():
    """Load the independently reviewed R2 collector without changing its source."""
    spec = importlib.util.spec_from_file_location("reviewed_clean_t3_collector_r2", R2_COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the reviewed terminal collector R2")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if sha256_file(R2_COLLECTOR) != R2_ADAPTER_SHA256:
        raise ValueError("reviewed R2 collector bytes changed")
    adapter.KERNEL = R4_KERNEL
    adapter.KERNEL_OUTPUT_LOG_NAME = R4_KERNEL.rsplit("/", 1)[-1] + ".log"
    return adapter


def parse_status_snapshot(path: Path, push_observed: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract one fresh R4 status row from a read-only poller snapshot."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10 * 1024**2:
        raise ValueError("status snapshot is missing, symlinked, or too large")
    raw = path.read_bytes()
    lines = raw.splitlines()
    if not lines:
        raise ValueError("status snapshot is empty")
    try:
        snapshot = json.loads(lines[-1])
    except json.JSONDecodeError:
        # Root poll receipts are often pretty-printed single JSON objects rather
        # than JSONL. Accept only that whole-file form after the last-line parse.
        snapshot = json.loads(raw)
    if not isinstance(snapshot, dict):
        raise ValueError("status snapshot must be a JSON object")
    observed = snapshot.get("observed_at_utc", snapshot.get("observed_utc"))
    if not isinstance(observed, str):
        raise ValueError("status snapshot lacks an observation timestamp")
    observed_dt = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    pushed_dt = datetime.fromisoformat(push_observed.replace("Z", "+00:00"))
    if observed_dt < pushed_dt:
        raise ValueError("status snapshot predates the admitted R4 push")
    if snapshot.get("cli_package_version") != "2.2.4" or not str(
        snapshot.get("source", "")
    ).startswith("read-only Kaggle SDK quota/status calls"):
        raise ValueError("status snapshot is not the expected read-only Kaggle 2.2.4 receipt")
    rows = snapshot.get("status")
    if not isinstance(rows, list):
        raise ValueError("status snapshot has no status rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("kernel") == R4_KERNEL]
    if len(matches) != 1 or matches[0].get("resource") != "gpu":
        raise ValueError("status snapshot does not uniquely bind the R4 GPU kernel")
    return snapshot, matches[0]


def unique_suffix_file(root: Path, suffix: str) -> Path:
    """Resolve exactly one regular output whose path ends with the frozen suffix."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"output root is missing or symlinked: {root}")
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one output ending in {suffix}; found {len(matches)}")
    return matches[0]


def sha_from_inventory(path: Path, output_root: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    """Require collected bytes to match the reviewed collector's fresh inventory hash."""
    relative = path.relative_to(output_root).as_posix()
    row = comparison.get("per_file_sha256", {}).get(relative)
    if not isinstance(row, dict):
        raise ValueError(f"output is not in the fresh provider inventory: {relative}")
    size = path.stat().st_size
    if row.get("size_bytes") != size:
        raise ValueError(f"output size differs from the fresh collector receipt: {relative}")
    digest = row.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"output SHA-256 is missing from the collector receipt: {relative}")
    return {"path": relative, "size_bytes": size, "sha256": digest}


def validate_npy(
    path: Path,
    expected_shape: list[int] | tuple[int, ...],
    expected_dtype: str,
    *,
    finite_rows: int | None = None,
) -> dict[str, Any]:
    """Validate one actual NPY header and scan finite values in bounded row chunks."""
    import numpy as np

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"prediction NPY is missing or symlinked: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        shape = tuple(int(value) for value in array.shape)
        dtype = np.dtype(array.dtype)
        if shape != tuple(expected_shape):
            raise ValueError(f"NPY shape mismatch: {path.name}: {shape}")
        if dtype != np.dtype(expected_dtype) or dtype.hasobject:
            raise ValueError(f"NPY dtype mismatch: {path.name}: {dtype}")
        if array.flags.f_contiguous and not array.flags.c_contiguous:
            raise ValueError(f"unexpected Fortran-order prediction array: {path.name}")
        expected_file_bytes = int(array.offset) + int(array.nbytes)
        if path.stat().st_size != expected_file_bytes:
            raise ValueError(f"NPY header/payload byte count mismatch: {path.name}")
        rows = shape[0]
        rows_to_scan = rows if finite_rows is None else min(rows, finite_rows)
        if rows_to_scan <= 0:
            raise ValueError("finite scan row count must be positive")
        chunk_rows = max(1, min(8, 2_000_000 // (shape[1] * shape[2] * dtype.itemsize)))
        for start in range(0, rows_to_scan, chunk_rows):
            block = array[start : min(start + chunk_rows, rows_to_scan)]
            if not bool(np.isfinite(block).all()):
                raise ValueError(f"prediction array contains NaN or infinity: {path.name}")
            del block
        return {
            "shape": list(shape),
            "dtype": dtype.name,
            "npy_bytes": path.stat().st_size,
            "rows_scanned": rows_to_scan,
            "all_values_finite": rows_to_scan == rows,
            "fortran_order": bool(array.flags.f_contiguous and not array.flags.c_contiguous),
        }
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def read_small_json(path: Path, maximum_bytes: int = 2 * 1024**2) -> dict[str, Any]:
    """Read a small worker, sidecar, or provenance receipt."""
    return read_json(path, maximum_bytes)


def validate_input_identity(
    path: Path,
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    """Check exact mounted-input hashes and only the declared deferred raw files."""
    identity = read_small_json(path)
    if identity.get("status") != "verified":
        raise ValueError(f"job input identity is not verified: {path.name}")
    observations = identity.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"job input identity has no observations: {path.name}")
    by_path = {
        row.get("path"): row
        for row in observations
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    expected_inputs = input_binding.get("external_inputs")
    if not isinstance(expected_inputs, list):
        raise ValueError("R4 input binding has no external input list")
    deferred: list[str] = []
    for expected_input in expected_inputs:
        name = expected_input["path"]
        row = by_path.get(name)
        if row is None:
            raise ValueError(f"job input identity omitted frozen external input: {name}")
        expected = row.get("expected", {})
        observed = row.get("observed", {})
        if expected.get("sha256") != expected_input["sha256"]:
            raise ValueError(f"job input receipt expected hash changed: {name}")
        if expected.get("size") != expected_input.get("size"):
            raise ValueError(f"job input receipt expected size changed: {name}")
        if name in {"datasets/train.parquet", "datasets/valid.parquet"}:
            if (
                row.get("status") != "deferred_to_worker_preflight"
                or observed.get("size") != expected_input.get("size")
                or observed.get("sha256") is not None
            ):
                raise ValueError(
                    f"raw source was not deferred to its strict worker preflight: {name}"
                )
            deferred.append(name)
        elif (
            row.get("status") != "match"
            or observed.get("size") != expected_input.get("size")
            or observed.get("sha256") != expected_input["sha256"]
        ):
            raise ValueError(f"mounted external input differs from the sealed packet: {name}")
    if set(deferred) != {"datasets/train.parquet", "datasets/valid.parquet"}:
        raise ValueError("the exact two declared raw sources were not deferred")
    return {"status": "verified", "deferred_raw_sources": sorted(deferred)}


def require_worker_stage(
    output_root: Path,
    stage: str,
    names: list[str],
    *,
    expected_artifacts: dict[str, list[str]],
    input_binding: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Validate one actual runner summary and all exact workers in its stage."""
    summary_path = unique_suffix_file(output_root, f"results/{stage}/summary.json")
    summary = read_small_json(summary_path)
    finished = summary.get("finished")
    if (
        not isinstance(finished, list)
        or summary.get("incident") is not None
        or summary.get("not_started")
    ):
        raise ValueError(f"{stage} stage summary has an incident or unstarted worker")
    finished_names = [row.get("name") for row in finished if isinstance(row, dict)]
    if sorted(finished_names) != sorted(names) or len(finished_names) != len(set(finished_names)):
        raise ValueError(f"{stage} stage summary does not list exactly the expected workers")
    workers = []
    for name in names:
        result_path = unique_suffix_file(output_root, f"results/{stage}/{name}/result.json")
        result = read_small_json(result_path)
        row = next(item for item in finished if item.get("name") == name)
        artifacts = result.get("artifacts", {})
        if (
            row.get("status") != "complete"
            or row.get("returncode") != 0
            or result.get("status") != "complete"
            or result.get("returncode") != 0
            or result.get("name") != name
            or artifacts.get("missing")
        ):
            raise ValueError(f"{stage} worker did not finish cleanly: {name}")
        declared = artifacts.get("files")
        if not isinstance(declared, list):
            raise ValueError(f"{stage} worker lacks an artifact file inventory: {name}")
        declared_paths = {
            item.get("path")
            for item in declared
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        for artifact in expected_artifacts.get(name, []):
            if artifact not in declared_paths:
                raise ValueError(f"{stage} worker omitted a declared artifact: {name}:{artifact}")
        worker_log = unique_suffix_file(output_root, f"results/{stage}/{name}/worker.log")
        if worker_log.stat().st_size == 0:
            raise ValueError(f"{stage} worker log is empty: {name}")
        input_path = unique_suffix_file(output_root, f"results/{stage}/{name}.input_identity.json")
        identity = validate_input_identity(input_path, input_binding)
        input_receipt_hash = sha_from_inventory(input_path, output_root, comparison)
        worker_receipt = sha_from_inventory(result_path, output_root, comparison)
        log_receipt = sha_from_inventory(worker_log, output_root, comparison)
        workers.append(
            {
                "name": name,
                "result": worker_receipt,
                "worker_log": log_receipt,
                "input_identity": input_receipt_hash,
                "input_status": "verified",
                "deferred_raw_sources": identity["deferred_raw_sources"],
                "artifact_count": len(declared),
            }
        )
    return {
        "summary": sha_from_inventory(summary_path, output_root, comparison),
        "workers": workers,
    }


def validate_pipeline_outputs(
    attempt_dir: Path,
    comparison: dict[str, Any],
    static_bindings: dict[str, Any],
    membership: dict[str, Any],
) -> dict[str, Any]:
    """Strictly validate pipeline receipts, fit ancestry and all nine future arrays."""
    output_root = attempt_dir / "outputs"
    plan = read_json(PACKET / "TEACHER_COHORT_PLAN_R3.json")
    input_binding = read_json(PACKET / "INPUT_BINDING_R3.json")
    seeds = plan["recipe"]["seeds"]
    fit_names = [f"fit_clean_t3_f0_muonall_h384_e3_s{seed}_r3" for seed in seeds]
    export_names = [
        f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r3_{kind}"
        for seed in seeds
        for kind in ("fit_train", "fit_valid", "full_valid")
    ]
    profile_names = plan["profile_stage"]["jobs"]

    pipeline_path = unique_suffix_file(output_root, "results/teacher_cohort_pipeline.json")
    pipeline = read_small_json(pipeline_path)
    if pipeline.get("status") != "complete":
        raise ValueError(f"teacher pipeline status is not complete: {pipeline.get('status')}")
    if len(pipeline.get("stages", [])) != 3:
        raise ValueError("teacher pipeline lacks the paired-profile, fit, and export stages")
    profile_guard = pipeline.get("paired_profile_guard", {})
    uuids = profile_guard.get("distinct_t4_uuids", [])
    if profile_guard.get("status") != "pass" or len(set(uuids)) != 2:
        raise ValueError("two-device profile guard did not pass")
    guard_file = unique_suffix_file(output_root, "results/paired_profile_guard_receipt.json")
    preflight_file = unique_suffix_file(output_root, "results/profile_preflight_inventory.json")
    pipeline_receipt = sha_from_inventory(pipeline_path, output_root, comparison)
    guard_receipt = sha_from_inventory(guard_file, output_root, comparison)
    preflight_receipt = sha_from_inventory(preflight_file, output_root, comparison)

    profile_summary_path = unique_suffix_file(output_root, "results/paired_profiles/summary.json")
    profile_summary = read_small_json(profile_summary_path)
    profile_finished = profile_summary.get("finished", [])
    if sorted(row.get("name") for row in profile_finished) != sorted(profile_names):
        raise ValueError("paired profile summary is incomplete")
    profile_workers = []
    for name in profile_names:
        result_path = unique_suffix_file(output_root, f"results/paired_profiles/{name}/result.json")
        log_path = unique_suffix_file(output_root, f"results/paired_profiles/{name}/worker.log")
        input_path = unique_suffix_file(
            output_root, f"results/paired_profiles/{name}.input_identity.json"
        )
        result = read_small_json(result_path)
        identity = read_small_json(input_path)
        if (
            result.get("status") != "complete"
            or result.get("returncode") != 0
            or identity.get("status") != "verified"
            or log_path.stat().st_size == 0
        ):
            raise ValueError(f"paired profile worker failed or lacks receipts: {name}")
        profile_workers.append(
            {
                "name": name,
                "result": sha_from_inventory(result_path, output_root, comparison),
                "worker_log": sha_from_inventory(log_path, output_root, comparison),
                "input_identity": sha_from_inventory(input_path, output_root, comparison),
            }
        )

    fit_expected = {
        item["fit_spec"].rsplit("/", 1)[-1].removesuffix(".json"): item["artifacts"]
        for item in plan["fit_stage"]
    }
    export_expected = {
        item["spec"].rsplit("/", 1)[-1].removesuffix(".json"): [
            item["npy"]["path"],
            item["sidecar"],
        ]
        for item in plan["export_stage"]
    }
    fit_stage = require_worker_stage(
        output_root,
        "teacher_fits",
        fit_names,
        expected_artifacts=fit_expected,
        input_binding=input_binding,
        comparison=comparison,
    )
    export_stage = require_worker_stage(
        output_root,
        "teacher_exports",
        export_names,
        expected_artifacts=export_expected,
        input_binding=input_binding,
        comparison=comparison,
    )

    model_files: dict[str, dict[str, str]] = {}
    for item in plan["fit_stage"]:
        run_id = item["run_id"]
        seed = item["seed"]
        run_suffix = f"artifacts/runs/{run_id}/"
        artifact_paths = {
            file_name: unique_suffix_file(output_root, run_suffix + file_name)
            for file_name in (
                "model_ep2.pt",
                "model.pt",
                "training_state.pt",
                "history.json",
                "provenance.json",
            )
        }
        file_records = {
            name: sha_from_inventory(path, output_root, comparison)
            for name, path in artifact_paths.items()
        }
        provenance = read_small_json(artifact_paths["provenance.json"])
        args = provenance.get("args", {})
        checkpoint_map = provenance.get("checkpoint_sha256", {})
        if (
            provenance.get("seed") != seed
            or args.get("name") != run_id
            or args.get("hidden") != 384
            or args.get("layers") != 2
            or provenance.get("selected_epoch") != 2
            or provenance.get("split_manifest_sha256") != SPLIT_SHA256
            or provenance.get("propensity_checkpoint_sha256") != PROPENSITY_SHA256
            or provenance.get("fit_row_groups") != membership["fit"]
            or provenance.get("evaluation_row_groups") != membership["evaluation_valid"]
            or checkpoint_map.get("model_ep2.pt") != file_records["model_ep2.pt"]["sha256"]
            or checkpoint_map.get("model.pt") != file_records["model.pt"]["sha256"]
        ):
            raise ValueError(f"fit checkpoint provenance or membership mismatch: {run_id}")
        model_files[run_id] = file_records

    predictions = []
    for item in plan["export_stage"]:
        run_id = item["run_id"]
        kind = item["kind"]
        expected_shape = item["npy"]["shape"]
        source_split = item["split"]
        if kind == "fit_train":
            expected_rows = membership["fit"]["train"]
            expected_scope = "train"
        elif kind == "fit_valid":
            expected_rows = membership["fit"]["valid"]
            expected_scope = "valid"
        elif kind == "full_valid":
            expected_rows = membership["full_valid_order"]
            expected_scope = "full-valid-feature-only"
        else:
            raise ValueError(f"unexpected export kind: {kind}")
        npy_path = unique_suffix_file(output_root, item["npy"]["path"])
        sidecar_path = unique_suffix_file(output_root, item["sidecar"])
        npy_receipt = sha_from_inventory(npy_path, output_root, comparison)
        sidecar_receipt = sha_from_inventory(sidecar_path, output_root, comparison)
        sidecar = read_small_json(sidecar_path)
        checkpoint_sha = model_files[run_id]["model_ep2.pt"]["sha256"]
        if (
            sidecar.get("run") != run_id
            or sidecar.get("split") != source_split
            or sidecar.get("epoch") != 2
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("prediction_sha256") != npy_receipt["sha256"]
            or sidecar.get("split_manifest_sha256") != SPLIT_SHA256
            or sidecar.get("campaign_split_manifest_sha256") != SPLIT_SHA256
            or sidecar.get("campaign_fit_scope") != expected_scope
            or sidecar.get("row_groups") != expected_rows
            or sidecar.get("inference_precision") != "fp32"
        ):
            raise ValueError(f"prediction sidecar identity/membership mismatch: {run_id}:{kind}")
        array_summary = validate_npy(
            npy_path,
            expected_shape,
            item["npy"]["dtype"],
            finite_rows=None,
        )
        if not array_summary["all_values_finite"]:
            raise ValueError(f"full finite scan was not completed: {run_id}:{kind}")
        predictions.append(
            {
                "run_id": run_id,
                "kind": kind,
                "npy": npy_receipt,
                "sidecar": sidecar_receipt,
                "checkpoint_sha256": checkpoint_sha,
                "epoch": sidecar["epoch"],
                "membership_count": len(expected_rows),
                "source_qualified_membership_verified": True,
                "array": array_summary,
            }
        )

    if len(predictions) != 9 or len({row["npy"]["path"] for row in predictions}) != 9:
        raise ValueError("terminal output does not contain exactly nine distinct predictions")
    return {
        "schema": "wnn.kaggle.clean_t3_f0_r4_strict_terminal_validation.r1",
        "status": "pass",
        "kernel": R4_KERNEL,
        "dataset": R4_DATASET,
        "dataset_version": 1,
        "root_dispatch_admission_sha256": R4_FINAL_DISPATCH_SHA256,
        "archive_sha256": R4_ARCHIVE_SHA256,
        "source_qualified_membership": {
            "split_manifest_sha256": SPLIT_SHA256,
            "counts": membership["counts"],
            "fit_train_valid_evaluation_components_disjoint": True,
        },
        "paired_profiles": {
            "status": "pass",
            "distinct_t4_uuids": sorted(set(uuids)),
            "profile_workers": profile_workers,
            "summary": sha_from_inventory(profile_summary_path, output_root, comparison),
            "guard_receipt": guard_receipt,
            "preflight_receipt": preflight_receipt,
        },
        "pipeline": pipeline_receipt,
        "fit_workers": fit_stage,
        "export_workers": export_stage,
        "fit_artifacts": model_files,
        "prediction_exports": predictions,
        "terminal_collection_sha256": sha256_file(
            attempt_dir.parent / "TERMINAL_COLLECTION_RECEIPT_R1.json"
        ),
        "provider_inventory_matches_download": comparison.get("matches") is True,
        "all_prediction_arrays_finite": True,
        "scientific_grade": "not performed by collector",
    }


def load_provider_status(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a fresh poller snapshot and require terminal status before download."""
    dispatch = read_json(ROOT_RELEASE / "ROOT_FINAL_DISPATCH_ADMISSION_R1.json")
    push = read_json(ROOT_RELEASE / "ROOT_KERNEL_PUSH_R1.json")
    snapshot, status_row = parse_status_snapshot(path, push["observed_utc"])
    return dispatch, {"snapshot": snapshot, "row": status_row}


def run_readonly_inventory(status_path: Path, inventory_dir: Path) -> dict[str, Any]:
    """Capture the exact terminal-output filename inventory without downloading."""
    static_bindings = validate_frozen_r4_bindings()
    membership = load_expected_membership()
    _, observed = load_provider_status(status_path)
    status_row = observed["row"]
    if not load_adapter().terminal.terminal_state(status_row.get("status")):
        raise RuntimeError(
            f"provider is not terminal ({status_row.get('status')}); no inventory request sent"
        )
    adapter = load_adapter()
    source_bindings = adapter.verify_collector_sources()
    if inventory_dir.exists():
        raise FileExistsError(
            f"inventory directory already exists; use a new identified attempt: {inventory_dir}"
        )
    inventory_dir.mkdir(parents=True, mode=0o700)
    inventory = adapter.collect_session_output_inventory(
        kernel=R4_KERNEL,
        inventory_dir=inventory_dir,
        deadline=time.monotonic() + MAX_INVENTORY_SECONDS,
    )
    adapter.write_immutable_json(inventory_dir / "PROVIDER_INVENTORY_R1.json", inventory)
    receipt = {
        "schema": "wnn.kaggle.r4_readonly_terminal_inventory.r1",
        "kernel": R4_KERNEL,
        "dataset": R4_DATASET,
        "dataset_version": 1,
        "provider_status": status_row.get("status"),
        "provider_failure_message": status_row.get("failure_message", ""),
        "provider_status_snapshot_sha256": sha256_file(status_path),
        "provider_status_observed_at_utc": observed["snapshot"].get(
            "observed_at_utc", observed["snapshot"].get("observed_utc")
        ),
        "endpoint_inventory": inventory,
        "endpoint_is_version_pinned": inventory.get("kernel_version_pinned_by_endpoint") is True,
        "remote_sizes_or_checksums_available": inventory.get("remote_size_or_checksum_available")
        is True,
        "download_performed": False,
        "reviewed_r2_collector_sha256": R2_ADAPTER_SHA256,
        "collector_review_receipt_sha256": R2_REVIEW_SHA256,
        "collector_source_bindings": source_bindings,
        "archive_sha256": static_bindings["archive_sha256"],
        "membership_counts": membership["counts"],
        "output_names": [row["name"] for row in inventory.get("files", [])],
        "output_file_count": len(inventory.get("files", [])),
    }
    receipt_path = inventory_dir / "R4_READONLY_INVENTORY_RECEIPT_R1.json"
    adapter.write_immutable_json(receipt_path, receipt)
    return {
        "inventory_complete": inventory.get("complete") is True,
        "provider_status": status_row.get("status"),
        "output_file_count": receipt["output_file_count"],
        "remote_sizes_available": receipt["remote_sizes_or_checksums_available"],
        "download_performed": False,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def run_collection(
    status_path: Path,
    collection_dir: Path,
    admission_path: Path,
    inventory_receipt_path: Path,
) -> dict[str, Any]:
    """Use reviewed R2 inventory/download primitives, then validate all R4 artifacts."""
    static_bindings = validate_frozen_r4_bindings()
    membership = load_expected_membership()
    _dispatch, observed = load_provider_status(status_path)
    adapter = load_adapter()
    source_bindings = adapter.verify_collector_sources()
    snapshot = observed["snapshot"]
    status_row = observed["row"]
    if not adapter.terminal.terminal_state(status_row.get("status")):
        raise RuntimeError(
            f"provider is not terminal ({status_row.get('status')}); no inventory or download started"
        )
    if status_row.get("status") != "KernelWorkerStatus.ERROR":
        raise RuntimeError("this root authorization permits only the exact R4 ERROR download")
    admission = validate_download_admission(
        admission_path, status_path, inventory_receipt_path, collection_dir
    )
    if collection_dir.exists():
        raise FileExistsError(
            f"collection directory already exists; preserve it and use a new identified attempt: {collection_dir}"
        )
    collection_dir.mkdir(parents=True, mode=0o700)
    inventory_dir = collection_dir / "inventory"
    inventory_dir.mkdir(mode=0o700)
    inventory = adapter.collect_session_output_inventory(
        kernel=R4_KERNEL,
        inventory_dir=inventory_dir,
        deadline=time.monotonic() + MAX_INVENTORY_SECONDS,
    )
    adapter.write_immutable_json(collection_dir / "PROVIDER_INVENTORY_R1.json", inventory)
    if not inventory.get("complete"):
        raise RuntimeError("fresh terminal inventory is incomplete; no download started")
    try:
        require_exact_inventory_names(admission["expected_output_names"], inventory.get("files"))
    except ValueError as error:
        raise RuntimeError(f"{error}; no download started") from error
    if not inventory.get("complete") or adapter.inventory_log_collisions(inventory):
        raise RuntimeError("fresh session-output inventory is incomplete or collides with CLI log")
    target = {
        "kernel": R4_KERNEL,
        "resource": "gpu",
        "artifact_dir": "attempts",
        "terminal_receipt": "TERMINAL_COLLECTION_RECEIPT_R1.json",
    }
    config = {
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_download_bytes": MAX_DOWNLOAD_BYTES,
        "max_runtime_seconds": MAX_DOWNLOAD_SECONDS,
        "max_attempts": 1,
        "kaggle_cli": str(MAIN / ".venv/bin/kaggle"),
        "expected_cli_version": "Kaggle CLI 2.2.4",
    }
    result = adapter.terminal.collect_terminal_target(
        target=target,
        status_row=status_row,
        config=config,
        repo_root=MAIN,
        config_dir=collection_dir,
    )
    receipt_path = collection_dir / target["terminal_receipt"]
    attempt, error = adapter.resolve_attempt_dir(result, receipt_path, collection_dir / "attempts")
    if attempt is None or error is not None:
        raise RuntimeError(f"terminal collector did not produce one safe attempt: {error}")
    comparison = adapter.compare_provider_inventory(
        inventory["files"], attempt, provider_response_log=inventory.get("response_log")
    )
    low_level_complete = adapter.terminal.collection_is_complete(
        receipt_path, collection_dir / "attempts"
    )
    complete = adapter.terminal_evidence_complete(low_level_complete, inventory, comparison)
    if not complete:
        raise RuntimeError("R2 terminal collection completeness/inventory comparison failed")
    validation_path: Path
    if status_row.get("status") == "KernelWorkerStatus.COMPLETE":
        validation = validate_pipeline_outputs(attempt, comparison, static_bindings, membership)
        validation_path = collection_dir / "R4_STRICT_VALIDATION_R1.json"
        validation["status"] = "pass"
    else:
        validation = partial_terminal_diagnostics(attempt, comparison, status_row)
        validation["strict_science_validation"] = "not_run_terminal_error"
        validation_path = collection_dir / "R4_TERMINAL_DIAGNOSTICS_R1.json"
    validation.update(
        {
            "observed_provider_status": status_row.get("status"),
            "provider_status_observed_at_utc": snapshot.get(
                "observed_at_utc", snapshot.get("observed_utc")
            ),
            "provider_status_snapshot_sha256": sha256_file(status_path),
            "provider_status_endpoint_version_pinned": False,
            "reviewed_r2_collector_sha256": R2_ADAPTER_SHA256,
            "collector_source_bindings": source_bindings,
            "collector_review_receipt_sha256": R2_REVIEW_SHA256,
            "terminal_inventory_sha256": sha256_file(collection_dir / "PROVIDER_INVENTORY_R1.json"),
            "attempt_directory": str(attempt),
            "collector_result": result,
            "quota_snapshot_sha256": hashlib.sha256(
                json.dumps(snapshot.get("quota", {}), sort_keys=True).encode()
            ).hexdigest(),
            "runner_terminal_complete": status_row.get("status") == "KernelWorkerStatus.COMPLETE",
            "dispatch_root_receipt_sha256": R4_FINAL_DISPATCH_SHA256,
            "root_download_admission_sha256": admission["admission_sha256"],
            "root_readonly_inventory_receipt_sha256": admission["inventory_receipt_sha256"],
        }
    )
    write_immutable_json(validation_path, validation)
    return {
        "collection_complete": True,
        "strict_validation_status": validation["status"],
        "validation_receipt": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "provider_status": status_row.get("status"),
        "prediction_arrays": len(validation.get("prediction_exports", [])),
        "downloaded_file_count": validation.get("downloaded_file_count"),
        "downloaded_total_bytes": validation.get("downloaded_total_bytes"),
    }


def partial_terminal_diagnostics(
    attempt_dir: Path,
    comparison: dict[str, Any],
    status_row: dict[str, Any],
) -> dict[str, Any]:
    """Describe a completely downloaded failed run without implying science passed."""
    output_root = attempt_dir / "outputs"
    relative_files = sorted(comparison.get("per_file_sha256", {}))
    present = set(relative_files)
    plan = read_json(PACKET / "TEACHER_COHORT_PLAN_R3.json")
    expected_fit_names = [
        f"fit_clean_t3_f0_muonall_h384_e3_s{seed}_r3" for seed in plan["recipe"]["seeds"]
    ]
    expected_export_names = [
        f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r3_{kind}"
        for seed in plan["recipe"]["seeds"]
        for kind in ("fit_train", "fit_valid", "full_valid")
    ]

    def parse_optional(relative: str) -> dict[str, Any] | None:
        matches = [name for name in relative_files if name.endswith(relative)]
        if len(matches) != 1:
            return None
        path = output_root / matches[0]
        try:
            value = read_small_json(path)
        except (OSError, ValueError):
            return {"path": matches[0], "parse_status": "invalid_or_oversized"}
        return {"path": matches[0], "value": value}

    pipeline_entry = parse_optional("results/teacher_cohort_pipeline.json")
    pipeline_summary: dict[str, Any] | None = None
    if pipeline_entry and "value" in pipeline_entry:
        raw_pipeline = pipeline_entry["value"]
        pipeline_summary = {
            "status": raw_pipeline.get("status"),
            "stages": [
                {
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "returncode": row.get("returncode"),
                }
                for row in raw_pipeline.get("stages", [])
                if isinstance(row, dict)
            ],
        }

    stage_summaries: dict[str, Any] = {}
    for stage in ("paired_profiles", "teacher_fits", "teacher_exports"):
        item = parse_optional(f"results/{stage}/summary.json")
        if item is None:
            stage_summaries[stage] = {"present": False}
            continue
        summary_value = item.get("value", {})
        stage_summaries[stage] = {
            "present": "value" in item,
            "path": item["path"],
            "finished": [
                {
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "returncode": row.get("returncode"),
                }
                for row in summary_value.get("finished", [])
                if isinstance(row, dict)
            ],
            "incident_present": summary_value.get("incident") is not None,
            "not_started": summary_value.get("not_started"),
        }

    known_log_files = [
        name
        for name in relative_files
        if name.endswith("/worker.log") or name.endswith("/provider_logs.txt")
    ]
    fit_worker_artifact_paths = [path for item in plan["fit_stage"] for path in item["artifacts"]]
    expected_prediction_paths = [item["npy"]["path"] for item in plan["export_stage"]]
    present_fits = sorted(path for path in fit_worker_artifact_paths if path in present)
    present_predictions = sorted(path for path in expected_prediction_paths if path in present)
    return {
        "schema": "wnn.kaggle.clean_t3_f0_r4_terminal_error_diagnostics.r1",
        "status": "terminal_download_complete_science_not_validated",
        "kernel": R4_KERNEL,
        "dataset": R4_DATASET,
        "dataset_version": 1,
        "provider_status": status_row.get("status"),
        "provider_failure_message": status_row.get("failure_message", ""),
        "provider_inventory_matches_download": comparison.get("matches") is True,
        "provider_file_count": comparison.get("provider_file_count"),
        "downloaded_file_count": comparison.get("downloaded_file_count"),
        "downloaded_total_bytes": sum(
            row.get("size_bytes", 0)
            for row in comparison.get("per_file_sha256", {}).values()
            if isinstance(row, dict)
        ),
        "inventory_remote_sizes_available": comparison.get("remote_sizes_available"),
        "teacher_pipeline": pipeline_summary or {"present": False},
        "stage_summaries": stage_summaries,
        "expected_fit_workers": expected_fit_names,
        "expected_export_workers": expected_export_names,
        "expected_fit_artifacts_present": present_fits,
        "expected_prediction_arrays_present": present_predictions,
        "worker_logs_present": known_log_files,
        "provider_files": [
            {
                "path": relative,
                "size_bytes": comparison["per_file_sha256"][relative].get("size_bytes"),
                "sha256": comparison["per_file_sha256"][relative].get("sha256"),
            }
            for relative in relative_files
        ],
        "scientific_grade": "not performed; ERROR terminal is execution evidence only",
    }


def write_immutable_json(path: Path, value: dict[str, Any]) -> str:
    """Write one JSON receipt with exclusive creation and no replacement."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    """Validate the sealed collector plan or collect after an external terminal receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-snapshot", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--inventory-dir", type=Path)
    parser.add_argument("--collection-dir", type=Path)
    parser.add_argument("--download-admission", type=Path)
    parser.add_argument("--inventory-receipt", type=Path)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    try:
        static = validate_frozen_r4_bindings()
        membership = load_expected_membership()
        if args.inventory_only:
            if args.status_snapshot is None or args.inventory_dir is None:
                parser.error("--inventory-only requires --status-snapshot and --inventory-dir")
            result = run_readonly_inventory(
                args.status_snapshot.resolve(), args.inventory_dir.resolve()
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if not args.collect:
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "collection_required": True,
                        "kernel": R4_KERNEL,
                        "archive_sha256": static["archive_sha256"],
                        "prediction_exports": 9,
                        "membership_counts": membership["counts"],
                        "next_action": "wait for exact R4 terminal status and root memory admission",
                    },
                    sort_keys=True,
                )
            )
            return 0
        if (
            args.status_snapshot is None
            or args.collection_dir is None
            or args.download_admission is None
            or args.inventory_receipt is None
        ):
            parser.error(
                "--collect requires --status-snapshot, --collection-dir, "
                "--download-admission, and --inventory-receipt"
            )
        result = run_collection(
            args.status_snapshot.resolve(),
            args.collection_dir.resolve(),
            args.download_admission.resolve(),
            args.inventory_receipt.resolve(),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, FileExistsError) as error:
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(error).__name__, "error": str(error)}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
