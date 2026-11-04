"""One-shot metadata recovery for the frozen R1 teacher/student continuation.

The R1 panel, one-shot receipts, teacher fits, and R1 stage-two source stay
immutable. This R2 entrypoint compacts semantically identical target metadata,
uses an explicitly owned compact finalizer for the still-pending Transformer
cache, then resumes only the two predeclared student-family queues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RECOVERY_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RECOVERY_ROOT), str(RECOVERY_ROOT / "src")]

from scripts.build_distill_cache import _target_order  # noqa: E402
from scripts.l4_teacher_capacity_20261104.continue_students import (  # noqa: E402
    _check_free_floor,
    _deadline,
    _receipt,
    _remaining,
    _run_stage2,
    _runtime,
    _student_family,
    _teacher_cells,
    _teacher_export,
    _wait_teachers,
    validate_panel,
)
from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (  # noqa: E402
    EXPECTED_RUNS,
    PR60_CACHE_METADATA_SHA256,
    PR60_FAMILIES_TRAIN_SHA256,
    PR60_MAIN_TRAIN_SHA256,
    SOURCE_TRAIN_INDEX_SEQUENCES,
    SPLIT_SHA256,
    TARGET_BYTES,
    TRAIN_SEQUENCES,
    combine_target_chunk,
    fit_source_positions,
    load_state,
    partial_root,
    positions_sha256,
    sequence_hash,
    state_path,
)
from scripts.l4_teacher_capacity_20261104.stage2_stream_finalize import (  # noqa: E402
    _array_record,
)
from wnn.distillation_cache import (  # noqa: E402
    CACHE_FORMAT,
    CACHE_VERSION,
    MAX_CACHE_METADATA_BYTES,
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
    validate_distillation_cache,
)
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_budget import utc_now  # noqa: E402
from wnn.multistage_search_owned_command import run_owned_command  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    base_environment,
    current_process_owner,
)
from wnn.splits import load_campaign_split  # noqa: E402

RECOVERY_ADMISSION_SCHEMA = "wnn.l4-student-recovery-source-admission.v1"
RECOVERY_STATE_SCHEMA = "wnn.l4-teacher-capacity-continuation-recovery.v1"
RECOVERY_RECEIPT_SCHEMA = "wnn.l4-teacher-stage2-metadata-recovery.v1"
CPU_AFFINITY = list(range(32, 40))
CONTROLLER_AFFINITY = list(range(64))


def _read_json(path: Path, limit: int = 32 << 20) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"expected a regular JSON object no larger than {limit} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"preserve existing recovery evidence; do not retry: {path}")
    atomic_write_json(path, value, replace=False)


def _sha_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _admission(path: Path, panel: dict[str, Any], panel_hash: str) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if (
        receipt.get("schema") != RECOVERY_ADMISSION_SCHEMA
        or not str(receipt.get("status", "")).startswith("approved")
        or receipt.get("continuation_panel_sha256") != panel_hash
    ):
        raise ValueError("recovery source admission does not approve this frozen R1 continuation")
    recovery_source_sha = receipt.get("source_binding_sha256")
    if not isinstance(recovery_source_sha, str) or len(recovery_source_sha) != 64:
        raise ValueError("recovery admission lacks its new source binding SHA256")
    expected = panel["identity"]
    for key in (
        "data_binding_sha256",
        "split_sha256",
        "fit_members_sha256",
        "baseline_ancestry_sha256",
    ):
        if receipt.get(key) != expected.get(key):
            raise ValueError(f"recovery admission differs from the frozen R1 panel on {key}")
    if expected.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("recovery panel split differs from the frozen chronology")
    return receipt, file_sha256(path)


def _check_controller_receipt(path: Path, panel_hash: str, recovery_source_sha: str) -> dict[str, Any]:
    receipt = _read_json(path)
    current = current_process_owner()
    owner = receipt.get("owner", {})
    if (
        receipt.get("schema") != "wnn.multistage-owned-process.v1"
        or receipt.get("status") != "running"
        or receipt.get("child") != current
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("recovery_source_binding_sha256") != recovery_source_sha
    ):
        raise ValueError("recovery controller is not running under its exact owned receipt")
    return {"path": str(path.resolve()), "child": current}


def _expected_stage2_receipt(
    panel: dict[str, Any],
    panel_hash: str,
    family: str,
    index: int,
    args: list[str],
    *,
    status: str,
    exit_code: int,
) -> tuple[Path, Path, dict[str, Any]]:
    cell_id = f"{family}_stage2_{index:02d}"
    cell = {"cell_id": cell_id}
    log_path, receipt_path = _receipt(panel, cell, "stage2")
    receipt = _read_json(receipt_path)
    command = receipt.get("command")
    paths = panel["paths"]
    suffix = [
        paths["python"],
        str(Path(paths["source_root"]) / "scripts/l4_teacher_capacity_20261104/stream_stage2_targets.py"),
        *args,
    ]
    owner = receipt.get("owner", {})
    if (
        receipt.get("status") != status
        or receipt.get("exit_code") != exit_code
        or receipt.get("timed_out") is not False
        or not isinstance(command, list)
        or len(command) < len(suffix) + 3
        or Path(command[0]).name != "taskset"
        or command[1:3] != ["-c", ",".join(map(str, CPU_AFFINITY))]
        or command[-len(suffix) :] != suffix
        or receipt.get("output_dir") != str(Path(args[args.index("--output-cache") + 1]).resolve())
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("cell_id") != cell_id
        or owner.get("run_name") != cell_id
        or owner.get("split_sha256") != panel["identity"]["split_sha256"]
    ):
        raise ValueError(f"{cell_id}: original owned stage2 command receipt differs")
    return log_path, receipt_path, receipt


def _stage2_args(panel: dict[str, Any], family: str, action: str, run: str | None = None) -> list[str]:
    output = panel["families"][family]["target_cache"]
    args = [action, "--family", family, "--output-cache", output]
    if action == "init":
        args.extend(
            [
                "--project-root",
                panel["paths"]["project_root"],
                "--split-manifest",
                panel["paths"]["split_manifest"],
                "--old-cache",
                panel["paths"]["old_cache"],
            ]
        )
    elif action == "add":
        if run is None:
            raise ValueError("stage2 add needs a fixed teacher run")
        args.extend(["--run", run])
    return args


def _check_tf_stage2_pending(
    panel: dict[str, Any], panel_hash: str
) -> tuple[Path, dict[str, Any], list[dict[str, str]], list[dict[str, Any]]]:
    """Bind the two TF adds and prove the compact finalizer has not been attempted."""
    family = "causal_transformer"
    output = Path(panel["families"][family]["target_cache"])
    partial = partial_root(output)
    if output.exists() or output.is_symlink() or not partial.is_dir() or partial.is_symlink():
        raise FileExistsError("Transformer compact finalizer requires its one active partial")
    state = load_state(output, family)
    expected_runs = list(EXPECTED_RUNS[family])
    if (
        state.get("schema") != "wnn.l4-teacher-stage2-stream.v2"
        or state.get("processed_runs") != expected_runs
        or state.get("expected_runs") != expected_runs
        or state.get("pending_run") is not None
        or state.get("split_manifest_sha256") != SPLIT_SHA256
        or not Path(state.get("accumulator", "")).is_file()
    ):
        raise ValueError("Transformer stage2 stream is not exactly two completed adds")
    split = load_campaign_split(
        Path(panel["paths"]["split_manifest"]), Path(panel["paths"]["project_root"])
    )
    fit_groups = list(map(int, split.train_groups))
    if state.get("fit_row_groups") != fit_groups:
        raise ValueError("Transformer stage2 fit rows differ from the exact chronological train split")
    source_groups = state.get("source_train_index_row_groups")
    positions = state.get("fit_source_positions")
    expected_positions = fit_source_positions(source_groups, fit_groups)
    if (
        not isinstance(source_groups, list)
        or len(source_groups) != SOURCE_TRAIN_INDEX_SEQUENCES
        or positions != expected_positions.tolist()
        or state.get("fit_source_positions_sha256") != positions_sha256(expected_positions)
    ):
        raise ValueError("Transformer stage2 source-to-fit row mapping differs")

    receipts = []
    for index, action, run in [
        (1, "init", None),
        *[(index, "add", run) for index, run in enumerate(expected_runs, start=2)],
    ]:
        args = _stage2_args(panel, family, action, run)
        _, receipt_path, _ = _expected_stage2_receipt(
            panel,
            panel_hash,
            family,
            index,
            args,
            status="completed",
            exit_code=0,
        )
        receipts.append({"path": str(receipt_path.resolve()), "sha256": file_sha256(receipt_path)})
    finalize_log, finalize_receipt = _receipt(
        panel,
        {"cell_id": f"{family}_stage2_{len(expected_runs) + 2:02d}"},
        "stage2",
    )
    if finalize_log.exists() or finalize_receipt.exists():
        raise ValueError("an R1 Transformer finalize attempt exists; preserve it and stop")

    teacher_rows = _teacher_cells(panel, family)
    by_run = {record.get("run"): record for record in state.get("teacher_records", [])}
    if set(by_run) != set(expected_runs):
        raise ValueError("Transformer stage2 teacher records do not cover both fixed runs")
    export_artifacts = []
    paths = panel["paths"]
    project = Path(paths["project_root"])
    for index, row in enumerate(teacher_rows, start=2):
        run = row["run"]
        cell_id = f"{family}_export_{index:02d}_{run}"
        _, export_receipt_path = _receipt(
            panel, {"cell_id": cell_id, "run_name": run}, "teacher_export"
        )
        receipt = _read_json(export_receipt_path)
        teacher_panel = _read_json(row["teacher_panel"])
        teacher_source = Path(teacher_panel["paths"]["source_root"])
        export_root = project / "cache" / "l4_teacher_sequence_exports" / run
        export_gpu = panel["families"][family]["export_gpu_uuid"]
        affinity = panel["families"][family].get("export_cpu_affinity", range(16, 24))
        command = [
            str(teacher_panel["paths"]["python"]),
            str(teacher_source / "scripts/predict_sequence.py"),
            "--checkpoint",
            str(project / "artifacts/runs" / run / "model.pt"),
            "--data-root",
            str(teacher_panel["paths"]["data_root"]),
            "--split-manifest",
            str(paths["split_manifest"]),
            "--output",
            str(export_root),
            "--membership",
            "fit",
            "--batch-size",
            "32",
        ]
        expected = [shutil.which("taskset"), "-c", ",".join(map(str, affinity)), *command]
        owner = receipt.get("owner", {})
        record = by_run[run]
        if (
            receipt.get("status") != "completed"
            or receipt.get("exit_code") != 0
            or receipt.get("timed_out") is not False
            or receipt.get("command") != expected
            or receipt.get("output_dir") != str(export_root.resolve())
            or receipt.get("gpu_key") != export_gpu
            or owner.get("continuation_panel_sha256") != panel_hash
            or owner.get("panel_sha256") != row["teacher_panel_sha256"]
            or owner.get("cell_id") != cell_id
            or owner.get("run_name") != run
            or owner.get("source_binding_sha256") != row["source_binding_sha256"]
            or owner.get("data_binding_sha256") != row["data_binding_sha256"]
            or record.get("prediction_manifest_path") != str(export_root / "manifest.json")
        ):
            raise ValueError(f"Transformer fit-only exporter receipt differs: {run}")
        manifest_path = export_root / "manifest.json"
        if (
            not export_root.is_dir()
            or export_root.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.is_symlink()
            or file_sha256(manifest_path) != record.get("prediction_manifest_sha256")
            or any(path.suffix == ".npy" for path in export_root.rglob("*"))
            or any(path.name != "manifest.json" for path in export_root.iterdir())
        ):
            raise ValueError(f"Transformer export retained a prediction bank or lost its manifest: {run}")
        export_artifacts.append(
            {
                "run_name": run,
                "cell_id": cell_id,
                "process_receipt_path": str(export_receipt_path.resolve()),
                "process_receipt_sha256": file_sha256(export_receipt_path),
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "prediction_manifest_path": record["prediction_manifest_path"],
                "prediction_manifest_sha256": record["prediction_manifest_sha256"],
                "fit_rows": record["fit_groups"],
                "fit_row_groups_sha256": record["fit_row_groups_sha256"],
                "endpoint_epoch": 3,
                "membership": "fit",
            }
        )
    return partial, state, receipts, export_artifacts


def _check_wide_r1_history(
    panel: dict[str, Any], panel_path: Path, panel_hash: str
) -> dict[str, Any]:
    runtime = _runtime(panel)
    start_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_START_R1.json"
    failure_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_FAILED_R1.json"
    start, failure = _read_json(start_path), _read_json(failure_path)
    if (
        start.get("schema") != "wnn.l4-teacher-capacity-continuation-state.v1"
        or start.get("status") != "running"
        or start.get("panel_sha256") != panel_hash
        or failure.get("schema") != "wnn.l4-teacher-capacity-continuation-failure.v1"
        or failure.get("status") != "failed"
        or failure.get("panel_sha256") != panel_hash
        or failure.get("family") != "wide_gru"
        or failure.get("phase") != "stage2_target"
        or "metadata exceeds its 512000-byte read bound" not in str(failure.get("error", ""))
    ):
        raise ValueError("R1 continuation did not fail at the known wide-GRU metadata-size gate")
    stage2_receipts = []
    binding = panel["families"]["wide_gru"]
    init_receipt_path = Path(binding["stage2_init_receipt"])
    if (
        not init_receipt_path.is_file()
        or init_receipt_path.is_symlink()
        or file_sha256(init_receipt_path) != binding.get("stage2_init_receipt_sha256")
    ):
        raise ValueError("wide-GRU root-approved initialization receipt differs from frozen panel")
    init_receipt = _read_json(init_receipt_path)
    if (
        init_receipt.get("status") not in ("completed", "approved", "complete")
        or init_receipt.get("exit_code") != 0
        or init_receipt.get("family") != "wide_gru"
        or Path(str(init_receipt.get("output_cache", binding["target_cache"]))).resolve()
        != Path(binding["target_cache"]).resolve()
        or not isinstance(init_receipt.get("accumulator_sha256"), str)
        or init_receipt.get("zero_filled") is not True
    ):
        raise ValueError("wide-GRU approved init receipt fails its empty-state binding")
    init_log, unexpected_init = _receipt(
        panel,
        {"cell_id": "wide_gru_stage2_01", "run_name": "wide_gru_stage2_01"},
        "stage2",
    )
    if init_log.exists() or unexpected_init.exists():
        raise ValueError("unexpected second wide-GRU stage2 init exists; preserve it")
    stage2_receipts.append(
        {"path": str(init_receipt_path.resolve()), "sha256": file_sha256(init_receipt_path)}
    )
    for index, run in enumerate(EXPECTED_RUNS["wide_gru"], start=2):
        _, add_receipt_path, _ = _expected_stage2_receipt(
            panel,
            panel_hash,
            "wide_gru",
            index,
            _stage2_args(panel, "wide_gru", "add", run),
            status="completed",
            exit_code=0,
        )
        stage2_receipts.append(
            {"path": str(add_receipt_path.resolve()), "sha256": file_sha256(add_receipt_path)}
        )
    finalize_args = _stage2_args(panel, "wide_gru", "finalize")
    log_path, receipt_path, receipt = _expected_stage2_receipt(
        panel,
        panel_hash,
        "wide_gru",
        len(EXPECTED_RUNS["wide_gru"]) + 2,
        finalize_args,
        status="failed",
        exit_code=1,
    )
    if "metadata exceeds its 512000-byte read bound" not in log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise ValueError("R1 finalize log lacks the exact metadata-size validation error")
    return {
        "start_receipt": {"path": str(start_path.resolve()), "sha256": file_sha256(start_path)},
        "failure_receipt": {
            "path": str(failure_path.resolve()),
            "sha256": file_sha256(failure_path),
        },
        "finalize_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": file_sha256(receipt_path),
            "log_path": str(log_path.resolve()),
            "log_sha256": file_sha256(log_path),
            "command": receipt["command"],
            "owner": receipt["owner"],
        },
        "stage2_init_add_receipts": stage2_receipts,
    }


def _worker_context(
    receipt_path: Path,
    panel: dict[str, Any],
    panel_hash: str,
    admission_sha: str,
    family: str,
    recovery_source_sha: str,
    worker_mode: str,
    panel_path: Path,
    admission_path: Path,
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(receipt_path)
    child = current_process_owner()
    owner = receipt.get("owner", {})
    entrypoint = Path(__file__).resolve()
    taskset = shutil.which("taskset")
    expected_command = [
        taskset,
        "-c",
        ",".join(map(str, CPU_AFFINITY)),
        panel["paths"]["python"],
        str(entrypoint),
        "--worker-mode",
        worker_mode,
        "--family",
        family,
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path.resolve()),
        "--worker-receipt",
        str(receipt_path.resolve()),
    ]
    target_cache = str(Path(panel["families"][family]["target_cache"]).resolve())
    cell_id = (
        "wide_gru_stage2_recovery_r2"
        if family == "wide_gru"
        else "causal_transformer_stage2_compact_finalize_r2"
    )
    command_sha = hashlib.sha256(
        json.dumps(expected_command, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        receipt.get("schema") != "wnn.multistage-owned-process.v1"
        or receipt.get("status") != "running"
        or receipt.get("child") != child
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("family") != family
        or owner.get("recovery_admission_receipt_sha256") != admission_sha
        or owner.get("recovery_source_binding_sha256") != recovery_source_sha
        or owner.get("recovery_entrypoint_path") != str(entrypoint)
        or owner.get("recovery_entrypoint_sha256") != file_sha256(entrypoint)
        or owner.get("recovery_admission_receipt_path") != str(admission_path.resolve())
        or owner.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or owner.get("split_sha256") != panel["identity"]["split_sha256"]
        or owner.get("fit_members_sha256") != panel["identity"]["fit_members_sha256"]
        or owner.get("baseline_ancestry_sha256") != panel["identity"]["baseline_ancestry_sha256"]
        or owner.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
        or owner.get("cell_id") != cell_id
        or owner.get("run_name") != cell_id
        or receipt.get("command") != expected_command
        or receipt.get("output_dir") != target_cache
        or receipt.get("gpu_key") != "cpu-stage2"
        or owner.get("command_sha256") != command_sha
    ):
        raise ValueError(f"{family}: recovery worker is not bound to its live owned receipt")
    return child, recovery_source_sha


def _check_complete_partial(
    panel: dict[str, Any], family: str, partial: Path
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Validate all existing streamed bytes before changing metadata or cache names."""
    output = Path(panel["families"][family]["target_cache"])
    if output.exists() or output.is_symlink() or not partial.is_dir() or partial.is_symlink():
        raise FileExistsError("recovery requires the one known partial and an absent final cache")
    if (partial / "stage2_state.json").exists() or (partial / "clipped_sum.npy").exists():
        raise ValueError("stage2 is not in its post-finalize pre-promotion state")
    if not (partial / "main_source_train.npy").is_file():
        raise FileNotFoundError("final train target array is missing from the known partial")
    if not (partial / "stage2_build_receipt.json").is_file():
        raise FileNotFoundError("final stage2 build receipt is missing")
    if not (partial / "metadata.json").is_file():
        raise FileNotFoundError("original pretty target metadata is missing")

    project = Path(panel["paths"]["project_root"])
    split = load_campaign_split(Path(panel["paths"]["split_manifest"]), project)
    if split.sha256 != SPLIT_SHA256 or len(split.train_groups) != TRAIN_SEQUENCES:
        raise ValueError("cache recovery requires the exact frozen chronological fit split")
    source_order = _target_order(project, "train")
    fit_order = list(map(int, split.train_groups))
    if len(source_order) != SOURCE_TRAIN_INDEX_SEQUENCES:
        raise ValueError("source train index differs from the full 10,607-row-group mapping")
    positions = fit_source_positions(source_order, fit_order)
    old_cache = Path(panel["paths"]["old_cache"])
    old = validate_distillation_cache(old_cache, project, split, "source", require_families=True)
    old_files = old["metadata"]["files"]
    if (
        old["metadata_sha256"] != PR60_CACHE_METADATA_SHA256
        or old_files["main_source_train"]["sha256"] != PR60_MAIN_TRAIN_SHA256
        or old_files["families_train"]["sha256"] != PR60_FAMILIES_TRAIN_SHA256
        or old_files["main_source_train"]["row_groups"] != fit_order
        or old_files["families_train"]["row_groups"] != fit_order
    ):
        raise ValueError("qualified PR60 parent cache identity/order differs")

    build_path = partial / "stage2_build_receipt.json"
    build = _read_json(build_path)
    expected_runs = list(EXPECTED_RUNS[family])
    if (
        build.get("schema") != "wnn.l4-teacher-stage2-stream.v2"
        or build.get("family") != family
        or build.get("expected_runs") != expected_runs
        or build.get("processed_runs") != expected_runs
        or build.get("pending_run") is not None
        or build.get("split_manifest_sha256") != split.sha256
        or build.get("fit_row_groups") != fit_order
        or build.get("fit_row_groups_sha256") != row_groups_sha256(fit_order)
        or build.get("source_train_index_row_groups") != source_order
        or build.get("source_train_index_sha256") != sequence_hash(np.asarray(source_order, dtype="<i8"))
        or build.get("fit_source_positions") != positions.tolist()
        or build.get("fit_source_positions_sha256") != positions_sha256(positions)
        or build.get("old_cache_metadata_sha256") != old["metadata_sha256"]
        or build.get("old_main_sha256") != PR60_MAIN_TRAIN_SHA256
        or len(build.get("teacher_records", [])) != len(expected_runs)
    ):
        raise ValueError("completed stage2 build receipt does not bind the exact R2 row mapping")
    teacher_records = build["teacher_records"]
    if [record.get("run") for record in teacher_records] != expected_runs or any(
        record.get("bank_discarded_after_aggregation") is not True
        for record in teacher_records
    ):
        raise ValueError("stage2 teacher order or bounded bank lifecycle differs")

    metadata_path = partial / "metadata.json"
    metadata_raw = metadata_path.read_bytes()
    if len(metadata_raw) <= MAX_CACHE_METADATA_BYTES:
        raise ValueError("wide cache metadata is no longer the known oversized pretty file")
    metadata = json.loads(metadata_raw)
    if not isinstance(metadata, dict):
        raise ValueError("pretty target metadata is not an object")
    target = metadata.get("target_contract", {})
    files = metadata.get("files", {})
    if (
        metadata.get("cache_profile") != "l4_teacher_capacity_stage2_stream_v2"
        or metadata.get("split_manifest_sha256") != split.sha256
        or metadata.get("fit_groups")
        != {"train": fit_order, "valid": list(split.valid_groups)}
        or metadata.get("evaluation_valid_groups") != list(split.evaluation_groups)
        or metadata.get("base_cache", {}).get("metadata_sha256") != old["metadata_sha256"]
        or metadata.get("base_cache", {}).get("main_source_train_sha256") != PR60_MAIN_TRAIN_SHA256
        or target.get("family") != family
        or target.get("teacher_runs_equal_weight") != expected_runs
        or target.get("fit_groups") != fit_order
        or target.get("fit_groups_sha256") != row_groups_sha256(fit_order)
        or target.get("source_train_index_rows") != len(source_order)
        or target.get("source_train_index_sha256") != sequence_hash(np.asarray(source_order, dtype="<i8"))
        or target.get("fit_source_positions") != positions.tolist()
        or target.get("fit_source_positions_sha256") != positions_sha256(positions)
        or target.get("teachers") != teacher_records
        or metadata.get("prediction_banks_persisted") is not False
    ):
        raise ValueError("pretty metadata differs semantically from the completed streamed state")
    if set(files) != {
        "main_source_train",
        "main_source_valid",
        "families_train",
        "families_valid",
    }:
        raise ValueError("pretty target metadata has an unexpected array inventory")
    for key in files:
        entry = files[key]
        path = partial / entry["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != entry.get("sha256")
            or path.stat().st_size != entry.get("bytes")
        ):
            raise ValueError(f"partial target array differs from its metadata record: {key}")
    main_entry = files["main_source_train"]
    if (
        main_entry.get("bytes") != TARGET_BYTES
        or main_entry.get("shape") != [TRAIN_SEQUENCES, SEQUENCE_ROWS, 2]
        or main_entry.get("dtype") != "float32"
        or main_entry.get("row_groups") != fit_order
    ):
        raise ValueError("partial main training target differs from the frozen shape/order")
    main = np.load(partial / main_entry["path"], mmap_mode="r", allow_pickle=False)
    if main.shape != (TRAIN_SEQUENCES, SEQUENCE_ROWS, 2) or main.dtype != np.float32:
        raise ValueError("partial main training target dtype/shape is invalid")
    del main
    for split_name in ("train", "valid"):
        target_path = partial / files[f"families_{split_name}"]["path"]
        old_path = old_cache / old_files[f"families_{split_name}"]["path"]
        target_stat, old_stat = target_path.stat(), old_path.stat()
        if (target_stat.st_dev, target_stat.st_ino) != (old_stat.st_dev, old_stat.st_ino):
            raise ValueError(f"retained auxiliary target lost its qualified hardlink: {split_name}")
    for run in expected_runs:
        if (project / "cache" / f"pred_{run}_train.npy").exists():
            raise ValueError(f"teacher bank unexpectedly remains after aggregation: {run}")
    return metadata, build, split, old


def _store_compact_metadata(partial: Path, metadata: dict[str, Any]) -> tuple[str, str]:
    """Replace only JSON whitespace and preserve the original object exactly."""
    metadata_path = partial / "metadata.json"
    original_raw = metadata_path.read_bytes()
    original_sha = hashlib.sha256(original_raw).hexdigest()
    compact_raw = (
        json.dumps(metadata, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if json.loads(compact_raw) != metadata:
        raise ValueError("compact metadata serializer changed its parsed JSON value")
    if not 0 < len(compact_raw) <= MAX_CACHE_METADATA_BYTES:
        raise ValueError("semantically identical compact metadata still exceeds the reader bound")
    temporary = partial / "metadata.json.recovery.partial"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"preserve previous metadata rewrite attempt: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(compact_raw)
        stream.flush()
        os.fsync(stream.fileno())
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != original_sha:
        raise ValueError("original pretty metadata changed before atomic replacement")
    os.replace(temporary, metadata_path)
    return original_sha, hashlib.sha256(compact_raw).hexdigest()


def _preserve_original_metadata(panel: dict[str, Any], original: bytes) -> Path:
    path = _runtime(panel) / "receipts/stage2/wide_gru_metadata_original_pretty_r1.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"preserve existing original metadata backup: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def recover_wide_partial(
    panel_path: Path,
    panel: dict[str, Any],
    panel_hash: str,
    admission_path: Path,
    admission_sha: str,
    recovery_source_sha: str,
    worker_receipt_path: Path,
) -> dict[str, Any]:
    """Validate, compact, and promote only the already-complete wide partial."""
    worker_child, _ = _worker_context(
        worker_receipt_path,
        panel,
        panel_hash,
        admission_sha,
        "wide_gru",
        recovery_source_sha,
        "wide-recover-partial",
        panel_path,
        admission_path,
    )
    history = _check_wide_r1_history(panel, panel_path, panel_hash)
    output = Path(panel["families"]["wide_gru"]["target_cache"])
    partial = partial_root(output)
    metadata, build, split, _ = _check_complete_partial(panel, "wide_gru", partial)
    original_raw = (partial / "metadata.json").read_bytes()
    original_sha = hashlib.sha256(original_raw).hexdigest()
    backup = _preserve_original_metadata(panel, original_raw)
    recovery_path = _runtime(panel) / "receipts/stage2/wide_gru_metadata_recovery_r2.json"
    start_path = _runtime(panel) / "receipts/stage2/wide_gru_metadata_recovery_r2_START.json"
    _write_once(
        start_path,
        {
            "schema": RECOVERY_RECEIPT_SCHEMA,
            "status": "running",
            "family": "wide_gru",
            "continuation_panel_sha256": panel_hash,
            "recovery_source_binding_sha256": recovery_source_sha,
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
            "split_sha256": panel["identity"]["split_sha256"],
            "fit_members_sha256": panel["identity"]["fit_members_sha256"],
            "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "r1_failure_receipts": history,
            "original_pretty_metadata_sha256": original_sha,
            "original_pretty_metadata_backup_path": str(backup.resolve()),
            "owned_worker_receipt_path": str(worker_receipt_path.resolve()),
            "owned_worker_child": worker_child,
        },
    )
    compact_sha = None
    try:
        observed_original_sha, compact_sha = _store_compact_metadata(partial, metadata)
        if observed_original_sha != original_sha:
            raise ValueError("metadata bytes changed between validation and compaction")
        result = validate_distillation_cache(
            partial, Path(panel["paths"]["project_root"]), split, "source", require_families=True
        )
        if result["metadata"] != metadata:
            raise ValueError("canonical cache validator decoded different compact metadata")
        if output.exists() or output.is_symlink():
            raise FileExistsError("final target path appeared during metadata recovery")
        partial.replace(output)
        result = validate_distillation_cache(
            output, Path(panel["paths"]["project_root"]), split, "source", require_families=True
        )
        if result["metadata"] != metadata:
            raise ValueError("promoted cache differs semantically from the original pretty metadata")
    except Exception as error:
        failed_path = _runtime(panel) / "receipts/stage2/wide_gru_metadata_recovery_r2_FAILED.json"
        if not failed_path.exists():
            _write_once(
                failed_path,
                {
                    "schema": RECOVERY_RECEIPT_SCHEMA,
                    "status": "failed",
                    "family": "wide_gru",
                    "continuation_panel_sha256": panel_hash,
                    "recovery_source_binding_sha256": recovery_source_sha,
                    "recovery_admission_receipt_path": str(admission_path.resolve()),
                    "recovery_admission_receipt_sha256": admission_sha,
                    "recovery_entrypoint_path": str(Path(__file__).resolve()),
                    "recovery_entrypoint_sha256": file_sha256(Path(__file__).resolve()),
                    "original_pretty_metadata_sha256": original_sha,
                    "compact_metadata_sha256": compact_sha,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_epoch": time.time(),
                },
            )
        raise
    _write_once(
        recovery_path,
        {
            "schema": RECOVERY_RECEIPT_SCHEMA,
            "status": "compact_finalized_and_promoted",
            "family": "wide_gru",
            "continuation_panel_sha256": panel_hash,
            "recovery_source_binding_sha256": recovery_source_sha,
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
            "split_sha256": panel["identity"]["split_sha256"],
            "fit_members_sha256": panel["identity"]["fit_members_sha256"],
            "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "recovery_entrypoint_path": str(Path(__file__).resolve()),
            "recovery_entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "r1_failure_receipts": history,
            "original_pretty_metadata_sha256": original_sha,
            "original_pretty_metadata_backup_path": str(backup.resolve()),
            "compact_metadata_sha256": compact_sha,
            "semantic_metadata_sha256": _sha_json(metadata),
            "stage2_build_receipt_path": str((output / "stage2_build_receipt.json").resolve()),
            "stage2_build_receipt_sha256": file_sha256(output / "stage2_build_receipt.json"),
            "stage2_init_add_receipts": history["stage2_init_add_receipts"],
            "target_cache_path": str(output.resolve()),
            "target_metadata_sha256": file_sha256(output / "metadata.json"),
            "target_main_sha256": result["metadata"]["files"]["main_source_train"]["sha256"],
            "target_cache_bytes": TARGET_BYTES,
            "target_contract_sha256": _sha_json(metadata["target_contract"]),
            "teacher_records": build["teacher_records"],
            "owned_worker_receipt_path": str(worker_receipt_path.resolve()),
            "owned_worker_child": worker_child,
            "published_at_utc": utc_now(),
        },
    )
    return _read_json(recovery_path)


def finalize_target_compact(output: Path, family: str) -> dict[str, Any]:
    """Apply the R1 equation and layout, with bounded compact metadata serialization."""
    state = load_state(output, family)
    if state["processed_runs"] != state["expected_runs"]:
        raise ValueError("compact finalization requires every frozen teacher add")
    partial = partial_root(output)
    if output.exists() or output.is_symlink() or not partial.is_dir() or partial.is_symlink():
        raise FileExistsError("compact finalizer requires the one active stage2 partial")
    if any(
        (partial / name).exists()
        for name in ("main_source_train.npy", "main_source_valid.npy", "families_train.npy", "metadata.json", "stage2_build_receipt.json")
    ):
        raise FileExistsError("partial already contains finalize output; do not finalize again")
    old_cache = Path(state["old_cache"])
    project_root = Path(state["project_root"])
    split_path = Path(state["split_manifest"])
    split = load_campaign_split(split_path, project_root)
    if (
        split.sha256 != SPLIT_SHA256
        or file_sha256(old_cache / "metadata.json") != state["old_cache_metadata_sha256"]
        or state["old_cache_metadata_sha256"] != PR60_CACHE_METADATA_SHA256
        or state["old_main_sha256"] != PR60_MAIN_TRAIN_SHA256
    ):
        raise ValueError("compact finalizer base cache or chronological split changed")
    old_metadata = state["old_metadata"]
    old_main_entry = old_metadata["files"]["main_source_train"]
    old_main_path = old_cache / old_main_entry["path"]
    if file_sha256(old_main_path) != state["old_main_sha256"]:
        raise ValueError("qualified PR60 main target changed after initialization")
    accumulator_path = Path(state["accumulator"])
    if not accumulator_path.is_file() or accumulator_path.is_symlink():
        raise FileNotFoundError("compact finalizer accumulator is missing")
    old_main = np.load(old_main_path, mmap_mode="r", allow_pickle=False)
    total = np.lib.format.open_memmap(accumulator_path, mode="r+")
    teacher_count = len(state["processed_runs"])
    for start in range(0, TRAIN_SEQUENCES, 32):
        stop = min(start + 32, TRAIN_SEQUENCES)
        total[start:stop] = combine_target_chunk(
            np.asarray(old_main[start:stop], dtype=np.float32),
            np.asarray(total[start:stop], dtype=np.float32),
            teacher_count,
        )
    total.flush()
    del total, old_main
    main_path = partial / "main_source_train.npy"
    accumulator_path.replace(main_path)
    train_groups = list(map(int, state["fit_row_groups"]))
    valid_groups: list[int] = []
    main = np.load(main_path, mmap_mode="r", allow_pickle=False)
    if main.shape != (TRAIN_SEQUENCES, SEQUENCE_ROWS, 2) or main.dtype != np.float32:
        raise ValueError("compact stage2 main target shape/dtype differs")
    files: dict[str, dict[str, Any]] = {
        "main_source_train": _array_record(main_path, train_groups, main)
    }
    del main
    valid_main_path = partial / "main_source_valid.npy"
    if valid_main_path.exists():
        raise FileExistsError("valid target already exists in stage2 partial")
    valid_main = np.empty((0, SEQUENCE_ROWS, 2), dtype=np.float32)
    np.save(valid_main_path, valid_main, allow_pickle=False)
    files["main_source_valid"] = _array_record(valid_main_path, valid_groups, valid_main)
    for split_name, groups, width in (
        ("train", train_groups, 8),
        ("valid", valid_groups, 8),
    ):
        old_entry = old_metadata["files"][f"families_{split_name}"]
        source = old_cache / old_entry["path"]
        target = partial / f"families_{split_name}.npy"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"retained auxiliary target already exists: {split_name}")
        os.link(source, target)
        if target.stat().st_size != old_entry["bytes"]:
            raise ValueError(f"retained family array size changed: {split_name}")
        files[f"families_{split_name}"] = dict(old_entry, path=target.name)
        if old_entry["shape"] != [len(groups), SEQUENCE_ROWS, width]:
            raise ValueError(f"retained family array schema changed: {split_name}")
    build_receipt_path = partial / "stage2_build_receipt.json"
    if build_receipt_path.exists():
        raise FileExistsError("stage2 build receipt already exists before compact finalization")
    state_file = state_path(output)
    if not state_file.is_file() or state_file.is_symlink():
        raise FileNotFoundError("stage2 state disappeared before compact finalization")
    state_file.replace(build_receipt_path)
    metadata = {
        key: old_metadata[key]
        for key in (
            "format",
            "version",
            "fold",
            "split_manifest",
            "split_manifest_sha256",
            "fit_groups",
            "evaluation_valid_groups",
            "family_order",
            "source_coefficients",
            "coefficient_sum",
            "homogeneous_family",
            "output_profile",
            "stored_targets",
            "epoch_convention",
            "sources",
        )
    }
    metadata.update(
        {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "cache_profile": "l4_teacher_capacity_stage2_stream_v2",
            "source_coefficients_scope": "unchanged PR60 base target and retained auxiliary heads",
            "base_cache": {
                "path": str(old_cache),
                "metadata_sha256": state["old_cache_metadata_sha256"],
                "main_source_train_sha256": state["old_main_sha256"],
                "families_train_sha256": old_metadata["files"]["families_train"]["sha256"],
            },
            "target_contract": {
                "formula": "float32(0.8*T_old + 0.2*0.8125*mean(float32_clip(each_new_teacher,-2,2)))",
                "family": family,
                "teacher_runs_equal_weight": list(state["expected_runs"]),
                "teacher_epoch_zero_based": 3,
                "teacher_mean": "ordered float32 clipped sum divided by teacher count",
                "clipping": {
                    "lower": -2.0,
                    "upper": 2.0,
                    "applied_per_teacher_before_mean": True,
                },
                "base_weight": 0.8,
                "new_family_weight": 0.2,
                "new_family_source_scale": 0.8125,
                "fit_groups": state["fit_row_groups"],
                "fit_groups_sha256": state["fit_row_groups_sha256"],
                "source_train_index_rows": len(state["source_train_index_row_groups"]),
                "source_train_index_sha256": state["source_train_index_sha256"],
                "fit_source_positions": state["fit_source_positions"],
                "fit_source_positions_sha256": state["fit_source_positions_sha256"],
                "fit_source_mapping": "for each fit-group target row, select this index from the full chronological train prediction bank",
                "teachers": state["teacher_records"],
            },
            "retained_auxiliary_targets": {
                "source_cache": str(old_cache),
                "family_order": old_metadata["family_order"],
                "train_sha256": files["families_train"]["sha256"],
                "train_hardlink_reused_without_writes": True,
                "valid_sha256": files["families_valid"]["sha256"],
            },
            "student_contract": {
                "init": "tl3c_fbc1k_s5@ep3",
                "inherit_init_preprocessing": True,
                "seed_set": [1001, 1002],
                "epochs": 2,
                "checkpoint_policy": "fixed",
                "checkpoint_epoch": 1,
                "precision": "fp32",
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "distill_alpha": 0.7,
                "distill_family_weight": 0.1,
                "hidden": 256,
                "layers": 2,
                "batch": 16,
                "chunk": 2000,
            },
            "files": files,
            "estimated_new_cache_bytes": TARGET_BYTES,
            "prediction_banks_persisted": False,
        }
    )
    metadata_path = partial / "metadata.json"
    if metadata_path.exists() or metadata_path.is_symlink():
        raise FileExistsError("compact metadata path already exists before finalization")
    compact_raw = (
        json.dumps(metadata, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if not 0 < len(compact_raw) <= MAX_CACHE_METADATA_BYTES:
        raise ValueError("Transformer compact metadata exceeds the unchanged reader size bound")
    with metadata_path.open("xb") as stream:
        stream.write(compact_raw)
        stream.flush()
        os.fsync(stream.fileno())
    validated = validate_distillation_cache(
        partial, project_root, split, "source", require_families=True
    )
    if validated["metadata"] != metadata:
        raise ValueError("canonical validator decoded different Transformer metadata")
    partial.replace(output)
    validated = validate_distillation_cache(
        output, project_root, split, "source", require_families=True
    )
    if validated["metadata"] != metadata:
        raise ValueError("promoted Transformer cache differs semantically")
    return {
        "status": "compact_finalized_and_promoted",
        "family": family,
        "target_cache_path": str(output.resolve()),
        "target_metadata_sha256": file_sha256(output / "metadata.json"),
        "target_main_sha256": files["main_source_train"]["sha256"],
        "target_cache_bytes": TARGET_BYTES,
        "stage2_build_receipt_path": str((output / "stage2_build_receipt.json").resolve()),
        "stage2_build_receipt_sha256": file_sha256(output / "stage2_build_receipt.json"),
        "compact_metadata_sha256": hashlib.sha256(compact_raw).hexdigest(),
        "semantic_metadata_sha256": _sha_json(metadata),
        "target_contract_sha256": _sha_json(metadata["target_contract"]),
        "teacher_records": state["teacher_records"],
        "stage2_init_add_receipts": [],
    }


def finalize_tf_from_owned_worker(
    panel_path: Path,
    panel: dict[str, Any],
    panel_hash: str,
    admission_path: Path,
    admission_sha: str,
    recovery_source_sha: str,
    worker_receipt_path: Path,
) -> dict[str, Any]:
    family = "causal_transformer"
    worker_child, _ = _worker_context(
        worker_receipt_path,
        panel,
        panel_hash,
        admission_sha,
        family,
        recovery_source_sha,
        "finalize-compact",
        panel_path,
        admission_path,
    )
    _partial, _state, stage2_receipts, export_receipts = _check_tf_stage2_pending(
        panel, panel_hash
    )
    output = Path(panel["families"][family]["target_cache"])
    summary_path = _runtime(panel) / "receipts/stage2/causal_transformer_metadata_recovery_r2.json"
    start_path = _runtime(panel) / "receipts/stage2/causal_transformer_metadata_recovery_r2_START.json"
    _write_once(
        start_path,
        {
            "schema": RECOVERY_RECEIPT_SCHEMA,
            "status": "running",
            "family": family,
            "continuation_panel_sha256": panel_hash,
            "recovery_source_binding_sha256": recovery_source_sha,
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "owned_worker_receipt_path": str(worker_receipt_path.resolve()),
            "owned_worker_child": worker_child,
            "original_r1_finalize_attempted": False,
            "stage2_init_add_receipts": stage2_receipts,
        },
    )
    try:
        result = finalize_target_compact(output, family)
    except Exception as error:
        failed_path = _runtime(panel) / "receipts/stage2/causal_transformer_metadata_recovery_r2_FAILED.json"
        if not failed_path.exists():
            _write_once(
                failed_path,
                {
                    "schema": RECOVERY_RECEIPT_SCHEMA,
                    "status": "failed",
                    "family": family,
                    "continuation_panel_sha256": panel_hash,
                    "recovery_source_binding_sha256": recovery_source_sha,
                    "recovery_admission_receipt_path": str(admission_path.resolve()),
                    "recovery_admission_receipt_sha256": admission_sha,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_epoch": time.time(),
                },
            )
        raise
    result.update(
        {
            "schema": RECOVERY_RECEIPT_SCHEMA,
            "continuation_panel_sha256": panel_hash,
            "recovery_source_binding_sha256": recovery_source_sha,
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
            "split_sha256": panel["identity"]["split_sha256"],
            "fit_members_sha256": panel["identity"]["fit_members_sha256"],
            "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "recovery_entrypoint_path": str(Path(__file__).resolve()),
            "recovery_entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "owned_worker_receipt_path": str(worker_receipt_path.resolve()),
            "owned_worker_child": worker_child,
            "stage2_init_add_receipts": stage2_receipts,
            "teacher_export_receipts": export_receipts,
            "original_r1_finalize_attempted": False,
            "published_at_utc": utc_now(),
        }
    )
    result["target_contract_sha256"] = _sha_json(
        validate_distillation_cache(
            output,
            Path(panel["paths"]["project_root"]),
            load_campaign_split(
                Path(panel["paths"]["split_manifest"]), Path(panel["paths"]["project_root"])
            ),
            "source",
            require_families=True,
        )["metadata"]["target_contract"]
    )
    result["stage2_build_receipt_path"] = str((output / "stage2_build_receipt.json").resolve())
    result["stage2_build_receipt_sha256"] = file_sha256(output / "stage2_build_receipt.json")
    _write_once(summary_path, result)
    return result


def _worker_cli(args: argparse.Namespace) -> int:
    panel_path = Path(args.panel)
    panel, panel_hash = validate_panel(panel_path)
    admission_path = Path(args.recovery_admission)
    admission, admission_sha = _admission(admission_path, panel, panel_hash)
    recovery_source_sha = admission["source_binding_sha256"]
    entrypoint = Path(__file__).resolve()
    worker_receipt_path = Path(args.worker_receipt)
    if args.worker_mode == "wide-recover-partial" and args.family == "wide_gru":
        expected_cell_id = "wide_gru_stage2_recovery_r2"
    elif args.worker_mode == "finalize-compact" and args.family == "causal_transformer":
        expected_cell_id = "causal_transformer_stage2_compact_finalize_r2"
    else:
        raise ValueError("worker mode/family is outside the two frozen recovery operations")
    _, expected_receipt = _receipt(
        panel,
        {"cell_id": expected_cell_id, "run_name": expected_cell_id},
        "stage2",
    )
    if worker_receipt_path.resolve() != expected_receipt.resolve():
        raise ValueError("worker receipt path differs from the exact wide recovery binding")
    target_cache = Path(panel["families"][args.family]["target_cache"]).resolve()
    if args.worker_mode == "wide-recover-partial" and target_cache.exists():
        raise FileExistsError("wide target already exists; recovery is one-shot and will not retry")
    _check_free_floor(panel)
    if args.worker_mode == "wide-recover-partial":
        result = recover_wide_partial(
            panel_path,
            panel,
            panel_hash,
            admission_path,
            admission_sha,
            recovery_source_sha,
            worker_receipt_path,
        )
    else:
        result = finalize_tf_from_owned_worker(
            panel_path,
            panel,
            panel_hash,
            admission_path,
            admission_sha,
            recovery_source_sha,
            worker_receipt_path,
        )
    if (
        result.get("status") != "compact_finalized_and_promoted"
        or result.get("recovery_entrypoint_path") != str(entrypoint)
        or result.get("recovery_entrypoint_sha256") != file_sha256(entrypoint)
    ):
        raise ValueError("wide recovery summary does not bind this approved entrypoint")
    return 0


def _expected_worker_command(
    panel: dict[str, Any],
    mode: str,
    family: str,
    panel_path: Path,
    admission_path: Path,
    receipt_path: Path,
) -> list[str]:
    return [
        shutil.which("taskset"),
        "-c",
        ",".join(map(str, CPU_AFFINITY)),
        panel["paths"]["python"],
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--family",
        family,
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path.resolve()),
        "--worker-receipt",
        str(receipt_path.resolve()),
    ]


def _wait_for_receipt(path: Path, deadline: int) -> dict[str, Any]:
    while not path.is_file():
        if _remaining(deadline) < 30:
            raise TimeoutError(f"owned receipt did not arrive before the collection reserve: {path}")
        time.sleep(5)
    while True:
        record = _read_json(path)
        if record.get("status") not in ("launching", "running"):
            return record
        if _remaining(deadline) < 30:
            raise TimeoutError(f"owned process is still running as the collection reserve begins: {path}")
        time.sleep(5)


def _validate_completed_worker(
    panel: dict[str, Any],
    panel_hash: str,
    admission_sha: str,
    recovery_source_sha: str,
    family: str,
    mode: str,
    panel_path: Path,
    admission_path: Path,
    receipt_path: Path,
    record: dict[str, Any],
) -> None:
    expected_command = _expected_worker_command(
        panel, mode, family, panel_path, admission_path, receipt_path
    )
    entrypoint = Path(__file__).resolve()
    owner = record.get("owner", {})
    cell_id = (
        "wide_gru_stage2_recovery_r2"
        if family == "wide_gru"
        else "causal_transformer_stage2_compact_finalize_r2"
    )
    if (
        record.get("schema") != "wnn.multistage-owned-process.v1"
        or record.get("status") != "completed"
        or record.get("exit_code") != 0
        or record.get("timed_out") is not False
        or record.get("command") != expected_command
        or record.get("command_sha256")
        != hashlib.sha256(json.dumps(expected_command, separators=(",", ":")).encode()).hexdigest()
        or record.get("output_dir") != str(Path(panel["families"][family]["target_cache"]).resolve())
        or record.get("gpu_key") != "cpu-stage2"
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
        or owner.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or owner.get("split_sha256") != panel["identity"]["split_sha256"]
        or owner.get("fit_members_sha256") != panel["identity"]["fit_members_sha256"]
        or owner.get("baseline_ancestry_sha256") != panel["identity"]["baseline_ancestry_sha256"]
        or owner.get("recovery_source_binding_sha256") != recovery_source_sha
        or owner.get("recovery_admission_receipt_sha256") != admission_sha
        or owner.get("recovery_admission_receipt_path") != str(admission_path.resolve())
        or owner.get("recovery_entrypoint_path") != str(entrypoint)
        or owner.get("recovery_entrypoint_sha256") != file_sha256(entrypoint)
        or owner.get("family") != family
        or owner.get("cell_id") != cell_id
        or owner.get("run_name") != cell_id
    ):
        raise ValueError(f"{family}: completed recovery worker receipt identity differs")


def _wait_cache_review(
    panel: dict[str, Any], panel_hash: str, family: str, deadline: int
) -> tuple[dict[str, Any], Path]:
    binding = panel["families"][family]
    cache = Path(binding["target_cache"])
    metadata_path = cache / "metadata.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError(f"{family}: finalized target metadata is missing")
    metadata_sha = file_sha256(metadata_path)
    review_path = Path(binding["cache_admission_receipt"])
    while not review_path.is_file():
        if _remaining(deadline) < 30:
            raise TimeoutError(f"{family}: independent cache review did not arrive in time")
        time.sleep(10)
    review = _read_json(review_path)
    if (
        not str(review.get("status", "")).startswith("approved")
        or review.get("family") != family
        or review.get("continuation_panel_sha256") != panel_hash
        or review.get("target_cache_path") != str(cache.resolve())
        or review.get("target_metadata_sha256") != metadata_sha
        or review.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
        or review.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or review.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(f"{family}: independent cache review differs from finalized target bytes")
    binding["target_metadata_sha256"] = metadata_sha
    return review, review_path


def _validate_teacher_admission(
    panel: dict[str, Any], family: str, rows: list[dict[str, Any]]
) -> None:
    binding = panel["families"][family]
    path = Path(binding["teacher_admission_receipt"])
    if not path.is_file() or path.is_symlink() or file_sha256(path) != binding.get(
        "teacher_admission_receipt_sha256"
    ):
        raise FileNotFoundError(f"{family}: frozen teacher/cache admission is required")
    admission = _read_json(path)
    if (
        not str(admission.get("status", "")).startswith("approved")
        or admission.get("family") != family
        or admission.get("teacher_panel_sha256") != rows[0]["teacher_panel_sha256"]
        or admission.get("source_binding_sha256") != rows[0]["source_binding_sha256"]
        or admission.get("data_binding_sha256") != rows[0]["data_binding_sha256"]
        or admission.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(f"{family}: frozen teacher/cache admission identity differs")


def _build_tf_target(
    panel: dict[str, Any],
    panel_path: Path,
    panel_hash: str,
    admission_path: Path,
    admission_sha: str,
    recovery_source_sha: str,
    deadline: int,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    family = "causal_transformer"
    rows = _wait_teachers(panel, family, deadline)
    _validate_teacher_admission(panel, family, rows)
    output = Path(panel["families"][family]["target_cache"])
    if output.exists() or output.is_symlink() or partial_root(output).exists():
        raise FileExistsError("Transformer stage2 target already exists; do not retry initialization")
    init_args = _stage2_args(panel, family, "init")
    _run_stage2(panel, panel_hash, family, init_args, deadline, 1)
    for index, row in enumerate(rows, start=2):
        export = _teacher_export(panel, panel_hash, family, row, deadline, index)
        add_args = _stage2_args(panel, family, "add", row["run"])
        add_args.extend(
            ["--sequence-export", str(export), "--teacher-panel", str(row["teacher_panel"])]
        )
        _run_stage2(panel, panel_hash, family, add_args, deadline, index)
    cell_id = "causal_transformer_stage2_compact_finalize_r2"
    log_path, receipt_path = _receipt(
        panel, {"cell_id": cell_id, "run_name": cell_id}, "stage2"
    )
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError("preserve existing Transformer compact-finalizer attempt")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for bounded CPU stage2 finalization")
    command = _expected_worker_command(
        panel,
        "finalize-compact",
        family,
        panel_path,
        admission_path,
        receipt_path,
    )
    owner = current_process_owner()
    owner.update(
        {
            "continuation_panel_sha256": panel_hash,
            "source_binding_sha256": panel["identity"]["source_binding_sha256"],
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
            "split_sha256": panel["identity"]["split_sha256"],
            "fit_members_sha256": panel["identity"]["fit_members_sha256"],
            "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
            "recovery_source_binding_sha256": recovery_source_sha,
            "recovery_admission_receipt_path": str(admission_path.resolve()),
            "recovery_admission_receipt_sha256": admission_sha,
            "recovery_entrypoint_path": str(Path(__file__).resolve()),
            "recovery_entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "family": family,
            "cell_id": cell_id,
            "run_name": cell_id,
        }
    )
    command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    owner["command_sha256"] = command_sha
    env = base_environment("", RECOVERY_ROOT)
    env.update(
        {
            "WNN_PROJECT_ROOT": str(panel["paths"]["project_root"]),
            "WNN_THREADS": "8",
            "WNN_LOADER_WORKERS": "2",
            "WNN_LOADER_RAM_GB": "8",
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        }
    )
    _check_free_floor(panel)
    code, timed_out, _ = run_owned_command(
        command,
        cwd=RECOVERY_ROOT,
        env=env,
        log_path=log_path,
        receipt_path=receipt_path,
        owner=owner,
        stage=3,
        trial_number=0,
        mode="l4tc_stage2_compact_finalize",
        gpu_key="cpu-stage2",
        output_dir=output,
        timeout_seconds=min(36_000, _remaining(deadline)),
    )
    if code != 0 or timed_out:
        raise RuntimeError("Transformer compact finalizer failed; preserve its owned receipt")
    worker_receipt = _read_json(receipt_path)
    _validate_completed_worker(
        panel,
        panel_hash,
        admission_sha,
        recovery_source_sha,
        family,
        "finalize-compact",
        panel_path,
        admission_path,
        receipt_path,
        worker_receipt,
    )
    summary_path = _runtime(panel) / "receipts/stage2/causal_transformer_metadata_recovery_r2.json"
    summary = _read_json(summary_path)
    if (
        summary.get("status") != "compact_finalized_and_promoted"
        or summary.get("continuation_panel_sha256") != panel_hash
        or summary.get("recovery_source_binding_sha256") != recovery_source_sha
        or summary.get("owned_worker_child") != worker_receipt.get("child")
    ):
        raise ValueError("Transformer compact-finalizer summary does not match its owner receipt")
    panel["families"][family]["teacher_export_artifacts"] = summary["teacher_export_receipts"]
    return summary, receipt_path, worker_receipt


def _controller_context(
    receipt_path: Path,
    panel: dict[str, Any],
    panel_path: Path,
    admission_path: Path,
    panel_hash: str,
    admission_sha: str,
    recovery_source_sha: str,
) -> dict[str, Any]:
    record = _read_json(receipt_path)
    child = current_process_owner()
    entrypoint = Path(__file__).resolve()
    command = [
        shutil.which("taskset"),
        "-c",
        ",".join(map(str, CONTROLLER_AFFINITY)),
        panel["paths"]["python"],
        str(entrypoint),
        "--controller-mode",
        "continue-students",
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path.resolve()),
        "--controller-receipt",
        str(receipt_path.resolve()),
    ]
    owner = record.get("owner", {})
    if (
        record.get("schema") != "wnn.multistage-owned-process.v1"
        or record.get("status") != "running"
        or record.get("child") != child
        or record.get("command") != command
        or record.get("command_sha256")
        != hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
        or record.get("output_dir") != str(Path(panel["paths"]["project_root"]).resolve())
        or record.get("gpu_key") != "cpu-stage2"
        or owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
        or owner.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or owner.get("split_sha256") != panel["identity"]["split_sha256"]
        or owner.get("fit_members_sha256") != panel["identity"]["fit_members_sha256"]
        or owner.get("baseline_ancestry_sha256") != panel["identity"]["baseline_ancestry_sha256"]
        or owner.get("recovery_source_binding_sha256") != recovery_source_sha
        or owner.get("recovery_admission_receipt_path") != str(admission_path.resolve())
        or owner.get("recovery_admission_receipt_sha256") != admission_sha
        or owner.get("recovery_entrypoint_path") != str(entrypoint)
        or owner.get("recovery_entrypoint_sha256") != file_sha256(entrypoint)
        or owner.get("cell_id") != "student_continuation_recovery_r2_controller"
        or owner.get("run_name") != "student_continuation_recovery_r2_controller"
        or owner.get("family") != "continuation"
    ):
        raise ValueError("continuation controller is not running under its exact owned receipt")
    return {"child": child, "command_sha256": record["command_sha256"]}


def _wait_wide_recovery(
    panel: dict[str, Any],
    panel_path: Path,
    panel_hash: str,
    admission_path: Path,
    admission_sha: str,
    recovery_source_sha: str,
    deadline: int,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    family = "wide_gru"
    receipt_path = _runtime(panel) / "receipts/stage2/wide_gru_stage2_recovery_r2.json"
    record = _wait_for_receipt(receipt_path, deadline)
    _validate_completed_worker(
        panel,
        panel_hash,
        admission_sha,
        recovery_source_sha,
        family,
        "wide-recover-partial",
        panel_path,
        admission_path,
        receipt_path,
        record,
    )
    summary_path = _runtime(panel) / "receipts/stage2/wide_gru_metadata_recovery_r2.json"
    summary = _read_json(summary_path)
    target_cache = Path(panel["families"][family]["target_cache"])
    if (
        summary.get("schema") != RECOVERY_RECEIPT_SCHEMA
        or summary.get("status") != "compact_finalized_and_promoted"
        or summary.get("family") != family
        or summary.get("continuation_panel_sha256") != panel_hash
        or summary.get("recovery_source_binding_sha256") != recovery_source_sha
        or summary.get("recovery_admission_receipt_sha256") != admission_sha
        or summary.get("owned_worker_receipt_path") != str(receipt_path.resolve())
        or summary.get("owned_worker_child") != record.get("child")
        or summary.get("target_cache_path") != str(target_cache.resolve())
        or summary.get("target_metadata_sha256") != file_sha256(target_cache / "metadata.json")
        or len(summary.get("stage2_init_add_receipts", [])) != 5
    ):
        raise ValueError("wide-GRU metadata recovery summary does not match its approved target")
    return summary, receipt_path, record


def _wide_export_artifacts(panel: dict[str, Any], panel_hash: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    family = "wide_gru"
    artifacts = []
    for index, teacher in enumerate(summary["teacher_records"], start=2):
        run = teacher["run"]
        cell_id = f"wide_gru_native_export_{index:02d}_{run}"
        _, receipt_path = _receipt(
            panel, {"cell_id": cell_id, "run_name": run}, "teacher_export"
        )
        record = _read_json(receipt_path)
        owner = record.get("owner", {})
        if (
            record.get("status") != "completed"
            or record.get("exit_code") != 0
            or record.get("timed_out") is not False
            or owner.get("continuation_panel_sha256") != panel_hash
            or owner.get("cell_id") != cell_id
            or owner.get("run_name") != run
            or owner.get("family") != family
        ):
            raise ValueError(f"wide-GRU native exporter receipt differs: {run}")
        artifacts.append(
            {
                "run_name": run,
                "cell_id": cell_id,
                "process_receipt_path": str(receipt_path.resolve()),
                "process_receipt_sha256": file_sha256(receipt_path),
                "prediction_path": teacher.get("prediction_path"),
                "prediction_sha256": teacher.get("prediction_sha256"),
                "prediction_sidecar_sha256": teacher.get("prediction_sidecar_sha256"),
                "checkpoint_path": teacher.get("checkpoint_path"),
                "checkpoint_sha256": teacher.get("checkpoint_sha256"),
                "provenance_sha256": teacher.get("provenance_sha256"),
                "source_train_index_rows": teacher.get("source_train_index_rows"),
                "source_train_index_sha256": teacher.get("source_train_index_sha256"),
                "fit_rows": teacher.get("fit_rows"),
                "fit_row_groups_sha256": teacher.get("fit_row_groups_sha256"),
                "fit_source_positions_sha256": teacher.get("fit_source_positions_sha256"),
                "campaign_export_scope": "chronological-full-train-index-feature-only",
                "feature_only": True,
                "endpoint_epoch": 3,
                "inference_precision": "fp32",
            }
        )
    panel["families"][family]["teacher_export_artifacts"] = artifacts
    return artifacts


def _family_completion(
    panel: dict[str, Any],
    panel_hash: str,
    family: str,
    student_result: dict[str, Any],
    recovery_summary: dict[str, Any],
    stage2_owned_receipt_path: Path,
) -> dict[str, Any]:
    grade_path = Path(student_result["grade_receipt_path"])
    grade_receipt = _read_json(grade_path)
    recovery_path = _runtime(panel) / "receipts/stage2" / (
        f"{family}_metadata_recovery_r2.json"
    )
    stage2_receipt = _read_json(stage2_owned_receipt_path)
    target_cache = Path(panel["families"][family]["target_cache"])
    result = {
        "family": family,
        "target_cache": str(target_cache.resolve()),
        "target_cache_metadata_sha256": file_sha256(target_cache / "metadata.json"),
        "stage2_recovery_receipt_path": str(recovery_path.resolve()),
        "stage2_recovery_receipt_sha256": file_sha256(recovery_path),
        "stage2_owned_receipt_path": str(stage2_owned_receipt_path.resolve()),
        "stage2_owned_receipt_sha256": file_sha256(stage2_owned_receipt_path),
        "stage2_owned_child": stage2_receipt.get("child"),
        "stage2_target_contract_sha256": recovery_summary.get("target_contract_sha256"),
        "stage2_build_receipt_path": recovery_summary["stage2_build_receipt_path"],
        "stage2_build_receipt_sha256": recovery_summary["stage2_build_receipt_sha256"],
        "cache_review_receipt_path": str(
            Path(panel["families"][family]["cache_admission_receipt"]).resolve()
        ),
        "cache_review_receipt_sha256": file_sha256(
            Path(panel["families"][family]["cache_admission_receipt"])
        ),
        "profiles": student_result["profiles"],
        "fits": student_result["fits"],
        "fit_artifacts": student_result["fit_artifacts"],
        "valid_exports": student_result["valid_exports"],
        "grade": student_result["grade"],
        "grade_receipt_path": str(grade_path.resolve()),
        "grade_receipt_sha256": file_sha256(grade_path),
        "grade_status": grade_receipt["status"],
        "grade_candidate_rows": grade_receipt["candidate_rows"],
    }
    return result


def run_recovery_controller(
    panel_path: Path,
    panel: dict[str, Any],
    panel_hash: str,
    admission_path: Path,
    admission_sha: str,
    recovery_source_sha: str,
    controller_receipt_path: Path,
) -> dict[str, Any]:
    control = _controller_context(
        controller_receipt_path,
        panel,
        panel_path,
        admission_path,
        panel_hash,
        admission_sha,
        recovery_source_sha,
    )
    deadline = _deadline(panel)
    runtime = _runtime(panel)
    start_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R2_START.json"
    complete_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R2_COMPLETE.json"
    failed_path = runtime / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R2_FAILED.json"
    if complete_path.exists() or failed_path.exists():
        raise FileExistsError("R2 controller has terminal evidence already; do not restart")
    history = _check_wide_r1_history(panel, panel_path, panel_hash)
    base = {
        "schema": RECOVERY_STATE_SCHEMA,
        "panel_sha256": panel_hash,
        "source_binding_sha256": panel["identity"]["source_binding_sha256"],
        "data_binding_sha256": panel["identity"]["data_binding_sha256"],
        "split_sha256": panel["identity"]["split_sha256"],
        "fit_members_sha256": panel["identity"]["fit_members_sha256"],
        "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
        "recovery_source_binding_sha256": recovery_source_sha,
        "recovery_admission_receipt_path": str(admission_path.resolve()),
        "recovery_admission_receipt_sha256": admission_sha,
        "controller_receipt_path": str(controller_receipt_path.resolve()),
        "controller_child": control["child"],
        "controller_command_sha256": control["command_sha256"],
        "r1_start_receipt_path": history["start_receipt"]["path"],
        "r1_start_receipt_sha256": history["start_receipt"]["sha256"],
        "r1_failure_receipt_path": history["failure_receipt"]["path"],
        "r1_failure_receipt_sha256": history["failure_receipt"]["sha256"],
        "hard_deadline_epoch": deadline,
    }
    _write_once(start_path, {**base, "status": "running", "started_epoch": time.time()})
    families: dict[str, Any] = {}
    try:
        wide_summary, wide_receipt_path, _ = _wait_wide_recovery(
            panel,
            panel_path,
            panel_hash,
            admission_path,
            admission_sha,
            recovery_source_sha,
            deadline,
        )
        _wide_export_artifacts(panel, panel_hash, wide_summary)
        _wait_cache_review(panel, panel_hash, "wide_gru", deadline)
        wide_students = _student_family(panel, panel_hash, "wide_gru", deadline)
        wide_result = _family_completion(
            panel, panel_hash, "wide_gru", wide_students, wide_summary, wide_receipt_path
        )
        _write_once(
            runtime / "receipts/continuation/wide_gru_RECOVERY_COMPLETE_R2.json",
            {"schema": RECOVERY_STATE_SCHEMA, "status": "complete", **base, **wide_result},
        )
        families["wide_gru"] = wide_result

        tf_summary, tf_receipt_path, _ = _build_tf_target(
            panel,
            panel_path,
            panel_hash,
            admission_path,
            admission_sha,
            recovery_source_sha,
            deadline,
        )
        _wait_cache_review(panel, panel_hash, "causal_transformer", deadline)
        tf_students = _student_family(panel, panel_hash, "causal_transformer", deadline)
        tf_result = _family_completion(
            panel, panel_hash, "causal_transformer", tf_students, tf_summary, tf_receipt_path
        )
        _write_once(
            runtime / "receipts/continuation/causal_transformer_RECOVERY_COMPLETE_R2.json",
            {"schema": RECOVERY_STATE_SCHEMA, "status": "complete", **base, **tf_result},
        )
        families["causal_transformer"] = tf_result
    except Exception as error:
        if not failed_path.exists():
            _write_once(
                failed_path,
                {
                    **base,
                    "schema": RECOVERY_STATE_SCHEMA,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_epoch": time.time(),
                    "completed_families": families,
                },
            )
        raise
    completion = {
        **base,
        "schema": RECOVERY_STATE_SCHEMA,
        "status": "complete",
        "finished_epoch": time.time(),
        "checkpoint_endpoint_epoch": 1,
        "canonical_fixed_weights": {"TL3": 0.9, "candidate": 0.1},
        "all_family_grades_complete": all(
            item.get("grade_status") == "complete" for item in families.values()
        ),
        "families": families,
    }
    _write_once(complete_path, completion)
    return completion


def _controller_cli(args: argparse.Namespace) -> int:
    panel_path = Path(args.panel)
    panel, panel_hash = validate_panel(panel_path)
    admission_path = Path(args.recovery_admission)
    admission, admission_sha = _admission(admission_path, panel, panel_hash)
    expected_receipt = (
        _runtime(panel)
        / "receipts/continuation/STUDENT_CONTINUATION_RECOVERY_R2_CONTROLLER.json"
    )
    receipt_path = Path(args.controller_receipt)
    if receipt_path.resolve() != expected_receipt.resolve():
        raise ValueError("controller receipt path differs from the frozen R2 control binding")
    _check_free_floor(panel)
    run_recovery_controller(
        panel_path,
        panel,
        panel_hash,
        admission_path,
        admission_sha,
        admission["source_binding_sha256"],
        receipt_path,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--worker-mode", choices=("wide-recover-partial", "finalize-compact")
    )
    modes.add_argument("--controller-mode", choices=("continue-students",))
    parser.add_argument("--family", choices=("wide_gru", "causal_transformer"))
    parser.add_argument("--panel", required=True)
    parser.add_argument("--recovery-admission", required=True)
    parser.add_argument("--worker-receipt")
    parser.add_argument("--controller-receipt")
    args = parser.parse_args(argv)
    if args.worker_mode is not None:
        if args.family is None or args.worker_receipt is None or args.controller_receipt is not None:
            parser.error("worker mode requires --family and --worker-receipt only")
        return _worker_cli(args)
    if args.family is not None or args.controller_receipt is None or args.worker_receipt is not None:
        parser.error("controller mode requires --controller-receipt and no worker-only arguments")
    return _controller_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
