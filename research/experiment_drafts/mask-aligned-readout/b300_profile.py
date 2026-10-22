"""Bounded per-fold B300 profile gate for the PR36 receiver."""

from __future__ import annotations

# Any represents trainer JSON records validated against the profile contract.
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from receiver_assets import (
    REPO_ROOT,
    Cell,
    ReceiverAssets,
    file_sha256,
    load_json,
    output_root,
    recipe_cells,
    select_cells,
)
from receiver_lock import claim_receiver_stage

PACKET = Path(__file__).resolve().parent
RECIPE_PATH = PACKET / "recipe.json"
PROFILE_CONFIG = load_json(RECIPE_PATH)["profile"]
PROFILE_BATCHES = int(PROFILE_CONFIG["measured_steps"])
PROFILE_WARMUP = int(PROFILE_CONFIG["warmup_steps"])


def profile_receipt_path(root: Path, cell: Cell) -> Path:
    """Return the immutable receipt path for one fold's B300 shape profile."""
    return output_root(root) / "profiles" / f"{cell.fold}.json"


def profile_run_name(cell: Cell) -> str:
    """Return the R2 profile run name bound to the selected fit endpoint."""
    return f"mask_readout_r2_e{cell.epochs}_ep{cell.checkpoint_epoch}_profile_{cell.fold.lower()}"


def _write_receipt(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"profile receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record(stdout: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records: dict[str, Any] = {}
    for line in stdout.splitlines():
        prefix, separator, payload = line.partition(" ")
        if separator and prefix in (
            "PROFILE",
            "PROFILE_UPDATE_EVIDENCE",
            "PROFILE_EXECUTION_RUNTIME",
        ):
            records[prefix] = json.loads(payload)
    metrics, evidence = records.get("PROFILE"), records.get("PROFILE_UPDATE_EVIDENCE")
    runtime = records.get("PROFILE_EXECUTION_RUNTIME")
    if (
        not isinstance(metrics, dict)
        or not isinstance(evidence, dict)
        or not isinstance(runtime, dict)
    ):
        raise ValueError(
            "canonical trainer did not emit profile, update and process-runtime records"
        )
    if not evidence.get("valid_for_throughput"):
        raise ValueError("canonical profile failed finite loss/gradient/update validation")
    attempts = int(evidence["optimizer_attempts"])
    if (
        attempts <= 0
        or evidence.get("optimizer_completed_updates") != attempts
        or evidence.get("amp_skipped_updates") != 0
        or evidence.get("gradient_finite") is not True
        or evidence.get("gradient_stats_steps") != attempts
        or evidence.get("loss_checks") != attempts
    ):
        raise ValueError("canonical profile lacks complete finite FP32 update evidence")
    for key in ("step_wall", "seq_per_s"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"canonical profile lacks a positive finite {key}")
    for key in (
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "cuda_free_memory_bytes",
        "cuda_total_memory_bytes",
    ):
        if not isinstance(metrics.get(key), int) or metrics[key] <= 0:
            raise ValueError(f"canonical profile lacks positive integer {key}")
    if (
        metrics["cuda_peak_reserved_bytes"]
        > PROFILE_CONFIG["maximum_peak_reserved_fraction"] * metrics["cuda_total_memory_bytes"]
        or metrics["cuda_free_memory_bytes"]
        < PROFILE_CONFIG["minimum_free_memory_fraction"] * metrics["cuda_total_memory_bytes"]
    ):
        raise ValueError("profile leaves less than ten percent of B300 memory headroom")
    for stage in ("process_start", "process_end"):
        sample = runtime.get(stage)
        if (
            not isinstance(sample, dict)
            or "b300" not in str(sample.get("device", "")).lower()
            or not sample.get("device_uuid")
            or not sample.get("driver_version")
            or sample.get("precision") != "fp32"
            or sample.get("matmul_tf32") is not False
            or sample.get("cudnn_tf32") is not False
            or sample.get("autocast_active_at_capture") is not False
        ):
            raise ValueError(f"profile {stage} lacks source-process strict-FP32 B300 identity")
    if any(
        runtime["process_start"].get(key) != runtime["process_end"].get(key)
        for key in ("device_uuid", "driver_version", "device", "torch_cuda_version")
    ):
        raise ValueError("profile process changed B300 hardware or CUDA driver during profiling")
    return metrics, evidence, runtime


def profile_command(cell: Cell, split_path: Path, profile_name: str) -> list[str]:
    """Build a treatment-shaped canonical profile without saving a checkpoint."""
    from receiver import fit_command

    command = fit_command(cell, "is_scored_weighted", split_path)
    command[command.index("--name") + 1] = profile_name
    command.extend(
        (
            "--profile-steps",
            str(PROFILE_BATCHES),
            "--profile-warmup",
            str(PROFILE_WARMUP),
            "--grad-stats",
        )
    )
    return command


def _require_receipt(assets: ReceiverAssets, cell: Cell) -> dict[str, Any]:
    path = profile_receipt_path(assets.root, cell)
    if not path.is_file():
        raise FileNotFoundError(f"required B300 profile receipt is missing: {path}")
    receipt = load_json(path)
    recipe = load_json(RECIPE_PATH)
    recipe_cell = next(item for item in recipe["cells"] if item["fold"] == cell.fold)
    expected = {
        "fold": cell.fold,
        "parent_run": cell.parent_run,
        "parent_checkpoint_sha256": file_sha256(assets.parent_checkpoints[cell.fold]),
        "parent_provenance_sha256": file_sha256(assets.parent_provenance[cell.fold]),
        "split_manifest_sha256": recipe_cell["split_manifest_sha256"],
        "recipe_sha256": file_sha256(RECIPE_PATH),
        "profile_batches": PROFILE_BATCHES,
        "profile_warmup": PROFILE_WARMUP,
        "minimum_free_memory_fraction": PROFILE_CONFIG["minimum_free_memory_fraction"],
        "maximum_peak_reserved_fraction": PROFILE_CONFIG["maximum_peak_reserved_fraction"],
        "batch_size": 32,
        "chunk_rows": 2000,
        "device": "cuda",
        "mask_arm": "is_scored_weighted",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{cell.fold} profile receipt does not match its pinned receiver cell")
    expected_command = profile_command(cell, assets.split_paths[cell.fold], profile_run_name(cell))
    if receipt.get("command") != expected_command:
        raise ValueError(f"{cell.fold} profile was not produced by the canonical bounded command")
    if not receipt.get("profile_evidence", {}).get("valid_for_throughput"):
        raise ValueError(f"{cell.fold} profile receipt is not a valid finite-update profile")
    evidence = receipt["profile_evidence"]
    if (
        evidence.get("optimizer_attempts") != receipt.get("profile_optimizer_attempts")
        or evidence.get("optimizer_completed_updates") != evidence.get("optimizer_attempts")
        or evidence.get("amp_skipped_updates") != 0
        or evidence.get("gradient_finite") is not True
    ):
        raise ValueError(f"{cell.fold} profile does not prove complete finite optimizer updates")
    metrics = receipt.get("canonical_metrics", {})
    if any(
        not isinstance(metrics.get(key), (int, float))
        or not math.isfinite(metrics[key])
        or metrics[key] <= 0
        for key in ("step_wall", "seq_per_s")
    ):
        raise ValueError(f"{cell.fold} profile timing evidence is nonfinite or incomplete")
    if any(
        not isinstance(metrics.get(key), int) or metrics[key] <= 0
        for key in (
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "cuda_free_memory_bytes",
            "cuda_total_memory_bytes",
        )
    ):
        raise ValueError(f"{cell.fold} profile memory evidence is incomplete")
    if (
        metrics.get("cuda_peak_reserved_bytes", 0)
        > PROFILE_CONFIG["maximum_peak_reserved_fraction"]
        * metrics.get("cuda_total_memory_bytes", 0)
        or metrics.get("cuda_free_memory_bytes", 0)
        < PROFILE_CONFIG["minimum_free_memory_fraction"] * metrics.get("cuda_total_memory_bytes", 0)
        or receipt.get("minimum_free_memory_fraction")
        != PROFILE_CONFIG["minimum_free_memory_fraction"]
        or receipt.get("maximum_peak_reserved_fraction")
        != PROFILE_CONFIG["maximum_peak_reserved_fraction"]
    ):
        raise ValueError(f"{cell.fold} B300 profile lacks ten percent memory headroom")
    if not receipt.get("trainer_implementation", {}).get("sha256"):
        raise ValueError(f"{cell.fold} profile receipt lacks its trainer source identity")
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.source_identity import training_source_identity

    current_source = training_source_identity(
        TrainingEnvironment(source_root=REPO_ROOT, root=assets.root)
    )
    if receipt["trainer_implementation"] != current_source:
        raise ValueError(f"{cell.fold} profile was made by a different trainer implementation")
    hardware = receipt.get("hardware_before", {})
    hardware_after = receipt.get("hardware_after", {})
    if "b300" not in str(hardware.get("device", "")).lower() or hardware.get(
        "device"
    ) != hardware_after.get("device"):
        raise ValueError(f"{cell.fold} profile receipt was not made on B300")
    from audit import require_same_device

    runtime = receipt.get("execution_runtime", {})
    require_same_device(runtime.get("process_start", {}), hardware)
    require_same_device(runtime.get("process_end", {}), hardware_after)
    return {"path": str(path), "sha256": file_sha256(path), **receipt}


def require_profile_receipts(assets: ReceiverAssets, cells: tuple[Cell, ...]) -> dict[str, Any]:
    """Fail closed unless the selected cell workers have passed B300 profiles."""
    return {cell.fold: _require_receipt(assets, cell) for cell in cells}


def run_profiles(assets: ReceiverAssets, fold: str | None = None) -> None:
    """Run bounded treatment-shaped profiles for selected paired-fit workers."""
    from audit import require_b300_runtime, require_same_device
    from receiver import command_environment, ensure_unwritten
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.source_identity import training_source_identity

    recipe = load_json(RECIPE_PATH)
    cells = select_cells(recipe_cells(recipe), fold)
    environment = command_environment(assets.root)
    source_identity = training_source_identity(
        TrainingEnvironment(source_root=REPO_ROOT, root=assets.root)
    )
    for cell in cells:
        name = profile_run_name(cell)
        run_dir = assets.root / "artifacts" / "runs" / name
        receipt_path = profile_receipt_path(assets.root, cell)
        with claim_receiver_stage(assets.root, "profile", cell.fold):
            ensure_unwritten([receipt_path, run_dir])
            hardware_before = require_b300_runtime()
            command = profile_command(cell, assets.split_paths[cell.fold], name)
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            metrics, evidence, execution_runtime = _record(completed.stdout)
            hardware_after = require_b300_runtime()
            require_same_device(execution_runtime["process_start"], hardware_before)
            require_same_device(execution_runtime["process_end"], hardware_after)
            if not run_dir.is_dir() or any(run_dir.iterdir()):
                raise ValueError(f"profile {cell.fold} unexpectedly wrote run artifacts: {run_dir}")
            run_dir.rmdir()
            recipe_cell = next(item for item in recipe["cells"] if item["fold"] == cell.fold)
            record = {
                "schema_version": 1,
                "stage": "bounded_profile_only_no_checkpoint_or_score",
                "fold": cell.fold,
                "parent_run": cell.parent_run,
                "parent_checkpoint_sha256": file_sha256(assets.parent_checkpoints[cell.fold]),
                "parent_provenance_sha256": file_sha256(assets.parent_provenance[cell.fold]),
                "split_manifest_sha256": recipe_cell["split_manifest_sha256"],
                "recipe_sha256": file_sha256(RECIPE_PATH),
                "trainer_implementation": source_identity,
                "device": "cuda",
                "mask_arm": "is_scored_weighted",
                "batch_size": 32,
                "chunk_rows": 2000,
                "profile_batches": PROFILE_BATCHES,
                "profile_warmup": PROFILE_WARMUP,
                "minimum_free_memory_fraction": PROFILE_CONFIG["minimum_free_memory_fraction"],
                "maximum_peak_reserved_fraction": PROFILE_CONFIG["maximum_peak_reserved_fraction"],
                "profile_optimizer_attempts": evidence["optimizer_attempts"],
                "canonical_metrics": metrics,
                "profile_evidence": evidence,
                "execution_runtime": execution_runtime,
                "hardware_before": hardware_before,
                "hardware_after": hardware_after,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
                "command": command,
            }
            _write_receipt(receipt_path, record)
