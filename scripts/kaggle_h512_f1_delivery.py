from __future__ import annotations

import gc
import hashlib
import os
import struct
import time
from pathlib import Path
from typing import Any

from kaggle_h512_f1_postfit import derive_swa345, validate_fit
from kaggle_h512_f1_worker_support import (
    EXPECTED,
    FINALIZER_RESERVE_SECONDS,
    FIT_NAME,
    OUTER_WALL_CAP_SECONDS,
    RECEIPT_PATH,
    SWA_NAME,
    TOTAL_CHUNK_UPDATES,
    WORKER_WALL_CAP_SECONDS,
    emit,
    run_cli,
    sha256,
)


def collect_export(
    root: Path,
    run_name: str,
    checkpoint_path: Path,
    expected_row_groups: list[int],
) -> dict[str, Any]:
    """Bind one FP32 feature-only export to its checkpoint and ordered F1 groups."""
    import json

    prediction = root / "cache" / f"pred_{run_name}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    checkpoint_sha = sha256(checkpoint_path)
    if metadata.get("run") != run_name or metadata.get("epoch") != 5:
        raise ValueError(f"Prediction sidecar run or epoch differs: {run_name}")
    if metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"Prediction checkpoint identity differs: {run_name}")
    if metadata.get("campaign_split_manifest_sha256") != EXPECTED["derived_manifest_sha256"]:
        raise ValueError(f"Prediction manifest identity differs: {run_name}")
    if (
        metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("feature_only") is not True
    ):
        raise ValueError(f"Prediction is not evaluation-only feature export: {run_name}")
    groups = metadata.get("row_groups")
    if not isinstance(groups, list) or list(map(int, groups)) != expected_row_groups:
        raise ValueError(f"Prediction row-group order differs from the F1 manifest: {run_name}")
    group_bytes = b"".join(struct.pack("<q", int(group)) for group in groups)
    return {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_path": prediction.relative_to(root).as_posix(),
        "prediction_size_bytes": prediction.stat().st_size,
        "prediction_sha256": sha256(prediction),
        "sidecar_path": sidecar.relative_to(root).as_posix(),
        "sidecar_sha256": sha256(sidecar),
        "campaign_export_scope": metadata["campaign_export_scope"],
        "feature_only": metadata["feature_only"],
        "campaign_split_manifest_sha256": metadata["campaign_split_manifest_sha256"],
        "row_group_count": len(groups),
        "ordered_row_groups_sha256": hashlib.sha256(group_bytes).hexdigest(),
    }


def _runtime_tf32_flags(torch: Any) -> dict[str, bool]:
    return {
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def fit_and_export(root: Path, started: float, state: dict[str, Any]) -> dict[str, Any]:
    """Run the admitted fresh fit, derive SWA345, and export its fixed endpoints."""
    gate = state["gate"]
    if (
        state["predicted_worker_exceeds_cap"]
        or state["predicted_outer_exceeds_cap"]
        or state["predicted_outer_exceeds_quota"]
        or state["predicted_outer_exceeds_kernel_deadline"]
    ):
        return {"status": "PROFILE_COMPLETE_FIT_SKIPPED_COST_GATE"}
    admission = state["admission"]
    if not admission.get("conditional_fit_admitted"):
        return {"status": "PROFILE_COMPLETE_FIT_NOT_ADMITTED"}

    import torch

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE=0 is required for strict FP32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before_fit_flags = _runtime_tf32_flags(torch)
    if any(before_fit_flags.values()):
        raise RuntimeError("TF32 flags must be false immediately before the full fit")
    gc.collect()
    torch.cuda.empty_cache()
    if any(
        (root / name).stat().st_mtime_ns != observed
        for name, observed in state["raw_mount_stats"].items()
    ):
        raise RuntimeError("Raw source metadata changed before full fit")
    emit(
        "fresh_full_fit_started",
        started,
        argv=state["fit_argv"],
        model_optimizer_rng="new canonical run; seed reset to 1004",
        expected_chunk_updates=TOTAL_CHUNK_UPDATES,
        tf32_flags=before_fit_flags,
    )
    fit_elapsed, _ = run_cli(root, state["fit_argv"])
    fit_audit = validate_fit(
        root,
        EXPECTED["derived_manifest_sha256"],
        list(map(int, state["split"].train_groups)),
        list(map(int, state["split"].valid_groups)),
        state["evaluation_order"],
    )
    if time.monotonic() - started + 2_700 > WORKER_WALL_CAP_SECONDS:
        return {
            "status": "FIT_COMPLETE_EXPORT_SKIPPED_WALL_RESERVE",
            "fit_elapsed_seconds": fit_elapsed,
            "fit_audit": fit_audit,
            "gate": gate,
            "device": state["device"],
        }
    proposal = state["proposal"]
    swa = derive_swa345(root, fit_audit)
    export_commands = proposal["recipe"]["evaluation_export_argv"]
    exports = []
    for index, (command, endpoint) in enumerate(
        (
            (export_commands, FIT_NAME),
            (proposal["recipe"]["swa345_evaluation_export_argv"], SWA_NAME),
        )
    ):
        reserve = 1_200 if index == 1 else 2_700
        deadline = float(os.environ["WNN_KERNEL_DEADLINE_MONOTONIC"])
        deadline_remaining = deadline - time.monotonic()
        if (
            time.monotonic() - started + reserve + FINALIZER_RESERVE_SECONDS
            >= OUTER_WALL_CAP_SECONDS
            or reserve + FINALIZER_RESERVE_SECONDS >= deadline_remaining
        ):
            return {
                "status": (
                    "FIT_COMPLETE_CP5_ONLY_EXPORT_SKIPPED_WALL_RESERVE"
                    if exports
                    else "FIT_COMPLETE_EXPORT_SKIPPED_WALL_RESERVE"
                ),
                "fit_elapsed_seconds": fit_elapsed,
                "fit_audit": fit_audit,
                "swa345": swa,
                "exports": exports,
                "gate": gate,
                "device": state["device"],
            }
        export_name = command[command.index("--output-name") + 1]
        if export_name != endpoint:
            raise ValueError("Prediction export name differs from the frozen grader endpoint")
        emit("feature_only_export_started", started, command=command, output_name=export_name)
        run_cli(root, list(command))
        flags = _runtime_tf32_flags(torch)
        if any(flags.values()):
            raise RuntimeError("Prediction export enabled TF32 after strict-FP32 setup")
        checkpoint_path = root / "artifacts/runs" / endpoint / "model_ep5.pt"
        exports.append(collect_export(root, endpoint, checkpoint_path, state["evaluation_order"]))
    receipt = {
        "status": "CONDITIONAL_FIT_AND_EXPORT_COMPLETE",
        "run_names": {"cp5": FIT_NAME, "swa345": SWA_NAME},
        "input_identity": state["identities"],
        "profile": state["records"],
        "gate": gate,
        "device": state["device"],
        "preflight_seconds": state["preflight_seconds"],
        "fit_elapsed_seconds": fit_elapsed,
        "fit_audit": fit_audit,
        "swa345": swa,
        "exports": exports,
        "precision_audit": {
            "nvidia_tf32_override": os.environ["NVIDIA_TF32_OVERRIDE"],
            "flags_after_profile": state["tf32_flags_after_profile"],
            "flags_before_full_fit": before_fit_flags,
            "flags_after_full_fit": fit_audit["runtime_tf32_flags_after_fit"],
            "flags_after_exports": _runtime_tf32_flags(torch),
        },
        "quality_fit_claim": None,
    }
    if RECEIPT_PATH.exists():
        raise FileExistsError("Refusing to overwrite worker receipt")
    return receipt
