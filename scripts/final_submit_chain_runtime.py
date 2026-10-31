"""Bounded worker and artifact helpers for the final-submit finite DAG."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import final_submit_chain_contract as contract
import numpy as np
import torch

from wnn.final_submit_manifest import FinalFitMembership
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import base_environment, hold_gpu_slot, process_identity
from wnn.training.final_submit_bridge import (
    validate_final_fit_parent,
    validate_final_fit_provenance,
)

MAX_NODE_SECONDS = 43_200


def ensure_source_indexes(data_root: Path, membership: FinalFitMembership) -> None:
    """Create or verify row-group-only source indexes; never place target labels in them."""
    cache = data_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, groups in (("train", membership.train_groups), ("valid", membership.valid_groups)):
        path = cache / f"{name}_targets.npz"
        expected = np.asarray(groups, dtype=np.int64)
        if not path.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name}-targets-", suffix=".npz", dir=cache
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    np.savez(stream, row_group=expected)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe final-fit row index: {path}")
        with np.load(path, allow_pickle=False) as content:
            if set(content.files) != {"row_group"}:
                raise ValueError(f"final-fit source index must contain only row_group: {path}")
            actual = np.asarray(content["row_group"])
        if (
            actual.ndim != 1
            or actual.dtype.kind not in "iu"
            or not np.array_equal(actual, expected)
        ):
            raise ValueError(f"final-fit source index order differs: {path}")


def gpu_is_idle_b300(gpu_uuid: str) -> None:
    """Require the selected device to be a mostly idle B300 before each GPU stage."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    matches = [
        line.split(", ") for line in result.stdout.splitlines() if line.startswith(gpu_uuid + ",")
    ]
    if len(matches) != 1 or len(matches[0]) != 4:
        raise RuntimeError(f"assigned GPU UUID is not uniquely visible: {gpu_uuid}")
    _, name, memory_used, utilization = matches[0]
    if "B300" not in name.upper() or int(memory_used) > 2048 or int(utilization) > 20:
        raise RuntimeError(
            f"assigned GPU is not an idle B300: {gpu_uuid}, {name}, {memory_used} MiB"
        )


def run_stage(
    *,
    argv: Sequence[str],
    source_root: Path,
    data_root: Path,
    controller_root: Path,
    gpu_uuid: str,
    gpu_visible: str,
    stage: int,
    trial: int,
    timeout: int,
    output_dir: Path,
    receipt: Path,
    log: Path,
    progress: Callable[[dict[str, object]], None],
    use_gpu: bool = True,
) -> dict[str, object]:
    """Run one bounded owned process with an isolated session, receipt, and GPU lock."""
    if receipt.exists() or log.exists():
        raise FileExistsError(f"identified stage already has a receipt or log: {receipt}")
    if timeout <= 10 or timeout > MAX_NODE_SECONDS:
        raise ValueError("stage timeout is outside the frozen node wall cap")
    env = base_environment(gpu_visible if use_gpu else "", source_root)
    env["WNN_PROJECT_ROOT"] = str(data_root)
    env["PYTHONUNBUFFERED"] = "1"
    last_write = 0.0

    def update_progress() -> None:
        nonlocal last_write
        now = time.monotonic()
        if receipt.is_file() and (last_write == 0 or now - last_write >= 5):
            value = contract.read_json(receipt)
            progress(value)
            last_write = now

    slot = hold_gpu_slot(controller_root, gpu_uuid) if use_gpu else nullcontext()
    with slot:
        if use_gpu:
            gpu_is_idle_b300(gpu_uuid)
        remaining = int(contract.HARD_DEADLINE - contract.COLLECTION_RESERVE - time.time())
        effective_timeout = min(timeout, remaining - 10)
        if effective_timeout <= 10:
            raise TimeoutError("the collection reserve leaves no time for another owned stage")
        command = list(argv)
        if (
            len(command) < 5
            or command[:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
            or not command[3].endswith("s")
        ):
            raise ValueError("owned stages require the frozen nested GNU timeout wrapper")
        inner_seconds = int(command[3][:-1])
        if inner_seconds > effective_timeout - 10:
            command[3] = f"{effective_timeout - 10}s"
        owner = process_identity(os.getpid())
        if owner is None:
            raise RuntimeError("controller process identity is unavailable")
        code, timed_out, record = run_owned_command(
            command,
            cwd=source_root,
            env=env,
            log_path=log,
            receipt_path=receipt,
            owner={**owner, "controller_root": str(controller_root)},
            stage=stage,
            trial_number=trial,
            mode="final_submit_finite_chain",
            gpu_key=gpu_uuid,
            output_dir=output_dir,
            timeout_seconds=effective_timeout,
            progress=update_progress,
        )
    progress(record)
    if timed_out or code != 0:
        raise RuntimeError(
            f"owned stage failed: exit={code}, timed_out={timed_out}, receipt={receipt}"
        )
    return record


def validate_fit_checkpoint(
    *,
    node: Mapping[str, Any],
    output_dir: Path,
    membership: FinalFitMembership,
) -> Path:
    """Require a complete fixed endpoint with final-fit provenance before exporting."""
    run = str(node["run"])
    if node.get("node_id") == "source4_lstm192x1":
        checkpoint_path = output_dir / "model.pt"
        terminal_path = output_dir / "terminal_status.json"
        if not checkpoint_path.is_file() or not terminal_path.is_file():
            raise FileNotFoundError("native LSTM fixed-endpoint artifacts are incomplete")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        terminal = contract.read_json(terminal_path)
        data_provenance = checkpoint.get("data")
        if not isinstance(data_provenance, Mapping):
            raise ValueError("native checkpoint omits its final-fit data provenance")
        validate_final_fit_provenance(data_provenance, membership)
        fit_identity = data_provenance.get("final_fit")
        if (
            checkpoint.get("checkpoint_schema") != "wnn.sequence-fit.v1"
            or checkpoint.get("epoch") != 7
            or checkpoint.get("completed_epochs") != 8
            or fit_identity is None
            or fit_identity.get("manifest_sha256") != membership.manifest_sha256
            or fit_identity.get("fit_members_sha256") != membership.fit_members_sha256
            or fit_identity.get("source_sha256")
            != {"train": membership.train_sha256, "valid": membership.valid_sha256}
            or fit_identity.get("source_validation_sha256") != membership.source_validation_sha256
            or fit_identity.get("independent_score_claim") != "forbidden"
            or data_provenance.get("evaluation_members") != []
            or terminal.get("status") != "completed"
            or terminal.get("completed_epochs") != 8
            or terminal.get("model_sha256") != contract.sha256_file(checkpoint_path)
        ):
            raise ValueError(
                "native LSTM checkpoint is partial or has different final-fit identity"
            )
        return checkpoint_path
    epoch = int(node["endpoint"])
    checkpoint_path = output_dir / f"model_ep{epoch}.pt"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"fixed TL3 checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != epoch
        or not isinstance(checkpoint.get("args"), Mapping)
        or checkpoint["args"].get("name") != run
        or checkpoint["args"].get("epochs") != node.get("epochs")
    ):
        raise ValueError(f"TL3 checkpoint is partial or misnamed: {run}")
    validate_final_fit_parent(checkpoint, membership)
    return checkpoint_path


def validate_export(data_root: Path, run: str, split: str, membership: FinalFitMembership) -> Path:
    """Check one feature-only export has source order and carries no score semantics."""
    sidecar_path = data_root / "cache" / f"pred_{run}_{split}.json"
    prediction_path = sidecar_path.with_suffix(".npy")
    sidecar = contract.read_json(sidecar_path)
    groups = membership.train_groups if split == "train" else membership.valid_groups
    if (
        sidecar.get("run") != run
        or sidecar.get("split") != split
        or sidecar.get("row_groups") != list(groups)
        or sidecar.get("final_fit_manifest_sha256") != membership.manifest_sha256
        or sidecar.get("final_fit_members_sha256") != membership.fit_members_sha256
        or sidecar.get("source_validation_sha256") != membership.source_validation_sha256
        or sidecar.get("feature_only") is not True
        or sidecar.get("campaign_export_scope") != "final-fit-source-index-feature-only"
        or sidecar.get("independent_score_claim") != "forbidden"
        or sidecar.get("prediction_role") != "fitting-only teacher features; no independent score"
        or contract.sha256_file(prediction_path) != sidecar.get("prediction_sha256")
    ):
        raise ValueError(f"feature-only export identity differs: {run}/{split}")
    return prediction_path


def cache_argv(
    *,
    python: Path,
    source_root: Path,
    data_root: Path,
    membership_path: Path,
    audit_path: Path,
    output: Path,
) -> list[str]:
    """Build the exact six-source homogeneous-strong W18 cache command."""
    argv = [
        str(python),
        str(source_root / "scripts/build_distill_cache.py"),
        "--data-root",
        str(data_root),
        "--teacher-cache-root",
        str(data_root / "cache"),
        "--checkpoint-root",
        str(data_root / "artifacts/runs"),
        "--final-fit-manifest",
        str(membership_path),
        "--final-fit-source-validation-receipt",
        str(audit_path),
        "--final-fit-source-validation-sha256",
        contract.SOURCE_AUDIT_SHA256,
        "--output",
        str(output),
    ]
    for run, family, coefficient, epoch in contract.W18_TEACHERS:
        argv.extend(("--teacher", f"{family}={run}@ep{epoch}*{coefficient}"))
    argv.extend(
        (
            "--homogeneous-family",
            "strong",
            "--expected-coefficient-sum",
            "0.8125",
            "--output-profile",
            "student-required",
        )
    )
    return argv
