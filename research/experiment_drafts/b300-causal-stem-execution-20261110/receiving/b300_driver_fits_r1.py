"""Run and validate the six fresh PR74 fixed4E fits after ROOT admission."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any  # Trainer metadata and receipts are JSON boundary documents.

from b300_driver_core_r1 import (
    CELLS,
    COMMIT,
    FIT_WAVES,
    SCORED_ROWS,
    SPLIT_SHA256,
    TRAIN_MEMBERS,
    UPDATES_PER_EPOCH,
    VALID_MEMBERS,
    atomic_json,
    canonical_sha256,
    read_json,
    sha256_file,
    validate_fit_admission,
)
from b300_driver_support_r1 import (
    _ensure_data_check,
    _read_clock,
    _run_acceptance,
    _run_stage,
)
from b300_owned_runner_r1 import ResourceGuard

FIT_RESULT_SCHEMA = "wnn.b300-pr74-fourpairs-fit-result.r1"


def _fit_metrics(
    root: Path,
    cell: str,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
    uuids: list[str],
) -> dict[str, Any]:
    """Verify a zero-exit 4E endpoint, all epoch full-valid events and train-only ancestry."""
    run_dir = root / "runs" / cell / "fit"
    terminal = read_json(run_dir / "terminal_status.json", 2_000_000)
    metadata = read_json(run_dir / "run_metadata.json", 1_000_000)
    model = run_dir / "model.pt"
    if model.is_symlink() or not model.is_file():
        raise ValueError("fixed endpoint model.pt is missing or linked")
    model_sha = sha256_file(model)
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != 4
        or terminal.get("global_updates") != 3_280
    ):
        raise ValueError("terminal status is not the configured fixed 4E endpoint")
    if terminal.get("model_sha256") != model_sha:
        raise ValueError("terminal model hash differs from the actual final checkpoint")
    actual_config = metadata.get("config")
    source = metadata.get("source")
    provenance = metadata.get("provenance")
    if (
        not isinstance(actual_config, dict)
        or canonical_sha256(actual_config) != config_sha
    ):
        raise ValueError("run metadata config differs from the profile-admitted bytes")
    if not isinstance(source, dict) or source.get("sha256") != source_sha:
        raise ValueError("run metadata source identity differs from the frozen capsule")
    if not isinstance(provenance, dict) or provenance.get("world_size") != 2:
        raise ValueError("fit provenance does not record world size two")
    data = provenance.get("data")
    if not isinstance(data, dict) or provenance.get("data_sha256") != canonical_sha256(
        data
    ):
        raise ValueError("fit data provenance digest is missing or invalid")
    fit_members, eval_members = data.get("fit_members"), data.get("evaluation_members")
    if (
        data.get("split_sha256") != SPLIT_SHA256
        or not isinstance(fit_members, list)
        or len(fit_members) != TRAIN_MEMBERS
        or any(
            not isinstance(member, list) or len(member) != 2 or member[0] != "train"
            for member in fit_members
        )
        or not isinstance(eval_members, list)
        or len(eval_members) != VALID_MEMBERS
        or any(
            not isinstance(member, list) or len(member) != 2 or member[0] != "valid"
            for member in eval_members
        )
        or data.get("training_mask") != "all_after_warmup"
        or not isinstance(data.get("normalization"), dict)
        or data["normalization"].get("scope") != "all_fit"
        or data["normalization"].get("sequence_count") != TRAIN_MEMBERS
        or not isinstance(data["normalization"].get("identity_sha256"), str)
        or len(data["normalization"]["identity_sha256"]) != 64
        or not isinstance(data["normalization"].get("moments_sha256"), str)
        or len(data["normalization"]["moments_sha256"]) != 64
        or data.get("teacher") is not None
        or data.get("reference") is not None
    ):
        raise ValueError(
            "fit memberships, normalization or ancestry differ from the official train-only split"
        )
    runtime = provenance.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, dict) else None
    precision = runtime.get("precision") if isinstance(runtime, dict) else None
    if (
        not isinstance(devices, list)
        or [str(item.get("uuid", "")).removeprefix("GPU-") for item in devices] != uuids
        or not isinstance(precision, dict)
        or precision.get("mode") != "bf16_bf16x9"
    ):
        raise ValueError("fit runtime differs from the assigned B300 pair/precision")
    terminal_identity = terminal.get("identity")
    if not isinstance(terminal_identity, dict) or any(
        terminal_identity.get(key) != value
        for key, value in {
            "source_sha256": source_sha,
            "config_sha256": config_sha,
            "data_sha256": provenance["data_sha256"],
            "world_size": 2,
        }.items()
    ):
        raise ValueError(
            "terminal identity is not bound to this source/config/data/world size"
        )
    event_path = run_dir / "monitor_evaluations.jsonl"
    if (
        event_path.is_symlink()
        or not event_path.is_file()
        or event_path.stat().st_size > 4_000_000
    ):
        raise ValueError("monitor evaluation journal is missing or oversized")
    raw = event_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("monitor journal ends with a partial JSON line")
    events = [json.loads(line) for line in raw.splitlines()]
    if len(events) != 4:
        raise ValueError("fixed 4E endpoint lacks four full-valid monitor events")
    metrics = []
    for epoch, event in enumerate(events, start=1):
        values = event.get("metrics")
        context = event.get("context")
        if (
            event.get("schema") != "wnn.monitor.v1"
            or event.get("kind") != "evaluation"
            or type(event.get("epoch")) is not int
            or event.get("epoch") != epoch
            or type(event.get("step")) is not int
            or event.get("step") != epoch * UPDATES_PER_EPOCH
            or not isinstance(values, dict)
            or values.get("valid/sequence_count") != VALID_MEMBERS
            or values.get("valid/selected_rows") != SCORED_ROWS
            or any(
                not isinstance(values.get(name), (int, float))
                or isinstance(values.get(name), bool)
                or not math.isfinite(float(values[name]))
                for name in ("valid/wp", "valid/wp/t0", "valid/wp/t1")
            )
            or not isinstance(context, dict)
            or context.get("split_sha256") != SPLIT_SHA256
            or context.get("evaluation_member_count") != VALID_MEMBERS
            or context.get("evaluation_population")
            != "complete official validation membership"
            or context.get("evaluation_mask") != "is_scored after warmup"
            or context.get("metric_id") != "global_wp_clip2_v1"
            or context.get("endpoint") != "raw standalone z; no reference or blend"
            or context.get("checkpoint_selection")
            != "fixed final epoch 4; monitoring is report only"
            or context.get("evaluation_exposure")
            != "development exposed; official source boundary"
        ):
            raise ValueError(f"E{epoch} full-valid monitor event is incomplete")
        metrics.append(values)
    return {
        "model_sha256": model_sha,
        "terminal_sha256": sha256_file(run_dir / "terminal_status.json"),
        "metadata_sha256": sha256_file(run_dir / "run_metadata.json"),
        "monitor_sha256": sha256_file(event_path),
        "data_sha256": provenance["data_sha256"],
        "final_metrics": metrics[-1],
        "epoch_metrics": metrics,
    }


def _fit_cell(
    *,
    cell: str,
    admitted: dict[str, Any],
    authority_config: dict[str, Any],
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
    root: Path,
    source: Path,
    python: Path,
    data_root: Path,
    split: Path,
    control: Path,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    fit_deadline: float,
    state_path: Path,
    state_lock: threading.Lock,
) -> dict[str, Any]:
    """Run one fresh fixed4E canonical DDP fit after the profile receipt gate passes."""
    result_path = control / "fits" / f"{cell}.json"
    run_dir = root / "runs" / cell / "fit"
    try:
        if (
            result_path.exists()
            or result_path.is_symlink()
            or run_dir.exists()
            or run_dir.is_symlink()
        ):
            raise FileExistsError(
                "fit output/receipt already exists; resume and automatic retries are disabled"
            )
        _run_acceptance(
            python=python,
            source=source,
            root=root,
            split=split,
            commit=COMMIT,
            control=control,
            guard=guard,
            owner=owner,
            cell=cell,
            mode="profiles",
            deadline=fit_deadline,
            state_path=state_path,
            state_lock=state_lock,
        )
        uuids = [
            str(uuid).removeprefix("GPU-") for uuid in authority_config["gpu_uuids"]
        ]
        config_path = Path(authority_config["path"])
        approved_path = root / "runs" / cell / "approved_profile.json"
        command = [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(source / "scripts" / "train_sequence.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(split),
            "--output",
            str(run_dir),
            "--profile-receipt",
            str(approved_path),
        ]
        owned = _run_stage(
            python=python,
            source=source,
            root=root,
            control=control,
            guard=guard,
            owner=owner,
            cell=cell,
            stage="fit",
            output=run_dir,
            command=command,
            gpu_uuids=uuids,
            deadline=fit_deadline,
            timeout=None,
            state_path=state_path,
            state_lock=state_lock,
        )
        evidence = _fit_metrics(root, cell, config, config_sha, source_sha, uuids)
        files = []
        for folder, dirs, names in os.walk(run_dir, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(folder) / name).is_symlink()]
            for name in names:
                path = Path(folder) / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        "fit output contains a symlink or non-regular artifact"
                    )
                files.append(
                    {
                        "path": path.relative_to(run_dir).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        result = {
            "schema": FIT_RESULT_SCHEMA,
            "cell": cell,
            "status": "fit_complete",
            "source_sha256": source_sha,
            "source_commit": COMMIT,
            "authority_sha256": sha256_file(control / "authority.json"),
            "fit_admission_sha256": sha256_file(control / "ROOT-fit-admission-r1.json"),
            "config_sha256": config_sha,
            "raw_config_sha256": authority_config["raw_sha256"],
            "pair": authority_config["pair"],
            "gpu_uuids": uuids,
            "fixed_epochs": 4,
            "global_updates": 3_280,
            "fit_members": TRAIN_MEMBERS,
            "evaluation_members": VALID_MEMBERS,
            "scored_rows": SCORED_ROWS,
            "owned_fit": owned,
            "model_sha256": evidence["model_sha256"],
            "terminal_sha256": evidence["terminal_sha256"],
            "metadata_sha256": evidence["metadata_sha256"],
            "monitor_sha256": evidence["monitor_sha256"],
            "data_sha256": evidence["data_sha256"],
            "final_metrics": evidence["final_metrics"],
            "epoch_metrics": evidence["epoch_metrics"],
            "artifacts": files,
            "completed_at_epoch": time.time(),
            "automatic_resume": False,
        }
    except Exception as error:  # noqa: BLE001 - a fit failure must be durably receipted.
        result = {
            "schema": FIT_RESULT_SCHEMA,
            "cell": cell,
            "status": "fit_failed_or_partial",
            "error": f"{type(error).__name__}: {error}",
            "source_sha256": source_sha,
            "config_sha256": config_sha,
            "automatic_resume": False,
            "automatic_retry": False,
            "observed_at_epoch": time.time(),
        }
    atomic_json(result_path, result)
    return result


def _fit_phase(
    *,
    authority: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    config_values: dict[str, dict[str, Any]],
    config_shas: dict[str, str],
    source_sha: str,
    root: Path,
    source: Path,
    python: Path,
    data_root: Path,
    split: Path,
    clock_path: Path,
    control: Path,
    fit_admission_path: Path,
    fit_admission_sha: str,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    state_path: Path,
    state_lock: threading.Lock,
) -> dict[str, Any]:
    """Verify ROOT's post-profile SHA admission and run six fresh fits in the amended pair waves."""
    clock = _read_clock(clock_path)
    first, fit_deadline, wave_deadline = clock
    if time.time() >= fit_deadline:
        raise TimeoutError("original five-hour fit window has already closed")
    actual_admission_sha = sha256_file(fit_admission_path)
    admission = read_json(fit_admission_path, 256_000)
    fit_admission = validate_fit_admission(
        admission,
        expected_sha=fit_admission_sha,
        actual_sha=actual_admission_sha,
        authority_sha=sha256_file(control / "authority.json"),
        source_sha=source_sha,
        source_root=source,
        configs=configs,
        root=root,
    )
    _ensure_data_check(
        root=root,
        source=source,
        python=python,
        data_root=data_root,
        split=split,
        control=control,
        guard=guard,
        owner=owner,
        state_path=state_path,
        state_lock=state_lock,
    )
    outcomes: dict[str, dict[str, Any]] = {}
    for wave_index, cells in enumerate(FIT_WAVES, start=1):
        with ThreadPoolExecutor(
            max_workers=len(cells), thread_name_prefix=f"fit-wave-{wave_index}"
        ) as pool:
            futures = {
                pool.submit(
                    _fit_cell,
                    cell=cell,
                    admitted=fit_admission[cell],
                    authority_config=configs[cell],
                    config=config_values[cell],
                    config_sha=config_shas[cell],
                    source_sha=source_sha,
                    root=root,
                    source=source,
                    python=python,
                    data_root=data_root,
                    split=split,
                    control=control,
                    guard=guard,
                    owner=owner,
                    fit_deadline=fit_deadline,
                    state_path=state_path,
                    state_lock=state_lock,
                ): cell
                for cell in cells
            }
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    outcomes[cell] = future.result()
                except Exception as error:  # noqa: BLE001 - record every child controller failure.
                    outcomes[cell] = {
                        "schema": FIT_RESULT_SCHEMA,
                        "cell": cell,
                        "status": "fit_controller_failed",
                        "error": f"{type(error).__name__}: {error}",
                        "automatic_resume": False,
                        "automatic_retry": False,
                    }
        atomic_json(
            control / "fit-wave-status.json",
            {
                "schema": "wnn.b300-pr74-fourpairs-fit-wave.r1",
                "wave": wave_index,
                "cells": {cell: outcomes[cell] for cell in cells},
                "updated_epoch": time.time(),
            },
            replace=True,
        )
    complete = set(outcomes) == set(CELLS) and all(
        value.get("status") == "fit_complete" for value in outcomes.values()
    )
    summary = {
        "schema": "wnn.b300-pr74-fourpairs-fit-summary.r1",
        "status": "complete" if complete else "partial_or_failed",
        "source_sha256": source_sha,
        "source_commit": COMMIT,
        "authority_sha256": sha256_file(control / "authority.json"),
        "fit_admission_sha256": fit_admission_sha,
        "split_sha256": SPLIT_SHA256,
        "first_profile_epoch": first,
        "fit_deadline_epoch": fit_deadline,
        "wave_deadline_epoch": wave_deadline,
        "fits": outcomes,
        "automatic_resume": False,
        "automatic_retry": False,
        "interpretation": "Fixed4E exposed official-valid screen under the ROOT-amended four-pair schedule; pair allocation differs from the original PR74 per-seed stream.",
    }
    atomic_json(control / "fit-complete.json", summary)
    return summary
