"""Run and validate the twelve PR74 production/peak engineering profiles."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any  # Native profile receipts contain recipe-specific JSON.

from b300_driver_core_r1 import (
    CELLS,
    COMMIT,
    PROFILE_WAVES,
    SPLIT_SHA256,
    atomic_json,
    canonical_sha256,
    sha256_file,
    validate_profile_receipt,
)
from b300_driver_support_r1 import (
    _create_clock_once,
    _ensure_data_check,
    _load_native_config,
    _peak_config,
    _run_acceptance,
    _run_stage,
)
from b300_owned_runner_r1 import OwnedRunError, ResourceGuard

MAX_PROFILE_SECONDS = 1800


def _profile_cell(
    *,
    cell: str,
    authority_config: dict[str, Any],
    production_config: dict[str, Any],
    config_sha: str,
    source_files: dict[str, str],
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
    """Run production then zero-warmup 50-update profiles on one fixed two-card pair."""
    run_dir = root / "runs" / cell
    production_config_path = Path(authority_config["path"])
    profile_output = run_dir / "profile"
    peak_output = run_dir / "profile-peak"
    if any(
        path.exists() or path.is_symlink() for path in (profile_output, peak_output)
    ):
        raise FileExistsError(
            f"profile output already exists for {cell}; retries are disabled"
        )
    gpu_uuids = [
        str(uuid).removeprefix("GPU-") for uuid in authority_config["gpu_uuids"]
    ]
    train_script = source / "scripts" / "train_sequence.py"
    base = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(train_script),
    ]
    common = [
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(split),
        "--profile-updates",
        "50",
    ]
    production_command = [
        *base,
        "--config",
        str(production_config_path),
        *common,
        "--output",
        str(profile_output),
    ]
    production_receipt = _run_stage(
        python=python,
        source=source,
        root=root,
        control=control,
        guard=guard,
        owner=owner,
        cell=cell,
        stage="profile-production",
        output=profile_output,
        command=production_command,
        gpu_uuids=gpu_uuids,
        deadline=fit_deadline,
        timeout=MAX_PROFILE_SECONDS,
        state_path=state_path,
        state_lock=state_lock,
    )
    if production_receipt.get("status") != "completed":
        raise OwnedRunError(f"production profile process failed for {cell}")
    production_evidence = validate_profile_receipt(
        profile_output / "profile_receipt.json",
        config=production_config,
        config_sha=config_sha,
        source_sha=source_sha,
        source_files=source_files,
        expected_split=SPLIT_SHA256,
        expected_uuids=gpu_uuids,
    )
    peak_config_path = run_dir / "profile-configs" / "peak.json"
    _peak_config(
        python=python,
        source=source,
        root=root,
        control=control,
        guard=guard,
        owner=owner,
        cell=cell,
        production_config=production_config_path,
        peak_config=peak_config_path,
        deadline=fit_deadline,
        state_path=state_path,
        state_lock=state_lock,
    )
    peak_config = _load_native_config(source, peak_config_path)
    expected_peak_config = dict(production_config, warmup_fraction=0.0)
    if peak_config != expected_peak_config:
        raise ValueError(f"normalized peak config changed more than warmup for {cell}")
    peak_receipt_config_sha = canonical_sha256(peak_config)
    peak_command = [
        *base,
        "--config",
        str(peak_config_path),
        *common,
        "--output",
        str(peak_output),
    ]
    peak_owned = _run_stage(
        python=python,
        source=source,
        root=root,
        control=control,
        guard=guard,
        owner=owner,
        cell=cell,
        stage="profile-zero-warmup-peak",
        output=peak_output,
        command=peak_command,
        gpu_uuids=gpu_uuids,
        deadline=fit_deadline,
        timeout=MAX_PROFILE_SECONDS,
        state_path=state_path,
        state_lock=state_lock,
    )
    if peak_owned.get("status") != "completed":
        raise OwnedRunError(f"zero-warmup profile process failed for {cell}")
    peak_evidence = validate_profile_receipt(
        peak_output / "profile_receipt.json",
        config=peak_config,
        config_sha=peak_receipt_config_sha,
        source_sha=source_sha,
        source_files=source_files,
        expected_split=SPLIT_SHA256,
        expected_uuids=gpu_uuids,
    )
    if (
        production_evidence["identity"]["runtime"]
        != peak_evidence["identity"]["runtime"]
    ):
        raise ValueError(
            f"production and peak profiles used different runtimes for {cell}"
        )
    result = {
        "schema": "wnn.b300-pr74-fourpairs-profile-cell.r1",
        "cell": cell,
        "status": "profiles_complete",
        "profile_updates_each": 50,
        "config_sha256": config_sha,
        "raw_config_sha256": authority_config["raw_sha256"],
        "pair": authority_config["pair"],
        "gpu_uuids": gpu_uuids,
        "production_receipt_sha256": sha256_file(
            profile_output / "profile_receipt.json"
        ),
        "peak_receipt_sha256": sha256_file(peak_output / "profile_receipt.json"),
        "peak_config_sha256": peak_receipt_config_sha,
        "production_runtime": production_evidence["identity"]["runtime"],
    }
    atomic_json(control / "profiles" / f"{cell}.json", result)
    return result


def _profile_phase(
    *,
    authority: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    config_values: dict[str, dict[str, Any]],
    config_shas: dict[str, str],
    source_files: dict[str, str],
    source_sha: str,
    root: Path,
    source: Path,
    python: Path,
    data_root: Path,
    split: Path,
    clock_path: Path,
    control: Path,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    state_path: Path,
    state_lock: threading.Lock,
) -> dict[str, Any]:
    """Run the exact 4-cell then 2-cell profile waves; never approve or start fits."""
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
    clock = _create_clock_once(clock_path)
    first, fit_deadline, wave_deadline = clock
    _run_acceptance(
        python=python,
        source=source,
        root=root,
        split=split,
        commit=COMMIT,
        control=control,
        guard=guard,
        owner=owner,
        cell="",
        mode="verify",
        deadline=fit_deadline,
        state_path=state_path,
        state_lock=state_lock,
    )
    wave_results: dict[str, dict[str, Any]] = {}
    for wave_index, cells in enumerate(PROFILE_WAVES, start=1):
        with ThreadPoolExecutor(
            max_workers=len(cells), thread_name_prefix=f"profile-wave-{wave_index}"
        ) as pool:
            futures = {
                pool.submit(
                    _profile_cell,
                    cell=cell,
                    authority_config=configs[cell],
                    production_config=config_values[cell],
                    config_sha=config_shas[cell],
                    source_files=source_files,
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
                    wave_results[cell] = future.result()
                except Exception as error:  # noqa: BLE001 - record every profile controller failure.
                    wave_results[cell] = {
                        "schema": "wnn.b300-pr74-fourpairs-profile-cell.r1",
                        "cell": cell,
                        "status": "profile_failed",
                        "error": f"{type(error).__name__}: {error}",
                        "automatic_retry": False,
                    }
        atomic_json(
            control / "profile-wave-status.json",
            {
                "schema": "wnn.b300-pr74-fourpairs-profile-wave.r1",
                "wave": wave_index,
                "cells": {cell: wave_results[cell] for cell in cells},
                "updated_epoch": time.time(),
            },
            replace=True,
        )
    successful = {
        cell: result
        for cell, result in wave_results.items()
        if result.get("status") == "profiles_complete"
    }
    complete = set(successful) == set(CELLS)
    summary = {
        "schema": "wnn.b300-pr74-fourpairs-profiles-summary.r1",
        "status": "profiles_complete" if complete else "profiles_partial_or_failed",
        "authority_sha256": sha256_file(control / "authority.json"),
        "source_sha256": source_sha,
        "source_commit": COMMIT,
        "split_sha256": SPLIT_SHA256,
        "first_profile_epoch": first,
        "fit_deadline_epoch": fit_deadline,
        "wave_deadline_epoch": wave_deadline,
        "cells": wave_results,
        "scientific_fits_started": 0,
        "automatic_retry": False,
        "root_manual_profile_approval_required": True,
    }
    atomic_json(control / "profiles-complete.json", summary)
    return summary
