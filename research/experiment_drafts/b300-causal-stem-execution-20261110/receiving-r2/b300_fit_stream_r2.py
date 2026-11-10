"""Fit-only R2 controller: four independent sequential streams over approved GPU pairs."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

LOG = logging.getLogger("b300_fit_stream_r2")
FIT_SCHEMA = "wnn.b300-pr74-fourpairs-fit-summary.r2"


def _pair_stream(
    pair: str, cells: tuple[str, ...], run_cell: Callable[[str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Run exactly one GPU-pair stream serially, continuing to its next approved cell."""
    results: dict[str, dict[str, Any]] = {}
    for cell in cells:
        try:
            results[cell] = run_cell(cell)
        except Exception as error:  # noqa: BLE001 - record each stream's complete failure state.
            results[cell] = {
                "cell": cell,
                "pair_stream": pair,
                "status": "fit_controller_failed",
                "error": f"{type(error).__name__}: {error}",
                "automatic_resume": False,
                "automatic_retry": False,
            }
    return results


def _authorized_cell(
    cell: str,
    validate_cell: Callable[[str], Any],
    run_cell: Callable[[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Require a fresh per-cell validation result before invoking its fit callback."""
    admitted = validate_cell(cell)
    return run_cell(cell, admitted)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit-only R2 controller using frozen R1 source/profile gates."
    )
    parser.add_argument("--phase", choices=("fit",), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--clock", type=Path)
    parser.add_argument("--fit-admission", type=Path)
    parser.add_argument("--fit-admission-sha256", required=True)
    parser.add_argument("--r1-receiver", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    return parser


def _import_r1(receiver: Path) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import only the frozen R1 receiver helpers from its exact staged directory."""
    resolved = receiver.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("--r1-receiver must name the frozen R1 receiver directory")
    path = str(resolved)
    if path not in sys.path:
        sys.path.insert(0, path)
    r2_path = str(Path(__file__).resolve().parent)
    if r2_path not in sys.path:
        sys.path.insert(0, r2_path)
    import b300_driver_core_r1 as core
    import b300_driver_fits_r1 as fits
    import b300_driver_r1 as driver
    import b300_driver_support_r1 as support

    runner = importlib.import_module("b300_owned_runner_r2")

    support.run_owned = runner.run_owned
    return core, fits, driver, support, runner.ResourceGuard, runner, resolved


def _validate_admission(
    *,
    core: Any,
    admission_path: Path,
    expected_sha: str,
    authority_sha: str,
    source_sha: str,
    source: Path,
    configs: dict[str, dict[str, Any]],
    root: Path,
) -> dict[str, dict[str, Any]]:
    """Re-read and revalidate ROOT's immutable approval immediately before each cell."""
    actual_sha = core.sha256_file(admission_path)
    admission = core.read_json(admission_path, 256_000)
    return core.validate_fit_admission(
        admission,
        expected_sha=expected_sha,
        actual_sha=actual_sha,
        authority_sha=authority_sha,
        source_sha=source_sha,
        source_root=source,
        configs=configs,
        root=root,
    )


def _fit_phase(
    *,
    core: Any,
    fits: Any,
    support: Any,
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
    admission_path: Path,
    admission_sha: str,
    control: Path,
    owner: dict[str, int | str],
    state_path: Path,
    state_lock: threading.Lock,
    guard: Any,
) -> dict[str, Any]:
    """Run four independent pair streams, validating authority before every cell."""
    first, fit_deadline, wave_deadline = support._read_clock(clock_path)
    if time.time() >= fit_deadline:
        raise TimeoutError("original five-hour fit window has already closed")
    authority_sha = core.sha256_file(control / "authority.json")
    _validate_admission(
        core=core,
        admission_path=admission_path,
        expected_sha=admission_sha,
        authority_sha=authority_sha,
        source_sha=source_sha,
        source=source,
        configs=configs,
        root=root,
    )
    support._ensure_data_check(
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

    def validate_cell(cell: str) -> dict[str, Any]:
        return _validate_admission(
            core=core,
            admission_path=admission_path,
            expected_sha=admission_sha,
            authority_sha=authority_sha,
            source_sha=source_sha,
            source=source,
            configs=configs,
            root=root,
        )[cell]

    def run_cell(cell: str, admitted: dict[str, Any]) -> dict[str, Any]:
        return fits._fit_cell(
            cell=cell,
            admitted=admitted,
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
        )

    def authorized_cell(cell: str) -> dict[str, Any]:
        return _authorized_cell(cell, validate_cell, run_cell)

    outcomes: dict[str, dict[str, Any]] = {}
    streams = core.FIT_STREAMS
    original_acceptance = fits._run_acceptance

    def recheck_after_profile_acceptance(**kwargs: Any) -> None:
        """Revalidate admission after native profile acceptance and just before launch."""
        original_acceptance(**kwargs)
        if kwargs.get("mode") == "profiles":
            if time.time() >= fit_deadline:
                raise TimeoutError(
                    "original five-hour fit deadline reached before cell launch"
                )
            _validate_admission(
                core=core,
                admission_path=admission_path,
                expected_sha=admission_sha,
                authority_sha=authority_sha,
                source_sha=source_sha,
                source=source,
                configs=configs,
                root=root,
            )

    fits._run_acceptance = recheck_after_profile_acceptance
    with ThreadPoolExecutor(
        max_workers=len(streams), thread_name_prefix="fit-pair-r2"
    ) as pool:
        futures = {
            pool.submit(_pair_stream, pair, cells, authorized_cell): (pair, cells)
            for pair, cells in streams.items()
        }
        completed_pairs: list[str] = []
        for future in as_completed(futures):
            pair, cells = futures[future]
            try:
                outcomes.update(future.result())
            except Exception as error:  # noqa: BLE001 - capture an unexpected stream failure.
                for cell in cells:
                    outcomes.setdefault(
                        cell,
                        {
                            "cell": cell,
                            "pair_stream": pair,
                            "status": "fit_stream_failed",
                            "error": f"{type(error).__name__}: {error}",
                            "automatic_resume": False,
                            "automatic_retry": False,
                        },
                    )
            completed_pairs.append(pair)
            core.atomic_json(
                control / "fit-r2-stream-status.json",
                {
                    "schema": "wnn.b300-pr74-fourpairs-stream-status.r2",
                    "completed_pair_streams": sorted(completed_pairs),
                    "fits": outcomes,
                    "updated_epoch": time.time(),
                },
                replace=True,
            )
    complete = set(outcomes) == set(core.CELLS) and all(
        item.get("status") == "fit_complete" for item in outcomes.values()
    )
    summary = {
        "schema": FIT_SCHEMA,
        "status": "complete" if complete else "partial_or_failed",
        "source_sha256": source_sha,
        "source_commit": core.COMMIT,
        "authority_sha256": authority_sha,
        "fit_admission_sha256": admission_sha,
        "split_sha256": core.SPLIT_SHA256,
        "first_profile_epoch": first,
        "fit_deadline_epoch": fit_deadline,
        "wave_deadline_epoch": wave_deadline,
        "fit_pair_streams": {pair: list(cells) for pair, cells in streams.items()},
        "fits": outcomes,
        "automatic_resume": False,
        "automatic_retry": False,
        "interpretation": (
            "Fixed4E official-valid screen; ROOT-amended four-pair schedule. Each pair runs "
            "its assigned cells sequentially, independent of other pair completion."
        ),
    }
    core.atomic_json(control / "fit-r2-complete.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run an explicitly ROOT-admitted fit-only phase; profiles are never repeated."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = _parser()
    args = parser.parse_args(argv)
    if args.phase != "fit" or args.detach:
        parser.error(
            "R2 is fit-only; omit --detach and launch under ROOT's owned controller"
        )
    if not args.fit_admission_sha256:
        parser.error("--fit-admission-sha256 is required")
    core, fits, driver, support, ResourceGuard, _runner, _ = _import_r1(
        args.r1_receiver
    )
    (
        authority,
        configs,
        config_values,
        config_shas,
        _source_files,
        root,
        source,
        data_root,
        split,
        python,
    ) = driver._prepare(args)
    control = root / "control" / "b300-driver-r1"
    authority_sha = core.sha256_file(control / "authority.json")
    if authority_sha != args.authority_sha256:
        raise ValueError("staged R1 authority differs from the requested R2 authority")
    clock_path = support._clock_path(root, args.clock)
    if clock_path.as_posix() != Path(authority["clock_path"]).as_posix():
        raise ValueError("clock path differs from the frozen authority")
    admission_path = args.fit_admission or Path(authority["fit_admission_path"])
    if admission_path.as_posix() != Path(authority["fit_admission_path"]).as_posix():
        raise ValueError("fit-admission path differs from the frozen authority")
    import fcntl

    lock_stream = (control / "phase.lock").open("a+b")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(
            "another profile/fit phase controller owns the run root"
        ) from error
    try:
        phase_control = control / "fit-r2"
        phase_control.mkdir(parents=True, exist_ok=True)
        owner_path = phase_control / "owner.json"
        state_path = control / "fit-r2-status.json"
        if any(path.exists() or path.is_symlink() for path in (owner_path, state_path)):
            raise FileExistsError(
                "fit R2 already has an owner or state file; retries are disabled"
            )
        owner = support._identity(root)
        try:
            core.atomic_json(
                owner_path,
                {
                    "schema": "wnn.b300-pr74-fit-r2-owner.r1",
                    "owner": owner,
                    "source_sha256": core.SOURCE_SHA256,
                    "source_commit": core.COMMIT,
                    "authority_sha256": authority_sha,
                    "fit_admission_sha256": args.fit_admission_sha256,
                    "started_epoch": time.time(),
                },
            )
            core.atomic_json(
                state_path,
                {
                    "schema": "wnn.b300-pr74-fit-r2-status.r1",
                    "status": "running",
                    "owner": owner,
                    "cells": {},
                    "started_epoch": time.time(),
                },
            )
            guard = ResourceGuard(root, control, support._caps(authority))
            if core.directory_bytes(root) > support._caps(authority)["output_bytes"]:
                raise OSError(
                    "native output directory exceeds the authority output cap"
                )
            if (
                os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
                < support._caps(authority)["minimum_free_bytes"]
            ):
                raise OSError("native free space is below the authority reserve")
            result = _fit_phase(
                core=core,
                fits=fits,
                support=support,
                authority=authority,
                configs=configs,
                config_values=config_values,
                config_shas=config_shas,
                source_sha=core.SOURCE_SHA256,
                root=root,
                source=source,
                python=python,
                data_root=data_root,
                split=split,
                clock_path=clock_path,
                admission_path=admission_path,
                admission_sha=args.fit_admission_sha256,
                control=control,
                owner=owner,
                state_path=state_path,
                state_lock=threading.Lock(),
                guard=guard,
            )
        except Exception as error:
            error_value = f"{type(error).__name__}: {error}"
            completed = time.time()
            if state_path.is_file() and not state_path.is_symlink():
                state = core.read_json(state_path, 4_000_000)
                state.update(
                    status="failed", completed_epoch=completed, error=error_value
                )
                core.atomic_json(state_path, state, replace=True)
            if owner_path.is_file() and not owner_path.is_symlink():
                phase_owner = core.read_json(owner_path, 64_000)
                phase_owner.update(
                    status="failed", completed_epoch=completed, error=error_value
                )
                core.atomic_json(owner_path, phase_owner, replace=True)
            raise
        state = core.read_json(state_path, 4_000_000)
        state.update(
            status=result["status"], completed_epoch=time.time(), summary=result
        )
        core.atomic_json(state_path, state, replace=True)
        phase_owner = core.read_json(owner_path, 64_000)
        phase_owner.update(
            status=result["status"], completed_epoch=state["completed_epoch"]
        )
        core.atomic_json(owner_path, phase_owner, replace=True)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "summary": str(control / "fit-r2-complete.json"),
                }
            )
        )
        return 0 if result["status"] == "complete" else 2
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
