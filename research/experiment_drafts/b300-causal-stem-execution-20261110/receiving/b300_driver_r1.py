"""ROOT-authorized profile/fit receiver for the exact PR74 B300 causal-stem source."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any  # Parsed ROOT/source receipts are JSON boundary documents.

from b300_driver_core_r1 import (
    COMMIT,
    SOURCE_SHA256,
    SPLIT_SHA256,
    atomic_json,
    directory_bytes,
    read_json,
    sha256_file,
    source_identity_files,
    validate_authority,
    validate_config,
    verify_source_capsule,
)
from b300_driver_fits_r1 import _fit_phase
from b300_driver_profiles_r1 import _profile_phase
from b300_driver_support_r1 import (
    _absolute_executable,
    _caps,
    _clock_path,
    _identity,
    _load_native_config,
    _write_owner,
)
from b300_owned_runner_r1 import ResourceGuard

LOG = logging.getLogger("b300_causal_stem_receiver")
STATE_SCHEMA = "wnn.b300-pr74-fourpairs-driver-state.r1"


def _prepare(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, str],
    Path,
    Path,
    Path,
    Path,
    Path,
]:
    """Bind authority, source capsule, configs and original data paths before work begins."""
    if os.name != "posix":
        raise RuntimeError(
            "receiving controller must run on the ROOT-owned Linux B300 host"
        )
    root = args.root.resolve(strict=True)
    source = (args.source or root / "source-r1").resolve(strict=True)
    if source != root / "source-r1":
        raise ValueError("source must be the authority's isolated source-r1 capsule")
    authority_sha = sha256_file(args.authority)
    if authority_sha != args.authority_sha256:
        raise ValueError("authority bytes differ from required --authority-sha256")
    authority = read_json(args.authority, 256_000)
    configs = validate_authority(
        authority,
        root=root,
        manifest_sha=args.manifest_sha256,
        commit=args.source_commit,
    )
    source_sha, files = verify_source_capsule(
        source,
        args.manifest or source / "source-manifest.json",
        args.manifest_sha256,
        args.source_commit,
    )
    if (
        source_sha != authority["source"]["source_sha256"]
        or source_sha != SOURCE_SHA256
    ):
        raise ValueError(
            "derived trainer source identity differs from frozen authority"
        )
    config_values: dict[str, dict[str, Any]] = {}
    config_shas: dict[str, str] = {}
    for cell, config_authority in configs.items():
        config_path = Path(config_authority["path"])
        if not config_path.resolve(strict=True).is_relative_to(source):
            raise ValueError(
                f"config path is outside the immutable source capsule for {cell}"
            )
        normalized = _load_native_config(source, config_path)
        config, config_sha = validate_config(
            config_path, config_authority, cell, normalized
        )
        config_authority["config_sha256"] = config_sha
        config_values[cell], config_shas[cell] = config, config_sha
    data_root = args.data_root or Path(authority["data_root"])
    split = args.split or Path(authority["split"])
    if str(data_root) != authority["data_root"] or str(split) != authority["split"]:
        raise ValueError("data-root/split overrides differ from the frozen authority")
    if sha256_file(split) != SPLIT_SHA256:
        raise ValueError(
            "official chronological split bytes differ from the pinned SHA"
        )
    python = _absolute_executable(args.python or Path(authority["python"]))
    return (
        authority,
        configs,
        config_values,
        config_shas,
        source_identity_files(files),
        root,
        source,
        data_root,
        split,
        python,
    )


def _detach(args: argparse.Namespace) -> dict[str, Any]:
    """Start one detached phase controller and write its lexical PID/birth receipt."""
    root = args.root.resolve(strict=True)
    control = root / "control" / "b300-driver-r1" / "detached"
    control.mkdir(parents=True, exist_ok=True)
    owner_path = control / f"{args.phase}-owner.json"
    if owner_path.exists() or owner_path.is_symlink():
        raise FileExistsError("detached phase already has an owner receipt")
    log_dir = control / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{args.phase}.log"
    if log_path.exists():
        raise FileExistsError("detached phase log already exists")
    child_args = [arg for arg in sys.argv[1:] if arg != "--detach"]
    command = [sys.executable, "-u", os.path.abspath(__file__), *child_args]
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    for _ in range(50):
        from b300_driver_core_r1 import proc_entry

        child = proc_entry(process.pid)
        if child is not None:
            break
        time.sleep(0.02)
    if child is None:
        raise RuntimeError(
            "detached controller exited before its process birth could be recorded"
        )
    value = {
        "schema": "wnn.b300-pr74-detached-owner.r1",
        "phase": args.phase,
        "controller": {"pid": child.pid, "birth_ticks": child.start_ticks},
        "command": command,
        "log_path": str(log_path),
        "created_epoch": time.time(),
    }
    atomic_json(owner_path, value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("profile", "fit"), required=True)
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
    parser.add_argument("--fit-admission-sha256")
    parser.add_argument("--detach", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one explicit profile or fit phase; never resume/retry or mutate source."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = _parser()
    args = parser.parse_args(argv)
    if args.phase == "fit" and not args.fit_admission_sha256:
        parser.error("--fit-admission-sha256 is required for --phase fit")
    if args.detach:
        print(json.dumps(_detach(args), sort_keys=True))
        return 0
    (
        authority,
        configs,
        config_values,
        config_shas,
        source_files,
        root,
        source,
        data_root,
        split,
        python,
    ) = _prepare(args)
    authority_path = root / "control" / "b300-driver-r1" / "authority.json"
    control = authority_path.parent
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "phase.lock"
    lock_stream = lock_path.open("a+b")
    import fcntl

    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(
            "another profile/fit phase controller owns the run root"
        ) from error
    if authority_path.exists() or authority_path.is_symlink():
        if sha256_file(authority_path) != args.authority_sha256:
            raise ValueError("controller's existing authority receipt differs")
    else:
        authority_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.authority, authority_path)
    clock_path = _clock_path(root, args.clock)
    if clock_path.as_posix() != Path(authority["clock_path"]).as_posix():
        raise ValueError("clock path differs from the frozen authority")
    phase_owner_path = control / args.phase / "owner.json"
    phase_owner_path.parent.mkdir(parents=True, exist_ok=True)
    owner = _identity(root)
    _write_owner(
        phase_owner_path,
        {
            "schema": "wnn.b300-pr74-phase-owner.r1",
            "phase": args.phase,
            "owner": owner,
            "authority_sha256": args.authority_sha256,
            "source_sha256": SOURCE_SHA256,
            "source_commit": COMMIT,
            "manifest_sha256": args.manifest_sha256,
            "started_epoch": time.time(),
            "python_path_lexical": str(python),
        },
    )
    state_path = control / "status.json"
    atomic_json(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "phase": args.phase,
            "status": "running",
            "owner": owner,
            "authority_sha256": args.authority_sha256,
            "source_sha256": SOURCE_SHA256,
            "cells": {},
            "started_epoch": time.time(),
        },
        replace=True,
    )
    state_lock = threading.Lock()
    guard = ResourceGuard(root, control, _caps(authority))
    if directory_bytes(root) > _caps(authority)["output_bytes"]:
        raise OSError(
            "native output directory already exceeds the 100 GB authority cap"
        )
    if shutil.disk_usage(root).free < _caps(authority)["minimum_free_bytes"]:
        raise OSError("native disk free space is below the 100 GB authority reserve")
    try:
        if args.phase == "profile":
            result = _profile_phase(
                authority=authority,
                configs=configs,
                config_values=config_values,
                config_shas=config_shas,
                source_files=source_files,
                source_sha=SOURCE_SHA256,
                root=root,
                source=source,
                python=python,
                data_root=data_root,
                split=split,
                clock_path=clock_path,
                control=control,
                guard=guard,
                owner=owner,
                state_path=state_path,
                state_lock=state_lock,
            )
        else:
            if args.fit_admission is None:
                args.fit_admission = Path(authority["fit_admission_path"])
            if str(args.fit_admission) != authority["fit_admission_path"]:
                raise ValueError("fit-admission path differs from the frozen authority")
            result = _fit_phase(
                authority=authority,
                configs=configs,
                config_values=config_values,
                config_shas=config_shas,
                source_sha=SOURCE_SHA256,
                root=root,
                source=source,
                python=python,
                data_root=data_root,
                split=split,
                clock_path=clock_path,
                control=control,
                fit_admission_path=args.fit_admission,
                fit_admission_sha=args.fit_admission_sha256,
                guard=guard,
                owner=owner,
                state_path=state_path,
                state_lock=state_lock,
            )
        state = read_json(state_path, 4_000_000)
        state["status"] = result["status"]
        state["completed_epoch"] = time.time()
        state["summary"] = result
        atomic_json(state_path, state, replace=True)
        phase_owner = read_json(phase_owner_path, 64_000)
        phase_owner.update(
            completed_epoch=state["completed_epoch"], status=result["status"]
        )
        atomic_json(phase_owner_path, phase_owner, replace=True)
    except Exception as error:
        state = read_json(state_path, 4_000_000)
        state.update(
            status="failed",
            completed_epoch=time.time(),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(state_path, state, replace=True)
        atomic_json(
            phase_owner_path,
            {
                "schema": "wnn.b300-pr74-phase-owner.r1",
                "phase": args.phase,
                "owner": owner,
                "authority_sha256": args.authority_sha256,
                "source_sha256": SOURCE_SHA256,
                "source_commit": COMMIT,
                "manifest_sha256": args.manifest_sha256,
                "started_epoch": state["started_epoch"],
                "completed_epoch": state["completed_epoch"],
                "status": "failed",
                "error": state["error"],
                "python_path_lexical": str(python),
            },
            replace=True,
        )
        raise
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "status": result["status"],
                "summary_path": str(
                    control
                    / (
                        "profiles-complete.json"
                        if args.phase == "profile"
                        else "fit-complete.json"
                    )
                ),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if result["status"] in ("profiles_complete", "complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
