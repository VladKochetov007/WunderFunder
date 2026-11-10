"""Shared preparation, clock, progress and canonical-stage helpers."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any  # JSON progress and authority documents are boundary data.

from b300_driver_core_r1 import (
    SPLIT_SHA256,
    TRAIN_MEMBERS,
    VALID_MEMBERS,
    atomic_json,
    parse_proc_stat,
    read_json,
    validate_clock,
)
from b300_owned_runner_r1 import OwnedRunError, ResourceGuard, run_owned

LOG = logging.getLogger("b300_causal_stem_receiver")
PACKET = Path("research/experiment_drafts/transformer-causal-input-stem-20261109")


def _absolute_executable(path: Path) -> Path:
    """Make a lexical absolute interpreter path without resolving a virtualenv symlink."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    if not candidate.is_file() or (
        os.name == "posix" and not os.access(candidate, os.X_OK)
    ):
        raise FileNotFoundError(
            f"interpreter is missing or not executable: {candidate}"
        )
    return candidate


def _load_native_config(source: Path, path: Path) -> dict[str, Any]:
    """Load with the exact verified PR74 config parser used by the trainer."""
    source_path = str(source / "src")
    sys_path_added = source_path not in sys.path
    if sys_path_added:
        sys.path.insert(0, source_path)
    old_emulation = os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION")
    os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    try:
        module = importlib.import_module("wnn.training.sequence_config")
        loaded = module.load_config(path)
        if not isinstance(loaded, dict):
            raise TypeError("canonical config loader returned a non-object")
        return loaded
    finally:
        if old_emulation is None:
            os.environ.pop("CUBLAS_EMULATE_SINGLE_PRECISION", None)
        else:
            os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"] = old_emulation
        if sys_path_added:
            sys.path.remove(source_path)


def _environment(source: Path, root: Path, gpu_uuids: list[str]) -> dict[str, str]:
    """Create the frozen PR74 runtime environment for a single two-rank pair."""
    env = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PYTHONPATH",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUBLAS_EMULATE_SINGLE_PRECISION",
    ):
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(source / "src"),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        NUMEXPR_NUM_THREADS="8",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUBLAS_EMULATE_SINGLE_PRECISION="1",
        NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(root / "scratch"),
        TORCHINDUCTOR_CACHE_DIR=str(root / "compiler" / "inductor"),
        TRITON_CACHE_DIR=str(root / "compiler" / "triton"),
        CUDA_VISIBLE_DEVICES=",".join(
            f"GPU-{uuid.removeprefix('GPU-')}" for uuid in gpu_uuids
        ),
    )
    if not gpu_uuids:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _caps(authority: dict[str, Any]) -> dict[str, int]:
    """Return resource limits from the exact authority after its contract was checked."""
    budget = authority["budget"]
    return {
        "output_bytes": int(budget["output_bytes"]),
        "minimum_free_bytes": int(budget["minimum_free_bytes"]),
        "owned_rss_bytes": int(budget["owned_rss_bytes"]),
    }


def _clock_path(root: Path, supplied: Path | None) -> Path:
    return supplied or root / "receipts" / "first-profile-r1.json"


def _read_clock(path: Path) -> tuple[float, float, float]:
    return validate_clock(read_json(path, 16_000))


def _create_clock_once(path: Path) -> tuple[float, float, float]:
    """Create the original clock only if ROOT has not already written it; never reset it."""
    if path.exists() or path.is_symlink():
        return _read_clock(path)
    now = time.time()
    value = {
        "first_profile_epoch": now,
        "fit_deadline_epoch": now + 18_000,
        "wave_deadline_epoch": now + 21_600,
    }
    try:
        atomic_json(path, value)
    except FileExistsError:
        LOG.info(
            "ROOT or another authorized profile worker created the first-profile clock"
        )
    return _read_clock(path)


def _progress_callback(
    state_path: Path, state_lock: threading.Lock, cell: str, stage: str, output: Path
) -> Any:
    """Build an owned-worker heartbeat callback that reads the canonical monitor progress file."""

    def update() -> None:
        progress: dict[str, Any] = {}
        path = output / "monitor_progress.json"
        if path.is_file() and not path.is_symlink():
            try:
                progress = read_json(path, 32_000)
            except (OSError, ValueError, json.JSONDecodeError):
                progress = {"progress_read_error": True}
        with state_lock:
            state = read_json(state_path, 4_000_000)
            state["updated_epoch"] = time.time()
            cells = state.setdefault("cells", {})
            cells[cell] = {
                **(cells.get(cell, {}) if isinstance(cells.get(cell), dict) else {}),
                "phase": stage,
                "last_progress": progress,
                "heartbeat_epoch": time.time(),
            }
            atomic_json(state_path, state, replace=True)

    return update


def _run_stage(
    *,
    python: Path,
    source: Path,
    root: Path,
    control: Path,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    cell: str,
    stage: str,
    output: Path,
    command: list[str],
    gpu_uuids: list[str],
    deadline: float,
    timeout: int | None,
    state_path: Path,
    state_lock: threading.Lock,
) -> dict[str, object]:
    env = _environment(source, root, gpu_uuids)
    progress = _progress_callback(state_path, state_lock, cell, stage, output)
    return run_owned(
        command,
        root=source,
        control=control,
        cell=cell,
        stage=stage,
        gpu_uuids=gpu_uuids,
        owner=owner,
        environment=env,
        deadline=deadline,
        stage_timeout=timeout,
        guard=guard,
        progress=progress,
    )


def _identity(root: Path) -> dict[str, int | str]:
    """Capture the driver process PID and Linux birth ticks for ownership receipts."""
    stat = Path("/proc") / str(os.getpid()) / "stat"
    current = parse_proc_stat(os.getpid(), stat.read_text(), os.sysconf("SC_PAGE_SIZE"))
    return {
        "pid": current.pid,
        "birth_ticks": current.start_ticks,
        "host": os.uname().nodename,
        "root": str(root),
    }


def _write_owner(path: Path, value: dict[str, Any]) -> None:
    atomic_json(path, value)


def _ensure_data_check(
    *,
    root: Path,
    source: Path,
    python: Path,
    data_root: Path,
    split: Path,
    control: Path,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    state_path: Path,
    state_lock: threading.Lock,
) -> None:
    """Run or verify PR74's canonical source/split check before any profile GPU work."""
    output = root / "data-check.json"
    if not (output.exists() or output.is_symlink()):
        command = [
            str(python),
            str(source / PACKET / "receiver_data.py"),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(split),
            "--output",
            str(output),
        ]
        _run_stage(
            python=python,
            source=source,
            root=root,
            control=control,
            guard=guard,
            owner=owner,
            cell="preflight",
            stage="data-check",
            output=root,
            command=command,
            gpu_uuids=[],
            deadline=time.time() + 1_800,
            timeout=1_800,
            state_path=state_path,
            state_lock=state_lock,
        )
    value = read_json(output, 32_000)
    if (
        value.get("schema") != "wnn.input-stem-data-check.v1"
        or value.get("split_sha256") != SPLIT_SHA256
        or value.get("fit_train") != TRAIN_MEMBERS
        or value.get("fit_valid") != 0
        or value.get("evaluation_valid") != VALID_MEMBERS
        or value.get("component_disjoint") is not True
        or value.get("new_model_executions") != 0
    ):
        raise ValueError(
            "canonical data check does not match train-only PR74 membership"
        )


def _run_acceptance(
    *,
    python: Path,
    source: Path,
    root: Path,
    split: Path,
    commit: str,
    control: Path,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    cell: str,
    mode: str,
    deadline: float,
    state_path: Path,
    state_lock: threading.Lock,
) -> None:
    packet = source / PACKET
    command = [
        str(python),
        str(packet / "acceptance.py"),
        "--mode",
        mode,
        "--root",
        str(root),
        "--commit",
        commit,
        "--split",
        str(split),
    ]
    if mode == "profiles":
        command.extend(("--cell", cell))
    receipt = _run_stage(
        python=python,
        source=source,
        root=root,
        control=control,
        guard=guard,
        owner=owner,
        cell=cell or "preflight",
        stage=f"acceptance-{mode}",
        output=root,
        command=command,
        gpu_uuids=[],
        deadline=deadline,
        timeout=900,
        state_path=state_path,
        state_lock=state_lock,
    )
    if receipt.get("status") != "completed":
        raise OwnedRunError(f"canonical acceptance {mode} failed")


def _peak_config(
    *,
    python: Path,
    source: Path,
    root: Path,
    control: Path,
    guard: ResourceGuard,
    owner: dict[str, int | str],
    cell: str,
    production_config: Path,
    peak_config: Path,
    deadline: float,
    state_path: Path,
    state_lock: threading.Lock,
) -> dict[str, Any]:
    """Use the pinned PR74 adapter to change only warmup for the peak-LR profile."""
    peak_config.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(source / PACKET / "profile_config.py"),
        "--config",
        str(production_config),
        "--output",
        str(peak_config),
    ]
    _run_stage(
        python=python,
        source=source,
        root=root,
        control=control,
        guard=guard,
        owner=owner,
        cell=cell,
        stage="make-peak-config",
        output=peak_config.parent,
        command=command,
        gpu_uuids=[],
        deadline=deadline,
        timeout=300,
        state_path=state_path,
        state_lock=state_lock,
    )
    production = read_json(production_config, 64_000)
    peak = read_json(peak_config, 64_000)
    expected = dict(production, warmup_fraction=0.0)
    if peak != expected:
        raise ValueError(
            f"peak profile config changed more than warmup_fraction for {cell}"
        )
    return peak
