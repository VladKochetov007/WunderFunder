from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import LaunchError, atomic_write_json, process_identity, sha256_file, utc_now
from .contract import CellPlan, load_panel_cells


def receipt_cells(output_root: Path, admitted: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert validated preflight records into durable cell entries before spawn."""
    result: dict[str, dict[str, Any]] = {}
    for cell_id, record in admitted["cells"].items():
        result[cell_id] = {
            key: value for key, value in record.items() if key != "identity"
        } | {
            "identity_sha256": record["identity_sha256"],
            "output": str(output_root / "runs" / cell_id),
            "log": str(output_root / "logs" / f"{cell_id}.log"),
            "state": "not_started",
        }
    return result


def start_fits(
    receipt_path: Path, receipt: dict[str, Any], admitted: dict[str, Any]
) -> dict[str, subprocess.Popen[bytes]]:
    """Start the validated single-device jobs with detached sessions and pinned CPUs."""
    output_root = Path(receipt["output_root"])
    (output_root / "runs").mkdir(mode=0o700)
    (output_root / "logs").mkdir(mode=0o700)
    plans = _plans_from_admitted(receipt, admitted)
    children: dict[str, subprocess.Popen[bytes]] = {}
    for cell in plans:
        Path(receipt["cells"][cell.cell_id]["output"]).mkdir(mode=0o700)
    for cell in plans:
        record = receipt["cells"][cell.cell_id]
        run_output = Path(record["output"])
        log_path = Path(record["log"])
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        log_file = os.fdopen(descriptor, "ab", buffering=0)
        env = fit_environment(Path(receipt["repo_root"]), cell, record)
        argv = [
            str(receipt["python"]),
            str(Path(receipt["repo_root"]) / "scripts" / "train_sequence.py"),
            "--config",
            str(cell.config_path),
            "--data-root",
            str(receipt["data_root"]),
            "--split-manifest",
            str(receipt["split_manifest"]),
            "--output",
            str(run_output),
            "--profile-receipt",
            str(cell.profile_path),
        ]
        try:
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                preexec_fn=affinity_setter(cell.cpus),
                env=env,
            )
        except Exception as error:
            record["state"] = "not_started"
            record["start_error"] = f"{type(error).__name__}: {error}"
            receipt["status"] = "launch_failed"
            atomic_write_json(receipt_path, receipt)
            break
        finally:
            log_file.close()
        children[cell.cell_id] = child
        try:
            identity = _wait_process_identity(child.pid)
        except LaunchError as error:
            record.update(
                {
                    "pid": child.pid,
                    "start_ticks": None,
                    "process_group": child.pid,
                    "state": "ownership_failed",
                    "start_error": str(error),
                }
            )
            receipt["status"] = "launch_failed"
            atomic_write_json(receipt_path, receipt)
            break
        if identity["process_group"] != child.pid:
            record.update(
                {
                    "pid": child.pid,
                    "start_ticks": identity["start_ticks"],
                    "process_group": identity["process_group"],
                    "state": "ownership_failed",
                    "start_error": "detached fit did not create its own process group",
                }
            )
            receipt["status"] = "launch_failed"
            atomic_write_json(receipt_path, receipt)
            break
        record.update(
            {
                "pid": child.pid,
                "start_ticks": identity["start_ticks"],
                "process_group": identity["process_group"],
                "argv": argv,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                    "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                    "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
                    "CUBLAS_WORKSPACE_CONFIG": env["CUBLAS_WORKSPACE_CONFIG"],
                },
                "state": "running",
                "started_at": utc_now(),
            }
        )
        receipt["status"] = "launching"
        atomic_write_json(receipt_path, receipt)
    receipt["status"] = "running" if len(children) == 8 else "launch_failed"
    if len(children) != 8:
        receipt["error"] = "one or more fixed panel fits could not be started"
    receipt["fits_started_at"] = utc_now()
    atomic_write_json(receipt_path, receipt)
    return children


def fit_environment(repo_root: Path, cell: CellPlan, record: dict[str, Any]) -> dict[str, str]:
    """Set the per-fit UUID, eight-CPU thread budget, and profile-bound cuBLAS state."""
    env = base_environment(repo_root)
    precision = cell.profile["identity"]["runtime"]["precision"]
    emulation = precision["cublas_emulate_single_precision"]
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(record["gpu_uuid"]),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "ARROW_NUM_THREADS": "8",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    if emulation is None:
        env.pop("CUBLAS_EMULATE_SINGLE_PRECISION", None)
    else:
        env["CUBLAS_EMULATE_SINGLE_PRECISION"] = str(emulation)
    return env


def base_environment(repo_root: Path) -> dict[str, str]:
    """Build a small environment with the isolated Python path and no inherited ranks."""
    allowed = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "CUDA_HOME",
        "CUDA_PATH",
        "TORCH_HOME",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(repo_root / "src"), str(repo_root / "scripts"))
            ),
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return env


def assert_reserved_root_fresh(output_root: Path) -> None:
    """Require the supervisor reservation to contain only its receipt and log."""
    permitted = {"launch_receipt.json", "supervisor.log"}
    if output_root.is_symlink() or not output_root.is_dir():
        raise LaunchError("reserved output root changed before supervisor admission")
    if any(path.name not in permitted for path in output_root.iterdir()):
        raise LaunchError("reserved output root has a preexisting fit, log, or foreign path")


def affinity_setter(cpus: tuple[int, ...]) -> Callable[[], None]:
    """Create the pre-exec affinity setter for one eight-core child process."""
    selected = set(cpus)

    def set_affinity() -> None:
        os.sched_setaffinity(0, selected)

    return set_affinity


def _plans_from_admitted(receipt: dict[str, Any], admitted: dict[str, Any]) -> list[CellPlan]:
    """Reload the immutable panel plans after admission and verify all pinned hashes."""
    if sha256_file(Path(receipt["panel"])) != admitted.get("panel_sha256"):
        raise LaunchError("PANEL changed after admission")
    profiles = _profile_map(list(receipt["profiles"]))
    _, cells = load_panel_cells(Path(receipt["panel"]), profiles)
    for cell in cells:
        admitted_cell = admitted["cells"].get(cell.cell_id)
        if (
            not isinstance(admitted_cell, dict)
            or cell.config_file_sha256 != admitted_cell.get("config_file_sha256")
            or cell.profile_sha256 != admitted_cell.get("profile_receipt_sha256")
        ):
            raise LaunchError(f"{cell.cell_id}: panel inputs changed after admission")
    return cells


def _profile_map(values: list[str]) -> dict[str, Path]:
    """Convert the persisted profile arguments into cell paths."""
    result: dict[str, Path] = {}
    for value in values:
        cell_id, separator, name = value.partition("=")
        if not separator or not cell_id or not name or cell_id in result:
            raise LaunchError("launch receipt profile arguments are malformed")
        result[cell_id] = Path(name)
    return result


def _wait_process_identity(pid: int) -> dict[str, int | str]:
    """Wait briefly for the new process's durable start-tick identity to appear."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        identity = process_identity(pid)
        if identity is not None:
            return identity
        time.sleep(0.01)
    raise LaunchError("new fit process did not appear in /proc for ownership recording")
