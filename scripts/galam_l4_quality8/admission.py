from __future__ import annotations

from pathlib import Path
from typing import Any

from wnn.training.sequence_provenance import validate_profile_receipt

from .common import LaunchError, sha256_file
from .contract import CellPlan, load_panel_cells
from .host import probe_host, validate_host_profiles
from .identity import build_cell_identity


def preflight(
    panel_path: Path,
    profile_args: list[str],
    data_root: Path,
    split_manifest: Path,
    output_root: Path,
    repo_root: Path,
    reserved_output: bool = False,
) -> dict[str, Any]:
    """Check all eight panel, source, data, profile, runtime, and output gates."""
    output_root = output_root.expanduser().absolute()
    if reserved_output:
        _assert_reserved_root(output_root)
    else:
        _assert_output_available(output_root)
    profiles = _parse_profiles(profile_args)
    panel, cells = load_panel_cells(panel_path, profiles)
    assert_child_paths_fresh(output_root, cells)
    host = probe_host()
    runtimes = validate_host_profiles(cells, host)
    panel_hash = sha256_file(panel_path)
    if split_manifest.is_symlink() or data_root.is_symlink():
        raise LaunchError("split manifest and data root cannot be symlinks")
    split_manifest = split_manifest.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    cell_records: dict[str, dict[str, Any]] = {}
    for cell in cells:
        identity = build_cell_identity(
            cell,
            data_root,
            split_manifest,
            repo_root.resolve(strict=True),
            runtimes[cell.cell_id],
        )
        receipt_hash = validate_profile_receipt(cell.profile_path, identity)
        if receipt_hash != cell.profile_sha256:
            raise LaunchError(f"{cell.cell_id}: profile receipt changed during preflight")
        cell_records[cell.cell_id] = {
            "gpu": cell.gpu,
            "gpu_uuid": host.gpus[cell.gpu]["uuid"],
            "cpu_affinity": list(cell.cpus),
            "numa_node": cell.numa_node,
            "config": str(cell.config_path),
            "config_file_sha256": cell.config_file_sha256,
            "config_sha256": identity["config_sha256"],
            "profile_receipt": str(cell.profile_path),
            "profile_receipt_sha256": receipt_hash,
            "identity_sha256": _canonical_sha256(identity),
            "identity": identity,
        }
    if sha256_file(panel_path) != panel_hash:
        raise LaunchError("PANEL changed during preflight")
    for cell in cells:
        if (
            sha256_file(cell.config_path) != cell.config_file_sha256
            or sha256_file(cell.profile_path) != cell.profile_sha256
        ):
            raise LaunchError(f"{cell.cell_id}: config or profile changed during preflight")
    manifest = {
        "schema": "wnn.galam-l4-quality8-preflight.v1",
        "panel": str(panel_path.resolve()),
        "panel_sha256": panel_hash,
        "data_root": str(data_root),
        "split_manifest": str(split_manifest),
        "split_sha256": cells[0].profile["identity"]["data"]["split_sha256"],
        "source_sha256": cells[0].profile["identity"]["source_sha256"],
        "launcher_sha256": _launcher_code_sha256(repo_root.resolve(strict=True)),
        "runtime": {
            "torch": str(__import__("torch").__version__),
            "cuda_runtime": str(__import__("torch").version.cuda),
            "nvidia_driver": host.driver,
            "gpu_uuids": [host.gpus[index]["uuid"] for index in range(8)],
            "gpu_memory": [
                {
                    "index": index,
                    "uuid": host.gpus[index]["uuid"],
                    "nvidia_smi_total_mib": host.gpus[index]["total_mib"],
                    "nvidia_smi_free_mib": host.gpus[index]["free_mib"],
                    "cuda_total_memory_bytes": host.cuda_devices[
                        host.gpus[index]["uuid"]
                    ]["total_memory_bytes"],
                }
                for index in range(8)
            ],
            "cublas_workspace_config": ":4096:8",
        },
        "cells": cell_records,
    }
    return manifest


def _assert_output_available(output_root: Path) -> None:
    """Fail before launch if the output root, its parent, or run paths are unsafe."""
    if output_root.exists() or output_root.is_symlink():
        raise LaunchError(f"output root already exists: {output_root}")
    parent = output_root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise LaunchError("output parent must exist and cannot be a symlink")


def _assert_reserved_root(output_root: Path) -> None:
    """Accept only the private launcher reservation created before supervision."""
    if output_root.is_symlink() or not output_root.is_dir():
        raise LaunchError("reserved output root is missing or unsafe")
    permitted = {"launch_receipt.json", "supervisor.log"}
    if any(item.name not in permitted for item in output_root.iterdir()):
        raise LaunchError("reserved output root contains an unexpected preexisting path")


def _parse_profiles(values: list[str]) -> dict[str, Path]:
    """Convert command-line profile references while preserving their cell key."""
    profiles: dict[str, Path] = {}
    for value in values:
        cell_id, separator, name = value.partition("=")
        if not separator or not cell_id or not name or cell_id in profiles:
            raise LaunchError("profiles must be unique CELL=PATH arguments")
        profiles[cell_id] = Path(name).expanduser().absolute()
    if len(profiles) != 8:
        raise LaunchError("exactly eight approved profile receipt arguments are required")
    return profiles


def assert_child_paths_fresh(output_root: Path, cells: list[CellPlan]) -> None:
    """Check all predetermined run and log paths before any fit process starts."""
    for cell in cells:
        for path in (output_root / "runs" / cell.cell_id, output_root / "logs" / f"{cell.cell_id}.log"):
            if path.exists() or path.is_symlink():
                raise LaunchError(f"fit output path already exists: {path}")


def _canonical_sha256(value: object) -> str:
    """Hash JSON identity in the same canonical form as the fit runtime."""
    from wnn.training.sequence_provenance import canonical_sha256

    return canonical_sha256(value)


def _launcher_code_sha256(repo_root: Path) -> str:
    """Bind the operational launcher files separately from the fit-source identity."""
    files = [repo_root / "scripts" / "launch_galam_quality8.py"]
    files.extend(sorted((repo_root / "scripts" / "galam_l4_quality8").rglob("*.py")))
    manifest: dict[str, str] = {}
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise LaunchError("launcher source contains a missing file or symlink")
        manifest[path.relative_to(repo_root).as_posix()] = sha256_file(path)
    from wnn.training.sequence_provenance import canonical_sha256

    return canonical_sha256(manifest)
