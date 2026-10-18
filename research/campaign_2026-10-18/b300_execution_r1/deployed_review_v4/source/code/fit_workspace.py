"""Prepare an isolated run root with links to frozen input datasets/cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapter_common import _CACHE_INPUTS, _json, _link, _sha256


def prepare_workspace(
    run_dir: Path, data_root: Path, split_manifest: Path, fold: int
) -> dict[str, Any]:
    """Create a local output root with read-only input links and a writable cache overlay."""
    run_dir = run_dir.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    if (
        run_dir == data_root
        or run_dir.is_relative_to(data_root)
        or data_root.is_relative_to(run_dir)
    ):
        raise ValueError("WNN_RUN_DIR and shared data root must be separate trees")
    for name in ("train.parquet", "valid.parquet"):
        if not (data_root / "datasets" / name).is_file():
            raise FileNotFoundError(data_root / "datasets" / name)
    split_manifest = split_manifest.resolve(strict=True)
    split = _json(split_manifest)
    if split.get("fold") != fold or split.get("n_folds") != 4:
        raise ValueError("split manifest fold does not match the requested four-fold cell")
    cache_source = data_root / "cache"
    for name in _CACHE_INPUTS:
        path = cache_source / name
        if not path.is_file():
            raise FileNotFoundError(path)
    _link(run_dir / "datasets", data_root / "datasets", directory=True)

    # Keep predictions and grader outputs local; only frozen inputs are symlinked.
    cache_overlay = run_dir / "cache"
    if cache_overlay.is_symlink():
        raise ValueError(
            "cache must be a local workspace directory, not a shared-directory symlink"
        )
    cache_overlay.mkdir(exist_ok=True)
    for name in _CACHE_INPUTS:
        _link(cache_overlay / name, cache_source / name, directory=False)
    source_identities = split.get("sources")
    if not isinstance(source_identities, dict):
        raise ValueError("split manifest lacks source identities")
    return {
        "schema": "tabdata-l4-inputs.v1",
        "data_root": str(data_root),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "fold": fold,
        "n_folds": 4,
        "fit_train_groups": split.get("fit", {}).get("train"),
        "fit_valid_groups": split.get("fit", {}).get("valid"),
        "evaluation_groups": split.get("evaluation", {}).get("valid"),
        "source_identities": source_identities,
        "cache_inputs": {
            name: {
                "path": str(cache_source / name),
                "bytes": (cache_source / name).stat().st_size,
                "mtime_ns": (cache_source / name).stat().st_mtime_ns,
                "sha256": _sha256(cache_source / name)
                if name in {"chains.npz", "chain_folds.npz"}
                else None,
            }
            for name in _CACHE_INPUTS
        },
        "workspace_links": {
            "datasets": str((run_dir / "datasets").resolve(strict=True)),
            "cache_is_local_overlay": True,
        },
    }


def prepare_export_workspace(export_dir: Path, fit_dir: Path) -> dict[str, Any]:
    """Create a fresh output root linked to immutable fit artifacts and frozen inputs."""
    fit_dir = fit_dir.resolve(strict=True)
    export_dir = export_dir.parent.resolve(strict=True) / export_dir.name
    if not export_dir.parent.is_dir():
        raise FileNotFoundError(f"export workspace parent must exist: {export_dir.parent}")
    data_root = (fit_dir / "datasets").resolve(strict=True).parent.resolve(strict=True)
    if (
        export_dir in (fit_dir, data_root)
        or export_dir.is_relative_to(fit_dir)
        or fit_dir.is_relative_to(export_dir)
        or export_dir.is_relative_to(data_root)
        or data_root.is_relative_to(export_dir)
    ):
        raise ValueError("export workspace must be separate from fit and shared data trees")
    if export_dir.exists():
        if not export_dir.is_dir() or any(export_dir.iterdir()):
            raise FileExistsError(f"export workspace must be fresh and empty: {export_dir}")
    else:
        export_dir.mkdir()
    _link(export_dir / "artifacts", fit_dir / "artifacts", directory=True)
    _link(export_dir / "datasets", fit_dir / "datasets", directory=True)
    cache = export_dir / "cache"
    cache.mkdir()
    for name in _CACHE_INPUTS:
        _link(cache / name, fit_dir / "cache" / name, directory=False)
    return {
        "schema": "tabdata-l4-export-workspace.v1",
        "export_dir": str(export_dir.resolve(strict=True)),
        "fit_dir": str(fit_dir),
        "artifacts_link": str((export_dir / "artifacts").resolve(strict=True)),
        "datasets_link": str((export_dir / "datasets").resolve(strict=True)),
        "cache_is_local_overlay": not cache.is_symlink(),
    }


def require_fresh_export_outputs(output_root: Path, run_name: str) -> tuple[Path, Path, Path]:
    """Return output paths only when this export workspace has no prior attempt."""
    if not output_root.is_dir():
        raise FileNotFoundError(f"export workspace is missing: {output_root}")
    prediction = output_root / "cache" / f"pred_{run_name}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    receipt = output_root / "export_receipt.json"
    command = output_root / "export_command.json"
    if any(path.exists() for path in (prediction, sidecar, receipt, command)):
        raise FileExistsError(
            "export workspace already has output; use a fresh export workspace linked to this fit"
        )
    if (output_root / "cache").is_symlink():
        raise ValueError("prediction output cache must be local to the export workspace")
    if not (output_root / "cache").is_dir():
        raise FileNotFoundError("export workspace has no local writable cache directory")
    return prediction, receipt, command
