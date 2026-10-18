"""Fresh PR15 workspaces linking only frozen tabdata inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CACHE_INPUTS = (
    "chains.npz",
    "chain_folds.npz",
    "train_targets.npz",
    "valid_targets.npz",
)


def _link(destination: Path, source: Path, *, directory: bool) -> None:
    """Create a new symlink to one existing immutable campaign input."""
    target = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace workspace path {destination}")
    destination.symlink_to(target, target_is_directory=directory)


def _fresh_work_directory(work: Path, data_root: Path) -> Path:
    """Create or accept only an empty run root outside the shared data tree."""
    requested = work.absolute()
    if requested.is_symlink():
        raise ValueError("worker work path cannot be a symlink")
    requested.parent.mkdir(parents=True, exist_ok=True)
    resolved = requested.resolve(strict=False)
    data = data_root.resolve(strict=True)
    if (
        resolved == data
        or resolved.is_relative_to(data)
        or data.is_relative_to(resolved)
    ):
        raise ValueError("worker output and shared data must be separate trees")
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise FileExistsError(
                f"worker work directory must be fresh and empty: {resolved}"
            )
    else:
        resolved.mkdir()
    return resolved


def prepare_workspace(
    root: Path, work: Path, fold: int, *, package_root: Path, include_scorer: bool
) -> dict[str, Any]:
    """Link PR15's datasets/cache and pin the current canonical fold manifest."""
    if not 0 <= fold < 4:
        raise ValueError("PR15 requires one of the four canonical folds")
    data_root = root / "data"
    datasets = data_root / "datasets"
    cache_source = data_root / "cache"
    split_path = root / "splits" / f"fold{fold}.json"
    for path in (datasets / "train.parquet", datasets / "valid.parquet", split_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required campaign input is missing: {path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        not isinstance(split, dict)
        or split.get("fold") != fold
        or split.get("n_folds") != 4
    ):
        raise ValueError(
            "current split manifest does not bind the requested four-fold cell"
        )
    if not isinstance(split.get("fit"), dict) or not isinstance(
        split.get("evaluation"), dict
    ):
        raise TypeError("current split manifest lacks fit/evaluation membership")
    cache_sources: dict[str, dict[str, Any]] = {}
    for filename in CACHE_INPUTS:
        path = cache_source / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required canonical cache is missing: {path}")
        cache_sources[filename] = {
            "path": str(path.resolve(strict=True)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }

    work = _fresh_work_directory(work, data_root)
    _link(work / "datasets", datasets, directory=True)
    local_cache = work / "cache"
    local_cache.mkdir()
    for filename in CACHE_INPUTS:
        _link(local_cache / filename, cache_source / filename, directory=False)

    # score_run.py uses its own script directory as the run root; this byte-identical
    # workspace copy makes it score only this worker's artifacts and cache overlay.
    scripts = work / "scripts"
    scripts.mkdir()
    source_root = root / "runtime" / "pr15-source-e238-v1"
    copied_scorers: dict[str, str] = {}
    if include_scorer:
        source_scorer = source_root / "scripts" / "score_run.py"
        support_scorer = package_root / "score_support" / "grade_prediction_bank.py"
        for source in (source_scorer, support_scorer):
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = scripts / source.name
            destination.write_bytes(source.read_bytes())
            copied_scorers[destination.name] = _sha256(destination)
    _link(work / "src", source_root / "src", directory=True)

    return {
        "schema": "pr15-host-workspace.v1",
        "work": str(work),
        "data_root": str(data_root.resolve(strict=True)),
        "datasets_input": str((work / "datasets").resolve(strict=True)),
        "cache_is_local": not local_cache.is_symlink(),
        "cache_inputs": cache_sources,
        "split_manifest": str(split_path.resolve(strict=True)),
        "fold": fold,
        "n_folds": 4,
        "fit_membership": split["fit"],
        "evaluation_membership": split["evaluation"],
        "split_manifest_sha256": _sha256(split_path),
        "scorer_copy_sha256": copied_scorers,
        "source_link": str((work / "src").resolve(strict=True)),
    }


def _sha256(path: Path) -> str:
    """Hash a small workspace identity file."""
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
