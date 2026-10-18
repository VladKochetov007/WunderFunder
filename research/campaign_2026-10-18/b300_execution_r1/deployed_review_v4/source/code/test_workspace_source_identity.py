"""Exercise overlay inputs against the actual frozen campaign split loader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fit_workspace import prepare_export_workspace, prepare_workspace


def test_fit_and_export_overlays_satisfy_all_frozen_source_identities(
    tmp_path, monkeypatch
) -> None:
    """Both target indexes are required even though training reads raw labels."""
    source_root = os.environ.get("TABDATA_FROZEN_SOURCE_ROOT")
    if not source_root:
        pytest.skip("set TABDATA_FROZEN_SOURCE_ROOT to the verified a266 source tree")
    source = Path(source_root) / "src/wnn/splits.py"
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("tabdata_frozen_splits_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "cache").mkdir()
    for name in ("train", "valid"):
        (data / "datasets" / f"{name}.parquet").write_bytes(name.encode())
        np.savez(
            data / "cache" / f"{name}_targets.npz",
            row_group=np.array([0, 1]),
            seq_ix=np.array([100, 101] if name == "train" else [200, 201]),
        )
    splits = np.array(["train", "train", "valid", "valid"])
    groups = np.array([0, 1, 0, 1])
    np.savez(data / "cache/chains.npz", split=splits, row_group=groups, next=np.full(4, -1))
    np.savez(data / "cache/chain_folds.npz", split=splits, row_group=groups, fold=[0, 0, 1, 0])
    sources = {}
    for path in [*(data / "datasets").iterdir(), *(data / "cache").iterdir()]:
        stat = path.stat()
        sources[path.relative_to(data).as_posix()] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "version": 1,
        "fold": 0,
        "n_folds": 4,
        "sources": sources,
        "sequences": {
            "train": [{"row_group": 0, "component": 0}, {"row_group": 1, "component": 1}],
            "valid": [{"row_group": 0, "component": 2}, {"row_group": 1, "component": 3}],
        },
        "fit": {"train": [0, 1], "valid": [0]},
        "evaluation": {"valid": [1]},
    }
    manifest_path = tmp_path / "fold0.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fit = tmp_path / "fit"
    fit.mkdir()
    try:
        prepare_workspace(fit, data, manifest_path, 0)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("directory symlinks require Windows Developer Mode or privilege")
        raise
    (fit / "artifacts").mkdir()
    loaded = module.load_campaign_split(manifest_path, fit)
    assert loaded.train_groups == (0, 1)
    assert (fit / "cache/train_targets.npz").resolve() == data / "cache/train_targets.npz"
    export = tmp_path / "export"
    prepare_export_workspace(export, fit)
    assert module.load_campaign_split(manifest_path, export).evaluation_groups == (1,)
