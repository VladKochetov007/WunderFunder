"""Cold-import trainer and prediction entrypoints with all data reads denied."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any


def _deny(calls: list[str], name: str):
    def denied(*args: Any, **kwargs: Any) -> None:
        calls.append(name)
        raise AssertionError(f"forbidden read during cold import: {name}")

    return denied


def main() -> int:
    """Import an isolated project copy and print its exact runtime file pins."""
    repo = pathlib.Path(__file__).resolve().parents[3]
    bundle = pathlib.Path(tempfile.mkdtemp(prefix="representation-import-smoke-"))
    source_root = bundle / "source"
    scripts_root = bundle / "scripts"
    shutil.copytree(
        repo / "src/wnn",
        source_root / "wnn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    scripts_root.mkdir()
    for name in ("train_rnn.py", "train_propensity.py", "predict_run.py"):
        shutil.copy2(repo / "scripts" / name, scripts_root / name)
    source_paths = [str(scripts_root), str(source_root)]
    external_paths = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            pathlib.Path(entry).resolve().relative_to(repo.resolve())
        except ValueError:
            external_paths.append(entry)
    sys.path[:] = source_paths + external_paths
    os.chdir(bundle)

    import numpy as np
    import torch

    calls: list[str] = []
    np.load = _deny(calls, "numpy.load")
    np.memmap = _deny(calls, "numpy.memmap")
    torch.load = _deny(calls, "torch.load")
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        parquet = None
    if parquet is not None:
        parquet.ParquetFile = _deny(calls, "pyarrow.parquet.ParquetFile")
        parquet.read_table = _deny(calls, "pyarrow.parquet.read_table")
        parquet.read_metadata = _deny(calls, "pyarrow.parquet.read_metadata")

    from wnn.data import SequenceFile

    SequenceFile.read = _deny(calls, "SequenceFile.read")
    imported = [
        importlib.import_module("train_rnn"),
        importlib.import_module("train_propensity"),
        importlib.import_module("predict_run"),
        importlib.import_module("wnn.training.input_representation"),
        importlib.import_module("wnn.training.input_representation_fit"),
        importlib.import_module("wnn.training.setup_preprocessing"),
        importlib.import_module("wnn.training.inputs"),
        importlib.import_module("wnn.training.checkpoints"),
        importlib.import_module("wnn.training.outputs"),
        importlib.import_module("wnn.training.setup_model"),
        importlib.import_module("wnn.training.mean_variance"),
    ]
    required = {
        "wnn.training.input_representation",
        "wnn.training.input_representation_fit",
        "wnn.training.setup_preprocessing",
        "wnn.training.inputs",
        "wnn.training.checkpoints",
        "wnn.training.outputs",
        "wnn.training.setup_model",
        "wnn.training.mean_variance",
    }
    runtime = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("wnn.") and getattr(module, "__file__", None)
    }
    missing = sorted(required - runtime.keys())
    wrong_origins = {}
    file_hashes = {}
    for name, module in runtime.items():
        path = pathlib.Path(module.__file__).resolve()
        try:
            relative = path.relative_to(source_root.resolve())
        except ValueError:
            wrong_origins[name] = str(path)
            continue
        file_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    entrypoint_paths = {}
    for name, module in zip(
        ("scripts/train_rnn.py", "scripts/train_propensity.py", "scripts/predict_run.py"),
        imported[:3],
        strict=True,
    ):
        path = pathlib.Path(module.__file__).resolve()
        entrypoint_paths[name] = str(path.relative_to(scripts_root.resolve()))
        file_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    passed = not calls and not missing and not wrong_origins
    receipt = {
        "schema": "strong_gru_tweaks.representation_import_smoke.v1",
        "status": "PASS" if passed else "FAIL",
        "isolated_copy": True,
        "runtime_wnn_module_count": len(runtime),
        "required_semantic_modules_present": not missing,
        "missing_semantic_modules": missing,
        "all_runtime_origins_inside_isolated_wnn_copy": not wrong_origins,
        "wrong_runtime_origins": wrong_origins,
        "data_read_sentinels": {
            "numpy.load": True,
            "numpy.memmap": True,
            "pyarrow.parquet": parquet is not None,
            "SequenceFile.read": True,
            "torch.load": True,
            "calls": calls,
        },
        "entrypoint_paths": entrypoint_paths,
        "runtime_file_sha256": dict(sorted(file_hashes.items())),
        "smoke_source": str(pathlib.Path(__file__).relative_to(repo)),
    }
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    shutil.rmtree(bundle)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
