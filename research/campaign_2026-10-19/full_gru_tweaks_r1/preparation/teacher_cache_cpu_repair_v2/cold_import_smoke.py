from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any


def deny(calls: list[str], name: str):
    def rejected(*args: Any, **kwargs: Any) -> None:
        calls.append(name)
        raise AssertionError(f"forbidden data read during cold CLI import: {name}")

    return rejected


def main() -> int:
    repo = pathlib.Path.cwd().resolve()
    bundle = pathlib.Path(tempfile.mkdtemp(prefix="teacher-cache-cold-import-"))
    source_root = bundle / "source"
    scripts_root = bundle / "scripts"
    shutil.copytree(
        repo / "src/wnn",
        source_root / "wnn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    scripts_root.mkdir()
    entrypoint_names = ("train_rnn.py", "predict_run.py", "build_distill_cache.py")
    for name in entrypoint_names:
        shutil.copy2(repo / "scripts" / name, scripts_root / name)
    external_paths = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            pathlib.Path(entry).resolve().relative_to(repo)
        except ValueError:
            external_paths.append(entry)
    sys.path[:] = [str(scripts_root), str(source_root), *external_paths]
    os.chdir(bundle)
    os.environ.pop("WNN_PROJECT_ROOT", None)

    import numpy as np
    import torch

    calls: list[str] = []
    np.load = deny(calls, "numpy.load")
    np.memmap = deny(calls, "numpy.memmap")
    torch.load = deny(calls, "torch.load")
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        parquet = None
    if parquet is not None:
        parquet.ParquetFile = deny(calls, "pyarrow.parquet.ParquetFile")
        parquet.read_table = deny(calls, "pyarrow.parquet.read_table")
        parquet.read_metadata = deny(calls, "pyarrow.parquet.read_metadata")

    from wnn.data import SequenceFile

    SequenceFile.read = deny(calls, "SequenceFile.read")
    imported = []
    failures = []
    for module_name in ("train_rnn", "predict_run", "build_distill_cache"):
        try:
            imported.append(importlib.import_module(module_name))
        except Exception as error:
            failures.append(f"{module_name}: {type(error).__name__}: {error}")
    if not failures:
        predictor = imported[1]
        args = predictor.build_parser().parse_args(
            [
                "teacher",
                "train",
                "--epoch",
                "2",
                "--campaign-split-manifest",
                "F0.json",
                "--inference-precision",
                "fp32",
                "--loader-workers",
                "0",
                "--profile-only",
                "--profile-output",
                "PROFILE.json",
            ]
        )
        if not args.profile_only or args.loader_workers != 0:
            failures.append("predict_run profile CLI did not preserve the fixed options")
        cache_builder = imported[2]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                cache_builder.main.__globals__["sys"].argv = ["build_distill_cache.py", "--help"]
                cache_builder.main()
            except SystemExit as error:
                if error.code != 0:
                    failures.append(f"cache builder --help exited {error.code}")
            except Exception as error:
                failures.append(f"cache builder CLI: {type(error).__name__}: {error}")

    required = {
        "wnn.cleanfold_t3_cache",
        "wnn.prediction_profile",
        "wnn.training.evaluation",
        "wnn.training.forward",
        "wnn.training.inputs",
        "wnn.training.models",
        "wnn.splits",
        "wnn.resources",
        "wnn.run_identity",
    }
    runtime = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("wnn") and getattr(module, "__file__", None)
    }
    missing = sorted(required - runtime.keys())
    wrong_origins: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    for name, module in runtime.items():
        path = pathlib.Path(module.__file__).resolve()
        try:
            relative = path.relative_to(source_root.resolve())
        except ValueError:
            wrong_origins[name] = str(path)
            continue
        file_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    entrypoint_paths = {}
    for name, module in zip(entrypoint_names, imported, strict=False):
        path = pathlib.Path(module.__file__).resolve()
        entrypoint_paths[f"scripts/{name}"] = str(path.relative_to(scripts_root.resolve()))
        file_hashes[f"scripts/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    passed = not calls and not missing and not wrong_origins and not failures
    smoke_bytes = pathlib.Path(__file__).read_bytes()
    receipt = {
        "schema": "strong_gru_tweaks.teacher_cache_cli_import_smoke.v1",
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
        "smoke_source": str(pathlib.Path(__file__).resolve()),
        "smoke_source_sha256": hashlib.sha256(smoke_bytes).hexdigest(),
        "failures": failures,
    }
    out = pathlib.Path("/tmp/teacher_cache_cold_import_receipt.json")
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "runtime_wnn_module_count": len(runtime), "out": str(out)}))
    shutil.rmtree(bundle)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
