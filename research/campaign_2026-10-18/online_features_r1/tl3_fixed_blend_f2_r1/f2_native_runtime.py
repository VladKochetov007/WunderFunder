"""Load exact ZIP callbacks whose native extensions use different module names."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from native_runtime import TL3Model


def load_callback_f2(package: Path) -> tuple[TL3Model, Path]:
    """Import one package callback and verify its native library stays in the ZIP."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("exact_f2_solution", package / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError("ZIP does not contain an importable root solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback = cast(TL3Model, module.PredictionModel())
    package_root = package.resolve()
    native_paths: list[Path] = []
    for name in ("wnn_cpp", "wnn_native"):
        native = sys.modules.get(name)
        path = getattr(native, "__file__", None)
        if path is not None:
            resolved = Path(path).resolve()
            if resolved.is_relative_to(package_root):
                native_paths.append(resolved)
    if not native_paths:
        raise RuntimeError("package did not load an in-package wnn_cpp or wnn_native library")
    return callback, native_paths[-1]
