"""Native inference and immutable-cache helpers for the precision probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def digest_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    """Reject missing or changed frozen inputs."""
    if not path.is_file() or digest_file(path) != expected:
        raise ValueError(f"missing or hash-mismatched frozen artifact: {path}")


def load_native_module(package: Path) -> tuple[ModuleType, ModuleType, Path, str]:
    """Load package source and its exact v3 extension without endpoint construction."""
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location(
        "precision_probe_solution", package / "solution.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("packaged solution.py is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native = sys.modules.get("wnn_cpp")
    if native is None:
        raise ImportError("solution.py did not import wnn_cpp")
    native_path = Path(native.__file__).resolve()
    if not native_path.is_relative_to(package.resolve()):
        raise RuntimeError("loaded native extension escaped the candidate package")
    level, backend = str(module.LEVEL), str(native.quant_backend())
    if level != "v3":
        raise RuntimeError(f"expected native LEVEL=v3, found {level}")
    return module, native, native_path, backend


def load_members(package: Path) -> list[dict[str, Any]]:
    """Read the package's fixed three-member registry."""
    members = json.loads((package / "members.json").read_text(encoding="utf-8"))
    if len(members) != 3 or [float(row["coef"]) for row in members] != [0.34, 0.33, 0.33]:
        raise ValueError("package member roster or fixed coefficients changed")
    return members


def build_ensembles(
    native: ModuleType,
    package: Path,
    members: list[dict[str, Any]],
    *,
    unit_weight: bool = False,
) -> list[tuple[Any, float]]:
    """Build models grouped exactly as the wrapper groups its fp32 member flag."""
    groups: dict[bool, list[dict[str, Any]]] = {}
    for member in members:
        groups.setdefault(bool(member.get("fp32", False)), []).append(member)
    result: list[tuple[Any, float]] = []
    for exact, group in groups.items():
        weights = [load_weight_dict(package / row["weights"]) for row in group]
        coefficients = [1.0 if unit_weight else float(row["coef"]) for row in group]
        model = native.Ensemble(
            weights,
            coefficients,
            fast_activations=True,
            half_weights=exact,
            dynamic_int8=not exact,
            quant_backend="auto",
        )
        result.append((model, 1.0))
    return result


def build_single_precision(
    native: ModuleType,
    package: Path,
    member: dict[str, Any],
    *,
    fast: bool,
    dynamic: bool,
) -> Any:
    """Construct the H512 member under one requested native precision."""
    weights = load_weight_dict(package / member["weights"])
    return native.Ensemble(
        [weights],
        [1.0],
        fast_activations=fast,
        half_weights=False,
        dynamic_int8=dynamic,
        quant_backend="auto",
    )


def load_weight_dict(path: Path) -> dict[str, np.ndarray]:
    """Materialize the exported NPZ tensors in the native wrapper's mapping format."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key], dtype=np.float32) for key in archive.files}


def role_step(models: list[tuple[Any, float]], features: np.ndarray) -> np.ndarray:
    """Advance precision groups once and sum their predictions."""
    output = np.zeros(2, dtype=np.float64)
    for model, scale in models:
        value = np.asarray(model.step(features), dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("native member emitted malformed predictions")
        output += scale * value
    return output.astype(np.float32)


def load_full_cache(
    path: Path,
    sidecar_path: Path,
    expected_sha: str,
    sidecar_sha: str,
    expected_groups: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate full F0-evaluation float32 coverage and its row-group mapping."""
    require_hash(path, expected_sha)
    require_hash(sidecar_path, sidecar_sha)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    groups = list(map(int, sidecar.get("row_groups", [])))
    if values.shape != (len(expected_groups), 20_000, 2) or values.dtype != np.float32:
        raise ValueError(f"prediction cache is not full-F0 float32: {path}")
    if len(groups) != len(expected_groups) or set(groups) != set(expected_groups):
        raise ValueError(f"prediction sidecar lacks full F0 row-group coverage: {path}")
    if sidecar.get("prediction_sha256") not in (None, expected_sha):
        raise ValueError(f"sidecar prediction hash differs from array: {path}")
    return values, {**sidecar, "sidecar_sha256": sidecar_sha, "row_groups": groups}


def run_features(
    source: Any, selected: list[dict[str, Any]], models: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Stream selected feature-only groups, resetting every native state."""
    from f0_replay import _read_features

    names = (
        "h512_dynamic_int8",
        "h512_float32_fast",
        "h512_float32_exact",
        "original_student",
        "retained_pair",
    )
    shape = (len(selected), source.sequence_length, 2)
    output = {name: np.zeros(shape, dtype=np.float32) for name in names}
    needs: list[np.ndarray] = []
    for position, member in enumerate(selected):
        row_group, seq_ix = int(member["row_group"]), int(member["seq_ix"])
        table = source.parquet.read_row_group(
            row_group, columns=list(source.columns), use_threads=False
        )
        features, need = _read_features(table, source, seq_ix)
        if features.shape != (20_000, len(source.feature_columns)):
            raise ValueError("selected feature group lacks a complete 20k sequence")
        for value in models.values():
            group = value if isinstance(value, list) else [(value, 1.0)]
            for model, _ in group:
                model.reset()
        needs.append(need.copy())
        for step, row in enumerate(features):
            values = {
                "h512_dynamic_int8": role_step([(models["h512_dynamic_int8"], 1.0)], row),
                "h512_float32_fast": role_step([(models["h512_float32_fast"], 1.0)], row),
                "h512_float32_exact": role_step([(models["h512_float32_exact"], 1.0)], row),
                "original_student": role_step(models["original_student"], row),
                "retained_pair": role_step(models["retained_pair"], row),
            }
            if need[step]:
                for name, prediction in values.items():
                    output[name][position, step] = prediction
        del table, features
    return output, needs


def cache_rows(
    cache: np.ndarray, sidecar: dict[str, Any], members: list[dict[str, Any]]
) -> np.ndarray:
    """Select cache rows by their frozen row-group identities."""
    index = {group: row for row, group in enumerate(sidecar["row_groups"])}
    return np.stack([cache[index[int(item["row_group"])]] for item in members])


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the maximum absolute discrepancy in float64 arithmetic."""
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
