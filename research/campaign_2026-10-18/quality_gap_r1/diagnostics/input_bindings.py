"""Bounded input validation for the F0 residual diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from receipt_validation import _validate_receipts
from residual_moments import POSITION_EDGES

from wnn.gmetric import METRIC_ID

MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_PREDICTION_BYTES = 32 * 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_ROWS = 1_000_000
SCORING_MASK = "need_prediction AND is_scored; compact rows already selected"
SCORE_TOLERANCE = 2e-10


def sha256_file(path: Path) -> str:
    """Hash a bounded input file in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash canonical contiguous little-endian array bytes."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError("object arrays are not accepted")
    value = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    """Resolve a manifest path independently of the caller's working directory."""
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (base / path).resolve(strict=True)


def _read_pinned_json(base: Path, item: dict[str, object], name: str) -> dict[str, object]:
    """Load a small JSON receipt only after checking its SHA-256."""
    path = _resolve(base, str(item["path"]))
    if path.stat().st_size > MAX_RECEIPT_BYTES or sha256_file(path) != item["sha256"]:
        raise ValueError(f"{name} receipt size or SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} receipt must contain a JSON object")
    return cast(dict[str, object], value)


def _load_inputs(
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Verify compact source, model predictions, and exact row-alignment bindings."""
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest_raw, dict)
        or manifest_raw.get("schema") != "wnn.quality_gap.inputs.v1"
    ):
        raise ValueError("unsupported input manifest")
    manifest = cast(dict[str, object], manifest_raw)
    base = manifest_path.resolve().parent
    source = cast(dict[str, object], manifest["source"])
    if source.get("metric_id") != METRIC_ID or source.get("mask_predicate") != SCORING_MASK:
        raise ValueError("metric or preselected-row mask differs from the official contract")
    bundle_spec = cast(dict[str, object], manifest["bundle"])
    bundle_path = _resolve(base, str(bundle_spec["path"]))
    if bundle_path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("compact bundle exceeds the configured byte bound")
    if bundle_path.stat().st_size != int(bundle_spec["bytes"]):
        raise ValueError("compact bundle byte count mismatch")
    bundle_sha = sha256_file(bundle_path)
    if bundle_sha != bundle_spec["sha256"]:
        raise ValueError("compact bundle SHA-256 mismatch")
    row_count = int(source["selected_rows"])
    if not 1 <= row_count <= MAX_ROWS:
        raise ValueError("selected row count is outside the bounded diagnostic limit")

    fold = cast(dict[str, object], source["fold_manifest"])
    fold_path = _resolve(base, str(fold["path"]))
    if fold_path.stat().st_size > MAX_RECEIPT_BYTES or sha256_file(fold_path) != fold["sha256"]:
        raise ValueError("fold manifest size or SHA-256 mismatch")
    receipts = cast(dict[str, dict[str, object]], manifest["receipts"])
    loaded_receipts = {name: _read_pinned_json(base, item, name) for name, item in receipts.items()}
    _validate_receipts(manifest, loaded_receipts, bundle_sha, row_count)

    keys = cast(dict[str, str | None], bundle_spec["keys"])
    with np.load(bundle_path, allow_pickle=False) as archive:
        absent = [key for key in keys.values() if key is not None and key not in archive.files]
        if absent:
            raise ValueError(f"compact bundle lacks required arrays: {absent}")
        arrays = {name: np.asarray(archive[key]) for name, key in keys.items() if key is not None}
    expected_hashes = cast(dict[str, str], bundle_spec["array_sha256"])
    for name, expected in expected_hashes.items():
        if name not in arrays or sha256_array(arrays[name]) != expected:
            raise ValueError(f"compact bundle array hash mismatch: {name}")
    scalar_bindings = cast(dict[str, str], bundle_spec["scalar_bindings"])
    for name, expected in scalar_bindings.items():
        value = arrays[name]
        if value.shape != () or str(value.item()) != expected:
            raise ValueError(f"compact bundle provenance scalar mismatch: {name}")
    targets = arrays["target"]
    if (
        targets.shape != (row_count, 2)
        or targets.dtype != np.float32
        or not np.isfinite(targets).all()
    ):
        raise ValueError("targets must be finite float32 with shape (selected_rows, 2)")
    for name in ("row_group", "component"):
        value = arrays[name]
        if value.shape != (row_count,) or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer vector aligned to selected rows")
    if len(np.unique(arrays["row_group"])) != int(source["row_groups"]):
        raise ValueError("row-group coverage differs from the pinned source")
    if len(np.unique(arrays["component"])) != int(source["components"]):
        raise ValueError("component coverage differs from the pinned source")
    position_key = keys.get("position")
    position = arrays.get("position") if position_key is not None else None
    if position is not None:
        if position.shape != (row_count,) or position.dtype.kind not in "iu":
            raise ValueError("position must be an aligned integer vector")
        if np.any(position < POSITION_EDGES[0]) or np.any(position >= POSITION_EDGES[-1]):
            raise ValueError("position values are outside the scored sequence range")

    models = cast(dict[str, dict[str, object]], manifest["models"])
    predictions: dict[str, np.ndarray] = {}
    for name, item in models.items():
        kind = item.get("kind")
        if kind == "bundle_key":
            key = str(item["key"])
            value = arrays.get(key)
            if value is None:
                raise ValueError(f"model {name} references missing compact array {key}")
            if sha256_array(value) != item["array_sha256"]:
                raise ValueError(f"model {name} array hash mismatch")
            predictions[name] = value
        elif kind == "npy":
            path = _resolve(base, str(item["path"]))
            if path.stat().st_size > MAX_PREDICTION_BYTES:
                raise ValueError(f"model {name} exceeds the compact-prediction byte bound")
            digest = sha256_file(path)
            if digest != item["sha256"]:
                raise ValueError(f"model {name} file hash mismatch")
            alignment = cast(dict[str, object], item["alignment_receipt"])
            receipt = _read_pinned_json(base, alignment, f"{name} alignment")
            fields = cast(dict[str, str], alignment["fields"])
            checks = {"bundle": bundle_sha, "prediction": digest, "rows": row_count}
            if any(receipt.get(fields[key]) != expected for key, expected in checks.items()):
                raise ValueError(f"model {name} alignment receipt does not bind exact paired rows")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            predictions[name] = value
        elif kind == "fixed_blend":
            continue
        else:
            raise ValueError(f"unsupported model input kind for {name}")
        value = predictions[name]
        if (
            value.shape != (row_count, 2)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"model {name} must be finite float32 with shape (selected_rows, 2)")

    for name, item in models.items():
        if item.get("kind") != "fixed_blend":
            continue
        wide = predictions[str(item["wide_model"])]
        deep = predictions[str(item["deep_model"])]
        wide_weight = np.float32(item["wide_weight"])
        deep_weight = np.float32(1.0 - float(wide_weight))
        prediction = np.add(
            np.multiply(wide, wide_weight), np.multiply(deep, deep_weight), dtype=np.float32
        )
        if sha256_array(prediction) != item["array_sha256"]:
            raise ValueError(f"fixed blend {name} does not match its frozen prediction hash")
        predictions[name] = prediction

    return manifest, arrays, predictions, loaded_receipts
